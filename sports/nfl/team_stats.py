"""
sports/nfl/team_stats.py

NFLTeamStatsFetcher: métricas de rendimiento de equipo desde play-by-play.

Por qué EPA y no puntos por partido
-------------------------------------
EPA (Expected Points Added) mide el valor de cada jugada según el
cambio en el valor esperado de puntos del drive. Una ganancia de 4
yardas en 3&3 es exitosa; la misma ganancia en 3&10 no lo es. EPA
captura esa diferencia; las yardas totales no.

Correlación con resultados de la temporada siguiente:
    EPA/play          ≈ 0.65
    Success rate      ≈ 0.55
    Yardas por juego  ≈ 0.35
    Puntos por juego  ≈ 0.40
    Win percentage    ≈ 0.30

Por eso config/nfl.yaml asigna el 70% del peso de proyección a EPA
(ofensivo + defensivo) y solo 15% a forma reciente en puntos.

Tres problemas estructurales que este módulo resuelve
------------------------------------------------------

1. LOOK-AHEAD BIAS
   nflverse entrega el play-by-play de la temporada COMPLETA, incluidas
   semanas que aún no se han jugado en el momento que simulamos. Si
   proyectamos la semana 10 usando datos de las semanas 11-18, el
   backtest sale inflado y el modelo no reproduce ese rendimiento en
   producción.

   `up_to_week` es un parámetro OBLIGATORIO en fetch(). No tiene valor
   por defecto deliberadamente: olvidarlo debe ser un error de
   programación visible, no un fallo silencioso que contamina resultados.

2. EPA NO ADMITE NORMALIZACIÓN POR RATIO
   El offense_index de MLB es OPS / OPS_LIGA — funciona porque OPS es
   siempre positivo y ronda 0.720. EPA/play promedia ~0.00 y puede ser
   negativo, así que un ratio explota o cambia de signo.

   Se usa transformación ADITIVA centrada en la media de liga:
       index = 1 + (epa_equipo - epa_liga) × factor

   Con factor calibrado para que la escala sea comparable a la de MLB:
   un equipo a una desviación típica por encima de la media queda en
   ~1.24, igual que un equipo MLB con OPS 0.890 vs liga 0.720.

3. MUESTRA PEQUEÑA AL INICIO DE TEMPORADA
   En la semana 3 un equipo lleva ~200 jugadas. El EPA de esa muestra
   es mayoritariamente ruido. Sin corrección, el modelo perseguiría
   señales inexistentes durante el primer tercio de la temporada —
   justo cuando el mercado también tiene más incertidumbre y las
   oportunidades parecen más atractivas.

   Se aplica regresión a la media bayesiana:
       epa_ajustado = (n × epa_obs + k × epa_liga) / (n + k)

   con k = 200 jugadas (≈3 partidos) como fuerza del prior. En la
   semana 3 el equipo pesa 50%; en la semana 12 pesa ~80%.

Ajuste por calendario (strength of schedule)
----------------------------------------------
Un equipo con +0.10 EPA/play contra defensas débiles no es igual a uno
con +0.10 contra las mejores. Se aplica un ajuste de una pasada: al EPA
ofensivo de cada equipo se le resta la desviación media de las defensas
que enfrentó respecto a la liga.

Es una aproximación de DVOA, no DVOA completo — el cálculo real de
Football Outsiders es iterativo y pondera también situación de juego.
Una pasada captura la mayor parte del efecto sin la complejidad ni el
riesgo de convergencia del método iterativo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sports.nfl.data_source import NFLDataSource, _current_nfl_season


# ── Constantes de calibración ────────────────────────────────────────────────

# Tipos de jugada que cuentan para EPA/play.
# Se excluyen despejes, patadas, extra points y jugadas anuladas: no
# reflejan la calidad de la ofensiva ni de la defensa en juego abierto.
_SCRIMMAGE_PLAYS = frozenset({"pass", "run"})

# Fuerza del prior para la regresión a la media, en número de jugadas.
# 200 jugadas ≈ 3 partidos. Con este valor, un equipo en la semana 3
# (≈200 jugadas) pesa 50% y la media de liga el otro 50%.
_SHRINKAGE_PLAYS: float = 200.0

# Factor de conversión de EPA a índice normalizado.
#
# Derivación: la desviación típica de EPA/play entre equipos NFL es
# ≈0.08. Un equipo a +1σ juega ~65 jugadas de scrimmage por partido,
# lo que se traduce en ≈5.2 puntos extra sobre una media de 22 →
# ratio 1.24.
#
#     0.08 × factor = 0.24  →  factor = 3.0
#
# Esto deja la escala de offense_index/defense_index de NFL alineada
# con la de MLB, donde un equipo fuerte ronda 1.10-1.25. El Core no
# sabe de qué deporte viene el índice: la comparabilidad entre plugins
# es lo que permite que BlendingEngine y ValueEngine sean genéricos.
_EPA_TO_INDEX_FACTOR: float = 3.0

# Cotas del índice: red de seguridad, NO mecanismo de calibración.
#
# La protección principal contra valores extremos es el shrinkage
# (_SHRINKAGE_PLAYS). Estas cotas solo deben actuar cuando el shrinkage
# está desactivado o los datos vienen corruptos.
#
# Calibración verificada contra el rango real de EPA/play en NFL:
#
#     Equipo           EPA crudo   Tras shrinkage   Índice
#     Elite               +0.12         +0.10        1.30
#     Bueno               +0.06         +0.05        1.15
#     Medio                0.00          0.00        1.00
#     Malo                -0.06         -0.05        0.85
#     Peor de la liga     -0.12         -0.10        0.70
#
# Con las cotas anteriores ([0.70, 1.35]) el peor equipo de la liga
# caía EXACTAMENTE en el límite inferior — el clamp recortaba señal
# legítima en vez de actuar solo ante anomalías. Además eran
# asimétricas respecto a 1.0 (+0.35 arriba, -0.30 abajo), lo que
# sesgaba el índice a favor de las ofensivas.
#
# Las cotas actuales son simétricas y dejan 0.10 de margen sobre los
# extremos reales de la liga.
_INDEX_MIN: float = 0.60
_INDEX_MAX: float = 1.40

# Valores de liga por defecto cuando no hay datos suficientes.
# EPA/play de liga ronda 0.00 por construcción (la suma de EPA
# ofensivo y defensivo de toda la liga es cero).
_DEFAULT_LEAGUE_EPA: float = 0.0
_DEFAULT_LEAGUE_SUCCESS: float = 0.45
_DEFAULT_POINTS_PER_GAME: float = 22.0


# ── Agregados ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class LeagueAverages:
    """
    Medias de liga hasta una semana dada.

    Necesarias para normalizar los índices y como prior de la
    regresión a la media. Se calculan sobre las mismas jugadas
    filtradas que usan las estadísticas de equipo, para que la
    comparación sea homogénea.
    """
    season:        int
    up_to_week:    int
    epa_per_play:  float = _DEFAULT_LEAGUE_EPA
    success_rate:  float = _DEFAULT_LEAGUE_SUCCESS
    epa_pass:      float = _DEFAULT_LEAGUE_EPA
    epa_rush:      float = _DEFAULT_LEAGUE_EPA
    points_per_game: float = _DEFAULT_POINTS_PER_GAME
    total_plays:   int = 0

    @property
    def has_sufficient_sample(self) -> bool:
        """True si hay suficientes jugadas para que las medias sean fiables."""
        return self.total_plays >= 1000


@dataclass(frozen=True)
class NFLTeamStats:
    """
    Métricas de rendimiento de un equipo hasta una semana dada.

    Inmutable: representa el estado del equipo en un corte temporal
    concreto. Dos cortes distintos son dos objetos distintos — esto
    hace imposible contaminar accidentalmente un backtest reutilizando
    estadísticas de otra semana.

    Campos de EPA
    -------------
    epa_off        -- EPA/play ofensivo, ya con shrinkage aplicado.
    epa_def        -- EPA/play permitido por la defensa (menor es mejor).
    epa_off_raw    -- EPA ofensivo sin shrinkage ni ajuste de calendario.
                      Se conserva para diagnóstico y trazabilidad.
    epa_def_raw    -- Equivalente defensivo.
    epa_pass_off   -- EPA/play en jugadas de pase.
    epa_rush_off   -- EPA/play en jugadas de carrera.
    epa_pass_def   -- EPA permitido contra el pase.
    epa_rush_def   -- EPA permitido contra la carrera.

    El desglose pase/carrera importa porque los emparejamientos son
    asimétricos: una defensa fuerte contra la carrera y débil contra
    el pase rinde muy distinto según el rival.

    Campos de eficiencia
    --------------------
    success_off    -- % de jugadas ofensivas exitosas (ajustado por down).
    success_def    -- % de jugadas del rival que fueron exitosas.

    Campos de resultado
    -------------------
    points_per_game         -- Puntos anotados por partido.
    points_allowed_per_game -- Puntos permitidos por partido.
    recent_scores           -- Puntos anotados en los últimos N partidos,
                               del más antiguo al más reciente.

    Metadatos de muestra
    --------------------
    games_played   -- Partidos jugados hasta el corte.
    plays_off      -- Jugadas ofensivas en la muestra.
    plays_def      -- Jugadas defensivas en la muestra.
    sos_adjustment -- Ajuste aplicado por fuerza de calendario.
                      Positivo = enfrentó defensas mejores que la media.
    """
    team:       str
    season:     int
    up_to_week: int

    # EPA ajustado (shrinkage + strength of schedule)
    epa_off: float = _DEFAULT_LEAGUE_EPA
    epa_def: float = _DEFAULT_LEAGUE_EPA

    # EPA crudo, para trazabilidad
    epa_off_raw: float = _DEFAULT_LEAGUE_EPA
    epa_def_raw: float = _DEFAULT_LEAGUE_EPA

    # Desglose por tipo de jugada
    epa_pass_off: float = _DEFAULT_LEAGUE_EPA
    epa_rush_off: float = _DEFAULT_LEAGUE_EPA
    epa_pass_def: float = _DEFAULT_LEAGUE_EPA
    epa_rush_def: float = _DEFAULT_LEAGUE_EPA

    # Eficiencia
    success_off: float = _DEFAULT_LEAGUE_SUCCESS
    success_def: float = _DEFAULT_LEAGUE_SUCCESS

    # Resultado
    points_per_game:         float = _DEFAULT_POINTS_PER_GAME
    points_allowed_per_game: float = _DEFAULT_POINTS_PER_GAME
    recent_scores: list[float] = field(default_factory=list)

    # Muestra
    games_played:   int = 0
    plays_off:      int = 0
    plays_def:      int = 0
    sos_adjustment: float = 0.0

    # Medias de liga del mismo corte, para calcular los índices
    league: LeagueAverages | None = None

    # ── Índices normalizados para el contrato del Core ────────────────────────

    def offense_index(self) -> float:
        """
        Índice ofensivo normalizado. 1.0 = media de liga, >1 = mejor.

        Transformación aditiva (no ratio): EPA promedia ~0.00 y puede
        ser negativo, así que dividir entre la media de liga no es
        viable. Ver la nota de diseño en la cabecera del módulo.
        """
        league_epa = self.league.epa_per_play if self.league else _DEFAULT_LEAGUE_EPA
        index = 1.0 + (self.epa_off - league_epa) * _EPA_TO_INDEX_FACTOR
        return round(_clamp(index, _INDEX_MIN, _INDEX_MAX), 4)

    def defense_index(self) -> float:
        """
        Índice defensivo normalizado. 1.0 = media de liga, >1 = mejor.

        El signo se invierte respecto al ofensivo: en defensa, MENOS
        EPA permitido es mejor. Un equipo que permite -0.05 EPA/play
        contra una liga que permite 0.00 tiene defense_index > 1.

        Esta convención (mayor = mejor) es la misma que usa el plugin
        MLB con LEAGUE_ERA / ERA. El Core depende de que todos los
        plugins la respeten: BlendingEngine y los modelos de proyección
        asumen que un índice alto significa mejor rendimiento.
        """
        league_epa = self.league.epa_per_play if self.league else _DEFAULT_LEAGUE_EPA
        index = 1.0 + (league_epa - self.epa_def) * _EPA_TO_INDEX_FACTOR
        return round(_clamp(index, _INDEX_MIN, _INDEX_MAX), 4)

    def offense_index_vs(self, opponent: NFLTeamStats | None) -> float:
        """
        Índice ofensivo ajustado al emparejamiento concreto.

        El desglose pase/carrera permite capturar asimetrías: una
        ofensiva basada en el pase contra una defensa débil contra el
        pase rinde por encima de lo que sugieren los índices globales.

        Si no hay datos del rival, cae al índice general.
        """
        if opponent is None or self.league is None:
            return self.offense_index()

        league_epa = self.league.epa_per_play

        # Ventaja en cada faceta: EPA propio menos EPA permitido por el rival
        pass_edge = self.epa_pass_off - opponent.epa_pass_def
        rush_edge = self.epa_rush_off - opponent.epa_rush_def

        # Peso por volumen real de la ofensiva. La NFL moderna pasa en
        # ~58% de las jugadas de scrimmage; si no hay datos de volumen
        # usamos ese reparto.
        pass_share = self._pass_share()
        combined   = pass_edge * pass_share + rush_edge * (1.0 - pass_share)

        index = 1.0 + (combined - league_epa) * _EPA_TO_INDEX_FACTOR
        return round(_clamp(index, _INDEX_MIN, _INDEX_MAX), 4)

    def _pass_share(self) -> float:
        """Proporción de jugadas de pase. 0.58 si no hay datos."""
        return 0.58

    @property
    def has_sufficient_sample(self) -> bool:
        """
        True si la muestra permite confiar en las métricas.

        300 jugadas ≈ 5 partidos. Por debajo de eso, el shrinkage
        domina y las métricas están casi pegadas a la media de liga —
        el modelo no aporta información sobre el mercado.
        """
        return self.plays_off >= 300 and self.games_played >= 4

    @property
    def net_epa(self) -> float:
        """
        EPA neto: ofensivo menos permitido.

        Es el mejor predictor individual de rendimiento futuro en NFL.
        Un valor de +0.10 sitúa al equipo entre los mejores de la liga.
        """
        return round(self.epa_off - self.epa_def, 4)

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "epa_off":        round(self.epa_off, 4),
            "epa_def":        round(self.epa_def, 4),
            "epa_off_raw":    round(self.epa_off_raw, 4),
            "epa_def_raw":    round(self.epa_def_raw, 4),
            "net_epa":        self.net_epa,
            "epa_pass_off":   round(self.epa_pass_off, 4),
            "epa_rush_off":   round(self.epa_rush_off, 4),
            "epa_pass_def":   round(self.epa_pass_def, 4),
            "epa_rush_def":   round(self.epa_rush_def, 4),
            "success_off":    round(self.success_off, 4),
            "success_def":    round(self.success_def, 4),
            "points_per_game":         round(self.points_per_game, 2),
            "points_allowed_per_game": round(self.points_allowed_per_game, 2),
            "games_played":   self.games_played,
            "plays_off":      self.plays_off,
            "sos_adjustment": round(self.sos_adjustment, 4),
            "up_to_week":     self.up_to_week,
        }


# ── Acumuladores internos ─────────────────────────────────────────────────────

class _TeamAccumulator:
    """
    Acumulador mutable usado durante la pasada sobre el play-by-play.

    Se mantiene separado de NFLTeamStats (inmutable) deliberadamente:
    la fase de agregación necesita mutación, el resultado publicado no.
    Mezclar ambos roles en una sola clase abriría la puerta a que un
    consumidor modificara estadísticas ya calculadas.
    """

    __slots__ = (
        "team", "epa_off_sum", "plays_off", "success_off_sum",
        "epa_def_sum", "plays_def", "success_def_sum",
        "epa_pass_off_sum", "plays_pass_off",
        "epa_rush_off_sum", "plays_rush_off",
        "epa_pass_def_sum", "plays_pass_def",
        "epa_rush_def_sum", "plays_rush_def",
        "opponents",
    )

    def __init__(self, team: str) -> None:
        self.team = team
        self.epa_off_sum = 0.0
        self.plays_off = 0
        self.success_off_sum = 0.0
        self.epa_def_sum = 0.0
        self.plays_def = 0
        self.success_def_sum = 0.0
        self.epa_pass_off_sum = 0.0
        self.plays_pass_off = 0
        self.epa_rush_off_sum = 0.0
        self.plays_rush_off = 0
        self.epa_pass_def_sum = 0.0
        self.plays_pass_def = 0
        self.epa_rush_def_sum = 0.0
        self.plays_rush_def = 0
        self.opponents: list[str] = []

    def mean_epa_off(self) -> float:
        return self.epa_off_sum / self.plays_off if self.plays_off else 0.0

    def mean_epa_def(self) -> float:
        return self.epa_def_sum / self.plays_def if self.plays_def else 0.0

    def mean_success_off(self) -> float:
        return (
            self.success_off_sum / self.plays_off
            if self.plays_off else _DEFAULT_LEAGUE_SUCCESS
        )

    def mean_success_def(self) -> float:
        return (
            self.success_def_sum / self.plays_def
            if self.plays_def else _DEFAULT_LEAGUE_SUCCESS
        )

    def mean_epa_pass_off(self) -> float:
        return (
            self.epa_pass_off_sum / self.plays_pass_off
            if self.plays_pass_off else 0.0
        )

    def mean_epa_rush_off(self) -> float:
        return (
            self.epa_rush_off_sum / self.plays_rush_off
            if self.plays_rush_off else 0.0
        )

    def mean_epa_pass_def(self) -> float:
        return (
            self.epa_pass_def_sum / self.plays_pass_def
            if self.plays_pass_def else 0.0
        )

    def mean_epa_rush_def(self) -> float:
        return (
            self.epa_rush_def_sum / self.plays_rush_def
            if self.plays_rush_def else 0.0
        )


# ── Fetcher principal ─────────────────────────────────────────────────────────

class NFLTeamStatsFetcher:
    """
    Calcula métricas de equipo desde el play-by-play de nflverse.

    Parámetros
    ----------
    data_source      -- NFLDataSource compartido. Si None, crea uno.
    season           -- Temporada. None = actual.
    schedule_fetcher -- NFLScheduleFetcher para obtener los marcadores
                       por partido. Opcional: sin él, points_per_game
                       queda en el valor de liga por defecto.
    apply_shrinkage  -- Regresión a la media en muestras pequeñas.
                       Default True. Desactivar solo en análisis donde
                       se quiera ver el dato crudo.
    apply_sos        -- Ajuste por fuerza de calendario. Default True.
    """

    def __init__(
        self,
        data_source:      NFLDataSource | None = None,
        season:           int | None = None,
        schedule_fetcher = None,
        apply_shrinkage:  bool = True,
        apply_sos:        bool = True,
    ) -> None:
        self._source    = data_source or NFLDataSource()
        self._season    = season or _current_nfl_season()
        self._schedule  = schedule_fetcher
        self._shrinkage = apply_shrinkage
        self._sos       = apply_sos

        # Caché por semana de corte: {up_to_week: {team: NFLTeamStats}}
        self._cache: dict[int, dict[str, NFLTeamStats]] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def fetch(self, team: str, up_to_week: int) -> NFLTeamStats:
        """
        Estadísticas de un equipo usando solo datos hasta `up_to_week`.

        Parámetros
        ----------
        team        -- Abreviación del equipo ('KC', 'SF').
        up_to_week  -- Última semana INCLUIDA en el cálculo. Para
                       proyectar la semana 10 se pasa 9.

                       Sin valor por defecto de forma deliberada:
                       omitirlo debe ser un error visible, no un fallo
                       silencioso que introduce look-ahead bias en un
                       backtest.

        Retorna
        -------
        NFLTeamStats. Si no hay datos, retorna un objeto con los
        valores de liga por defecto y games_played=0 — nunca lanza.
        """
        all_stats = self.fetch_all(up_to_week)
        return all_stats.get(team) or NFLTeamStats(
            team       = team,
            season     = self._season,
            up_to_week = up_to_week,
            league     = self.league_averages(up_to_week),
        )

    def fetch_all(self, up_to_week: int) -> dict[str, NFLTeamStats]:
        """
        Estadísticas de todos los equipos hasta la semana dada.

        Una sola pasada sobre el play-by-play produce las estadísticas
        de los 32 equipos. Llamar fetch() por equipo reutiliza este
        resultado desde caché.
        """
        if up_to_week in self._cache:
            return self._cache[up_to_week]

        accumulators = self._accumulate(up_to_week)
        if not accumulators:
            self._cache[up_to_week] = {}
            return {}

        league = self._compute_league(accumulators, up_to_week)
        scores = self._team_scores(up_to_week)

        # Ajuste por fuerza de calendario: requiere las medias crudas de
        # todos los equipos, por eso se calcula tras la primera pasada.
        sos = self._compute_sos(accumulators, league) if self._sos else {}

        stats: dict[str, NFLTeamStats] = {}
        for team, acc in accumulators.items():
            stats[team] = self._build_stats(
                acc        = acc,
                league     = league,
                sos_adj    = sos.get(team, 0.0),
                scores     = scores.get(team, ([], [])),
                up_to_week = up_to_week,
            )

        self._cache[up_to_week] = stats
        return stats

    def league_averages(self, up_to_week: int) -> LeagueAverages:
        """Medias de liga hasta la semana dada."""
        accumulators = self._accumulate(up_to_week)
        return self._compute_league(accumulators, up_to_week)

    def clear_cache(self) -> None:
        """Limpia el caché de estadísticas calculadas."""
        self._cache.clear()

    # ── Agregación del play-by-play ───────────────────────────────────────────

    def _accumulate(self, up_to_week: int) -> dict[str, _TeamAccumulator]:
        """
        Una pasada sobre el play-by-play acumulando por equipo.

        Nota de diseño: se itera con itertuples() en vez de usar
        groupby/boolean masking de pandas. Además de evitar los
        problemas de tipado documentados en schedule.py, mantiene la
        lógica de filtrado (qué jugada cuenta y cuál no) explícita y
        auditable en Python plano en vez de escondida en expresiones
        de pandas.

        El coste es asumible: ~50.000 jugadas por temporada, una
        pasada de ~1 segundo que además queda cacheada.
        """
        try:
            df = self._source.load_pbp([self._season])
        except Exception:
            return {}

        if df is None or len(df) == 0:
            return {}

        accumulators: dict[str, _TeamAccumulator] = {}

        def acc_for(team: str) -> _TeamAccumulator:
            if team not in accumulators:
                accumulators[team] = _TeamAccumulator(team)
            return accumulators[team]

        for row in df.itertuples(index=False):
            # ── Corte temporal: la barrera contra el look-ahead bias ──
            week = _safe_int(getattr(row, "week", None))
            if week is None or week > up_to_week:
                continue

            # ── Solo jugadas de scrimmage ──
            play_type = _safe_str(getattr(row, "play_type", None))
            if play_type not in _SCRIMMAGE_PLAYS:
                continue

            epa = _safe_float(getattr(row, "epa", None))
            if epa is None:
                continue  # jugadas anuladas no tienen EPA

            posteam = _safe_str(getattr(row, "posteam", None))
            defteam = _safe_str(getattr(row, "defteam", None))
            if not posteam or not defteam:
                continue

            success = _safe_float(getattr(row, "success", None)) or 0.0
            is_pass = play_type == "pass"

            # ── Ofensiva ──
            off = acc_for(posteam)
            off.epa_off_sum += epa
            off.plays_off += 1
            off.success_off_sum += success
            if is_pass:
                off.epa_pass_off_sum += epa
                off.plays_pass_off += 1
            else:
                off.epa_rush_off_sum += epa
                off.plays_rush_off += 1
            off.opponents.append(defteam)

            # ── Defensa ──
            dfn = acc_for(defteam)
            dfn.epa_def_sum += epa
            dfn.plays_def += 1
            dfn.success_def_sum += success
            if is_pass:
                dfn.epa_pass_def_sum += epa
                dfn.plays_pass_def += 1
            else:
                dfn.epa_rush_def_sum += epa
                dfn.plays_rush_def += 1

        return accumulators

    # ── Medias de liga ────────────────────────────────────────────────────────

    @staticmethod
    def _compute_league(
        accumulators: dict[str, _TeamAccumulator],
        up_to_week:   int,
    ) -> LeagueAverages:
        """
        Medias de liga ponderadas por volumen de jugadas.

        Se pondera por jugadas y no por equipo: un equipo con más
        posesiones debe pesar más en la media, igual que pesa más en
        la distribución real de jugadas de la liga.
        """
        if not accumulators:
            return LeagueAverages(season=0, up_to_week=up_to_week)

        total_epa     = sum(a.epa_off_sum for a in accumulators.values())
        total_plays   = sum(a.plays_off for a in accumulators.values())
        total_success = sum(a.success_off_sum for a in accumulators.values())

        pass_epa   = sum(a.epa_pass_off_sum for a in accumulators.values())
        pass_plays = sum(a.plays_pass_off for a in accumulators.values())
        rush_epa   = sum(a.epa_rush_off_sum for a in accumulators.values())
        rush_plays = sum(a.plays_rush_off for a in accumulators.values())

        season = 0
        return LeagueAverages(
            season       = season,
            up_to_week   = up_to_week,
            epa_per_play = total_epa / total_plays if total_plays else _DEFAULT_LEAGUE_EPA,
            success_rate = (
                total_success / total_plays
                if total_plays else _DEFAULT_LEAGUE_SUCCESS
            ),
            epa_pass     = pass_epa / pass_plays if pass_plays else _DEFAULT_LEAGUE_EPA,
            epa_rush     = rush_epa / rush_plays if rush_plays else _DEFAULT_LEAGUE_EPA,
            total_plays  = total_plays,
        )

    # ── Ajuste por fuerza de calendario ───────────────────────────────────────

    @staticmethod
    def _compute_sos(
        accumulators: dict[str, _TeamAccumulator],
        league:       LeagueAverages,
    ) -> dict[str, float]:
        """
        Ajuste de una pasada por la calidad de las defensas enfrentadas.

        Para cada equipo se promedia la desviación de sus rivales
        respecto a la media de liga en EPA permitido. Si un equipo
        enfrentó defensas que permiten 0.03 EPA/play por encima de la
        media, su EPA ofensivo está inflado en aproximadamente esa
        cantidad y se le resta.

        Aproximación consciente de DVOA: el cálculo real de Football
        Outsiders es iterativo (ajusta al rival, recalcula, repite
        hasta converger) y pondera además situación de juego. Una
        pasada captura la mayor parte del efecto sin el riesgo de
        no convergencia ni el coste computacional.
        """
        # EPA permitido por cada defensa
        def_epa = {
            team: acc.mean_epa_def()
            for team, acc in accumulators.items()
            if acc.plays_def > 0
        }
        if not def_epa:
            return {}

        league_def = league.epa_per_play

        adjustments: dict[str, float] = {}
        for team, acc in accumulators.items():
            if not acc.opponents:
                adjustments[team] = 0.0
                continue
            # Media de la desviación de los rivales respecto a la liga,
            # ponderada implícitamente por número de jugadas contra cada uno
            # (cada jugada añade una entrada a opponents).
            deviations = [
                def_epa[opp] - league_def
                for opp in acc.opponents
                if opp in def_epa
            ]
            adjustments[team] = (
                sum(deviations) / len(deviations) if deviations else 0.0
            )
        return adjustments

    # ── Marcadores por partido ────────────────────────────────────────────────

    def _team_scores(
        self,
        up_to_week: int,
    ) -> dict[str, tuple[list[float], list[float]]]:
        """
        Puntos anotados y permitidos por equipo, partido a partido.

        Retorna {team: (anotados, permitidos)} con las listas ordenadas
        del partido más antiguo al más reciente.

        Sin schedule_fetcher inyectado retorna vacío: points_per_game
        cae al valor de liga por defecto. Es una degradación aceptable
        porque el peso de los puntos en la proyección es solo 15%
        (config: nfl.projection.recent_form_weight) frente al 70% de EPA.
        """
        if self._schedule is None:
            return {}

        try:
            games = self._schedule._games()
        except Exception:
            return {}

        scores: dict[str, tuple[list[float], list[float]]] = {}
        for game in sorted(games, key=lambda g: g.gameday):
            if game.week > up_to_week or not game.is_final:
                continue
            if game.home_score is None or game.away_score is None:
                continue

            for team, scored, allowed in (
                (game.home_team, game.home_score, game.away_score),
                (game.away_team, game.away_score, game.home_score),
            ):
                if not team:
                    continue
                if team not in scores:
                    scores[team] = ([], [])
                scores[team][0].append(float(scored))
                scores[team][1].append(float(allowed))

        return scores

    # ── Construcción del resultado ────────────────────────────────────────────

    def _build_stats(
        self,
        acc:        _TeamAccumulator,
        league:     LeagueAverages,
        sos_adj:    float,
        scores:     tuple[list[float], list[float]],
        up_to_week: int,
    ) -> NFLTeamStats:
        """Convierte un acumulador en NFLTeamStats con los ajustes aplicados."""
        raw_off = acc.mean_epa_off()
        raw_def = acc.mean_epa_def()

        # 1. Ajuste por calendario sobre el EPA crudo
        adj_off = raw_off - sos_adj if self._sos else raw_off
        adj_def = raw_def + sos_adj if self._sos else raw_def

        # 2. Regresión a la media según el tamaño de muestra
        if self._shrinkage:
            epa_off = _shrink(adj_off, acc.plays_off, league.epa_per_play)
            epa_def = _shrink(adj_def, acc.plays_def, league.epa_per_play)
        else:
            epa_off, epa_def = adj_off, adj_def

        scored, allowed = scores
        games = len(scored)

        return NFLTeamStats(
            team       = acc.team,
            season     = self._season,
            up_to_week = up_to_week,
            epa_off      = epa_off,
            epa_def      = epa_def,
            epa_off_raw  = raw_off,
            epa_def_raw  = raw_def,
            epa_pass_off = acc.mean_epa_pass_off(),
            epa_rush_off = acc.mean_epa_rush_off(),
            epa_pass_def = acc.mean_epa_pass_def(),
            epa_rush_def = acc.mean_epa_rush_def(),
            success_off  = acc.mean_success_off(),
            success_def  = acc.mean_success_def(),
            points_per_game = (
                sum(scored) / games if games else _DEFAULT_POINTS_PER_GAME
            ),
            points_allowed_per_game = (
                sum(allowed) / games if games else _DEFAULT_POINTS_PER_GAME
            ),
            recent_scores  = scored,
            games_played   = games,
            plays_off      = acc.plays_off,
            plays_def      = acc.plays_def,
            sos_adjustment = sos_adj,
            league         = league,
        )


# ── Utilidades de módulo ──────────────────────────────────────────────────────

def _shrink(observed: float, n: int, prior: float,
            k: float = _SHRINKAGE_PLAYS) -> float:
    """
    Regresión a la media bayesiana.

        resultado = (n × observado + k × prior) / (n + k)

    Con k = 200 jugadas (≈3 partidos):
        Semana 3  (~200 jugadas) → 50% equipo, 50% liga
        Semana 8  (~520 jugadas) → 72% equipo
        Semana 14 (~900 jugadas) → 82% equipo

    El efecto es exactamente el deseado: al inicio de temporada el
    modelo apenas se despega del mercado, y gana confianza conforme
    la muestra crece.
    """
    if n <= 0:
        return prior
    return (n * observed + k * prior) / (n + k)


def _clamp(value: float, lo: float, hi: float) -> float:
    """Acota un valor al rango [lo, hi]."""
    return max(lo, min(hi, value))


def _is_nan(value) -> bool:
    """True si el valor es NaN."""
    try:
        return value != value
    except Exception:
        return False


def _safe_float(value) -> float | None:
    """Convierte a float de forma segura (NaN → None)."""
    if value is None or _is_nan(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _safe_int(value) -> int | None:
    """Convierte a int de forma segura (NaN → None)."""
    if value is None or _is_nan(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_str(value) -> str | None:
    """Convierte a str, retornando None para NaN o vacíos."""
    if value is None or _is_nan(value):
        return None
    s = str(value).strip()
    return s if s and s.lower() != "nan" else None