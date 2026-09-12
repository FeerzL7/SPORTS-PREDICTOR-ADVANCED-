"""
sports/nfl/h2h.py

NFLH2HFetcher: historial de enfrentamientos directos en NFL.

Advertencia sobre la fuerza real de esta señal
-----------------------------------------------
El H2H es MUCHO más débil en NFL que en MLB, y el módulo está diseñado
para reflejar esa debilidad en vez de disimularla.

    Enfrentamientos por temporada     MLB          NFL
      dentro de división              13-19        2
      fuera de división               6-7          ~0.3
    Rotación de plantilla anual       ~10%         ~25%
    Cambios de cuerpo técnico         raros        frecuentes

Con esa rotación, un partido de hace cuatro temporadas describe a dos
equipos que en la práctica ya no existen: cambió el quarterback, la
línea ofensiva y probablemente el coordinador defensivo.

Fuera de división el problema es peor: el calendario NFL empareja a dos
equipos de conferencias distintas una vez cada tres o cuatro años. Un
"historial" de uno o dos partidos no es una muestra, es una anécdota.

Por eso este módulo hace tres cosas que el de MLB no necesita:

    1. DECAIMIENTO POR ANTIGÜEDAD
       Cada temporada de distancia reduce el peso del encuentro. No es
       un adorno: sin él, un 3-0 histórico de 2019 pesaría igual que
       uno de la temporada pasada.

    2. VENTANA MÁS CORTA
       Tres temporadas por defecto, no cinco. Más allá, la rotación de
       plantilla hace que los datos describan equipos distintos.

    3. CONFIANZA DIFERENCIADA POR DIVISIÓN
       Un H2H divisional con 6 encuentros en 3 años es utilizable. Uno
       no divisional con 1 encuentro no lo es, y el módulo lo declara
       con `is_reliable = False` en vez de entregar un win_rate de
       0% o 100% que el modelo tomaría en serio.

Lo que sí está documentado: la compresión divisional
------------------------------------------------------
Independientemente del historial concreto, los partidos de división
quedan sistemáticamente más cerrados de lo que sugieren los ratings de
los equipos. Los rivales se enfrentan dos veces al año, se conocen los
esquemas y preparan específicamente contra ellos.

Ese efecto NO viene del H2H sino de la propia condición divisional, y
config/nfl.yaml lo recoge como `divisional_spread_compression: 0.85`.
El módulo lo expone en `spread_compression()` porque es el consumidor
natural: quien pregunta por el historial entre dos equipos es quien
necesita saber si son rivales de división.

Reutilización del Core
------------------------
El cálculo estadístico genérico (win rate, medias, totales) vive en
core/utils/h2h_base.py y lo comparten todos los deportes. Este módulo
aporta solo lo específico de NFL: obtener los encuentros desde nflverse,
el decaimiento por antigüedad y la lectura divisional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.utils.h2h_base import H2HMetrics, compute_h2h

from sports.nfl.schedule import NFLGameInfo, GAME_TYPE_REGULAR


# ── Dependencias inyectadas ──────────────────────────────────────────────────

@runtime_checkable
class ScheduleSource(Protocol):
    """
    Interfaz mínima del calendario que este módulo consume.

    Mismo patrón que rest.py y context.py: declarar solo lo usado
    documenta la dependencia real y evita que el type checker propague
    `Unknown` desde una dependencia sin anotar.
    """

    def get_team_schedule(self, team: str) -> list[NFLGameInfo]:
        """Partidos de un equipo ordenados por fecha."""
        ...

    def get_game_info(self, game_id: str) -> NFLGameInfo | None:
        """Metadatos de un partido, o None si no existe."""
        ...


# ── Constantes de calibración ────────────────────────────────────────────────

# Ventana histórica por defecto, en temporadas.
#
# Tres y no cinco: con ~25% de rotación anual de plantilla, a las cuatro
# temporadas el solapamiento de jugadores entre los dos equipos del
# encuentro histórico y los actuales es minoritario.
_DEFAULT_SEASONS_BACK: int = 3

# Factor de decaimiento por temporada de antigüedad.
#
# Un encuentro de la temporada en curso pesa 1.0; el de la anterior
# 0.65; el de dos atrás 0.42. Calibrado de forma que a las tres
# temporadas el peso caiga por debajo de 0.3 — coherente con la tasa de
# rotación de plantilla.
_SEASON_DECAY: float = 0.65

# Encuentros mínimos para considerar el H2H informativo.
#
# Se diferencia por tipo de emparejamiento porque la tasa de
# acumulación es radicalmente distinta: dos rivales de división juegan
# 2 veces al año (6 en la ventana), mientras que dos equipos de
# conferencias distintas pueden no haberse cruzado nunca en 3 años.
_MIN_MEETINGS_DIVISIONAL:     int = 4
_MIN_MEETINGS_NON_DIVISIONAL: int = 3

# Compresión del spread en partidos divisionales.
# Valor por defecto si no hay ConfigLoader; coincide con nfl.yaml.
_DEFAULT_DIVISIONAL_COMPRESSION: float = 0.85


@dataclass(frozen=True)
class NFLH2HResult:
    """
    Resultado del análisis H2H entre dos equipos NFL.

    Campos
    ------
    home_team      -- Equipo local del partido a proyectar.
    away_team      -- Equipo visitante.
    metrics        -- Métricas genéricas calculadas por el Core.
    n_meetings     -- Encuentros hallados en la ventana.
    is_divisional  -- True si son rivales de división.
    is_reliable    -- True si la muestra alcanza el mínimo para el tipo
                      de emparejamiento. Cuando es False, el modelo debe
                      ignorar las métricas: un win_rate de 100% sobre un
                      único partido no es información.
    weighted_margin -- Margen medio ponderado por antigüedad, desde la
                      perspectiva del local. Positivo = el local suele
                      ganar estos enfrentamientos.
    weighted_total  -- Total medio ponderado por antigüedad.
    effective_sample -- Suma de los pesos de decaimiento. Mide la
                      muestra REAL en términos de información: seis
                      encuentros repartidos en tres temporadas dan
                      menos información que seis en una.
    seasons_covered -- Temporadas distintas presentes en la muestra.
    """
    home_team:       str
    away_team:       str
    metrics:         H2HMetrics
    n_meetings:      int = 0
    is_divisional:   bool = False
    is_reliable:     bool = False
    weighted_margin: float | None = None
    weighted_total:  float | None = None
    effective_sample: float = 0.0
    seasons_covered: int = 0

    @property
    def has_data(self) -> bool:
        """True si se encontró al menos un encuentro."""
        return self.n_meetings > 0

    def to_metadata(self) -> dict:
        """
        Convierte a dict para TeamFeatures.sport_metadata.

        Las métricas numéricas se emiten SOLO si la muestra es fiable.
        Emitir un win_rate calculado sobre un único encuentro invitaría
        al modelo a usarlo como si fuera señal, que es exactamente el
        error que `is_reliable` existe para prevenir.
        """
        base = {
            "h2h_n_meetings":       self.n_meetings,
            "h2h_is_divisional":    self.is_divisional,
            "h2h_is_reliable":      self.is_reliable,
            "h2h_effective_sample": round(self.effective_sample, 3),
            "h2h_seasons_covered":  self.seasons_covered,
        }
        if self.is_reliable:
            base.update({
                "h2h_home_win_rate":  self.metrics.win_rate_a,
                "h2h_avg_total":      self.metrics.avg_total,
                "h2h_weighted_margin": self.weighted_margin,
                "h2h_weighted_total":  self.weighted_total,
            })
        return base


class NFLH2HFetcher:
    """
    Historial de enfrentamientos directos entre equipos NFL.

    Parámetros
    ----------
    schedule_fetcher -- Fuente del calendario. Obligatorio: los
                        encuentros se derivan de él.
    config_loader    -- ConfigLoader con nfl.yaml, para la compresión
                        divisional.
    seasons_back     -- Ventana histórica en temporadas. Default 3.
    include_playoffs -- Si True, incluye encuentros de postemporada.
                        Default True: un enfrentamiento de playoffs es
                        tan informativo como uno de temporada regular,
                        y la muestra NFL es lo bastante escasa para no
                        descartar datos válidos.
    """

    def __init__(
        self,
        schedule_fetcher: ScheduleSource,
        config_loader    = None,
        seasons_back:     int = _DEFAULT_SEASONS_BACK,
        include_playoffs: bool = True,
    ) -> None:
        self._schedule: ScheduleSource = schedule_fetcher
        self._config       = config_loader
        self._seasons_back = seasons_back
        self._playoffs     = include_playoffs

        self._divisional_compression = self._cfg(
            "nfl.divisional_spread_compression",
            _DEFAULT_DIVISIONAL_COMPRESSION,
        )

        # Caché: {(home, away, season, is_divisional): NFLH2HResult}
        #
        # `is_divisional` forma parte de la clave porque CAMBIA el
        # resultado: determina el umbral de fiabilidad (4 encuentros
        # para divisionales, 3 para el resto). Omitirlo hacía que una
        # segunda consulta con el valor contrario devolviera el
        # resultado cacheado de la primera — un fallo silencioso que
        # solo se manifestaba como una fiabilidad mal calculada.
        self._cache: dict[
            tuple[str, str, int, bool | None], NFLH2HResult
        ] = {}

    # ── API pública ───────────────────────────────────────────────────────────

    def get_h2h(
        self,
        home_team:      str,
        away_team:      str,
        current_season: int,
        is_divisional:  bool | None = None,
    ) -> NFLH2HResult:
        """
        Historial entre dos equipos hasta la temporada dada.

        Parámetros
        ----------
        home_team      -- Local del partido a proyectar.
        away_team      -- Visitante.
        current_season -- Temporada del partido a proyectar. Los
                          encuentros se buscan en las `seasons_back`
                          temporadas anteriores más la actual.
        is_divisional  -- Si se conoce (desde NFLGameInfo.div_game),
                          pasarlo evita inferirlo. Si es None se deduce
                          de la frecuencia de encuentros: dos equipos
                          que se ven 2 veces por temporada son rivales
                          de división por construcción del calendario.

        Retorna
        -------
        NFLH2HResult. Nunca lanza: sin datos retorna un resultado con
        `has_data=False` e `is_reliable=False`.
        """
        key = (home_team, away_team, current_season, is_divisional)
        if key in self._cache:
            return self._cache[key]

        result = self._compute(home_team, away_team, current_season, is_divisional)
        self._cache[key] = result
        return result

    def spread_compression(self, is_divisional: bool) -> float:
        """
        Factor de compresión del spread proyectado.

        Los partidos divisionales quedan sistemáticamente más cerrados
        de lo que sugieren los ratings: los equipos se enfrentan dos
        veces al año, conocen los esquemas rivales y preparan
        específicamente contra ellos.

        El efecto es independiente del historial concreto — deriva de
        la condición divisional en sí. Un spread proyectado de +7.0 en
        un partido de división se comprime a +5.95 con el factor por
        defecto de 0.85.

        Retorna 1.0 (sin compresión) para partidos no divisionales.
        """
        return self._divisional_compression if is_divisional else 1.0

    def clear_cache(self) -> None:
        """Limpia el caché de resultados."""
        self._cache.clear()

    # ── Cálculo ───────────────────────────────────────────────────────────────

    def _compute(
        self,
        home_team:      str,
        away_team:      str,
        current_season: int,
        is_divisional:  bool | None,
    ) -> NFLH2HResult:
        meetings = self._find_meetings(home_team, away_team, current_season)

        empty_metrics = compute_h2h(home_team, away_team, [])

        if not meetings:
            return NFLH2HResult(
                home_team     = home_team,
                away_team     = away_team,
                metrics       = empty_metrics,
                is_divisional = bool(is_divisional),
            )

        # Métricas genéricas via el Core
        metrics = compute_h2h(
            team_a_id = home_team,
            team_b_id = away_team,
            meetings  = [m["record"] for m in meetings],
        )

        # ── Divisional: dato explícito o inferido de la frecuencia ──
        seasons = {m["season"] for m in meetings}
        divisional = (
            bool(is_divisional) if is_divisional is not None
            else self._infer_divisional(len(meetings), len(seasons))
        )

        # ── Agregados ponderados por antigüedad ──
        weighted = self._weighted_aggregates(meetings, reference_team=home_team)

        min_required = (
            _MIN_MEETINGS_DIVISIONAL if divisional
            else _MIN_MEETINGS_NON_DIVISIONAL
        )

        return NFLH2HResult(
            home_team        = home_team,
            away_team        = away_team,
            metrics          = metrics,
            n_meetings       = len(meetings),
            is_divisional    = divisional,
            is_reliable      = len(meetings) >= min_required,
            weighted_margin  = weighted["margin"],
            weighted_total   = weighted["total"],
            effective_sample = weighted["weight_sum"],
            seasons_covered  = len(seasons),
        )

    def _find_meetings(
        self,
        home_team:      str,
        away_team:      str,
        current_season: int,
    ) -> list[dict]:
        """
        Encuentros directos en la ventana histórica.

        Se recorre el calendario del equipo local y se filtran los
        partidos finalizados contra el visitante. Solo partidos con
        marcador: uno programado a futuro no aporta información.

        Cada entrada lleva el peso de decaimiento ya calculado, para
        que los agregados ponderados no tengan que recalcularlo.
        """
        try:
            schedule = self._schedule.get_team_schedule(home_team)
        except Exception:
            return []

        if not isinstance(schedule, list):
            return []

        min_season = current_season - self._seasons_back

        meetings: list[dict] = []
        for game in schedule:
            if not isinstance(game, NFLGameInfo):
                continue
            if game.season < min_season or game.season > current_season:
                continue
            if not self._playoffs and game.game_type != GAME_TYPE_REGULAR:
                continue
            if not game.is_final:
                continue

            # ¿Enfrentamiento entre estos dos equipos?
            teams = {game.home_team, game.away_team}
            if teams != {home_team, away_team}:
                continue
            if game.home_score is None or game.away_score is None:
                continue

            age = current_season - game.season
            meetings.append({
                "season": game.season,
                "weight": _SEASON_DECAY ** age,
                "record": {
                    "home_id":    game.home_team,
                    "away_id":    game.away_team,
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                },
            })

        return meetings

    @staticmethod
    def _weighted_aggregates(
        meetings:       list[dict],
        reference_team: str,
    ) -> dict:
        """
        Margen y total medios ponderados por antigüedad.

        El margen se normaliza a la perspectiva de `reference_team` —
        el local del partido a proyectar — NO del equipo que fue local
        en cada encuentro histórico.

        Por qué la normalización es imprescindible
        ------------------------------------------
        Los rivales de división juegan uno en cada estadio. Si el
        margen se tomara crudo como `home_score - away_score`, una
        serie donde un equipo domina daría un promedio cercano a cero
        por cancelación de signos:

            KC local  27-20  →  hs-aws = +7
            KC visita 17-24  →  hs-aws = -7   ← KC GANÓ, pero resta
                                 ────────
                                 media: 0.0

        El primer cálculo de este módulo tenía exactamente ese defecto:
        KC ganaba 4 de 6 enfrentamientos y el margen ponderado salía
        NEGATIVO. El docstring afirmaba que normalizaba la perspectiva,
        pero el código no lo hacía — una discrepancia que solo se
        detectó al comprobar el signo contra el registro de victorias.

        Con la normalización, el mismo ejemplo da:

            KC local  27-20  →  +7
            KC visita 24-17  →  +7   (invertido: KC es la referencia)
                                 ────
                                 media: +7.0

        El total NO necesita normalización: la suma de puntos es
        simétrica y no depende de qué equipo fue local.
        """
        weight_sum = sum(m["weight"] for m in meetings)
        if weight_sum <= 0:
            return {"margin": None, "total": None, "weight_sum": 0.0}

        margin_sum = 0.0
        total_sum  = 0.0

        for m in meetings:
            rec = m["record"]
            w   = m["weight"]
            hs  = float(rec["home_score"])
            aws = float(rec["away_score"])

            # Signo desde la perspectiva del equipo de referencia
            if rec["home_id"] == reference_team:
                margin_sum += w * (hs - aws)
            else:
                margin_sum += w * (aws - hs)

            total_sum += w * (hs + aws)

        return {
            "margin":     round(margin_sum / weight_sum, 3),
            "total":      round(total_sum / weight_sum, 3),
            "weight_sum": round(weight_sum, 4),
        }

    @staticmethod
    def _infer_divisional(n_meetings: int, n_seasons: int) -> bool:
        """
        Deduce si dos equipos son rivales de división.

        El calendario NFL garantiza dos enfrentamientos anuales entre
        rivales de división y como máximo uno entre equipos de
        divisiones distintas. Así que una frecuencia cercana a 2 por
        temporada implica división.

        Se usa solo cuando NFLGameInfo.div_game no está disponible. El
        dato explícito siempre es preferible: esta inferencia falla si
        el historial es corto o si hubo un cruce de playoffs que eleva
        artificialmente la frecuencia.
        """
        if n_seasons <= 0:
            return False
        return (n_meetings / n_seasons) >= 1.5

    # ── Config ────────────────────────────────────────────────────────────────

    def _cfg(self, key: str, default: float) -> float:
        """Lee un valor del ConfigLoader con fallback documentado."""
        if self._config is None:
            return default
        try:
            val = self._config.get(key, default=default)
            return float(val) if val is not None else default
        except (ValueError, TypeError, AttributeError):
            return default


# ── Adaptador de metadatos ────────────────────────────────────────────────────

def h2h_metadata(result: NFLH2HResult) -> dict:
    """
    Adaptador para TeamFeatures.sport_metadata.

    Función pública equivalente a la del plugin MLB, para que
    NFLDataProvider las consuma con la misma forma. Traduce el objeto
    de dominio al dict plano que el contrato del Core espera.
    """
    return result.to_metadata()