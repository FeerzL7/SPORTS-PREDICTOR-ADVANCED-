"""
sports/nfl/projections.py

NFLProjectionModel: convierte TeamFeatures en Projection para NFL.
Implementa core/pipeline/stage.py:ProjectionModel.

Decisión de diseño central: composición ADITIVA de señales
-----------------------------------------------------------
El plugin MLB compone los índices de forma multiplicativa:

    proj_home = recent_avg × offense_index_home / defense_index_away

Funciona en béisbol porque el rango de índices es estrecho (~0.85-1.15)
y las carreras son pocas, así que el producto no se descontrola.

En NFL ese enfoque falla por dos razones:

    RANGO MÁS AMPLIO. Los índices de team_stats.py van de 0.60 a 1.40.
    La mejor ofensiva contra la peor defensa daría
    22 × 1.30/0.70 = 40.9 puntos — un valor que ocurre, pero como
    resultado extremo, no como proyección central.

    LOS ERRORES SE MULTIPLICAN. Si el índice ofensivo tiene un error
    del 10% y el defensivo otro 10%, el producto arrastra ~21% de
    error. Con solo 17 partidos por temporada, ambos índices ya
    cargan ruido considerable.

Los pesos de config/nfl.yaml describen otra estructura:

    epa_offense_weight:   0.40
    epa_defense_weight:   0.30
    success_rate_weight:  0.15
    recent_form_weight:   0.15
                          ────
                          1.00

Cuatro señales que suman exactamente 1.0 es la firma de un PROMEDIO
PONDERADO, no de un producto. Expresando cada señal como multiplicador
sobre la media de liga y promediándolas:

    m = 0.40·(off_idx) + 0.30·(1/def_idx_rival)
      + 0.15·(success_rel) + 0.15·(forma_rel)

    proj = media_liga × m

El mismo caso extremo da 29.1 puntos, y los errores de las cuatro
señales se promedian en vez de acumularse. Es además robusto ante
señales ausentes: si falta el success rate, se redistribuye su peso
entre las demás en vez de anular el cálculo.

Orden de aplicación de los ajustes
------------------------------------
El orden importa porque hay ajustes multiplicativos y aditivos:

    1. Base            promedio ponderado de las cuatro señales
    2. Multiplicativos clima × altitud       (afectan la escala)
    3. Aditivos        campo propio, descanso, lesiones, viaje
    4. Cotas           [proj_min, proj_max] de nfl.yaml
    5. Compresión      divisional, aplicada SOLO al margen

Los multiplicativos van antes que los aditivos porque modelan
condiciones que escalan la anotación (un partido con viento de 25 mph
reduce todo el juego aéreo), mientras que los aditivos son
transferencias de puntos concretas (un QB fuera vale ~7 puntos
independientemente de cuánto anote el equipo).

La compresión divisional se aplica al final y solo al margen: los
rivales de división producen partidos más cerrados, pero no
necesariamente con menos puntos totales.

Distribución Normal, no Poisson
---------------------------------
La anotación NFL no es un proceso de conteo de eventos raros: cada
drive puede producir 0, 3, 6, 7 u 8 puntos. La distribución empírica
de puntos por equipo se aproxima bien con una Normal de μ≈22, σ≈10.

Dos sigmas distintas, como documenta nfl.yaml:

    σ_margen = 13.5   para SPREAD y ML
    σ_total  = 10.0   para TOTAL

Que la σ del margen supere a la del total parece un error pero es
correcto: los errores de proyección de ambos equipos se cancelan
parcialmente al sumar y se amplifican al restar.

Empates
---------
NFL permite empates tras tiempo extra (~0.4% de partidos). El contrato
Projection exige que las probabilidades sumen 1.0, así que hay que
modelarlos. Se calcula desde la densidad de la Normal en el entorno de
cero, acotado — es una aproximación, y está documentada como tal: un
empate exacto no es un intervalo continuo sino la coincidencia de dos
marcadores discretos.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from core.contracts.features import TeamFeatures
from core.contracts.projection import Projection


_MODEL_VERSION = "nfl-v1.0.0"

# ── Valores de liga por defecto ──────────────────────────────────────────────
# Usados cuando no hay medias de liga en sport_metadata.
_LEAGUE_PPG:          float = 22.0
_LEAGUE_SUCCESS_RATE: float = 0.45

# ── Pesos por defecto de las cuatro señales ──────────────────────────────────
# Coinciden con config/nfl.yaml sección nfl.projection.
_W_EPA_OFFENSE:  float = 0.40
_W_EPA_DEFENSE:  float = 0.30
_W_SUCCESS_RATE: float = 0.15
_W_RECENT_FORM:  float = 0.15

# ── Ajustes aditivos por defecto ─────────────────────────────────────────────
_DEFAULT_HFA: float = 1.8   # ventaja de campo, nfl.home_field_advantage

# ── Sigmas de la Normal ──────────────────────────────────────────────────────
_DEFAULT_SIGMA_MARGIN: float = 13.5
_DEFAULT_SIGMA_TOTAL:  float = 10.0
_DEFAULT_SIGMA_MIN:    float = 8.0
_DEFAULT_SIGMA_MAX:    float = 18.0

# ── Cotas de proyección ──────────────────────────────────────────────────────
_DEFAULT_PROJ_MIN: float = 6.0
_DEFAULT_PROJ_MAX: float = 45.0

# ── Empates ──────────────────────────────────────────────────────────────────
# Ancho de la banda alrededor de cero que se considera empate al
# integrar la densidad Normal. Calibrado para que un partido
# equilibrado (margen 0) produzca ~1.5% de probabilidad de empate y la
# media sobre todos los partidos quede cerca del 0.4% histórico.
_TIE_BAND: float = 0.5
_TIE_MAX:  float = 0.03   # techo: ni el partido más parejo supera 3%

# ── Penalizaciones de confianza ──────────────────────────────────────────────
_CONF_INSUFFICIENT_SAMPLE: float = 0.15
_CONF_STALE_INJURY:        float = 0.10
_CONF_QB_OUT:              float = 0.12
_CONF_EXTREME_WEATHER:     float = 0.08
_CONF_EARLY_SEASON:        float = 0.10
_CONF_MIN:                 float = 0.10

# Semana hasta la cual se considera "temporada temprana".
# Con 4 partidos jugados la muestra de EPA sigue dominada por el
# shrinkage y el modelo aporta poca información sobre el mercado.
_EARLY_SEASON_WEEK: int = 4

# Umbral de factor climático para considerarlo extremo.
_EXTREME_WEATHER_FACTOR: float = 0.95


@dataclass(frozen=True)
class _SignalSet:
    """
    Las cuatro señales que componen la proyección de un equipo.

    Cada una expresada como MULTIPLICADOR sobre la media de liga, para
    que sean promediables entre sí. None indica señal no disponible: su
    peso se redistribuye entre las presentes.

    Campos
    ------
    epa_offense  -- Índice ofensivo propio (offense_index).
    epa_defense  -- Inverso del índice defensivo del rival. Se invierte
                    porque una defensa rival débil (índice bajo) debe
                    AUMENTAR nuestra proyección.
    success_rate -- Success rate propio relativo a la liga.
    recent_form  -- Puntos recientes propios relativos a la liga.
    """
    epa_offense:  float | None = None
    epa_defense:  float | None = None
    success_rate: float | None = None
    recent_form:  float | None = None

    def blend(
        self,
        w_offense: float,
        w_defense: float,
        w_success: float,
        w_recent:  float,
    ) -> tuple[float, float]:
        """
        Promedio ponderado de las señales disponibles.

        Retorna (multiplicador, cobertura) donde cobertura es la
        fracción del peso total que sí tenía datos. Una cobertura de
        0.70 significa que el 30% de las señales faltaba y su peso se
        redistribuyó — información que el llamador usa para ajustar la
        confianza de la proyección.

        Redistribuir en vez de tratar la señal ausente como 1.0 evita
        un sesgo sistemático hacia la media: un equipo sin datos de
        success rate no es un equipo promedio en success rate, es un
        equipo del que no sabemos su success rate.
        """
        pairs = (
            (self.epa_offense,  w_offense),
            (self.epa_defense,  w_defense),
            (self.success_rate, w_success),
            (self.recent_form,  w_recent),
        )
        available = [(v, w) for v, w in pairs if v is not None and w > 0]
        if not available:
            return 1.0, 0.0

        weight_sum = sum(w for _, w in available)
        if weight_sum <= 0:
            return 1.0, 0.0

        total_weight = w_offense + w_defense + w_success + w_recent
        blended = sum(v * w for v, w in available) / weight_sum
        coverage = weight_sum / total_weight if total_weight > 0 else 0.0

        return blended, coverage


class NFLProjectionModel:
    """
    Modelo de proyección de puntos para NFL.

    Parámetros
    ----------
    config_loader     -- ConfigLoader con nfl.yaml. Sin él usa los
                         valores por defecto documentados arriba.
    injury_fetcher    -- NFLInjuryFetcher. Opcional: si no se inyecta,
                         las penalizaciones por lesión se leen de
                         sport_metadata si están presentes.
    rest_fetcher      -- NFLRestFetcher. Opcional, mismo criterio.
    h2h_fetcher       -- NFLH2HFetcher. Opcional: aporta la compresión
                         divisional.
    """

    def __init__(
        self,
        config_loader  = None,
        injury_fetcher = None,
        rest_fetcher   = None,
        h2h_fetcher    = None,
    ) -> None:
        self._config = config_loader
        self._injuries = injury_fetcher
        self._rest     = rest_fetcher
        self._h2h      = h2h_fetcher

        # Pesos de las señales
        self._w_offense = self._cfg("nfl.projection.epa_offense_weight",  _W_EPA_OFFENSE)
        self._w_defense = self._cfg("nfl.projection.epa_defense_weight",  _W_EPA_DEFENSE)
        self._w_success = self._cfg("nfl.projection.success_rate_weight", _W_SUCCESS_RATE)
        self._w_recent  = self._cfg("nfl.projection.recent_form_weight",  _W_RECENT_FORM)

        # Ajustes
        self._hfa = self._cfg("nfl.home_field_advantage", _DEFAULT_HFA)

        # Sigmas
        self._sigma_margin = self._cfg("simulation.nfl.normal.default_sigma", _DEFAULT_SIGMA_MARGIN)
        self._sigma_total  = self._cfg("simulation.nfl.normal.total_sigma",   _DEFAULT_SIGMA_TOTAL)
        self._sigma_min    = self._cfg("simulation.nfl.normal.min_sigma",     _DEFAULT_SIGMA_MIN)
        self._sigma_max    = self._cfg("simulation.nfl.normal.max_sigma",     _DEFAULT_SIGMA_MAX)

        # Cotas
        self._proj_min = self._cfg("ensemble.proj_min", _DEFAULT_PROJ_MIN)
        self._proj_max = self._cfg("ensemble.proj_max", _DEFAULT_PROJ_MAX)

    def model_version(self) -> str:
        return _MODEL_VERSION

    # ── ProjectionModel Protocol ──────────────────────────────────────────────

    def project(
        self,
        home_features: TeamFeatures,
        away_features: TeamFeatures,
        context:       dict,
    ) -> Projection:
        """
        Proyecta la anotación de ambos equipos.

        Nunca lanza: ante datos ausentes cae a la media de liga y lo
        refleja bajando `confidence`. Un pipeline que aborta por falta
        de un dato secundario es peor que uno que proyecta con
        incertidumbre declarada.
        """
        home_meta = home_features.sport_metadata or {}
        away_meta = away_features.sport_metadata or {}
        ctx = context or {}

        league_ppg = _safe_float(ctx.get("league_ppg")) or _LEAGUE_PPG

        # ── 1. Base: promedio ponderado de las cuatro señales ──────
        home_signals = self._build_signals(home_features, away_features, home_meta, away_meta)
        away_signals = self._build_signals(away_features, home_features, away_meta, home_meta)

        home_mult, home_cov = home_signals.blend(
            self._w_offense, self._w_defense, self._w_success, self._w_recent
        )
        away_mult, away_cov = away_signals.blend(
            self._w_offense, self._w_defense, self._w_success, self._w_recent
        )

        base_home = league_ppg * home_mult
        base_away = league_ppg * away_mult

        # ── 2. Multiplicativos: clima y altitud ────────────────────
        weather_factor = _safe_float(ctx.get("weather_factor")) or 1.0
        venue_factor   = _safe_float(ctx.get("venue_total_factor")) or 1.0
        scale = weather_factor * venue_factor

        scaled_home = base_home * scale
        scaled_away = base_away * scale

        # ── 3. Aditivos ────────────────────────────────────────────
        adj = self._additive_adjustments(
            home_features, away_features, home_meta, away_meta, ctx
        )

        proj_home = scaled_home + adj["home_total"]
        proj_away = scaled_away + adj["away_total"]

        # ── 4. Cotas ───────────────────────────────────────────────
        proj_home = _clamp(proj_home, self._proj_min, self._proj_max)
        proj_away = _clamp(proj_away, self._proj_min, self._proj_max)

        # ── 5. Compresión divisional, solo al margen ───────────────
        is_divisional = bool(ctx.get("is_divisional", False))
        compression = self._divisional_compression(is_divisional)

        if compression != 1.0:
            # Comprimir el margen conservando el total: se acerca cada
            # equipo a la media del partido, no se reduce la anotación.
            midpoint = (proj_home + proj_away) / 2.0
            proj_home = midpoint + (proj_home - midpoint) * compression
            proj_away = midpoint + (proj_away - midpoint) * compression

        proj_home = round(proj_home, 3)
        proj_away = round(proj_away, 3)

        # ── 6. Probabilidades desde la Normal ──────────────────────
        sigma = self._effective_sigma(home_meta, away_meta, ctx)
        margin = proj_home - proj_away
        probs = _normal_outcome_probs(margin, sigma)

        # ── 7. Confianza ───────────────────────────────────────────
        confidence = self._compute_confidence(
            home_features, away_features, home_meta, away_meta, ctx,
            coverage=min(home_cov, away_cov),
        )

        # ── 8. Trazabilidad ────────────────────────────────────────
        model_inputs = {
            # Señales
            "home_signals": _signals_dict(home_signals),
            "away_signals": _signals_dict(away_signals),
            "home_multiplier": round(home_mult, 4),
            "away_multiplier": round(away_mult, 4),
            "signal_coverage_home": round(home_cov, 3),
            "signal_coverage_away": round(away_cov, 3),
            # Base y escalado
            "league_ppg":     league_ppg,
            "base_home":      round(base_home, 3),
            "base_away":      round(base_away, 3),
            "weather_factor": weather_factor,
            "venue_factor":   venue_factor,
            # Aditivos
            **{f"adj_{k}": v for k, v in adj.items()},
            # Compresión y distribución
            "is_divisional":  is_divisional,
            "compression":    compression,
            "sigma_margin":   sigma,
            "margin":         round(margin, 3),
        }

        event_id = str(ctx.get("event_id") or home_meta.get("nfl_game_id") or "")

        return Projection(
            event_id       = event_id,
            sport          = "nfl",
            expected_home  = proj_home,
            expected_away  = proj_away,
            expected_total = round(proj_home + proj_away, 3),
            home_win_prob  = probs["home"],
            away_win_prob  = probs["away"],
            draw_prob      = probs["draw"],
            distribution   = "normal",
            distribution_params = {
                "mu_home":      proj_home,
                "mu_away":      proj_away,
                "sigma_margin": sigma,
                "sigma_total":  self._sigma_total,
            },
            confidence     = round(confidence, 4),
            model_version  = _MODEL_VERSION,
            model_inputs   = model_inputs,
        )

    # ── Construcción de señales ───────────────────────────────────────────────

    def _build_signals(
        self,
        own:        TeamFeatures,
        opponent:   TeamFeatures,
        own_meta:   dict,
        opp_meta:   dict,
    ) -> _SignalSet:
        """
        Convierte features en las cuatro señales, como multiplicadores.

        Cada señal es None si el dato subyacente falta, para que el
        peso se redistribuya en vez de asumir valor promedio.
        """
        league_ppg     = _LEAGUE_PPG
        league_success = _safe_float(own_meta.get("league_success_rate")) or _LEAGUE_SUCCESS_RATE

        # Señal 1: índice ofensivo propio.
        # Ya viene normalizado a 1.0 = media de liga desde team_stats.py.
        epa_offense = _safe_float(own.offense_index)

        # Señal 2: inverso del índice defensivo del rival.
        # Se invierte porque defense_index > 1 significa DEFENSA FUERTE,
        # lo que debe REDUCIR nuestra proyección. Sin la inversión, el
        # modelo premiaría atacar contra las mejores defensas.
        opp_def = _safe_float(opponent.defense_index)
        epa_defense = (
            1.0 / opp_def if opp_def is not None and opp_def > 0 else None
        )

        # Señal 3: success rate propio relativo a la liga.
        own_success = _safe_float(own_meta.get("success_off"))
        success_rate = (
            own_success / league_success
            if own_success is not None and league_success > 0 else None
        )

        # Señal 4: forma reciente en puntos, relativa a la liga.
        # Peso bajo por diseño (0.15): con 17 partidos por temporada los
        # promedios recientes son muy ruidosos.
        recent = _safe_float(own.recent_avg)
        if recent is None or recent <= 0:
            recent = _safe_float(own_meta.get("points_per_game"))
        recent_form = (
            recent / league_ppg
            if recent is not None and recent > 0 else None
        )

        return _SignalSet(
            epa_offense  = epa_offense,
            epa_defense  = epa_defense,
            success_rate = success_rate,
            recent_form  = recent_form,
        )

    # ── Ajustes aditivos ──────────────────────────────────────────────────────

    def _additive_adjustments(
        self,
        home:      TeamFeatures,
        away:      TeamFeatures,
        home_meta: dict,
        away_meta: dict,
        ctx:       dict,
    ) -> dict:
        """
        Ajustes en puntos, no multiplicadores.

        Modelan transferencias concretas: un QB fuera vale ~7 puntos
        independientemente de cuánto anote su equipo, a diferencia del
        clima que escala toda la anotación.

        Retorna el desglose completo para trazabilidad, más los totales
        agregados por equipo.
        """
        # Ventaja de campo: solo al local
        hfa = self._hfa

        # Lesiones: penalización propia, desde el fetcher o el metadata
        home_injury = self._injury_penalty(home_meta, home.team_id, ctx)
        away_injury = self._injury_penalty(away_meta, away.team_id, ctx)

        # Descanso: el diferencial ya viene desde la perspectiva del local
        rest_diff = self._rest_differential(ctx)

        # Viaje: penaliza al visitante. El contexto lo entrega con signo
        # negativo desde la perspectiva del visitante.
        travel = _safe_float(ctx.get("travel_adjustment")) or 0.0

        return {
            "hfa":          round(hfa, 3),
            "home_injury":  round(-home_injury, 3),
            "away_injury":  round(-away_injury, 3),
            "rest_diff":    round(rest_diff, 3),
            "travel":       round(travel, 3),
            # Totales por equipo
            "home_total":   round(hfa - home_injury + rest_diff, 3),
            "away_total":   round(-away_injury + travel, 3),
        }

    def _injury_penalty(self, meta: dict, team_id: str, ctx: dict) -> float:
        """
        Penalización por lesiones en puntos.

        Prioridad: fetcher inyectado > metadata precalculado > 0.

        El fetcher es preferible porque aplica los pesos del config,
        pero el provider puede haber precalculado la penalización y
        dejarla en sport_metadata — en ese caso se reutiliza en vez de
        recalcular.
        """
        if self._injuries is not None:
            week = _safe_int(ctx.get("week"))
            if week is not None and team_id:
                try:
                    return float(self._injuries.penalty_for(team_id, week))
                except Exception:
                    pass

        precomputed = _safe_float(meta.get("injury_penalty"))
        return precomputed if precomputed is not None else 0.0

    def _rest_differential(self, ctx: dict) -> float:
        """
        Diferencial de descanso desde la perspectiva del local.

        Prioridad: contexto precalculado > fetcher > 0.

        El contexto se prefiere porque NFLDataProvider ya lo resolvió
        con el game_id correcto; el fetcher necesitaría ese id y no
        siempre está en el contexto.
        """
        from_ctx = _safe_float(ctx.get("rest_differential"))
        if from_ctx is not None:
            return from_ctx

        if self._rest is not None:
            game_id = ctx.get("event_id") or ctx.get("nfl_game_id")
            if game_id:
                try:
                    return float(self._rest.differential(str(game_id)))
                except Exception:
                    pass
        return 0.0

    def _divisional_compression(self, is_divisional: bool) -> float:
        """Factor de compresión del margen en partidos divisionales."""
        if self._h2h is not None:
            try:
                return float(self._h2h.spread_compression(is_divisional))
            except Exception:
                pass
        if not is_divisional:
            return 1.0
        return self._cfg("nfl.divisional_spread_compression", 0.85)

    # ── Sigma efectiva ────────────────────────────────────────────────────────

    def _effective_sigma(
        self,
        home_meta: dict,
        away_meta: dict,
        ctx:       dict,
    ) -> float:
        """
        Sigma del margen ajustada a la incertidumbre del partido.

        Parte de la sigma base (13.5) y la AUMENTA cuando hay factores
        que hacen el resultado menos predecible:

            QB titular fuera   → el suplente es una incógnita
            Clima extremo      → más varianza en el juego aéreo
            Temporada temprana → muestra insuficiente en ambos equipos

        Aumentar sigma en vez de solo bajar `confidence` tiene un efecto
        concreto: ensancha la distribución y acerca las probabilidades
        al 50%, lo que reduce el EV calculado y hace que los filtros
        descarten el pick. Es la respuesta correcta a la incertidumbre
        — no apostar — en lugar de apostar con una etiqueta de baja
        confianza.
        """
        sigma = self._sigma_margin

        if home_meta.get("injury_qb_out") or away_meta.get("injury_qb_out"):
            sigma *= 1.12

        weather = _safe_float(ctx.get("weather_factor"))
        if weather is not None and weather < _EXTREME_WEATHER_FACTOR:
            sigma *= 1.08

        week = _safe_int(ctx.get("week"))
        if week is not None and week <= _EARLY_SEASON_WEEK:
            sigma *= 1.10

        return round(_clamp(sigma, self._sigma_min, self._sigma_max), 3)

    # ── Confianza ─────────────────────────────────────────────────────────────

    def _compute_confidence(
        self,
        home:      TeamFeatures,
        away:      TeamFeatures,
        home_meta: dict,
        away_meta: dict,
        ctx:       dict,
        coverage:  float,
    ) -> float:
        """
        Score de confianza en [0.10, 1.0].

        Se multiplica por la calidad de datos de ambos equipos y por la
        cobertura de señales, y se descuenta por factores concretos de
        incertidumbre.
        """
        confidence = 1.0

        # Muestra insuficiente
        if not home.has_sufficient_sample:
            confidence -= _CONF_INSUFFICIENT_SAMPLE
        if not away.has_sufficient_sample:
            confidence -= _CONF_INSUFFICIENT_SAMPLE

        # Calidad de datos declarada por el provider
        dq_home = _safe_float(home.data_quality)
        dq_away = _safe_float(away.data_quality)
        dq = ((dq_home if dq_home is not None else 1.0)
              + (dq_away if dq_away is not None else 1.0)) / 2.0
        confidence *= dq

        # Cobertura de señales: cuántas de las cuatro tenían datos
        confidence *= (0.5 + 0.5 * coverage)

        # Reporte de lesiones desactualizado (latencia de nflverse)
        if home_meta.get("injury_is_stale") or away_meta.get("injury_is_stale"):
            confidence -= _CONF_STALE_INJURY

        # QB titular fuera
        if home_meta.get("injury_qb_out") or away_meta.get("injury_qb_out"):
            confidence -= _CONF_QB_OUT

        # Clima extremo
        weather = _safe_float(ctx.get("weather_factor"))
        if weather is not None and weather < _EXTREME_WEATHER_FACTOR:
            confidence -= _CONF_EXTREME_WEATHER

        # Temporada temprana
        week = _safe_int(ctx.get("week"))
        if week is not None and week <= _EARLY_SEASON_WEEK:
            confidence -= _CONF_EARLY_SEASON

        return max(_CONF_MIN, min(1.0, confidence))

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


# ── Probabilidades desde la Normal ────────────────────────────────────────────

def _normal_cdf(x: float) -> float:
    """
    Función de distribución acumulada de la Normal estándar.

    Se usa math.erf en vez de scipy para no añadir una dependencia
    pesada por una sola función. La relación es exacta:

        Φ(x) = (1 + erf(x/√2)) / 2
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_pdf(x: float) -> float:
    """Densidad de la Normal estándar."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _normal_outcome_probs(margin: float, sigma: float) -> dict:
    """
    Probabilidades de victoria local, visitante y empate.

    El margen real se modela como Normal(margin, sigma). La
    probabilidad de victoria local es P(margen_real > 0) = Φ(margin/σ).

    Sobre los empates
    -----------------
    Un empate en NFL requiere marcadores exactamente iguales tras
    tiempo extra: es un evento discreto, no un intervalo de la Normal.
    Aproximarlo integrando la densidad en una banda estrecha alrededor
    de cero es precisamente eso — una aproximación.

    Se documenta como tal y se acota al 3%. La tasa histórica real es
    ~0.4% de los partidos, y solo los muy parejos se acercan al techo.
    Modelarlo con más precisión no aportaría nada: ningún mercado NFL
    cotiza el empate como opción.
    """
    if sigma <= 0:
        # Sin varianza el resultado es determinista
        if margin > 0:
            return {"home": 1.0, "away": 0.0, "draw": 0.0}
        if margin < 0:
            return {"home": 0.0, "away": 1.0, "draw": 0.0}
        return {"home": 0.5, "away": 0.5, "draw": 0.0}

    z = margin / sigma

    # Empate: densidad en la banda alrededor de cero
    draw = min(_normal_pdf(z) / sigma * _TIE_BAND * 2.0, _TIE_MAX)

    # Victorias, repartiendo el resto según la CDF
    home_raw = _normal_cdf(z)
    remaining = 1.0 - draw
    home = home_raw * remaining
    away = remaining - home

    return {
        "home": round(home, 4),
        "away": round(away, 4),
        "draw": round(draw, 4),
    }


# ── Utilidades ────────────────────────────────────────────────────────────────

def _signals_dict(signals: _SignalSet) -> dict:
    """Serializa un _SignalSet para model_inputs."""
    return {
        "epa_offense":  _round_or_none(signals.epa_offense),
        "epa_defense":  _round_or_none(signals.epa_defense),
        "success_rate": _round_or_none(signals.success_rate),
        "recent_form":  _round_or_none(signals.recent_form),
    }


def _round_or_none(value: float | None) -> float | None:
    return round(value, 4) if value is not None else None


def _clamp(value: float, lo: float, hi: float) -> float:
    """Acota un valor al rango [lo, hi]."""
    return max(lo, min(hi, value))


def _is_nan(value) -> bool:
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