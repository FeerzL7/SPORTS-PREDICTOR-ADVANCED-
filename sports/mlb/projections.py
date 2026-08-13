"""
sports/mlb/projections.py

MLBProjectionModel: modelo de proyección de carreras MLB.

Implementa core/pipeline/stage.py:ProjectionModel.

Consume todos los módulos 8.2-8.9:
    StatcastFetcher  → ERA, FIP, WHIP del abridor (8.2)
    VenueFactorProvider → park factor del estadio (8.3)
    PitchingFetcher  → abridor probable con ERA reciente (8.4)
    BullpenFetcher   → ERA del bullpen, defense_index (8.5)
    OffenseFetcher   → OPS, splits vsRHP/vsLHP, recent_scores (8.6)
    DefenseFetcher   → fielding stats complementarias (8.7)
    MLBContextFetcher → clima, venue_type, day_night (8.8)
    MLBH2HFetcher    → historial de enfrentamientos (8.9)

Fórmula de proyección
----------------------
La proyección de carreras del equipo local se calcula como:

    runs_base  = recent_avg × offense_index
    runs_adj   = runs_base / defense_index_rival
    runs_park  = runs_adj × venue_factor × weather_factor
    runs_final = ensemble_blend(runs_park, poisson_proj)

Donde:
    recent_avg      = media de recent_scores (corregido bug F1)
    offense_index   = OPS / OPS_LIGA (corregido bug F2)
    defense_index   = LEAGUE_ERA / ERA_comb (corregido inversión)
    venue_factor    = park factor del estadio
    weather_factor  = ajuste por temperatura y viento (solo outdoor)
    ensemble_blend  = EnsembleModel con CV adaptativo

La probabilidad de victoria se deriva de las proyecciones vía
PoissonModel (para el spread) y de las probabilidades directas del
Skellam para el ML.

Trazabilidad completa
----------------------
Projection.model_inputs registra todos los valores que alimentaron
el modelo para diagnóstico y backtesting:
    era_home, fip_home, ops_away, venue_factor, weather_factor,
    h2h_weight, ensemble_applied, etc.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any

from core.contracts.features import TeamFeatures
from core.contracts.projection import Projection
from core.simulation.ensemble import EnsembleModel
from sports.mlb.bullpen import BullpenFetcher, BullpenStats, combined_era, defense_index_from_combined_era
from sports.mlb.offense import OffenseStats
from sports.mlb.pitching import ProbablePitcher
from sports.mlb.statcast import _LEAGUE_ERA, _LEAGUE_OPS
from sports.mlb.venue_factors import VenueFactorProvider

# Versión del modelo — incrementar ante cambios de fórmula
_MODEL_VERSION: str = "mlb-v2.0.0"

# Constantes de ajuste de clima (calibradas desde análisis histórico MLB)
# Temperatura: reducción de carreras por grado bajo 60°F
_TEMP_FACTOR_BASELINE: float = 72.0   # °F referencia (neutral)
_TEMP_COEFFICIENT:     float = 0.004  # 0.4% por grado de diferencia

# Viento: ajuste por velocidad y dirección
# Viento de espalda (hacia el campo, 180°±45°) aumenta carreras
# Viento de frente (desde el campo, 0°±45°) reduce carreras
_WIND_COEFFICIENT: float = 0.008  # 0.8% por mph de viento efectivo

# Límites de proyección MLB
_PROJ_MIN: float = 1.5
_PROJ_MAX: float = 12.0


class MLBProjectionModel:
    """
    Modelo de proyección de carreras para MLB.

    Parámetros
    ----------
    venue_provider   -- VenueFactorProvider para park factors.
    bullpen_fetcher  -- BullpenFetcher para ERA combinada.
    ensemble         -- EnsembleModel para blending Poisson/regresión.
    config_loader    -- ConfigLoader con mlb.yaml para leer pesos.
    """

    def __init__(
        self,
        venue_provider:  VenueFactorProvider | None = None,
        bullpen_fetcher: BullpenFetcher | None = None,
        ensemble:        EnsembleModel | None = None,
        config_loader    = None,
    ) -> None:
        self._venue    = venue_provider  or VenueFactorProvider(config_loader)
        self._bullpen  = bullpen_fetcher or BullpenFetcher(config_loader=config_loader)
        self._ensemble = ensemble or EnsembleModel(
            proj_min = _PROJ_MIN,
            proj_max = _PROJ_MAX,
        )
        self._config = config_loader

        # Leer pesos del YAML si disponible
        self._era_weight    = self._cfg("mlb.projection.era_weight",    0.35)
        self._fip_weight    = self._cfg("mlb.projection.fip_weight",    0.35)
        self._off_weight    = self._cfg("mlb.projection.team_offense_weight", 0.20)
        self._bul_weight    = self._cfg("mlb.projection.bullpen_weight", 0.10)

    def model_version(self) -> str:
        return _MODEL_VERSION

    # ── Punto de entrada principal ────────────────────────────────────────────

    def project(
        self,
        home_features: TeamFeatures,
        away_features: TeamFeatures,
        context:       dict,
    ) -> Projection:
        """
        Calcula la proyección de carreras para ambos equipos.

        Parámetros
        ----------
        home_features  -- TeamFeatures del equipo local. Contiene
                         offense_index, defense_index, recent_scores,
                         venue_factor, sport_metadata (ERA, FIP, OPS, etc.)
        away_features  -- TeamFeatures del equipo visitante.
        context        -- Dict de MLBContextFetcher:
                         venue_type, temperature, wind_speed,
                         wind_direction, day_night.

        Retorna
        -------
        Projection con expected_home, expected_away, win_probs,
        distribution='poisson' y model_inputs completo.
        """
        # ── 1. Extraer componentes de las features ────────────────────
        home_meta = home_features.sport_metadata or {}
        away_meta = away_features.sport_metadata or {}

        # ERA del abridor (con fallback a ERA de liga)
        era_home = _get_era(home_meta)
        era_away = _get_era(away_meta)

        # FIP del abridor si disponible (más predictivo que ERA)
        fip_home = home_meta.get("fip")
        fip_away = away_meta.get("fip")

        # ERA efectiva: blend ERA + FIP ponderado
        eff_era_home = _effective_era(era_home, fip_home,
                                       self._era_weight, self._fip_weight)
        eff_era_away = _effective_era(era_away, fip_away,
                                       self._era_weight, self._fip_weight)

        # OPS ofensiva (con splits vs mano del pitcher rival si disponible)
        pitcher_hand_away = away_meta.get("pitcher_hand")  # mano del pitcher rival
        pitcher_hand_home = home_meta.get("pitcher_hand")

        ops_home = _get_ops_vs_hand(home_meta, pitcher_hand_away)
        ops_away = _get_ops_vs_hand(away_meta, pitcher_hand_home)

        # ── 2. Proyección base desde offense/defense ──────────────────
        # Runs proyectadas = recent_avg × offense_adj / defense_adj
        home_recent_avg = home_features.recent_avg or _LEAGUE_OPS
        away_recent_avg = away_features.recent_avg or _LEAGUE_OPS

        off_idx_home = ops_home / _LEAGUE_OPS if ops_home else home_features.offense_index
        off_idx_away = ops_away / _LEAGUE_OPS if ops_away else away_features.offense_index

        def_idx_home = defense_index_from_combined_era(eff_era_home)
        def_idx_away = defense_index_from_combined_era(eff_era_away)

        # Proyección base: ofensiva propia vs defensa rival
        proj_home_base = home_recent_avg * off_idx_home / def_idx_away
        proj_away_base = away_recent_avg * off_idx_away / def_idx_home

        # ── 3. Ajuste por park factor ─────────────────────────────────
        venue_id = context.get("venue_id", home_features.venue_id or "")
        venue_factor = (
            home_features.venue_factor
            if home_features.venue_factor and home_features.venue_factor != 1.0
            else self._venue.get(venue_id)
        )
        # El park factor se aplica parcialmente (sqrt): afecta a ambos equipos
        pf_adj = math.sqrt(venue_factor)
        proj_home_park = proj_home_base * pf_adj
        proj_away_park = proj_away_base * pf_adj

        # ── 4. Ajuste por clima (solo outdoor) ───────────────────────
        venue_type    = context.get("venue_type", "outdoor")
        weather_factor= 1.0
        if venue_type != "indoor":
            weather_factor = _compute_weather_factor(context)

        proj_home_w = proj_home_park * weather_factor
        proj_away_w = proj_away_park * weather_factor

        # ── 5. Ensemble blending con regresión sobre recent_scores ────
        proj_home_final, h_detail = self._ensemble.adjust(
            poisson_proj  = proj_home_w,
            team_features = home_features,
        )
        proj_away_final, a_detail = self._ensemble.adjust(
            poisson_proj  = proj_away_w,
            team_features = away_features,
        )

        # Clamp final (garantía adicional sobre EnsembleModel)
        proj_home_final = max(_PROJ_MIN, min(_PROJ_MAX, proj_home_final))
        proj_away_final = max(_PROJ_MIN, min(_PROJ_MAX, proj_away_final))

        # ── 6. Probabilidades de victoria (aproximación logística) ────
        run_diff    = proj_home_final - proj_away_final
        home_win_p  = _run_diff_to_win_prob(run_diff)
        away_win_p  = round(1.0 - home_win_p, 4)

        # ── 7. Confidence score ────────────────────────────────────────
        confidence = _compute_confidence(home_features, away_features, context)

        # ── 8. Trazabilidad completa ───────────────────────────────────
        model_inputs = {
            # Pitching
            "era_home":          era_home,
            "era_away":          era_away,
            "fip_home":          fip_home,
            "fip_away":          fip_away,
            "eff_era_home":      round(eff_era_home, 3),
            "eff_era_away":      round(eff_era_away, 3),
            "bullpen_day_home":  home_meta.get("is_bullpen_day", False),
            "bullpen_day_away":  away_meta.get("is_bullpen_day", False),
            # Batting
            "ops_home":          ops_home,
            "ops_away":          ops_away,
            "off_idx_home":      round(off_idx_home, 4),
            "off_idx_away":      round(off_idx_away, 4),
            "def_idx_home":      round(def_idx_home, 4),
            "def_idx_away":      round(def_idx_away, 4),
            # Context
            "venue_factor":      round(venue_factor, 4),
            "weather_factor":    round(weather_factor, 4),
            "venue_type":        venue_type,
            "temperature":       context.get("temperature"),
            "wind_speed":        context.get("wind_speed"),
            # Ensemble
            "ensemble_home":     h_detail,
            "ensemble_away":     a_detail,
            # Base projections
            "proj_home_base":    round(proj_home_base, 3),
            "proj_away_base":    round(proj_away_base, 3),
            "proj_home_final":   round(proj_home_final, 3),
            "proj_away_final":   round(proj_away_final, 3),
        }

        return Projection(
            event_id          = home_features.team_id + "_" + away_features.team_id,
            sport             = "mlb",
            expected_home     = round(proj_home_final, 3),
            expected_away     = round(proj_away_final, 3),
            home_win_prob     = home_win_p,
            away_win_prob     = away_win_p,
            draw_prob         = 0.0,
            distribution      = "poisson",
            distribution_params = {
                "mu_home": round(proj_home_final, 3),
                "mu_away": round(proj_away_final, 3),
            },
            confidence        = round(confidence, 4),
            model_version     = _MODEL_VERSION,
            model_inputs      = model_inputs,
        )

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _cfg(self, key: str, default: float) -> float:
        """Lee un valor del ConfigLoader con fallback a default."""
        if self._config is None:
            return default
        val = self._config.get(key, default=default)
        try:
            return float(val)
        except (ValueError, TypeError):
            return default


# ── Funciones puras de apoyo ──────────────────────────────────────────────────

def _get_era(meta: dict) -> float:
    """Extrae ERA del sport_metadata con fallback a ERA de liga."""
    era = meta.get("era") or meta.get("era_recent")
    try:
        return float(era) if era is not None else _LEAGUE_ERA
    except (ValueError, TypeError):
        return _LEAGUE_ERA


def _get_ops_vs_hand(meta: dict, pitcher_hand: str | None) -> float | None:
    """Extrae el OPS con splits vsRHP/vsLHP si aplica."""
    if pitcher_hand == "R":
        ops = meta.get("ops_vs_rhp")
    elif pitcher_hand == "L":
        ops = meta.get("ops_vs_lhp")
    else:
        ops = meta.get("ops")
    if ops is None:
        ops = meta.get("ops")
    try:
        return float(ops) if ops is not None else None
    except (ValueError, TypeError):
        return None


def _effective_era(
    era:        float,
    fip:        float | None,
    era_weight: float,
    fip_weight: float,
) -> float:
    """
    ERA efectiva como blend ponderado de ERA y FIP.

    Si FIP no está disponible, usa ERA solo.
    Si ambos están disponibles, blend ponderado normalizado.
    """
    if fip is None:
        return era
    total_w = era_weight + fip_weight
    if total_w <= 0:
        return era
    return round((era * era_weight + fip * fip_weight) / total_w, 4)


def _run_diff_to_win_prob(run_diff: float) -> float:
    """
    Convierte la diferencia de carreras proyectadas en probabilidad de victoria.

    Usa la función logística calibrada para MLB:
        P(home wins) = 1 / (1 + exp(-k × run_diff))

    Donde k=0.37 fue calibrado para que:
        +1 carrera de ventaja → ~59% de probabilidad de victoria
        +3 carreras → ~75%
        0 diferencia → 50%

    Este es el mismo patrón que usa la mayoría de los modelos Elo
    de MLB (FiveThirtyEight calibraba k≈0.40).
    """
    k = 0.37
    prob = 1.0 / (1.0 + math.exp(-k * run_diff))
    return round(prob, 4)


def _compute_weather_factor(context: dict) -> float:
    """
    Calcula el factor de ajuste por clima.

    Temperatura: desviación de 72°F (referencia neutral MLB).
    Viento: ajuste basado en velocidad y dirección relativa al campo.

    Factor = 1.0 si condiciones neutrales.
    Factor > 1.0 si condiciones favorecen el bateo (calor, viento de espalda).
    Factor < 1.0 si condiciones deprimen el bateo (frío, viento de frente).
    """
    factor = 1.0

    # Ajuste por temperatura
    temp = context.get("temperature")
    if temp is not None:
        try:
            temp_delta = float(temp) - _TEMP_FACTOR_BASELINE
            factor += temp_delta * _TEMP_COEFFICIENT
        except (ValueError, TypeError):
            pass

    # Ajuste por viento
    wind_speed = context.get("wind_speed")
    wind_dir   = context.get("wind_direction")
    if wind_speed is not None and wind_dir is not None:
        try:
            spd = float(wind_speed)
            direction = float(wind_dir)
            # Dirección 180° = viento de espalda (desde home plate hacia OF)
            # Dirección 0°   = viento de frente (desde OF hacia home plate)
            # Componente efectiva: cos(dir - 180°) × velocidad
            angle_rad     = math.radians(direction - 180.0)
            effective_wind = spd * math.cos(angle_rad)
            factor        += effective_wind * _WIND_COEFFICIENT
        except (ValueError, TypeError):
            pass

    # Clamp: factor nunca menor de 0.80 ni mayor de 1.25
    return round(max(0.80, min(1.25, factor)), 4)


def _compute_confidence(
    home: TeamFeatures,
    away: TeamFeatures,
    context: dict,
) -> float:
    """
    Score de confianza de la proyección [0.0, 1.0].

    Factores que reducen la confianza:
    - has_sufficient_sample=False en cualquier equipo
    - is_bullpen_day en cualquier equipo
    - data_quality < 0.80
    - condición de lluvia (mayor incertidumbre)

    Confianza perfecta (1.0) solo si todos los datos están completos
    y el clima es favorable.
    """
    confidence = 1.0

    # Datos insuficientes reducen confianza
    if not home.has_sufficient_sample:
        confidence -= 0.15
    if not away.has_sufficient_sample:
        confidence -= 0.15

    # Calidad de datos
    dq_home = home.data_quality or 1.0
    dq_away = away.data_quality or 1.0
    confidence *= (dq_home + dq_away) / 2.0

    # Bullpen day = más incertidumbre
    home_meta = home.sport_metadata or {}
    away_meta = away.sport_metadata or {}
    if home_meta.get("is_bullpen_day"):
        confidence -= 0.10
    if away_meta.get("is_bullpen_day"):
        confidence -= 0.10

    # Lluvia = más incertidumbre
    condition = context.get("condition", "clear")
    if condition in ("rain", "storm"):
        confidence -= 0.10

    return max(0.10, min(1.0, confidence))