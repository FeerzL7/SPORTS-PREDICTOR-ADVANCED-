"""
core/bankroll/staking.py

StakingStrategy: interfaz y cuatro implementaciones para sizing de stake.

Migrado de bankroll/staking.py del sistema MLB con tres correcciones
documentadas en SPORTS_PREDICTOR_ARCHITECTURE.md §7.2:

1. Opera sobre CandidatePick tipado, no sobre dict mutable.
   El sistema MLB parseaba 'mejor_pick' (string "ML: Team") para extraer
   el mercado y luego accedía a campos por nombre de string. Con
   CandidatePick tipado, pick.ev, pick.blended_prob y pick.kelly_fraction
   son campos directos — sin parseo frágil.

2. movement_confirms como parámetro explícito, no campo del dict.
   El sistema MLB leía partido.get("mov_confirma"). Aquí el caller
   extrae esta información del pick.reasons del LineMovementDetector
   y la pasa explícitamente — más limpio y testeable.

3. Thresholds por (sport, market) en YAML via ConfigLoader.
   Los thresholds del sistema MLB estaban hardcodeados o en atributos
   del dataclass. Aquí IntegerPercentStaking los lee del ConfigLoader
   por mercado, con defaults del sistema MLB como fallback.

Implementaciones
-----------------
IntegerPercentStaking  — migrado de MLB. Stake conservador entre min y max
                         pct ajustado por EV, probabilidad y confirmación
                         de movimiento de línea. Thresholds en YAML.

KellyStaking           — usa pick.kelly_fraction como referencia. Si
                         kelly_fraction=0 (EV negativo), retorna 0 sin
                         mínimo forzado. Convierte fracción a entero
                         con multiplier configurable.

FlatStaking            — siempre min_pct. Para backtesting sin sesgo de
                         sizing: mide valor del sistema de filtrado puro.

AdaptiveStaking        — ajusta entre min y max según hit_rate reciente.
                         Más agresivo en rachas positivas. Requiere que
                         el caller provea hit_rate desde BankrollTracker
                         — desacopla staking del tracker.

Uso típico
-----------
    from core.bankroll.staking import IntegerPercentStaking, apply_staking
    from core.utils.config_loader import load_config

    strategy = IntegerPercentStaking(config=load_config(sport='mlb'))
    apply_staking(pick, strategy, movement_confirms=True)
    # pick.stake_pct ahora está fijado por la estrategia
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.contracts.pick import CandidatePick


# ── Protocolo ─────────────────────────────────────────────────────────────────

@runtime_checkable
class StakingStrategy(Protocol):
    """
    Interfaz de estrategia de sizing de stake.

    Una implementación devuelve un entero en % del bankroll.
    El caller (pipeline Stage 8) fija pick.stake_pct con ese valor
    antes de pasarlo a BankrollTracker.register().

    Parámetros de stake_pct
    ------------------------
    pick                -- CandidatePick con ev, edge, blended_prob y
                          kelly_fraction ya calculados (Stages 6).
    movement_confirms   -- True si LineMovementDetector marcó que el
                          movimiento de línea va en la misma dirección
                          que el pick. El caller extrae esto del trail
                          de reasons del pick antes de llamar a la
                          estrategia.
    """

    def stake_pct(
        self,
        pick: CandidatePick,
        movement_confirms: bool = False,
    ) -> int:
        """Retorna el stake recomendado en % del bankroll (entero)."""
        ...


# ── IntegerPercentStaking ─────────────────────────────────────────────────────

# Defaults del sistema MLB — preservados como fallback para deportes
# sin configuración YAML. Calibrados para la distribución de EV/prob
# observada en MLB TOTAL (el mercado más rentable del sistema original).
_MLB_MIN_PCT:   int   = 1
_MLB_MAX_PCT:   int   = 3

# Thresholds para subir stake de min a min+1
_EV_THRESHOLD_L1:   float = 24.0  # TOTAL: EV >= 24 y prob >= 0.58 → +1
_PROB_THRESHOLD_L1: float = 0.58

# Thresholds para subir de min+1 a min+2 (con movimiento confirmado)
_EV_THRESHOLD_L2:   float = 34.0
_PROB_THRESHOLD_L2: float = 0.60

# Thresholds ML/RL (más conservadores que TOTAL)
_EV_ML_L1:   float = 10.0
_PROB_ML_L1: float = 0.55
_EV_ML_L2:   float = 16.0
_PROB_ML_L2: float = 0.57


@dataclass
class IntegerPercentStaking:
    """
    Stake conservador en porcentajes enteros — migrado de MLB.

    Lógica: parte de min_pct y sube hasta max_pct según la calidad
    de la señal (EV, probabilidad, confirmación de movimiento de línea).
    Baja 1 pct si hay data_quality_flags en el pick.

    Thresholds configurables por (sport, market) via ConfigLoader:

        # config/mlb.yaml
        staking:
          TOTAL:
            min_pct: 1
            max_pct: 3
            ev_l1: 24.0
            prob_l1: 0.58
            ev_l2: 34.0
            prob_l2: 0.60
          ML:
            min_pct: 1
            max_pct: 2
            ev_l1: 10.0
            prob_l1: 0.55

    Si config=None, usa los defaults del sistema MLB como fallback.
    """

    min_pct: int   = _MLB_MIN_PCT
    max_pct: int   = _MLB_MAX_PCT
    config         = None  # ConfigLoader opcional

    def __init__(
        self,
        min_pct: int   = _MLB_MIN_PCT,
        max_pct: int   = _MLB_MAX_PCT,
        config         = None,
    ) -> None:
        if min_pct < 0:
            raise ValueError(f"min_pct={min_pct} debe ser >= 0.")
        if max_pct < min_pct:
            raise ValueError(f"max_pct={max_pct} debe ser >= min_pct={min_pct}.")
        self.min_pct = min_pct
        self.max_pct = max_pct
        self.config  = config
        self._cache: dict[str, dict] = {}

    def stake_pct(
        self,
        pick: CandidatePick,
        movement_confirms: bool = False,
    ) -> int:
        market = pick.market.upper()
        cfg    = self._get_market_config(market)

        min_pct = cfg["min_pct"]
        max_pct = cfg["max_pct"]
        stake   = min_pct

        ev   = pick.ev
        prob = pick.blended_prob

        if market == "TOTAL":
            if ev >= cfg["ev_l1"] and prob >= cfg["prob_l1"]:
                stake += 1
            if ev >= cfg["ev_l2"] and prob >= cfg["prob_l2"] and movement_confirms:
                stake += 1
        else:
            # ML y SPREAD — umbrales más conservadores
            if ev >= cfg["ev_l1"] and prob >= cfg["prob_l1"]:
                stake += 1
            if ev >= cfg["ev_l2"] and prob >= cfg["prob_l2"] and movement_confirms:
                stake += 1

        # Penalizar por flags de calidad de datos en el pick
        if pick.reasons and any("fallback" in r.lower() for r in pick.reasons):
            stake = max(min_pct, stake - 1)

        return int(max(min_pct, min(stake, max_pct)))

    def _get_market_config(self, market: str) -> dict:
        """Lee thresholds del ConfigLoader por mercado, con fallback a defaults."""
        if market in self._cache:
            return self._cache[market]

        def get(key: str, default):
            if self.config is None:
                return default
            return self.config.get(f"staking.{market}.{key}", default=default)

        cfg = {
            "min_pct": int(get("min_pct", self.min_pct)),
            "max_pct": int(get("max_pct", self.max_pct)),
            "ev_l1":   float(get("ev_l1",   _EV_THRESHOLD_L1 if market == "TOTAL" else _EV_ML_L1)),
            "prob_l1": float(get("prob_l1", _PROB_THRESHOLD_L1 if market == "TOTAL" else _PROB_ML_L1)),
            "ev_l2":   float(get("ev_l2",   _EV_THRESHOLD_L2 if market == "TOTAL" else _EV_ML_L2)),
            "prob_l2": float(get("prob_l2", _PROB_THRESHOLD_L2 if market == "TOTAL" else _PROB_ML_L2)),
        }
        self._cache[market] = cfg
        return cfg

    def clear_cache(self) -> None:
        self._cache.clear()


# ── KellyStaking ──────────────────────────────────────────────────────────────

@dataclass
class KellyStaking:
    """
    Stake basado directamente en pick.kelly_fraction del modelo.

    Si kelly_fraction=0.0 (EV negativo o pick sin valor), retorna 0.
    No aplica mínimo forzado cuando kelly_fraction=0 — si el modelo
    no ve valor, no hay stake.

    Conversión:
        stake = round(kelly_fraction × multiplier)
        stake = max(min_pct, stake) si stake > 0, else 0
        stake = min(stake, max_pct)

    Ejemplo con defaults (multiplier=100):
        kelly_fraction=0.018 → round(1.8) = 2 → stake=2%
        kelly_fraction=0.0   → stake=0 (sin apuesta)
        kelly_fraction=0.005 → round(0.5) = 1 → stake=max(1,1)=1%

    Parámetros
    ----------
    multiplier  -- Factor de escala para convertir la fracción a %.
                  Default 100: kelly_fraction=0.018 → 1.8 → 2%.
    min_pct     -- Mínimo aplicado solo cuando kelly > 0.
    max_pct     -- Techo absoluto de stake.
    """

    multiplier: float = 100.0
    min_pct:    int   = _MLB_MIN_PCT
    max_pct:    int   = _MLB_MAX_PCT

    def __post_init__(self) -> None:
        if self.multiplier <= 0:
            raise ValueError(f"multiplier={self.multiplier} debe ser > 0.")
        if self.max_pct < self.min_pct:
            raise ValueError(f"max_pct={self.max_pct} debe ser >= min_pct={self.min_pct}.")

    def stake_pct(
        self,
        pick: CandidatePick,
        movement_confirms: bool = False,
    ) -> int:
        kf = pick.kelly_fraction or 0.0
        if kf <= 0.0:
            return 0

        raw   = round(kf * self.multiplier)
        stake = max(self.min_pct, raw)

        # Bonus por confirmación de movimiento
        if movement_confirms:
            stake = min(stake + 1, self.max_pct)

        return int(min(stake, self.max_pct))


# ── FlatStaking ───────────────────────────────────────────────────────────────

@dataclass
class FlatStaking:
    """
    Stake fijo — siempre retorna min_pct sin importar la señal.

    Uso principal: backtesting para medir el valor puro del sistema
    de filtrado sin el sesgo de sizing. Si FlatStaking tiene ROI
    positivo, el sistema encuentra valor real; si solo
    IntegerPercentStaking lo tiene, el sizing puede estar sobreajustado.

    También útil para entornos de producción conservadores donde se
    quiere control total sobre el tamaño de apuesta.
    """

    min_pct: int = _MLB_MIN_PCT

    def __post_init__(self) -> None:
        if self.min_pct < 0:
            raise ValueError(f"min_pct={self.min_pct} debe ser >= 0.")

    def stake_pct(
        self,
        pick: CandidatePick,
        movement_confirms: bool = False,
    ) -> int:
        return self.min_pct


# ── AdaptiveStaking ───────────────────────────────────────────────────────────

@dataclass
class AdaptiveStaking:
    """
    Stake adaptativo ajustado por el hit rate reciente del modelo.

    Más agresivo (stake alto) durante rachas positivas; más conservador
    (stake bajo) durante rachas negativas. Implementa gestión de riesgo
    dinámica sin tocar los filtros del ValueEngine.

    El caller provee recent_hit_rate calculado desde BankrollTracker —
    desacopla AdaptiveStaking del tracker y evita dependencias circulares.

    Lógica de ajuste:
        hit_rate >= high_threshold → max_pct
        hit_rate <= low_threshold  → min_pct
        En medio                   → interpolación lineal

    Parámetros
    ----------
    recent_hit_rate  -- Hit rate reciente del modelo (0-100).
                       Calculado por el caller desde
                       BankrollTracker.metrics(date_from=...).hit_rate.
    min_pct          -- Stake mínimo (en rachas negativas).
    max_pct          -- Stake máximo (en rachas positivas).
    low_threshold    -- Hit rate por debajo del cual se usa min_pct.
                       Default: 45% (por debajo del breakeven ~52%).
    high_threshold   -- Hit rate por encima del cual se usa max_pct.
                       Default: 60% (racha claramente positiva).
    """

    recent_hit_rate:  float
    min_pct:          int   = _MLB_MIN_PCT
    max_pct:          int   = _MLB_MAX_PCT
    low_threshold:    float = 45.0
    high_threshold:   float = 60.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.recent_hit_rate <= 100.0:
            raise ValueError(
                f"recent_hit_rate={self.recent_hit_rate} debe estar en [0, 100]."
            )
        if self.max_pct < self.min_pct:
            raise ValueError(f"max_pct={self.max_pct} debe ser >= min_pct={self.min_pct}.")
        if self.low_threshold >= self.high_threshold:
            raise ValueError(
                f"low_threshold={self.low_threshold} debe ser < "
                f"high_threshold={self.high_threshold}."
            )

    def stake_pct(
        self,
        pick: CandidatePick,
        movement_confirms: bool = False,
    ) -> int:
        hr = self.recent_hit_rate

        if hr <= self.low_threshold:
            stake = self.min_pct
        elif hr >= self.high_threshold:
            stake = self.max_pct
        else:
            # Interpolación lineal entre low y high threshold
            t = (hr - self.low_threshold) / (self.high_threshold - self.low_threshold)
            stake = round(self.min_pct + t * (self.max_pct - self.min_pct))

        stake = int(max(self.min_pct, min(stake, self.max_pct)))

        # Bonus por confirmación de movimiento cuando el modelo está en racha
        if movement_confirms and stake < self.max_pct:
            stake = min(stake + 1, self.max_pct)

        return stake


# ── Función pública de aplicación ─────────────────────────────────────────────

def apply_staking(
    pick: CandidatePick,
    strategy: StakingStrategy,
    movement_confirms: bool = False,
) -> int:
    """
    Aplica la estrategia de staking al pick y fija pick.stake_pct.

    Wrapper conveniente para Stage 8 del pipeline — una línea en vez
    de dos (calcular + asignar).

    Parámetros
    ----------
    pick               -- CandidatePick a fijar. stake_pct se actualiza
                         directamente en el objeto.
    strategy           -- Implementación de StakingStrategy a usar.
    movement_confirms  -- True si LineMovementDetector confirmó la
                         dirección del pick con movimiento de línea.

    Retorna
    -------
    int — el stake_pct fijado, para uso en logging y trazabilidad.
    """
    pct = strategy.stake_pct(pick, movement_confirms=movement_confirms)
    pick.stake_pct = pct
    return pct