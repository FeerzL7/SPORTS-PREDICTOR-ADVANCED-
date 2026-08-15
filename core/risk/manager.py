"""
core/risk/manager.py

RiskManager: Stage 9 del pipeline — control de exposición diaria,
límites de picks, veto por movimiento de línea contrario.

Migrado de utils/risk_management.py del sistema MLB con cuatro
correcciones documentadas en SPORTS_PREDICTOR_ARCHITECTURE.md §9:

1. Opera sobre list[CandidatePick] tipados, no list[dict] mutables.
   El sistema MLB usaba partido["riesgo_estado"] = "activo". Aquí
   pick.activate() y pick.deactivate(reason) son los métodos del
   contrato — pick.active y pick.inactive_reason son los campos.

2. Límites configurables por (sport, date) via RiskProfile.
   MAX_PICKS_DIARIOS=3 y MAX_EXPOSICION_DIARIA_PCT=6 estaban
   hardcodeados. Aquí RiskProfile es un dataclass configurable
   y el pipeline puede crear perfiles distintos por deporte.

3. Detección de movimiento contradictorio desde pick.reasons.
   El sistema MLB leía partido.get("mov_contradice") — un bool
   en el dict. Aquí LineMovementDetector.annotate_pick() añade
   reasons con "MOVEMENT[✗]" — el manager detecta esta señal
   en el trail del pick sin acoplarse al formato del detector.

4. Selección greedy correcta — el sistema MLB descartaba por orden
   de iteración sin reconsiderar si un pick de menor prioridad
   pudiera ceder su cupo a uno de mayor valor. Aquí se seleccionan
   los mejores picks por score antes de aplicar límites, garantizando
   el portfolio óptimo dentro de las restricciones.

Separación de responsabilidades
---------------------------------
RiskManager hace:
    - Desactivar picks contradichos por movimiento de línea
    - Ordenar candidatos por (confirmación, market_priority, ev)
    - Aplicar límite máximo de picks diarios
    - Aplicar límite máximo de exposición diaria (suma de stake_pct)
    - Recortar stake_pct si excede el límite por pick

NO hace:
    - Calcular EV, Kelly, blending (ValueEngine — Stage 6)
    - Sizing de stake (StakingStrategy — Stage 8)
    - Settlement de resultados (SettlementProvider)
    - Calcular el movimiento de línea (LineMovementDetector — Stage 7)

Garantías de diseño
---------------------
- Todo pick en la lista de entrada queda con active=True o
  active=False al salir — nunca en estado ambiguo.
- El orden relativo de los picks activos en la lista de salida
  es el mismo que en la entrada — para reproducibilidad de logs.
- Un pick desactivado por veto de movimiento nunca se reactiva
  aunque quepan más picks en el presupuesto.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.contracts.pick import CandidatePick
from core.odds.line_movement import (
    MOVEMENT_CONFIRMS_PREFIX,
    MOVEMENT_CONTRADICTS_PREFIX,
)


# ── Señal de movimiento contradictorio ───────────────────────────────────────
#
# CORRECCIÓN DE CONTRATO (auditoría 2026-08): estas dos constantes vivían
# antes como copias locales de un literal ("MOVEMENT[✗]"/"MOVEMENT[✓]")
# también duplicado en core/odds/line_movement.py. Un tercer módulo
# (core/pipeline/runner.py) terminó importando el nombre equivocado desde
# un cuarto módulo que nunca lo definió — el síntoma directo de no tener
# una única fuente de verdad. Ahora se importan desde
# core.odds.line_movement, que es el dueño canónico del formato.

def _pick_has_contradicting_movement(pick: CandidatePick) -> bool:
    """True si el trail del pick contiene señal de movimiento contradictorio."""
    return any(
        MOVEMENT_CONTRADICTS_PREFIX in reason
        for reason in pick.reasons
    )


def _pick_has_confirming_movement(pick: CandidatePick) -> bool:
    """True si el trail del pick contiene señal de movimiento confirmatorio."""
    return any(
        MOVEMENT_CONFIRMS_PREFIX in reason
        for reason in pick.reasons
    )


# ── Perfil de riesgo ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskProfile:
    """
    Parámetros de riesgo para un deporte y configuración dados.

    Inmutable: el perfil no cambia durante la aplicación del risk
    manager en un pipeline. Si los parámetros cambian (ej. al ajustar
    la exposición diaria), se crea un nuevo RiskProfile.

    Campos
    ------
    max_picks_daily        -- Máximo de picks activos por día.
                            Default 3 (conservador — sistema MLB).
    max_exposure_pct       -- Máximo de exposición diaria total en %
                            del bankroll (suma de stake_pct de picks
                            activos). Default 6.
    max_stake_per_pick_pct -- Techo de stake_pct por pick individual.
                            Si StakingStrategy asigna más, se recorta.
                            Default 3.
    veto_on_contradiction  -- Si True, desactiva picks contradichos por
                            movimiento de línea. Default True.
                            Puede desactivarse para backtesting puro.
    market_priority        -- Mapa de market → prioridad para ordenar
                            candidatos cuando hay competencia por cupo.
                            Menor número = mayor prioridad.
                            Default: TOTAL=0, ML=1, SPREAD=2.
                            Sigue el principio del sistema MLB donde
                            TOTAL tenía prioridad estructural.
    """
    max_picks_daily:        int             = 3
    max_exposure_pct:       int             = 6
    max_stake_per_pick_pct: int             = 3
    veto_on_contradiction:  bool            = True
    market_priority:        dict[str, int]  = field(default_factory=lambda: {
        "TOTAL":  0,
        "ML":     1,
        "SPREAD": 2,
        "1X2":    1,
    })

    def get_market_priority(self, market: str) -> int:
        """
        Retorna la prioridad de un mercado.
        Mercados no registrados reciben prioridad baja (99).
        """
        return self.market_priority.get(market.upper(), 99)


# ── Resultado de la aplicación del risk manager ───────────────────────────────

@dataclass(frozen=True)
class RiskSummary:
    """
    Resumen de las decisiones tomadas por RiskManager.apply().

    Inmutable: snapshot de lo que ocurrió en esta ejecución.
    Útil para logging y para el dashboard de operaciones.

    Campos
    ------
    picks_active          -- Picks marcados como activos.
    picks_vetoed          -- Desactivados por movimiento contradictorio.
    picks_over_exposure   -- Desactivados por límite de exposición diaria.
    picks_over_limit      -- Desactivados por límite de picks diarios.
    picks_stake_trimmed   -- Picks donde stake_pct fue recortado por
                            exceder max_stake_per_pick_pct.
    total_exposure_pct    -- Suma de stake_pct de picks activos.
    profile_used          -- RiskProfile aplicado.
    """
    picks_active:         list[CandidatePick]
    picks_vetoed:         list[CandidatePick]
    picks_over_exposure:  list[CandidatePick]
    picks_over_limit:     list[CandidatePick]
    picks_stake_trimmed:  list[CandidatePick]
    total_exposure_pct:   int
    profile_used:         RiskProfile

    def log_summary(self) -> str:
        """Descripción compacta para logs del pipeline."""
        return (
            f"Risk: {len(self.picks_active)} activos | "
            f"exposición={self.total_exposure_pct}% | "
            f"vetados={len(self.picks_vetoed)} | "
            f"por_límite={len(self.picks_over_limit)} | "
            f"por_exposición={len(self.picks_over_exposure)}"
        )


# ── Motor principal ───────────────────────────────────────────────────────────

class RiskManager:
    """
    Stage 9 del pipeline — control de exposición y límites diarios.

    Recibe la lista completa de picks que pasaron filtros (Stage 6)
    y staking (Stage 8), y decide cuáles quedan activos aplicando:

    1. Veto por movimiento de línea contradictorio
    2. Límite de stake por pick (max_stake_per_pick_pct)
    3. Ordenación por score: (confirma, -prioridad_mercado, ev)
    4. Selección greedy respetando max_picks_daily y max_exposure_pct

    Parámetros
    ----------
    profile  -- RiskProfile con los límites configurados.
               Default: RiskProfile() con los valores del sistema MLB.
    """

    def __init__(self, profile: RiskProfile | None = None) -> None:
        self._profile = profile or RiskProfile()

    @classmethod
    def from_config(cls, config) -> RiskManager:
        """
        Factory: construye RiskManager desde un ConfigLoader.

        Lee los parámetros de riesgo del YAML bajo la key 'risk':
            risk:
              max_picks_daily: 3
              max_exposure_pct: 6
              max_stake_per_pick_pct: 3
              veto_on_contradiction: true
        """
        def get(key: str, default):
            return config.get(f"risk.{key}", default=default)

        profile = RiskProfile(
            max_picks_daily        = int(get("max_picks_daily",        3)),
            max_exposure_pct       = int(get("max_exposure_pct",       6)),
            max_stake_per_pick_pct = int(get("max_stake_per_pick_pct", 3)),
            veto_on_contradiction  = bool(get("veto_on_contradiction", True)),
        )
        return cls(profile=profile)

    def apply(self, picks: list[CandidatePick]) -> RiskSummary:
        """
        Aplica las reglas de riesgo a la lista de picks.

        Cada pick queda con active=True o active=False al finalizar.
        El orden de la lista de entrada se preserva en la salida
        (para reproducibilidad de logs y tests).

        Parámetros
        ----------
        picks  -- Picks que pasaron ValueEngine (Stage 6), fueron
                 anotados por LineMovementDetector (Stage 7) y
                 tienen stake_pct fijado por StakingStrategy (Stage 8).
                 Pueden tener active=True/False de etapas anteriores —
                 el RiskManager solo procesa los que tienen stake_pct > 0.

        Retorna
        -------
        RiskSummary con las decisiones tomadas y listas de picks
        por categoría de decisión.
        """
        profile = self._profile

        vetoed:          list[CandidatePick] = []
        over_limit:      list[CandidatePick] = []
        over_exposure:   list[CandidatePick] = []
        stake_trimmed:   list[CandidatePick] = []
        candidates:      list[CandidatePick] = []

        # ── Paso 1: Veto por movimiento contradictorio ────────────────
        for pick in picks:
            if pick.stake_pct <= 0:
                # Sin stake asignado — desactivar silenciosamente
                pick.deactivate("stake_pct=0 tras staking strategy")
                continue

            if (profile.veto_on_contradiction
                    and _pick_has_contradicting_movement(pick)):
                pick.deactivate("movimiento de línea contradice el pick")
                vetoed.append(pick)
                continue

            candidates.append(pick)

        # ── Paso 2: Recortar stake por encima del límite por pick ─────
        for pick in candidates:
            if pick.stake_pct > profile.max_stake_per_pick_pct:
                original = pick.stake_pct
                pick.stake_pct = profile.max_stake_per_pick_pct
                pick.add_reason(
                    f"stake recortado {original}%→{profile.max_stake_per_pick_pct}% "
                    f"(max_stake_per_pick_pct)"
                )
                stake_trimmed.append(pick)

        # ── Paso 3: Ordenar candidatos por score de selección ─────────
        # Score: (confirmación, prioridad_mercado, ev)
        # Mayor confirmación > menor prioridad numérica > mayor EV
        def score(pick: CandidatePick) -> tuple:
            confirms = 1 if _pick_has_confirming_movement(pick) else 0
            priority = profile.get_market_priority(pick.market)
            return (confirms, -priority, pick.ev)

        ordered = sorted(candidates, key=score, reverse=True)

        # ── Paso 4: Selección greedy por límites ─────────────────────
        active:           list[CandidatePick] = []
        total_exposure:   int                 = 0

        for pick in ordered:
            if len(active) >= profile.max_picks_daily:
                pick.deactivate(
                    f"límite de picks diarios alcanzado "
                    f"({profile.max_picks_daily})"
                )
                over_limit.append(pick)
                continue

            # Verificar si este pick cabe en la exposición restante
            remaining = profile.max_exposure_pct - total_exposure
            if pick.stake_pct > remaining:
                if remaining > 0:
                    # Recortar stake para caber exactamente en el presupuesto
                    pick.stake_pct = remaining
                    pick.add_reason(
                        f"stake recortado a {remaining}% "
                        f"(exposición diaria max={profile.max_exposure_pct}%)"
                    )
                    if pick not in stake_trimmed:
                        stake_trimmed.append(pick)
                else:
                    pick.deactivate(
                        f"exposición diaria máxima alcanzada "
                        f"({profile.max_exposure_pct}%)"
                    )
                    over_exposure.append(pick)
                    continue

            if pick.stake_pct <= 0:
                pick.deactivate("stake_pct=0 tras recorte de exposición")
                over_exposure.append(pick)
                continue

            pick.activate()
            active.append(pick)
            total_exposure += pick.stake_pct

        # Los candidatos que no fueron seleccionados ya fueron
        # desactivados en el loop anterior con su razón específica.

        return RiskSummary(
            picks_active        = active,
            picks_vetoed        = vetoed,
            picks_over_exposure = over_exposure,
            picks_over_limit    = over_limit,
            picks_stake_trimmed = stake_trimmed,
            total_exposure_pct  = total_exposure,
            profile_used        = profile,
        )