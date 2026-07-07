"""
core/value/filters.py

MarketFilters: umbrales configurables por deporte y mercado para
determinar si un CandidatePick tiene suficiente valor para ser activo.

Posición en el pipeline
------------------------
    CandidatePick (BlendingEngine + KellyCriterion)
        ↓
    MarketFilters.evaluate(pick) → FilterResult
        ↓
    FilterResult.passed → True: pick activo → StakingStrategy
                        → False: pick rechazado → registrar motivo

Motivación
-----------
El sistema MLB-PREDICTOR-ADVANCED tenía todos los umbrales hardcodeados
como constantes Python en analysis/value.py:

    UMBRAL_EV_ML    = 6       UMBRAL_EV_TOTAL    = 15
    UMBRAL_PROB_ML  = 0.535   UMBRAL_PROB_TOTAL  = 0.56
    MIN_CUOTA_ML    = 1.88    MAX_CUOTA_ML       = 2.12
    EDGE_MIN_ML     = 0.035   EDGE_MAX_ML        = 0.180

Tres problemas estructurales documentados en MLB_EDGE_AUDIT.md:

1. No recalibrables sin tocar código: cambiar un umbral requería commit,
   deploy y verificación manual de que no se rompió nada más.

2. Iguales para todos los deportes: un MIN_CUOTA=1.88 tiene sentido
   para MLB donde los precios rondan 2.0, pero no para Soccer 1X2
   donde los precios de draw están en 3.0+.

3. Sin trazabilidad de rechazo: el sistema retornaba True/False sin
   registrar qué criterio específico causó el rechazo. El backtester
   no podía distinguir "rechazado por EV bajo" de "rechazado por cuota
   fuera de rango" — información crítica para calibración.

Configuración YAML esperada
-----------------------------
# config/mlb.yaml
filters:
  ML:
    min_ev: 6
    min_prob: 0.535
    min_odds: 1.88
    max_odds: 2.12
    min_edge: 0.035
    max_edge: 0.180
  TOTAL:
    min_ev: 15
    min_prob: 0.56
    min_odds: 1.75
    max_odds: 2.20
    min_line_diff: 1.0      # discrepancia mínima proyección-línea

# config/soccer.yaml
filters:
  '1X2':
    min_ev: 8
    min_prob: 0.50
    min_odds: 1.70
    max_odds: 2.50
  TOTAL:
    min_ev: 12
    min_prob: 0.55
    min_line_diff: 0.5      # goles mínimos de discrepancia

Todos los umbrales son opcionales (None = no aplica).
Un deporte nuevo puede empezar con solo min_ev activo.

Por qué max_edge existe
------------------------
Un edge demasiado alto (ej. 0.25) es estadísticamente sospechoso:
significa que el modelo cree tener 25% más de confianza que el mercado
eficiente. En la práctica, esto casi siempre indica un error en los
datos de entrada (una cuota incorrecta, una probabilidad de modelo
calculada con datos stale) más que valor real. El sistema MLB filtró
con EDGE_MAX=0.180, lo que redujo picks con datos corruptos.

Por qué min_line_diff es parámetro separado en evaluate()
-----------------------------------------------------------
CandidatePick.line es la línea DE MERCADO (ej. 8.5 carreras).
projected_value es la proyección DEL MODELO (ej. 10.2 carreras).
La diferencia |10.2 - 8.5| = 1.7 > min_line_diff=1.0 → pasa el filtro.
CandidatePick no almacena projected_value — es información del
ProjectionModel que el ValueEngine tiene disponible cuando llama
evaluate(). Se pasa como argumento explícito opcional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.contracts.pick import CandidatePick

# Default mínimo de EV para deportes sin configuración YAML.
# 3.0% es el mínimo estadístico para que un pick tenga sentido de
# valor esperado positivo con cualquier nivel razonable de confianza.
# Conservador pero operativo: no bloquea todo el sistema para deportes
# nuevos, pero sí filtra picks sin valor aparente.
_DEFAULT_MIN_EV: float = 3.0


# ── Configuración de umbrales ─────────────────────────────────────────────────

@dataclass(frozen=True)
class FilterThresholds:
    """
    Umbrales de filtrado para un (sport, market) específico.

    Inmutable: representa la configuración vigente en el momento de
    evaluar el pick. Todos los campos son opcionales — None significa
    que ese filtro no se aplica para este (sport, market).

    Permite habilitar filtros incrementalmente: un deporte nuevo puede
    empezar con solo min_ev activo y agregar min_edge cuando tenga
    suficiente backtesting para calibrar ese umbral.

    Campos
    ------
    min_ev         -- EV mínimo (%). Rechaza picks con valor esperado
                     por debajo de este umbral.
    min_prob       -- Probabilidad blended mínima. Evita apostar en
                     selecciones donde el modelo tiene muy baja confianza.
    min_odds       -- Cuota decimal mínima. Rechaza favoritos extremos
                     donde el pago no justifica el riesgo de error del
                     modelo.
    max_odds       -- Cuota decimal máxima. Rechaza underdogs extremos
                     donde la liquidez del mercado es baja y la cuota
                     puede ser ineficiente por razones ajenas al valor.
    min_edge       -- Edge mínimo (model_prob_raw - market_prob).
                     Evita picks donde el modelo no tiene ventaja
                     estadística real sobre el mercado.
    max_edge       -- Edge máximo. Rechaza picks donde la ventaja
                     percibida es tan alta que es estadísticamente
                     sospechosa (posible error en datos, no valor real).
    min_line_diff  -- Diferencia mínima |projected_value - line|.
                     Específico para mercados TOTAL y SPREAD: evita
                     apostar cuando la discrepancia entre proyección
                     y línea es demasiado pequeña para ser concluyente.
                     Ignorado si projected_value no se provee en
                     evaluate().
    """
    min_ev:        float | None = _DEFAULT_MIN_EV
    min_prob:      float | None = None
    min_odds:      float | None = None
    max_odds:      float | None = None
    min_edge:      float | None = None
    max_edge:      float | None = None
    min_line_diff: float | None = None

    def __post_init__(self) -> None:
        # Validar consistencia cuando ambos extremos están definidos
        if self.min_odds is not None and self.max_odds is not None:
            if self.min_odds >= self.max_odds:
                raise ValueError(
                    f"min_odds={self.min_odds} debe ser < max_odds={self.max_odds}."
                )
        if self.min_edge is not None and self.max_edge is not None:
            if self.min_edge >= self.max_edge:
                raise ValueError(
                    f"min_edge={self.min_edge} debe ser < max_edge={self.max_edge}."
                )


# ── Resultado del filtrado ────────────────────────────────────────────────────

@dataclass(frozen=True)
class FilterResult:
    """
    Resultado completo de la evaluación de filtros para un pick.

    Inmutable: representa una decisión tomada en un instante del pipeline.
    Se serializa al trail de reasons de CandidatePick para trazabilidad
    completa en backtesting — permite distinguir "rechazado por EV bajo"
    de "rechazado por cuota fuera de rango", información crítica para
    calibrar umbrales con datos históricos.

    Campos
    ------
    passed           -- True si todos los criterios activos pasaron.
                       False si al menos uno falló.
    failed_criteria  -- Lista de criterios fallidos con el valor real
                       del pick y el umbral que no se cumplió.
                       Vacía si passed=True.
    passed_criteria  -- Lista de criterios que sí se cumplieron.
                       Útil para debug y auditoría.
    thresholds_used  -- FilterThresholds aplicados. Para auditoría:
                       detectar si un pick fue evaluado con la
                       configuración equivocada de sport/market.
    """
    passed:           bool
    failed_criteria:  list[str]
    passed_criteria:  list[str]
    thresholds_used:  FilterThresholds

    def to_reason(self) -> str:
        """
        Serialización compacta para CandidatePick.add_reason().
        """
        if self.passed:
            return f"FILTERS PASSED ({len(self.passed_criteria)} criterios)"
        return (
            f"FILTERS REJECTED: "
            f"{'; '.join(self.failed_criteria)}"
        )


# ── Motor de filtros ──────────────────────────────────────────────────────────

class MarketFilters:
    """
    Motor de filtrado calibrable por deporte y mercado.

    Lee umbrales desde ConfigLoader (YAML por deporte/mercado) con
    defaults conservadores para deportes sin configuración previa.

    Cachea instancias de FilterThresholds por (sport, market) —
    mismo patrón que BlendingEngine y KellyCriterion.

    Parámetros
    ----------
    config  -- ConfigLoader opcional. Si se provee, los umbrales
               se leen de YAML bajo filters.{MARKET}.{umbral}.
               Si es None, se aplica solo min_ev=3.0 para todos
               los mercados.
    """

    def __init__(self, config=None) -> None:
        self._config = config
        self._cache: dict[tuple[str, str], FilterThresholds] = {}

    def evaluate(
        self,
        pick: CandidatePick,
        projected_value: float | None = None,
    ) -> FilterResult:
        """
        Evalúa si un CandidatePick cumple todos los umbrales activos.

        Parámetros
        ----------
        pick             -- CandidatePick a evaluar. Se usan pick.ev,
                           pick.edge, pick.price, pick.blended_prob y
                           pick.event.sport + pick.market para resolver
                           la configuración de umbrales.
        projected_value   -- Proyección del modelo para este mercado
                           (ej. total de carreras proyectadas para un
                           pick TOTAL). Solo se usa para evaluar
                           min_line_diff. Si es None, ese filtro se
                           omite aunque esté configurado.

        Retorna
        -------
        FilterResult con trazabilidad completa de qué criterios
        pasaron y cuáles fallaron.
        """
        sport  = pick.event.sport.lower()
        market = pick.market.upper()

        thresholds = self._get_thresholds(sport, market)
        failed: list[str]  = []
        passed_: list[str] = []

        # ── min_ev ────────────────────────────────────────────────────
        if thresholds.min_ev is not None:
            if pick.ev < thresholds.min_ev:
                failed.append(
                    f"ev={pick.ev:.2f} < min_ev={thresholds.min_ev}"
                )
            else:
                passed_.append(f"ev={pick.ev:.2f} ✓ min_ev={thresholds.min_ev}")

        # ── min_prob ──────────────────────────────────────────────────
        if thresholds.min_prob is not None:
            if pick.blended_prob < thresholds.min_prob:
                failed.append(
                    f"prob={pick.blended_prob:.4f} < min_prob={thresholds.min_prob}"
                )
            else:
                passed_.append(
                    f"prob={pick.blended_prob:.4f} ✓ min_prob={thresholds.min_prob}"
                )

        # ── min_odds / max_odds ───────────────────────────────────────
        if thresholds.min_odds is not None:
            if pick.price < thresholds.min_odds:
                failed.append(
                    f"odds={pick.price} < min_odds={thresholds.min_odds}"
                )
            else:
                passed_.append(f"odds={pick.price} ✓ min_odds={thresholds.min_odds}")

        if thresholds.max_odds is not None:
            if pick.price > thresholds.max_odds:
                failed.append(
                    f"odds={pick.price} > max_odds={thresholds.max_odds}"
                )
            else:
                passed_.append(f"odds={pick.price} ✓ max_odds={thresholds.max_odds}")

        # ── min_edge / max_edge ───────────────────────────────────────
        if thresholds.min_edge is not None:
            if pick.edge < thresholds.min_edge:
                failed.append(
                    f"edge={pick.edge:.4f} < min_edge={thresholds.min_edge}"
                )
            else:
                passed_.append(
                    f"edge={pick.edge:.4f} ✓ min_edge={thresholds.min_edge}"
                )

        if thresholds.max_edge is not None:
            if pick.edge > thresholds.max_edge:
                failed.append(
                    f"edge={pick.edge:.4f} > max_edge={thresholds.max_edge}"
                )
            else:
                passed_.append(
                    f"edge={pick.edge:.4f} ✓ max_edge={thresholds.max_edge}"
                )

        # ── min_line_diff (solo si projected_value se provee) ─────────
        if thresholds.min_line_diff is not None and projected_value is not None:
            if pick.line is None:
                # Sin línea de mercado no se puede calcular la diferencia
                passed_.append("line_diff=N/A (mercado sin línea)")
            else:
                line_diff = abs(projected_value - pick.line)
                if line_diff < thresholds.min_line_diff:
                    failed.append(
                        f"line_diff={line_diff:.2f} < "
                        f"min_line_diff={thresholds.min_line_diff}"
                    )
                else:
                    passed_.append(
                        f"line_diff={line_diff:.2f} ✓ "
                        f"min_line_diff={thresholds.min_line_diff}"
                    )

        return FilterResult(
            passed=len(failed) == 0,
            failed_criteria=failed,
            passed_criteria=passed_,
            thresholds_used=thresholds,
        )

    def get_thresholds(
        self,
        sport: str,
        market: str,
    ) -> FilterThresholds:
        """
        API pública para acceder a los umbrales configurados.

        Útil para el backtester y para debuggear qué umbrales
        están activos para un (sport, market) dado.
        """
        return self._get_thresholds(sport.lower(), market.upper())

    def clear_cache(self) -> None:
        """Limpia el caché de FilterThresholds."""
        self._cache.clear()

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _get_thresholds(self, sport: str, market: str) -> FilterThresholds:
        """
        Obtiene o construye FilterThresholds para (sport, market).
        Caché por (sport, market): evitar re-parsear YAML en cada pick.
        """
        key = (sport, market)
        if key in self._cache:
            return self._cache[key]

        thresholds = self._build_thresholds(market)
        self._cache[key] = thresholds
        return thresholds

    def _build_thresholds(self, market: str) -> FilterThresholds:
        """
        Construye FilterThresholds desde YAML o defaults.

        Lee cada umbral individualmente desde la ruta
        filters.{MARKET}.{umbral} con fallback a None (no aplica)
        excepto min_ev que tiene default _DEFAULT_MIN_EV.
        """
        if self._config is None:
            return FilterThresholds()

        def get(key: str, default=None):
            return self._config.get( # type: ignore
                f"filters.{market}.{key}",
                default=default,
            )

        raw_min_ev = get('min_ev', _DEFAULT_MIN_EV)
        raw_min_ev = float(raw_min_ev) if raw_min_ev is not None else _DEFAULT_MIN_EV

        return FilterThresholds(
            min_ev        = raw_min_ev,
            min_prob      = _to_float(get('min_prob')),
            min_odds      = _to_float(get('min_odds')),
            max_odds      = _to_float(get('max_odds')),
            min_edge      = _to_float(get('min_edge')),
            max_edge      = _to_float(get('max_edge')),
            min_line_diff = _to_float(get('min_line_diff')),
        )


# ── Utilidades ────────────────────────────────────────────────────────────────

def _to_float(value) -> float | None:
    """Convierte un valor de YAML a float, retorna None si es None."""
    if value is None:
        return None
    return float(value)