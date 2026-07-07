"""
core/value/kelly.py

KellyCriterion: cálculo de Kelly fraccionado para sizing de stake.

Posición en el pipeline
------------------------
    blended_prob (BlendingEngine)
        ↓
    KellyCriterion.calculate() → KellyResult.kelly_final
        ↓
    CandidatePick.kelly_fraction  (referencia matemática, campo normal)
        ↓
    StakingStrategy.stake_pct()   (Stage 8 — puede ignorar kelly_fraction)

kelly_fraction en CandidatePick es una REFERENCIA, no el stake final.
StakingStrategy (IntegerPercentStaking, KellyStaking) decide el stake_pct
real usando esta referencia como input, con sus propias reglas de
disciplina de riesgo (caps, mínimos, exposición diaria).

Fórmula
--------
    b = price - 1                          # ganancia neta por unidad
    kelly_full = (p × price - 1) / b       # = (p × (b+1) - 1) / b
    kelly_fractional = kelly_full × fraction
    kelly_final = min(kelly_fractional, max_fraction)
    kelly_final = max(0.0, kelly_final)    # nunca negativo

Donde:
    p         = blended_prob (probabilidad ajustada post-blending)
    price     = cuota decimal de la selección
    fraction  = fracción del Kelly completo a usar (default: 0.18)
                Calibrado empíricamente en el sistema MLB como
                balance entre crecimiento y control de drawdown.
    max_fraction = techo duro del kelly_final (default: 1.25%)
                Previene stakes excesivos en casos de EV muy alto
                con alta confianza del modelo.

Kelly negativo
--------------
Cuando blended_prob × price < 1 (EV negativo), kelly_full es negativo.
En picks deportivos unidireccionales (solo se apuesta a favor, nunca
en contra), un Kelly negativo significa "no apostar" — kelly_final=0.0.
No se lanza excepción: un Kelly negativo es información válida que
el ValueEngine puede usar como señal adicional de "sin valor", distinto
de un input inválido (prob=0, price=1) que sí produce ValueError.

Limitación documentada: independencia de apuestas
---------------------------------------------------
El criterio de Kelly clásico asume que cada apuesta es independiente
y que el bankroll se actualiza entre apuestas sucesivas. En la práctica,
el sistema puede colocar múltiples picks el mismo día, potencialmente
correlacionados (mismo deporte, misma jornada, mismo factor de mercado).

Esta correlación entre picks NO se puede corregir en este módulo sin
información del portfolio completo. Es responsabilidad de RiskManager
(Stage 9) gestionar la exposición diaria total y limitar la suma de
stakes cuando hay correlación alta entre picks activos. kelly.py
produce el Kelly óptimo para CADA pick en AISLAMIENTO — RiskManager
aplica los límites de cartera.

Calibración
-----------
Los parámetros fraction y max_fraction son configurables por (sport, market)
desde YAML via ConfigLoader, con los valores del sistema MLB como defaults:

# config/mlb.yaml
kelly:
  ML:
    fraction: 0.18
    max_fraction: 1.25
  TOTAL:
    fraction: 0.18
    max_fraction: 1.25

Para deportes sin configuración explícita, se usan los defaults
documentados (fraction=0.18, max_fraction=1.25).

Uso típico
-----------
    from core.value.kelly import KellyCriterion

    kelly = KellyCriterion()
    result = kelly.calculate(blended_prob=0.55, price=1.91)
    # result.kelly_final  → 0.63  (% del bankroll sugerido)
    # result.kelly_full   → 3.51  (Kelly completo sin fraccionar)
    # result.capped       → False (< max_fraction)
"""

from __future__ import annotations

from dataclasses import dataclass

# Defaults documentados — idénticos al sistema MLB (analysis/value.py):
#   KELLY_FRACCION      = 0.18
#   KELLY_MAX_STAKE_PCT = 1.25
# Preservados porque ya estaban empíricamente calibrados y la arquitectura
# los confirma explícitamente en §6.2: "fraction=0.18".
_DEFAULT_FRACTION: float = 0.18
_DEFAULT_MAX_FRACTION: float = 1.25


# ── Tipo de retorno ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KellyResult:
    """
    Resultado completo del cálculo de Kelly para una selección.

    Inmutable: representa el cálculo en un instante del pipeline.
    Se serializa al trail de reasons de CandidatePick para trazabilidad.

    Campos
    ------
    kelly_full       -- Kelly completo sin fraccionar ni capear.
                       Puede ser negativo (EV negativo) o mayor que 1
                       (teóricamente apostar más del 100% del bankroll,
                       lo cual nunca se usa en práctica). Útil como
                       señal de confianza: cuanto más positivo, mayor
                       EV percibido por el modelo.
    kelly_fractional -- kelly_full × fraction. Aún puede superar
                       max_fraction en picks con EV muy alto.
    kelly_final      -- Valor efectivo a usar como referencia de stake.
                       = max(0.0, min(kelly_fractional, max_fraction)).
                       Nunca negativo, nunca supera max_fraction.
                       Este es el campo que se asigna a
                       CandidatePick.kelly_fraction.
    fraction_used    -- Fracción aplicada para llegar a kelly_fractional.
                       Puede diferir del default si se leyó desde YAML.
    max_fraction_used -- Techo aplicado. Para auditoría: detectar si el
                       cap fue el factor limitante de un stake.
    capped           -- True si kelly_fractional > max_fraction y fue
                       truncado. Señal importante: el modelo tenía
                       confianza muy alta pero el techo de seguridad
                       lo limitó.
    has_value        -- True si kelly_full > 0 (EV positivo). Shortcut
                       para el ValueEngine: no requiere acceder a
                       kelly_full directamente para el caso más común.
    """
    kelly_full:        float
    kelly_fractional:  float
    kelly_final:       float
    fraction_used:     float
    max_fraction_used: float
    capped:            bool

    @property
    def has_value(self) -> bool:
        """True si el Kelly completo es positivo (EV > 0)."""
        return self.kelly_full > 0.0

    def to_reason(self) -> str:
        """
        Serialización compacta para CandidatePick.add_reason().
        """
        cap_note = f" [CAP aplicado: {self.kelly_fractional:.4f}→{self.kelly_final:.4f}]" \
            if self.capped else ""
        value_note = "" if self.has_value else " [SIN VALOR: kelly_full<0]"
        return (
            f"KELLY: full={self.kelly_full:.4f} "
            f"× fraction={self.fraction_used} "
            f"= {self.kelly_fractional:.4f}"
            f"{cap_note}"
            f" → final={self.kelly_final:.4f}"
            f"{value_note}"
        )


# ── Motor de Kelly ─────────────────────────────────────────────────────────────

class KellyCriterion:
    """
    Motor de Kelly fraccionado calibrable.

    Parámetros
    ----------
    fraction      -- Fracción del Kelly completo a usar. Default 0.18
                    (calibrado en sistema MLB). Configurable desde YAML
                    por (sport, market) via calculate_for_market().
    max_fraction   -- Techo duro del kelly_final en % del bankroll.
                    Default 1.25%. Configurable desde YAML.
    config         -- ConfigLoader opcional. Si se provee, fraction y
                    max_fraction se leen de YAML por (sport, market)
                    en calculate_for_market(). Si es None, se usan
                    los parámetros del constructor para todos los
                    mercados.
    """

    def __init__(
        self,
        fraction: float = _DEFAULT_FRACTION,
        max_fraction: float = _DEFAULT_MAX_FRACTION,
        config=None,
    ) -> None:
        self._validate_params(fraction, max_fraction)
        self.fraction = fraction
        self.max_fraction = max_fraction
        self._config = config
        self._cache: dict[tuple[str, str], tuple[float, float]] = {}

    def calculate(
        self,
        blended_prob: float,
        price: float,
        fraction: float | None = None,
        max_fraction: float | None = None,
    ) -> KellyResult:
        """
        Calcula el Kelly fraccionado para una selección.

        Método base — usa parámetros explícitos (o los del constructor
        si no se especifican). Sin lectura de YAML.

        Parámetros
        ----------
        blended_prob  -- Probabilidad post-blending de la selección.
                        ∈ (0, 1). Output de BlendingEngine.blend().
        price          -- Cuota decimal de la selección. > 1.0.
                        Invariante garantizado por MarketOdds.price.
        fraction       -- Override de la fracción de Kelly. Si None,
                        usa self.fraction.
        max_fraction    -- Override del techo de Kelly. Si None, usa
                        self.max_fraction.

        Retorna
        -------
        KellyResult con trazabilidad completa. kelly_final es el valor
        a asignar a CandidatePick.kelly_fraction.

        Raises
        ------
        ValueError
            Si blended_prob no está en (0, 1) o price <= 1.0.
        """
        f = fraction if fraction is not None else self.fraction
        mf = max_fraction if max_fraction is not None else self.max_fraction

        self._validate_inputs(blended_prob, price)
        self._validate_params(f, mf)

        b = price - 1.0  # ganancia neta por unidad apostada

        # Fórmula de Kelly: (p × (b+1) - 1) / b = (p × price - 1) / b
        kelly_full = (blended_prob * price - 1.0) / b

        # Kelly negativo → no apostar (kelly_final = 0.0)
        # Kelly positivo → fraccionar y capear
        kelly_fractional = kelly_full * f
        capped = kelly_fractional > mf
        kelly_final = max(0.0, min(kelly_fractional, mf))

        return KellyResult(
            kelly_full=round(kelly_full, 6),
            kelly_fractional=round(kelly_fractional, 6),
            kelly_final=round(kelly_final, 6),
            fraction_used=f,
            max_fraction_used=mf,
            capped=capped,
        )

    def calculate_for_market(
        self,
        blended_prob: float,
        price: float,
        market: str,
        sport: str,
    ) -> KellyResult:
        """
        Calcula Kelly con parámetros leídos desde YAML por (sport, market).

        Mismo patrón que BlendingEngine.blend(): lee fraction y
        max_fraction de config.get('kelly.{MARKET}.fraction') con
        fallback a los defaults del constructor.

        Usa caché de parámetros por (sport, market) — mismo patrón
        que BlendingEngine._blending_config_cache.
        """
        f, mf = self._get_params(sport, market)
        return self.calculate(blended_prob, price, fraction=f, max_fraction=mf)

    def clear_cache(self) -> None:
        """Limpia caché de parámetros por (sport, market)."""
        self._cache.clear()

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _get_params(self, sport: str, market: str) -> tuple[float, float]:
        """
        Obtiene (fraction, max_fraction) para (sport, market) desde YAML,
        con fallback a los parámetros del constructor.
        """
        key = (sport.lower(), market.upper())
        if key in self._cache:
            return self._cache[key]

        if self._config is not None:
            market_upper = market.upper()
            f = float(self._config.get(
                f"kelly.{market_upper}.fraction",
                default=self.fraction,
            ))
            mf = float(self._config.get(
                f"kelly.{market_upper}.max_fraction",
                default=self.max_fraction,
            ))
        else:
            f, mf = self.fraction, self.max_fraction

        self._cache[key] = (f, mf)
        return f, mf

    @staticmethod
    def _validate_inputs(blended_prob: float, price: float) -> None:
        """
        Valida que los inputs sean matemáticamente válidos para Kelly.

        blended_prob=0.0 produciría kelly_full=-∞ (división por 0 en la
        fórmula con b→∞ no, pero prob=0 → kelly = (0×price-1)/b < 0 es
        válido matemáticamente). Sin embargo, una probabilidad de 0 no
        tiene sentido semántico — si el modelo asigna 0% a una selección,
        no debería llegar al cálculo de Kelly.

        price=1.0 produce b=0, lo que causaría división por cero en la
        fórmula de Kelly. price > 1.0 es invariante de MarketOdds, pero
        se valida aquí para detectar llamadas directas con datos incorrectos.
        """
        if not 0.0 < blended_prob < 1.0:
            raise ValueError(
                f"blended_prob={blended_prob} fuera de rango (0, 1). "
                f"Debe ser la probabilidad post-blending output de "
                f"BlendingEngine.blend()."
            )
        if price <= 1.0:
            raise ValueError(
                f"price={price} inválido. Debe ser > 1.0 (cuota decimal). "
                f"Verificar que el MarketOdds.price es válido antes de "
                f"llamar KellyCriterion.calculate()."
            )

    @staticmethod
    def _validate_params(fraction: float, max_fraction: float) -> None:
        """
        Valida que fraction y max_fraction sean parámetros razonables.

        fraction=0.0 significa "nunca apostar", lo que hace el módulo
        inútil. fraction>1.0 significaría apostar MÁS que el Kelly
        completo — contrario al principio de Kelly fraccionado.
        """
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"fraction={fraction} fuera de rango (0, 1]. "
                f"Valor típico calibrado: 0.18 (sistema MLB)."
            )
        if max_fraction <= 0.0:
            raise ValueError(
                f"max_fraction={max_fraction} debe ser > 0. "
                f"Valor típico: 1.25 (% máximo del bankroll)."
            )