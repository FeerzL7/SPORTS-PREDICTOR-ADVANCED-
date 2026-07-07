"""
core/value/blending.py

BlendingEngine: mezcla calibrable de probabilidad de modelo con
probabilidad de mercado no-vig.

Contexto y motivación
----------------------
El sistema MLB-PREDICTOR-ADVANCED calculaba blended_prob con pesos
hardcodeados como constantes Python en analysis/value.py:

    PROB_MODEL_WEIGHT_ML    = 0.35
    PROB_MODEL_WEIGHT_RL    = 0.35
    PROB_MODEL_WEIGHT_TOTAL = 0.62

Dos problemas estructurales documentados en MLB_EDGE_AUDIT.md:

1. No recalibrables sin tocar código: cualquier ajuste requería un
   commit, un deploy y esperanza de que no se rompiera nada más en
   el mismo archivo. En la práctica, estos pesos nunca se tocaron
   aunque el backtesting mostraba ROI=-21.56% para ML.

2. El prob_cap se aplicaba DESPUÉS del blend:
       blended = model * w + market * (1-w)
       blended = min(blended, prob_cap)  ← INCORRECTO
   Esto truncaba blended_prob FUERA del rango convexo
   [min(model, market), max(model, market)], produciendo valores
   imposibles para una combinación lineal legítima. En este sistema,
   CandidatePick.__post_init__ rechaza esos valores con ValueError.

Correcciones aplicadas aquí
-----------------------------
1. Pesos en YAML por (sport, market) — configurables sin tocar código.
2. prob_cap aplicado a model_prob ANTES del blend — preserva la
   propiedad convexa del resultado y respeta el contrato de
   CandidatePick.

Fórmula correcta
-----------------
    model_prob_clamped = min(model_prob_raw, prob_cap)
    blended = model_weight × model_prob_clamped
              + (1 - model_weight) × market_prob
    blended = max(MIN_BLENDED_PROB, blended)

Donde model_weight ∈ [0, 1] y prob_cap ∈ (0, 1].

Esta fórmula garantiza:
    blended ∈ [min(model_clamped, market_prob),
               max(model_clamped, market_prob)]
    (rango convexo — propiedad garantizada por la combinación lineal)

Configuración YAML esperada (ejemplos)
----------------------------------------
# config/mlb.yaml
blending:
  ML:
    model_weight: 0.35   # 65% mercado — mercado MLB-ML es eficiente
    prob_cap: 0.58
  RL:
    model_weight: 0.35
    prob_cap: 0.57
  TOTAL:
    model_weight: 0.62   # modelo más informativo en totales
    prob_cap: 0.62

# config/soccer.yaml
blending:
  1X2:
    model_weight: 0.45   # mercado soccer menos eficiente que MLB-ML
    prob_cap: 0.65
  TOTAL:
    model_weight: 0.55
    prob_cap: 0.60

Uso típico
-----------
    from core.value.blending import BlendingEngine

    engine = BlendingEngine(config=load_config(sport='mlb'))
    result = engine.blend(
        model_prob=0.58,
        market_prob=0.51,
        market='TOTAL',
        sport='mlb',
    )
    # result.blended_prob → 0.553  (62% modelo + 38% mercado)
    # result.cap_applied  → False   (0.58 < prob_cap=0.62)
    # result.weight_used  → 0.62
"""

from __future__ import annotations

from dataclasses import dataclass

# Piso mínimo de blended_prob — nunca retornar 0.0 o negativo.
# 2% es el mínimo estadísticamente sensato: una probabilidad por debajo
# de esto en un mercado de apuestas deportivas casi siempre indica
# un dato corrupto o un modelo completamente desconectado del mercado.
_MIN_BLENDED_PROB: float = 0.02

# Defaults seguros para deportes/mercados sin configuración en YAML.
# model_weight=0.50: punto de indiferencia entre modelo y mercado —
# el más neutro posible para un deporte sin calibración previa.
# prob_cap=1.0: sin cap — no limita la probabilidad del modelo.
_DEFAULT_MODEL_WEIGHT: float = 0.50
_DEFAULT_PROB_CAP: float = 1.0


# ── Tipos de retorno ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BlendingConfig:
    """
    Parámetros de blending para un (sport, market) específico.

    Inmutable: representa la configuración vigente en el momento de
    calcular el blend. Si los parámetros cambian (recalibración via
    backtesting), se crea un nuevo BlendingConfig — el BlendingEngine
    nunca muta su configuración en mitad de una ejecución de pipeline.

    Campos
    ------
    model_weight  -- Peso del modelo deportivo en la mezcla. ∈ [0, 1].
                     0.0 → 100% mercado (ignorar el modelo completamente).
                     1.0 → 100% modelo (ignorar el mercado completamente).
                     Valor típico calibrado: 0.35–0.65 según deporte/mercado.
    prob_cap       -- Límite superior de model_prob antes del blend. ∈ (0, 1].
                     Evita que el modelo tenga más del prob_cap de confianza
                     independientemente de lo que calcule. 1.0 = sin límite.
                     Valor típico calibrado: 0.57–0.65 según mercado.
    """
    model_weight: float
    prob_cap: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.model_weight <= 1.0:
            raise ValueError(
                f"model_weight={self.model_weight} debe estar en [0, 1]."
            )
        if not 0.0 < self.prob_cap <= 1.0:
            raise ValueError(
                f"prob_cap={self.prob_cap} debe estar en (0, 1]."
            )


@dataclass(frozen=True)
class BlendResult:
    """
    Resultado completo del proceso de blending para una selección.

    Inmutable: un BlendResult producido representa una decisión tomada
    en un instante del pipeline. Se serializa al trail de reasons de
    CandidatePick para trazabilidad completa en backtesting.

    Campos
    ------
    blended_prob        -- Probabilidad final tras mezcla. Siempre en
                          [min(model_prob_clamped, market_prob),
                           max(model_prob_clamped, market_prob)].
                          Garantizado por la fórmula de combinación
                          lineal convexa.
    model_prob_raw       -- Probabilidad del modelo antes de cualquier
                          transformación. Para trazabilidad — permite
                          comparar con model_prob_clamped y detectar
                          si el cap fue la causa de un cambio de pick.
    model_prob_clamped   -- Probabilidad del modelo después de aplicar
                          prob_cap. Igual a model_prob_raw si no se
                          aplicó cap (cap_applied=False).
    market_prob          -- Probabilidad no-vig del mercado. Input directo
                          de no_vig_probabilities().
    weight_used          -- model_weight efectivo usado en este blend.
                          Puede diferir del configurado si se aplicó
                          algún ajuste dinámico (reservado para
                          extensiones futuras — hoy siempre igual al
                          configurado).
    cap_applied          -- True si model_prob_raw > prob_cap y fue
                          truncado. Señal importante para el trail de
                          decisión: "el modelo tenía confianza alta pero
                          el cap la limitó".
    config               -- BlendingConfig usada. Para auditoría:
                          permite detectar si un pick fue calculado con
                          la configuración equivocada de sport/market.
    """
    blended_prob:       float
    model_prob_raw:     float
    model_prob_clamped: float
    market_prob:        float
    weight_used:        float
    cap_applied:        bool
    config:             BlendingConfig

    def to_reason(self) -> str:
        """
        Serialización compacta para CandidatePick.add_reason().

        Formato legible por humanos, reproducible para debugging.
        """
        cap_note = f" [CAP aplicado: {self.model_prob_raw:.4f}→{self.model_prob_clamped:.4f}]" \
            if self.cap_applied else ""
        return (
            f"BLEND: model={self.model_prob_raw:.4f}{cap_note} "
            f"market={self.market_prob:.4f} "
            f"weight={self.weight_used:.2f} "
            f"→ blended={self.blended_prob:.4f}"
        )


# ── Motor de blending ──────────────────────────────────────────────────────────

class BlendingEngine:
    """
    Motor de blending calibrable modelo-mercado.

    Lee pesos de blending desde ConfigLoader (YAML por deporte/mercado)
    con defaults seguros para deportes sin configuración previa.

    Cachea instancias de BlendingConfig por (sport, market) para evitar
    parsear el YAML en cada llamada durante un pipeline con cientos de
    eventos — mismo patrón que DistributionFactory.

    Parámetros
    ----------
    config  -- ConfigLoader opcional. Si se provee, los parámetros de
               blending se leen de YAML bajo
               blending.{MARKET}.model_weight y
               blending.{MARKET}.prob_cap.
               Si es None, se usan _DEFAULT_MODEL_WEIGHT y
               _DEFAULT_PROB_CAP para todos los mercados.
    """

    def __init__(self, config=None) -> None:
        self._config = config
        self._blending_config_cache: dict[tuple[str, str], BlendingConfig] = {}

    def _get_blending_config(self, sport: str, market: str) -> BlendingConfig:
        """
        Obtiene o construye el BlendingConfig para (sport, market).

        Caché por (sport, market): evitar re-parsear YAML en cada
        llamada al blend durante un pipeline completo.
        """
        key = (sport.lower(), market.upper())
        if key in self._blending_config_cache:
            return self._blending_config_cache[key]

        market_upper = market.upper()

        if self._config is not None:
            weight = self._config.get(
                f"blending.{market_upper}.model_weight",
                default=_DEFAULT_MODEL_WEIGHT,
            )
            cap = self._config.get(
                f"blending.{market_upper}.prob_cap",
                default=_DEFAULT_PROB_CAP,
            )
        else:
            weight = _DEFAULT_MODEL_WEIGHT
            cap = _DEFAULT_PROB_CAP

        bc = BlendingConfig(model_weight=float(weight), prob_cap=float(cap))
        self._blending_config_cache[key] = bc
        return bc

    def blend(
        self,
        model_prob: float,
        market_prob: float,
        market: str,
        sport: str,
    ) -> BlendResult:
        """
        Calcula la probabilidad blended para una selección.

        Opera sobre una selección individual — compatible con mercados
        binarios (ML, TOTAL) y N-arios (Soccer 1X2, Golf outright).
        Para N selecciones, llamar blend() N veces con los mismos
        parámetros de (sport, market).

        Parámetros
        ----------
        model_prob   -- Probabilidad estimada por el modelo deportivo
                       para esta selección. ∈ (0, 1).
        market_prob  -- Probabilidad no-vig del mercado para esta
                       selección. Output de no_vig_probabilities().
                       no_vig_prob debe estar poblado en MarketOdds
                       antes de llamar blend().
        market        -- Identificador del mercado ('ML', 'TOTAL',
                       'SPREAD', '1X2', etc.). Determina qué pesos
                       de blending leer del YAML.
        sport         -- Identificador del deporte ('mlb', 'nba', etc.).
                       Determina qué sección de configuración usar.

        Retorna
        -------
        BlendResult
            Resultado completo con trazabilidad. blended_prob está
            garantizado en [min(model_prob_clamped, market_prob),
            max(model_prob_clamped, market_prob)], respetando el
            contrato de CandidatePick._validate_blended_prob_is_reachable.

        Raises
        ------
        ValueError
            Si model_prob o market_prob están fuera de (0, 1).
            Detecta datos corruptos en el punto de entrada.
        """
        self._validate_inputs(model_prob, market_prob, market, sport)

        bc = self._get_blending_config(sport, market)

        # Aplicar prob_cap a model_prob ANTES del blend — garantiza
        # que blended_prob caiga dentro del rango convexo
        # [min(model_clamped, market_prob), max(model_clamped, market_prob)].
        cap_applied = model_prob > bc.prob_cap
        model_clamped = min(model_prob, bc.prob_cap)

        # Mezcla lineal convexa
        blended = (
            bc.model_weight * model_clamped
            + (1.0 - bc.model_weight) * market_prob
        )

        # Piso mínimo — nunca retornar probabilidad <= 0
        blended = max(_MIN_BLENDED_PROB, blended)
        blended = round(blended, 6)

        return BlendResult(
            blended_prob=blended,
            model_prob_raw=model_prob,
            model_prob_clamped=round(model_clamped, 6),
            market_prob=market_prob,
            weight_used=bc.model_weight,
            cap_applied=cap_applied,
            config=bc,
        )

    def clear_cache(self) -> None:
        """
        Limpia el caché de BlendingConfig.

        Útil en tests, o si el ConfigLoader cambió y se necesita
        releer los parámetros de blending actualizados.
        """
        self._blending_config_cache.clear()

    @staticmethod
    def _validate_inputs(
        model_prob: float,
        market_prob: float,
        market: str,
        sport: str,
    ) -> None:
        """
        Valida que model_prob y market_prob sean probabilidades válidas.

        0.0 exacto se rechaza para model_prob (un modelo que asigna 0%
        a una selección que tiene cuota en el mercado es casi siempre
        un error de cálculo). 0.0 para market_prob es imposible si
        price > 1.0 (invariante de MarketOdds), pero se valida para
        detectar llamadas directas con datos incorrectos.
        """
        for name, value in (("model_prob", model_prob), ("market_prob", market_prob)):
            if not 0.0 < value <= 1.0:
                raise ValueError(
                    f"{name}={value} fuera de rango (0, 1] para "
                    f"sport='{sport}', market='{market}'. "
                    f"Verificar que no_vig_probabilities() se ejecutó "
                    f"antes de llamar blend(), y que el modelo deportivo "
                    f"no asigna probabilidad 0 o negativa."
                )