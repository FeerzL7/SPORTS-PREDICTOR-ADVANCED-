"""
core/contracts/pick.py

Resultado del proceso de valoración de un mercado de apuestas.

CandidatePick es producido exclusivamente por el ValueEngine (Stage 6
del pipeline) y enriquecido progresivamente por los stages posteriores:
Stage 7 (line movement) añade razones de confirmación/contradicción,
Stage 8 (staking) fija stake_pct, Stage 9 (risk management) decide
active/inactive_reason.

Origen del diseño — corrección directa del hallazgo de mayor severidad
en MLB_EDGE_AUDIT.md:

    "EV calculado no correlaciona con rentabilidad real en ML y RL.
    El EV medio de ML es 46.83, pero el ROI es −22.43%. Esta brecha
    de ~70 puntos indica que el EV calculado está gravemente
    sobreestimado para ML."

    En el sistema MLB, ev se calculaba manualmente con _calc_ev() en
    tres lugares distintos (_decidir_ml, _decidir_rl, _decidir_total)
    sin garantía estructural de que las tres implementaciones
    coincidieran o se mantuvieran sincronizadas al ajustar el blending.

    Aquí ev y edge son PROPIEDADES CALCULADAS, no campos de dataclass:
    es matemáticamente imposible construir un CandidatePick cuyo ev
    no corresponda exactamente a (blended_prob × price - 1) × 100.
    Ningún código, en ningún stage, puede asignar un ev "a mano".

Uso típico:
    pick = CandidatePick(
        event=event, market="TOTAL", selection="over", line=8.5,
        price=1.91, model_prob_raw=0.58, market_prob=0.51,
        blended_prob=0.55, kelly_fraction=0.018,
        price_at_pick=1.91,
    )
    pick.ev    # -> calculado: (0.55 * 1.91 - 1) * 100 = 5.05
    pick.edge  # -> calculado: 0.58 - 0.51 = 0.07

    pick.add_reason("EV=5.05 no alcanza umbral mínimo de 6.0 para TOTAL")
    pick.stake_pct = 0
    # active permanece False hasta que RiskManager lo apruebe explícitamente
"""

from __future__ import annotations

from dataclasses import dataclass, field

from core.contracts.event import Event


# Tolerancia para la validación cruzada de blended_prob contra el rango
# convexo [min(model_prob_raw, market_prob), max(model_prob_raw, market_prob)].
# Permite pequeños errores de redondeo del BlendingEngine sin rechazar
# picks legítimos por imprecisión de punto flotante.
BLENDED_PROB_RANGE_TOLERANCE = 0.01


@dataclass
class CandidatePick:
    """
    Decisión de apuesta candidata, con trazabilidad completa.

    Mutable a propósito: se crea en Stage 6 y se enriquece en los
    stages 7, 8 y 9 sin reconstrucción. ev y edge, en cambio, NUNCA
    se mutan directamente — son propiedades de solo lectura derivadas
    de los campos base, recalculadas en cada acceso.

    Campos
    ------
    Identidad y mercado:
        event      -- Event de origen. Provee matchup, sport, league
                     para display y agrupación en ledger/dashboard.
        market      -- Igual que MarketOdds.market.
        selection   -- Igual que MarketOdds.selection.
        line        -- Igual que MarketOdds.line.
        price       -- Cuota decimal usada para este pick.

    Probabilidades:
        model_prob_raw -- Probabilidad del modelo deportivo puro,
                         antes de mezclar con el mercado. Viene de
                         Projection (home_win_prob, away_win_prob, o
                         la probabilidad de spread/total calculada por
                         ProbabilityModel).
        market_prob     -- Probabilidad no-vig del mercado para esta
                         selección. Viene de no_vig_probabilities()
                         sobre el conjunto de MarketOdds del mercado.
        blended_prob    -- Probabilidad final tras blending modelo-
                         mercado. VALIDADA en __post_init__: debe caer
                         dentro de [min(model_prob_raw, market_prob),
                         max(model_prob_raw, market_prob)] + tolerancia.
                         Una blended_prob fuera de ese rango convexo es
                         matemáticamente imposible para una mezcla
                         lineal legítima — señal de un BlendingEngine
                         con un bug real (peso corrupto, error de
                         signo), no un dato de baja calidad. Por eso
                         se rechaza con ValueError, no se registra como
                         data_quality_flag: el problema es de lógica
                         de cálculo, no de dato de entrada.

    Métricas de valor:
        edge            -- PROPIEDAD CALCULADA, no campo. Siempre
                         = model_prob_raw - market_prob. Ver docstring
                         del módulo.
        ev               -- PROPIEDAD CALCULADA, no campo. Siempre
                         = (blended_prob * price - 1) * 100. Ver
                         docstring del módulo.
        kelly_fraction   -- Campo normal (no calculado como ev/edge).
                         Es la fracción de Kelly de referencia que
                         calcula el ValueEngine, pero StakingStrategy
                         puede ignorarla completamente — el mismo
                         patrón que IntegerPercentStaking en el sistema
                         MLB, que sobrescribía el Kelly preliminar con
                         su propia escalera de porcentajes enteros.
        stake_pct        -- Campo normal. Lo fija StakingStrategy en
                         Stage 8. Default 0 — sin stake hasta que una
                         estrategia explícita lo determine.

    Calidad y trazabilidad:
        data_quality_flags -- Flags de DATOS de entrada de baja
                            confianza (lookup_fallback, recent_scores
                            vacío, etc.). NO se usa para errores de
                            lógica de blending — eso aborta la
                            construcción con ValueError.
        active              -- Aprobado por RiskManager. DEFAULT FALSE:
                            un pick recién construido en Stage 6 aún
                            no pasó por Risk Management (Stage 9). Si
                            el default fuera True, un bug que saltee
                            Stage 9 produciría picks aprobados por
                            omisión. Con default False, ese mismo bug
                            produce picks inactivos — el fallo seguro.
        inactive_reason     -- Por qué active=False. None si el pick
                            aún no fue evaluado por RiskManager.
        reasons             -- Trail append-only de la decisión. Usar
                            add_reason(), no reasons.append() directo.

    CLV tracking:
        price_at_pick  -- Cuota en el momento de registrar el pick.
                         Necesaria porque price puede mutar
                         conceptualmente entre el cálculo de valor y
                         el registro final, y CLV requiere comparar
                         contra la cuota exacta del pick, no contra
                         cualquier price posterior.
        closing_price   -- Cuota de cierre de mercado, si está
                         disponible. None hasta que el snapshot final
                         del día se procese.
        clv             -- PROPIEDAD CALCULADA cuando closing_price
                         no es None: (closing_price / price_at_pick - 1)
                         * 100. None si closing_price aún no existe.
                         Corrige la carencia identificada como más
                         crítica en MLB_EDGE_AUDIT.md: "CLV no se mide,
                         y es la métrica más importante ausente."
    """

    # ── Identidad y mercado ────────────────────────────────────────────────────
    event:     Event
    market:    str
    selection: str
    line:      float | None
    price:     float

    # ── Probabilidades ─────────────────────────────────────────────────────────
    model_prob_raw: float
    market_prob:    float
    blended_prob:   float

    # ── Métricas de valor (kelly_fraction y stake_pct son campos normales;
    #    ev y edge son propiedades — ver más abajo) ─────────────────────────────
    kelly_fraction: float = 0.0
    stake_pct:      int   = 0

    # ── Calidad y trazabilidad ─────────────────────────────────────────────────
    data_quality_flags: list[str] = field(default_factory=list)
    active:             bool      = False
    inactive_reason:    str | None = None
    reasons:             list[str] = field(default_factory=list)

    # ── CLV tracking ───────────────────────────────────────────────────────────
    price_at_pick: float = 0.0
    closing_price: float | None = None

    # ── Validación e invariantes ──────────────────────────────────────────────

    def __post_init__(self) -> None:
        """
        Aplica las invariantes estructurales del contrato.

        Si price_at_pick no se especifica explícitamente, se asume
        igual a price (el caso normal: el pick se registra con la
        misma cuota que se usó para calcularlo).
        """
        if self.price_at_pick == 0.0:
            self.price_at_pick = self.price

        self._validate_probability_ranges()
        self._validate_blended_prob_is_reachable()

    def _validate_probability_ranges(self) -> None:
        """model_prob_raw, market_prob y blended_prob deben estar en [0, 1]."""
        for name, value in (
            ("model_prob_raw", self.model_prob_raw),
            ("market_prob", self.market_prob),
            ("blended_prob", self.blended_prob),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name}={value} fuera de rango [0, 1] para "
                    f"event_id='{self.event.event_id}', market="
                    f"'{self.market}', selection='{self.selection}'."
                )

    def _validate_blended_prob_is_reachable(self) -> None:
        """
        blended_prob debe ser alcanzable como combinación convexa de
        model_prob_raw y market_prob.

        Una mezcla lineal blended = model*w + market*(1-w) con w en
        [0, 1] SIEMPRE cae dentro de [min(model, market), max(model,
        market)]. Si blended_prob cae fuera de ese rango (con
        tolerancia de redondeo), el BlendingEngine que la produjo
        tiene un bug real — peso fuera de [0,1], error de signo, o
        probabilidad de mercado mal calculada. Se rechaza aquí, en el
        punto de entrada al ValueEngine, en vez de dejar que un pick
        con probabilidad imposible reciba stake real.
        """
        lower = min(self.model_prob_raw, self.market_prob) - BLENDED_PROB_RANGE_TOLERANCE
        upper = max(self.model_prob_raw, self.market_prob) + BLENDED_PROB_RANGE_TOLERANCE

        if not (lower <= self.blended_prob <= upper):
            raise ValueError(
                f"blended_prob={self.blended_prob} no es alcanzable como "
                f"combinación convexa de model_prob_raw="
                f"{self.model_prob_raw} y market_prob={self.market_prob} "
                f"para event_id='{self.event.event_id}', market="
                f"'{self.market}', selection='{self.selection}'. Rango "
                f"válido: [{round(lower, 4)}, {round(upper, 4)}]. Esto "
                f"indica un bug en el BlendingEngine que produjo este "
                f"pick, no un problema de calidad de dato."
            )

    # ── Métricas de valor calculadas ───────────────────────────────────────────

    @property
    def edge(self) -> float:
        """
        edge = model_prob_raw - market_prob.

        Propiedad de solo lectura, recalculada en cada acceso. Ningún
        código puede asignar pick.edge directamente — intentarlo
        lanza AttributeError, porque no existe un setter.
        """
        return round(self.model_prob_raw - self.market_prob, 4)

    @property
    def ev(self) -> float:
        """
        ev = (blended_prob * price - 1) * 100.

        Propiedad de solo lectura, recalculada en cada acceso. Esta
        es la corrección estructural del hallazgo de mayor severidad
        en MLB_EDGE_AUDIT.md — ver docstring del módulo.
        """
        return round((self.blended_prob * self.price - 1) * 100, 2)

    # ── CLV tracking calculado ─────────────────────────────────────────────────

    @property
    def clv(self) -> float | None:
        """
        clv = (closing_price / price_at_pick - 1) * 100.

        Fórmula exacta documentada en SPORTS_PREDICTOR_ARCHITECTURE.md
        (sección 10.2): CLV = (closing_price / opening_price - 1) × 100.

        None mientras closing_price no esté disponible. Se recalcula
        automáticamente en cada acceso una vez que closing_price se
        asigna — no requiere ningún paso manual adicional, mismo
        patrón que implied_prob en MarketOdds.

        NOTA SOBRE CONVENCIÓN DE SIGNO (detectado en validación):
        Con esta fórmula literal, cuando la cuota BAJA después del pick
        —que es la señal de que el mercado confirma la dirección del
        pick, porque más dinero entró en esa selección— el resultado
        es NEGATIVO (closing/opening < 1). Cuando la cuota SUBE
        —mercado se aleja del pick—, el resultado es POSITIVO.

        Esto es matemáticamente correcto para la fórmula tal como está
        documentada, pero es opuesto a la interpretación intuitiva
        "CLV positivo = edge confirmado" que describe la arquitectura
        en su texto explicativo. core/evaluation/clv.py (Bloque 6 del
        roadmap) es responsable de aplicar la interpretación correcta
        por selección al momento de generar reportes y validar
        hipótesis — este contrato expone el cálculo crudo de la
        fórmula documentada, sin reinterpretar el signo aquí, para no
        introducir una segunda convención distinta a la que ya
        consume cualquier código que lea closing_price/price_at_pick
        directamente.

        CLV negativo (esta fórmula): el mercado se movió hacia la
            selección del pick — señal de edge real.
        CLV positivo (esta fórmula): el mercado se alejó de la
            selección del pick — señal de falta de edge.
        """
        if self.closing_price is None:
            return None
        return round((self.closing_price / self.price_at_pick - 1) * 100, 2)

    # ── Trazabilidad ───────────────────────────────────────────────────────────

    def add_reason(self, text: str) -> None:
        """
        Añade una entrada al trail de decisión.

        Punto único de instrumentación: usar este método en vez de
        pick.reasons.append(...) directo permite añadir timestamp o
        logging centralizado en el futuro sin auditar cada llamada
        dispersa por los 12 stages del pipeline.
        """
        self.reasons.append(text)

    def deactivate(self, reason: str) -> None:
        """
        Marca el pick como inactivo con una razón explícita.

        Usado por RiskManager (Stage 9) y por validaciones de stages
        anteriores (ej. line movement contradice la dirección del pick).
        Registra automáticamente la razón también en el trail de
        reasons para trazabilidad completa.
        """
        self.active = False
        self.inactive_reason = reason
        self.add_reason(f"DEACTIVATED: {reason}")

    def activate(self) -> None:
        """
        Marca el pick como activo, aprobado por RiskManager.

        Limpia inactive_reason — un pick activo no debería conservar
        una razón de inactividad obsoleta de una evaluación anterior.
        """
        self.active = True
        self.inactive_reason = None
        self.add_reason("ACTIVATED: aprobado por risk management")