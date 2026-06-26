"""
core/contracts/market_odds.py

Cuota de mercado normalizada para una selección específica.

MarketOdds representa un hecho del mercado en un instante dado: esta
cuota, en este bookmaker, a esta hora, para esta selección de este
mercado de este evento. Es la puerta de entrada de la información
que determina la rentabilidad del sistema (ver MLB_EDGE_AUDIT.md,
sección "¿Qué módulo aporta más valor?").

Diseño deliberadamente NO binario: a diferencia de un diseño que
asumiera "cuota A vs cuota B", MarketOdds representa UNA selección
de UN mercado. Un mercado puede tener 2 selecciones (ML: home/away),
3 (1X2: home/draw/away) o N (outright de golf: 150+ jugadores). El
cálculo de no-vig probability —que requiere conocer TODAS las
selecciones del mismo mercado— se delega deliberadamente a
core/odds/no_vig.py, que opera sobre list[MarketOdds], no aquí.

Esta separación es la razón por la que no_vig_prob NUNCA se calcula
en __post_init__: MarketOdds no tiene visibilidad de sus selecciones
hermanas, y forzar ese cálculo aquí asumiría implícitamente que todo
mercado es binario — una asunción que se rompe en Fase 3 (Soccer 1X2)
y Fase 5 (Golf outright) del roadmap.

Uso típico:
    over = MarketOdds(
        event_id  = "a1b2c3d4-...",
        market    = "TOTAL",
        selection = "over",
        line      = 8.5,
        price     = 1.91,
        bookmaker = "Pinnacle",
        timestamp = "2026-06-08T17:30:00Z",
    )
    # over.implied_prob ya está calculado: round(1/1.91, 4) = 0.5236

    under = MarketOdds(
        event_id="a1b2c3d4-...", market="TOTAL", selection="under",
        line=8.5, price=1.95, bookmaker="Pinnacle",
        timestamp="2026-06-08T17:30:00Z",
    )

    # no_vig_prob se calcula EXTERNAMENTE, sobre el par:
    #   from core.odds.no_vig import no_vig_probabilities
    #   probs = no_vig_probabilities([over, under])
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Identificadores de mercado conocidos por convención, usados únicamente
# por la propiedad heurística requires_line. NO es un conjunto cerrado
# de validación (a diferencia de VALID_DISTRIBUTIONS en projection.py):
# cada sport plugin introduce sus propios nombres de mercado libremente
# ('1X2' para soccer, 'AH' para handicap asiático, 'Puck Line' para NHL)
# y el Core nunca hace lookup directo contra esta lista — solo la usa
# como ayuda de depuración.
MARKETS_WITHOUT_LINE: frozenset[str] = frozenset({
    "ML", "MONEYLINE", "1X2", "WINNER", "OUTRIGHT",
})


@dataclass(frozen=True)
class MarketOdds:
    """
    Cuota normalizada para una selección de un mercado específico.
    Inmutable: representa un hecho del mercado en un instante dado.

    A diferencia de TeamFeatures y Projection (mutables, construidos
    progresivamente), MarketOdds se obtiene tal cual de la API y nunca
    se ajusta después. Si la cuota cambia, se crea un MarketOdds nuevo
    con timestamp distinto — el mismo patrón que ya usa
    data/line_movement.py en el sistema MLB para detectar movimiento
    de línea comparando snapshots, aquí hecho estructural.

    Campos
    ------
    Identidad:
        event_id   -- Igual a Event.event_id. Vincula la cuota con su
                      evento de origen.
        market      -- Identificador de mercado, libre por deporte.
                      MLB: 'ML', 'RL', 'TOTAL'. Soccer: '1X2', 'AH',
                      'BTTS'. NHL: 'ML', 'Puck Line', 'TOTAL'. Tennis:
                      'ML', 'Sets O/U'. El Core nunca interpreta este
                      string directamente — lo pasa a MarketDefinitions
                      del sport plugin correspondiente.
        selection   -- Selección dentro del mercado. 'home', 'away',
                      'draw', 'over', 'under', o el nombre de un
                      jugador/equipo en mercados outright.

    Línea y precio:
        line        -- Línea del spread o total. None para mercados sin
                      línea (ML, 1X2). ÚNICO campo genuinamente opcional
                      del contrato: no existe un valor "neutral" sensato
                      para una línea que estructuralmente no aplica —
                      a diferencia de venue_factor=1.0 en TeamFeatures,
                      forzar un default numérico aquí sería engañoso.
        price       -- Cuota decimal. VALIDADO en __post_init__: debe
                      ser > 1.0. Una cuota ≤ 1.0 es matemáticamente
                      inválida (implica probabilidad implícita ≥100%)
                      y casi siempre indica un error de parseo de la
                      API upstream — se rechaza en el punto de entrada,
                      no tres stages después cuando ya contaminó un EV.

    Procedencia:
        bookmaker   -- Nombre de la casa de apuestas.
        timestamp   -- ISO-8601 UTC de cuándo se observó esta cuota.

    Probabilidades derivadas:
        implied_prob -- CALCULADO AUTOMÁTICAMENTE en __post_init__
                      como 1/price. Es una función pura de un único
                      campo del propio objeto — se calcula aquí porque
                      no depende de ninguna otra selección.
        no_vig_prob  -- NO se calcula aquí. Requiere conocer TODAS las
                      cuotas del mismo mercado (la contraparte en un
                      mercado binario, o las N selecciones en 1X2/
                      outright). Lo puebla externamente
                      core/odds/no_vig.py, operando sobre
                      list[MarketOdds]. Permanece None hasta que ese
                      paso se ejecute explícitamente.

    Propiedades derivadas
    ----------------------
        requires_line -- Heurística (no validación dura) basada en
                        MARKETS_WITHOUT_LINE: útil para detectar a
                        simple vista un MarketOdds sospechoso (ej.
                        market='TOTAL' con line=None). No exhaustiva
                        para todo mercado futuro de los 7 deportes
                        objetivo.
    """

    # ── Identidad ──────────────────────────────────────────────────────────────
    event_id:  str
    market:    str
    selection: str

    # ── Línea y precio ─────────────────────────────────────────────────────────
    line:  float | None
    price: float

    # ── Procedencia ────────────────────────────────────────────────────────────
    bookmaker: str
    timestamp: str

    # ── Probabilidades derivadas ───────────────────────────────────────────────
    # implied_prob se calcula SIEMPRE en __post_init__.
    # no_vig_prob NUNCA se calcula aquí — ver docstring de la clase.
    implied_prob: float | None = field(default=None)
    no_vig_prob:  float | None = field(default=None)

    # ── Validación e invariantes ──────────────────────────────────────────────

    def __post_init__(self) -> None:
        """
        Aplica las invariantes estructurales del contrato.

        frozen=True impide reasignación normal de atributos, por lo que
        la asignación de implied_prob usa object.__setattr__ —patrón
        estándar de dataclasses congeladas para poblar campos derivados
        durante __post_init__.
        """
        self._validate_price()
        self._compute_implied_prob()

    def _validate_price(self) -> None:
        """
        price debe ser > 1.0.

        Una cuota decimal ≤ 1.0 es matemáticamente inválida y casi
        siempre resulta de un error de parseo upstream (campo null
        convertido a 0, o un endpoint que devuelve cuota americana
        sin convertir). Rechazar aquí evita que un EV calculado sobre
        una cuota corrupta llegue a registrarse como pick real — el
        riesgo que MLB_EDGE_AUDIT.md identifica como crítico en la
        "puerta de entrada" de datos de mercado.
        """
        if self.price <= 1.0:
            raise ValueError(
                f"price={self.price} inválido para event_id="
                f"'{self.event_id}', market='{self.market}', "
                f"selection='{self.selection}'. Una cuota decimal debe "
                f"ser > 1.0. Esto casi siempre indica un error de "
                f"parseo en la fuente de datos upstream — revisar el "
                f"cliente de odds antes de descartar este evento."
            )

    def _compute_implied_prob(self) -> None:
        """
        implied_prob = 1 / price, redondeado a 4 decimales.

        Es la única probabilidad que MarketOdds puede calcular por sí
        solo: depende exclusivamente de su propio campo price, sin
        necesidad de conocer otras selecciones del mismo mercado.
        """
        object.__setattr__(self, "implied_prob", round(1 / self.price, 4))

    # ── Propiedades derivadas ──────────────────────────────────────────────────

    @property
    def requires_line(self) -> bool:
        """
        Heurística: True si este tipo de mercado normalmente requiere
        una línea (spread, total, handicap), False para mercados de
        resultado directo (moneyline, 1X2, outright).

        NO es validación dura — MARKETS_WITHOUT_LINE no es exhaustivo
        para los mercados que introducirán deportes futuros. Útil
        únicamente como ayuda de depuración para detectar a simple
        vista una combinación sospechosa (market='TOTAL' con line=None).
        """
        return self.market.upper() not in MARKETS_WITHOUT_LINE