"""
core/contracts/ledger.py

Registro permanente en el ledger financiero. Una vez liquidado,
inmutable en sus resultados.

BetLedgerEntry es la fuente de verdad del rendimiento del sistema
(ver SPORTS_PREDICTOR_ARCHITECTURE.md, sección 7.1). Cada pick
registrado en producción se convierte en una entrada de este tipo,
y el BankrollTracker (core/bankroll/tracker.py, Bloque 5 del roadmap)
opera exclusivamente sobre listas de BetLedgerEntry para calcular
ROI, yield, hit rate y la equity curve.

Origen del diseño — dos correcciones directas de deuda documentada:

1. Columna 'sport' obligatoria, sin default (ver
   SPORTS_PREDICTOR_ASSESSMENT.md, hallazgo 8.7 y 9.6):
       "Ausencia de columna sport en el ledger... genera deuda
       histórica... requiere migración retroactiva"
   El sistema MLB original no tenía esta columna. Aquí es el
   segundo campo posicional del contrato, sin valor por defecto —
   construir un BetLedgerEntry sin especificar sport es un
   TypeError en construcción, no un dato faltante descubierto
   meses después al filtrar el dashboard por deporte.

2. settle() como única vía para liquidar un pick, con protección
   estructural contra re-liquidación. En el bankroll/tracker.py
   original, profit_amount/bankroll_after/yield_pct podían
   asignarse por separado sin garantía de que permanecieran
   consistentes entre sí — mismo tipo de riesgo que ya se eliminó
   en CandidatePick.ev/edge, aplicado aquí al cálculo financiero
   más crítico del sistema completo.

Uso típico:
    entry = BetLedgerEntry(
        entry_id="a1b2c3d4-...",
        sport="mlb",
        league="MLB",
        date="2026-06-08",
        event="Boston Red Sox @ New York Yankees",
        market="TOTAL",
        selection="over 8.5",
        price=1.91,
        model_prob=0.55,
        ev=5.05,
        stake_pct=2,
        stake_amount=20.0,
        bankroll_before=1000.0,
        model_version="mlb-v1.0",
        created_at="2026-06-08T17:35:00Z",
    )
    # entry.result == 'pending', bankroll_after/profit_amount/yield_pct == None

    entry.settle(result="win", settled_at="2026-06-08T23:50:00Z")
    # profit_amount, bankroll_after, yield_pct calculados automáticamente
    # entry.settle(...) de nuevo -> ValueError: ya está liquidado
"""

from __future__ import annotations

from dataclasses import dataclass


# Estados válidos de result. Conjunto cerrado, mismo patrón que
# EventStatus.ALL y VALID_DISTRIBUTIONS.
#
# 'void' se añade respecto al sistema MLB original (que solo manejaba
# pending/win/lose/null) porque los deportes ya comprometidos en el
# roadmap (NFL Fase 4, Soccer/NHL Fase 3) tienen cancelaciones,
# postergaciones y abandonos con más frecuencia estructural que MLB.
# 'void' = el pick nunca debió contar (evento cancelado/pospuesto).
# 'null' = push/empate en el handicap — resultado deportivo válido
#          donde el stake se devuelve sin ganancia ni pérdida.
# Mezclar ambos casos bajo 'null' contaminaría su significado
# semántico ya bien establecido en el sistema MLB.
VALID_RESULTS: frozenset[str] = frozenset({
    "pending",
    "win",
    "lose",
    "null",
    "void",
})

# Resultados terminales: una vez alcanzado uno de estos, el pick está
# liquidado y settle() no puede invocarse de nuevo sobre la misma entrada.
TERMINAL_RESULTS: frozenset[str] = frozenset({"win", "lose", "null", "void"})


@dataclass
class BetLedgerEntry:
    """
    Registro permanente de una apuesta en el ledger.

    Mutable únicamente en su transición de result='pending' a un
    estado terminal, y exclusivamente a través de settle(). Ningún
    otro campo debe mutarse después de la construcción — a diferencia
    de TeamFeatures o CandidatePick, que se enriquecen progresivamente
    en múltiples puntos del pipeline, BetLedgerEntry se construye una
    vez (al registrar el pick) y se liquida una vez (al resolverse el
    resultado real). No hay un tercer momento de mutación legítimo.

    Campos
    ------
    Identidad:
        entry_id    -- ID único de esta entrada. NO se autogenera en
                      __post_init__ (a diferencia de un UUID interno):
                      se recibe como parámetro para que ROITracker
                      pueda construirlo de forma determinista desde
                      (fecha, evento, mercado, selección) y así
                      preservar la deduplicación que usaba
                      clave_apuesta() en el sistema MLB original.
        sport        -- OBLIGATORIO, sin default. Ver nota de diseño
                      en el docstring del módulo.
        league       -- 'MLB', 'EPL', 'NBA', etc.
        date         -- Fecha del evento, 'YYYY-MM-DD'.

    Mercado:
        event        -- Formato 'Away @ Home', igual a Event.matchup.
        market        -- 'ML', 'TOTAL', '1X2', etc.
        selection      -- Selección apostada, con línea si aplica
                        (ej. 'over 8.5').
        price          -- Cuota decimal usada en el pick.
        model_prob     -- Probabilidad final usada para el cálculo de
                        EV (equivalente a CandidatePick.blended_prob
                        en el momento del registro).
        ev              -- EV en el momento del registro. A diferencia
                        de CandidatePick.ev, aquí SÍ es un campo
                        normal, no una propiedad calculada: el ledger
                        registra un snapshot histórico del EV tal
                        como era al momento del pick, no necesita
                        recalcularse si blended_prob cambiara después
                        (de hecho, CandidatePick ya es inmutable en
                        la práctica para este punto del pipeline).

    Stake:
        stake_pct     -- Porcentaje de stake (entero), fijado por
                        StakingStrategy.
        stake_amount   -- Monto absoluto = bankroll_before * stake_pct/100.

    Resultado y bankroll (ver settle()):
        result          -- Validado contra VALID_RESULTS. Default
                          'pending'.
        bankroll_before  -- Bankroll inmediatamente antes de este pick.
                          Campo de entrada normal, conocido en el
                          momento de construcción.
        bankroll_after    -- None hasta settle(). CALCULADO, no
                          asignable directamente tras la construcción
                          inicial salvo a través de settle().
        profit_amount      -- None hasta settle(). CALCULADO.
        yield_pct           -- None hasta settle(). CALCULADO =
                          profit_amount / stake_amount * 100.

    CLV:
        clv          -- None hasta que se asigne explícitamente
                      (normalmente copiado desde
                      CandidatePick.clv una vez que closing_price
                      esté disponible). No se recalcula aquí porque
                      el ledger no tiene acceso a price_at_pick vs
                      closing_price por separado — ese cálculo
                      pertenece a CandidatePick.

    Trazabilidad temporal:
        created_at    -- ISO-8601 UTC de cuándo se registró el pick.
        settled_at     -- None hasta settle(). ISO-8601 UTC de cuándo
                       se liquidó.

    Versionado del modelo:
        model_version -- Identificador de versión del modelo que
                       generó este pick. Para picks migrados desde el
                       ledger histórico de MLB sin esta información,
                       usar el valor explícito documentado
                       "legacy-unversioned" — NO un default silencioso,
                       sino una categoría reconocida que BacktestEngine
                       puede excluir deliberadamente de comparaciones
                       entre versiones de modelo.

    Propiedades derivadas
    ----------------------
        is_settled      -- True si result != 'pending'.
        is_profitable    -- True si profit_amount existe y es > 0.
    """

    # ── Identidad ──────────────────────────────────────────────────────────────
    entry_id: str
    sport:    str          # Obligatorio — sin default, ver docstring del módulo.
    league:   str
    date:     str

    # ── Mercado ─────────────────────────────────────────────────────────────────
    event:      str
    market:     str
    selection:  str
    price:      float
    model_prob: float
    ev:         float

    # ── Stake ──────────────────────────────────────────────────────────────────
    stake_pct:    int
    stake_amount: float

    # ── Bankroll (bankroll_before es entrada; el resto se llena en settle()) ────
    bankroll_before: float
    result:          str = "pending"
    bankroll_after:  float | None = None
    profit_amount:   float | None = None
    yield_pct:       float | None = None

    # ── CLV ────────────────────────────────────────────────────────────────────
    clv: float | None = None

    # ── Trazabilidad temporal ───────────────────────────────────────────────────
    created_at: str = ""
    settled_at: str | None = None

    # ── Versionado del modelo ───────────────────────────────────────────────────
    model_version: str = "legacy-unversioned"

    # ── Validación e invariantes ──────────────────────────────────────────────

    def __post_init__(self) -> None:
        """Aplica las invariantes estructurales del contrato en construcción."""
        self._validate_result()
        self._validate_consistency_with_result()

    def _validate_result(self) -> None:
        """result debe estar en VALID_RESULTS."""
        if self.result not in VALID_RESULTS:
            raise ValueError(
                f"result='{self.result}' inválido para entry_id="
                f"'{self.entry_id}'. Valores válidos: "
                f"{sorted(VALID_RESULTS)}."
            )

    def _validate_consistency_with_result(self) -> None:
        """
        Si se construye directamente con un resultado terminal (caso de
        migración de ledger histórico, tarea 8.16 del roadmap), los
        campos calculados deben venir poblados de forma consistente —
        no se permite construir una entrada 'win' con bankroll_after=None.
        Para el flujo normal (construcción en 'pending', liquidación
        posterior vía settle()), esta validación no aplica.
        """
        if self.result in TERMINAL_RESULTS:
            faltantes = [
                name for name, value in (
                    ("bankroll_after", self.bankroll_after),
                    ("profit_amount", self.profit_amount),
                    ("yield_pct", self.yield_pct),
                    ("settled_at", self.settled_at),
                )
                if value is None
            ]
            if faltantes:
                raise ValueError(
                    f"entry_id='{self.entry_id}' se construyó con "
                    f"result='{self.result}' (terminal) pero le faltan "
                    f"campos calculados: {faltantes}. Si es un pick "
                    f"nuevo, constrúyelo con result='pending' (default) "
                    f"y liquídalo con settle(). Si es una migración de "
                    f"ledger histórico, provee todos los campos "
                    f"calculados explícitamente."
                )

    # ── Liquidación ─────────────────────────────────────────────────────────────

    def settle(self, result: str, settled_at: str) -> None:
        """
        Liquida el pick: única vía legítima para poblar profit_amount,
        bankroll_after y yield_pct de forma garantizadamente consistente.

        Fórmula (idéntica a calcular_profit() del sistema MLB original):
            win:  profit = stake_amount * (price - 1)
            lose: profit = -stake_amount
            null: profit = 0.0   (push — stake se devuelve)
            void: profit = 0.0   (evento cancelado/pospuesto — stake se devuelve)

        bankroll_after = bankroll_before + profit_amount
        yield_pct = profit_amount / stake_amount * 100

        Protección contra re-liquidación: si result ya es un estado
        terminal (no 'pending'), lanza ValueError inmediatamente. Esto
        hace estructuralmente imposible que un bug de reintento, un
        cron job duplicado o una llamada accidental corrompan el
        bankroll histórico liquidando el mismo pick dos veces.

        Parámetros
        ----------
        result      -- Debe estar en TERMINAL_RESULTS ('win', 'lose',
                      'null', 'void'). Pasar 'pending' aquí no tiene
                      sentido y se rechaza explícitamente.
        settled_at  -- Timestamp ISO-8601 UTC de la liquidación.
        """
        if self.result != "pending":
            raise ValueError(
                f"No se puede liquidar entry_id='{self.entry_id}': ya "
                f"tiene result='{self.result}' (settled_at="
                f"'{self.settled_at}'). Un BetLedgerEntry solo puede "
                f"liquidarse una vez. Si esto es un reintento "
                f"accidental, verificar la lógica de ROITracker antes "
                f"de forzar una segunda liquidación."
            )

        if result not in TERMINAL_RESULTS:
            raise ValueError(
                f"result='{result}' no es un estado terminal válido "
                f"para settle(). Valores válidos: "
                f"{sorted(TERMINAL_RESULTS)}. 'pending' no es un "
                f"resultado de liquidación."
            )

        profit = self._calculate_profit(result)

        self.result = result
        self.profit_amount = round(profit, 2)
        self.bankroll_after = round(self.bankroll_before + profit, 2)
        self.yield_pct = (
            round(profit / self.stake_amount * 100, 2)
            if self.stake_amount else 0.0
        )
        self.settled_at = settled_at

    def _calculate_profit(self, result: str) -> float:
        """
        Fórmula de profit, idéntica a calcular_profit() del sistema
        MLB original (bankroll/tracker.py). Aislada en un método
        privado para que settle() permanezca legible y para que
        scripts/migrate_ledger.py (tarea 8.16) pueda reutilizarla
        sin duplicar la lógica.
        """
        if result == "win":
            return self.stake_amount * (self.price - 1)
        if result == "lose":
            return -self.stake_amount
        # 'null' (push) y 'void' (evento cancelado) devuelven el stake
        # sin ganancia ni pérdida.
        return 0.0

    # ── Propiedades derivadas ──────────────────────────────────────────────────

    @property
    def is_settled(self) -> bool:
        """True si el pick ya fue liquidado (result distinto de 'pending')."""
        return self.result != "pending"

    @property
    def is_profitable(self) -> bool:
        """True si el pick está liquidado y generó ganancia positiva."""
        return self.profit_amount is not None and self.profit_amount > 0