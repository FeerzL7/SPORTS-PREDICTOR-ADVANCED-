"""
core/tracking/protocols.py

Protocolos del subsistema de tracking y settlement.

Posición en el pipeline
------------------------
    Stage 10 — Settlement:

    BetLedgerEntry (pending)
        ↓
    SettlementProvider.get_result(entry) → SettlementResult
        ↓
    BankrollTracker.settle(entry_id, result, settled_at)
    BankrollTracker.update_clv(entry_id, clv)

Separación de responsabilidades
---------------------------------
SettlementProvider  — sabe qué resultado tuvo un pick.
                      Es un concepto deportivo: necesita conocer
                      el score final del partido, la línea apostada
                      y las reglas del mercado. Lo implementa el
                      sport plugin, no el core.

BankrollTracker     — sabe cómo liquidar financieramente un pick
                      dado su resultado. No sabe qué deporte es.

Esta separación es la corrección central respecto al sistema MLB,
donde tracking/roi_tracker.py mezclaba ambas responsabilidades:
_resolver_resultado() interpretaba ML/RL/TOTAL con lógica baseball-
específica (runline de 1.5, carreras) dentro del mismo módulo que
actualizaba el bankroll.

CLV (Closing Line Value)
--------------------------
CLV = (model_price - closing_price) / closing_price × 100

Donde model_price es la cuota al momento del pick y closing_price
es la cuota de cierre del mercado (justo antes del evento).

CLV > 0: el pick fue apostado a mejor precio que el cierre → edge real.
CLV < 0: el mercado se movió en contra → señal de pick con escaso valor.

SettlementResult incluye closing_price como campo opcional — si el
sport plugin puede obtenerlo (requiere snapshot de cierre de mercado),
BankrollTracker.update_clv() se llama con el CLV calculado.
Si closing_price=None, CLV queda None en el ledger.

Resultados válidos
-------------------
Los resultados posibles están definidos en core/contracts/ledger.py:

    TERMINAL_RESULTS = {'win', 'lose', 'null', 'void'}

    win  — El pick ganó. profit = stake × (price - 1).
    lose — El pick perdió. profit = -stake.
    null — Push/empate. profit = 0, stake devuelto.
    void — Anulado (evento cancelado, etc.). profit = 0.

'pending' no es un resultado de settlement — es el estado inicial
antes de que el SettlementProvider procese el pick.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.contracts.ledger import BetLedgerEntry, TERMINAL_RESULTS


# ── Resultado de settlement ───────────────────────────────────────────────────

@dataclass(frozen=True)
class SettlementResult:
    """
    Resultado inmutable producido por SettlementProvider para un pick.

    Inmutable: representa un hecho deportivo — una vez determinado
    el resultado del partido y la cuota de cierre, no cambian.

    Campos
    ------
    entry_id       -- ID del BetLedgerEntry a liquidar. Garantiza que
                     el resultado se aplica al pick correcto.
    result         -- Resultado del pick. Debe estar en TERMINAL_RESULTS
                     ('win', 'lose', 'null', 'void'). El SettlementProvider
                     es responsable de interpretar el score del partido
                     y determinar si el pick ganó o perdió.
    settled_at     -- Timestamp ISO-8601 UTC del momento de liquidación.
                     Generalmente el momento en que el score final del
                     partido fue confirmado por la fuente de datos.
    closing_price  -- Cuota del mercado justo antes del inicio del
                     evento (precio de cierre). Necesaria para calcular
                     CLV. None si el sport plugin no puede obtenerla
                     (ej. datos históricos sin snapshot de cierre).
    sport_context  -- Información adicional del partido para trazabilidad
                     en logs y backtesting. Formato libre — el sport
                     plugin incluye lo que considera relevante.
                     Ejemplos:
                         MLB: {'home_score': 5, 'away_score': 3,
                               'innings': 9}
                         NBA: {'home_score': 112, 'away_score': 108,
                               'overtime': False}
                         Soccer: {'home_score': 2, 'away_score': 1,
                                  'home_ht': 1, 'away_ht': 0}
    """
    entry_id:      str
    result:        str
    settled_at:    str
    closing_price: float | None        = None
    sport_context: dict                = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.result not in TERMINAL_RESULTS:
            raise ValueError(
                f"SettlementResult.result='{self.result}' no es un resultado "
                f"terminal válido. Valores permitidos: "
                f"{sorted(TERMINAL_RESULTS)}. "
                f"'pending' no es un resultado de settlement."
            )
        if self.closing_price is not None and self.closing_price <= 1.0:
            raise ValueError(
                f"closing_price={self.closing_price} debe ser > 1.0 "
                f"(cuota decimal). Si no hay precio de cierre disponible, "
                f"usar closing_price=None."
            )
        # Inicializar sport_context a dict vacío si es None
        if self.sport_context is None:
            object.__setattr__(self, 'sport_context', {})

    def clv(self, pick_price: float) -> float | None:
        """
        Calcula el CLV para este settlement dado el precio del pick.

        CLV = (pick_price - closing_price) / closing_price × 100

        Retorna None si closing_price no está disponible.

        Parámetros
        ----------
        pick_price  -- Cuota al momento del pick (BetLedgerEntry.price).

        Ejemplo:
            Pick apostado a 2.10, cierre del mercado en 1.95.
            CLV = (2.10 - 1.95) / 1.95 × 100 = +7.69%
            → El modelo encontró valor real: el mercado confirmó el edge.
        """
        if self.closing_price is None:
            return None
        return round(
            (pick_price - self.closing_price) / self.closing_price * 100,
            4,
        )


# ── Protocolo ─────────────────────────────────────────────────────────────────

@runtime_checkable
class SettlementProvider(Protocol):
    """
    Protocolo para proveedores de liquidación de picks.

    Implementado por cada sport plugin — no por el core. El core
    no sabe qué es un runline, un handicap de -1.5, ni cómo se
    determina si un partido de tennis en best-of-5 cubrió el spread.

    Cada deporte implementa su propia lógica de resolución:

    MLB plugin:
        Consulta la MLB Stats API para el score final.
        Interpreta ML/RL/TOTAL con las reglas de baseball
        (runline ±1.5, extras, etc.).

    NBA plugin:
        Consulta la NBA API para el score final.
        Interpreta ML/SPREAD/TOTAL con las reglas de basketball
        (incluyendo overtime si aplica).

    Soccer plugin:
        Consulta la football-data API o similar.
        Interpreta 1X2/TOTAL/BTTS con tiempo reglamentario
        (90 minutos, sin contar prórroga salvo indicación).

    Garantías del protocolo
    ------------------------
    1. get_result() nunca retorna 'pending' — solo resultados terminales.
    2. Si el score no está disponible aún, retorna None (no None-como-pending).
    3. entry_id en el SettlementResult debe coincidir con el del input.

    El pipeline llama get_result() en un loop sobre todos los
    BetLedgerEntry pendientes. Si retorna None, el pick sigue pending
    y se reintenta en la siguiente ejecución.
    """

    def get_result(
        self,
        entry: BetLedgerEntry,
    ) -> SettlementResult | None:
        """
        Determina el resultado de un pick pendiente.

        Parámetros
        ----------
        entry  -- BetLedgerEntry con result='pending'. El provider
                 usa entry.event (nombre del partido), entry.market,
                 entry.selection y entry.date para identificar el
                 partido y calcular el resultado.

        Retorna
        -------
        SettlementResult si el resultado ya está disponible.
        None si el partido no ha terminado o el score no está
        disponible aún — el pick permanece 'pending'.

        No lanza excepciones por datos faltantes — las condiciones
        esperadas (partido no terminado, API no disponible) se
        expresan como None, no como errores.
        """
        ...

    def get_closing_price(
        self,
        entry: BetLedgerEntry,
    ) -> float | None:
        """
        Obtiene el precio de cierre del mercado para calcular CLV.

        Puede ser None si:
        - El sport plugin no tiene acceso a snapshots de cierre
        - El mercado no tenía precio de cierre disponible
        - La implementación no soporta CLV tracking

        Este método es opcional en la práctica — si un sport plugin
        no lo implementa con datos reales, CLV queda None en el
        ledger, lo cual es preferible a un CLV incorrecto.
        """
        ...


# ── Función utilitaria ────────────────────────────────────────────────────────

def apply_settlement(
    entry:    BetLedgerEntry,
    result:   SettlementResult,
    tracker,  # BankrollTracker — sin import circular
) -> None:
    """
    Aplica un SettlementResult al ledger via BankrollTracker.

    Función utilitaria del pipeline de settlement — encapsula la
    secuencia: settle() + update_clv() en una sola llamada.

    Parámetros
    ----------
    entry    -- BetLedgerEntry pendiente a liquidar.
    result   -- SettlementResult del sport plugin.
    tracker  -- BankrollTracker instancia (sin tipo explícito para
               evitar import circular entre tracking y bankroll).

    El CLV se calcula y registra automáticamente si closing_price
    está disponible en el SettlementResult.
    """
    tracker.settle(
        entry_id   = result.entry_id,
        result     = result.result,
        settled_at = result.settled_at,
    )

    clv = result.clv(pick_price=entry.price)
    if clv is not None:
        tracker.update_clv(
            entry_id = result.entry_id,
            clv      = clv,
        )