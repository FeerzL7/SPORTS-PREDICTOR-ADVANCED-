from collections.abc import Iterable

from core.models.ledger_entry import LedgerEntry


class Ledger:
    """
    Libro contable central del sistema.

    Responsabilidades:

    - Almacenar apuestas históricas
    - Buscar apuestas
    - Liquidar apuestas
    - Obtener pendientes
    - Obtener cerradas
    - Calcular estadísticas básicas

    NO calcula:

    - ROI
    - Yield
    - Bankroll

    Eso pertenece a módulos superiores.
    """

    def __init__(self):
        self._entries: list[LedgerEntry] = []

    # --------------------------------------------------
    # Escritura
    # --------------------------------------------------

    def add_entry(
        self,
        entry: LedgerEntry,
    ) -> None:

        self._entries.append(entry)

    def add_entries(
        self,
        entries: Iterable[LedgerEntry],
    ) -> None:

        self._entries.extend(entries)

    # --------------------------------------------------
    # Lectura
    # --------------------------------------------------

    def all_entries(self) -> list[LedgerEntry]:

        return list(self._entries)

    def pending_entries(self) -> list[LedgerEntry]:

        return [
            entry
            for entry in self._entries
            if entry.is_pending
        ]

    def settled_entries(self) -> list[LedgerEntry]:

        return [
            entry
            for entry in self._entries
            if not entry.is_pending
        ]

    def winning_entries(self) -> list[LedgerEntry]:

        return [
            entry
            for entry in self._entries
            if entry.is_win
        ]

    def losing_entries(self) -> list[LedgerEntry]:

        return [
            entry
            for entry in self._entries
            if entry.is_loss
        ]

    def push_entries(self) -> list[LedgerEntry]:

        return [
            entry
            for entry in self._entries
            if entry.is_push
        ]

    # --------------------------------------------------
    # Búsqueda
    # --------------------------------------------------

    def find_by_event(
        self,
        event_id: str,
    ) -> list[LedgerEntry]:

        return [
            entry
            for entry in self._entries
            if entry.event_id == event_id
        ]

    def find_first(
        self,
        event_id: str,
    ) -> LedgerEntry | None:

        for entry in self._entries:

            if entry.event_id == event_id:
                return entry

        return None

    # --------------------------------------------------
    # Liquidación
    # --------------------------------------------------

    def settle_entry(
        self,
        event_id: str,
        result: str,
        profit_loss: float,
    ) -> bool:

        entry = self.find_first(event_id)

        if entry is None:
            return False

        entry.settle(
            result=result,
            profit_loss=profit_loss,
        )

        return True

    # --------------------------------------------------
    # Resumen rápido
    # --------------------------------------------------

    def total_entries(self) -> int:

        return len(self._entries)

    def total_pending(self) -> int:

        return len(
            self.pending_entries()
        )

    def total_settled(self) -> int:

        return len(
            self.settled_entries()
        )

    def total_wins(self) -> int:

        return len(
            self.winning_entries()
        )

    def total_losses(self) -> int:

        return len(
            self.losing_entries()
        )

    def total_pushes(self) -> int:

        return len(
            self.push_entries()
        )

    # --------------------------------------------------
    # Profit/Loss histórico
    # --------------------------------------------------

    def total_profit_loss(self) -> float:

        return round(
            sum(
                entry.profit_loss
                for entry in self.settled_entries()
            ),
            2,
        )

    # --------------------------------------------------
    # Utilidades
    # --------------------------------------------------

    def clear(self) -> None:

        self._entries.clear()

    def __len__(self) -> int:

        return len(self._entries)

    def __iter__(self):

        return iter(self._entries)