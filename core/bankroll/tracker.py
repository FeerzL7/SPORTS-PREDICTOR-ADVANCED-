"""
core/bankroll/tracker.py

BankrollTracker: ledger financiero multi-deporte con soporte para
CLV, model_version y métricas filtrables por sport/market/fecha.

Migrado de bankroll/tracker.py del sistema MLB con cuatro correcciones
documentadas en SPORTS_PREDICTOR_ARCHITECTURE.md §7.1:

1. Columna `sport` obligatoria en BetLedgerEntry.
   El sistema MLB mezclaba todos los deportes en un único ledger sin
   esta columna — imposible calcular ROI por deporte o filtrar el
   dashboard por MLB vs NBA. Aquí es el segundo campo del contrato,
   sin default, forzando su provisión en construcción.

2. Columna `clv` (Closing Line Value) como métrica de primera clase.
   Ausente en el sistema MLB. CLV positivo en 20-30 picks es señal
   estadísticamente más fuerte de edge que ROI en 100 picks — el
   instrumento de diagnóstico más rápido disponible para detectar
   si el modelo tiene ventaja real sobre el mercado.

3. `metrics()` filtrable por sport, market, model_version, fechas.
   El sistema MLB solo tenía métricas globales. BankrollTracker ahora
   expone metrics(sport='mlb') o metrics(sport='nba', market='TOTAL')
   para que el dashboard y el backtester puedan desglosar rendimiento.

4. LedgerStore como Protocol — CSV es un detalle de infraestructura.
   Mismo principio que SnapshotStore en line_movement.py. CsvLedgerStore
   es compatible con el sistema MLB existente. Tests usan
   InMemoryLedgerStore. Migración futura a SQLite no toca BankrollTracker.

Separación de responsabilidades
---------------------------------
BankrollTracker hace:
    - Persistencia y lectura del ledger (via LedgerStore)
    - Registro y liquidación de picks
    - Cálculo de métricas financieras con filtros

NO hace:
    - Settlement deportivo (sport plugin / SettlementProvider)
    - Decisión de stake (StakingStrategy)
    - Cálculo de CLV (CandidatePick — se registra aquí ya calculado)
    - Generación de picks (ValueEngine)
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import groupby
from typing import Protocol, runtime_checkable

from core.contracts.ledger import BetLedgerEntry, TERMINAL_RESULTS


# ── Protocolo de persistencia ─────────────────────────────────────────────────

@runtime_checkable
class LedgerStore(Protocol):
    """
    Interfaz de persistencia del ledger financiero.

    Permite sustituir CsvLedgerStore por SQLite, Redis u otro backend
    sin modificar BankrollTracker. Tests usan InMemoryLedgerStore.
    """

    def save(self, entry: BetLedgerEntry) -> bool:
        """
        Persiste una nueva entrada.
        Retorna True si se insertó, False si entry_id ya existía.
        """
        ...

    def load_all(self) -> list[BetLedgerEntry]:
        """Retorna todas las entradas ordenadas por created_at."""
        ...

    def load_by_id(self, entry_id: str) -> BetLedgerEntry | None:
        """Retorna una entrada por su entry_id, o None si no existe."""
        ...

    def update(self, entry: BetLedgerEntry) -> None:
        """
        Sobreescribe la entrada con el mismo entry_id.
        Usado por settle() y update_clv() tras modificar el objeto.
        """
        ...


# ── Implementación CSV ────────────────────────────────────────────────────────

_CSV_FIELDS = [
    "entry_id", "sport", "league", "date", "event", "market",
    "selection", "price", "model_prob", "ev", "stake_pct",
    "stake_amount", "bankroll_before", "result", "bankroll_after",
    "profit_amount", "yield_pct", "clv", "created_at", "settled_at",
    "model_version",
]


def _float_or_none(val: str) -> float | None:
    if val in ("", "None"):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def _str_or_none(val: str) -> str | None:
    return None if val in ("", "None") else val


def _row_to_entry(row: dict) -> BetLedgerEntry | None:
    """Convierte una fila CSV a BetLedgerEntry. Retorna None si falla."""
    try:
        result = row.get("result", "pending")

        # Campos calculados — solo presentes si está liquidado
        bankroll_after = _float_or_none(row.get("bankroll_after", ""))
        profit_amount  = _float_or_none(row.get("profit_amount", ""))
        yield_pct      = _float_or_none(row.get("yield_pct", ""))
        settled_at     = _str_or_none(row.get("settled_at", ""))

        entry = BetLedgerEntry(
            entry_id       = row["entry_id"],
            sport          = row["sport"],
            league         = row.get("league", ""),
            date           = row["date"],
            event          = row["event"],
            market         = row["market"],
            selection      = row["selection"],
            price          = float(row["price"]),
            model_prob     = float(row["model_prob"]),
            ev             = float(row["ev"]),
            stake_pct      = int(row["stake_pct"]),
            stake_amount   = float(row["stake_amount"]),
            bankroll_before= float(row["bankroll_before"]),
            result         = result,
            bankroll_after = bankroll_after,
            profit_amount  = profit_amount,
            yield_pct      = yield_pct,
            clv            = _float_or_none(row.get("clv", "")),
            created_at     = row.get("created_at", ""),
            settled_at     = settled_at,
            model_version  = row.get("model_version", "legacy-unversioned"),
        )
        return entry
    except (KeyError, ValueError, TypeError):
        return None


class CsvLedgerStore:
    """
    Implementación de LedgerStore que persiste en CSV.

    Compatible con el formato del sistema MLB (output/roi_tracking.csv).
    El encoding utf-8-sig garantiza compatibilidad con Excel en Windows.

    Parámetros
    ----------
    path  -- Ruta al archivo CSV. Default: 'output/roi_tracking.csv'.
    """

    def __init__(self, path: str = "output/roi_tracking.csv") -> None:
        self._path = path
        self._ensure_file()

    def _ensure_file(self) -> None:
        os.makedirs(os.path.dirname(self._path) if os.path.dirname(self._path) else ".", exist_ok=True)
        if not os.path.exists(self._path):
            self._write_all([])

    def _write_all(self, entries: list[BetLedgerEntry]) -> None:
        with open(self._path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
            writer.writeheader()
            for e in entries:
                writer.writerow(self._to_row(e))

    @staticmethod
    def _to_row(entry: BetLedgerEntry) -> dict:
        return {
            "entry_id":       entry.entry_id,
            "sport":          entry.sport,
            "league":         entry.league,
            "date":           entry.date,
            "event":          entry.event,
            "market":         entry.market,
            "selection":      entry.selection,
            "price":          entry.price,
            "model_prob":     entry.model_prob,
            "ev":             entry.ev,
            "stake_pct":      entry.stake_pct,
            "stake_amount":   entry.stake_amount,
            "bankroll_before":entry.bankroll_before,
            "result":         entry.result,
            "bankroll_after": entry.bankroll_after if entry.bankroll_after is not None else "",
            "profit_amount":  entry.profit_amount  if entry.profit_amount  is not None else "",
            "yield_pct":      entry.yield_pct      if entry.yield_pct      is not None else "",
            "clv":            entry.clv            if entry.clv            is not None else "",
            "created_at":     entry.created_at,
            "settled_at":     entry.settled_at     if entry.settled_at     is not None else "",
            "model_version":  entry.model_version,
        }

    def save(self, entry: BetLedgerEntry) -> bool:
        entries = self.load_all()
        existing_ids = {e.entry_id for e in entries}
        if entry.entry_id in existing_ids:
            return False
        entries.append(entry)
        self._write_all(entries)
        return True

    def load_all(self) -> list[BetLedgerEntry]:
        if not os.path.exists(self._path):
            return []
        with open(self._path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            entries = []
            for row in reader:
                e = _row_to_entry(row)
                if e:
                    entries.append(e)
        return sorted(entries, key=lambda e: e.created_at)

    def load_by_id(self, entry_id: str) -> BetLedgerEntry | None:
        for e in self.load_all():
            if e.entry_id == entry_id:
                return e
        return None

    def update(self, entry: BetLedgerEntry) -> None:
        entries = self.load_all()
        updated = [entry if e.entry_id == entry.entry_id else e for e in entries]
        self._write_all(updated)


class InMemoryLedgerStore:
    """
    Implementación en memoria para tests y entornos sin disco.

    No persiste entre instancias. Mantiene el orden de inserción.
    """

    def __init__(self) -> None:
        self._entries: dict[str, BetLedgerEntry] = {}
        self._order:   list[str] = []

    def save(self, entry: BetLedgerEntry) -> bool:
        if entry.entry_id in self._entries:
            return False
        self._entries[entry.entry_id] = entry
        self._order.append(entry.entry_id)
        return True

    def load_all(self) -> list[BetLedgerEntry]:
        return [self._entries[eid] for eid in self._order if eid in self._entries]

    def load_by_id(self, entry_id: str) -> BetLedgerEntry | None:
        return self._entries.get(entry_id)

    def update(self, entry: BetLedgerEntry) -> None:
        if entry.entry_id in self._entries:
            self._entries[entry.entry_id] = entry

    def clear(self) -> None:
        self._entries.clear()
        self._order.clear()


# ── Métricas financieras ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class BankrollMetrics:
    """
    Snapshot inmutable de métricas financieras del ledger.

    Calculado por BankrollTracker.metrics() con filtros opcionales
    por sport, market, model_version y rango de fechas.

    Campos estándar (migrados de MLB)
    -----------------------------------
    picks_total     -- Total de picks registrados (incluyendo pendientes).
    picks_resolved  -- Picks resueltos (win + lose + null + void).
    wins            -- Picks ganados.
    losses          -- Picks perdidos.
    pushes          -- Picks en push/empate (null).
    voids           -- Picks anulados (void).
    pending         -- Picks aún pendientes.
    hit_rate        -- wins / (wins + losses) × 100.
    stake_total     -- Suma de stake_amount de picks resueltos.
    profit_total    -- Suma de profit_amount de picks resueltos.
    roi             -- profit_total / stake_total × 100.
    yield_pct       -- Idéntico a roi (preservado de MLB).
    bankroll_current -- Bankroll actual (último bankroll_after resuelto).
    max_drawdown_pct -- Máximo drawdown observado en la equity curve.

    Campos nuevos (ausentes en MLB)
    ---------------------------------
    clv_mean        -- Media de CLV para picks con clv no-None.
                      CLV > 0 = evidencia de edge más rápida que ROI.
                      None si no hay picks con CLV calculado.
    brier_score     -- Mean((model_prob - outcome_binary)²) para
                      picks resueltos (win=1, lose=0, null/void omitidos).
                      Mide calibración del modelo. None si no hay datos.
    edge_real       -- hit_rate/100 - (1/avg_price) para picks resueltos.
                      Compara el hit rate real vs el esperado por la cuota.
                      Positivo = el modelo bate al mercado.
    avg_price       -- Cuota media de picks resueltos.
    avg_ev          -- EV medio de picks registrados (modelo en el momento
                      del pick).
    filters_applied -- Descripción de los filtros usados para calcular
                      estas métricas (para trazabilidad en logs).
    """
    picks_total:      int
    picks_resolved:   int
    wins:             int
    losses:           int
    pushes:           int
    voids:            int
    pending:          int
    hit_rate:         float
    stake_total:      float
    profit_total:     float
    roi:              float
    yield_pct:        float
    bankroll_current: float
    max_drawdown_pct: float

    # Nuevas métricas
    clv_mean:         float | None
    brier_score:      float | None
    edge_real:        float | None
    avg_price:        float
    avg_ev:           float
    filters_applied:  str


# ── Motor principal ───────────────────────────────────────────────────────────

class BankrollTracker:
    """
    Ledger financiero multi-deporte.

    Fuente de verdad del rendimiento del sistema. Opera sobre
    BetLedgerEntry — no tiene conceptos deportivos propios.

    Parámetros
    ----------
    store             -- LedgerStore de persistencia. Default:
                        CsvLedgerStore() compatible con MLB.
    initial_bankroll  -- Bankroll inicial. Solo se usa como base
                        para el primer pick si el ledger está vacío.
    """

    def __init__(
        self,
        store: LedgerStore | None = None,
        initial_bankroll: float = 1000.0,
    ) -> None:
        self._store = store or CsvLedgerStore()
        self._initial_bankroll = initial_bankroll

    # ── Registro ───────────────────────────────────────────────────────────────

    def register(self, entry: BetLedgerEntry) -> bool:
        """
        Registra un nuevo pick en el ledger.

        Retorna True si se insertó, False si entry_id ya existía
        (deduplicación sin excepción — permite reintentos seguros).
        """
        return self._store.save(entry)

    # ── Liquidación ────────────────────────────────────────────────────────────

    def settle(
        self,
        entry_id: str,
        result: str,
        settled_at: str,
    ) -> BetLedgerEntry:
        """
        Liquida un pick pendiente.

        Delega en BetLedgerEntry.settle() — la lógica de profit y
        protección contra re-liquidación está en el contrato.

        Raises
        ------
        KeyError   -- Si entry_id no existe en el ledger.
        ValueError -- Si el pick ya está liquidado o result es inválido.
        """
        entry = self._store.load_by_id(entry_id)
        if entry is None:
            raise KeyError(
                f"entry_id='{entry_id}' no encontrado en el ledger. "
                f"Verificar que register() fue llamado antes de settle()."
            )
        # BetLedgerEntry.settle() lanza ValueError si ya está liquidado
        entry.settle(result=result, settled_at=settled_at)
        self._store.update(entry)
        return entry

    # ── CLV ────────────────────────────────────────────────────────────────────

    def update_clv(self, entry_id: str, clv: float) -> None:
        """
        Actualiza el CLV de un pick existente.

        CLV se calcula cuando closing_price está disponible (cierre del
        mercado), independientemente de si el pick ya fue liquidado.
        No requiere re-liquidar el pick.

        Raises
        ------
        KeyError -- Si entry_id no existe en el ledger.
        """
        entry = self._store.load_by_id(entry_id)
        if entry is None:
            raise KeyError(
                f"entry_id='{entry_id}' no encontrado. "
                f"No se puede actualizar CLV."
            )
        entry.clv = round(clv, 4)
        self._store.update(entry)

    # ── Bankroll actual ────────────────────────────────────────────────────────

    def current_bankroll(self) -> float:
        """
        Retorna el bankroll actual.

        Toma el bankroll_after del último pick resuelto en orden
        cronológico. Si no hay picks resueltos, retorna el bankroll
        inicial (el ledger está "vacío" financieramente).
        """
        entries = self._store.load_all()
        resolved = [
            e for e in entries
            if e.result in TERMINAL_RESULTS and e.bankroll_after is not None
        ]
        if not resolved:
            return self._initial_bankroll
        return resolved[-1].bankroll_after  # type: ignore[return-value]

    # ── Métricas ───────────────────────────────────────────────────────────────

    def metrics(
        self,
        sport:         str | None = None,
        market:        str | None = None,
        date_from:     str | None = None,
        date_to:       str | None = None,
        model_version: str | None = None,
    ) -> BankrollMetrics:
        """
        Calcula métricas financieras con filtros opcionales.

        Todos los filtros son acumulativos (AND). Pasar sport='mlb' y
        market='TOTAL' calcula métricas solo para totales de MLB.

        Parámetros
        ----------
        sport          -- Filtrar por deporte ('mlb', 'nba', etc.).
        market          -- Filtrar por mercado ('ML', 'TOTAL', 'SPREAD').
        date_from       -- Filtrar desde fecha 'YYYY-MM-DD' (incluida).
        date_to         -- Filtrar hasta fecha 'YYYY-MM-DD' (incluida).
        model_version   -- Filtrar por versión del modelo.
        """
        all_entries = self._store.load_all()
        filtered = self._apply_filters(
            all_entries, sport, market, date_from, date_to, model_version
        )

        # Describir filtros aplicados para trazabilidad
        filter_parts = []
        if sport:         filter_parts.append(f"sport={sport}")
        if market:        filter_parts.append(f"market={market}")
        if date_from:     filter_parts.append(f"from={date_from}")
        if date_to:       filter_parts.append(f"to={date_to}")
        if model_version: filter_parts.append(f"model={model_version}")
        filters_str = ",".join(filter_parts) if filter_parts else "none"

        resolved = [e for e in filtered if e.result in TERMINAL_RESULTS]
        wins     = [e for e in resolved if e.result == "win"]
        losses   = [e for e in resolved if e.result == "lose"]
        pushes   = [e for e in resolved if e.result == "null"]
        voids    = [e for e in resolved if e.result == "void"]
        pending  = [e for e in filtered if e.result == "pending"]

        win_lose = wins + losses
        hit_rate = len(wins) / len(win_lose) * 100 if win_lose else 0.0

        stake_total  = sum(e.stake_amount for e in win_lose)
        profit_total = sum(e.profit_amount for e in win_lose if e.profit_amount is not None)
        roi = profit_total / stake_total * 100 if stake_total else 0.0

        prices     = [e.price for e in resolved if e.price > 1.0]
        avg_price  = sum(prices) / len(prices) if prices else 0.0
        evs        = [e.ev for e in filtered]
        avg_ev     = sum(evs) / len(evs) if evs else 0.0

        # CLV medio (solo picks con clv calculado)
        clv_values = [e.clv for e in resolved if e.clv is not None]
        clv_mean   = round(sum(clv_values) / len(clv_values), 4) if clv_values else None

        # Brier Score: mean((model_prob - outcome)²) para win/lose
        brier_items = [
            (e.model_prob - (1.0 if e.result == "win" else 0.0)) ** 2
            for e in win_lose
        ]
        brier_score = round(sum(brier_items) / len(brier_items), 6) if brier_items else None

        # Edge real: hit_rate - (1/avg_price)
        edge_real = None
        if win_lose and avg_price > 1.0:
            edge_real = round(hit_rate / 100 - 1.0 / avg_price, 4)

        # Equity curve para calcular bankroll actual y drawdown
        curve     = self._equity_curve_entries(filtered)
        bankroll  = curve[-1]["bankroll_after"] if curve else self._initial_bankroll
        drawdown  = self._max_drawdown(curve)

        return BankrollMetrics(
            picks_total      = len(filtered),
            picks_resolved   = len(resolved),
            wins             = len(wins),
            losses           = len(losses),
            pushes           = len(pushes),
            voids            = len(voids),
            pending          = len(pending),
            hit_rate         = round(hit_rate, 2),
            stake_total      = round(stake_total, 2),
            profit_total     = round(profit_total, 2),
            roi              = round(roi, 2),
            yield_pct        = round(roi, 2),
            bankroll_current = round(bankroll, 2),
            max_drawdown_pct = round(drawdown, 2),
            clv_mean         = clv_mean,
            brier_score      = brier_score,
            edge_real        = edge_real,
            avg_price        = round(avg_price, 4),
            avg_ev           = round(avg_ev, 2),
            filters_applied  = filters_str,
        )

    def equity_curve(
        self,
        sport:     str | None = None,
        market:    str | None = None,
        date_from: str | None = None,
        date_to:   str | None = None,
    ) -> list[dict]:
        """
        Retorna la equity curve filtrada como lista de dicts.

        Cada elemento tiene: date, event, market, selection, result,
        stake_amount, profit_amount, bankroll_after, drawdown_pct.
        """
        all_entries = self._store.load_all()
        filtered    = self._apply_filters(all_entries, sport, market, date_from, date_to)
        return self._equity_curve_entries(filtered)

    def monthly_stats(
        self,
        sport: str | None = None,
    ) -> list[dict]:
        """
        Estadísticas mensuales agrupadas por YYYY-MM.

        Si sport está especificado, solo retorna el deporte dado.
        Si None, agrupa todas las entradas sin distinción de deporte.

        Compatible con el formato de bankroll_monthly_stats.csv del
        sistema MLB — el dashboard puede leer ambos formatos.
        """
        all_entries = self._store.load_all()
        filtered    = self._apply_filters(all_entries, sport=sport)

        # Agrupar por mes
        by_month: dict[str, list[BetLedgerEntry]] = defaultdict(list)
        for entry in filtered:
            month = entry.date[:7] if len(entry.date) >= 7 else "unknown"
            by_month[month].append(entry)

        rows = []
        for month in sorted(by_month.keys()):
            entries = by_month[month]
            resolved = [e for e in entries if e.result in TERMINAL_RESULTS]
            wins     = sum(1 for e in resolved if e.result == "win")
            losses   = sum(1 for e in resolved if e.result == "lose")
            pushes   = sum(1 for e in resolved if e.result == "null")
            stake_total  = sum(e.stake_amount for e in resolved if e.result in ("win", "lose"))
            profit_total = sum(
                e.profit_amount for e in resolved
                if e.result in ("win", "lose") and e.profit_amount is not None
            )
            n = wins + losses
            roi = profit_total / stake_total * 100 if stake_total else 0.0
            hit_rate = wins / n * 100 if n else 0.0
            prices = [e.price for e in resolved if e.price > 1.0]
            avg_odds = sum(prices) / len(prices) if prices else 0.0

            # Bankroll start/end del mes
            curve_month = self._equity_curve_entries(entries)
            bank_start = curve_month[0]["bankroll_before"] if curve_month else self._initial_bankroll
            bank_end   = curve_month[-1]["bankroll_after"] if curve_month else bank_start
            growth = (bank_end - bank_start) / bank_start * 100 if bank_start else 0.0
            drawdown = self._max_drawdown(curve_month)

            rows.append({
                "month":           month,
                "sport":           sport or "all",
                "picks":           n,
                "wins":            wins,
                "losses":          losses,
                "pushes":          pushes,
                "stake_total":     round(stake_total, 2),
                "profit":          round(profit_total, 2),
                "roi_pct":         round(roi, 2),
                "yield_pct":       round(roi, 2),
                "hit_rate_pct":    round(hit_rate, 2),
                "bankroll_start":  round(bank_start, 2),
                "bankroll_end":    round(bank_end, 2),
                "growth_pct":      round(growth, 2),
                "max_drawdown_pct":round(drawdown, 2),
                "avg_odds":        round(avg_odds, 3),
            })
        return rows

    # ── Export ─────────────────────────────────────────────────────────────────

    def export_csv(self, path: str) -> None:
        """
        Exporta todas las entradas del ledger a un CSV.

        Útil para backup, auditoría y compatibilidad con el dashboard
        del sistema MLB (bankroll_history.csv).
        """
        entries = self._store.load_all()
        curve   = self._equity_curve_entries(entries)

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            fields = [
                "date", "event", "market", "selection", "sport",
                "result", "stake_pct", "stake_amount", "profit_amount",
                "bankroll_before", "bankroll_after", "drawdown_pct",
                "clv", "model_version",
            ]
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for row in curve:
                writer.writerow({k: row.get(k, "") for k in fields})

    # ── Helpers privados ───────────────────────────────────────────────────────

    @staticmethod
    def _apply_filters(
        entries:       list[BetLedgerEntry],
        sport:         str | None = None,
        market:        str | None = None,
        date_from:     str | None = None,
        date_to:       str | None = None,
        model_version: str | None = None,
    ) -> list[BetLedgerEntry]:
        result = entries
        if sport:
            result = [e for e in result if e.sport.lower() == sport.lower()]
        if market:
            result = [e for e in result if e.market.upper() == market.upper()]
        if date_from:
            result = [e for e in result if e.date >= date_from]
        if date_to:
            result = [e for e in result if e.date <= date_to]
        if model_version:
            result = [e for e in result if e.model_version == model_version]
        return result

    def _equity_curve_entries(
        self,
        entries: list[BetLedgerEntry],
    ) -> list[dict]:
        """
        Construye la equity curve ordenada cronológicamente.

        Solo incluye entradas resueltas con bankroll_after poblado.
        Calcula drawdown respecto al high-water mark acumulado.
        """
        resolved = [
            e for e in entries
            if e.result in TERMINAL_RESULTS and e.bankroll_after is not None
        ]
        resolved.sort(key=lambda e: e.created_at)

        hwm    = self._initial_bankroll
        curve  = []
        for e in resolved:
            after = e.bankroll_after
            hwm   = max(hwm, after)  # type: ignore[type-var]
            dd    = (after - hwm) / hwm * 100 if hwm else 0.0 # type: ignore
            curve.append({
                "date":            e.date,
                "event":           e.event,
                "market":          e.market,
                "selection":       e.selection,
                "sport":           e.sport,
                "result":          e.result,
                "stake_pct":       e.stake_pct,
                "stake_amount":    e.stake_amount,
                "profit_amount":   e.profit_amount,
                "bankroll_before": e.bankroll_before,
                "bankroll_after":  after,
                "drawdown_pct":    round(dd, 2),
                "clv":             e.clv,
                "model_version":   e.model_version,
            })
        return curve

    @staticmethod
    def _max_drawdown(curve: list[dict]) -> float:
        """Retorna el drawdown máximo (más negativo) de la equity curve."""
        if not curve:
            return 0.0
        return min((row["drawdown_pct"] for row in curve), default=0.0)