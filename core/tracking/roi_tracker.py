"""
core/tracking/roi_tracker.py

ROITracker: fachada genérica sobre BankrollTracker + SettlementProvider.

Posición en el pipeline
------------------------
Stage 10 (Settlement + Tracking):

    CandidatePick (activo, stake_pct fijado por RiskManager)
        ↓
    ROITracker.register_pick(pick, event_info) → BetLedgerEntry
        ↓
    ... pipeline continúa con otros eventos ...
        ↓
    ROITracker.settle_pending(provider) → list[SettlementResult]
        ↓
    BankrollTracker.settle() + update_clv()

Diferencias con el sistema MLB (tracking/roi_tracker.py)
----------------------------------------------------------
El sistema MLB mezclaba tres responsabilidades en un solo módulo:

1. Registro de picks → BankrollTracker.register() aquí.
2. Settlement deportivo → SettlementProvider (sport plugin) aquí.
3. Resolución de resultado (ML/RL/TOTAL) → _resolver_resultado()
   usaba statsapi.schedule y comparaba scores con lógica hardcodeada
   para baseball. Aquí eso pertenece exclusivamente al sport plugin.

La corrección central: settle_pending() no sabe qué deporte es,
no llama a ninguna API deportiva, no tiene lógica de "si mercado=RL
y el equipo tiene handicap -1.5...". Delega completamente en el
SettlementProvider que provee el sport plugin.

Idempotencia
-------------
settle_pending() es idempotente: puede llamarse múltiples veces
sin duplicar liquidaciones. BetLedgerEntry.settle() protege contra
re-liquidación (lanza ValueError si result != 'pending') y
BankrollTracker.register() protege contra duplicados de entry_id.

Esto permite al pipeline llamar settle_pending() al inicio de cada
ejecución diaria para resolver picks de días anteriores que quedaron
pendientes, sin riesgo de corrupción del ledger.

Exportación por deporte
------------------------
export_sport_views() genera ledgers separados por deporte
(roi_tracking_mlb.csv, roi_tracking_nba.csv, etc.) además del
ledger completo (roi_tracking.csv). El dashboard puede consumir
la vista por deporte para análisis independiente.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from core.bankroll.tracker import BankrollTracker
from core.contracts.ledger import BetLedgerEntry
from core.contracts.pick import CandidatePick
from core.tracking.protocols import SettlementProvider, SettlementResult, apply_settlement


# ── Resultado de registro ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class RegistrationResult:
    """
    Resultado de ROITracker.register_pick().

    Inmutable: snapshot del registro en el momento de la llamada.

    Campos
    ------
    entry         -- BetLedgerEntry creado y persistido.
    registered    -- True si se insertó, False si entry_id ya existía
                   (deduplicación — el pick estaba ya registrado).
    entry_id      -- ID del entry para referencia rápida.
    """
    entry:      BetLedgerEntry
    registered: bool
    entry_id:   str


# ── Motor principal ───────────────────────────────────────────────────────────

class ROITracker:
    """
    Fachada genérica sobre BankrollTracker + SettlementProvider.

    No contiene lógica deportiva. No sabe qué deporte es ni cómo
    se resuelve un resultado de apuesta. Delega en:
        - BankrollTracker: persistencia y métricas financieras
        - SettlementProvider: resolución del resultado deportivo

    Parámetros
    ----------
    tracker         -- BankrollTracker configurado con el LedgerStore
                      y el bankroll inicial del sistema.
    model_version   -- Identificador de la versión del modelo que
                      generó los picks. Se registra en cada
                      BetLedgerEntry para permitir comparación de
                      versiones en backtesting.
                      Formato sugerido: 'mlb-v2.1.0' o similar.
    """

    def __init__(
        self,
        tracker: BankrollTracker,
        model_version: str = "unversioned",
    ) -> None:
        self._tracker       = tracker
        self._model_version = model_version

    # ── Registro de picks ─────────────────────────────────────────────────────

    def register_pick(
        self,
        pick:       CandidatePick,
        event_info: dict[str, Any],
    ) -> RegistrationResult:
        """
        Registra un CandidatePick activo en el ledger como 'pending'.

        Idempotente: si el pick ya está registrado (mismo entry_id),
        retorna registered=False sin duplicar la entrada.

        Parámetros
        ----------
        pick        -- CandidatePick con active=True, stake_pct fijado
                      por StakingStrategy y aprobado por RiskManager.
                      No registra picks con active=False o stake_pct=0.
        event_info  -- Información del evento para el ledger. Campos
                      esperados (todos opcionales con fallback):
                          'sport':  str  — 'mlb', 'nba', etc.
                          'league': str  — 'MLB', 'NBA', etc.
                          'date':   str  — 'YYYY-MM-DD'
                          'event':  str  — 'Away @ Home'

        Retorna
        -------
        RegistrationResult con el entry creado y si fue nuevo.

        Raises
        ------
        ValueError -- Si pick.active=False o pick.stake_pct=0.
                     El tracker solo registra picks aprobados por
                     RiskManager.
        """
        if not pick.active:
            raise ValueError(
                f"register_pick() recibió un pick con active=False. "
                f"Solo picks aprobados por RiskManager deben registrarse. "
                f"inactive_reason='{pick.inactive_reason}'"
            )
        if pick.stake_pct <= 0:
            raise ValueError(
                f"register_pick() recibió pick con stake_pct={pick.stake_pct}. "
                f"StakingStrategy debe haber asignado stake > 0 antes del registro."
            )

        bankroll = self._tracker.current_bankroll()
        stake_amount = round(bankroll * pick.stake_pct / 100, 2)
        now = datetime.now(timezone.utc).isoformat()

        entry = BetLedgerEntry(
            entry_id        = self._make_entry_id(pick),
            sport           = event_info.get("sport", pick.event.sport),
            league          = event_info.get("league", ""),
            date            = event_info.get("date", ""),
            event           = event_info.get("event",
                               f"{pick.event.away_team} @ {pick.event.home_team}"),
            market          = pick.market,
            selection       = pick.selection,
            price           = pick.price,
            model_prob      = pick.model_prob_raw,
            ev              = pick.ev,
            stake_pct       = pick.stake_pct,
            stake_amount    = stake_amount,
            bankroll_before = bankroll,
            result          = "pending",
            model_version   = self._model_version,
            created_at      = now,
        )

        registered = self._tracker.register(entry)

        return RegistrationResult(
            entry      = entry,
            registered = registered,
            entry_id   = entry.entry_id,
        )

    # ── Settlement de picks pendientes ────────────────────────────────────────

    def settle_pending(
        self,
        provider: SettlementProvider,
        sport:    str | None = None,
    ) -> list[SettlementResult]:
        """
        Resuelve todos los picks pendientes usando el SettlementProvider.

        Idempotente: picks ya liquidados son ignorados. Picks cuyo
        resultado no está disponible aún (provider retorna None)
        permanecen 'pending' para la siguiente ejecución.

        Parámetros
        ----------
        provider  -- Implementación de SettlementProvider del sport
                    plugin. Solo sabe resolver picks de su deporte.
        sport     -- Si se especifica, solo intenta resolver picks
                    de ese deporte. Útil cuando el pipeline procesa
                    un deporte a la vez y hay picks pendientes de
                    múltiples deportes en el ledger.

        Retorna
        -------
        list[SettlementResult] de picks efectivamente liquidados
        en esta llamada (no incluye los que siguen pending).
        """
        all_entries = self._tracker._store.load_all()

        pending = [
            e for e in all_entries
            if e.result == "pending"
            and (sport is None or e.sport.lower() == sport.lower())
        ]

        settled: list[SettlementResult] = []

        for entry in pending:
            try:
                result = provider.get_result(entry)
            except Exception:
                # Error del provider (API caída, timeout) —
                # el pick permanece pending, el pipeline continúa
                continue

            if result is None:
                # Partido no terminado aún — pick permanece pending
                continue

            try:
                apply_settlement(
                    entry   = entry,
                    result  = result,
                    tracker = self._tracker,
                )
                settled.append(result)
            except (KeyError, ValueError):
                # KeyError: entry_id no encontrado (no debería ocurrir)
                # ValueError: pick ya liquidado (idempotencia — ignorar)
                continue

        return settled

    # ── Métricas y exportación ────────────────────────────────────────────────

    def roi_summary(
        self,
        sport:     str | None = None,
        market:    str | None = None,
        date_from: str | None = None,
        date_to:   str | None = None,
    ) -> dict:
        """
        Retorna resumen de ROI filtrable como dict para logging y Telegram.

        Delega en BankrollTracker.metrics() — no duplica lógica.

        Retorna dict con las claves que el sistema MLB usaba en el log:
            total_apuestas, wins, roi, ganancias, bankroll, pendientes.
        Más las nuevas:
            clv_mean, brier_score, sport (filtro aplicado).
        """
        m = self._tracker.metrics(
            sport     = sport,
            market    = market,
            date_from = date_from,
            date_to   = date_to,
        )
        return {
            "total_apuestas": m.picks_resolved,
            "wins":           m.wins,
            "losses":         m.losses,
            "pushes":         m.pushes,
            "pendientes":     m.pending,
            "roi":            m.roi,
            "yield":          m.yield_pct,
            "hit_rate":       m.hit_rate,
            "ganancias":      m.profit_total,
            "bankroll":       m.bankroll_current,
            "max_drawdown":   m.max_drawdown_pct,
            "clv_mean":       m.clv_mean,
            "brier_score":    m.brier_score,
            "sport":          sport or "all",
            "filters":        m.filters_applied,
        }

    def export_sport_views(self, base_dir: str = "output/ledger") -> list[str]:
        """
        Exporta vistas del ledger filtradas por deporte.

        Genera:
            {base_dir}/roi_tracking.csv           — ledger completo
            {base_dir}/roi_tracking_{sport}.csv   — por deporte

        Retorna la lista de rutas de archivos generados.
        """
        import os
        os.makedirs(base_dir, exist_ok=True)

        paths = []

        # Ledger completo
        full_path = os.path.join(base_dir, "roi_tracking.csv")
        self._tracker.export_csv(full_path)
        paths.append(full_path)

        # Vistas por deporte
        all_entries = self._tracker._store.load_all()
        sports = {e.sport for e in all_entries if e.sport}

        for sport in sorted(sports):
            sport_path = os.path.join(base_dir, f"roi_tracking_{sport}.csv")
            # Crear tracker temporal filtrado por sport para export
            # Reutilizar export_csv con filtro via monthly_stats no es directo;
            # escribimos el CSV manualmente con las entradas filtradas
            self._export_filtered(
                entries=[e for e in all_entries if e.sport == sport],
                path=sport_path,
            )
            paths.append(sport_path)

        return paths

    def monthly_stats(self, sport: str | None = None) -> list[dict]:
        """Stats mensuales delegados al BankrollTracker."""
        return self._tracker.monthly_stats(sport=sport)

    # ── Helpers privados ───────────────────────────────────────────────────────

    @staticmethod
    def _make_entry_id(pick: CandidatePick) -> str:
        """
        Genera un entry_id determinístico basado en el pick.

        Determinístico: mismo pick en la misma ejecución → mismo ID.
        Esto garantiza la idempotencia de register_pick(): si el pipeline
        se ejecuta dos veces con el mismo pick, el segundo intento
        detecta el duplicado y retorna registered=False.

        Formato: {event_id}_{market}_{selection_slug}
        donde selection_slug es la selección sin espacios en minúsculas.
        """
        selection_slug = pick.selection.lower().replace(" ", "_")[:20]
        return f"{pick.event.event_id}_{pick.market}_{selection_slug}"

    @staticmethod
    def _export_filtered(entries: list[BetLedgerEntry], path: str) -> None:
        """Exporta una lista filtrada de entries a CSV."""
        import csv
        fields = [
            "date", "event", "market", "selection", "sport", "league",
            "result", "stake_pct", "stake_amount", "profit_amount",
            "bankroll_before", "bankroll_after", "clv", "model_version",
        ]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for e in entries:
                writer.writerow({
                    "date":            e.date,
                    "event":           e.event,
                    "market":          e.market,
                    "selection":       e.selection,
                    "sport":           e.sport,
                    "league":          e.league,
                    "result":          e.result,
                    "stake_pct":       e.stake_pct,
                    "stake_amount":    e.stake_amount,
                    "profit_amount":   e.profit_amount if e.profit_amount is not None else "",
                    "bankroll_before": e.bankroll_before,
                    "bankroll_after":  e.bankroll_after if e.bankroll_after is not None else "",
                    "clv":             e.clv if e.clv is not None else "",
                    "model_version":   e.model_version,
                })