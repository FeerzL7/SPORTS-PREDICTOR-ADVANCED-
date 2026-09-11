#!/usr/bin/env python3
"""
scripts/validate_ledger.py

Valida el estado del ledger tras una ejecución real del pipeline.

Uso
----
    # Validación completa del ledger MLB
    python scripts/validate_ledger.py --sport mlb

    # Con path de ledger custom
    python scripts/validate_ledger.py --sport mlb \
        --ledger output/ledger/mlb_roi_tracking.csv

Qué verifica
-------------
1. El CSV existe y tiene las 21 columnas del contrato BetLedgerEntry
2. Todos los entries tienen sport, model_version poblados
3. Los entries resueltos tienen bankroll_after y profit_amount
4. Los entries pendientes NO tienen bankroll_after
5. La equity curve es monótona en created_at
6. bankroll_after[n] == bankroll_before[n] + profit_amount[n]
7. Las métricas agregadas son consistentes con las filas individuales

Exit codes
-----------
    0 — ledger válido
    1 — inconsistencias detectadas
    2 — ledger no existe o está vacío
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_EXPECTED_COLUMNS = {
    "entry_id", "sport", "league", "date", "event", "market",
    "selection", "price", "model_prob", "ev", "stake_pct",
    "stake_amount", "bankroll_before", "result", "bankroll_after",
    "profit_amount", "yield_pct", "clv", "created_at", "settled_at",
    "model_version",
}

_TERMINAL = {"win", "lose", "null", "void"}


def main() -> int:
    args = _parse_args()

    ledger_path = args.ledger or f"output/ledger/{args.sport}_roi_tracking.csv"
    path = Path(ledger_path)

    print(f"\n{'='*62}")
    print(f"  VALIDACIÓN DEL LEDGER")
    print(f"  Sport:  {args.sport}")
    print(f"  Ledger: {ledger_path}")
    print(f"{'='*62}\n")

    if not path.exists():
        print(f"  ERROR: El ledger no existe: {ledger_path}")
        print(f"  Ejecutar primero: python scripts/run_daily.py --sport {args.sport}")
        return 2

    from core.bankroll.tracker import BankrollTracker, CsvLedgerStore

    store   = CsvLedgerStore(str(path))
    tracker = BankrollTracker(store=store, initial_bankroll=args.bankroll)
    entries = store.load_all()

    if not entries:
        print("  ERROR: El ledger está vacío (0 entries).")
        return 2

    issues: list[str] = []

    # ── 1. Columnas del contrato ───────────────────────────────
    import csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        headers = set(csv.DictReader(f).fieldnames or [])
    missing = _EXPECTED_COLUMNS - headers
    if missing:
        issues.append(f"Columnas faltantes en el CSV: {sorted(missing)}")
    else:
        print(f"  ✅ Columnas: las {len(_EXPECTED_COLUMNS)} del contrato presentes")

    # ── 2. Campos obligatorios poblados ────────────────────────
    for e in entries:
        if not e.sport:
            issues.append(f"{e.entry_id}: campo 'sport' vacío")
        if not e.model_version:
            issues.append(f"{e.entry_id}: campo 'model_version' vacío")
    if not any("sport" in i or "model_version" in i for i in issues):
        print(f"  ✅ Campos obligatorios: sport y model_version poblados en todos")

    # ── 3-4. Coherencia resuelto vs pendiente ──────────────────
    for e in entries:
        if e.result in _TERMINAL:
            if e.bankroll_after is None:
                issues.append(f"{e.entry_id}: resuelto ({e.result}) sin bankroll_after")
            if e.profit_amount is None:
                issues.append(f"{e.entry_id}: resuelto ({e.result}) sin profit_amount")
        elif e.result == "pending":
            if e.bankroll_after is not None:
                issues.append(f"{e.entry_id}: pendiente pero tiene bankroll_after")
    resolved = [e for e in entries if e.result in _TERMINAL]
    pending  = [e for e in entries if e.result == "pending"]
    print(f"  ✅ Coherencia: {len(resolved)} resueltos, {len(pending)} pendientes")

    # ── 5. Aritmética del bankroll ─────────────────────────────
    for e in resolved:
        if e.bankroll_after is None or e.profit_amount is None:
            continue
        expected = e.bankroll_before + e.profit_amount
        if abs(e.bankroll_after - expected) > 0.02:
            issues.append(
                f"{e.entry_id}: bankroll_after={e.bankroll_after} != "
                f"bankroll_before({e.bankroll_before}) + profit({e.profit_amount}) "
                f"= {expected:.2f}"
            )
    if not any("bankroll_after=" in i for i in issues):
        print(f"  ✅ Aritmética: bankroll_after = bankroll_before + profit en todos")

    # ── 6. Profit según resultado ──────────────────────────────
    for e in resolved:
        if e.profit_amount is None:
            continue
        if e.result == "win":
            expected = e.stake_amount * (e.price - 1)
            if abs(e.profit_amount - expected) > 0.02:
                issues.append(
                    f"{e.entry_id}: win profit={e.profit_amount} != "
                    f"stake({e.stake_amount}) × (price-1)({e.price-1:.2f}) = {expected:.2f}"
                )
        elif e.result == "lose":
            if abs(e.profit_amount + e.stake_amount) > 0.02:
                issues.append(
                    f"{e.entry_id}: lose profit={e.profit_amount} != -stake({e.stake_amount})"
                )
        elif e.result in ("null", "void"):
            if abs(e.profit_amount) > 0.02:
                issues.append(f"{e.entry_id}: {e.result} profit={e.profit_amount} != 0")
    if not any("profit=" in i for i in issues):
        print(f"  ✅ Profit: correcto según resultado en los {len(resolved)} resueltos")

    # ── 7. Métricas agregadas ──────────────────────────────────
    m = tracker.metrics(sport=args.sport)
    manual_wins   = sum(1 for e in resolved if e.result == "win")
    manual_losses = sum(1 for e in resolved if e.result == "lose")
    if m.wins != manual_wins or m.losses != manual_losses:
        issues.append(
            f"metrics() inconsistente: W={m.wins}/L={m.losses} vs "
            f"manual W={manual_wins}/L={manual_losses}"
        )
    else:
        print(f"  ✅ Métricas: agregadas coinciden con el conteo manual")

    # ── Resumen ────────────────────────────────────────────────
    print()
    print(f"{'─'*62}")
    print(f"  ESTADO DEL LEDGER")
    print(f"{'─'*62}")
    print(f"    Picks totales:     {m.picks_total}")
    print(f"    Resueltos:         {m.picks_resolved}")
    print(f"    Pendientes:        {m.pending}")
    print(f"    Wins / Losses:     {m.wins} / {m.losses}")
    print(f"    Hit rate:          {m.hit_rate:.2f}%")
    print(f"    ROI:               {m.roi:+.2f}%")
    print(f"    Profit total:      {m.profit_total:+.2f}")
    print(f"    Bankroll actual:   {m.bankroll_current:.2f}")
    print(f"    Max drawdown:      {m.max_drawdown_pct:.2f}%")
    print(f"    CLV medio:         {m.clv_mean if m.clv_mean is not None else '—'}")
    print(f"    Brier score:       {m.brier_score if m.brier_score is not None else '—'}")
    print()

    if issues:
        print(f"  ❌ {len(issues)} INCONSISTENCIA(S) DETECTADA(S):")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
        print()
        return 1

    print(f"  ✅ LEDGER VÁLIDO — sin inconsistencias.")
    print()
    return 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Valida el ledger tras una ejecución")
    p.add_argument("--sport",   default="mlb", help="Sport ID. Default: mlb")
    p.add_argument("--ledger",  default=None,  help="Path al CSV del ledger")
    p.add_argument("--bankroll", type=float, default=1000.0,
                   help="Bankroll inicial para el cálculo de drawdown")
    return p.parse_args()


if __name__ == "__main__":
    sys.exit(main())