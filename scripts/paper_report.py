#!/usr/bin/env python3
"""
scripts/paper_report.py

Informe de paper trading multideporte con CLV por deporte y mercado.

Qué responde este informe
---------------------------
Una sola pregunta: ¿qué combinaciones de deporte y mercado tienen
ventaja real, y cuáles solo rellenan calendario?

Esa distinción importa porque la cobertura multideporte es lo que
permite dar picks a diario sin aflojar filtros — MLB cubre de abril a
septiembre, NFL los domingos de septiembre a febrero — pero solo
funciona si cada deporte aporta picks con valor. Rellenar el martes con
un mercado que pierde es el mismo error que bajar los umbrales, con
otro disfraz.

El audit de MLB ya mostró esa asimetría dentro de un mismo deporte:
TOTAL con edge confirmado (56% de acierto, +11.15% de ROI sobre 92
picks), mientras que ML y RL no lo tenían. Un informe agregado lo
habría ocultado.

Por qué el CLV manda sobre el ROI
-----------------------------------
El ROI arrastra el ruido del resultado: puedes estimar bien la
probabilidad y perder igual. El CLV compara tu precio con el de cierre,
que se observa exactamente.

    Detectar 1% de ventaja vía ROI : ~5.000 picks
    Detectar 1% de ventaja vía CLV : ~150-250 picks

Con CLV positivo y ROI plano, lo que falta es tiempo. Con CLV negativo
y ROI positivo, lo que sobra es suerte — y no se sostendrá.

Uso
----
    # Todos los deportes con ledger
    python scripts/paper_report.py

    # Uno solo, con el detalle por mercado
    python scripts/paper_report.py --sport mlb --detail

    # Desde una fecha
    python scripts/paper_report.py --from 2026-09-01

    # Exportar para análisis externo
    python scripts/paper_report.py --output informe.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_N = NormalDist()

# Picks mínimos para que el CLV sea interpretable.
#
# Con menos de 30 la media de CLV tiene un error estándar comparable a
# la propia media, y leerla como señal es engañarse.
_MIN_PICKS_CLV: int = 30

# Picks mínimos para que el ROI diga algo.
#
# Muy superior al de CLV porque el ROI carga la varianza del resultado
# del partido además de la del modelo.
_MIN_PICKS_ROI: int = 150


@dataclass
class Bucket:
    """Métricas de un subconjunto de picks (deporte, mercado o total)."""
    label:   str
    n:       int = 0
    wins:    int = 0
    losses:  int = 0
    pushes:  int = 0
    pending: int = 0
    staked:  float = 0.0
    profit:  float = 0.0
    clvs:    list[float] = field(default_factory=list)
    prices:  list[float] = field(default_factory=list)

    def add(self, entry) -> None:
        self.n += 1
        self.prices.append(entry.price)
        if entry.clv is not None:
            self.clvs.append(entry.clv)

        if entry.result == "win":
            self.wins += 1
        elif entry.result == "lose":
            self.losses += 1
        elif entry.result in ("null", "void"):
            self.pushes += 1
        else:
            self.pending += 1
            return

        self.staked += entry.stake_amount or 0.0
        self.profit += entry.profit_amount or 0.0

    # ── Resultado ─────────────────────────────────────────────────

    @property
    def resolved(self) -> int:
        return self.wins + self.losses

    @property
    def hit_rate(self) -> float | None:
        return 100.0 * self.wins / self.resolved if self.resolved else None

    @property
    def roi(self) -> float | None:
        return 100.0 * self.profit / self.staked if self.staked else None

    @property
    def breakeven(self) -> float | None:
        """Acierto necesario para no perder a la cuota media."""
        if not self.prices:
            return None
        return 100.0 / (sum(self.prices) / len(self.prices))

    # ── CLV ───────────────────────────────────────────────────────

    @property
    def clv_mean(self) -> float | None:
        return sum(self.clvs) / len(self.clvs) if self.clvs else None

    @property
    def clv_positive_pct(self) -> float | None:
        if not self.clvs:
            return None
        return 100.0 * sum(1 for c in self.clvs if c > 0) / len(self.clvs)

    @property
    def clv_p_value(self) -> float | None:
        """
        p de que el CLV medio sea > 0 por azar.

        Test t de una muestra contra cero. Con n grande la t converge a
        la normal, y a partir de 30 la diferencia es despreciable para
        lo que aquí se decide.
        """
        n = len(self.clvs)
        if n < 5:
            return None
        mean = sum(self.clvs) / n
        var = sum((c - mean) ** 2 for c in self.clvs) / (n - 1)
        if var <= 0:
            return 0.0 if mean > 0 else 1.0
        se = math.sqrt(var / n)
        return 1.0 - _N.cdf(mean / se)

    @property
    def verdict(self) -> str:
        """Lectura del CLV, con el tamaño de muestra en cuenta."""
        if len(self.clvs) < _MIN_PICKS_CLV:
            return f"muestra corta ({len(self.clvs)}/{_MIN_PICKS_CLV})"
        p = self.clv_p_value
        mean = self.clv_mean
        if p is None or mean is None:
            return "sin datos"
        if p < 0.05 and mean > 0:
            return "✅ CLV positivo significativo"
        if mean > 0:
            return f"CLV positivo, no concluyente (p={p:.2f})"
        return "❌ CLV negativo"


def main() -> int:
    args = _parse_args()

    from core.bankroll.tracker import BankrollTracker, CsvLedgerStore

    ledger_dir = Path("output/ledger")
    if not ledger_dir.exists():
        print("No hay ledgers en output/ledger/.")
        print("Ejecutar primero: python scripts/run_daily.py --sport <deporte>")
        return 2

    sports = ([args.sport] if args.sport
              else sorted(p.stem.replace("_roi_tracking", "")
                          for p in ledger_dir.glob("*_roi_tracking.csv")))
    if not sports:
        print("No se encontró ningún ledger.")
        return 2

    print(f"\n{'=' * 70}")
    print("  PAPER TRADING — CLV por deporte y mercado")
    print(f"{'=' * 70}")
    if args.date_from:
        print(f"  Desde: {args.date_from}")
    print()

    overall = Bucket("TOTAL")
    by_sport: dict[str, Bucket] = {}
    by_combo: dict[tuple[str, str], Bucket] = {}
    all_entries = []

    for sport in sports:
        path = ledger_dir / f"{sport}_roi_tracking.csv"
        if not path.exists():
            continue
        entries = CsvLedgerStore(str(path)).load_all()
        if args.date_from:
            entries = [e for e in entries if e.date >= args.date_from]
        if not entries:
            continue

        sport_bucket = by_sport.setdefault(sport, Bucket(sport.upper()))
        for e in entries:
            all_entries.append(e)
            sport_bucket.add(e)
            overall.add(e)
            combo = (sport, e.market)
            by_combo.setdefault(combo, Bucket(f"{sport.upper()} {e.market}")).add(e)

    if not overall.n:
        print("  Sin picks registrados en el rango indicado.")
        return 0

    # ── Tabla por deporte y mercado ───────────────────────────────
    print(f"  {'Deporte/Mercado':<20s} {'n':>4s} {'W-L-P':>10s} "
          f"{'hit':>7s} {'ROI':>8s} {'CLV':>8s} {'+CLV':>6s}")
    print("  " + "-" * 68)

    for sport in sorted(by_sport):
        _print_row(by_sport[sport], bold=True)
        combos = sorted(c for c in by_combo if c[0] == sport)
        for combo in combos:
            _print_row(by_combo[combo], indent=True)
        print()

    if len(by_sport) > 1:
        print("  " + "-" * 68)
        _print_row(overall, bold=True)
        print()

    # ── Veredicto por combinación ─────────────────────────────────
    print(f"{'=' * 70}")
    print("  VEREDICTO POR COMBINACIÓN")
    print(f"{'=' * 70}")
    print()
    print("  El CLV decide antes que el ROI: necesita ~20 veces menos")
    print("  muestra para separar señal de ruido.")
    print()

    ready, waiting, failing = [], [], []
    for combo in sorted(by_combo):
        b = by_combo[combo]
        verdict = b.verdict
        if verdict.startswith("✅"):
            ready.append(b)
        elif verdict.startswith("❌"):
            failing.append(b)
        else:
            waiting.append(b)

    for group, title in ((ready,   "Con evidencia de ventaja"),
                         (waiting, "Sin muestra suficiente todavía"),
                         (failing, "Sin ventaja — revisar antes de publicar")):
        if not group:
            continue
        print(f"  {title}")
        for b in group:
            clv = f"{b.clv_mean:+.2f}%" if b.clv_mean is not None else "—"
            print(f"    {b.label:<20s} n={b.n:<4d} CLV={clv:>8s}   {b.verdict}")
        print()

    # ── Cobertura del calendario ──────────────────────────────────
    _calendar_coverage(all_entries)

    # ── Lectura final ─────────────────────────────────────────────
    _final_read(overall, ready, failing)

    if args.output:
        _export(by_combo, overall, args.output)

    return 0


def _print_row(b: Bucket, indent: bool = False, bold: bool = False) -> None:
    prefix = "    ↳ " if indent else "  "
    label = b.label if not indent else b.label.split(" ", 1)[-1]
    wl = f"{b.wins}-{b.losses}-{b.pushes}"
    hit = f"{b.hit_rate:.1f}%" if b.hit_rate is not None else "—"
    roi = f"{b.roi:+.2f}%" if b.roi is not None else "—"
    clv = f"{b.clv_mean:+.2f}%" if b.clv_mean is not None else "—"
    pos = f"{b.clv_positive_pct:.0f}%" if b.clv_positive_pct is not None else "—"
    width = 18 if indent else 20
    print(f"{prefix}{label:<{width}s} {b.n:>4d} {wl:>10s} "
          f"{hit:>7s} {roi:>8s} {clv:>8s} {pos:>6s}")


def _calendar_coverage(entries) -> None:
    """
    Picks por mes y deporte.

    La cobertura multideporte es lo que permite dar picks a diario sin
    tocar los umbrales: MLB cubre de abril a septiembre, NFL los
    domingos de septiembre a febrero. Esta tabla muestra si el
    calendario está realmente cubierto o si hay huecos que tentarían a
    aflojar filtros para rellenarlos.
    """
    by_month: dict[str, dict[str, int]] = {}
    for e in entries:
        month = (e.date or "")[:7]
        if not month:
            continue
        by_month.setdefault(month, {}).setdefault(e.sport, 0)
        by_month[month][e.sport] += 1

    if len(by_month) < 2:
        return

    sports = sorted({s for m in by_month.values() for s in m})
    print(f"{'=' * 70}")
    print("  COBERTURA DEL CALENDARIO")
    print(f"{'=' * 70}")
    print()
    header = "  " + "Mes".ljust(10) + "".join(s.upper().rjust(8) for s in sports)
    print(header + "total".rjust(8))
    print("  " + "-" * (10 + 8 * (len(sports) + 1)))
    for month in sorted(by_month):
        row = "  " + month.ljust(10)
        total = 0
        for s in sports:
            n = by_month[month].get(s, 0)
            total += n
            row += (str(n) if n else "·").rjust(8)
        print(row + str(total).rjust(8))
    print()


def _final_read(overall: Bucket, ready: list, failing: list) -> None:
    print(f"{'=' * 70}")
    print("  LECTURA")
    print(f"{'=' * 70}")
    print()

    n_clv = len(overall.clvs)
    if n_clv < _MIN_PICKS_CLV:
        print(f"  {n_clv} picks con CLV registrado. Hacen falta al menos")
        print(f"  {_MIN_PICKS_CLV} para que la media sea interpretable.")
        print()
        print("  Si el número es bajo pese a llevar tiempo operando,")
        print("  revisar que capture_closing.py se esté ejecutando antes")
        print("  del inicio de los partidos.")
        return

    mean = overall.clv_mean or 0.0
    p = overall.clv_p_value

    print(f"  CLV global : {mean:+.2f}% sobre {n_clv} picks")
    if p is not None:
        print(f"  p          : {p:.4f}")
    if overall.roi is not None:
        print(f"  ROI        : {overall.roi:+.2f}% "
              f"({overall.resolved} resueltos)")
    print()

    if ready:
        combos = ", ".join(b.label for b in ready)
        print(f"  Con evidencia de ventaja: {combos}")
        print()
        print("  Esas combinaciones son las que sostienen un servicio de")
        print("  picks. Publicar solo esas, con el registro visible, es un")
        print("  producto defendible.")
    else:
        print("  Ninguna combinación alcanza evidencia de ventaja todavía.")
        print()
        print("  Publicar picks en este estado significa vender jugadas que")
        print("  no se ha demostrado que ganen. El coste no es solo del")
        print("  cliente: un servicio que pierde muere por abandono, no por")
        print("  falta de marketing.")

    if failing:
        combos = ", ".join(b.label for b in failing)
        print()
        print(f"  Sin ventaja: {combos}")
        print("  Rellenar calendario con estas combinaciones es el mismo")
        print("  error que bajar los umbrales, con otro disfraz.")

    if overall.resolved < _MIN_PICKS_ROI:
        print()
        print(f"  Nota: {overall.resolved} picks resueltos siguen por debajo")
        print(f"  de los {_MIN_PICKS_ROI} que hacen interpretable el ROI. El")
        print("  CLV es el indicador válido en esta fase.")


def _export(by_combo: dict, overall: Bucket, path: str) -> None:
    rows = []
    for b in list(by_combo.values()) + [overall]:
        rows.append({
            "grupo": b.label, "n": b.n,
            "wins": b.wins, "losses": b.losses, "pushes": b.pushes,
            "pending": b.pending,
            "hit_rate": round(b.hit_rate, 2) if b.hit_rate is not None else "",
            "breakeven": round(b.breakeven, 2) if b.breakeven is not None else "",
            "roi": round(b.roi, 2) if b.roi is not None else "",
            "clv_mean": round(b.clv_mean, 3) if b.clv_mean is not None else "",
            "clv_positive_pct": (round(b.clv_positive_pct, 1)
                                 if b.clv_positive_pct is not None else ""),
            "clv_n": len(b.clvs),
            "clv_p": (round(b.clv_p_value, 4)
                      if b.clv_p_value is not None else ""),
            "veredicto": b.verdict,
        })
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Exportado: {path} ({len(rows)} filas)")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Informe de paper trading con CLV por deporte y mercado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sport", default=None,
                        help="Limitar a un deporte. Default: todos")
    parser.add_argument("--from", dest="date_from", default=None,
                        help="Fecha inicial YYYY-MM-DD")
    parser.add_argument("--detail", action="store_true",
                        help="Incluir el desglose por mercado")
    parser.add_argument("--output", default=None,
                        help="CSV donde exportar el informe")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())