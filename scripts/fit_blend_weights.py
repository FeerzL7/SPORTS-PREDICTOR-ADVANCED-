#!/usr/bin/env python3
"""
scripts/fit_blend_weights.py

Estima `blending.{MERCADO}.model_weight` con datos y lo valida en una
temporada reservada. Dice, mercado por mercado, si el modelo aporta
información que el mercado de cierre no tenga.

Para qué
---------
El peso del modelo se fijaba a mano. En fútbol la config usaba 0.45
cuando el peso que minimiza la log-loss es 0.00: el modelo no añadía
nada al cierre, y los picks salían de sus discrepancias con el
mercado, que son ruido. Este script convierte esa comprobación en un
paso obligatorio antes de sacar un mercado de solo-papel.

Método y veredicto: ver core/evaluation/blend_fit.py.

Entrada
--------
Cualquier CSV de backtest con estas columnas:

    season, market, model_prob, market_prob, result

`result` en {win, lose}; push y anulados se descartan. El partido se
identifica con `match`+`date`, o `game_id`, o `matchup`+`week`, para
el bootstrap agrupado.

IMPORTANTE: el CSV debe traer TODOS los candidatos, no solo los que
pasaron el filtro. Ajustar el peso solo con los picks filtrados mide
el modelo justo donde más discrepa del mercado: un sesgo de selección
que lo hace parecer peor (o mejor) de lo que es. Los backtests de
fútbol exportan todos con `--no-filter` (columna passed_filter).

Uso
----
    python scripts/fit_blend_weights.py bt_full.csv
    python scripts/fit_blend_weights.py bt_full.csv bt_latam.csv --holdout 2023
    python scripts/fit_blend_weights.py bt_open.csv --market TOTAL

Código de salida: 0 si se evaluó al menos un mercado, 1 si no.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from core.evaluation.blend_fit import (  # noqa: E402
    VERDICT_INFORMATIVE,
    Observation,
    evaluate_market,
)

_REQUIRED = ("season", "market", "model_prob", "market_prob", "result")


def _group_key(row: dict, line_num: int) -> str:
    """Identificador del partido, de la columna más específica disponible."""
    if row.get("match"):
        return f"{row.get('date', '')}|{row['match']}"
    if row.get("game_id"):
        return row["game_id"]
    if row.get("matchup"):
        return f"{row.get('season', '')}|{row.get('week', '')}|{row['matchup']}"
    return f"fila-{line_num}"


def _load(paths: list[str]) -> tuple[dict[str, list[Observation]], list[str]]:
    """Lee los CSV y agrupa las observaciones por mercado."""
    by_market: dict[str, list[Observation]] = {}
    problems: list[str] = []

    for path in paths:
        with open(path, encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in _REQUIRED if c not in (reader.fieldnames or [])]
            if missing:
                problems.append(
                    f"{path}: faltan columnas {missing}. Sin la probabilidad "
                    f"del mercado por selección no se puede medir qué añade "
                    f"el modelo."
                )
                continue
            if "passed_filter" not in (reader.fieldnames or []):
                problems.append(
                    f"{path}: sin columna passed_filter — si solo contiene los "
                    f"picks filtrados, el ajuste tiene sesgo de selección."
                )
            for row in reader:
                if row["result"] not in ("win", "lose"):
                    continue
                try:
                    obs = Observation(
                        model_prob  = float(row["model_prob"]),
                        market_prob = float(row["market_prob"]),
                        outcome     = 1 if row["result"] == "win" else 0,
                        season      = int(float(row["season"])),
                        group       = f"{path}|{_group_key(row, reader.line_num)}",
                    )
                except (TypeError, ValueError):
                    continue
                if not (0 < obs.model_prob < 1 and 0 < obs.market_prob < 1):
                    continue
                by_market.setdefault(row["market"].upper(), []).append(obs)

    return by_market, problems


def main() -> int:
    args = _parse_args()
    by_market, problems = _load(args.csv)

    for p in problems:
        print(f"  AVISO  {p}")

    if args.market:
        by_market = {m: o for m, o in by_market.items() if m == args.market.upper()}
    if not by_market:
        print("\n  No hay observaciones evaluables.")
        return 1

    print()
    print("=" * 78)
    print("  PESO DEL MODELO FRENTE AL MERCADO — validación fuera de muestra")
    print("=" * 78)

    fits = []
    for market, obs in sorted(by_market.items()):
        seasons = sorted({o.season for o in obs})
        holdout = args.holdout or [seasons[-1]]
        try:
            fit = evaluate_market(market, obs, holdout, n_boot=args.boot)
        except ValueError as e:
            print(f"\n  {market}: {e}")
            continue
        fits.append(fit)

        print(f"\n  {market}")
        print(f"    entrenamiento : {fit.train_seasons}  n={fit.n_train}")
        print(f"    prueba        : {fit.test_seasons}  n={fit.n_test}")
        print(f"    peso óptimo   : {fit.weight:.2f}   (ajustado solo en entrenamiento)")
        print(f"    Δlog-loss     : entrenamiento {fit.delta_train:+.5f} | "
              f"prueba {fit.delta_test:+.5f}  IC95 [{fit.ci_low:+.5f}, {fit.ci_high:+.5f}]")
        print(f"    combinación logística: modelo {fit.stack_model_coef:+.2f} · "
              f"mercado {fit.stack_market_coef:+.2f}")
        if fit.stack_model_coef < 0:
            print("      coeficiente del modelo NEGATIVO: sus discrepancias con el")
            print("      mercado apuntan en la dirección equivocada.")
        print(f"    veredicto     : {fit.verdict}")

    if not fits:
        return 1

    print()
    print("-" * 78)
    print("  Lectura")
    print("-" * 78)
    print("  Δlog-loss < 0 con el intervalo entero por debajo de 0 = el modelo")
    print("  mejora al mercado en datos que el ajuste no vio. Es condición")
    print("  NECESARIA para sacar un mercado de solo-papel; la suficiente es")
    print("  CLV > 0 en el ledger de papel con n ≥ 100.")
    print()
    print("  Sugerencia para config/{sport}.yaml:")
    print()
    paper = [f.market for f in fits if f.verdict != VERDICT_INFORMATIVE]
    print("    blending:")
    for f in fits:
        print(f"      {f.market!r}:\n        model_weight: {f.weight:.2f}")
    print("    risk:")
    print(f"      paper_only_markets: {paper}")
    print()
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ajusta y valida el peso del modelo frente al mercado",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv", nargs="+", help="CSV(s) de backtest")
    parser.add_argument("--holdout", type=int, nargs="+", default=None,
                        help="Temporada(s) reservadas. Default: la última de cada mercado")
    parser.add_argument("--market", default=None, help="Evaluar solo este mercado")
    parser.add_argument("--boot", type=int, default=1000,
                        help="Remuestreos del bootstrap (default 1000)")
    return parser.parse_args()


if __name__ == "__main__":
    sys.exit(main())
