#!/usr/bin/env python3
"""
scripts/analyze_backtest.py

Analiza un CSV de backtest y prueba filtros alternativos.

Para qué
---------
El backtest dice SI el sistema gana. Este script dice DÓNDE pierde y
qué umbral lo arreglaría, sin volver a descargar datos ni reproyectar:
trabaja sobre el CSV que exporta backtest_soccer.py, que ya contiene la
probabilidad del modelo, la del mercado, el precio y el resultado de
cada pick.

El problema que motivó escribirlo
-----------------------------------
El backtest de fútbol sobre 2021-2023 dio ROI -5.50% con un déficit de
84 victorias sobre lo que implicaban las cuotas. El desglose señaló al
1X2: el 86% del déficit con el 65% de los picks.

La causa es que `min_ev` no es comparable entre precios. El EV es
derivada del precio —d(EV)/d(p) = precio— así que un error de 1 punto
en la probabilidad produce 6 puntos de EV a cuota 6.00 y solo 1.8 a
cuota 1.80.

Con un umbral uniforme del 3%:

    cuota 1.80  el modelo necesita 3.70 pp de desacuerdo
    cuota 6.00  le basta 1.11 pp

El listón es tres veces más bajo donde las estimaciones son tres veces
más ruidosas. Cualquier ruido del modelo en probabilidades bajas genera
picks, y el modelo tiene ruido: todos lo tienen.

Qué prueba
------------
Cinco familias de filtro sobre los mismos picks:

    ACTUAL        El que produjo el resultado, como referencia.
    EV ESCALADO   min_ev proporcional al precio, para que la relación
                  señal-ruido sea constante.
    EDGE RELATIVO Exige un desacuerdo porcentual, no absoluto.
    BANDA DE PRECIO  Restringe el rango operable.
    SOLO TOTAL    El mercado que menos pierde, aislado.

No es optimización: probar veinte umbrales y quedarse con el mejor
sobre los mismos datos produce un resultado que no se reproduce. El
informe marca cuántas configuraciones se probaron y advierte del
sobreajuste.

Uso
----
    python scripts/analyze_backtest.py bt.csv
    python scripts/analyze_backtest.py bt.csv --market 1X2
    python scripts/analyze_backtest.py bt.csv --split 2023
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

_N = NormalDist()

# Precio de referencia para escalar el umbral de EV.
#
# 2.00 es la cuota de una apuesta a par. Escalar respecto a ella hace
# que el umbral crezca proporcionalmente con la volatilidad del EV.
_REFERENCE_PRICE = 2.0


@dataclass
class Pick:
    comp:        str
    season:      int
    market:      str
    price:       float
    model_prob:  float
    market_prob: float
    ev:          float
    result:      str
    profit:      float
    confidence:  float

    @property
    def edge(self) -> float:
        return self.model_prob - self.market_prob

    @property
    def relative_edge(self) -> float:
        """Desacuerdo como fracción de la probabilidad implícita."""
        return self.edge / self.market_prob if self.market_prob > 0 else 0.0

    @property
    def resolved(self) -> bool:
        return self.result in ("win", "lose")


@dataclass
class Result:
    label:   str
    n:       int = 0
    wins:    int = 0
    losses:  int = 0
    profit:  float = 0.0
    expected: float = 0.0

    @property
    def resolved(self) -> int:
        return self.wins + self.losses

    @property
    def roi(self) -> float | None:
        return 100.0 * self.profit / self.n if self.n else None

    @property
    def hit_rate(self) -> float | None:
        return 100.0 * self.wins / self.resolved if self.resolved else None

    @property
    def surplus(self) -> float:
        return self.wins - self.expected


def _load(path: str) -> list[Pick]:
    picks: list[Pick] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                picks.append(Pick(
                    comp=row["comp"],
                    season=int(row["season"]),
                    market=row["market"],
                    price=float(row["price"]),
                    model_prob=float(row["model_prob"]),
                    market_prob=float(row["market_prob"]),
                    ev=float(row["ev"]),
                    result=row["result"],
                    profit=float(row["profit"]),
                    confidence=float(row["confidence"]),
                ))
            except (KeyError, ValueError):
                continue
    return picks


def _evaluate(picks: list[Pick], label: str, keep) -> Result:
    """Aplica un criterio y agrega el resultado."""
    result = Result(label=label)
    for p in picks:
        if not keep(p):
            continue
        result.n += 1
        result.profit += p.profit
        if p.result == "win":
            result.wins += 1
            result.expected += 1.0 / p.price
        elif p.result == "lose":
            result.losses += 1
            result.expected += 1.0 / p.price
    return result


def _report(results: list[Result], titulo: str) -> None:
    print(f"\n  {titulo}")
    print(f"    {'Criterio':30s} {'n':>5s} {'hit':>7s} {'ROI':>8s} {'excedente':>10s}")
    print("    " + "-" * 62)
    for r in results:
        if r.n == 0:
            print(f"    {r.label:30s} {0:5d}        —        —          —")
            continue
        hit = f"{r.hit_rate:.2f}%" if r.hit_rate is not None else "—"
        roi = f"{r.roi:+.2f}%" if r.roi is not None else "—"
        print(f"    {r.label:30s} {r.n:5d} {hit:>7s} {roi:>8s} "
              f"{r.surplus:>+10.1f}")


def _price_bands(picks: list[Pick]) -> None:
    """
    Rendimiento por banda de precio.

    Es el diagnóstico más directo: si el ROI empeora conforme sube el
    precio, el umbral de EV está dejando pasar ruido en los longshots.
    """
    bandas = [
        ("1.50 - 2.00", 1.50, 2.00),
        ("2.00 - 2.75", 2.00, 2.75),
        ("2.75 - 3.75", 2.75, 3.75),
        ("3.75 - 5.00", 3.75, 5.00),
        ("5.00 +",      5.00, 99.0),
    ]
    results = [
        _evaluate(picks, label, lambda p, lo=lo, hi=hi: lo <= p.price < hi)
        for label, lo, hi in bandas
    ]
    _report(results, "POR BANDA DE PRECIO")

    con_datos = [r for r in results if r.n >= 50 and r.roi is not None]
    if len(con_datos) >= 3:
        peor = min(con_datos, key=lambda r: r.roi)
        mejor = max(con_datos, key=lambda r: r.roi)
        print()
        print(f"    Mejor banda : {mejor.label} ({mejor.roi:+.2f}%)")
        print(f"    Peor banda  : {peor.label} ({peor.roi:+.2f}%)")
        if peor.label.startswith(("3.75", "5.00")):
            print()
            print("    El ROI empeora al subir el precio: es la firma de un")
            print("    umbral de EV que acepta ruido en los longshots.")


def _ev_bands(picks: list[Pick]) -> None:
    """
    Rendimiento por EV declarado.

    Si el ROI no crece con el EV, el EV no está midiendo lo que dice
    medir — y filtrar por él no ayuda.
    """
    bandas = [(f"EV {lo}-{hi}%", lo, hi) for lo, hi in
              ((3, 5), (5, 8), (8, 12), (12, 20), (20, 999))]
    results = [
        _evaluate(picks, label, lambda p, lo=lo, hi=hi: lo <= p.ev < hi)
        for label, lo, hi in bandas
    ]
    _report(results, "POR EV DECLARADO")

    con_datos = [r for r in results if r.n >= 50 and r.roi is not None]
    if len(con_datos) >= 3:
        monotono = all(
            con_datos[i].roi <= con_datos[i + 1].roi
            for i in range(len(con_datos) - 1)
        )
        print()
        if monotono:
            print("    El ROI crece con el EV: el EV discrimina.")
        else:
            print("    El ROI NO crece con el EV declarado. El EV no está")
            print("    midiendo ventaja real — filtrar por él no ayuda, y")
            print("    subirle el umbral tampoco lo arreglaría.")


def _alternatives(picks: list[Pick]) -> int:
    """Prueba familias de filtro alternativas. Retorna cuántas se probaron."""
    probadas = 0

    # ── EV escalado por precio ─────────────────────────────────
    results = [_evaluate(picks, "actual (min_ev ≥ 3%)", lambda p: True)]
    for base in (3.0, 4.0, 5.0):
        label = f"EV ≥ {base:.0f}% × (precio/2)"
        results.append(_evaluate(
            picks, label,
            lambda p, b=base: p.ev >= b * (p.price / _REFERENCE_PRICE),
        ))
        probadas += 1
    _report(results, "EV ESCALADO POR PRECIO")
    print()
    print("    El EV es derivada del precio: un error de 1 pp en la")
    print("    probabilidad da 6 pp de EV a cuota 6.00 y 1.8 a cuota 1.80.")
    print("    Escalar el umbral mantiene constante la relación señal-ruido.")

    # ── Edge relativo ──────────────────────────────────────────
    results = [_evaluate(picks, "actual", lambda p: True)]
    for umbral in (0.08, 0.12, 0.18, 0.25):
        results.append(_evaluate(
            picks, f"edge relativo ≥ {umbral:.0%}",
            lambda p, u=umbral: p.relative_edge >= u,
        ))
        probadas += 1
    _report(results, "EDGE RELATIVO (desacuerdo porcentual)")

    # ── Banda de precio ────────────────────────────────────────
    results = [_evaluate(picks, "actual (1.70 - 6.50)", lambda p: True)]
    for lo, hi in ((1.70, 3.00), (1.70, 2.50), (2.00, 4.00), (1.50, 2.50)):
        results.append(_evaluate(
            picks, f"precio {lo:.2f} - {hi:.2f}",
            lambda p, a=lo, b=hi: a <= p.price <= b,
        ))
        probadas += 1
    _report(results, "BANDA DE PRECIO OPERABLE")

    # ── Por confianza ──────────────────────────────────────────
    results = [_evaluate(picks, "actual", lambda p: True)]
    for umbral in (0.70, 0.80, 0.90):
        results.append(_evaluate(
            picks, f"confianza ≥ {umbral:.2f}",
            lambda p, u=umbral: p.confidence >= u,
        ))
        probadas += 1
    _report(results, "POR CONFIANZA DEL MODELO")

    return probadas


def _split_validation(picks: list[Pick], holdout: int) -> None:
    """
    Separa una temporada como validación.

    Sin esto, cualquier umbral que se elija mirando todos los datos
    está sobreajustado por construcción. La pregunta útil no es qué
    filtro habría funcionado mejor en 2021-2023, sino si el que elijo
    mirando 2021-2022 sigue funcionando en 2023.
    """
    train = [p for p in picks if p.season != holdout]
    test = [p for p in picks if p.season == holdout]

    if not train or not test:
        return

    print()
    print("=" * 70)
    print(f"  VALIDACIÓN FUERA DE MUESTRA — temporada {holdout} reservada")
    print("=" * 70)
    print(f"\n  Entrenamiento: {len(train)} picks   Validación: {len(test)} picks")

    candidatos = [
        ("actual", lambda p: True),
        ("EV ≥ 4% × (precio/2)", lambda p: p.ev >= 4.0 * (p.price / _REFERENCE_PRICE)),
        ("edge relativo ≥ 12%", lambda p: p.relative_edge >= 0.12),
        ("precio 1.70 - 2.50", lambda p: 1.70 <= p.price <= 2.50),
        ("solo TOTAL", lambda p: p.market == "TOTAL"),
    ]

    print(f"\n  {'Criterio':26s} {'ROI train':>11s} {'ROI test':>11s} {'n test':>8s}")
    print("  " + "-" * 60)
    for label, keep in candidatos:
        r_train = _evaluate(train, label, keep)
        r_test = _evaluate(test, label, keep)
        roi_tr = f"{r_train.roi:+.2f}%" if r_train.roi is not None else "—"
        roi_te = f"{r_test.roi:+.2f}%" if r_test.roi is not None else "—"
        print(f"  {label:26s} {roi_tr:>11s} {roi_te:>11s} {r_test.n:>8d}")

    print()
    print("  Un criterio que funciona en entrenamiento y falla en")
    print("  validación está sobreajustado. Solo los que funcionan en")
    print("  AMBOS merecen consideración — y aun así con muestra corta.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analiza un CSV de backtest y prueba filtros alternativos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv", help="CSV exportado por backtest_soccer.py")
    parser.add_argument("--market", default=None,
                        help="Limitar a un mercado (1X2, TOTAL)")
    parser.add_argument("--split", type=int, default=None,
                        help="Temporada a reservar como validación")
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: no existe {args.csv}")
        return 1

    picks = _load(args.csv)
    if not picks:
        print(f"ERROR: {args.csv} no contiene picks legibles")
        return 1

    if args.market:
        picks = [p for p in picks if p.market.upper() == args.market.upper()]
        if not picks:
            print(f"ERROR: sin picks del mercado {args.market}")
            return 1

    base = _evaluate(picks, "todos", lambda p: True)

    print("=" * 70)
    print("  ANÁLISIS DEL BACKTEST")
    print("=" * 70)
    print(f"  Picks     : {base.n}")
    if args.market:
        print(f"  Mercado   : {args.market.upper()}")
    print(f"  ROI       : {base.roi:+.2f}%" if base.roi is not None else "")
    print(f"  Excedente : {base.surplus:+.1f} victorias sobre lo implícito")

    _price_bands(picks)
    _ev_bands(picks)
    probadas = _alternatives(picks)

    if args.split:
        _split_validation(picks, args.split)

    print()
    print("=" * 70)
    print("  ADVERTENCIA DE SOBREAJUSTE")
    print("=" * 70)
    print()
    print(f"  Se probaron {probadas} configuraciones sobre los mismos datos.")
    print("  Con ese número, alguna saldrá bien por azar aunque ninguna")
    print("  tenga ventaja real: es el problema de las comparaciones")
    print("  múltiples.")
    print()
    print("  Un criterio solo merece consideración si:")
    print("    1. Funciona fuera de muestra (--split)")
    print("    2. Tiene una razón MECÁNICA, no solo estadística")
    print("    3. Mantiene muestra suficiente (n ≥ 150)")
    print()
    print("  El EV escalado por precio cumple el punto 2: corrige una")
    print("  propiedad conocida del EV —su sensibilidad al precio— en vez")
    print("  de buscar el número que mejor encaja con estos datos.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())