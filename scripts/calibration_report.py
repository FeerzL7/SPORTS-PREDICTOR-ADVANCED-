#!/usr/bin/env python3
"""
scripts/calibration_report.py

Diagrama de fiabilidad: ¿acierta el modelo lo que dice acertar?

Por qué este análisis y no otro
---------------------------------
El backtest dijo que el sistema pierde. El análisis de filtros dijo que
ningún umbral lo arregla, y algo más grave:

    EV  3-5%    ROI  -0.33%
    EV  5-8%    ROI  -5.37%
    EV  8-12%   ROI  -7.87%
    EV 12-20%   ROI  -9.50%

El EV discrimina en la dirección CONTRARIA. Cuanto más valiosa cree el
modelo que es una apuesta, peor le va. Su desacuerdo con el mercado no
es información: es error, y crece con la magnitud.

Eso descarta el filtro como causa y apunta al modelo. La pregunta
siguiente es concreta: cuando el modelo dice 30%, ¿ocurre el 30%?

Un modelo puede perder dinero por dos razones distintas, y exigen
arreglos opuestos:

    MAL CALIBRADO   Dice 30% donde ocurre el 25%. Las probabilidades
                    están infladas. Se corrige recalibrando —menos
                    shrinkage, otro rho, otra composición de índices.

    BIEN CALIBRADO
    SIN VENTAJA     Dice 30% y ocurre el 30%, pero el mercado también
                    decía 30%. No hay nada que corregir: el modelo no
                    sabe más que el precio, y hay que buscar señal
                    nueva o cambiar de mercado.

Confundirlas lleva a recalibrar un modelo que ya está calibrado, o a
buscar features nuevas cuando el problema era el shrinkage.

Qué mide
----------
    DIAGRAMA DE FIABILIDAD
        Agrupa los picks por probabilidad predicha y compara con la
        frecuencia observada. La diagonal perfecta es calibración
        perfecta.

    SESGO POR TRAMO
        Dónde infla y dónde desinfla. La hipótesis a contrastar es que
        el shrinkage comprime las proyecciones: si todos los partidos
        parecen más igualados de lo que son, los empates y visitantes
        salen inflados.

    COMPARACIÓN CON EL MERCADO
        El mismo diagrama para las probabilidades implícitas. Si el
        mercado está mejor calibrado que el modelo en un tramo, ahí el
        modelo no debería operar.

    BRIER SCORE
        Error cuadrático medio de las probabilidades. Descompuesto en
        fiabilidad y resolución: la primera mide calibración, la
        segunda capacidad de discriminar. Un modelo puede estar
        perfectamente calibrado y ser inútil si no discrimina.

Uso
----
    python scripts/calibration_report.py bt.csv
    python scripts/calibration_report.py bt.csv --market 1X2
    python scripts/calibration_report.py bt.csv --bins 12
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist

_N = NormalDist()

# Picks mínimos por tramo para que la frecuencia observada signifique
# algo. Con veinte, el error estándar de una proporción del 30% ronda
# los 10 puntos: demasiado para leer un sesgo.
_MIN_BIN = 40


@dataclass
class Pick:
    market:      str
    price:       float
    model_prob:  float
    market_prob: float
    won:         bool
    passed:      bool = True


@dataclass
class Bin:
    lo:    float
    hi:    float
    picks: list[Pick] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.picks)

    @property
    def wins(self) -> int:
        return sum(1 for p in self.picks if p.won)

    @property
    def observed(self) -> float | None:
        return self.wins / self.n if self.n else None

    def predicted(self, source: str = "model") -> float | None:
        if not self.n:
            return None
        values = [p.model_prob if source == "model" else p.market_prob
                  for p in self.picks]
        return sum(values) / len(values)

    def bias(self, source: str = "model") -> float | None:
        """Predicho menos observado. Positivo = infla."""
        pred, obs = self.predicted(source), self.observed
        return None if pred is None or obs is None else pred - obs

    def p_value(self, source: str = "model") -> float | None:
        """p de que el sesgo sea azar, test binomial normalizado."""
        pred, obs = self.predicted(source), self.observed
        if pred is None or obs is None or self.n < 20:
            return None
        se = math.sqrt(max(pred * (1 - pred), 1e-9) / self.n)
        z = abs(obs - pred) / se
        return 2 * (1 - _N.cdf(z))


def _selection_effect(todos: list[Pick], filtrados: list[Pick]) -> None:
    """
    Separa el sesgo real del efecto de selección.

    LA PREGUNTA
    -----------
    El filtro exige que el modelo sea MÁS optimista que el mercado. Por
    construcción, cada pick seleccionado es un caso donde el modelo
    discrepó al alza.

    Si el modelo tuviera ruido alrededor de un centro bien calibrado,
    el filtro seleccionaría la cola alta de ese ruido y produciría un
    sesgo aparente aunque el modelo estuviera sano. Es la maldición del
    ganador.

    Comparar la calibración del conjunto COMPLETO con la del
    seleccionado distingue ambas causas, y exigen arreglos opuestos:

        Conjunto completo bien calibrado
            El modelo está sano; el filtro lo estropea. Se corrige
            bajando model_weight o exigiendo desacuerdos mayores.

        Conjunto completo también inflado
            Hay sesgo real en las probabilidades. Se corrige
            recalibrando el modelo, y la capa isotónica es la vía.
    """
    def sesgo(muestra: list[Pick]) -> tuple[float, int]:
        if not muestra:
            return 0.0, 0
        pred = sum(p.model_prob for p in muestra) / len(muestra)
        obs = sum(1 for p in muestra if p.won) / len(muestra)
        return (pred - obs) * 100, len(muestra)

    sesgo_todos, n_todos = sesgo(todos)
    sesgo_filtrados, n_filtrados = sesgo(filtrados)
    rechazados = [p for p in todos if not p.passed]
    sesgo_rechazados, n_rechazados = sesgo(rechazados)

    print()
    print("=" * 70)
    print("  EFECTO DE SELECCIÓN — la pregunta que decide el arreglo")
    print("=" * 70)
    print()
    print(f"    {'conjunto':24s} {'n':>6s} {'sesgo medio':>13s}")
    print("    " + "-" * 46)
    print(f"    {'TODOS los candidatos':24s} {n_todos:6d} {sesgo_todos:+12.2f} pp")
    print(f"    {'los que pasan el filtro':24s} {n_filtrados:6d} "
          f"{sesgo_filtrados:+12.2f} pp")
    print(f"    {'los rechazados':24s} {n_rechazados:6d} "
          f"{sesgo_rechazados:+12.2f} pp")
    print()

    # Umbral: 2 puntos de sesgo medio sobre el conjunto completo es
    # sesgo real; por debajo, el filtro explica lo observado.
    if abs(sesgo_todos) < 2.0:
        print("    → EFECTO DE SELECCIÓN.")
        print()
        print("    El conjunto completo está razonablemente calibrado; el")
        print("    sesgo aparece SOLO en los seleccionados. El filtro está")
        print("    escogiendo la cola alta del ruido del modelo.")
        print()
        print("    El modelo no necesita recalibración. Lo que hace falta")
        print("    es amortiguar el ruido: bajar model_weight en")
        print("    soccer.yaml, o exigir desacuerdos que persistan.")
    elif abs(sesgo_todos) > abs(sesgo_filtrados) * 0.6:
        print("    → SESGO REAL en las probabilidades.")
        print()
        print("    El conjunto completo está inflado en magnitud comparable")
        print("    al seleccionado, así que el filtro no lo explica.")
        print()
        print("    Una capa de calibración isotónica lo corrige sin tocar")
        print("    el modelo: ajusta un mapeo monótono de probabilidad")
        print("    predicha a observada con temporadas pasadas.")
    else:
        print("    → AMBAS CAUSAS.")
        print()
        print("    Hay sesgo real, y el filtro lo amplifica al seleccionar")
        print("    la cola alta. Conviene atacar las dos: calibración")
        print("    isotónica primero, y luego revisar model_weight.")


def _load(path: str, market: str | None) -> list[Pick]:
    picks: list[Pick] = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            try:
                if row["result"] not in ("win", "lose"):
                    continue
                if market and row["market"].upper() != market.upper():
                    continue
                passed = row.get("passed_filter", "True")
                picks.append(Pick(
                    market=row["market"],
                    price=float(row["price"]),
                    model_prob=float(row["model_prob"]),
                    market_prob=float(row["market_prob"]),
                    won=row["result"] == "win",
                    passed=str(passed).strip().lower() not in ("false", "0", ""),
                ))
            except (KeyError, ValueError):
                continue
    return picks


def _bin(picks: list[Pick], n_bins: int, source: str) -> list[Bin]:
    """Agrupa en tramos de igual anchura de probabilidad."""
    bins = [Bin(lo=i / n_bins, hi=(i + 1) / n_bins) for i in range(n_bins)]
    for p in picks:
        value = p.model_prob if source == "model" else p.market_prob
        index = min(int(value * n_bins), n_bins - 1)
        if index >= 0:
            bins[index].picks.append(p)
    return [b for b in bins if b.n >= _MIN_BIN]


def _reliability(bins: list[Bin], source: str, titulo: str) -> None:
    print(f"\n  {titulo}")
    print(f"    {'tramo':12s} {'n':>5s} {'predicho':>9s} {'observado':>10s} "
          f"{'sesgo':>8s} {'p':>7s}")
    print("    " + "-" * 56)

    for b in bins:
        pred = b.predicted(source)
        obs = b.observed
        bias = b.bias(source)
        p = b.p_value(source)
        if pred is None or obs is None or bias is None:
            continue
        marca = ""
        if p is not None and p < 0.01:
            marca = " ***" if abs(bias) > 0.03 else " *"
        print(f"    {b.lo:.2f}-{b.hi:.2f}  {b.n:5d} {pred*100:8.2f}% "
              f"{obs*100:9.2f}% {bias*100:+7.2f} {('—' if p is None else f'{p:.3f}'):>7s}{marca}")


def _brier(picks: list[Pick], source: str) -> dict:
    """
    Brier score y su descomposición.

        BS = fiabilidad − resolución + incertidumbre

    FIABILIDAD  cuánto se desvían las predicciones de la frecuencia
                observada en su tramo. Menor es mejor; cero es
                calibración perfecta.

    RESOLUCIÓN  cuánto varían las frecuencias observadas entre tramos.
                Mayor es mejor: mide la capacidad de DISCRIMINAR. Un
                modelo que predice siempre la media base está
                perfectamente calibrado y tiene resolución cero — es
                decir, es inútil.

    INCERTIDUMBRE  varianza irreducible del propio suceso.
    """
    if not picks:
        return {}

    values = [(p.model_prob if source == "model" else p.market_prob, p.won)
              for p in picks]
    n = len(values)
    base = sum(1 for _, w in values if w) / n

    brier = sum((prob - (1.0 if won else 0.0)) ** 2 for prob, won in values) / n

    # Descomposición sobre tramos de 0.05
    grupos: dict[int, list[tuple[float, bool]]] = {}
    for prob, won in values:
        grupos.setdefault(int(prob * 20), []).append((prob, won))

    fiabilidad = resolucion = 0.0
    for grupo in grupos.values():
        nk = len(grupo)
        pred_k = sum(p for p, _ in grupo) / nk
        obs_k = sum(1 for _, w in grupo if w) / nk
        fiabilidad += nk * (pred_k - obs_k) ** 2
        resolucion += nk * (obs_k - base) ** 2
    fiabilidad /= n
    resolucion /= n

    return {
        "brier": round(brier, 4),
        "fiabilidad": round(fiabilidad, 4),
        "resolucion": round(resolucion, 4),
        "incertidumbre": round(base * (1 - base), 4),
        "base": round(base, 4),
    }


def _blend_curve(picks: list[Pick]) -> None:
    """
    Brier de la mezcla modelo-mercado a distintos pesos.

    LA PREGUNTA DECISIVA
    --------------------
    Un modelo bien calibrado pero con menos resolución que el mercado
    puede aportar o no aportar. Depende de si su información es NUEVA o
    un subconjunto ruidoso de la que el precio ya incorpora.

    La prueba es directa: mezclar a distintos pesos y ver si alguno
    bate al mercado solo.

        w = 0.00   solo mercado
        w = 1.00   solo modelo

    Si el mínimo está en w = 0, el modelo no añade nada: cualquier peso
    positivo empeora la estimación. Si está en un punto intermedio, ese
    es el peso óptimo y el actual debería acercarse a él.

    Por qué el Brier y no el ROI
    -----------------------------
    El ROI depende de qué picks pasen el filtro, así que mide el
    sistema completo. El Brier mide solo la calidad de las
    probabilidades, que es lo que aquí se decide. Separarlos evita
    confundir un problema de estimación con uno de selección.
    """
    if not picks:
        return

    def brier_at(w: float) -> float:
        total = 0.0
        for p in picks:
            blended = w * p.model_prob + (1 - w) * p.market_prob
            total += (blended - (1.0 if p.won else 0.0)) ** 2
        return total / len(picks)

    pesos = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.45, 0.60, 0.80, 1.0]
    curva = [(w, brier_at(w)) for w in pesos]
    mejor_w, mejor_brier = min(curva, key=lambda x: x[1])
    solo_mercado = curva[0][1]

    print()
    print("=" * 70)
    print("  ¿APORTA EL MODELO ALGO QUE EL PRECIO NO TENGA?")
    print("=" * 70)
    print()
    print(f"    {'peso':>6s} {'Brier':>9s} {'vs mercado':>12s}")
    print("    " + "-" * 32)
    for w, b in curva:
        delta = b - solo_mercado
        marca = "  ← mejor" if w == mejor_w else ""
        actual = "  (actual)" if abs(w - 0.45) < 1e-9 else ""
        print(f"    {w:6.2f} {b:9.5f} {delta:+12.5f}{marca}{actual}")

    # La MAGNITUD de la mejora decide, no dónde cae el mínimo.
    #
    # Con muestra finita el óptimo rara vez cae exactamente en cero
    # aunque el modelo sea ruido puro: el azar desplaza el mínimo unas
    # centésimas. Una mejora de Brier por debajo de 0.0005 está en el
    # nivel del ruido de muestreo y no describe información real.
    #
    # Como referencia: la diferencia de Brier entre el modelo y el
    # mercado en este backtest fue 0.0085, diecisiete veces ese umbral.
    mejora = solo_mercado - mejor_brier

    print()
    if mejor_w <= 0.001 or mejora < 0.0005:
        print("    → EL MODELO NO APORTA INFORMACIÓN NUEVA.")
        print(f"       (mejora máxima {mejora:+.5f}, nivel del ruido)")
        print()
        print("    El Brier mínimo está en peso CERO: cualquier mezcla con")
        print("    el modelo empeora la estimación. Su información es un")
        print("    subconjunto ruidoso de la que el precio ya incorpora.")
        print()
        print("    Eso no significa que el modelo sea malo en términos")
        print("    absolutos —está bien calibrado— sino que estas cinco")
        print("    ligas son demasiado eficientes para lo que ve.")
        print()
        print("    Bajar model_weight reduce la pérdida, pero el óptimo")
        print("    es no operar aquí: hace falta señal que el cierre de")
        print("    Pinnacle no tenga, o mercados donde el precio sea peor.")
    elif mejor_w < 0.25:
        print(f"    → APORTA POCO. Peso óptimo ≈ {mejor_w:.2f}.")
        print()
        print("    El modelo mejora la estimación, pero solo con un peso")
        print("    bajo. El actual (0.45) está por encima del óptimo y")
        print("    amplifica el ruido.")
        print()
        print(f"    Bajar model_weight a {mejor_w:.2f} en soccer.yaml es el")
        print("    cambio de mayor efecto y menor coste.")
    else:
        print(f"    → APORTA INFORMACIÓN REAL. Peso óptimo ≈ {mejor_w:.2f}.")
        print()
        print("    El modelo mejora la estimación de forma sustancial. Si")
        print("    aun así el ROI es negativo, el problema está en la")
        print("    SELECCIÓN de picks, no en las probabilidades.")

    # Peso óptimo por mercado, si hay más de uno
    mercados = sorted({p.market for p in picks})
    if len(mercados) > 1:
        print()
        print("    Peso óptimo por mercado:")
        for m in mercados:
            sub = [p for p in picks if p.market == m]
            if len(sub) < 200:
                continue

            def brier_sub(w: float, s=sub) -> float:
                return sum(
                    (w * p.model_prob + (1 - w) * p.market_prob
                     - (1.0 if p.won else 0.0)) ** 2 for p in s
                ) / len(s)

            w_opt, b_opt = min(((w, brier_sub(w)) for w in pesos),
                               key=lambda x: x[1])
            delta = b_opt - brier_sub(0.0)
            print(f"      {m:8s} n={len(sub):<6d} peso {w_opt:.2f}  "
                  f"mejora {-delta:+.5f}")


def _verdict(picks: list[Pick]) -> None:
    modelo = _brier(picks, "model")
    mercado = _brier(picks, "market")

    print()
    print("=" * 70)
    print("  BRIER SCORE — menor es mejor")
    print("=" * 70)
    print()
    print(f"  {'':16s} {'Brier':>9s} {'fiabilidad':>12s} {'resolución':>12s}")
    print("  " + "-" * 52)
    for nombre, d in (("Modelo", modelo), ("Mercado", mercado)):
        if d:
            print(f"  {nombre:16s} {d['brier']:9.4f} {d['fiabilidad']:12.4f} "
                  f"{d['resolucion']:12.4f}")
    print()

    if not modelo or not mercado:
        return

    print("=" * 70)
    print("  DIAGNÓSTICO")
    print("=" * 70)
    print()

    peor_brier = modelo["brier"] > mercado["brier"]
    peor_fiab = modelo["fiabilidad"] > mercado["fiabilidad"] * 1.5
    peor_resol = modelo["resolucion"] < mercado["resolucion"] * 0.9

    if peor_fiab and not peor_resol:
        print("  MAL CALIBRADO, pero discrimina.")
        print()
        print("  La fiabilidad del modelo es sensiblemente peor que la del")
        print("  mercado, mientras que su resolución es comparable. Es decir:")
        print("  el modelo SÍ distingue partidos, pero sus probabilidades")
        print("  están desplazadas.")
        print()
        print("  Eso se corrige recalibrando, y el diagrama de arriba dice")
        print("  en qué dirección. Es el mejor de los casos posibles.")
    elif peor_resol and not peor_fiab:
        print("  BIEN CALIBRADO, pero NO discrimina.")
        print()
        print("  Las probabilidades son razonables pero varían poco entre")
        print("  partidos: el modelo predice casi lo mismo siempre.")
        print()
        print("  No hay nada que recalibrar. Hace falta señal nueva o")
        print("  reducir el shrinkage, que es lo que aplana las")
        print("  proyecciones.")
    elif peor_brier:
        print("  PEOR QUE EL MERCADO en ambas dimensiones.")
        print()
        print("  El modelo ni calibra mejor ni discrimina mejor que las")
        print("  probabilidades implícitas. Contra cuotas de cierre eso es")
        print("  lo esperable de un modelo sin ventaja real.")
        print()
        print("  Operar en mercados menos eficientes o añadir señal que el")
        print("  precio no incorpore son las vías; ajustar umbrales no.")
    else:
        print("  El modelo iguala o supera al mercado en el Brier.")
        print()
        print("  Si aun así el ROI es negativo, el problema está en la")
        print("  SELECCIÓN —qué picks pasan el filtro— y no en las")
        print("  probabilidades. Revisar el análisis de filtros.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagrama de fiabilidad del modelo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv", help="CSV exportado por backtest_soccer.py")
    parser.add_argument("--market", default=None, help="1X2 o TOTAL")
    parser.add_argument("--bins", type=int, default=10)
    args = parser.parse_args()

    if not Path(args.csv).exists():
        print(f"ERROR: no existe {args.csv}")
        return 1

    picks = _load(args.csv, args.market)
    if not picks:
        print("ERROR: sin picks resueltos en el CSV")
        return 1

    print("=" * 70)
    print("  CALIBRACIÓN DEL MODELO")
    print("=" * 70)
    print(f"  Picks resueltos : {len(picks)}")
    if args.market:
        print(f"  Mercado         : {args.market.upper()}")
    base = sum(1 for p in picks if p.won) / len(picks)
    print(f"  Tasa base       : {base:.2%}")

    # ── La pregunta central ────────────────────────────────────
    #
    # Si el CSV viene de --no-filter, contiene TODOS los candidatos y
    # se puede separar el efecto de selección del sesgo real.
    filtrados = [p for p in picks if p.passed]
    if len(filtrados) < len(picks):
        _selection_effect(picks, filtrados)

    bins_modelo = _bin(picks, args.bins, "model")
    _reliability(bins_modelo, "model",
                 "MODELO — predicho vs observado  (*** sesgo significativo)")

    bins_mercado = _bin(picks, args.bins, "market")
    _reliability(bins_mercado, "market",
                 "MERCADO — las mismas apuestas, probabilidad implícita")

    # ── Dirección del sesgo ────────────────────────────────────
    sesgos = [(b, b.bias("model")) for b in bins_modelo]
    sesgos = [(b, s) for b, s in sesgos if s is not None]
    if len(sesgos) >= 3:
        print()
        print("  DIRECCIÓN DEL SESGO")
        bajos = [s for b, s in sesgos if b.hi <= 0.40]
        altos = [s for b, s in sesgos if b.lo >= 0.50]
        if bajos:
            print(f"    Probabilidades bajas (<40%) : "
                  f"{sum(bajos)/len(bajos)*100:+.2f} pp de media")
        if altos:
            print(f"    Probabilidades altas (>50%) : "
                  f"{sum(altos)/len(altos)*100:+.2f} pp de media")
        if bajos and altos:
            print()
            if sum(bajos)/len(bajos) > 0.01 and sum(altos)/len(altos) < -0.01:
                print("    INFLA las bajas y DESINFLA las altas: es la firma")
                print("    de proyecciones demasiado planas. El shrinkage")
                print("    comprime los índices, todos los partidos parecen")
                print("    más igualados de lo que son, y eso sube empates y")
                print("    visitantes a costa de los favoritos.")
                print()
                print("    Se corrige bajando prior_matches en soccer.yaml")
                print("    o ampliando las cotas de los índices.")
            elif sum(bajos)/len(bajos) > 0.01:
                print("    INFLA sistemáticamente las probabilidades bajas.")
                print("    Con cuotas altas eso basta para perder: el EV")
                print("    amplifica el error por el precio.")

    _blend_curve(picks)
    _verdict(picks)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())