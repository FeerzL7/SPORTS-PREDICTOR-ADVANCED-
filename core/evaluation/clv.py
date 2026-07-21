"""
core/evaluation/clv.py

CLVTracker: análisis de Closing Line Value (CLV) para detección
rápida de edge del modelo.

Por qué CLV es la métrica de primera clase
--------------------------------------------
El audit del sistema MLB identificó CLV como la métrica más importante
ausente. Con ROI en muestras pequeñas (50-100 picks), el intervalo de
confianza es muy amplio — incluso con edge real, puede mostrar ROI
negativo por varianza. CLV positivo en 20-30 picks es una señal
estadísticamente más fuerte de edge que ROI en 100 picks.

La razón: el mercado de cierre es el mejor estimador disponible de
la probabilidad real del evento. Si el sistema apostó consistentemente
a mejores precios que el cierre, tiene ventaja informacional sobre el
mercado — independientemente del resultado final de cada pick.

Fórmula adoptada (estándar de la industria)
---------------------------------------------
    CLV = (pick_price - closing_price) / closing_price × 100

    CLV > 0: el sistema apostó a mejor precio que el cierre.
             El mercado se movió en la dirección del pick después
             de la apuesta — señal de edge real.

    CLV < 0: el mercado se movió en contra del pick.
             La cuota bajó antes del cierre — señal de que el
             pick apostó contra la acción sharp.

    CLV = 0: el precio no se movió — sin señal de valor capturado.

Nota sobre el spec (reconciliación de fórmulas)
-------------------------------------------------
El spec de la arquitectura indica:
    CLV = (closing_price / opening_price - 1) × 100

Esta fórmula es equivalente a la estándar cuando pick_price == opening_price
(precio de apertura == precio al momento del pick), que es el caso más
común. Sin embargo, si el precio cambió entre la apertura del mercado y
el momento del pick, las fórmulas difieren. Se adopta la fórmula estándar
de la industria (Buchdahl, Joseph 2016) porque:
1. Tiene interpretación directa: "capturé X% de valor sobre el cierre"
2. Es consistente con SettlementResult.clv() ya implementado
3. Es la más usada en literatura académica de betting

Workflow completo
------------------
Stage 5  (Odds Ingestion):
    CLVTracker.record_opening_line(entry_id, opening_price, market, sport)
    → Registra el precio de apertura para calcular CLV después.

Stage 10 (Settlement):
    SettlementProvider.get_closing_price(entry) → closing_price
    BankrollTracker.update_clv(entry_id, clv)
    → CLV persiste en el ledger.

Análisis (post-settlement):
    CLVTracker.analyze(entries) → CLVReport
    → Agrega CLV individual en métricas estadísticas de edge.

Uso típico
-----------
    from core.evaluation.clv import CLVTracker

    tracker = CLVTracker()

    # Al registrar el pick (Stage 5)
    tracker.record_opening_line('e1_ML_home', 2.10, 'ML', 'mlb')

    # Después del settlement con closing_price disponible
    clv = tracker.calculate_clv(pick_price=2.10, closing_price=1.95)
    # clv = (2.10 - 1.95) / 1.95 × 100 = +7.69%

    # Análisis agregado
    entries = bankroll_tracker._store.load_all()
    report  = tracker.analyze(entries)
    print(report.summary())
"""

from __future__ import annotations

import math
import os
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from core.contracts.ledger import BetLedgerEntry


# ── Registro de línea de apertura ─────────────────────────────────────────────

@dataclass
class OpeningLineRecord:
    """
    Registro de la línea de apertura de un pick.

    Mutable: se actualiza cuando el closing_price llega.

    Campos
    ------
    entry_id       -- ID del BetLedgerEntry correspondiente.
    opening_price  -- Cuota al momento de registrar el pick.
    market         -- Mercado del pick ('ML', 'TOTAL', etc.).
    sport          -- Deporte del pick.
    recorded_at    -- Timestamp ISO-8601 del registro.
    closing_price  -- Cuota de cierre. None hasta que esté disponible.
    clv            -- CLV calculado. None hasta que closing_price exista.
    """
    entry_id:      str
    opening_price: float
    market:        str
    sport:         str
    recorded_at:   str
    closing_price: float | None = None
    clv:           float | None = None


# ── Reporte de CLV ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CLVReport:
    """
    Análisis agregado de CLV para un conjunto de picks.

    Inmutable: snapshot de métricas en un instante.

    Campos estándar
    ----------------
    n_total          -- Total de picks en el ledger input.
    n_with_clv       -- Picks que tienen CLV calculado (closing_price
                       disponible). Es el denominador de todas las
                       métricas de CLV.
    n_positive_clv   -- Picks con CLV > 0 (apostaron a mejor precio
                       que el cierre — señal de edge capturado).
    n_negative_clv   -- Picks con CLV < 0 (el mercado se movió en contra).
    clv_mean         -- Media de CLV. El indicador principal de edge.
                       Target: clv_mean > 0 de forma consistente.
    clv_std          -- Desviación estándar de CLV.
    clv_min          -- CLV mínimo observado (peor pick individual).
    clv_max          -- CLV máximo observado (mejor pick individual).
    positive_rate    -- n_positive_clv / n_with_clv. % de picks con
                       edge positivo capturado.
    edge_significance -- t-estadístico para H0: clv_mean = 0.
                        Valores |t| > 2.0 indican edge estadísticamente
                        significativo al 95% de confianza.
    is_significant   -- True si |edge_significance| > 2.0 y n_with_clv >= 20.
                       Condición mínima para afirmar edge real.
    by_sport         -- CLV medio por deporte {sport: clv_mean}.
    by_market        -- CLV medio por mercado {market: clv_mean}.
    label            -- Etiqueta del conjunto analizado.
    """
    n_total:           int
    n_with_clv:        int
    n_positive_clv:    int
    n_negative_clv:    int
    clv_mean:          float | None
    clv_std:           float | None
    clv_min:           float | None
    clv_max:           float | None
    positive_rate:     float | None
    edge_significance: float | None
    is_significant:    bool
    by_sport:          dict[str, float]
    by_market:         dict[str, float]
    label:             str

    def summary(self) -> str:
        """Resumen compacto para logging y Telegram."""
        if self.n_with_clv == 0:
            return f"[{self.label}] CLV: sin datos de cierre disponibles"
        mean  = f"{self.clv_mean:+.2f}%" if self.clv_mean is not None else "N/A"
        sig   = " ✓ SIGNIFICATIVO" if self.is_significant else ""
        rate  = f"{self.positive_rate:.1%}" if self.positive_rate is not None else "N/A"
        return (
            f"[{self.label}] CLV: mean={mean} | "
            f"positivos={rate} | n={self.n_with_clv}{sig}"
        )

    def has_positive_edge(self, min_picks: int = 20) -> bool:
        """
        True si hay evidencia de edge positivo.

        Condición: CLV medio > 0, significativo estadísticamente
        y con muestra suficiente.
        """
        if self.n_with_clv < min_picks:
            return False
        if self.clv_mean is None or self.clv_mean <= 0:
            return False
        return self.is_significant


# ── Motor principal ───────────────────────────────────────────────────────────

class CLVTracker:
    """
    Tracker de Closing Line Value para detección de edge del modelo.

    Dos responsabilidades:
    1. Registro de opening lines (Estado en memoria — persiste vía
       BankrollTracker.update_clv(), no por sí mismo).
    2. Análisis agregado de CLV sobre BetLedgerEntry ya liquidados.

    Parámetros
    ----------
    significance_threshold  -- |t| mínimo para considerar CLV significativo.
                              Default 2.0 (≈ 95% confianza bilateral).
    min_picks_significant   -- Mínimo de picks con CLV para calcular
                              significancia. Con n < 20 el t-test es
                              poco fiable. Default 20.
    """

    def __init__(
        self,
        significance_threshold: float = 2.0,
        min_picks_significant:  int   = 20,
    ) -> None:
        self._sig_threshold  = significance_threshold
        self._min_picks_sig  = min_picks_significant
        # Registro en memoria de opening lines pendientes de closing
        self._opening_lines: dict[str, OpeningLineRecord] = {}

    # ── Registro de líneas ─────────────────────────────────────────────────────

    def record_opening_line(
        self,
        entry_id:      str,
        opening_price: float,
        market:        str,
        sport:         str,
        recorded_at:   str | None = None,
    ) -> OpeningLineRecord:
        """
        Registra el precio de apertura de un pick para CLV posterior.

        Llamar inmediatamente después de BankrollTracker.register()
        en Stage 5 del pipeline.

        Parámetros
        ----------
        entry_id       -- ID del BetLedgerEntry.
        opening_price  -- Cuota al momento del pick (≡ BetLedgerEntry.price).
        market         -- Mercado del pick.
        sport          -- Deporte del pick.
        recorded_at    -- Timestamp ISO-8601. Default: ahora UTC.
        """
        from datetime import datetime, timezone
        ts     = recorded_at or datetime.now(timezone.utc).isoformat()
        record = OpeningLineRecord(
            entry_id      = entry_id,
            opening_price = opening_price,
            market        = market,
            sport         = sport,
            recorded_at   = ts,
        )
        self._opening_lines[entry_id] = record
        return record

    def record_closing_price(
        self,
        entry_id:      str,
        closing_price: float,
    ) -> float | None:
        """
        Registra el precio de cierre y calcula CLV para un pick.

        Parámetros
        ----------
        entry_id       -- ID del pick.
        closing_price  -- Cuota de cierre del mercado.

        Retorna
        -------
        CLV calculado, o None si no hay registro de opening line
        para este entry_id (pick registrado antes del CLVTracker).
        """
        record = self._opening_lines.get(entry_id)
        if record is None:
            return None

        clv = self.calculate_clv(
            pick_price    = record.opening_price,
            closing_price = closing_price,
        )
        record.closing_price = closing_price
        record.clv           = clv
        return clv

    # ── Cálculo de CLV individual ─────────────────────────────────────────────

    @staticmethod
    def calculate_clv(
        pick_price:    float,
        closing_price: float,
    ) -> float:
        """
        Calcula el CLV para un pick individual.

            CLV = (pick_price - closing_price) / closing_price × 100

        CLV > 0: el sistema apostó a mejor precio que el cierre.
        CLV < 0: el mercado se movió en contra.

        Parámetros
        ----------
        pick_price     -- Cuota al momento del pick.
        closing_price  -- Cuota de cierre del mercado justo antes del evento.

        Raises
        ------
        ValueError -- Si closing_price <= 1.0 (cuota inválida).
        """
        if closing_price <= 1.0:
            raise ValueError(
                f"closing_price={closing_price} debe ser > 1.0 (cuota decimal)."
            )
        if pick_price <= 1.0:
            raise ValueError(
                f"pick_price={pick_price} debe ser > 1.0 (cuota decimal)."
            )
        return round(
            (pick_price - closing_price) / closing_price * 100,
            4,
        )

    # ── Análisis agregado ─────────────────────────────────────────────────────

    def analyze(
        self,
        entries: Sequence[BetLedgerEntry],
        label:   str = "all",
    ) -> CLVReport:
        """
        Analiza el CLV de un conjunto de BetLedgerEntry.

        Usa el campo `clv` del ledger — ya calculado por
        SettlementResult.clv() y persistido por update_clv().
        No recalcula CLV desde precios — solo agrega los valores
        ya almacenados.

        Parámetros
        ----------
        entries  -- Entradas del ledger. El caller puede filtrar por
                   sport, market, model_version o fecha antes de llamar.
        label    -- Etiqueta descriptiva para el reporte.

        Retorna
        -------
        CLVReport con métricas agregadas.
        """
        n_total    = len(entries)
        with_clv   = [e for e in entries if e.clv is not None]
        n_with_clv = len(with_clv)

        if n_with_clv == 0:
            return CLVReport(
                n_total=n_total, n_with_clv=0,
                n_positive_clv=0, n_negative_clv=0,
                clv_mean=None, clv_std=None,
                clv_min=None, clv_max=None,
                positive_rate=None, edge_significance=None,
                is_significant=False,
                by_sport={}, by_market={}, label=label,
            )

        clv_values   = [e.clv for e in with_clv]  # type: ignore[misc]
        n_positive   = sum(1 for v in clv_values if v > 0) # type: ignore
        n_negative   = sum(1 for v in clv_values if v < 0) # type: ignore
        clv_mean     = sum(clv_values) / n_with_clv # type: ignore
        clv_std      = _std(clv_values, clv_mean) # type: ignore
        clv_min      = min(clv_values) # type: ignore
        clv_max      = max(clv_values) # type: ignore
        positive_rate= n_positive / n_with_clv

        # t-estadístico para H0: clv_mean = 0
        t_stat: float | None = None
        if clv_std is not None and clv_std > 0 and n_with_clv >= self._min_picks_sig:
            t_stat = clv_mean / (clv_std / math.sqrt(n_with_clv))

        is_significant = (
            t_stat is not None
            and abs(t_stat) >= self._sig_threshold
            and n_with_clv >= self._min_picks_sig
        )

        # Desglose por sport y market
        by_sport  = self._mean_by_field(with_clv, "sport")
        by_market = self._mean_by_field(with_clv, "market")

        return CLVReport(
            n_total          = n_total,
            n_with_clv       = n_with_clv,
            n_positive_clv   = n_positive,
            n_negative_clv   = n_negative,
            clv_mean         = round(clv_mean, 4),
            clv_std          = round(clv_std, 4) if clv_std is not None else None,
            clv_min          = round(clv_min, 4),
            clv_max          = round(clv_max, 4),
            positive_rate    = round(positive_rate, 4),
            edge_significance= round(t_stat, 4) if t_stat is not None else None,
            is_significant   = is_significant,
            by_sport         = by_sport,
            by_market        = by_market,
            label            = label,
        )

    def compare(
        self,
        report_a: CLVReport,
        report_b: CLVReport,
    ) -> dict:
        """
        Compara dos CLVReport (ej. v1.0 vs v2.0, o mlb vs nba).

        Retorna dict con deltas y si la mejora es significativa.
        """
        def delta(va, vb):
            if va is None or vb is None:
                return None
            return round(vb - va, 4)

        mean_d = delta(report_a.clv_mean, report_b.clv_mean)
        return {
            "label_a":         report_a.label,
            "label_b":         report_b.label,
            "n_a":             report_a.n_with_clv,
            "n_b":             report_b.n_with_clv,
            "clv_mean_a":      report_a.clv_mean,
            "clv_mean_b":      report_b.clv_mean,
            "clv_mean_delta":  mean_d,
            "clv_improved":    mean_d > 0 if mean_d is not None else None,
            "sig_a":           report_a.is_significant,
            "sig_b":           report_b.is_significant,
            "positive_rate_a": report_a.positive_rate,
            "positive_rate_b": report_b.positive_rate,
            "sufficient_data": (
                report_a.n_with_clv >= 20 and report_b.n_with_clv >= 20
            ),
        }

    # ── Exportación ────────────────────────────────────────────────────────────

    def export_csv(
        self,
        entries: Sequence[BetLedgerEntry],
        path:    str,
    ) -> None:
        """
        Exporta el análisis CLV por pick a CSV.

        Columnas: date, event, market, sport, price, closing_price
        (si disponible en entry), clv, result.
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        fields = [
            "date", "event", "market", "selection", "sport",
            "price", "clv", "result", "model_version",
        ]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for e in entries:
                writer.writerow({
                    "date":          e.date,
                    "event":         e.event,
                    "market":        e.market,
                    "selection":     e.selection,
                    "sport":         e.sport,
                    "price":         e.price,
                    "clv":           e.clv if e.clv is not None else "",
                    "result":        e.result,
                    "model_version": e.model_version,
                })

    # ── Helpers privados ───────────────────────────────────────────────────────

    @staticmethod
    def _mean_by_field(
        entries: list[BetLedgerEntry],
        field_name: str,
    ) -> dict[str, float]:
        """Calcula CLV medio agrupado por un campo del ledger."""
        groups: dict[str, list[float]] = defaultdict(list)
        for e in entries:
            if e.clv is None:
                continue
            key = getattr(e, field_name, "unknown") or "unknown"
            groups[key].append(e.clv)
        return {
            k: round(sum(v) / len(v), 4)
            for k, v in groups.items()
            if v
        }


# ── Utilidades matemáticas ────────────────────────────────────────────────────

def _std(values: list[float], mean: float) -> float | None:
    """Desviación estándar muestral (ddof=1)."""
    n = len(values)
    if n < 2:
        return None
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)