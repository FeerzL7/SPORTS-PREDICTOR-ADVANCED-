"""
core/backtesting/engine.py

BacktestEngine: orquestador de análisis histórico multi-deporte.

Migrado de backtesting/backtesting.py del sistema MLB con seis mejoras
documentadas en SPORTS_PREDICTOR_ARCHITECTURE.md §9.1:

1. Cross-sport backtesting con filtro por sport/market/fecha.
   El sistema MLB no tenía columna sport — mezclaba todo.

2. Segmentación multi-dimensión: por deporte, mercado, EV band,
   cuota, mes, model_version.

3. Sensitivity analysis automático: sweep de umbrales EV/prob/edge
   para encontrar el óptimo calibrado por backtesting.

4. Model version comparison: ROI de v1.0 vs v2.0 en el mismo período.

5. Bootstrap confidence intervals: error estándar de ROI y hit_rate.

6. Hypothesis validation: integra HypothesisTracker para validar
   H1..HN automáticamente cuando hay suficiente muestra.

BacktestEngine NO reimplementa lógica de métricas — orquesta:
    BankrollTracker.metrics()    → métricas por segmento
    CalibrationEngine.calculate() → curva de calibración
    CLVTracker.analyze()          → CLV por segmento
    HypothesisTracker.evaluate_all() → validación de hipótesis

Uso típico
-----------
    from core.backtesting.engine import BacktestEngine

    engine = BacktestEngine(tracker=bankroll_tracker)

    # Análisis completo MLB
    report = engine.run(
        sport='mlb',
        date_from='2026-01-01',
        date_to='2026-07-15',
    )
    engine.export(report, 'output/backtest/')
    print(report.summary())

    # Sensitivity analysis sobre min_ev
    sensitivity = engine.sensitivity_analysis(
        sport='mlb', market='TOTAL',
        param='min_ev', values=[3,6,9,12,15,18,21,24,27,30],
    )

    # Comparar versiones
    comparison = engine.compare_versions(
        sport='mlb', v1='mlb-v1.0', v2='mlb-v2.0',
    )
"""

from __future__ import annotations

import csv
import math
import os
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Sequence

from core.bankroll.tracker import BankrollTracker
from core.contracts.ledger import BetLedgerEntry
from core.evaluation.calibration import CalibrationEngine, CalibrationResult
from core.evaluation.clv import CLVTracker, CLVReport
from core.evaluation.metrics import HypothesisTracker


# ── Resultado de backtesting ──────────────────────────────────────────────────

@dataclass(frozen=True)
class SegmentResult:
    """
    Métricas para un segmento específico (sport, market, EV band, etc.).

    Inmutable: snapshot de métricas en el momento del cálculo.
    """
    label:           str
    n_picks:         int
    n_resolved:      int
    wins:            int
    losses:          int
    hit_rate:        float
    roi:             float
    profit:          float
    stake_total:     float
    avg_odds:        float
    avg_ev:          float
    avg_edge:        float
    clv_mean:        float | None
    brier_score:     float | None
    # Bootstrap confidence intervals (95%)
    roi_ci_low:      float | None = None
    roi_ci_high:     float | None = None
    hit_rate_ci_low: float | None = None
    hit_rate_ci_high: float | None = None


@dataclass
class BacktestReport:
    """
    Reporte completo de un backtesting.

    Mutable durante la construcción — congelado al retornar de engine.run().

    Campos
    ------
    sport          -- Deporte analizado ('mlb', 'nba', 'all').
    date_from      -- Inicio del período.
    date_to        -- Fin del período.
    overall        -- Métricas globales del período completo.
    by_market      -- Métricas por mercado.
    by_month       -- Métricas por mes YYYY-MM.
    by_ev_band     -- Métricas por banda de EV (0-5%, 5-10%, etc.).
    by_odds_band   -- Métricas por banda de cuota.
    by_model_version -- Métricas por versión del modelo.
    calibration    -- CalibrationResult del período.
    clv_report     -- CLVReport del período.
    hypotheses     -- Estado de hipótesis registradas.
    generated_at   -- Timestamp de generación.
    """
    sport:              str
    date_from:          str
    date_to:            str
    overall:            SegmentResult | None           = None
    by_market:          dict[str, SegmentResult]       = field(default_factory=dict)
    by_month:           dict[str, SegmentResult]       = field(default_factory=dict)
    by_ev_band:         dict[str, SegmentResult]       = field(default_factory=dict)
    by_odds_band:       dict[str, SegmentResult]       = field(default_factory=dict)
    by_model_version:   dict[str, SegmentResult]       = field(default_factory=dict)
    calibration:        CalibrationResult | None       = None
    clv_report:         CLVReport | None               = None
    hypotheses:         dict[str, str]                 = field(default_factory=dict)
    generated_at:       str                            = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def summary(self) -> str:
        """Resumen compacto para logging."""
        if self.overall is None:
            return f"[{self.sport}/{self.date_from}→{self.date_to}] Sin datos"
        o = self.overall
        cal = f" | BS={self.calibration.brier_score:.4f}" if self.calibration and self.calibration.brier_score else ""
        clv = f" | CLV={self.clv_report.clv_mean:+.2f}%" if self.clv_report and self.clv_report.clv_mean else ""
        return (
            f"[{self.sport}/{self.date_from}→{self.date_to}] "
            f"n={o.n_resolved} | HR={o.hit_rate:.1f}% | "
            f"ROI={o.roi:+.2f}%{cal}{clv}"
        )


@dataclass(frozen=True)
class SensitivityPoint:
    """Un punto en el análisis de sensibilidad."""
    param_value:  float
    n_picks:      int
    roi:          float
    hit_rate:     float
    profit:       float


@dataclass(frozen=True)
class ModelComparison:
    """Comparación entre dos versiones del modelo."""
    sport:          str
    version_a:      str
    version_b:      str
    result_a:       SegmentResult | None
    result_b:       SegmentResult | None
    roi_delta:      float | None
    hit_rate_delta: float | None
    clv_delta:      float | None
    improved:       bool | None

    def summary(self) -> str:
        if self.result_a is None or self.result_b is None:
            return f"Datos insuficientes para comparar {self.version_a} vs {self.version_b}"
        sign = "✓" if self.improved else "✗"
        return (
            f"[{sign}] {self.version_b} vs {self.version_a}: "
            f"ΔROI={self.roi_delta:+.2f}% "
            f"ΔHR={self.hit_rate_delta:+.2f}%"
        )


# ── Motor principal ───────────────────────────────────────────────────────────

class BacktestEngine:
    """
    Orquestador de análisis histórico multi-deporte.

    Parámetros
    ----------
    tracker          -- BankrollTracker con LedgerStore configurado.
    calibration_engine -- CalibrationEngine. Default: instancia nueva.
    clv_tracker      -- CLVTracker. Default: instancia nueva.
    hypothesis_tracker -- HypothesisTracker pre-configurado con
                         las hipótesis a validar. None = sin hipótesis.
    bootstrap_n      -- Número de iteraciones bootstrap para CI.
                       Default 1000. Reducir a 100 para tests rápidos.
    random_seed      -- Semilla para reproducibilidad del bootstrap.
    """

    def __init__(
        self,
        tracker:             BankrollTracker,
        calibration_engine:  CalibrationEngine | None = None,
        clv_tracker:         CLVTracker | None        = None,
        hypothesis_tracker:  HypothesisTracker | None = None,
        bootstrap_n:         int                      = 1000,
        random_seed:         int                      = 42,
    ) -> None:
        self._tracker            = tracker
        self._calibration_engine = calibration_engine or CalibrationEngine()
        self._clv_tracker        = clv_tracker or CLVTracker()
        self._hypothesis_tracker = hypothesis_tracker
        self._bootstrap_n        = bootstrap_n
        self._random_seed        = random_seed

    # ── Análisis principal ────────────────────────────────────────────────────

    def run(
        self,
        sport:         str   = "all",
        date_from:     str | None = None,
        date_to:       str | None = None,
        model_version: str | None = None,
    ) -> BacktestReport:
        """
        Ejecuta el backtesting completo para el período y filtros dados.

        Parámetros
        ----------
        sport          -- 'all' para todos los deportes, o 'mlb', 'nba', etc.
        date_from      -- Inicio del período 'YYYY-MM-DD'. None = sin límite.
        date_to        -- Fin del período 'YYYY-MM-DD'. None = hoy.
        model_version  -- Filtrar por versión del modelo. None = todas.

        Retorna
        -------
        BacktestReport con métricas por segmento y evaluación de hipótesis.
        """
        sport_filter = None if sport == "all" else sport
        entries = self._tracker._store.load_all()
        filtered = self._tracker._apply_filters(
            entries,
            sport         = sport_filter,
            date_from     = date_from,
            date_to       = date_to,
            model_version = model_version,
        )

        report = BacktestReport(
            sport     = sport,
            date_from = date_from or "inicio",
            date_to   = date_to   or "hoy",
        )

        if not filtered:
            return report

        # Overall
        report.overall = self._segment_result(filtered, label=sport)

        # Por mercado
        for market in sorted({e.market for e in filtered}):
            seg_entries = [e for e in filtered if e.market == market]
            report.by_market[market] = self._segment_result(
                seg_entries, label=market
            )

        # Por mes
        for month in sorted({e.date[:7] for e in filtered if len(e.date) >= 7}):
            seg_entries = [e for e in filtered if e.date[:7] == month]
            report.by_month[month] = self._segment_result(
                seg_entries, label=month
            )

        # Por banda de EV
        for band_label, (lo, hi) in _EV_BANDS.items():
            seg_entries = [
                e for e in filtered
                if lo <= e.ev < hi
            ]
            if seg_entries:
                report.by_ev_band[band_label] = self._segment_result(
                    seg_entries, label=band_label
                )

        # Por banda de cuota
        for band_label, (lo, hi) in _ODDS_BANDS.items():
            seg_entries = [
                e for e in filtered
                if lo <= e.price < hi
            ]
            if seg_entries:
                report.by_odds_band[band_label] = self._segment_result(
                    seg_entries, label=band_label
                )

        # Por model_version
        for ver in sorted({e.model_version for e in filtered}):
            seg_entries = [e for e in filtered if e.model_version == ver]
            report.by_model_version[ver] = self._segment_result(
                seg_entries, label=ver
            )

        # Calibración
        report.calibration = self._calibration_engine.calculate(
            filtered,
            label=f"{sport}/{date_from}→{date_to}",
        )

        # CLV
        report.clv_report = self._clv_tracker.analyze(
            filtered,
            label=f"{sport}/{date_from}→{date_to}",
        )

        # Hipótesis
        if self._hypothesis_tracker is not None:
            self._hypothesis_tracker.evaluate_all(filtered)
            report.hypotheses = self._hypothesis_tracker.summary()

        return report

    # ── Sensitivity analysis ──────────────────────────────────────────────────

    def sensitivity_analysis(
        self,
        param:     str,
        values:    list[float],
        sport:     str | None = None,
        market:    str | None = None,
        date_from: str | None = None,
        date_to:   str | None = None,
    ) -> list[SensitivityPoint]:
        """
        Analiza la sensibilidad de ROI y hit_rate a un umbral dado.

        Para cada valor del parámetro, filtra los picks que superan ese
        umbral y calcula las métricas. Permite encontrar el umbral óptimo
        de forma empírica en vez de manual (reemplaza el --calibrar de MLB).

        Parámetros soportados
        ----------------------
        'min_ev'    -- Filtrar picks con ev >= value
        'min_prob'  -- Filtrar picks con model_prob >= value
        'min_edge'  -- Filtrar picks con edge >= value
        'min_odds'  -- Filtrar picks con price >= value
        'max_odds'  -- Filtrar picks con price <= value

        Retorna
        -------
        list[SensitivityPoint] ordenada por param_value ascendente.
        """
        entries = self._tracker._store.load_all()
        base = self._tracker._apply_filters(
            entries, sport=sport, market=market,
            date_from=date_from, date_to=date_to,
        )
        base = [e for e in base if e.result in ("win", "lose")]

        results = []
        for v in sorted(values):
            filtered = self._apply_param_filter(base, param, v)
            if not filtered:
                continue
            wins   = sum(1 for e in filtered if e.result == "win")
            losses = sum(1 for e in filtered if e.result == "lose")
            n      = wins + losses
            stake  = sum(e.stake_amount for e in filtered)
            profit = sum(e.profit_amount for e in filtered if e.profit_amount is not None)
            roi    = profit / stake * 100 if stake else 0.0
            hr     = wins / n * 100 if n else 0.0
            results.append(SensitivityPoint(
                param_value = v,
                n_picks     = n,
                roi         = round(roi, 2),
                hit_rate    = round(hr, 2),
                profit      = round(profit, 2),
            ))
        return results

    # ── Comparación de versiones ──────────────────────────────────────────────

    def compare_versions(
        self,
        v1:        str,
        v2:        str,
        sport:     str | None = None,
        date_from: str | None = None,
        date_to:   str | None = None,
    ) -> ModelComparison:
        """
        Compara métricas de dos versiones del modelo.

        Parámetros
        ----------
        v1, v2  -- Valores de model_version en BetLedgerEntry.
                  Ej: 'mlb-v1.0' vs 'mlb-v2.0'.
        """
        entries = self._tracker._store.load_all()
        filtered = self._tracker._apply_filters(
            entries, sport=sport, date_from=date_from, date_to=date_to
        )

        e_v1 = [e for e in filtered if e.model_version == v1]
        e_v2 = [e for e in filtered if e.model_version == v2]

        r1 = self._segment_result(e_v1, label=v1) if e_v1 else None
        r2 = self._segment_result(e_v2, label=v2) if e_v2 else None

        roi_delta      = round(r2.roi - r1.roi, 2)         if r1 and r2 else None
        hr_delta       = round(r2.hit_rate - r1.hit_rate, 2) if r1 and r2 else None
        clv_a          = r1.clv_mean if r1 else None
        clv_b          = r2.clv_mean if r2 else None
        clv_delta      = round(clv_b - clv_a, 4) if clv_a is not None and clv_b is not None else None
        improved       = roi_delta > 0 if roi_delta is not None else None

        return ModelComparison(
            sport          = sport or "all",
            version_a      = v1,
            version_b      = v2,
            result_a       = r1,
            result_b       = r2,
            roi_delta      = roi_delta,
            hit_rate_delta = hr_delta,
            clv_delta      = clv_delta,
            improved       = improved,
        )

    # ── Exportación ───────────────────────────────────────────────────────────

    def export(self, report: BacktestReport, output_dir: str) -> list[str]:
        """
        Exporta el BacktestReport a CSV en output_dir.

        Archivos generados:
            {sport}_{date}_summary.csv
            {sport}_{date}_by_market.csv
            {sport}_{date}_sensitivity.csv
            {sport}_{date}_calibration.csv
            {sport}_{date}_hypotheses.csv
        """
        os.makedirs(output_dir, exist_ok=True)
        today  = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        prefix = f"{report.sport}_{today}"
        paths  = []

        # Summary
        p = os.path.join(output_dir, f"{prefix}_summary.csv")
        self._export_segment_csv([report.overall] if report.overall else [], p)
        paths.append(p)

        # By market
        p = os.path.join(output_dir, f"{prefix}_by_market.csv")
        self._export_segment_csv(list(report.by_market.values()), p)
        paths.append(p)

        # By month
        p = os.path.join(output_dir, f"{prefix}_by_month.csv")
        self._export_segment_csv(list(report.by_month.values()), p)
        paths.append(p)

        # Calibración
        if report.calibration:
            p = os.path.join(output_dir, f"{prefix}_calibration.csv")
            self._calibration_engine.export_csv(report.calibration, p)
            paths.append(p)

        # Hipótesis
        if report.hypotheses:
            p = os.path.join(output_dir, f"{prefix}_hypotheses.csv")
            self._export_hypotheses_csv(report.hypotheses, p)
            paths.append(p)

        return paths

    def export_sensitivity(
        self,
        points:     list[SensitivityPoint],
        path:       str,
        param_name: str = "param",
    ) -> None:
        """Exporta los resultados de sensitivity analysis a CSV."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fields = [param_name, "n_picks", "roi", "hit_rate", "profit"]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for pt in points:
                writer.writerow({
                    param_name: pt.param_value,
                    "n_picks":  pt.n_picks,
                    "roi":      pt.roi,
                    "hit_rate": pt.hit_rate,
                    "profit":   pt.profit,
                })

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _segment_result(
        self,
        entries: list[BetLedgerEntry],
        label:   str,
        with_ci: bool = True,
    ) -> SegmentResult:
        """
        Calcula métricas para un segmento de entries.
        Incluye bootstrap CI si with_ci=True y n >= 30.
        """
        resolved = [e for e in entries if e.result in ("win", "lose")]
        wins     = [e for e in resolved if e.result == "win"]
        losses   = [e for e in resolved if e.result == "lose"]
        n        = len(resolved)

        stake_total = sum(e.stake_amount for e in resolved)
        profit      = sum(
            e.profit_amount for e in resolved
            if e.profit_amount is not None
        )
        roi      = profit / stake_total * 100 if stake_total else 0.0
        hit_rate = len(wins) / n * 100 if n else 0.0

        prices = [e.price for e in resolved if e.price > 1.0]
        evs    = [e.ev for e in entries]
        edges  = [e.ev for e in entries]  # ev como proxy de edge disponible

        clv_vals = [e.clv for e in resolved if e.clv is not None]
        clv_mean = round(sum(clv_vals)/len(clv_vals), 4) if clv_vals else None

        brier_items = [
            (e.model_prob - (1.0 if e.result == "win" else 0.0)) ** 2
            for e in resolved
        ]
        brier = round(sum(brier_items)/len(brier_items), 6) if brier_items else None

        # Bootstrap CI
        roi_ci_low = roi_ci_high = hr_ci_low = hr_ci_high = None
        if with_ci and n >= 30:
            roi_boots, hr_boots = self._bootstrap(resolved)
            roi_ci_low, roi_ci_high = _percentile(roi_boots, 2.5), _percentile(roi_boots, 97.5)
            hr_ci_low, hr_ci_high   = _percentile(hr_boots, 2.5),  _percentile(hr_boots, 97.5)

        return SegmentResult(
            label            = label,
            n_picks          = len(entries),
            n_resolved       = n,
            wins             = len(wins),
            losses           = len(losses),
            hit_rate         = round(hit_rate, 2),
            roi              = round(roi, 2),
            profit           = round(profit, 2),
            stake_total      = round(stake_total, 2),
            avg_odds         = round(sum(prices)/len(prices), 4) if prices else 0.0,
            avg_ev           = round(sum(evs)/len(evs), 2) if evs else 0.0,
            avg_edge         = round(sum(edges)/len(edges), 4) if edges else 0.0,
            clv_mean         = clv_mean,
            brier_score      = brier,
            roi_ci_low       = round(roi_ci_low, 2) if roi_ci_low is not None else None,
            roi_ci_high      = round(roi_ci_high, 2) if roi_ci_high is not None else None,
            hit_rate_ci_low  = round(hr_ci_low, 2) if hr_ci_low is not None else None,
            hit_rate_ci_high = round(hr_ci_high, 2) if hr_ci_high is not None else None,
        )

    def _bootstrap(
        self,
        resolved: list[BetLedgerEntry],
    ) -> tuple[list[float], list[float]]:
        """
        Bootstrap de ROI y hit_rate para intervalos de confianza al 95%.

        Retorna (roi_samples, hit_rate_samples) de longitud bootstrap_n.
        """
        rng = random.Random(self._random_seed)
        n   = len(resolved)
        roi_boots: list[float] = []
        hr_boots:  list[float] = []

        for _ in range(self._bootstrap_n):
            sample  = [resolved[rng.randint(0, n-1)] for _ in range(n)]
            wins    = sum(1 for e in sample if e.result == "win")
            stake   = sum(e.stake_amount for e in sample)
            profit  = sum(
                e.profit_amount for e in sample
                if e.profit_amount is not None
            )
            roi_boots.append(profit / stake * 100 if stake else 0.0)
            hr_boots.append(wins / n * 100 if n else 0.0)

        return roi_boots, hr_boots

    @staticmethod
    def _apply_param_filter(
        entries: list[BetLedgerEntry],
        param:   str,
        value:   float,
    ) -> list[BetLedgerEntry]:
        """Aplica un filtro de umbral por parámetro para sensitivity analysis."""
        if param == "min_ev":
            return [e for e in entries if e.ev >= value]
        if param == "min_prob":
            return [e for e in entries if e.model_prob >= value]
        if param == "min_odds":
            return [e for e in entries if e.price >= value]
        if param == "max_odds":
            return [e for e in entries if e.price <= value]
        return entries

    @staticmethod
    def _export_segment_csv(
        segments: list[SegmentResult | None],
        path:     str,
    ) -> None:
        """Exporta segmentos a CSV."""
        segs = [s for s in segments if s is not None]
        if not segs:
            return
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fields = [
            "label", "n_picks", "n_resolved", "wins", "losses",
            "hit_rate", "roi", "profit", "stake_total",
            "avg_odds", "avg_ev", "clv_mean", "brier_score",
            "roi_ci_low", "roi_ci_high",
            "hit_rate_ci_low", "hit_rate_ci_high",
        ]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for s in segs:
                writer.writerow({
                    "label":            s.label,
                    "n_picks":          s.n_picks,
                    "n_resolved":       s.n_resolved,
                    "wins":             s.wins,
                    "losses":           s.losses,
                    "hit_rate":         s.hit_rate,
                    "roi":              s.roi,
                    "profit":           s.profit,
                    "stake_total":      s.stake_total,
                    "avg_odds":         s.avg_odds,
                    "avg_ev":           s.avg_ev,
                    "clv_mean":         s.clv_mean if s.clv_mean is not None else "",
                    "brier_score":      s.brier_score if s.brier_score is not None else "",
                    "roi_ci_low":       s.roi_ci_low if s.roi_ci_low is not None else "",
                    "roi_ci_high":      s.roi_ci_high if s.roi_ci_high is not None else "",
                    "hit_rate_ci_low":  s.hit_rate_ci_low if s.hit_rate_ci_low is not None else "",
                    "hit_rate_ci_high": s.hit_rate_ci_high if s.hit_rate_ci_high is not None else "",
                })

    @staticmethod
    def _export_hypotheses_csv(hypotheses: dict[str, str], path: str) -> None:
        """Exporta estado de hipótesis a CSV."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "status"])
            writer.writeheader()
            for h_id, status in sorted(hypotheses.items()):
                writer.writerow({"id": h_id, "status": status})


# ── Bandas de segmentación ────────────────────────────────────────────────────

_EV_BANDS: dict[str, tuple[float, float]] = {
    "EV 0-5%":   (0.0,  5.0),
    "EV 5-10%":  (5.0,  10.0),
    "EV 10-15%": (10.0, 15.0),
    "EV 15-20%": (15.0, 20.0),
    "EV 20-30%": (20.0, 30.0),
    "EV 30%+":   (30.0, 9999.0),
}

_ODDS_BANDS: dict[str, tuple[float, float]] = {
    "1.50-1.75": (1.50, 1.75),
    "1.75-1.90": (1.75, 1.90),
    "1.90-2.00": (1.90, 2.00),
    "2.00-2.20": (2.00, 2.20),
    "2.20-2.50": (2.20, 2.50),
    "2.50+":     (2.50, 9999.0),
}


# ── Utilidades estadísticas ───────────────────────────────────────────────────

def _percentile(data: list[float], pct: float) -> float:
    """Percentil simple sobre lista ordenada."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (len(sorted_data) - 1) * pct / 100
    lo  = int(idx)
    hi  = min(lo + 1, len(sorted_data) - 1)
    frac = idx - lo
    return sorted_data[lo] * (1 - frac) + sorted_data[hi] * frac