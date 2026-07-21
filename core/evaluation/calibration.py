"""
core/evaluation/calibration.py

CalibrationEngine: métricas de calibración del modelo predictivo.

Instrumentos de diagnóstico
-----------------------------
Brier Score (BS):
    BS = mean((p_predicted - outcome)²)
    Perfecto: 0.0 | Random (p=0.5 siempre): 0.25 | Peor posible: 1.0
    Target: BS < 0.24 para tener edge sobre random.
    Interpretación: mide calibración + sharpness juntos.

Log-Loss (LL):
    LL = -mean(y×log(p) + (1-y)×log(1-p))
    Penaliza más las predicciones confiadas incorrectas que las
    inciertas incorrectas. Un modelo que dice 95% y pierde recibe
    mayor penalización que uno que dice 60% y pierde.
    Valores típicos: LL < 0.65 indica modelo útil (random ≈ 0.693).

Calibration Curve:
    Para cada bin de probabilidad [0.50-0.54, 0.55-0.59, ...]:
        hit_rate_real = wins / picks en ese bin
    Modelo perfectamente calibrado: prob_estimada ≈ hit_rate_real.
    Sistema MLB tenía desviación sistemática positiva (overconfianza).

Overconfidence Index (OCI):
    OCI = mean(prob_estimada) - mean(hit_rate_real)
    Sistema MLB audit: OCI ≈ 0.16 en ML (estimaba 55%, realidad 39%).
    Target: |OCI| < 0.03 (máximo 3% de sesgo sistemático).
    OCI > 0: modelo sobreconfiado.
    OCI < 0: modelo subconfiado (infrecuente en la práctica).

Motivación histórica
----------------------
El audit del sistema MLB identificó OCI ≈ 0.16 como causa raíz del
ROI=-21.56% en ML: el modelo creía tener ventaja donde no la había.
Los picks pasaban los filtros de EV porque el EV estaba calculado
con probabilidades infladas. CalibrationEngine permite detectar y
cuantificar este fenómeno en producción continua y por versión del
modelo — sin este instrumento, el problema puede persistir durante
meses sin diagnóstico claro.

Uso típico
-----------
    from core.evaluation.calibration import CalibrationEngine
    from core.bankroll.tracker import BankrollTracker

    engine  = CalibrationEngine()
    tracker = BankrollTracker(store=CsvLedgerStore())
    entries = tracker._store.load_all()

    # Calibración completa
    result = engine.calculate(entries)
    print(f"Brier Score: {result.brier_score}")
    print(f"OCI: {result.oci}")

    # Por deporte
    mlb_entries = [e for e in entries if e.sport == 'mlb']
    mlb_result  = engine.calculate(mlb_entries, label='mlb')

    # Comparar versiones
    v1 = engine.calculate([e for e in entries if e.model_version == 'v1.0'])
    v2 = engine.calculate([e for e in entries if e.model_version == 'v2.0'])
    diff = engine.compare(v1, v2)
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from core.contracts.ledger import BetLedgerEntry


# ── Constantes ─────────────────────────────────────────────────────────────────

# Epsilon para clamp numérico en log-loss — evita log(0) = -inf.
# Valor estándar de sklearn y scipy: 1e-7.
_LOG_LOSS_EPS: float = 1e-7

# Tamaño mínimo de bin para considerarlo estadísticamente fiable.
_DEFAULT_MIN_BIN_SIZE: int = 10

# Ancho de cada bin de calibración en probabilidad.
_DEFAULT_BIN_WIDTH: float = 0.05  # bins de 5pp: [0.50-0.55), [0.55-0.60), ...

# Rango de probabilidades de picks esperados (el sistema filtra picks < 0.50)
_BIN_START: float = 0.50
_BIN_END:   float = 1.00


# ── Punto de calibración ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationPoint:
    """
    Un punto de la curva de calibración: métricas para un bin de probabilidad.

    Inmutable: representa un hecho histórico calculado sobre los picks
    en un rango de probabilidad específico.

    Campos
    ------
    bin_low         -- Límite inferior del bin (incluido).
    bin_high        -- Límite superior del bin (excluido).
    bin_center      -- Centro del bin = (bin_low + bin_high) / 2.
                      Valor representativo para gráficos.
    n_picks         -- Número de picks (win + lose) en este bin.
                      No incluye push/void que no tienen outcome binario.
    hit_rate_real   -- wins / (wins + losses) en este bin.
                      None si n_picks == 0.
    prob_mean       -- Media de model_prob de los picks en este bin.
                      Puede diferir del bin_center si la distribución
                      de picks no es uniforme dentro del bin.
    overconfidence  -- prob_mean - hit_rate_real.
                      > 0: modelo sobreestimó la probabilidad real.
                      None si hit_rate_real es None.
    is_reliable     -- True si n_picks >= min_bin_size.
                      Bins no fiables tienen hit_rate estadísticamente
                      ruidoso — útil para el dashboard (mostrar en gris).
    """
    bin_low:        float
    bin_high:       float
    bin_center:     float
    n_picks:        int
    hit_rate_real:  float | None
    prob_mean:      float | None
    overconfidence: float | None
    is_reliable:    bool


# ── Resultado de calibración ──────────────────────────────────────────────────

@dataclass(frozen=True)
class CalibrationResult:
    """
    Todas las métricas de calibración para un conjunto de picks.

    Calculado por CalibrationEngine.calculate() sobre una lista
    de BetLedgerEntry resueltos (win/lose).

    Campos estándar
    ----------------
    brier_score      -- mean((model_prob - outcome)²) sobre picks
                       win/lose. None si n_resolved == 0.
    log_loss         -- -mean(y×log(p) + (1-y)×log(1-p)). None si
                       n_resolved == 0.
    oci              -- Overconfidence Index global = mean(prob) -
                       mean(hit_rate). None si n_resolved == 0.
    calibration_curve -- Lista de CalibrationPoint por bin de
                       probabilidad. Incluye bins vacíos como puntos
                       con n_picks=0 para que el gráfico sea continuo.
    n_total          -- Total de BetLedgerEntry en el input (incluyendo
                       pending, push, void).
    n_resolved       -- Picks win + lose usados en el cálculo.
    n_clamped        -- Picks donde model_prob fue clamped a [ε, 1-ε]
                       para log-loss. Alta cantidad indica probabilidades
                       extremas — posible señal de sobreconfianza.
    mean_model_prob  -- Media de model_prob de picks resueltos.
    mean_hit_rate    -- mean(outcome) = wins / n_resolved.
    label            -- Etiqueta del conjunto analizado (sport, versión,
                       fecha). Para identificar el resultado en logs.
    """
    brier_score:       float | None
    log_loss:          float | None
    oci:               float | None
    calibration_curve: list[CalibrationPoint]
    n_total:           int
    n_resolved:        int
    n_clamped:         int
    mean_model_prob:   float | None
    mean_hit_rate:     float | None
    label:             str

    def summary(self) -> str:
        """Descripción compacta para logging."""
        bs  = f"{self.brier_score:.4f}" if self.brier_score is not None else "N/A"
        ll  = f"{self.log_loss:.4f}"    if self.log_loss    is not None else "N/A"
        oci = f"{self.oci:+.4f}"        if self.oci         is not None else "N/A"
        return (
            f"[{self.label}] n={self.n_resolved} | "
            f"BS={bs} | LL={ll} | OCI={oci}"
        )

    def is_overconfident(self, threshold: float = 0.03) -> bool:
        """True si |OCI| > threshold (sobrepasa el target de calibración)."""
        if self.oci is None:
            return False
        return abs(self.oci) > threshold


# ── Motor principal ───────────────────────────────────────────────────────────

class CalibrationEngine:
    """
    Calcula métricas de calibración sobre picks históricos resueltos.

    No tiene estado mutable entre llamadas — calculate() recibe
    los datos y retorna el resultado. Puede llamarse múltiples veces
    con distintos subconjuntos de entries sin efecto secundario.

    Parámetros
    ----------
    min_bin_size  -- Mínimo de picks para que un bin sea fiable.
                   Default: 10.
    bin_width     -- Ancho de cada bin de calibración. Default: 0.05
                   (bins de 5 puntos porcentuales).
    """

    def __init__(
        self,
        min_bin_size: int   = _DEFAULT_MIN_BIN_SIZE,
        bin_width:    float = _DEFAULT_BIN_WIDTH,
    ) -> None:
        if min_bin_size < 1:
            raise ValueError(f"min_bin_size={min_bin_size} debe ser >= 1.")
        if not 0.01 <= bin_width <= 0.5:
            raise ValueError(f"bin_width={bin_width} debe estar en [0.01, 0.5].")
        self._min_bin_size = min_bin_size
        self._bin_width    = bin_width

    # ── Cálculo principal ──────────────────────────────────────────────────────

    def calculate(
        self,
        entries: Sequence[BetLedgerEntry],
        label:   str = "all",
    ) -> CalibrationResult:
        """
        Calcula todas las métricas de calibración para el conjunto dado.

        Solo usa picks con result en ('win', 'lose') — push/void/pending
        no tienen outcome binario útil para calibración.

        Parámetros
        ----------
        entries  -- Entradas del ledger a evaluar. Pueden estar
                   filtradas por el caller por sport, market, fecha
                   o model_version antes de pasarlas aquí.
        label    -- Etiqueta descriptiva del conjunto para logging.
                   Ej: 'mlb', 'mlb-TOTAL', 'v2.0', '2026-04'.

        Retorna
        -------
        CalibrationResult con todas las métricas calculadas.
        Si no hay picks win/lose, los campos numéricos son None.
        """
        n_total   = len(entries)
        resolved  = [e for e in entries if e.result in ("win", "lose")]
        n_resolved = len(resolved)

        if n_resolved == 0:
            return CalibrationResult(
                brier_score       = None,
                log_loss          = None,
                oci               = None,
                calibration_curve = self._empty_curve(),
                n_total           = n_total,
                n_resolved        = 0,
                n_clamped         = 0,
                mean_model_prob   = None,
                mean_hit_rate     = None,
                label             = label,
            )

        outcomes   = [1.0 if e.result == "win" else 0.0 for e in resolved]
        probs      = [e.model_prob for e in resolved]

        # ── Brier Score ────────────────────────────────────────────────
        brier = sum(
            (p - o) ** 2
            for p, o in zip(probs, outcomes)
        ) / n_resolved

        # ── Log-Loss con clamp ─────────────────────────────────────────
        n_clamped = 0
        ll_sum = 0.0
        for p, o in zip(probs, outcomes):
            p_clamped = p
            if p < _LOG_LOSS_EPS or p > 1 - _LOG_LOSS_EPS:
                p_clamped = max(_LOG_LOSS_EPS, min(1 - _LOG_LOSS_EPS, p))
                n_clamped += 1
            ll_sum += o * math.log(p_clamped) + (1 - o) * math.log(1 - p_clamped)
        log_loss = -ll_sum / n_resolved

        # ── Estadísticas globales ──────────────────────────────────────
        mean_prob     = sum(probs) / n_resolved
        mean_hit_rate = sum(outcomes) / n_resolved
        oci           = round(mean_prob - mean_hit_rate, 6)

        # ── Curva de calibración ───────────────────────────────────────
        curve = self._calibration_curve(resolved, outcomes)

        return CalibrationResult(
            brier_score       = round(brier, 6),
            log_loss          = round(log_loss, 6),
            oci               = oci,
            calibration_curve = curve,
            n_total           = n_total,
            n_resolved        = n_resolved,
            n_clamped         = n_clamped,
            mean_model_prob   = round(mean_prob, 4),
            mean_hit_rate     = round(mean_hit_rate, 4),
            label             = label,
        )

    # ── Comparación de versiones ───────────────────────────────────────────────

    def compare(
        self,
        result_a: CalibrationResult,
        result_b: CalibrationResult,
    ) -> dict:
        """
        Compara dos CalibrationResult (ej. v1.0 vs v2.0 del modelo).

        Retorna dict con diferencias absolutas y dirección de mejora
        para cada métrica. La convención es b - a (positivo = b mejoró).

        Campos del dict retornado:
            label_a, label_b
            brier_delta     -- b.brier_score - a.brier_score (negativo = b mejor)
            log_loss_delta  -- b.log_loss - a.log_loss (negativo = b mejor)
            oci_delta       -- |b.oci| - |a.oci| (negativo = b mejor calibrado)
            n_a, n_b        -- muestras de cada versión
            brier_improved  -- True si b.brier_score < a.brier_score
            oci_improved    -- True si |b.oci| < |a.oci|
            sufficient_data -- True si ambas versiones tienen n >= 30
        """
        def delta(va, vb):
            if va is None or vb is None:
                return None
            return round(vb - va, 6)

        brier_d   = delta(result_a.brier_score, result_b.brier_score)
        ll_d      = delta(result_a.log_loss,    result_b.log_loss)
        oci_a_abs = abs(result_a.oci) if result_a.oci is not None else None
        oci_b_abs = abs(result_b.oci) if result_b.oci is not None else None
        oci_d     = delta(oci_a_abs, oci_b_abs)

        return {
            "label_a":        result_a.label,
            "label_b":        result_b.label,
            "n_a":            result_a.n_resolved,
            "n_b":            result_b.n_resolved,
            "brier_a":        result_a.brier_score,
            "brier_b":        result_b.brier_score,
            "brier_delta":    brier_d,
            "brier_improved": brier_d < 0 if brier_d is not None else None,
            "log_loss_a":     result_a.log_loss,
            "log_loss_b":     result_b.log_loss,
            "log_loss_delta": ll_d,
            "oci_a":          result_a.oci,
            "oci_b":          result_b.oci,
            "oci_delta":      oci_d,
            "oci_improved":   oci_d < 0 if oci_d is not None else None,
            "sufficient_data": (
                result_a.n_resolved >= 30 and result_b.n_resolved >= 30
            ),
        }

    # ── Exportación ────────────────────────────────────────────────────────────

    def export_csv(
        self,
        result: CalibrationResult,
        path:   str,
    ) -> None:
        """
        Exporta la curva de calibración a CSV.

        Formato compatible con el spec:
        {sport}_{date}_calibration.csv
        Columnas: bin_center, n_picks, prob_mean, hit_rate_real,
                  overconfidence, is_reliable.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        fields = [
            "bin_low", "bin_high", "bin_center", "n_picks",
            "prob_mean", "hit_rate_real", "overconfidence", "is_reliable",
        ]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for pt in result.calibration_curve:
                writer.writerow({
                    "bin_low":        pt.bin_low,
                    "bin_high":       pt.bin_high,
                    "bin_center":     pt.bin_center,
                    "n_picks":        pt.n_picks,
                    "prob_mean":      pt.prob_mean   if pt.prob_mean   is not None else "",
                    "hit_rate_real":  pt.hit_rate_real if pt.hit_rate_real is not None else "",
                    "overconfidence": pt.overconfidence if pt.overconfidence is not None else "",
                    "is_reliable":    pt.is_reliable,
                })

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _calibration_curve(
        self,
        resolved: list[BetLedgerEntry],
        outcomes: list[float],
    ) -> list[CalibrationPoint]:
        """Construye la curva de calibración por bins de probabilidad."""
        # Agrupar por bin
        bins: dict[float, list[tuple[float, float]]] = defaultdict(list)
        for entry, outcome in zip(resolved, outcomes):
            bin_low = self._bin_for(entry.model_prob)
            bins[bin_low].append((entry.model_prob, outcome))

        # Construir todos los bins del rango, incluyendo vacíos
        curve: list[CalibrationPoint] = []
        bin_low = _BIN_START
        while bin_low < _BIN_END - 1e-9:
            bin_high   = round(bin_low + self._bin_width, 10)
            bin_center = round(bin_low + self._bin_width / 2, 10)
            pairs      = bins.get(bin_low, [])
            n          = len(pairs)

            if n == 0:
                curve.append(CalibrationPoint(
                    bin_low        = bin_low,
                    bin_high       = bin_high,
                    bin_center     = bin_center,
                    n_picks        = 0,
                    hit_rate_real  = None,
                    prob_mean      = None,
                    overconfidence = None,
                    is_reliable    = False,
                ))
            else:
                probs_in_bin  = [p for p, _ in pairs]
                outcomes_in   = [o for _, o in pairs]
                prob_mean     = sum(probs_in_bin) / n
                hit_rate_real = sum(outcomes_in) / n
                curve.append(CalibrationPoint(
                    bin_low        = bin_low,
                    bin_high       = bin_high,
                    bin_center     = bin_center,
                    n_picks        = n,
                    hit_rate_real  = round(hit_rate_real, 4),
                    prob_mean      = round(prob_mean, 4),
                    overconfidence = round(prob_mean - hit_rate_real, 4),
                    is_reliable    = n >= self._min_bin_size,
                ))

            bin_low = bin_high

        return curve

    def _bin_for(self, prob: float) -> float:
        """Retorna el límite inferior del bin al que pertenece prob."""
        import math as _math
        n_bins   = (prob - _BIN_START) / self._bin_width
        bin_idx  = int(_math.floor(n_bins))
        bin_low  = round(_BIN_START + bin_idx * self._bin_width, 10)
        # Clamp al último bin si prob está en el límite superior
        max_bin  = round(_BIN_END - self._bin_width, 10)
        return min(bin_low, max_bin)

    def _empty_curve(self) -> list[CalibrationPoint]:
        """Curva vacía para cuando no hay datos resueltos."""
        curve  = []
        b      = _BIN_START
        while b < _BIN_END - 1e-9:
            bh = round(b + self._bin_width, 10)
            curve.append(CalibrationPoint(
                bin_low=b, bin_high=bh,
                bin_center=round(b + self._bin_width / 2, 10),
                n_picks=0, hit_rate_real=None, prob_mean=None,
                overconfidence=None, is_reliable=False,
            ))
            b = bh
        return curve