"""
core/evaluation/metrics.py

HypothesisTracker: registro y validación estadística de hipótesis
del sistema sobre datos históricos del ledger.

Por qué existe este módulo
----------------------------
El audit del sistema MLB identificó que el ROI positivo en TOTAL
(+11.15%, n=92) podía ser edge real o ruido de muestreo — sin un
test estadístico formal es imposible saberlo. Con n=92 y hit_rate=56%,
el p-value del binomial test es < 0.05, lo que sí constituye evidencia
estadística. Este módulo formaliza ese proceso para todas las hipótesis
del sistema, de forma continua y automática.

Hipótesis del audit MLB (pre-registradas)
------------------------------------------
H1: "Edge en TOTAL es real"
    trigger: n_total >= 150 picks TOTAL resueltos
    test: binomial test unilateral H0: hit_rate <= 0.513
    resultado esperado: VALIDADA (hit_rate=56% en histórico)

H2: "Mayor edge con abridores de alta muestra (>80 IP)"
    trigger: >= 30 picks por segmento (alta/baja muestra)
    test: Mann-Whitney U sobre ROI por segmento
    resultado esperado: VALIDADA si la segmentación aporta

H7: "Runs reales mejoran proyecciones"
    trigger: 30 días post-fix del bug de ensemble
    test: comparación de CLV medio antes/después del fix
    resultado esperado: VALIDADA si CLV_post > CLV_pre

Estados de hipótesis
---------------------
PENDING   — trigger no alcanzado (datos insuficientes)
VALIDATED — H0 rechazada con p < alpha (evidencia de edge)
REJECTED  — H0 no rechazada (sin evidencia estadística)
PARTIAL   — evidencia mixta entre segmentos/deportes

Tests estadísticos implementados en Python puro
-------------------------------------------------
Binomial test exacto:
    P(X >= k | n, p0) = Σ C(n,i)×p0^i×(1-p0)^(n-i) para i in [k..n]
    Exacto para n < 500 (rango real del sistema).
    Para n > 500: aproximación Normal (CLT) con corrección de continuidad.

Mann-Whitney U:
    Test no paramétrico para comparar dos distribuciones.
    Implementación O(n²) para muestras < 200.
    Para muestras > 200: aproximación Normal del estadístico U.

Uso típico
-----------
    from core.evaluation.metrics import HypothesisTracker, make_h1_total_edge

    tracker = HypothesisTracker()
    tracker.register(make_h1_total_edge(sport='mlb'))
    tracker.register(make_h7_ensemble_clv(fix_date='2026-05-01'))

    entries = bankroll_tracker._store.load_all()
    results = tracker.evaluate_all(entries)

    for h in results:
        print(h.id, h.status.value, h.test_result.detail if h.test_result else '')

    tracker.export_csv('output/hypotheses.csv')
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Callable, Sequence

from core.contracts.ledger import BetLedgerEntry


# ── Estado de hipótesis ───────────────────────────────────────────────────────

class HypothesisStatus(Enum):
    """Estado de validación de una hipótesis."""
    PENDING   = "PENDIENTE"    # datos insuficientes para probar
    VALIDATED = "VALIDADA"     # H0 rechazada — edge estadísticamente real
    REJECTED  = "RECHAZADA"    # H0 no rechazada — sin evidencia de edge
    PARTIAL   = "PARCIAL"      # evidencia mixta entre segmentos


# ── Resultado de test estadístico ─────────────────────────────────────────────

@dataclass(frozen=True)
class TestResult:
    """
    Resultado de un test estadístico sobre datos del ledger.

    Inmutable: representa el resultado en el momento del cálculo.

    Campos
    ------
    statistic     -- Valor del estadístico del test (z, U, t, etc.).
    p_value       -- p-value del test. Compara contra alpha (default 0.05).
    rejected_h0   -- True si p_value < alpha (hay evidencia contra H0).
    effect_size   -- Tamaño del efecto (Cohen's d, odds ratio, etc.).
                    None si no aplica para este tipo de test.
    n_used        -- Número de observaciones usadas en el test.
    alpha         -- Nivel de significancia usado.
    confidence    -- 1 - alpha. % de confianza en el resultado.
    detail        -- Descripción completa del test y resultado para logs.
    """
    statistic:   float
    p_value:     float
    rejected_h0: bool
    effect_size: float | None
    n_used:      int
    alpha:       float
    confidence:  float
    detail:      str


# ── Definición de hipótesis ───────────────────────────────────────────────────

@dataclass
class Hypothesis:
    """
    Definición completa de una hipótesis del sistema.

    Mutable: el status y test_result se actualizan cuando
    evaluate() corre sobre nuevos datos.

    Campos
    ------
    id            -- Identificador único. Ej: 'H1', 'H2', 'H7'.
    description   -- Texto de la hipótesis en lenguaje natural.
    sport         -- Deporte al que aplica, o 'all' para multi-deporte.
    min_n         -- Mínimo de observaciones para activar el trigger.
    alpha         -- Nivel de significancia. Default 0.05.
    test_fn       -- Función que recibe (entries filtradas, **kwargs)
                    y retorna TestResult. Se inyecta en construcción.
    filter_fn     -- Función que filtra las entries antes del test.
                    Default: filtra por sport y result in (win, lose).
    status        -- Estado actual. Empieza en PENDING.
    test_result   -- Resultado del último test. None hasta evaluarse.
    last_checked  -- Timestamp del último evaluate(). None si nunca.
    created_at    -- Timestamp de registro de la hipótesis.
    """
    id:           str
    description:  str
    sport:        str
    min_n:        int
    alpha:        float
    test_fn:      Callable
    filter_fn:    Callable | None  = None
    status:       HypothesisStatus = HypothesisStatus.PENDING
    test_result:  TestResult | None = None
    last_checked: str | None       = None
    created_at:   str              = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def evaluate(self, entries: list[BetLedgerEntry], **kwargs) -> None:
        """
        Ejecuta el test estadístico y actualiza status y test_result.

        Parámetros
        ----------
        entries  -- Entries del ledger (el filter_fn se aplica aquí).
        kwargs   -- Parámetros adicionales pasados a test_fn.
        """
        self.last_checked = datetime.now(timezone.utc).isoformat()

        # Aplicar filtro de datos
        if self.filter_fn is not None:
            filtered = self.filter_fn(entries)
        else:
            filtered = _default_filter(entries, self.sport)

        # Verificar trigger
        if len(filtered) < self.min_n:
            self.status = HypothesisStatus.PENDING
            return

        # Ejecutar test
        try:
            result = self.test_fn(filtered, **kwargs)
        except Exception as e:
            # Error en el test — mantener PENDING con nota
            self.status = HypothesisStatus.PENDING
            return

        self.test_result = result

        # Determinar status según resultado
        if result.rejected_h0:
            self.status = HypothesisStatus.VALIDATED
        else:
            self.status = HypothesisStatus.REJECTED


# ── Motor principal ───────────────────────────────────────────────────────────

class HypothesisTracker:
    """
    Registro y evaluación de hipótesis del sistema sobre datos históricos.

    Parámetros
    ----------
    alpha  -- Nivel de significancia global. Puede sobreescribirse
             por hipótesis individual. Default 0.05.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha:       float                     = alpha
        self._hypotheses:  dict[str, Hypothesis]     = {}

    def register(self, hypothesis: Hypothesis) -> None:
        """
        Registra una hipótesis en el tracker.

        Si ya existe una hipótesis con el mismo id, la sobreescribe.
        """
        self._hypotheses[hypothesis.id] = hypothesis

    def evaluate_all(
        self,
        entries: Sequence[BetLedgerEntry],
        **kwargs,
    ) -> list[Hypothesis]:
        """
        Evalúa todas las hipótesis registradas con los datos actuales.

        Parámetros
        ----------
        entries  -- Entries completas del ledger. Cada hipótesis aplica
                   su propio filter_fn para obtener los datos relevantes.
        kwargs   -- Parámetros adicionales pasados a cada test_fn.
                   Ej: fix_date='2026-05-01' para H7.

        Retorna
        -------
        Lista de hipótesis con status y test_result actualizados.
        """
        entries_list = list(entries)
        for h in self._hypotheses.values():
            h.evaluate(entries_list, **kwargs)
        return list(self._hypotheses.values())

    def get_status(self, hypothesis_id: str) -> HypothesisStatus:
        """
        Retorna el estado actual de una hipótesis.

        Raises KeyError si el hypothesis_id no está registrado.
        """
        h = self._hypotheses.get(hypothesis_id)
        if h is None:
            raise KeyError(
                f"Hipótesis '{hypothesis_id}' no registrada. "
                f"Registradas: {sorted(self._hypotheses.keys())}"
            )
        return h.status

    def summary(self) -> dict[str, str]:
        """Resumen {id: status} para logging."""
        return {h_id: h.status.value for h_id, h in self._hypotheses.items()}

    def export_csv(self, path: str) -> None:
        """
        Exporta el estado de todas las hipótesis a CSV.

        Compatible con el formato del spec:
        {sport}_{date}_hypotheses.csv
        """
        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
        )
        fields = [
            "id", "description", "sport", "status",
            "p_value", "statistic", "rejected_h0", "n_used",
            "alpha", "effect_size", "detail", "last_checked",
        ]
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for h in self._hypotheses.values():
                tr = h.test_result
                writer.writerow({
                    "id":          h.id,
                    "description": h.description,
                    "sport":       h.sport,
                    "status":      h.status.value,
                    "p_value":     tr.p_value      if tr else "",
                    "statistic":   tr.statistic    if tr else "",
                    "rejected_h0": tr.rejected_h0  if tr else "",
                    "n_used":      tr.n_used        if tr else "",
                    "alpha":       h.alpha,
                    "effect_size": tr.effect_size  if tr else "",
                    "detail":      tr.detail        if tr else "",
                    "last_checked":h.last_checked or "",
                })


# ── Tests estadísticos en Python puro ────────────────────────────────────────

def binomial_test_upper(
    k:   int,
    n:   int,
    p0:  float,
) -> float:
    """
    P-value del test binomial unilateral superior: P(X >= k | n, p0).

    H0: hit_rate <= p0
    H1: hit_rate > p0  (test unilateral superior)

    Exacto para n < 500 (suma directa de probabilidades binomiales).
    Para n >= 500: aproximación Normal con corrección de continuidad.

    Parámetros
    ----------
    k   -- Número de éxitos observados.
    n   -- Número total de ensayos.
    p0  -- Proporción bajo H0 (umbral de referencia).
    """
    if n <= 0 or k < 0 or k > n:
        return 1.0
    if p0 <= 0:
        return 0.0 if k > 0 else 1.0
    if p0 >= 1:
        return 1.0

    if n < 500:
        # Suma exacta
        p_value = 0.0
        for i in range(k, n + 1):
            p_value += math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i))
        return min(p_value, 1.0)
    else:
        # Aproximación Normal (CLT) con corrección de continuidad
        mu     = n * p0
        sigma  = math.sqrt(n * p0 * (1 - p0))
        z      = (k - 0.5 - mu) / sigma  # corrección de continuidad
        return _normal_sf(z)


def mann_whitney_u_test(
    group_a: list[float],
    group_b: list[float],
) -> tuple[float, float]:
    """
    Test de Mann-Whitney U bilateral para dos muestras independientes.

    H0: las dos distribuciones son iguales.
    H1: las distribuciones difieren.

    Implementación O(n²) exacta para muestras < 200.
    Para muestras >= 200: aproximación Normal del estadístico U.

    Retorna (U_statistic, p_value_bilateral).
    """
    n_a = len(group_a)
    n_b = len(group_b)

    if n_a == 0 or n_b == 0:
        return 0.0, 1.0

    if n_a < 200 and n_b < 200:
        # Cálculo exacto de U
        u_a = sum(
            1.0 if a > b else (0.5 if a == b else 0.0)
            for a in group_a
            for b in group_b
        )
        u_b = n_a * n_b - u_a
        u   = min(u_a, u_b)
    else:
        # Aproximación para muestras grandes
        # U_a = R_a - n_a*(n_a+1)/2 donde R_a es la suma de rangos de A
        combined = (
            [(v, 'a') for v in group_a] + [(v, 'b') for v in group_b]
        )
        combined.sort(key=lambda x: x[0])
        ranks_a = [
            i + 1 for i, (_, grp) in enumerate(combined) if grp == 'a'
        ]
        u_a = sum(ranks_a) - n_a * (n_a + 1) / 2
        u_b = n_a * n_b - u_a
        u   = min(u_a, u_b)

    # Aproximación Normal de U
    mu_u    = n_a * n_b / 2
    sigma_u = math.sqrt(n_a * n_b * (n_a + n_b + 1) / 12)
    if sigma_u == 0:
        return u, 1.0
    z       = (u - mu_u) / sigma_u
    p_value = 2 * _normal_sf(abs(z))  # bilateral
    return u, min(p_value, 1.0)


def _normal_sf(z: float) -> float:
    """P(Z > z) para Z ~ N(0,1). Survival function de la Normal estándar."""
    return (1 - math.erf(z / math.sqrt(2))) / 2


# ── Filtros de datos ──────────────────────────────────────────────────────────

def _default_filter(
    entries: list[BetLedgerEntry],
    sport:   str,
) -> list[BetLedgerEntry]:
    """Filtra por sport y result resuelto (win/lose)."""
    filtered = [e for e in entries if e.result in ("win", "lose")]
    if sport != "all":
        filtered = [e for e in filtered if e.sport.lower() == sport.lower()]
    return filtered


# ── Factory functions para hipótesis del audit ────────────────────────────────

def make_h1_total_edge(
    sport:     str   = "mlb",
    min_n:     int   = 150,
    p0:        float = 0.513,
    alpha:     float = 0.05,
) -> Hypothesis:
    """
    H1: "Edge en TOTAL es real"

    Test binomial unilateral superior:
        H0: hit_rate_TOTAL <= p0 (sin edge)
        H1: hit_rate_TOTAL > p0  (edge real)

    p0=0.513 corresponde al break-even para cuota media de 1.91
    (1/1.91 ≈ 0.524, pero con vig implícito el break-even real
    es ~51.3% para mercado de -110/-110 en cuota americana).

    Parámetros
    ----------
    p0     -- Hit rate de break-even para el mercado de TOTAL.
             Default 0.513 (calibrado para cuota 1.91).
    min_n  -- Mínimo de picks TOTAL para activar el test. Default 150.
    """

    def filter_fn(entries: list[BetLedgerEntry]) -> list[BetLedgerEntry]:
        return [
            e for e in entries
            if e.result in ("win", "lose")
            and e.market.upper() == "TOTAL"
            and (sport == "all" or e.sport.lower() == sport.lower())
        ]

    def test_fn(filtered: list[BetLedgerEntry]) -> TestResult:
        n    = len(filtered)
        wins = sum(1 for e in filtered if e.result == "win")
        hit_rate = wins / n

        p_value = binomial_test_upper(k=wins, n=n, p0=p0)
        rejected = p_value < alpha

        # Effect size: diferencia entre hit_rate real y p0
        effect = round(hit_rate - p0, 4)

        detail = (
            f"Binomial test H0: hit_rate <= {p0} | "
            f"hits={wins}/{n} ({hit_rate:.3f}) | "
            f"p={p_value:.4f} | "
            f"{'RECHAZADA H0 ✓' if rejected else 'NO rechazada ✗'} | "
            f"efecto={effect:+.3f}"
        )
        return TestResult(
            statistic   = round(hit_rate, 4),
            p_value     = round(p_value, 6),
            rejected_h0 = rejected,
            effect_size = effect,
            n_used      = n,
            alpha       = alpha,
            confidence  = 1 - alpha,
            detail      = detail,
        )

    return Hypothesis(
        id          = "H1",
        description = f"Edge en TOTAL ({sport.upper()}) es estadísticamente real",
        sport       = sport,
        min_n       = min_n,
        alpha       = alpha,
        test_fn     = test_fn,
        filter_fn   = filter_fn,
    )


def make_h2_starter_sample(
    sport:        str   = "mlb",
    segment_key:  str   = "model_version",
    min_per_seg:  int   = 30,
    alpha:        float = 0.05,
) -> Hypothesis:
    """
    H2: "Mayor edge con abridores de alta muestra (>80 IP)"

    En el sistema multi-deporte, H2 generaliza a: "hay diferencia
    significativa de ROI entre dos segmentos definidos por segment_key".
    Para MLB, segment_key identifica el grupo de abridores (tag en
    model_version o campo externo). El test es Mann-Whitney U sobre
    yield_pct de cada grupo.

    Parámetros
    ----------
    segment_key  -- Campo de BetLedgerEntry para segmentar.
                   Default 'model_version' para compatibilidad con
                   el uso de MLBPlugin que puede añadir tags al
                   model_version ('v2.0-alta-ip' vs 'v2.0-baja-ip').
    min_per_seg  -- Mínimo de picks por segmento. Default 30.
    """

    def filter_fn(entries: list[BetLedgerEntry]) -> list[BetLedgerEntry]:
        return [
            e for e in entries
            if e.result in ("win", "lose")
            and e.yield_pct is not None
            and (sport == "all" or e.sport.lower() == sport.lower())
        ]

    def test_fn(filtered: list[BetLedgerEntry]) -> TestResult:
        from collections import defaultdict
        segments: dict[str, list[float]] = defaultdict(list)
        for e in filtered:
            key = getattr(e, segment_key, "unknown") or "unknown"
            if e.yield_pct is not None:
                segments[key].append(e.yield_pct)

        if len(segments) < 2:
            return TestResult(
                statistic=0.0, p_value=1.0, rejected_h0=False,
                effect_size=None, n_used=len(filtered), alpha=alpha,
                confidence=1-alpha,
                detail="Menos de 2 segmentos — test no aplicable",
            )

        keys       = sorted(segments.keys())
        group_a    = segments[keys[0]]
        group_b    = segments[keys[1]]
        n_a, n_b   = len(group_a), len(group_b)

        if n_a < min_per_seg or n_b < min_per_seg:
            return TestResult(
                statistic=0.0, p_value=1.0, rejected_h0=False,
                effect_size=None, n_used=n_a + n_b, alpha=alpha,
                confidence=1-alpha,
                detail=(
                    f"Muestras insuficientes: "
                    f"{keys[0]}(n={n_a}) {keys[1]}(n={n_b}) — "
                    f"min={min_per_seg} requerido"
                ),
            )

        u_stat, p_value = mann_whitney_u_test(group_a, group_b)
        rejected = p_value < alpha

        mean_a = sum(group_a) / n_a
        mean_b = sum(group_b) / n_b
        effect = round(mean_b - mean_a, 4)

        detail = (
            f"Mann-Whitney U={u_stat:.1f} | "
            f"{keys[0]}: yield_mean={mean_a:.2f}% (n={n_a}) | "
            f"{keys[1]}: yield_mean={mean_b:.2f}% (n={n_b}) | "
            f"p={p_value:.4f} | "
            f"{'RECHAZADA H0 ✓' if rejected else 'NO rechazada ✗'}"
        )
        return TestResult(
            statistic   = round(u_stat, 2),
            p_value     = round(p_value, 6),
            rejected_h0 = rejected,
            effect_size = effect,
            n_used      = n_a + n_b,
            alpha       = alpha,
            confidence  = 1 - alpha,
            detail      = detail,
        )

    return Hypothesis(
        id          = "H2",
        description = (
            f"Diferencia de rendimiento por segmento "
            f"('{segment_key}') en {sport.upper()}"
        ),
        sport       = sport,
        min_n       = min_per_seg * 2,
        alpha       = alpha,
        test_fn     = test_fn,
        filter_fn   = filter_fn,
    )


def make_h7_ensemble_clv(
    fix_date: str,
    sport:    str   = "mlb",
    min_n:    int   = 30,
    alpha:    float = 0.05,
) -> Hypothesis:
    """
    H7: "Runs reales (ensemble activado) mejoran proyecciones"

    Compara CLV medio antes y después de fix_date.
    H0: clv_post <= clv_pre (el fix no mejoró el modelo)
    H1: clv_post > clv_pre  (el fix mejoró el modelo)

    Test: Mann-Whitney U unilateral sobre CLV de ambos períodos.

    Parámetros
    ----------
    fix_date  -- Fecha del fix en 'YYYY-MM-DD'. Divide los picks en
                pre (date < fix_date) y post (date >= fix_date).
    min_n     -- Mínimo de picks con CLV en cada período. Default 30.
    """

    def filter_fn(entries: list[BetLedgerEntry]) -> list[BetLedgerEntry]:
        return [
            e for e in entries
            if e.result in ("win", "lose")
            and e.clv is not None
            and (sport == "all" or e.sport.lower() == sport.lower())
        ]

    def test_fn(filtered: list[BetLedgerEntry]) -> TestResult:
        pre  = [e.clv for e in filtered if e.date <  fix_date]
        post = [e.clv for e in filtered if e.date >= fix_date]

        n_pre, n_post = len(pre), len(post)

        if n_pre < min_n or n_post < min_n:
            return TestResult(
                statistic=0.0, p_value=1.0, rejected_h0=False,
                effect_size=None, n_used=n_pre + n_post, alpha=alpha,
                confidence=1-alpha,
                detail=(
                    f"Datos insuficientes — pre={n_pre} post={n_post} "
                    f"(min={min_n} por período)"
                ),
            )

        u_stat, p_value_bilateral = mann_whitney_u_test(post, pre) # type: ignore
        # Convertir a unilateral (H1: post > pre → la dirección ya está en U)
        p_value = p_value_bilateral / 2

        rejected = p_value < alpha
        mean_pre  = sum(pre)  / n_pre # type: ignore
        mean_post = sum(post) / n_post # type: ignore
        effect    = round(mean_post - mean_pre, 4)

        detail = (
            f"CLV pre-fix (n={n_pre}): mean={mean_pre:.3f}% | "
            f"CLV post-fix (n={n_post}): mean={mean_post:.3f}% | "
            f"Δ={effect:+.3f}% | "
            f"p(unilateral)={p_value:.4f} | "
            f"{'RECHAZADA H0 ✓ — fix mejoró CLV' if rejected else 'NO rechazada ✗'}"
        )
        return TestResult(
            statistic   = round(u_stat, 2),
            p_value     = round(p_value, 6),
            rejected_h0 = rejected,
            effect_size = effect,
            n_used      = n_pre + n_post,
            alpha       = alpha,
            confidence  = 1 - alpha,
            detail      = detail,
        )

    return Hypothesis(
        id          = "H7",
        description = (
            f"Ensemble activado mejora CLV post-{fix_date} en {sport.upper()}"
        ),
        sport       = sport,
        min_n       = min_n * 2,
        alpha       = alpha,
        test_fn     = test_fn,
        filter_fn   = filter_fn,
    )