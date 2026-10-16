"""
tests/core/test_blend_fit.py

Ajuste del peso del modelo frente al mercado (core/evaluation/blend_fit.py).

Los datos son sintéticos y con semilla fija: el resultado de cada
selección se sortea con una probabilidad "verdadera" conocida, así que
se sabe de antemano qué fuente tiene información y cuál es ruido.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import math
import random

import pytest

from core.evaluation.blend_fit import (
    VERDICT_INFORMATIVE,
    VERDICT_NO_SIGNAL,
    Observation,
    blend,
    evaluate_market,
    fit_linear_weight,
    fit_logistic_stack,
    log_loss,
)


def _clip(p):
    return min(max(p, 0.02), 0.98)


def _synthetic(informed: str, n_per_season: int = 1500, seed: int = 11):
    """
    Observaciones en tres temporadas.

    informed='market': el mercado conoce la probabilidad verdadera y el
                       modelo es mercado + ruido (el caso de fútbol).
    informed='model':  el modelo la conoce y el mercado es ruidoso.
    """
    rng = random.Random(seed)
    obs = []
    for season in (2021, 2022, 2023):
        for i in range(n_per_season):
            true_p = rng.uniform(0.15, 0.85)
            noisy = _clip(true_p + rng.gauss(0, 0.10))
            y = 1 if rng.random() < true_p else 0
            model, market = ((noisy, true_p) if informed == "market"
                             else (true_p, noisy))
            obs.append(Observation(model, market, y, season, f"{season}-{i}"))
    return obs


# ── Funciones básicas ─────────────────────────────────────────────────────────

class TestBasics:
    def test_log_loss_of_coin_flip(self):
        assert log_loss([0.5, 0.5], [1, 0]) == pytest.approx(math.log(2))

    def test_log_loss_survives_certainty(self):
        # p=1 con resultado 0 no debe producir infinito
        assert math.isfinite(log_loss([1.0], [0]))

    def test_log_loss_rejects_empty(self):
        with pytest.raises(ValueError):
            log_loss([], [])

    @pytest.mark.parametrize("w,expected", [(0.0, 0.40), (1.0, 0.60), (0.5, 0.50)])
    def test_blend_matches_blending_engine(self, w, expected):
        assert blend(0.60, 0.40, w) == pytest.approx(expected)


# ── Ajuste ────────────────────────────────────────────────────────────────────

class TestFit:
    def test_noise_model_gets_near_zero_weight(self):
        w = fit_linear_weight(_synthetic("market"))
        assert w <= 0.10

    def test_informed_model_gets_high_weight(self):
        w = fit_linear_weight(_synthetic("model"))
        assert w >= 0.80

    def test_stack_coefficients_point_to_the_informed_source(self):
        _, b_model, b_market = fit_logistic_stack(_synthetic("model"))
        assert b_model > 0.5 > b_market

    def test_noise_model_has_no_positive_stack_coefficient(self):
        _, b_model, b_market = fit_logistic_stack(_synthetic("market"))
        assert b_model < 0.15
        assert b_market > 0.8


# ── Validación fuera de muestra ───────────────────────────────────────────────

class TestEvaluateMarket:
    def test_noise_model_has_no_signal(self):
        fit = evaluate_market("1X2", _synthetic("market"), [2023], n_boot=200)
        assert fit.verdict == VERDICT_NO_SIGNAL
        assert fit.test_seasons == (2023,)
        assert fit.train_seasons == (2021, 2022)

    def test_informed_model_is_informative(self):
        fit = evaluate_market("TOTAL", _synthetic("model"), [2023], n_boot=200)
        assert fit.verdict == VERDICT_INFORMATIVE
        assert fit.delta_test < 0
        assert fit.ci_high < 0

    def test_zero_weight_gives_exactly_zero_interval(self):
        # El mercado da 0.9 al lado que gana y 0.1 al que pierde; el
        # modelo dice 0.5 siempre. Cualquier w > 0 acerca la mezcla a
        # 0.5 y empeora la log-loss en TODAS las filas, así que w=0 es
        # forzoso. Y con w=0 la mezcla ES el mercado: Δ vale 0 en cada
        # remuestreo.
        obs = [Observation(0.5, 0.9 if y else 0.1, y, season, f"{season}-{i}")
               for season in (2021, 2022, 2023)
               for i, y in enumerate([1, 0] * 200)]
        fit = evaluate_market("ML", obs, [2023], n_boot=50)
        assert fit.weight == 0.0
        assert fit.ci_low == fit.ci_high == fit.delta_test == 0.0
        assert fit.verdict == VERDICT_NO_SIGNAL

    def test_weight_never_sees_the_test_season(self):
        # Alterar SOLO la temporada de prueba no puede cambiar el peso.
        obs = _synthetic("market")
        flipped = [Observation(o.model_prob, o.market_prob,
                               1 - o.outcome if o.season == 2023 else o.outcome,
                               o.season, o.group) for o in obs]
        a = evaluate_market("X", obs, [2023], n_boot=20)
        b = evaluate_market("X", flipped, [2023], n_boot=20)
        assert a.weight == b.weight

    def test_bootstrap_is_reproducible(self):
        obs = _synthetic("model", n_per_season=400)
        a = evaluate_market("X", obs, [2023], n_boot=100, seed=3)
        b = evaluate_market("X", obs, [2023], n_boot=100, seed=3)
        assert (a.ci_low, a.ci_high) == (b.ci_low, b.ci_high)

    def test_requires_both_halves(self):
        obs = [o for o in _synthetic("market") if o.season == 2023]
        with pytest.raises(ValueError):
            evaluate_market("X", obs, [2023])
