"""
tests/core/test_importability.py

Red de seguridad mínima (Fase 0 del roadmap de remediación, tarea 0.3).

Este test no verifica lógica de negocio — verifica que los paquetes
principales del sistema (core y cada sport plugin) se puedan importar sin
excepción. Es deliberadamente el test más simple posible, y por eso mismo
el más barato de mantener: cualquier `ImportError`/`ModuleNotFoundError`
causado por un módulo faltante o un símbolo importado que no existe en su
origen (contract drift entre archivos) lo atrapa aquí, en segundos, en
lugar de descubrirse en producción al ejecutar `run_daily.py`.

Historial: los 4 bugs P0 documentados en AUDITORIA_BUGS_CRITICOS.md
(módulo sports/mlb/defense.py inexistente; símbolos H2HRecord/H2HStats/
compute_h2h_stats/filter_recent inexistentes en core/utils/h2h_base.py;
_MOVEMENT_CONFIRMS_PREFIX importado del módulo equivocado en
core/pipeline/runner.py; TeamFeatures construido con kwargs inexistentes
sport/season/venue_id) eran TODOS detectables por este test o por un
type checker en modo estricto — ninguno requería ejecutar el pipeline
contra las APIs reales para descubrirse.
"""

import importlib

import pytest

# Cada entrada es un módulo que el sistema necesita poder importar para
# funcionar. Si se añade un sport plugin nuevo (NBA, NFL, Soccer...),
# agregar su módulo de entrada aquí.
_CRITICAL_MODULES = [
    # Core
    "core.contracts",
    "core.contracts.event",
    "core.contracts.features",
    "core.contracts.projection",
    "core.contracts.market_odds",
    "core.contracts.pick",
    "core.contracts.ledger",
    "core.pipeline.stage",
    "core.pipeline.runner",
    "core.pipeline.context",
    "core.odds.client",
    "core.odds.normalizer",
    "core.odds.line_movement",
    "core.odds.no_vig",
    "core.odds.market_registry",
    "core.value.blending",
    "core.value.kelly",
    "core.value.filters",
    "core.value.engine",
    "core.bankroll.staking",
    "core.bankroll.tracker",
    "core.risk.manager",
    "core.tracking.protocols",
    "core.tracking.roi_tracker",
    "core.simulation.poisson",
    "core.simulation.normal",
    "core.simulation.skellam",
    "core.simulation.bivariate_poisson",
    "core.simulation.bradley_terry",
    "core.simulation.ensemble",
    "core.simulation.factory",
    "core.evaluation.clv",
    "core.evaluation.calibration",
    "core.evaluation.metrics",
    "core.backtesting.engine",
    "core.notifications.telegram",
    "core.utils.config_loader",
    "core.utils.h2h_base",
    "core.utils.cache",
    "core.utils.logger",
    "core.utils.math.poisson_math",
    # Plugin MLB — el punto de entrada real que usa scripts/run_daily.py
    "sports.mlb.plugin",
    "sports.mlb.provider",
    "sports.mlb.projections",
    "sports.mlb.pitching",
    "sports.mlb.bullpen",
    "sports.mlb.offense",
    "sports.mlb.defense",
    "sports.mlb.h2h",
    "sports.mlb.context",
    "sports.mlb.statcast",
    "sports.mlb.venue_factors",
    "sports.mlb.markets",
    "sports.mlb.settlement",
]


@pytest.mark.parametrize("module_name", _CRITICAL_MODULES)
def test_module_imports_without_error(module_name: str) -> None:
    """
    Falla con ImportError/ModuleNotFoundError/AttributeError si el
    módulo (o cualquier módulo que importe a nivel de archivo) tiene
    un import roto — sin necesidad de red, API keys, ni ejecutar
    ninguna lógica de negocio.
    """
    importlib.import_module(module_name)


def test_mlb_plugin_instantiates() -> None:
    """
    Un nivel más allá de "importa": confirma que MLBPlugin() se puede
    construir sin argumentos (todos sus componentes tienen defaults) y
    que sus factory methods no lanzan por typos de firma como
    factory.build() vs factory.get_model() — sin llamar a ninguna API
    externa (los fetchers son lazy, no hacen red hasta que se les pide
    fetch() explícitamente).
    """
    from sports.mlb.plugin import MLBPlugin

    plugin = MLBPlugin()
    assert plugin.sport_id == "mlb"
    assert plugin.league_id == "MLB"

    # Estos factory methods instancian objetos pero no hacen red —
    # deben poder llamarse en cualquier entorno, con o sin API keys.
    assert plugin.get_data_provider() is not None
    assert plugin.get_projection_model() is not None
    assert plugin.get_probability_model() is not None
    assert plugin.get_settlement_provider() is not None
    assert plugin.get_market_definitions() is not None
