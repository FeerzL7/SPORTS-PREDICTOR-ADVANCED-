"""
sports/mlb/markets.py

MLBMarketDefinitions: catálogo de mercados disponibles para MLB.

Implementa core/pipeline/stage.py:MarketDefinitions.

El PipelineRunner llama get_core_markets() para saber qué pedir
al OddsAPIClient en la request principal (bajo costo en créditos).
get_extended_markets() para mercados adicionales opcionales.
get_preferred_line() para que OddsNormalizer sepa qué runline usar.

Mercados MLB por tier
----------------------
CORE (endpoint principal, bajo costo):
    h2h       → ML  (moneyline)
    totals    → TOTAL (over/under carreras)
    spreads   → SPREAD (runline ±1.5)

EXTENDED (endpoint por evento, mayor costo):
    h2h_1st_5_innings  → ML_F5
    totals_1st_5_innings → TOTAL_F5
    pitcher_strikeouts → PITCHER_K
    batter_hits        → BATTER_H
    batter_home_runs   → BATTER_HR
    batter_total_bases → BATTER_TB

Preferred lines MLB
--------------------
SPREAD (Runline): -1.5 siempre
    En MLB el runline estándar es siempre ±1.5.
    No hay spreads alternativos en el sentido de NBA/NFL.
    OddsNormalizer.extract_best() usará preferred_line=-1.5
    para seleccionar la línea correcta si hay múltiples.

TOTAL: None (usar línea de mayor consenso)
    Los totales MLB varían por partido (7.5 a 11.5 típicamente).
    No hay una línea "estándar" — OddsNormalizer elige la de
    mayor consenso entre bookmakers.

ML: None (sin línea — mercado binario)
"""

from __future__ import annotations

from core.odds.market_registry import MarketDefinition, MarketRegistry, MarketTier, default_registry


class MLBMarketDefinitions:
    """
    Catálogo de mercados para MLB.

    Implementa core/pipeline/stage.py:MarketDefinitions Protocol.

    Parámetros
    ----------
    registry     -- MarketRegistry base. Default: default_registry()
                   que ya incluye los mercados MLB registrados.
    include_props -- Si True, incluye props de jugadores en extended.
                    Default True. Desactivar si el plan de API no
                    incluye props (ahorrar créditos).
    """

    def __init__(
        self,
        registry:      MarketRegistry | None = None,
        include_props: bool = True,
    ) -> None:
        self._registry      = registry or default_registry()
        self._include_props = include_props

    # ── MarketDefinitions Protocol ────────────────────────────────────────────

    def get_core_markets(self) -> list[str]:
        """
        API keys de mercados CORE para MLB.

        Una sola request a The Odds API cubre todos los partidos
        del día para estos mercados.
        """
        return ["h2h", "totals", "spreads"]

    def get_extended_markets(self) -> list[str]:
        """
        API keys de mercados EXTENDED para MLB.

        Requieren endpoint por evento (mayor costo en créditos).
        El pipeline decide si solicitarlos según créditos disponibles.
        """
        markets = [
            "h2h_1st_5_innings",
            "totals_1st_5_innings",
        ]
        if self._include_props:
            markets.extend([
                "pitcher_strikeouts",
                "batter_hits",
                "batter_home_runs",
                "batter_total_bases",
            ])
        return markets

    def get_preferred_line(self, market: str) -> float | None:
        """
        Línea preferida para un mercado MLB.

        SPREAD (Runline): siempre -1.5 en MLB.
            La línea estándar de runline es -1.5 para el favorito.
            OddsNormalizer usará esta línea para seleccionar la cuota
            correcta cuando hay múltiples líneas disponibles.

        TOTAL: None — la línea varía por partido (consenso).

        ML/F5: None — sin línea (mercados binarios).
        """
        m = market.upper()
        if m in ("SPREAD", "RL"):
            return -1.5
        return None

    # ── Conveniencia ──────────────────────────────────────────────────────────

    def get_all_markets(self) -> list[str]:
        """Todos los mercados MLB (core + extended)."""
        return self.get_core_markets() + self.get_extended_markets()

    def is_supported(self, api_key: str) -> bool:
        """True si el mercado está soportado en MLB."""
        return api_key.lower() in self.get_all_markets()

    def chunk_extended(self, size: int = 12) -> list[list[str]]:
        """
        Divide los mercados extended en chunks para la API.

        The Odds API acepta hasta 12 markets por request en el
        endpoint de event-level.
        """
        return MarketRegistry.chunk(self.get_extended_markets(), size=size)

    def summary(self) -> dict:
        """Resumen del catálogo para logging."""
        return {
            "sport":            "mlb",
            "core_markets":     self.get_core_markets(),
            "extended_markets": self.get_extended_markets(),
            "total_markets":    len(self.get_all_markets()),
            "include_props":    self._include_props,
            "preferred_lines":  {
                "SPREAD": self.get_preferred_line("SPREAD"),
                "TOTAL":  self.get_preferred_line("TOTAL"),
                "ML":     self.get_preferred_line("ML"),
            },
        }