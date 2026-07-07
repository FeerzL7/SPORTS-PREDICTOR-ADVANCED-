"""
core/odds/market_registry.py

MarketRegistry: catálogo central de mercados de apuestas por deporte y tier.

Migrado de data/odds_markets.py del sistema MLB con dos correcciones
arquitectónicas documentadas en MLB_EDGE_AUDIT.md:

1. Separación estricta entre mercados universales y sport-específicos.
   El sistema MLB mezclaba en MARKET_GROUPS grupos universales ('core')
   con grupos baseball-específicos ('baseball_game', 'mlb_player_props')
   sin documentar qué funcionaba para qué deporte. Un consumer que
   intentara usar 'pitcher_strikeouts' para NBA recibiría un error 404
   de la API sin explicación clara en el código.

2. Clasificación CORE vs EXTENDED en vez de featured vs event-level.
   featured/event-level es un detalle de implementación de The Odds API.
   Si en el futuro se usa Sportradar u otro provider, no habrá esa
   distinción. CORE/EXTENDED es un concepto de dominio independiente
   del provider: CORE = siempre disponibles, bajo costo en créditos;
   EXTENDED = disponibilidad variable, mayor costo o endpoint adicional.

Modelo de datos
----------------
MarketDefinition es la unidad atómica: un mercado con su API key,
nombre interno, tier y conjunto de deportes que lo soportan.

sports=set() vacío significa "universal" (aplica a todos los deportes).
Solo mercados sport-específicos listan explícitamente sus deportes:
    pitcher_strikeouts → sports={'mlb'}
    btts              → sports={'soccer'}
    h2h               → sports=set()  (universal)

Registro por sport plugin
--------------------------
default_registry() retorna un registry pre-poblado con mercados
universales. Los sport plugins registran sus mercados adicionales:

    registry = default_registry()
    registry.register(MarketDefinition(
        api_key='pitcher_strikeouts',
        internal_name='PITCHER_K',
        tier=MarketTier.EXTENDED,
        sports={'mlb'},
        description='Strikeouts de pitcher (prop MLB)',
    ))

Uso típico
-----------
    from core.odds.market_registry import default_registry

    registry = default_registry()

    # El OddsAPIClient usa API keys para pedir datos
    core_keys = registry.get_core_markets(sport='mlb')
    # ['h2h', 'totals', 'spreads']

    # El sport plugin puede añadir extended markets para su deporte
    extended_keys = registry.get_extended_markets(sport='mlb')
    # ['alternate_totals', 'team_totals', 'pitcher_strikeouts', ...]

    # Chunking para respetar límites de la API (max 12 markets/request)
    chunks = registry.chunk(extended_keys, size=12)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


# ── Tier de mercado ───────────────────────────────────────────────────────────

class MarketTier(Enum):
    """
    Clasificación de mercados por disponibilidad y costo.

    CORE:
        Siempre disponibles en el endpoint principal de la API.
        Bajo costo en créditos (una request cubre todos los eventos).
        Ejemplos: h2h (ML), spreads, totals, h2h_3_way.

    EXTENDED:
        Disponibilidad variable según plan de la API.
        Mayor costo: requieren endpoint por evento o tienen cobertura
        reducida de bookmakers.
        Ejemplos: props de jugadores, mercados por período, alternativos.
    """
    CORE     = auto()
    EXTENDED = auto()


# ── Definición de mercado ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class MarketDefinition:
    """
    Definición inmutable de un mercado de apuestas.

    Representa un único mercado con su identificador en la API,
    nombre convencional interno, clasificación por tier y conjunto
    de deportes donde está disponible.

    Campos
    ------
    api_key        -- Identificador del mercado en The Odds API.
                    Ejemplos: 'h2h', 'totals', 'pitcher_strikeouts'.
                    Es el string que se pasa a OddsAPIClient.get_events().
    internal_name  -- Nombre convencional interno del sistema.
                    Debe coincidir con MARKET_NAME_MAP de normalizer.py
                    para mercados que OddsNormalizer procesa.
                    Ejemplos: 'ML', 'TOTAL', 'SPREAD', '1X2'.
    tier           -- CORE o EXTENDED. Determina en qué endpoint
                    del cliente se solicita el mercado.
    sports         -- Conjunto de sport_ids donde este mercado está
                    disponible. Set vacío = universal (todos los deportes).
                    Ejemplos: {'mlb'}, {'soccer'}, set() (universal).
    description    -- Descripción legible para documentación y logs.
    """
    api_key:       str
    internal_name: str
    tier:          MarketTier
    sports:        frozenset[str] = field(default_factory=frozenset)
    description:   str           = ""

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ValueError("api_key no puede estar vacío.")
        if not self.internal_name or not self.internal_name.strip():
            raise ValueError("internal_name no puede estar vacío.")

    def supports(self, sport: str) -> bool:
        """
        True si este mercado está disponible para el deporte dado.

        sports vacío significa universal — soporta cualquier deporte.
        """
        if not self.sports:
            return True
        return sport.lower() in self.sports


# ── Registry principal ────────────────────────────────────────────────────────

class MarketRegistry:
    """
    Catálogo central de mercados de apuestas.

    Permite registrar mercados por deporte y consultarlos por tier,
    facilitando que OddsAPIClient sepa exactamente qué pedir y con
    qué costo en créditos.

    Thread-safety: el registry no está diseñado para modificación
    concurrente. En un pipeline multi-sport, cada sport plugin debe
    operar con su propia instancia o registrar todos sus mercados
    antes de iniciar el procesamiento paralelo.

    Uso típico:
        registry = default_registry()           # mercados universales
        registry.register(my_sport_market)      # añadir sport-específico
        keys = registry.get_core_markets('mlb') # consultar
    """

    def __init__(self) -> None:
        # Almacena definiciones por api_key — clave única
        self._definitions: dict[str, MarketDefinition] = {}

    # ── Registro ───────────────────────────────────────────────────────────────

    def register(self, definition: MarketDefinition) -> None:
        """
        Registra un mercado en el catálogo.

        Si ya existe un mercado con el mismo api_key, lo sobreescribe.
        Esto permite que un sport plugin personalice la definición de
        un mercado universal (ej. cambiar tier o añadir deportes).
        """
        self._definitions[definition.api_key.lower()] = definition

    def register_many(self, definitions: list[MarketDefinition]) -> None:
        """Registra múltiples mercados en una sola llamada."""
        for definition in definitions:
            self.register(definition)

    # ── Consulta ───────────────────────────────────────────────────────────────

    def get_core_markets(self, sport: str) -> list[str]:
        """
        Retorna los API keys de mercados CORE disponibles para el deporte.

        Son los mercados que se piden en el endpoint principal de la API
        (bajo costo en créditos — una request cubre todos los eventos).

        Orden preservado: los mercados se retornan en el orden en que
        fueron registrados, que corresponde al orden de prioridad para
        ese deporte.
        """
        return [
            defn.api_key
            for defn in self._definitions.values()
            if defn.tier == MarketTier.CORE and defn.supports(sport)
        ]

    def get_extended_markets(self, sport: str) -> list[str]:
        """
        Retorna los API keys de mercados EXTENDED disponibles para el deporte.

        Son los mercados que requieren endpoint por evento o tienen
        mayor costo en créditos. El pipeline decide cuándo vale la pena
        solicitarlos según el plan de la API y los créditos restantes.
        """
        return [
            defn.api_key
            for defn in self._definitions.values()
            if defn.tier == MarketTier.EXTENDED and defn.supports(sport)
        ]

    def get_all_markets(self, sport: str) -> list[str]:
        """Retorna todos los API keys disponibles para el deporte (CORE + EXTENDED)."""
        return [
            defn.api_key
            for defn in self._definitions.values()
            if defn.supports(sport)
        ]

    def get_definition(self, api_key: str) -> MarketDefinition | None:
        """Retorna la definición de un mercado por su API key, o None si no existe."""
        return self._definitions.get(api_key.lower())

    def is_supported(self, api_key: str, sport: str) -> bool:
        """
        True si el mercado está registrado y soporta el deporte dado.

        Útil para validar antes de solicitar un mercado al cliente:
        evita requests que garantizadamente retornarán 404 o vacíos.
        """
        defn = self._definitions.get(api_key.lower())
        if defn is None:
            return False
        return defn.supports(sport)

    def get_internal_name(self, api_key: str) -> str:
        """
        Retorna el nombre interno del mercado para un API key dado.

        Si el mercado no está registrado, retorna api_key.upper() como
        fallback — mismo comportamiento que normalize_market_name() en
        normalizer.py para mantener consistencia.
        """
        defn = self._definitions.get(api_key.lower())
        if defn is None:
            return api_key.upper()
        return defn.internal_name

    # ── Utilidades ─────────────────────────────────────────────────────────────

    @staticmethod
    def chunk(markets: list[str], size: int = 12) -> list[list[str]]:
        """
        Divide una lista de market keys en chunks de tamaño máximo.

        The Odds API acepta hasta 12 markets por request en el endpoint
        de event-level. El tamaño por defecto refleja este límite.

        Parámetros
        ----------
        markets  -- Lista de API keys a dividir.
        size     -- Tamaño máximo de cada chunk. Default: 12.

        Retorna
        -------
        Lista de listas. Si markets está vacío, retorna [].
        Si len(markets) <= size, retorna [[markets]].
        """
        if not markets:
            return []
        return [markets[i:i + size] for i in range(0, len(markets), size)]

    def summary(self, sport: str) -> dict:
        """
        Retorna un resumen del catálogo para un deporte dado.

        Útil para logging al inicio del pipeline y para debugging.
        """
        core_keys     = self.get_core_markets(sport)
        extended_keys = self.get_extended_markets(sport)
        return {
            "sport":          sport,
            "core_markets":   core_keys,
            "extended_markets": extended_keys,
            "total_markets":  len(core_keys) + len(extended_keys),
        }

    def __len__(self) -> int:
        return len(self._definitions)

    def __contains__(self, api_key: str) -> bool:
        return api_key.lower() in self._definitions


# ── Mercados universales pre-definidos ────────────────────────────────────────

# Mercados CORE universales — disponibles en todos los deportes
# que The Odds API soporta, en el endpoint principal (bajo costo).
_UNIVERSAL_CORE: list[MarketDefinition] = [
    MarketDefinition(
        api_key="h2h",
        internal_name="ML",
        tier=MarketTier.CORE,
        sports=frozenset(),
        description="Head-to-head / Moneyline. Ganador del partido.",
    ),
    MarketDefinition(
        api_key="totals",
        internal_name="TOTAL",
        tier=MarketTier.CORE,
        sports=frozenset(),
        description="Over/Under del total de puntos/carreras/goles.",
    ),
    MarketDefinition(
        api_key="spreads",
        internal_name="SPREAD",
        tier=MarketTier.CORE,
        sports=frozenset(),
        description="Handicap / Spread / Runline.",
    ),
    MarketDefinition(
        api_key="h2h_3_way",
        internal_name="1X2",
        tier=MarketTier.CORE,
        sports=frozenset(),
        description="1X2 con empate. Principal para Soccer.",
    ),
]

# Mercados EXTENDED universales — disponibles en varios deportes
# pero requieren endpoint adicional o mayor costo en créditos.
_UNIVERSAL_EXTENDED: list[MarketDefinition] = [
    MarketDefinition(
        api_key="alternate_spreads",
        internal_name="ALT_SPREAD",
        tier=MarketTier.EXTENDED,
        sports=frozenset(),
        description="Spreads alternativos (líneas adicionales).",
    ),
    MarketDefinition(
        api_key="alternate_totals",
        internal_name="ALT_TOTAL",
        tier=MarketTier.EXTENDED,
        sports=frozenset(),
        description="Totales alternativos (líneas adicionales).",
    ),
    MarketDefinition(
        api_key="team_totals",
        internal_name="TEAM_TOTAL",
        tier=MarketTier.EXTENDED,
        sports=frozenset(),
        description="Total de puntos/carreras por equipo individual.",
    ),
]

# Mercados MLB-específicos
_MLB_MARKETS: list[MarketDefinition] = [
    MarketDefinition(
        api_key="h2h_1st_5_innings",
        internal_name="ML_F5",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"mlb"}),
        description="Moneyline primeros 5 innings (MLB).",
    ),
    MarketDefinition(
        api_key="totals_1st_5_innings",
        internal_name="TOTAL_F5",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"mlb"}),
        description="Total primeros 5 innings (MLB).",
    ),
    MarketDefinition(
        api_key="pitcher_strikeouts",
        internal_name="PITCHER_K",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"mlb"}),
        description="Prop: strikeouts del pitcher (MLB).",
    ),
    MarketDefinition(
        api_key="batter_hits",
        internal_name="BATTER_H",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"mlb"}),
        description="Prop: hits del bateador (MLB).",
    ),
    MarketDefinition(
        api_key="batter_home_runs",
        internal_name="BATTER_HR",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"mlb"}),
        description="Prop: home runs del bateador (MLB).",
    ),
    MarketDefinition(
        api_key="batter_total_bases",
        internal_name="BATTER_TB",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"mlb"}),
        description="Prop: total bases del bateador (MLB).",
    ),
]

# Mercados Soccer-específicos
_SOCCER_MARKETS: list[MarketDefinition] = [
    MarketDefinition(
        api_key="btts",
        internal_name="BTTS",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"soccer"}),
        description="Ambos equipos anotan (Soccer).",
    ),
    MarketDefinition(
        api_key="draw_no_bet",
        internal_name="DNB",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"soccer"}),
        description="Draw No Bet — apuesta sin empate (Soccer).",
    ),
    MarketDefinition(
        api_key="double_chance",
        internal_name="DC",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"soccer"}),
        description="Doble oportunidad: 1X, 12, X2 (Soccer).",
    ),
]

# Mercados NBA-específicos
_NBA_MARKETS: list[MarketDefinition] = [
    MarketDefinition(
        api_key="player_points",
        internal_name="PLAYER_PTS",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"basketball_nba"}),
        description="Prop: puntos del jugador (NBA).",
    ),
    MarketDefinition(
        api_key="player_rebounds",
        internal_name="PLAYER_REB",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"basketball_nba"}),
        description="Prop: rebotes del jugador (NBA).",
    ),
    MarketDefinition(
        api_key="player_assists",
        internal_name="PLAYER_AST",
        tier=MarketTier.EXTENDED,
        sports=frozenset({"basketball_nba"}),
        description="Prop: asistencias del jugador (NBA).",
    ),
]


# ── Factory function ──────────────────────────────────────────────────────────

def default_registry() -> MarketRegistry:
    """
    Crea y retorna un MarketRegistry pre-poblado con todos los mercados
    definidos en este módulo: universales + MLB + Soccer + NBA.

    Los sport plugins pueden registrar mercados adicionales sobre este
    registry base, o crear uno vacío con MarketRegistry() y registrar
    solo lo que necesitan.

    Retorna una nueva instancia en cada llamada — no es un singleton.
    Esto permite que cada test o pipeline tenga su propio registry
    sin estado compartido.
    """
    registry = MarketRegistry()
    registry.register_many(_UNIVERSAL_CORE)
    registry.register_many(_UNIVERSAL_EXTENDED)
    registry.register_many(_MLB_MARKETS)
    registry.register_many(_SOCCER_MARKETS)
    registry.register_many(_NBA_MARKETS)
    return registry