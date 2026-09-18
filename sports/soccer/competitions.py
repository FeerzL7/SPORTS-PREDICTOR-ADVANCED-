"""
sports/soccer/competitions.py

Registro de competiciones de fútbol.

Por qué esto es la primera pieza del plugin
---------------------------------------------
MLB y NFL son cada uno UNA competición: 30 y 32 equipos jugando un
único torneo con un calendario y un conjunto de reglas. El plugin
podía tratar "el deporte" y "la competición" como sinónimos.

En fútbol no. Un equipo del Manchester City juega Premier League,
Champions, FA Cup y Carabao simultáneamente, con reglas distintas en
cada una. Y las competiciones difieren entre sí más de lo que difieren
MLB y NFL en varios aspectos:

    Calendario     Europa va de agosto a mayo; Liga MX tiene DOS
                   torneos por año natural; Brasileirão sigue el año
                   calendario.

    Formato        Liga regular con ida y vuelta, fase de grupos,
                   eliminatorias a doble partido.

    Empates        Una liga los permite; una eliminatoria no (hay
                   prórroga y penaltis), lo que cambia el mercado
                   disponible.

    Datos          xG gratis para las cinco grandes europeas; parcial
                   o inexistente para otras.

La decisión de diseño: UN plugin `soccer` con las competiciones como
configuración, no un plugin por competición. Comparten el 90% de la
lógica —proyección de goles, Dixon-Coles, liquidación de 1X2— y
duplicarla sería el error que evitamos en el Core con h2h_base.

Lo que varía entre competiciones vive aquí, en datos, no en código.

Tiers de calidad de datos
---------------------------
No todas las competiciones se pueden modelar con la misma precisión, y
el sistema debe declararlo en vez de fingir uniformidad.

    TIER_FULL     xG por partido y por equipo, más cuotas históricas
                  de cierre. Permite el modelo completo y backtesting
                  contra el cierre.

    TIER_PARTIAL  Resultados y cuotas, sin xG. El modelo funciona con
                  goles y forma, notablemente menos predictivo:
                  la correlación de xG con resultados futuros ronda
                  0.60, la de goles marcados ronda 0.40.

    TIER_MINIMAL  Solo resultados. Sin cuotas históricas no hay
                  backtest posible, así que estas competiciones no
                  deberían operarse con dinero real hasta tener
                  histórico propio.

`data_quality` del provider se calibra con este tier, y de ahí baja la
`confidence` del modelo — el mismo encadenamiento que en NFL hace que
los filtros rechacen picks sobre datos pobres.

Alcance actual
----------------
Las cinco grandes europeas, que es exactamente la cobertura de
Understat para xG. Champions League, Liga MX y Brasileirão quedan
declaradas con sus identificadores pero desactivadas: añadirlas es
cambiar un flag y validar la fuente, no reescribir el plugin.

Libertadores y Concachampions quedan fuera deliberadamente: sin xG
fiable el modelo sería de otra categoría de precisión, y mezclarlas
con las demás daría una falsa impresión de homogeneidad.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Tiers de datos ───────────────────────────────────────────────────────────

TIER_FULL    = "full"      # xG + cuotas históricas
TIER_PARTIAL = "partial"   # cuotas, sin xG
TIER_MINIMAL = "minimal"   # solo resultados

# Peso de calidad de datos que cada tier aporta al provider.
# Un tier partial arranca en 0.70 porque pierde la señal principal
# (xG) pero conserva resultados, forma y cuotas.
_TIER_QUALITY: dict[str, float] = {
    TIER_FULL:    1.00,
    TIER_PARTIAL: 0.70,
    TIER_MINIMAL: 0.45,
}


# ── Formatos de calendario ───────────────────────────────────────────────────

# La temporada cruza el año natural: 2024 significa 2024/25.
# Es el caso de todas las ligas europeas.
SEASON_SPLIT_YEAR = "split_year"

# La temporada coincide con el año natural. Brasileirão, MLS.
SEASON_CALENDAR = "calendar"

# Dos torneos independientes por año natural. Liga MX (Apertura y
# Clausura), y varias ligas sudamericanas. Cada torneo tiene su propia
# tabla y su propio campeón, así que la forma de un equipo NO se
# arrastra entre ellos como sí se arrastra dentro de una temporada.
SEASON_SPLIT_TOURNAMENT = "split_tournament"


# ── Formatos de competición ──────────────────────────────────────────────────

FORMAT_LEAGUE     = "league"       # todos contra todos, ida y vuelta
FORMAT_GROUP_KO   = "group_ko"     # grupos + eliminatorias
FORMAT_KNOCKOUT   = "knockout"     # eliminación directa


@dataclass(frozen=True)
class Competition:
    """
    Una competición de fútbol y sus identificadores por fuente.

    Campos de identidad
    -------------------
    comp_id   -- Identificador interno, usado como clave en todo el
                 plugin y en las claves de configuración de
                 soccer.yaml.
    name      -- Nombre para mostrar.
    country   -- País o confederación.

    Identificadores externos
    ------------------------
    Cada fuente usa su propia nomenclatura para la misma competición.
    Centralizarlos aquí evita que aparezcan dispersos por los fetchers,
    que es donde se convierten en constantes mágicas imposibles de
    auditar.

    understat_id        -- 'EPL', 'La_liga', 'Serie_A'...
    fbref_id            -- 'ENG-Premier League', formato de soccerdata.
    football_data_code  -- 'E0', 'SP1'... Códigos de
                           football-data.co.uk, que es la fuente de
                           cuotas históricas de cierre.
    odds_api_key        -- 'soccer_epl'. Identificador de The Odds API,
                           equivalente a odds_api_sport_id en NFL.

    Características del torneo
    --------------------------
    tier          -- Calidad de datos disponible (ver la nota del módulo).
    season_type   -- Cómo se numeran las temporadas.
    comp_format   -- Liga, grupos+eliminatorias o eliminación directa.
    n_teams       -- Equipos participantes. Determina cuántos partidos
                     juega cada uno y por tanto el tamaño de muestra
                     disponible.
    allows_draws  -- False en eliminatorias a partido único, donde hay
                     prórroga y penaltis. Cambia qué mercados existen:
                     sin empate no hay 1X2, hay moneyline a dos vías.
    enabled       -- Si el plugin la procesa. Permite declarar
                     competiciones futuras sin activarlas.

    Parámetros de liga
    ------------------
    avg_home_goals / avg_away_goals
        Medias históricas de la competición. Difieren de forma
        apreciable entre ligas: la Bundesliga promedia ~3.1 goles por
        partido y la Ligue 1 ~2.6. Usar una media global metería un
        sesgo sistemático en los totales de ambas.
    """
    comp_id:  str
    name:     str
    country:  str

    understat_id:       str | None = None
    fbref_id:           str | None = None
    football_data_code: str | None = None
    odds_api_key:       str = ""

    tier:         str = TIER_MINIMAL
    season_type:  str = SEASON_SPLIT_YEAR
    comp_format:  str = FORMAT_LEAGUE
    n_teams:      int = 20
    allows_draws: bool = True
    enabled:      bool = False

    avg_home_goals: float = 1.45
    avg_away_goals: float = 1.15

    # ── Propiedades derivadas ────────────────────────────────────────────────

    @property
    def has_xg(self) -> bool:
        """True si hay datos de xG para esta competición."""
        return self.tier == TIER_FULL

    @property
    def has_historical_odds(self) -> bool:
        """
        True si hay cuotas históricas de cierre para backtesting.

        Sin ellas no se puede backtestear contra el mercado, que es la
        única validación previa a arriesgar capital. Una competición
        sin esto no debería operarse hasta acumular histórico propio.
        """
        return self.tier in (TIER_FULL, TIER_PARTIAL)

    @property
    def data_quality_base(self) -> float:
        """Calidad de datos base que aporta el tier de la competición."""
        return _TIER_QUALITY.get(self.tier, _TIER_QUALITY[TIER_MINIMAL])

    @property
    def matches_per_team(self) -> int:
        """
        Partidos que juega cada equipo en la fase regular.

        Determina el tamaño de muestra disponible para el modelo, igual
        que los 17 partidos de NFL determinaban allí la agresividad del
        shrinkage. Una liga de 20 equipos da 38 partidos: más del doble
        que NFL, pero muy por debajo de los 162 de MLB.
        """
        if self.comp_format == FORMAT_LEAGUE:
            return (self.n_teams - 1) * 2
        if self.comp_format == FORMAT_GROUP_KO:
            return 6   # fase de grupos típica
        return 1

    @property
    def avg_total_goals(self) -> float:
        """Media de goles por partido de la competición."""
        return round(self.avg_home_goals + self.avg_away_goals, 3)

    def season_label(self, season: int) -> str:
        """
        Etiqueta legible de una temporada.

        Con SEASON_SPLIT_YEAR, el entero 2024 designa la temporada
        2024/25. Esa convención es la que usan Understat y FBref, y
        coincide con la decisión que tomamos en NFL: la temporada se
        identifica por su año de inicio.
        """
        if self.season_type == SEASON_SPLIT_YEAR:
            return f"{season}/{str(season + 1)[-2:]}"
        return str(season)

    def football_data_season(self, season: int) -> str:
        """
        Código de temporada de football-data.co.uk.

        Esa fuente usa el formato '2425' para 2024/25, sin separador y
        con dos dígitos por año — distinto tanto del entero que usan
        Understat y FBref como de la etiqueta legible.
        """
        if self.season_type == SEASON_SPLIT_YEAR:
            return f"{str(season)[-2:]}{str(season + 1)[-2:]}"
        return str(season)


# ── Catálogo ─────────────────────────────────────────────────────────────────
#
# Las medias de goles son valores históricos estables de cada
# competición. No son arbitrarias: la Bundesliga promedia sistemática y
# apreciablemente más goles que la Ligue 1, y usar una media global
# sesgaría los totales de ambas en direcciones opuestas.

_COMPETITIONS: dict[str, Competition] = {

    # ── Las cinco grandes europeas ───────────────────────────────
    # Cobertura exacta de Understat para xG, y de football-data.co.uk
    # para cuotas históricas de cierre.

    "epl": Competition(
        comp_id="epl", name="Premier League", country="Inglaterra",
        understat_id="EPL", fbref_id="ENG-Premier League",
        football_data_code="E0", odds_api_key="soccer_epl",
        tier=TIER_FULL, n_teams=20, enabled=True,
        avg_home_goals=1.53, avg_away_goals=1.25,
    ),
    "laliga": Competition(
        comp_id="laliga", name="La Liga", country="España",
        understat_id="La_liga", fbref_id="ESP-La Liga",
        football_data_code="SP1", odds_api_key="soccer_spain_la_liga",
        tier=TIER_FULL, n_teams=20, enabled=True,
        avg_home_goals=1.43, avg_away_goals=1.10,
    ),
    "seriea": Competition(
        comp_id="seriea", name="Serie A", country="Italia",
        understat_id="Serie_A", fbref_id="ITA-Serie A",
        football_data_code="I1", odds_api_key="soccer_italy_serie_a",
        tier=TIER_FULL, n_teams=20, enabled=True,
        avg_home_goals=1.48, avg_away_goals=1.20,
    ),
    "bundesliga": Competition(
        comp_id="bundesliga", name="Bundesliga", country="Alemania",
        understat_id="Bundesliga", fbref_id="GER-Bundesliga",
        football_data_code="D1", odds_api_key="soccer_germany_bundesliga",
        tier=TIER_FULL, n_teams=18, enabled=True,
        avg_home_goals=1.72, avg_away_goals=1.38,
    ),
    "ligue1": Competition(
        comp_id="ligue1", name="Ligue 1", country="Francia",
        understat_id="Ligue_1", fbref_id="FRA-Ligue 1",
        football_data_code="F1", odds_api_key="soccer_france_ligue_one",
        tier=TIER_FULL, n_teams=18, enabled=True,
        avg_home_goals=1.48, avg_away_goals=1.15,
    ),

    # ── Declaradas, no activas ───────────────────────────────────
    # Añadir cualquiera de estas es cambiar `enabled` y validar que su
    # fuente responde — no reescribir el plugin. Esa es la razón de
    # que el registro exista.

    "ucl": Competition(
        comp_id="ucl", name="Champions League", country="UEFA",
        fbref_id="INT-Champions League", odds_api_key="soccer_uefa_champs_league",
        tier=TIER_PARTIAL, comp_format=FORMAT_GROUP_KO,
        n_teams=36, enabled=False,
        avg_home_goals=1.55, avg_away_goals=1.22,
    ),
    "ligamx": Competition(
        comp_id="ligamx", name="Liga MX", country="México",
        fbref_id="MEX-Liga MX", odds_api_key="soccer_mexico_ligamx",
        tier=TIER_PARTIAL, season_type=SEASON_SPLIT_TOURNAMENT,
        n_teams=18, enabled=False,
        avg_home_goals=1.58, avg_away_goals=1.18,
    ),
    "brasileirao": Competition(
        comp_id="brasileirao", name="Brasileirão Série A", country="Brasil",
        fbref_id="BRA-Serie A", odds_api_key="soccer_brazil_campeonato",
        tier=TIER_PARTIAL, season_type=SEASON_CALENDAR,
        n_teams=20, enabled=False,
        avg_home_goals=1.35, avg_away_goals=1.00,
    ),
}


# ── API del registro ─────────────────────────────────────────────────────────

def get_competition(comp_id: str) -> Competition | None:
    """Competición por su identificador interno, o None."""
    return _COMPETITIONS.get(str(comp_id).strip().lower())


def enabled_competitions() -> list[Competition]:
    """
    Competiciones activas, en orden de declaración.

    El pipeline itera sobre estas. Una competición declarada pero
    desactivada no genera peticiones a ninguna fuente ni consume
    créditos de API.
    """
    return [c for c in _COMPETITIONS.values() if c.enabled]


def all_competitions() -> list[Competition]:
    """Todas las competiciones del catálogo, activas o no."""
    return list(_COMPETITIONS.values())


def competitions_with_xg() -> list[Competition]:
    """Competiciones activas con datos de xG disponibles."""
    return [c for c in enabled_competitions() if c.has_xg]


def by_understat_id(understat_id: str) -> Competition | None:
    """Competición desde su identificador de Understat."""
    target = str(understat_id).strip()
    for comp in _COMPETITIONS.values():
        if comp.understat_id == target:
            return comp
    return None


def by_odds_api_key(key: str) -> Competition | None:
    """
    Competición desde su identificador de The Odds API.

    Necesaria en el camino inverso: cuando llega una respuesta de
    cuotas hay que saber a qué competición pertenece para aplicar sus
    parámetros.
    """
    target = str(key).strip().lower()
    for comp in _COMPETITIONS.values():
        if comp.odds_api_key.lower() == target:
            return comp
    return None


def odds_api_keys() -> list[str]:
    """Claves de The Odds API de las competiciones activas."""
    return [c.odds_api_key for c in enabled_competitions() if c.odds_api_key]


def summary() -> dict:
    """Resumen del catálogo, para logs y diagnóstico."""
    enabled = enabled_competitions()
    return {
        "total":        len(_COMPETITIONS),
        "enabled":      len(enabled),
        "with_xg":      len(competitions_with_xg()),
        "enabled_ids":  [c.comp_id for c in enabled],
        "disabled_ids": [c.comp_id for c in _COMPETITIONS.values() if not c.enabled],
        "by_tier": {
            tier: [c.comp_id for c in _COMPETITIONS.values() if c.tier == tier]
            for tier in (TIER_FULL, TIER_PARTIAL, TIER_MINIMAL)
        },
    }