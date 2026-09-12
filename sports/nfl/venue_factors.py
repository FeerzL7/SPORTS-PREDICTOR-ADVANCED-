"""
sports/nfl/venue_factors.py

NFLVenueFactors: propiedades de estadio y ajustes derivados.

Diferencia estructural con el módulo equivalente de MLB
---------------------------------------------------------
sports/mlb/venue_factors.py devuelve un MULTIPLICADOR: Coors Field
1.36, Petco Park 0.91. Funciona porque en béisbol el estadio afecta
las carreras de forma proporcional — la altitud de Denver alarga los
batazos y eso se traduce directamente en más anotación.

En NFL los factores de estadio son de cuatro naturalezas distintas y
mezclarlas en un solo multiplicador perdería información:

    Techo        → BOOLEANO. Determina si el clima aplica o no.
                   Un partido en domo ignora viento y temperatura.

    Altitud      → MULTIPLICADOR, pero mucho más débil que en MLB.
                   Denver inflaba +36% las carreras en Coors; en NFL
                   el efecto es ~+2.5% sobre el total.

    Huso horario → ADITIVO y depende de AMBOS equipos, no del estadio
                   solo. Un partido en Buffalo perjudica a Seattle por
                   el desfase, pero no a Nueva Inglaterra.

    Superficie   → Sin ajuste de anotación. El efecto documentado de
                   grass vs turf es sobre riesgo de lesión, que este
                   sistema no modela.

Por eso la API expone métodos separados en vez de un único `get()`
que devuelva un factor agregado.

El ajuste de viaje: lo que quedó pendiente de 10.6
----------------------------------------------------
NFLRestFetcher cubre el descanso entre partidos pero dejó fuera el
efecto de viaje, que requiere coordenadas y husos horarios.

El caso mejor documentado: un equipo de la costa oeste jugando a las
13:00 ET tiene el reloj biológico en las 10:00 de la mañana. Los
estudios sobre rendimiento circadiano en deportes profesionales
sitúan el efecto entre 1 y 2 puntos de spread.

El efecto NO es simétrico. Viajar al este para un partido temprano es
peor que viajar al oeste para uno tardío: en el segundo caso el
desfase juega a favor (el cuerpo cree que es más tarde, cerca del
pico de rendimiento vespertino).

Prioridad del dato de partido sobre el catálogo
-------------------------------------------------
El catálogo estático sabe que Arizona tiene techo retráctil, pero no
si estuvo abierto en un partido concreto — eso depende del clima y de
la decisión del equipo local ese día. nflverse SÍ lo sabe: el campo
`roof` de cada partido trae 'open' o 'closed' para los retráctiles.

Los métodos aceptan un `game_roof` opcional que, cuando está presente,
prevalece sobre el catálogo. Sin esa prioridad, un partido en Arizona
con techo cerrado recibiría ajuste de clima que no corresponde.

Fuente de las coordenadas y altitudes
---------------------------------------
Coordenadas verificadas contra los datos públicos de los estadios.
Altitudes en pies sobre el nivel del mar. Solo Denver (5280 ft) tiene
magnitud suficiente para justificar un ajuste; el siguiente más alto
es Arizona (~1070 ft), donde el efecto es indistinguible del ruido.
"""

from __future__ import annotations

from dataclasses import dataclass


# ── Tipos de techo ───────────────────────────────────────────────────────────

ROOF_OUTDOORS    = "outdoors"
ROOF_DOME        = "dome"         # techo fijo cerrado
ROOF_CLOSED      = "closed"       # retráctil, cerrado ese partido
ROOF_OPEN        = "open"         # retráctil, abierto ese partido
ROOF_RETRACTABLE = "retractable"  # catálogo: tiene retráctil, estado desconocido

# Techos bajo los que el clima es irrelevante
_WEATHER_PROOF = frozenset({ROOF_DOME, ROOF_CLOSED})


# ── Husos horarios ───────────────────────────────────────────────────────────
# Offset respecto a la hora del Este, que es el estándar de kickoff NFL.
#   0 = Eastern, -1 = Central, -2 = Mountain, -3 = Pacific
#
# Arizona es el caso especial: no observa horario de verano. Durante la
# mayor parte de la temporada NFL (septiembre-octubre, cuando el resto
# del país está en DST) Arizona coincide con Pacific, no con Mountain.
# Se modela como -3 porque ese es el desfase efectivo en la ventana
# donde se juegan más partidos.

TZ_EASTERN  = 0
TZ_CENTRAL  = -1
TZ_MOUNTAIN = -2
TZ_PACIFIC  = -3


# ── Catálogo de estadios ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class NFLVenueInfo:
    """
    Propiedades estáticas del estadio de una franquicia.

    Se indexa por abreviación del equipo local, coherente con la
    decisión de schedule.py de usar `venue_id = home_team`: nflverse no
    expone un venue_id estable, pero cada franquicia juega en un
    estadio fijo durante la temporada.

    Campos
    ------
    team        -- Abreviación de la franquicia local.
    stadium     -- Nombre del estadio.
    lat / lon   -- Coordenadas para NFLContextFetcher (clima).
    altitude_ft -- Altitud en pies sobre el nivel del mar.
    roof        -- Tipo de techo según el catálogo. Para retráctiles
                   indica ROOF_RETRACTABLE: el estado real de cada
                   partido lo aporta nflverse.
    surface     -- 'grass' o 'turf'.
    tz_offset   -- Desfase respecto a hora del Este (0/-1/-2/-3).
    """
    team:        str
    stadium:     str
    lat:         float
    lon:         float
    altitude_ft: int
    roof:        str
    surface:     str
    tz_offset:   int

    @property
    def is_weatherproof(self) -> bool:
        """
        True si el catálogo garantiza que el clima no aplica.

        Los retráctiles devuelven False: podrían estar abiertos. Para
        saberlo hay que consultar el dato del partido concreto.
        """
        return self.roof in _WEATHER_PROOF


# Catálogo de los 32 estadios.
#
# Comparticiones: LAR y LAC juegan en SoFi Stadium; NYG y NYJ en
# MetLife. Se registran por separado porque el equipo local determina
# quién NO viaja, y eso importa para el ajuste de husos horarios.
#
# SoFi merece una nota: tiene techo fijo translúcido pero laterales
# abiertos, así que el viento y la temperatura sí influyen. nflverse lo
# clasifica como 'outdoors' y se respeta ese criterio.
_VENUES: dict[str, NFLVenueInfo] = {
    # ── AFC East ──
    "BUF": NFLVenueInfo("BUF", "Highmark Stadium",       42.7738, -78.7870,  600, ROOF_OUTDOORS,    "turf",  TZ_EASTERN),
    "MIA": NFLVenueInfo("MIA", "Hard Rock Stadium",      25.9580, -80.2389,    8, ROOF_OUTDOORS,    "grass", TZ_EASTERN),
    "NE":  NFLVenueInfo("NE",  "Gillette Stadium",       42.0909, -71.2643,  289, ROOF_OUTDOORS,    "turf",  TZ_EASTERN),
    "NYJ": NFLVenueInfo("NYJ", "MetLife Stadium",        40.8135, -74.0745,    7, ROOF_OUTDOORS,    "turf",  TZ_EASTERN),

    # ── AFC North ──
    "BAL": NFLVenueInfo("BAL", "M&T Bank Stadium",       39.2780, -76.6227,   33, ROOF_OUTDOORS,    "grass", TZ_EASTERN),
    "CIN": NFLVenueInfo("CIN", "Paycor Stadium",         39.0955, -84.5161,  490, ROOF_OUTDOORS,    "turf",  TZ_EASTERN),
    "CLE": NFLVenueInfo("CLE", "Cleveland Browns Stadium", 41.5061, -81.6995, 571, ROOF_OUTDOORS,   "grass", TZ_EASTERN),
    "PIT": NFLVenueInfo("PIT", "Acrisure Stadium",       40.4468, -80.0158,  740, ROOF_OUTDOORS,    "grass", TZ_EASTERN),

    # ── AFC South ──
    "HOU": NFLVenueInfo("HOU", "NRG Stadium",            29.6847, -95.4107,   49, ROOF_RETRACTABLE, "turf",  TZ_CENTRAL),
    "IND": NFLVenueInfo("IND", "Lucas Oil Stadium",      39.7601, -86.1639,  715, ROOF_RETRACTABLE, "turf",  TZ_EASTERN),
    "JAX": NFLVenueInfo("JAX", "EverBank Stadium",       30.3239, -81.6373,   16, ROOF_OUTDOORS,    "grass", TZ_EASTERN),
    "TEN": NFLVenueInfo("TEN", "Nissan Stadium",         36.1665, -86.7713,  450, ROOF_OUTDOORS,    "grass", TZ_CENTRAL),

    # ── AFC West ──
    "DEN": NFLVenueInfo("DEN", "Empower Field at Mile High", 39.7439, -105.0201, 5280, ROOF_OUTDOORS, "grass", TZ_MOUNTAIN),
    "KC":  NFLVenueInfo("KC",  "Arrowhead Stadium",      39.0489, -94.4839,  750, ROOF_OUTDOORS,    "grass", TZ_CENTRAL),
    "LV":  NFLVenueInfo("LV",  "Allegiant Stadium",      36.0909, -115.1833, 2030, ROOF_DOME,       "grass", TZ_PACIFIC),
    "LAC": NFLVenueInfo("LAC", "SoFi Stadium",           33.9535, -118.3392,  125, ROOF_OUTDOORS,   "turf",  TZ_PACIFIC),

    # ── NFC East ──
    "DAL": NFLVenueInfo("DAL", "AT&T Stadium",           32.7473, -97.0945,  595, ROOF_RETRACTABLE, "turf",  TZ_CENTRAL),
    "NYG": NFLVenueInfo("NYG", "MetLife Stadium",        40.8135, -74.0745,    7, ROOF_OUTDOORS,    "turf",  TZ_EASTERN),
    "PHI": NFLVenueInfo("PHI", "Lincoln Financial Field", 39.9008, -75.1675,   39, ROOF_OUTDOORS,   "grass", TZ_EASTERN),
    "WAS": NFLVenueInfo("WAS", "Northwest Stadium",      38.9076, -76.8645,  200, ROOF_OUTDOORS,    "grass", TZ_EASTERN),

    # ── NFC North ──
    "CHI": NFLVenueInfo("CHI", "Soldier Field",          41.8623, -87.6167,  597, ROOF_OUTDOORS,    "grass", TZ_CENTRAL),
    "DET": NFLVenueInfo("DET", "Ford Field",             42.3400, -83.0456,  600, ROOF_DOME,        "turf",  TZ_EASTERN),
    "GB":  NFLVenueInfo("GB",  "Lambeau Field",          44.5013, -88.0622,  640, ROOF_OUTDOORS,    "grass", TZ_CENTRAL),
    "MIN": NFLVenueInfo("MIN", "U.S. Bank Stadium",      44.9738, -93.2578,  830, ROOF_DOME,        "turf",  TZ_CENTRAL),

    # ── NFC South ──
    "ATL": NFLVenueInfo("ATL", "Mercedes-Benz Stadium",  33.7554, -84.4008, 1050, ROOF_RETRACTABLE, "turf",  TZ_EASTERN),
    "CAR": NFLVenueInfo("CAR", "Bank of America Stadium", 35.2258, -80.8528, 751, ROOF_OUTDOORS,    "turf",  TZ_EASTERN),
    "NO":  NFLVenueInfo("NO",  "Caesars Superdome",      29.9511, -90.0812,    3, ROOF_DOME,        "turf",  TZ_CENTRAL),
    "TB":  NFLVenueInfo("TB",  "Raymond James Stadium",  27.9759, -82.5033,   26, ROOF_OUTDOORS,    "grass", TZ_EASTERN),

    # ── NFC West ──
    "ARI": NFLVenueInfo("ARI", "State Farm Stadium",     33.5276, -112.2626, 1070, ROOF_RETRACTABLE, "grass", TZ_PACIFIC),
    "LAR": NFLVenueInfo("LAR", "SoFi Stadium",           33.9535, -118.3392,  125, ROOF_OUTDOORS,   "turf",  TZ_PACIFIC),
    "SF":  NFLVenueInfo("SF",  "Levi's Stadium",         37.4030, -121.9698,   26, ROOF_OUTDOORS,   "grass", TZ_PACIFIC),
    "SEA": NFLVenueInfo("SEA", "Lumen Field",            47.5952, -122.3316,   26, ROOF_OUTDOORS,   "turf",  TZ_PACIFIC),
}

# Aliases de franquicias reubicadas o renombradas, para que el
# backtesting histórico resuelva los estadios correctamente.
_TEAM_ALIASES: dict[str, str] = {
    "OAK": "LV",    # Oakland Raiders → Las Vegas
    "SD":  "LAC",   # San Diego Chargers → Los Angeles
    "STL": "LAR",   # St. Louis Rams → Los Angeles
    "LA":  "LAR",   # abreviación ambigua usada en algunos datasets
    "WSH": "WAS",   # variante de Washington
    "ARZ": "ARI",   # variante de Arizona
}


# ── Constantes de calibración ────────────────────────────────────────────────

# Umbral de altitud para aplicar ajuste. Solo Denver (5280 ft) lo supera.
# El siguiente estadio más alto es Las Vegas (2030 ft, y es domo), luego
# Arizona (1070 ft) — alturas donde el efecto es indistinguible del ruido.
_ALTITUDE_THRESHOLD_FT: int = 3000

# Multiplicador del total en altitud. MUY inferior al de MLB (1.36 en
# Coors) porque la anotación en football no depende del vuelo del balón
# como en béisbol. El efecto documentado en Denver actúa sobre dos vías:
# field goals de mayor distancia y fatiga del visitante en el último
# cuarto. Ambas son reales pero modestas.
_ALTITUDE_TOTAL_FACTOR: float = 1.025

# Penalización en puntos para el visitante por desfase horario.
#
# El efecto NO es simétrico. Viajar al este para un partido temprano es
# lo más perjudicial: un equipo de la costa oeste jugando a las 13:00 ET
# tiene el reloj biológico en las 10:00. Viajar al oeste para un partido
# tardío puede incluso favorecer, porque el cuerpo cree que es más tarde
# y está más cerca del pico de rendimiento vespertino.
_TZ_PENALTY_PER_ZONE_EARLY: float = 0.5   # viaje al este, kickoff temprano
_TZ_PENALTY_PER_ZONE_LATE:  float = 0.15  # viaje al este, kickoff tardío
_TZ_BONUS_WESTWARD:         float = 0.10  # viaje al oeste: leve ventaja
_TZ_MAX_PENALTY:            float = 2.0   # techo del ajuste total

# Hora (ET) por debajo de la cual un kickoff cuenta como "temprano".
# Los partidos de las 13:00 ET son el caso problemático para equipos
# de la costa oeste; los de 16:25 y los nocturnos ya no.
_EARLY_KICKOFF_HOUR_ET: int = 15


class NFLVenueFactors:
    """
    Propiedades de estadio y ajustes derivados para NFL.

    Parámetros
    ----------
    config_loader -- ConfigLoader con nfl.yaml. Permite sobreescribir
                     los factores desde YAML sin tocar código, igual
                     que el módulo equivalente de MLB.
    """

    def __init__(self, config_loader = None) -> None:
        self._config = config_loader
        self._cache: dict[str, float] = {}

    # ── Consulta del catálogo ─────────────────────────────────────────────────

    def get(self, team: str) -> NFLVenueInfo | None:
        """
        Propiedades del estadio de una franquicia.

        Resuelve aliases de equipos reubicados para que el backtesting
        histórico funcione (OAK → LV, SD → LAC, STL → LAR).

        Retorna None si la abreviación no se reconoce.
        """
        return _VENUES.get(self._normalize(team))

    def get_all(self) -> dict[str, NFLVenueInfo]:
        """Catálogo completo de los 32 estadios."""
        return dict(_VENUES)

    # ── Techo y aplicabilidad del clima ───────────────────────────────────────

    def is_weatherproof(
        self,
        team:      str,
        game_roof: str | None = None,
    ) -> bool:
        """
        True si el clima NO afecta al partido.

        El parámetro `game_roof` tiene PRIORIDAD sobre el catálogo.
        Razón: el catálogo sabe que Arizona tiene techo retráctil, pero
        no si estuvo abierto ese día — eso depende del clima y de la
        decisión del equipo local. nflverse sí lo sabe y lo expone en
        el campo `roof` de cada partido ('open' o 'closed').

        Sin esta prioridad, un partido en Arizona con techo cerrado
        recibiría ajustes de viento y temperatura que no corresponden,
        y uno con techo abierto los perdería.
        """
        if game_roof:
            return game_roof.strip().lower() in _WEATHER_PROOF

        venue = self.get(team)
        return venue.is_weatherproof if venue else False

    def effective_roof(
        self,
        team:      str,
        game_roof: str | None = None,
    ) -> str:
        """
        Tipo de techo efectivo, priorizando el dato del partido.

        Devuelve ROOF_OUTDOORS para equipos desconocidos: es el
        supuesto conservador, porque asumir domo desactivaría los
        ajustes de clima sin justificación.
        """
        if game_roof:
            return game_roof.strip().lower()
        venue = self.get(team)
        return venue.roof if venue else ROOF_OUTDOORS

    # ── Altitud ───────────────────────────────────────────────────────────────

    def total_factor(self, team: str) -> float:
        """
        Multiplicador del total de puntos por altitud.

        1.0 para 31 de los 32 estadios. Solo Denver supera el umbral de
        3000 pies con margen suficiente para justificar un ajuste.

        El valor (1.025) es deliberadamente modesto comparado con el
        equivalente de MLB (Coors Field 1.36). En béisbol la altitud
        alarga los batazos y eso se traduce en carreras directamente;
        en football el efecto actúa por vías más indirectas (field
        goals de más distancia, fatiga del visitante) cuyo impacto
        agregado sobre la anotación es mucho menor.
        """
        key = self._normalize(team)
        if key in self._cache:
            return self._cache[key]

        factor = self._resolve_total_factor(key)
        self._cache[key] = factor
        return factor

    def is_high_altitude(self, team: str) -> bool:
        """True si el estadio está por encima del umbral de altitud."""
        venue = self.get(team)
        return venue is not None and venue.altitude_ft >= _ALTITUDE_THRESHOLD_FT

    # ── Viaje y husos horarios ────────────────────────────────────────────────

    def travel_adjustment(
        self,
        home_team:    str,
        away_team:    str,
        kickoff_hour_et: int | None = None,
    ) -> float:
        """
        Penalización en puntos para el equipo VISITANTE por el viaje.

        Retorna un valor negativo o cero: nunca penaliza al local, que
        no viaja. El signo está desde la perspectiva del visitante, así
        que el modelo lo suma a la proyección del visitante o lo resta
        del spread.

        Lógica del desfase
        ------------------
        El efecto no es simétrico y depende de dos cosas: la dirección
        del viaje y la hora del kickoff.

            Oeste → Este, kickoff temprano (13:00 ET)
                El caso más perjudicial. Un equipo de Seattle jugando a
                las 13:00 ET tiene el reloj biológico en las 10:00.
                Penalización de 0.5 puntos por huso cruzado.

            Oeste → Este, kickoff tardío (16:25 ET o nocturno)
                Mucho menos dañino: el cuerpo ya está en horario de
                rendimiento. 0.15 puntos por huso.

            Este → Oeste
                Puede favorecer ligeramente. El cuerpo cree que es más
                tarde de lo que marca el reloj local, más cerca del
                pico vespertino. Bonificación de 0.10 por huso.

        Parámetros
        ----------
        home_team       -- Abreviación del local (define el estadio).
        away_team       -- Abreviación del visitante (define el origen).
        kickoff_hour_et -- Hora del kickoff en horario del Este. Si es
                           None se asume tardío, el supuesto conservador
                           (penaliza menos ante datos ausentes).
        """
        home = self.get(home_team)
        away = self.get(away_team)
        if home is None or away is None:
            return 0.0

        # Husos cruzados. Positivo = el visitante viaja hacia el este.
        # Ej: SEA (-3) jugando en BUF (0) → 0 - (-3) = 3 husos al este.
        zones_east = home.tz_offset - away.tz_offset

        if zones_east == 0:
            return 0.0

        if zones_east > 0:
            # Viaje al este: perjudicial, más aún con kickoff temprano
            is_early = (
                kickoff_hour_et is not None
                and kickoff_hour_et < _EARLY_KICKOFF_HOUR_ET
            )
            per_zone = (
                _TZ_PENALTY_PER_ZONE_EARLY if is_early
                else _TZ_PENALTY_PER_ZONE_LATE
            )
            penalty = -per_zone * zones_east
            return round(max(penalty, -_TZ_MAX_PENALTY), 3)

        # Viaje al oeste: leve ventaja circadiana
        bonus = _TZ_BONUS_WESTWARD * abs(zones_east)
        return round(min(bonus, _TZ_MAX_PENALTY), 3)

    def timezone_delta(self, home_team: str, away_team: str) -> int:
        """
        Husos horarios que cruza el visitante. Positivo = hacia el este.

        Expuesto aparte del ajuste para que el modelo pueda registrarlo
        en `model_inputs` como dato de trazabilidad.
        """
        home = self.get(home_team)
        away = self.get(away_team)
        if home is None or away is None:
            return 0
        return home.tz_offset - away.tz_offset

    # ── Metadatos ─────────────────────────────────────────────────────────────

    def to_metadata(
        self,
        home_team: str,
        away_team: str,
        game_roof: str | None = None,
        kickoff_hour_et: int | None = None,
    ) -> dict:
        """Dict de factores de estadio para TeamFeatures.sport_metadata."""
        venue = self.get(home_team)
        return {
            "venue_stadium":     venue.stadium if venue else None,
            "venue_altitude_ft": venue.altitude_ft if venue else None,
            "venue_surface":     venue.surface if venue else None,
            "venue_roof":        self.effective_roof(home_team, game_roof),
            "venue_weatherproof": self.is_weatherproof(home_team, game_roof),
            "venue_total_factor": self.total_factor(home_team),
            "travel_tz_delta":   self.timezone_delta(home_team, away_team),
            "travel_adjustment": self.travel_adjustment(
                home_team, away_team, kickoff_hour_et
            ),
        }

    def clear_cache(self) -> None:
        """Limpia el caché de factores resueltos."""
        self._cache.clear()

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _resolve_total_factor(self, key: str) -> float:
        """
        Resuelve el multiplicador de total por prioridad:
        1. YAML (nfl.venue_factors.{team}), 2. altitud, 3. neutro.
        """
        if self._config is not None:
            try:
                val = self._config.get(
                    f"nfl.venue_factors.{key.lower()}", default=None
                )
                if val is not None:
                    return float(val)
            except (ValueError, TypeError, AttributeError):
                pass

        venue = _VENUES.get(key)
        if venue is None:
            return 1.0
        if venue.altitude_ft >= _ALTITUDE_THRESHOLD_FT:
            return _ALTITUDE_TOTAL_FACTOR
        return 1.0

    @staticmethod
    def _normalize(team: str) -> str:
        """Normaliza la abreviación y resuelve aliases históricos."""
        abbr = str(team).strip().upper()
        return _TEAM_ALIASES.get(abbr, abbr)


# ── Función de conveniencia ───────────────────────────────────────────────────

def get_venue(team: str) -> NFLVenueInfo | None:
    """
    Propiedades del estadio de un equipo sin instanciar la clase.

    Útil para scripts y tests. El pipeline usa la instancia inyectada
    por NFLPlugin para aprovechar el caché.
    """
    return NFLVenueFactors().get(team)