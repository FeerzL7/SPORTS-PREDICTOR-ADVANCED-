"""
sports/nfl/context.py

NFLContextFetcher: contexto situacional de partidos NFL.
Implementa SportDataProvider.get_context(event) → dict.

Dos diferencias sustantivas con el módulo equivalente de MLB
--------------------------------------------------------------

1. FUENTE DUAL DE CLIMA, CON PRIORIDAD AL DATO OBSERVADO

   El schedule de nflverse incluye `temp` y `wind` para los partidos
   ya jugados. Eso es MEJOR que un pronóstico: es el valor real medido,
   sin error de predicción.

   Open-Meteo solo hace falta para partidos futuros. La prioridad es:

       1. nflverse (observado)  → partidos pasados, backtesting
       2. Open-Meteo (forecast) → partidos próximos, producción
       3. Neutro (sin ajuste)   → sin datos

   Usar el pronóstico donde existe el dato observado introduciría en
   el backtest un error que no estuvo presente en la realidad, e
   inflaría o deflactaría el rendimiento histórico del modelo de forma
   artificial.

2. EN NFL IMPORTA LA VELOCIDAD DEL VIENTO, NO SU DIRECCIÓN

   El módulo de MLB modela la dirección del viento con trigonometría:
   viento hacia los jardines aumenta los home runs, viento de frente
   los reduce. Tiene sentido porque el vuelo del balón es el mecanismo
   directo de la anotación.

   En NFL no aplica por dos razones:

       El mecanismo es distinto. El viento cruzado degrada la precisión
       del pase y la fiabilidad del pateo INDEPENDIENTEMENTE de su
       dirección. Un viento lateral de 20 mph arruina un field goal
       igual que uno frontal.

       No tenemos el dato. La orientación de cada campo respecto al
       norte no está en nflverse. Calcular una componente direccional
       sin conocer la orientación del estadio produciría un número con
       apariencia de precisión y contenido de ruido.

   Por eso `weather_factor()` pondera velocidad y descarta dirección.
   La dirección se conserva en el dict de contexto por trazabilidad,
   pero no entra en el cálculo.

Rango climático mucho más extremo que MLB
-------------------------------------------
La temporada de béisbol va de abril a septiembre: el rango de
temperatura relevante es 50-95 °F. La NFL juega hasta febrero en Green
Bay, Buffalo y Chicago, con partidos bajo 0 °F documentados.

Eso obliga a una curva de ajuste por temperatura asimétrica y con más
recorrido en el extremo frío que la de MLB.

Cortocircuito en estadios cubiertos
-------------------------------------
Si el partido se juega bajo techo cerrado, `get_context()` retorna sin
consultar ninguna API de clima. La decisión de si el estadio está
cubierto se delega a NFLVenueFactors (10.7), que ya resuelve la
prioridad entre el catálogo estático y el estado real del techo
retráctil en ese partido concreto.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from core.contracts.event import Event

from sports.nfl.schedule import NFLGameInfo
from sports.nfl.venue_factors import NFLVenueFactors

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]


_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ── Dependencias inyectadas ──────────────────────────────────────────────────

@runtime_checkable
class ScheduleSource(Protocol):
    """
    Interfaz mínima del calendario que este módulo consume.

    Mismo patrón que rest.py: declarar solo los métodos usados documenta
    la dependencia real y permite inyectar dobles de prueba sin
    arrastrar toda la implementación de NFLScheduleFetcher. Sin la
    anotación, el type checker trataría los retornos como Unknown y
    propagaría esa indeterminación a cada punto de uso.
    """

    def get_game_info(self, game_id: str) -> NFLGameInfo | None:
        """Metadatos de un partido, o None si no existe."""
        ...


# ── Condiciones climáticas ───────────────────────────────────────────────────

CONDITION_CLEAR  = "clear"
CONDITION_CLOUDY = "cloudy"
CONDITION_RAIN   = "rain"
CONDITION_SNOW   = "snow"
CONDITION_UNKNOWN = "unknown"

SOURCE_OBSERVED = "nflverse"   # dato real medido
SOURCE_FORECAST = "open-meteo" # pronóstico
SOURCE_NONE     = "none"       # sin datos


# ── Curvas de ajuste ─────────────────────────────────────────────────────────
#
# VIENTO: el factor más impactante del clima en NFL. Degrada la precisión
# del pase y la fiabilidad del pateo. Umbrales en mph con el multiplicador
# resultante sobre el total de puntos.
#
# El efecto es no lineal: por debajo de 10 mph es indistinguible del
# ruido, y a partir de 20 mph se vuelve severo porque los equipos
# abandonan el juego aéreo profundo y los field goals largos.
_WIND_BANDS: tuple[tuple[float, float], ...] = (
    (10.0, 1.000),   # <10 mph: sin efecto medible
    (15.0, 0.985),   # 10-15:   leve
    (20.0, 0.965),   # 15-20:   significativo
    (25.0, 0.930),   # 20-25:   severo
    (999.0, 0.900),  # >25:     extremo — se abandona el juego aéreo
)

# TEMPERATURA: curva asimétrica con más recorrido en el extremo frío.
# La NFL juega hasta febrero en climas donde el béisbol nunca opera.
_TEMP_BANDS: tuple[tuple[float, float], ...] = (
    (10.0,  0.955),  # <10 °F:  frío extremo, manejo del balón comprometido
    (20.0,  0.970),  # 10-20:   muy frío
    (32.0,  0.985),  # 20-32:   bajo cero (°C)
    (50.0,  0.995),  # 32-50:   fresco
    (80.0,  1.000),  # 50-80:   rango neutro
    (90.0,  1.005),  # 80-90:   calor, defensas se fatigan
    (999.0, 1.000),  # >90:     calor extremo, se compensa por fatiga mutua
)

# PRECIPITACIÓN: afecta el manejo del balón y aumenta los balones sueltos.
_PRECIP_FACTOR_RAIN: float = 0.975
_PRECIP_FACTOR_SNOW: float = 0.955

# Umbral de probabilidad para considerar que llueve.
_PRECIP_THRESHOLD_PCT: float = 50.0

# Cotas del factor combinado. Red de seguridad ante datos corruptos.
_WEATHER_FACTOR_MIN: float = 0.85
_WEATHER_FACTOR_MAX: float = 1.02


@dataclass(frozen=True)
class NFLWeather:
    """
    Condiciones climáticas de un partido.

    Campos
    ------
    temperature_f     -- Temperatura en °F.
    wind_speed_mph    -- Velocidad del viento en mph. El factor que más
                         pesa en el ajuste.
    wind_direction    -- Dirección en grados. Se conserva por
                         trazabilidad pero NO entra en el cálculo del
                         factor: en NFL el viento cruzado daña igual que
                         el frontal, y no conocemos la orientación de
                         los campos.
    precipitation_pct -- Probabilidad de precipitación (0-100).
    condition         -- 'clear', 'cloudy', 'rain', 'snow', 'unknown'.
    source            -- 'nflverse' (observado), 'open-meteo'
                         (pronóstico) o 'none'.
    """
    temperature_f:     float | None = None
    wind_speed_mph:    float | None = None
    wind_direction:    int | None = None
    precipitation_pct: float | None = None
    condition:         str = CONDITION_UNKNOWN
    source:            str = SOURCE_NONE

    @property
    def is_observed(self) -> bool:
        """True si es dato real medido, no pronóstico."""
        return self.source == SOURCE_OBSERVED

    @property
    def has_data(self) -> bool:
        """True si hay al menos una medición utilizable."""
        return self.temperature_f is not None or self.wind_speed_mph is not None

    def weather_factor(self) -> float:
        """
        Multiplicador del total de puntos por condiciones climáticas.

        Combina viento, temperatura y precipitación. Retorna 1.0 (sin
        ajuste) cuando no hay datos: es el supuesto correcto, no una
        degradación — asumir clima adverso sin evidencia sesgaría todas
        las proyecciones a la baja.

        El viento pesa más que la temperatura porque su efecto sobre el
        juego aéreo y el pateo es más directo e inmediato que el del
        frío, al que los equipos se adaptan con el plan de juego.
        """
        if not self.has_data:
            return 1.0

        factor = 1.0

        if self.wind_speed_mph is not None:
            factor *= _band_lookup(self.wind_speed_mph, _WIND_BANDS)

        if self.temperature_f is not None:
            factor *= _band_lookup(self.temperature_f, _TEMP_BANDS)

        if self.condition == CONDITION_SNOW:
            factor *= _PRECIP_FACTOR_SNOW
        elif self.condition == CONDITION_RAIN:
            factor *= _PRECIP_FACTOR_RAIN
        elif (
            self.precipitation_pct is not None
            and self.precipitation_pct >= _PRECIP_THRESHOLD_PCT
        ):
            factor *= _PRECIP_FACTOR_RAIN

        return round(
            max(_WEATHER_FACTOR_MIN, min(_WEATHER_FACTOR_MAX, factor)), 4
        )

    def to_dict(self) -> dict:
        """Serializa para el dict de contexto."""
        return {
            "temperature":       self.temperature_f,
            "wind_speed":        self.wind_speed_mph,
            "wind_direction":    self.wind_direction,
            "precipitation_pct": self.precipitation_pct,
            "condition":         self.condition,
            "weather_source":    self.source,
            "weather_observed":  self.is_observed,
            "weather_factor":    self.weather_factor(),
        }


class NFLContextFetcher:
    """
    Contexto situacional de partidos NFL.

    Parámetros
    ----------
    schedule_fetcher  -- Fuente del calendario, para leer el clima
                         observado y los metadatos del partido.
    venue_factors     -- NFLVenueFactors. Si None, crea uno. Resuelve
                         techo, altitud y ajuste de viaje.
    cache_ttl_seconds -- TTL del caché de pronósticos. Default 3600.
    timeout           -- Timeout HTTP. Default 10s.
    """

    def __init__(
        self,
        schedule_fetcher:  ScheduleSource,
        venue_factors:     NFLVenueFactors | None = None,
        cache_ttl_seconds: int = 3600,
        timeout:           int = 10,
    ) -> None:
        self._schedule: ScheduleSource  = schedule_fetcher
        self._venues:   NFLVenueFactors = venue_factors or NFLVenueFactors()
        self._cache_ttl = cache_ttl_seconds
        self._timeout   = timeout
        self._cache: dict[str, tuple[dict, float]] = {}

    # ── SportDataProvider Protocol ────────────────────────────────────────────

    def get_context(self, event: Event) -> dict:
        """
        Contexto situacional del partido.

        Claves del dict retornado
        -------------------------
        venue_type        -- Techo efectivo: 'outdoors', 'dome', 'closed', 'open'.
        weatherproof      -- True si el clima no aplica.
        week              -- Semana de temporada.
        game_type         -- 'REG', 'WC', 'DIV', 'CON', 'SB'.
        is_divisional     -- True si es partido de división.
        is_primetime      -- True si es TNF, SNF o MNF.
        venue_total_factor -- Multiplicador por altitud.
        travel_adjustment -- Penalización al visitante por husos horarios.
        travel_tz_delta   -- Husos cruzados (positivo = hacia el este).

        Y si el estadio es descubierto, además:
        temperature, wind_speed, wind_direction, precipitation_pct,
        condition, weather_source, weather_observed, weather_factor.

        Nunca lanza: ante cualquier fallo retorna el contexto que haya
        podido construir. El pipeline no debe abortar por no conocer
        el clima.
        """
        try:
            return self._build_context(event)
        except Exception:
            return {}

    # ── Construcción del contexto ─────────────────────────────────────────────

    def _build_context(self, event: Event) -> dict:
        game = self._safe_game_info(event.event_id)

        home = event.home_team_id
        away = event.away_team_id
        game_roof = game.roof if game else None

        kickoff_hour = _kickoff_hour_et(event.start_time)

        context: dict = {
            "venue_type":   self._venues.effective_roof(home, game_roof),
            "weatherproof": self._venues.is_weatherproof(home, game_roof),
            "venue_total_factor": self._venues.total_factor(home),
            "travel_adjustment":  self._venues.travel_adjustment(
                home, away, kickoff_hour
            ),
            "travel_tz_delta":    self._venues.timezone_delta(home, away),
            "kickoff_hour_et":    kickoff_hour,
        }

        # Metadatos del calendario
        if game is not None:
            context.update({
                "week":          game.week,
                "game_type":     game.game_type,
                "is_divisional": game.div_game,
                "is_primetime":  game.is_primetime,
                "weekday":       game.weekday,
                "surface":       game.surface,
            })

        # ── Cortocircuito en estadios cubiertos ───────────────────
        # Sin consultar ninguna API: si el techo está cerrado el clima
        # es irrelevante y el factor es neutro por definición.
        if context["weatherproof"]:
            context["weather_factor"] = 1.0
            context["weather_source"] = SOURCE_NONE
            return context

        weather = self._resolve_weather(event, game)
        context.update(weather.to_dict())
        return context

    def _resolve_weather(
        self,
        event: Event,
        game:  NFLGameInfo | None,
    ) -> NFLWeather:
        """
        Resuelve el clima con prioridad al dato observado.

        Orden:
            1. nflverse — valor real medido. Disponible para partidos
               ya jugados, que es exactamente el caso del backtesting.
            2. Open-Meteo — pronóstico. Necesario para partidos futuros.
            3. Neutro — sin datos, factor 1.0.

        Usar el pronóstico donde existe el dato observado metería en el
        backtest un error de predicción que no ocurrió en la realidad.
        """
        observed = _weather_from_game(game)
        if observed is not None:
            return observed

        forecast = self._fetch_forecast(event)
        if forecast is not None:
            return forecast

        return NFLWeather(source=SOURCE_NONE)

    # ── Pronóstico ────────────────────────────────────────────────────────────

    def _fetch_forecast(self, event: Event) -> NFLWeather | None:
        """Pronóstico desde Open-Meteo, con caché por estadio y hora."""
        venue = self._venues.get(event.home_team_id)
        if venue is None or _requests is None:
            return None

        game_date = event.date
        hour_key  = (event.start_time or "")[:13] or game_date
        cache_key = f"{venue.team}_{hour_key}"

        if cache_key in self._cache:
            data, ts = self._cache[cache_key]
            if time.monotonic() - ts < self._cache_ttl:
                return NFLWeather(**data) if data else None

        weather = self._call_open_meteo(
            lat=venue.lat, lon=venue.lon,
            game_date=game_date, start_time=event.start_time,
        )
        if weather is not None:
            payload = {
                "temperature_f":     weather.temperature_f,
                "wind_speed_mph":    weather.wind_speed_mph,
                "wind_direction":    weather.wind_direction,
                "precipitation_pct": weather.precipitation_pct,
                "condition":         weather.condition,
                "source":            weather.source,
            }
            self._cache[cache_key] = (payload, time.monotonic())
        return weather

    def _call_open_meteo(
        self,
        lat:        float,
        lon:        float,
        game_date:  str,
        start_time: str | None,
    ) -> NFLWeather | None:
        """Consulta Open-Meteo (gratuita, sin API key)."""
        if _requests is None or not game_date:
            return None

        params = {
            "latitude":  lat,
            "longitude": lon,
            "hourly": (
                "temperature_2m,wind_speed_10m,wind_direction_10m,"
                "precipitation_probability,weather_code"
            ),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit":  "mph",
            "timezone":   "UTC",
            "start_date": game_date,
            "end_date":   game_date,
        }
        try:
            resp = _requests.get(_OPEN_METEO_URL, params=params, timeout=self._timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None

        return _parse_open_meteo(data, start_time)

    # ── Utilidades internas ───────────────────────────────────────────────────

    def _safe_game_info(self, game_id: str) -> NFLGameInfo | None:
        """
        Metadatos del partido, tolerando fallos de la fuente.

        Se aísla en un método propio porque get_game_info() viene de una
        dependencia inyectada: un doble de prueba o una versión futura
        podría lanzar, y ese fallo no debe impedir construir el resto
        del contexto (techo, viaje, altitud no dependen del calendario).
        """
        try:
            result = self._schedule.get_game_info(game_id)
        except Exception:
            return None
        return result if isinstance(result, NFLGameInfo) else None

    def clear_cache(self) -> None:
        """Limpia el caché de pronósticos."""
        self._cache.clear()


# ── Funciones de módulo ───────────────────────────────────────────────────────

def _band_lookup(
    value: float,
    bands: tuple[tuple[float, float], ...],
) -> float:
    """
    Busca el multiplicador correspondiente en una tabla de umbrales.

    Las bandas se declaran ordenadas por umbral ascendente; se devuelve
    el factor de la primera cuyo umbral supera el valor.

    Se usan bandas discretas en vez de una función continua porque los
    efectos documentados del clima en NFL se reportan por rangos, no
    como coeficientes. Inventar una interpolación suave daría una
    falsa impresión de precisión sobre datos que no la tienen.
    """
    for threshold, factor in bands:
        if value < threshold:
            return factor
    return bands[-1][1]


def _weather_from_game(game: NFLGameInfo | None) -> NFLWeather | None:
    """
    Extrae el clima observado desde los metadatos de nflverse.

    Retorna None si no hay ninguna medición, para que el llamador pueda
    distinguir "no hay dato observado" de "hay dato y es neutro".

    CORRECCIÓN: la versión anterior leía estos campos con
    `getattr(game, "temp", None)` asumiendo que NFLGameInfo podría
    tenerlos "en una versión futura". No los tenía, así que el getattr
    devolvía None siempre y esta función nunca retornaba nada — el
    pronóstico de Open-Meteo se usaba incluso para partidos ya jugados,
    metiendo en el backtest un error de predicción que no existió en la
    realidad. El acceso directo hace que un campo ausente sea un error
    visible en vez de una degradación silenciosa.
    """
    if game is None:
        return None

    temp = _safe_float(game.temp)
    wind = _safe_float(game.wind)

    if temp is None and wind is None:
        return None

    # nflverse no reporta condición ni precipitación: se infiere lo que
    # se puede de la temperatura y se deja el resto sin determinar.
    condition = CONDITION_UNKNOWN
    if temp is not None and temp <= 32.0:
        # Por debajo de 0 °C cualquier precipitación sería nieve. No
        # afirma que nevara — solo que si precipitó, fue en forma de
        # nieve. La condición real no está en el dataset.
        condition = CONDITION_UNKNOWN

    return NFLWeather(
        temperature_f  = temp,
        wind_speed_mph = wind,
        condition      = condition,
        source         = SOURCE_OBSERVED,
    )


def _parse_open_meteo(data: dict, start_time: str | None) -> NFLWeather | None:
    """Parsea la respuesta de Open-Meteo para la hora del kickoff."""
    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        return None

    target = (start_time or "")[:13]
    idx = 0
    for i, t in enumerate(times):
        if str(t)[:13] == target:
            idx = i
            break

    def at(key: str):
        vals = hourly.get(key, [])
        return vals[idx] if idx < len(vals) else None

    temp   = _safe_float(at("temperature_2m"))
    wind   = _safe_float(at("wind_speed_10m"))
    wdir   = _safe_float(at("wind_direction_10m"))
    precip = _safe_float(at("precipitation_probability"))
    code   = at("weather_code")

    return NFLWeather(
        temperature_f     = round(temp, 1) if temp is not None else None,
        wind_speed_mph    = round(wind, 1) if wind is not None else None,
        wind_direction    = int(wdir) if wdir is not None else None,
        precipitation_pct = precip,
        condition         = _wmo_to_condition(code),
        source            = SOURCE_FORECAST,
    )


def _wmo_to_condition(code) -> str:
    """
    Traduce un código WMO de Open-Meteo a nuestra condición.

    Los códigos de nieve (71-77) se distinguen de los de lluvia porque
    su penalización es notablemente mayor: la nieve compromete el
    manejo del balón además de la visibilidad.
    """
    if code is None:
        return CONDITION_UNKNOWN
    try:
        c = int(code)
    except (ValueError, TypeError):
        return CONDITION_UNKNOWN

    if c <= 1:
        return CONDITION_CLEAR
    if c <= 3:
        return CONDITION_CLOUDY
    if 71 <= c <= 77 or c in (85, 86):
        return CONDITION_SNOW
    if c <= 82:
        return CONDITION_RAIN
    if c <= 99:
        return CONDITION_RAIN   # tormenta: se trata como lluvia
    return CONDITION_UNKNOWN


def _kickoff_hour_et(start_time: str | None) -> int | None:
    """
    Hora del kickoff en horario del Este, desde un start_time UTC.

    schedule.py construye start_time en UTC aplicando el offset de ET
    (-4 en septiembre-octubre, -5 de noviembre en adelante). Aquí se
    invierte esa conversión para recuperar la hora ET, que es lo que
    necesita NFLVenueFactors.travel_adjustment() para decidir si el
    kickoff cuenta como temprano.

    El offset se deduce del mes presente en el propio start_time, sin
    depender de una zona horaria del sistema.
    """
    if not start_time or len(start_time) < 13:
        return None
    try:
        utc_hour = int(start_time[11:13])
        month    = int(start_time[5:7])
    except (ValueError, IndexError):
        return None

    offset = 4 if month in (9, 10) else 5
    return (utc_hour - offset) % 24


def _is_nan(value) -> bool:
    """True si el valor es NaN."""
    try:
        return value != value
    except Exception:
        return False


def _safe_float(value) -> float | None:
    """Convierte a float de forma segura (NaN → None)."""
    if value is None or _is_nan(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None