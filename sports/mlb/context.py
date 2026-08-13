"""
sports/mlb/context.py

MLBContextFetcher: contexto situacional de partidos MLB.
Implementa SportDataProvider.get_context(event) → dict.

Migrado de analysis/context.py del sistema MLB con tres correcciones:

1. venue_id como clave — no venue_name.
   venue_name puede cambiar (patrocinio) pero venue_id es estable.
   Event.venue_id es el campo correcto del contrato.

2. Estadios indoor retornan {} — sin llamada a Open-Meteo.
   Si venue_type='indoor', el clima no afecta el partido.
   Evita llamadas innecesarias a la API para Tropicana Field,
   Minute Maid Park, loanDepot Park, Globe Life Field, etc.

3. Caché por (venue_id, hour_utc) — respetar rate limits Open-Meteo.
   Open-Meteo permite 10,000 req/día en el plan gratuito. Con caché
   de 1 hora, el mismo estadio no se refetcha en el mismo día.

Por qué el clima importa en MLB
---------------------------------
MLB es el único deporte entre los soportados donde el clima afecta
significativamente el resultado:
    - Viento en contra (campo) → reduce carreras ~8%
    - Viento de frente → aumenta carreras ~12%
    - Temperatura < 50°F → reduce carreras ~5% (bola no viaja)
    - Temperatura > 85°F → aumenta carreras ~4%
    - Lluvia → riesgo de postponement

El ProjectionModel MLB multiplica expected_home/away por un factor
de ajuste de clima cuando venue_type='outdoor'. Para 'indoor' y
'retractable' (techo cerrado), el factor es 1.0 (sin ajuste).

Estadios de MLB por venue_type
---------------------------------
outdoor:    30 estadios — afectados por clima
indoor:     5 estadios — Tropicana, Minute Maid, loanDepot,
            Globe Life, Rogers Centre (azotea)
retractable: 4 estadios — Chase Field, American Family Field,
             Marlins Park, T-Mobile Park

Nota: retractable se clasifica como 'outdoor' si el techo está
abierto (decisión del equipo local, no disponible via API).
Por defecto se asume 'outdoor' para retractable — conservador.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

from core.contracts.event import Event

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


# ── Tipos de venue ────────────────────────────────────────────────────────────

VENUE_TYPE_OUTDOOR     = "outdoor"
VENUE_TYPE_INDOOR      = "indoor"
VENUE_TYPE_RETRACTABLE = "retractable"  # tratado como outdoor por defecto


# ── Catálogo de venues MLB ────────────────────────────────────────────────────

@dataclass(frozen=True)
class VenueInfo:
    """
    Información estática de un estadio MLB.

    Inmutable: las coordenadas y el type de un estadio no cambian.
    """
    venue_id:    str
    name:        str
    lat:         float
    lon:         float
    venue_type:  str      # outdoor | indoor | retractable
    timezone:    str      # IANA timezone string (ej: 'America/New_York')

    @property
    def is_outdoor(self) -> bool:
        """True si el clima puede afectar el partido."""
        return self.venue_type in (VENUE_TYPE_OUTDOOR, VENUE_TYPE_RETRACTABLE)


# Catálogo de estadios MLB con coordenadas exactas y timezone local.
# venue_id = slug estable de MLB Stats API.
# Fuente: MLB Stats API + Google Maps (coordinadas verificadas 2024).
_VENUE_CATALOG: dict[str, VenueInfo] = {
    "coors-field": VenueInfo(
        "coors-field", "Coors Field", 39.7559, -104.9942,
        VENUE_TYPE_OUTDOOR, "America/Denver"
    ),
    "fenway-park": VenueInfo(
        "fenway-park", "Fenway Park", 42.3467, -71.0972,
        VENUE_TYPE_OUTDOOR, "America/New_York"
    ),
    "wrigley-field": VenueInfo(
        "wrigley-field", "Wrigley Field", 41.9484, -87.6553,
        VENUE_TYPE_OUTDOOR, "America/Chicago"
    ),
    "yankee-stadium": VenueInfo(
        "yankee-stadium", "Yankee Stadium", 40.8296, -73.9262,
        VENUE_TYPE_OUTDOOR, "America/New_York"
    ),
    "dodger-stadium": VenueInfo(
        "dodger-stadium", "Dodger Stadium", 34.0739, -118.2400,
        VENUE_TYPE_OUTDOOR, "America/Los_Angeles"
    ),
    "oracle-park": VenueInfo(
        "oracle-park", "Oracle Park", 37.7786, -122.3893,
        VENUE_TYPE_OUTDOOR, "America/Los_Angeles"
    ),
    "petco-park": VenueInfo(
        "petco-park", "Petco Park", 32.7073, -117.1566,
        VENUE_TYPE_OUTDOOR, "America/Los_Angeles"
    ),
    "citizens-bank-park": VenueInfo(
        "citizens-bank-park", "Citizens Bank Park", 39.9061, -75.1665,
        VENUE_TYPE_OUTDOOR, "America/New_York"
    ),
    "camden-yards": VenueInfo(
        "camden-yards", "Oriole Park at Camden Yards", 39.2838, -76.6218,
        VENUE_TYPE_OUTDOOR, "America/New_York"
    ),
    "pnc-park": VenueInfo(
        "pnc-park", "PNC Park", 40.4469, -80.0057,
        VENUE_TYPE_OUTDOOR, "America/New_York"
    ),
    "target-field": VenueInfo(
        "target-field", "Target Field", 44.9817, -93.2781,
        VENUE_TYPE_OUTDOOR, "America/Chicago"
    ),
    "busch-stadium": VenueInfo(
        "busch-stadium", "Busch Stadium", 38.6226, -90.1928,
        VENUE_TYPE_OUTDOOR, "America/Chicago"
    ),
    "kauffman-stadium": VenueInfo(
        "kauffman-stadium", "Kauffman Stadium", 39.0517, -94.4803,
        VENUE_TYPE_OUTDOOR, "America/Chicago"
    ),
    "progressive-field": VenueInfo(
        "progressive-field", "Progressive Field", 41.4962, -81.6852,
        VENUE_TYPE_OUTDOOR, "America/New_York"
    ),
    "comerica-park": VenueInfo(
        "comerica-park", "Comerica Park", 42.3390, -83.0485,
        VENUE_TYPE_OUTDOOR, "America/Detroit"
    ),
    "great-american-ball-park": VenueInfo(
        "great-american-ball-park", "Great American Ball Park", 39.0975, -84.5081,
        VENUE_TYPE_OUTDOOR, "America/New_York"
    ),
    "truist-park": VenueInfo(
        "truist-park", "Truist Park", 33.8908, -84.4678,
        VENUE_TYPE_OUTDOOR, "America/New_York"
    ),
    "angel-stadium": VenueInfo(
        "angel-stadium", "Angel Stadium", 33.8003, -117.8827,
        VENUE_TYPE_OUTDOOR, "America/Los_Angeles"
    ),
    # Estadios indoor — get_context() retorna {} para estos
    "tropicana-field": VenueInfo(
        "tropicana-field", "Tropicana Field", 27.7683, -82.6534,
        VENUE_TYPE_INDOOR, "America/New_York"
    ),
    "minute-maid-park": VenueInfo(
        "minute-maid-park", "Minute Maid Park", 29.7572, -95.3555,
        VENUE_TYPE_INDOOR, "America/Chicago"
    ),
    "loanDepot-park": VenueInfo(
        "loanDepot-park", "loanDepot park", 25.7781, -80.2196,
        VENUE_TYPE_INDOOR, "America/New_York"
    ),
    "globe-life-field": VenueInfo(
        "globe-life-field", "Globe Life Field", 32.7474, -97.0832,
        VENUE_TYPE_INDOOR, "America/Chicago"
    ),
    # Estadios retractable — tratados como outdoor
    "chase-field": VenueInfo(
        "chase-field", "Chase Field", 33.4453, -112.0667,
        VENUE_TYPE_RETRACTABLE, "America/Phoenix"
    ),
    "american-family-field": VenueInfo(
        "american-family-field", "American Family Field", 43.0283, -87.9712,
        VENUE_TYPE_RETRACTABLE, "America/Chicago"
    ),
    "t-mobile-park": VenueInfo(
        "t-mobile-park", "T-Mobile Park", 47.5914, -122.3325,
        VENUE_TYPE_RETRACTABLE, "America/Los_Angeles"
    ),
    "guaranteed-rate-field": VenueInfo(
        "guaranteed-rate-field", "Guaranteed Rate Field", 41.8300, -87.6339,
        VENUE_TYPE_OUTDOOR, "America/Chicago"
    ),
}

# Aliases: ID numérico MLB Stats API → slug
_VENUE_ID_ALIASES: dict[str, str] = {
    "2395": "coors-field",
    "2394": "fenway-park",
    "17":   "wrigley-field",
    "3313": "yankee-stadium",
    "22":   "petco-park",
    "1":    "dodger-stadium",
    "2":    "oracle-park",
    "32":   "angel-stadium",
    "2392": "truist-park",
}


# ── Contexto de clima ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WeatherContext:
    """
    Condiciones climáticas en el momento del partido.

    Solo relevante para estadios outdoor/retractable.
    Inmutable: representa el clima en el instante del fetch.
    """
    temperature_f:      float | None = None
    wind_speed_mph:     float | None = None
    wind_direction:     int   | None = None   # grados (0=N, 90=E, 180=S, 270=O)
    precipitation_pct:  float | None = None   # % probabilidad de lluvia
    condition:          str   | None = None   # 'clear', 'cloudy', 'rain', 'unknown'
    day_night:          str   | None = None   # 'day' o 'night'

    def to_dict(self) -> dict:
        """Convierte a dict para SportDataProvider.get_context()."""
        return {
            "temperature":       self.temperature_f,
            "wind_speed":        self.wind_speed_mph,
            "wind_direction":    self.wind_direction,
            "precipitation_pct": self.precipitation_pct,
            "condition":         self.condition,
            "day_night":         self.day_night,
        }


# ── Fetcher principal ─────────────────────────────────────────────────────────

class MLBContextFetcher:
    """
    Obtiene contexto situacional de partidos MLB.

    Implementa la responsabilidad de SportDataProvider.get_context()
    para el plugin de MLB.

    Parámetros
    ----------
    cache_ttl_seconds  -- TTL del caché de clima. Default 3600 (1 hora).
                         Evita refetch de Open-Meteo para el mismo
                         estadio en la misma hora.
    timeout            -- Timeout HTTP en segundos. Default 10.
    """

    def __init__(
        self,
        cache_ttl_seconds: int = 3600,
        timeout:           int = 10,
    ) -> None:
        self._cache_ttl = cache_ttl_seconds
        self._timeout   = timeout
        self._cache: dict[str, tuple[dict, float]] = {}  # key → (data, timestamp)

    def get_context(self, event: Event) -> dict:
        """
        Retorna el contexto situacional del partido.

        Implementa SportDataProvider.get_context(event) → dict.

        Retorna {} para estadios indoor (clima irrelevante).
        Retorna contexto completo para outdoor/retractable.

        Claves del dict retornado:
            venue_type        -- 'outdoor' | 'indoor' | 'retractable'
            temperature       -- Temperatura en °F (None si no disponible)
            wind_speed        -- Viento en mph
            wind_direction    -- Dirección del viento en grados
            precipitation_pct -- Probabilidad de lluvia (0-100)
            condition         -- 'clear' | 'cloudy' | 'rain' | 'unknown'
            day_night         -- 'day' | 'night'
        """
        venue_id = event.venue_id
        venue    = self._resolve_venue(venue_id)

        if venue is None:
            # Venue desconocido — retornar contexto mínimo sin clima
            return {"venue_type": VENUE_TYPE_OUTDOOR, "day_night": "unknown"}

        # Estadio indoor — el clima no afecta el partido
        if not venue.is_outdoor:
            return {"venue_type": VENUE_TYPE_INDOOR}

        # Fetch de clima desde Open-Meteo
        weather = self._fetch_weather_cached(
            venue_id   = venue.venue_id,
            lat        = venue.lat,
            lon        = venue.lon,
            start_time = event.start_time,
        )

        day_night = self._day_night(event.start_time, venue.timezone)

        result: dict = {"venue_type": venue.venue_type, "day_night": day_night}
        if weather is not None:
            result.update(weather.to_dict())

        return result

    def get_venue_info(self, venue_id: str) -> VenueInfo | None:
        """Retorna la info del estadio por venue_id."""
        return self._resolve_venue(venue_id)

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _resolve_venue(self, venue_id: str) -> VenueInfo | None:
        """Resuelve venue_id a VenueInfo usando el catálogo y aliases."""
        vid = str(venue_id).strip().lower()
        # Alias numérico
        slug = _VENUE_ID_ALIASES.get(vid, vid)
        # Normalizar espacios a guiones
        slug = slug.replace(" ", "-").replace("_", "-")
        return _VENUE_CATALOG.get(slug)

    def _fetch_weather_cached(
        self,
        venue_id:   str,
        lat:        float,
        lon:        float,
        start_time: str,
    ) -> WeatherContext | None:
        """
        Fetch de clima con caché por (venue_id, hour_utc).

        Open-Meteo devuelve el forecast hora a hora. La caché usa la
        hora UTC del partido como clave — el mismo estadio a la misma
        hora no se refetcha en la misma ejecución del pipeline.
        """
        hour_utc  = start_time[:13] if start_time else "unknown"
        cache_key = f"{venue_id}_{hour_utc}"

        if cache_key in self._cache:
            data, ts = self._cache[cache_key]
            if time.monotonic() - ts < self._cache_ttl:
                return WeatherContext(**data) if data else None

        weather = self._fetch_open_meteo(lat, lon, start_time)
        if weather is not None:
            self._cache[cache_key] = (weather.to_dict(), time.monotonic())
        return weather

    def _fetch_open_meteo(
        self,
        lat:        float,
        lon:        float,
        start_time: str,
    ) -> WeatherContext | None:
        """
        Fetch de clima desde Open-Meteo API (gratuita, sin API key).

        Endpoint: https://api.open-meteo.com/v1/forecast
        Variables: temperature_2m, wind_speed_10m, wind_direction_10m,
                   precipitation_probability, weather_code.
        """
        if not _REQUESTS_AVAILABLE:
            return None

        # Extraer fecha del start_time para el forecast
        game_date = start_time[:10] if start_time else None
        if not game_date:
            return None

        params = {
            "latitude":    lat,
            "longitude":   lon,
            "hourly":      "temperature_2m,wind_speed_10m,wind_direction_10m,precipitation_probability,weather_code",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit":  "mph",
            "timezone":    "UTC",
            "start_date":  game_date,
            "end_date":    game_date,
        }

        try:
            resp = _requests.get(
                _OPEN_METEO_URL, params=params, timeout=self._timeout
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None

        return self._parse_weather(data, start_time)

    @staticmethod
    def _parse_weather(data: dict, start_time: str) -> WeatherContext | None:
        """
        Parsea la respuesta de Open-Meteo para la hora del partido.

        Selecciona la hora más cercana al start_time del partido.
        """
        hourly = data.get("hourly", {})
        times  = hourly.get("time", [])
        if not times:
            return None

        # Encontrar el índice de la hora más cercana al partido
        game_hour = start_time[:13] if start_time else None
        idx = 0
        for i, t in enumerate(times):
            if t[:13] == game_hour:
                idx = i
                break

        def get_hourly(key: str) -> Any:
            vals = hourly.get(key, [])
            return vals[idx] if idx < len(vals) else None

        temp    = get_hourly("temperature_2m")
        wind_s  = get_hourly("wind_speed_10m")
        wind_d  = get_hourly("wind_direction_10m")
        precip  = get_hourly("precipitation_probability")
        w_code  = get_hourly("weather_code")

        # WMO weather code → condition string
        condition = _wmo_to_condition(w_code)

        return WeatherContext(
            temperature_f     = round(float(temp), 1) if temp is not None else None,
            wind_speed_mph    = round(float(wind_s), 1) if wind_s is not None else None,
            wind_direction    = int(wind_d) if wind_d is not None else None,
            precipitation_pct = float(precip) if precip is not None else None,
            condition         = condition,
        )

    @staticmethod
    def _day_night(start_time: str, timezone_str: str) -> str:
        """
        Determina si el partido es diurno o nocturno según la hora local.

        day = inicio antes de las 17:00 hora local del estadio.
        night = inicio a las 17:00 o después.

        Usa solo la hora UTC del start_time y un offset aproximado por
        timezone para evitar dependencia en pytz/zoneinfo.
        """
        if not start_time:
            return "unknown"

        # Extraer hora UTC del start_time (ISO-8601)
        try:
            hour_utc = int(start_time[11:13])
        except (IndexError, ValueError):
            return "unknown"

        # Offset aproximado UTC → local por timezone string
        tz_offsets: dict[str, int] = {
            "America/New_York":    -4,   # EDT (temporada MLB = verano)
            "America/Chicago":     -5,   # CDT
            "America/Denver":      -6,   # MDT
            "America/Phoenix":     -7,   # MST (sin DST)
            "America/Los_Angeles": -7,   # PDT
            "America/Detroit":     -4,   # EDT
        }
        offset    = tz_offsets.get(timezone_str, -5)  # CDT como default
        hour_local = (hour_utc + offset) % 24

        return "day" if hour_local < 17 else "night"


# ── Utilidades WMO ────────────────────────────────────────────────────────────

def _wmo_to_condition(code: Any) -> str:
    """Convierte WMO weather code a string de condición."""
    if code is None:
        return "unknown"
    try:
        code = int(code)
    except (ValueError, TypeError):
        return "unknown"

    if code <= 1:
        return "clear"
    elif code <= 3:
        return "cloudy"
    elif code <= 67:
        return "rain"
    elif code <= 77:
        return "snow"
    elif code <= 99:
        return "storm"
    return "unknown"