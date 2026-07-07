"""
core/odds/client.py

OddsAPIClient: cliente HTTP tipado para The Odds API v4.

Migrado de data/odds_api.py del sistema MLB-PREDICTOR-ADVANCED con
tres correcciones arquitectónicas documentadas en MLB_EDGE_AUDIT.md:

    1. SPORT como parámetro de método, no constante hardcodeada.
       El sistema MLB tenía SPORT="baseball_mlb" en constants.py —
       imposible llamar para NBA y MLB en el mismo proceso. Un único
       OddsAPIClient sirve todos los deportes del roadmap.

    2. OddsAPIResponse tipado con error_type en vez de excepciones.
       Errores de autenticación (401/403) y rate limit son condiciones
       esperadas en producción, no bugs. Modelarlos como resultado
       tipado permite al pipeline decidir limpiamente qué hacer:
       auth_error → abortar; rate_limit → continuar con datos ya
       cargados; network_error → reintentar. El sistema MLB capturaba
       HTTPError y retornaba [] silenciosamente — perdía la señal.

    3. requests_remaining expuesto en cada respuesta.
       The Odds API devuelve x-requests-remaining en cada header HTTP.
       El sistema MLB ignoraba este header. Exponer el valor permite
       que el pipeline implemente lógica de ahorro de créditos sin
       modificar el cliente.

Separación de responsabilidades
---------------------------------
Este módulo hace SOLO HTTP y deserialización mínima a RawOddsEvent.
NO extrae best prices, NO selecciona bookmakers, NO construye
MarketOdds — esa responsabilidad pertenece a core/odds/normalizer.py.

Esta separación corrige la deuda técnica de analizar_mercados() en el
sistema MLB, que mezclaba fetch + parse + best_price en ~200 líneas.

Soporte de endpoints
---------------------
The Odds API v4 tiene dos endpoints relevantes:

    /sports/{sport}/odds
        → mercados featured (ML, TOTAL, SPREAD) para TODOS los eventos
        → Una sola request, bajo costo en créditos

    /sports/{sport}/events/{eventId}/odds
        → mercados avanzados (props, innings) para UN evento específico
        → Una request por evento, costo alto en créditos

OddsAPIClient soporta ambos. El caller decide qué mercados pedir y
cuándo vale la pena el costo adicional de event-level markets.

Configuración
--------------
OddsAPIConfig se construye desde ConfigLoader (YAML + env vars):

# .env o config/secrets.yaml
ODDS_API_KEY=your_key_here

# config/base.yaml
odds_api:
  base_url: "https://api.the-odds-api.com/v4"
  timeout_seconds: 20
  max_retries: 0           # 0 = sin retry por defecto
  retry_backoff_base: 2.0  # segundos, backoff exponencial
  regions: "us"
  bookmakers: ""           # vacío = usar regions, no bookmakers específicos

Uso típico
-----------
    from core.odds.client import OddsAPIClient, OddsAPIConfig

    config = OddsAPIConfig(api_key="...", regions="us")
    client = OddsAPIClient(config)

    response = client.get_events(
        sport="baseball_mlb",
        markets=["h2h", "totals", "spreads"],
    )

    if not response.success:
        if response.error_type == "auth_error":
            raise RuntimeError("API key inválida")
        elif response.error_type == "rate_limit":
            logger.warning(f"Rate limit alcanzado. Remaining: {response.requests_remaining}")
    else:
        raw_events = response.events
        # Pasar a OddsNormalizer en Stage 5
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

# Tipos de error que puede retornar OddsAPIResponse.error_type
ERROR_AUTH         = "auth_error"         # 401 / 403
ERROR_RATE_LIMIT   = "rate_limit"         # 429
ERROR_NOT_FOUND    = "not_found"          # 404
ERROR_SERVER       = "server_error"       # 5xx
ERROR_NETWORK      = "network_error"      # timeout, connection refused
ERROR_PARSE        = "parse_error"        # JSON inválido
ERROR_UNAVAILABLE  = "requests_unavailable"  # requests no instalado


# ── Configuración del cliente ─────────────────────────────────────────────────

@dataclass(frozen=True)
class OddsAPIConfig:
    """
    Configuración inmutable del cliente The Odds API.

    Inmutable: la configuración no cambia durante la vida del cliente.
    Si se necesita cambiar la API key o el timeout, se crea un nuevo
    OddsAPIClient — no se muta el existente.

    Campos
    ------
    api_key           -- Clave de autenticación. Nunca se loggea ni
                        se incluye en mensajes de error (se redacta).
    base_url           -- URL base de la API. Raramente necesita cambiar
                        salvo para tests contra un mock server.
    timeout_seconds     -- Timeout HTTP por request. 20s es suficiente
                        para la mayoría de condiciones de red. La API
                        responde en <2s en condiciones normales.
    max_retries         -- Reintentos para errores transitorios (network,
                        5xx). 0 = sin retry. Nunca reintenta 401/403.
    retry_backoff_base  -- Base en segundos para backoff exponencial.
                        retry 1: base^1, retry 2: base^2, etc.
    regions             -- Regiones para filtrar bookmakers ('us', 'uk',
                        'eu', 'au'). Ignorado si bookmakers está poblado.
    bookmakers          -- Lista específica de bookmakers separada por
                        comas. Si está poblado, regions se ignora.
                        Vacío = usar regions.
    """
    api_key:            str
    base_url:           str   = "https://api.the-odds-api.com/v4"
    timeout_seconds:    int   = 20
    max_retries:        int   = 0
    retry_backoff_base: float = 2.0
    regions:            str   = "us"
    bookmakers:         str   = ""

    def __post_init__(self) -> None:
        if not self.api_key or not self.api_key.strip():
            raise ValueError(
                "OddsAPIConfig.api_key no puede estar vacío. "
                "Definir ODDS_API_KEY en .env o en config/secrets.yaml."
            )
        if self.max_retries < 0:
            raise ValueError(f"max_retries={self.max_retries} debe ser >= 0.")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds={self.timeout_seconds} debe ser > 0.")

    def redacted_key(self) -> str:
        """API key con solo los primeros 4 caracteres visibles."""
        if len(self.api_key) <= 4:
            return "****"
        return self.api_key[:4] + "*" * (len(self.api_key) - 4)


# ── Tipos de datos crudos ─────────────────────────────────────────────────────

@dataclass
class RawOddsEvent:
    """
    Evento crudo de The Odds API, con tipado mínimo.

    Contiene los campos de identidad y el bloque bookmakers como dict
    crudo — OddsNormalizer (core/odds/normalizer.py) extrae los
    precios y construye list[MarketOdds].

    Deliberadamente NO tipado en detalle: la estructura de bookmakers
    varía según los mercados solicitados, y el normalizer es el
    responsable de navegar esa estructura, no el cliente HTTP.

    Campos
    ------
    event_id      -- ID único del evento en The Odds API.
    sport         -- Identificador del deporte ('baseball_mlb', etc.).
    home_team     -- Nombre del equipo local (tal como lo devuelve la API).
    away_team     -- Nombre del equipo visitante.
    commence_time -- Hora de inicio en ISO-8601 UTC.
    bookmakers    -- Bloque crudo de bookmakers con sus mercados y cuotas.
                    Estructura: list[dict] con keys 'key', 'title',
                    'last_update', 'markets'.
    loaded_markets -- Conjunto de market keys cargados para este evento.
                    Vacío si solo se cargaron featured markets.
    """
    event_id:       str
    sport:          str
    home_team:      str
    away_team:      str
    commence_time:  str
    bookmakers:     list[dict] = field(default_factory=list)
    loaded_markets: set[str]   = field(default_factory=set)


@dataclass(frozen=True)
class OddsAPIResponse:
    """
    Resultado tipado de cualquier llamada al OddsAPIClient.

    Modela tanto éxito como error como resultado explícito — no usa
    excepciones para condiciones esperadas de negocio (auth, rate
    limit). El caller decide la acción según error_type.

    Campos
    ------
    success             -- True si la request fue exitosa y events
                          está poblado (o vacío por ausencia de datos,
                          no por error).
    events              -- Eventos obtenidos. Lista vacía si success=True
                          pero no hay eventos disponibles (deporte fuera
                          de temporada, por ejemplo).
    error_type          -- None si success=True. Uno de las constantes
                          ERROR_* definidas en este módulo si hay error.
    error_message        -- Descripción del error sin API key.
    requests_remaining   -- Créditos restantes en el plan actual.
                          None si no se pudo leer el header (error
                          antes de recibir respuesta HTTP).
    requests_used        -- Créditos usados en esta request.
                          None si no disponible en el header.
    """
    success:             bool
    events:              list[RawOddsEvent]
    error_type:          str | None       = None
    error_message:       str | None       = None
    requests_remaining:  int | None       = None
    requests_used:       int | None       = None

    @property
    def has_events(self) -> bool:
        return bool(self.events)

    @property
    def is_auth_error(self) -> bool:
        return self.error_type == ERROR_AUTH

    @property
    def is_rate_limit(self) -> bool:
        return self.error_type == ERROR_RATE_LIMIT


# ── Cliente principal ─────────────────────────────────────────────────────────

class OddsAPIClient:
    """
    Cliente HTTP tipado para The Odds API v4.

    Una instancia sirve todos los deportes — sport es parámetro de
    método, no atributo del cliente.

    Parámetros
    ----------
    config  -- OddsAPIConfig con API key y parámetros de red.
               Construir desde ConfigLoader:
                   config = OddsAPIConfig(
                       api_key=loader.get('ODDS_API_KEY'),
                       regions=loader.get('odds_api.regions', 'us'),
                   )
    """

    def __init__(self, config: OddsAPIConfig) -> None:
        self._config = config

    # ── API pública ────────────────────────────────────────────────────────────

    def get_events(
        self,
        sport: str,
        markets: list[str] | None = None,
        odds_format: str = "decimal",
    ) -> OddsAPIResponse:
        """
        Obtiene eventos con mercados featured para un deporte.

        Endpoint: GET /sports/{sport}/odds
        Costo: 1 request (todos los eventos del deporte en una llamada).

        Parámetros
        ----------
        sport       -- Identificador del deporte para The Odds API.
                      Ejemplos: 'baseball_mlb', 'basketball_nba',
                      'americanfootball_nfl', 'soccer_epl'.
        markets      -- Lista de market keys a solicitar.
                      Default: ['h2h', 'totals', 'spreads'] (featured).
                      The Odds API solo permite featured markets aquí;
                      para props usar get_event_markets().
        odds_format  -- Formato de cuotas. 'decimal' (default) o
                      'american'. El sistema usa decimal en todas
                      partes — este parámetro existe para compatibilidad
                      con integraciones externas.

        Retorna
        -------
        OddsAPIResponse con events poblado si success=True.
        Si la API no tiene datos para este sport, success=True y
        events=[] — no es un error, es ausencia de datos.
        """
        if markets is None:
            markets = ["h2h", "totals", "spreads"]

        params = self._build_params(
            markets=markets,
            odds_format=odds_format,
        )

        url = f"{self._config.base_url}/sports/{sport}/odds"
        raw_response = self._request(url, params)

        if not raw_response["success"]:
            return OddsAPIResponse(
                success=False,
                events=[],
                error_type=raw_response["error_type"],
                error_message=raw_response["error_message"],
                requests_remaining=raw_response.get("requests_remaining"),
                requests_used=raw_response.get("requests_used"),
            )

        events = self._parse_events(raw_response["data"], sport)
        return OddsAPIResponse(
            success=True,
            events=events,
            requests_remaining=raw_response.get("requests_remaining"),
            requests_used=raw_response.get("requests_used"),
        )

    def get_event_markets(
        self,
        sport: str,
        event_id: str,
        markets: list[str],
        odds_format: str = "decimal",
    ) -> OddsAPIResponse:
        """
        Obtiene mercados avanzados para un evento específico.

        Endpoint: GET /sports/{sport}/events/{eventId}/odds
        Costo: 1 request por evento (más costoso que get_events).
        Usar solo para mercados avanzados (props, innings) cuando el
        plan de la API lo permita.

        Parámetros
        ----------
        sport     -- Identificador del deporte.
        event_id  -- ID del evento en The Odds API (de RawOddsEvent.event_id).
        markets   -- Lista de market keys avanzados.
                    Ejemplos MLB: ['pitcher_strikeouts', 'batter_hits',
                    'h2h_1st_5_innings', 'totals_1st_5_innings']

        Retorna
        -------
        OddsAPIResponse. Si 401/403: is_auth_error=True — la API key
        no tiene acceso a event-level markets en el plan actual.
        """
        params = self._build_params(
            markets=markets,
            odds_format=odds_format,
        )

        url = f"{self._config.base_url}/sports/{sport}/events/{event_id}/odds"
        raw_response = self._request(url, params)

        if not raw_response["success"]:
            return OddsAPIResponse(
                success=False,
                events=[],
                error_type=raw_response["error_type"],
                error_message=raw_response["error_message"],
                requests_remaining=raw_response.get("requests_remaining"),
                requests_used=raw_response.get("requests_used"),
            )

        # El endpoint de event-level retorna un único evento, no lista
        data = raw_response["data"]
        if isinstance(data, list) and data:
            events = self._parse_events(data, sport)
        elif isinstance(data, dict) and data.get("id"):
            events = self._parse_events([data], sport)
        else:
            events = []

        return OddsAPIResponse(
            success=True,
            events=events,
            requests_remaining=raw_response.get("requests_remaining"),
            requests_used=raw_response.get("requests_used"),
        )

    # ── Construcción de parámetros ─────────────────────────────────────────────

    def _build_params(
        self,
        markets: list[str],
        odds_format: str,
    ) -> dict:
        """
        Construye el dict de parámetros para la query string.

        Usa bookmakers si está configurado, regions si no.
        La API no acepta ambos simultáneamente.
        """
        params: dict = {
            "apiKey":     self._config.api_key,
            "markets":    ",".join(markets),
            "oddsFormat": odds_format,
        }

        if self._config.bookmakers:
            params["bookmakers"] = self._config.bookmakers
        else:
            params["regions"] = self._config.regions

        return params

    # ── HTTP core ──────────────────────────────────────────────────────────────

    def _request(self, url: str, params: dict) -> dict:
        """
        Ejecuta la request HTTP con retry configurable.

        Retorna un dict interno con keys:
            success, data, error_type, error_message,
            requests_remaining, requests_used.

        No lanza excepciones para errores de API — los encapsula en el
        dict de resultado. Solo puede lanzar si requests no está
        instalado (ImportError en inicialización del módulo — fallo
        en construcción del cliente, no en uso).
        """
        if not _REQUESTS_AVAILABLE:
            return {
                "success":      False,
                "data":         None,
                "error_type":   ERROR_UNAVAILABLE,
                "error_message": (
                    "El paquete 'requests' no está instalado. "
                    "Ejecutar: pip install requests"
                ),
            }

        last_error: dict = {}
        max_attempts = 1 + max(0, self._config.max_retries)

        for attempt in range(max_attempts):

            if attempt > 0:
                backoff = self._config.retry_backoff_base ** attempt
                time.sleep(backoff)

            try:
                response = _requests.get( # type: ignore
                    url,
                    params=params,
                    timeout=self._config.timeout_seconds,
                )
            except _requests.exceptions.Timeout: # type: ignore
                last_error = {
                    "success":       False,
                    "data":          None,
                    "error_type":    ERROR_NETWORK,
                    "error_message": (
                        f"Timeout después de {self._config.timeout_seconds}s "
                        f"en {url}"
                    ),
                }
                continue
            except _requests.exceptions.ConnectionError as e: # type: ignore
                last_error = {
                    "success":       False,
                    "data":          None,
                    "error_type":    ERROR_NETWORK,
                    "error_message": f"Error de conexión: {e}",
                }
                continue
            except _requests.exceptions.RequestException as e: # type: ignore
                last_error = {
                    "success":       False,
                    "data":          None,
                    "error_type":    ERROR_NETWORK,
                    "error_message": f"Error de red: {e}",
                }
                continue

            # Extraer headers de créditos antes de procesar el status
            remaining = self._parse_int_header(
                response, "x-requests-remaining"
            )
            used = self._parse_int_header(response, "x-requests-used")

            # 401 / 403 — error permanente, no reintentar
            if response.status_code in (401, 403):
                return {
                    "success":            False,
                    "data":               None,
                    "error_type":         ERROR_AUTH,
                    "error_message":      (
                        f"Autenticación fallida ({response.status_code}). "
                        f"Verificar que ODDS_API_KEY sea válida y activa. "
                        f"Key usada: {self._config.redacted_key()}"
                    ),
                    "requests_remaining": remaining,
                    "requests_used":      used,
                }

            # 429 — rate limit, no reintentar
            if response.status_code == 429:
                return {
                    "success":            False,
                    "data":               None,
                    "error_type":         ERROR_RATE_LIMIT,
                    "error_message":      (
                        "Rate limit alcanzado. "
                        f"Créditos restantes: {remaining}. "
                        "Revisar plan de The Odds API."
                    ),
                    "requests_remaining": remaining,
                    "requests_used":      used,
                }

            # 404 — recurso no encontrado (sport inválido, event_id incorrecto)
            if response.status_code == 404:
                return {
                    "success":            False,
                    "data":               None,
                    "error_type":         ERROR_NOT_FOUND,
                    "error_message":      (
                        f"Recurso no encontrado (404): {url}. "
                        f"Verificar que el sport identifier sea válido."
                    ),
                    "requests_remaining": remaining,
                    "requests_used":      used,
                }

            # 5xx — error del servidor, reintentable
            if response.status_code >= 500:
                last_error = {
                    "success":            False,
                    "data":               None,
                    "error_type":         ERROR_SERVER,
                    "error_message":      (
                        f"Error del servidor ({response.status_code}). "
                        f"Reintento {attempt + 1}/{max_attempts}."
                    ),
                    "requests_remaining": remaining,
                    "requests_used":      used,
                }
                continue

            # 2xx — éxito
            try:
                data = response.json()
            except ValueError:
                return {
                    "success":            False,
                    "data":               None,
                    "error_type":         ERROR_PARSE,
                    "error_message":      (
                        "Respuesta no es JSON válido. "
                        f"Content-Type: {response.headers.get('content-type')}"
                    ),
                    "requests_remaining": remaining,
                    "requests_used":      used,
                }

            # La API puede retornar {"message": "..."} en respuestas de error
            if isinstance(data, dict) and data.get("message"):
                return {
                    "success":            False,
                    "data":               None,
                    "error_type":         ERROR_SERVER,
                    "error_message":      self._redact(str(data["message"])),
                    "requests_remaining": remaining,
                    "requests_used":      used,
                }

            return {
                "success":            True,
                "data":               data,
                "requests_remaining": remaining,
                "requests_used":      used,
            }

        # Todos los reintentos agotados
        return last_error

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_events(
        self,
        data: list[dict],
        sport: str,
    ) -> list[RawOddsEvent]:
        """
        Convierte la lista de dicts de la API a list[RawOddsEvent].

        No hace parsing de bookmakers/markets — eso pertenece a
        OddsNormalizer. Solo extrae los campos de identidad del evento
        y deja bookmakers como dict crudo.

        Eventos con campos de identidad faltantes se omiten silenciosamente
        con un registro en loaded_markets vacío — el normalizer los
        descartará al no encontrar bookmakers útiles.
        """
        events: list[RawOddsEvent] = []

        for raw in data:
            event_id     = raw.get("id")
            home_team    = raw.get("home_team")
            away_team    = raw.get("away_team")
            commence     = raw.get("commence_time", "")
            bookmakers   = raw.get("bookmakers", [])

            # Campos de identidad obligatorios
            if not event_id or not home_team or not away_team:
                continue

            # Detectar qué markets fueron efectivamente cargados
            loaded: set[str] = set()
            for bm in bookmakers:
                for mkt in bm.get("markets", []):
                    key = mkt.get("key")
                    if key:
                        loaded.add(key)

            events.append(RawOddsEvent(
                event_id=event_id,
                sport=sport,
                home_team=home_team,
                away_team=away_team,
                commence_time=commence,
                bookmakers=bookmakers,
                loaded_markets=loaded,
            ))

        return events

    # ── Utilidades ─────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_int_header(response, header_name: str) -> int | None:
        """Extrae un header HTTP numérico de forma segura."""
        value = response.headers.get(header_name)
        if value is None:
            return None
        try:
            return int(value)
        except (ValueError, TypeError):
            return None

    def _redact(self, text: str) -> str:
        """Reemplaza la API key en cualquier string para evitar leaks en logs."""
        if not self._config.api_key:
            return text
        return text.replace(self._config.api_key, self._config.redacted_key())