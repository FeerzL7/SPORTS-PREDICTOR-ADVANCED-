"""
core/contracts/event.py

Unidad de trabajo primaria del pipeline.

Un Event representa un partido o evento deportivo de cualquier deporte
soportado por el sistema. Se crea en Stage 1 (EVENT DISCOVERY) y viaja
inmutable a través de los 12 stages del pipeline.

Deportes soportados:
    mlb, nba, nfl, nhl, soccer, tennis, golf

Uso típico:
    event = Event(
        event_id     = "a1b2c3d4-...",
        sport        = "mlb",
        league       = "MLB",
        season_start = 2026,
        season_end   = 2026,
        date         = "2026-06-08",
        start_time   = "2026-06-08T18:10:00Z",
        home_team_id = "yankees",
        away_team_id = "redsox",
        home_team    = "New York Yankees",
        away_team    = "Boston Red Sox",
        venue_id     = "yankee-stadium",
        venue_name   = "Yankee Stadium",
        status       = EventStatus.SCHEDULED,
        provider_ids = {
            "odds_api":  "abc123",
            "mlb_stats": "745302",
        },
    )
"""

from __future__ import annotations

from dataclasses import dataclass


class EventStatus:
    """
    Constantes de estado de un evento.

    Se implementa como clase con atributos de clase en lugar de Enum
    para garantizar compatibilidad directa con serialización JSON/CSV
    sin encoders personalizados.

    Uso:
        event.status == EventStatus.FINAL
        status = EventStatus.SCHEDULED
    """

    SCHEDULED = "scheduled"   # Partido programado, no iniciado
    LIVE      = "live"        # En curso
    FINAL     = "final"       # Finalizado con resultado oficial
    POSTPONED = "postponed"   # Pospuesto (sin nueva fecha confirmada)
    CANCELLED = "cancelled"   # Cancelado definitivamente

    # Conjunto para validación rápida sin instanciar la clase
    ALL: frozenset[str] = frozenset({
        SCHEDULED,
        LIVE,
        FINAL,
        POSTPONED,
        CANCELLED,
    })


@dataclass(frozen=True)
class Event:
    """
    Representa un partido o evento deportivo. Inmutable una vez creado.

    Se crea en Stage 1 del pipeline (SportDataProvider.get_events) y
    es consumido por todos los stages posteriores sin modificación.
    La inmutabilidad (frozen=True) garantiza trazabilidad completa:
    ningún stage puede alterar los datos de identidad del evento.

    Campos
    ------
    Identidad:
        event_id     -- UUID generado por el sistema al crear el evento.
                        Clave primaria en el ledger y los snapshots.
        sport        -- Identificador de deporte en minúsculas.
                        Valores válidos: 'mlb', 'nba', 'nfl', 'nhl',
                        'soccer', 'tennis', 'golf'.
        league       -- Nombre de la liga o competición.
                        Ejemplos: 'MLB', 'EPL', 'NBA', 'ATP'.
        season_start -- Año de inicio de la temporada (int).
                        MLB 2026 → 2026. NHL 2025-26 → 2025.
        season_end   -- Año de fin de la temporada (int).
                        MLB 2026 → 2026. NHL 2025-26 → 2026.
                        Para temporadas de año único: season_end == season_start.

    Temporalidad:
        date         -- Fecha del partido en formato 'YYYY-MM-DD'.
                        Zona horaria: local del venue (para display).
        start_time   -- Hora de inicio en ISO-8601 UTC.
                        Ejemplo: '2026-06-08T18:10:00Z'.
                        Base para todos los cálculos de tiempo.

    Participantes:
        home_team_id -- ID canónico del equipo local, estable entre temporadas.
                        No usar nombres como IDs (cambian con franquicias).
        away_team_id -- ID canónico del equipo visitante.
        home_team    -- Nombre completo para display ('New York Yankees').
        away_team    -- Nombre completo para display ('Boston Red Sox').

    Venue:
        venue_id     -- ID canónico del estadio.
                        Usado para lookup de park factors, coordenadas
                        climáticas y clasificación indoor/outdoor.
        venue_name   -- Nombre para display ('Yankee Stadium').

    Estado:
        status       -- Estado actual del evento.
                        Usar constantes de EventStatus.

    IDs externos:
        provider_ids -- Mapa de IDs del evento en APIs externas.
                        El Core no asume qué fuentes existen.
                        Cada sport plugin popula las claves relevantes.
                        Ejemplo: {'odds_api': 'abc123', 'mlb_stats': '745302'}.

    Propiedades derivadas
    ---------------------
        matchup              -- String 'Away @ Home' para display y ledger.
        season_label         -- String legible de la temporada ('2026' o '2025-26').
        is_single_year_season -- True si la temporada es de año único.
    """

    # ── Identidad ──────────────────────────────────────────────────────────────
    event_id:     str
    sport:        str
    league:       str
    season_start: int
    season_end:   int

    # ── Temporalidad ───────────────────────────────────────────────────────────
    date:         str   # 'YYYY-MM-DD'
    start_time:   str   # ISO-8601 UTC, ej: '2026-06-08T18:10:00Z'

    # ── Participantes ──────────────────────────────────────────────────────────
    home_team_id: str
    away_team_id: str
    home_team:    str
    away_team:    str

    # ── Venue ──────────────────────────────────────────────────────────────────
    venue_id:     str
    venue_name:   str

    # ── Estado ─────────────────────────────────────────────────────────────────
    status:       str   # Usar EventStatus.SCHEDULED, etc.

    # ── IDs externos ───────────────────────────────────────────────────────────
    provider_ids: dict[str, str]

    # ── Propiedades derivadas ──────────────────────────────────────────────────

    @property
    def matchup(self) -> str:
        """
        Formato canónico 'Away @ Home'.

        Usado como identificador de partido en el ledger, los CSV de
        predicciones y los mensajes de Telegram. Consistente con el
        formato del sistema MLB original.

        Ejemplo:
            'Boston Red Sox @ New York Yankees'
        """
        return f"{self.away_team} @ {self.home_team}"

    @property
    def season_label(self) -> str:
        """
        Representación legible de la temporada.

        Temporada de año único  → '2026'
        Temporada que cruza año → '2025-26'

        Usado para display en el dashboard y agrupación en reportes.
        Para comparaciones y filtros usar season_start / season_end directamente.
        """
        if self.season_start == self.season_end:
            return str(self.season_start)
        # Año final como dos dígitos: 2025-26, no 2025-2026
        return f"{self.season_start}-{str(self.season_end)[-2:]}"

    @property
    def is_single_year_season(self) -> bool:
        """
        True para deportes con temporada de año calendario único.

        True:  MLB, NFL, golf (temporada dentro de un año)
        False: NHL, NBA, soccer europeo (temporada que cruza dos años)
        """
        return self.season_start == self.season_end