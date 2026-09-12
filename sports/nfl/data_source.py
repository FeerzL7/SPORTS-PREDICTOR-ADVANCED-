"""
sports/nfl/data_source.py

NFLDataSource: capa de acceso y caché centralizada sobre nfl_data_py.

Por qué existe esta capa (no tiene equivalente en MLB)
-------------------------------------------------------
La MLB Stats API es de consulta puntual: pides un partido, te devuelve
un partido. Cada fetcher del plugin MLB puede llamar directamente a la
API sin desperdiciar ancho de banda.

nfl_data_py funciona al revés: descarga DataFrames completos de
temporada desde el CDN de nflverse. Un `import_pbp_data([2026])` son
~50 MB. Si cada fetcher del plugin NFL llamara directamente:

    team_stats.py  → import_pbp_data([2026])      ~50 MB
    injuries.py    → import_injuries([2026])       ~2 MB
    rest.py        → import_schedules([2026])      ~1 MB
    h2h.py         → import_schedules([2022..26])  ~5 MB
                                          Total:  ~58 MB por evento

Con 16 partidos en un domingo de NFL, eso serían ~900 MB de descargas
redundantes por ejecución del pipeline. NFLDataSource descarga una vez
por temporada y sirve a todos los fetchers desde caché.

Dos niveles de caché
---------------------
1. MEMORIA — durante una ejecución del pipeline, el mismo DataFrame se
   sirve N veces sin tocar disco ni red. Se pierde al terminar el
   proceso.

2. DISCO (parquet) — entre ejecuciones, evita re-descargar de internet.
   Persistente en output/cache/nfl/.

TTL diferenciado por naturaleza del dato
------------------------------------------
Los datos de temporadas pasadas son inmutables: un partido de 2023 no
va a cambiar nunca. Los de la temporada en curso se actualizan los
martes tras cada jornada.

    Temporada pasada  → TTL infinito (nunca re-descargar)
    Temporada actual  → TTL 1 día (re-descargar tras cada jornada)
    Injuries          → TTL 12 horas (cambian durante la semana)

Latencia conocida de nflverse
-------------------------------
Los datos se publican los martes tras cada semana de juegos, con ~24h
de latencia post-partido. Esto significa que el injury report del
miércoles/jueves NO estará disponible hasta el martes siguiente.

Mitigación: usar el injury report de la semana previa como baseline y
reducir data_quality en TeamFeatures cuando la información sea stale.
El NFLProjectionModel lo refleja bajando confidence en la Projection.

Dependencia aislada
--------------------
pandas y nfl_data_py son dependencias EXCLUSIVAS del plugin NFL. El
Core nunca las importa. Si nfl_data_py no está instalado, este módulo
falla con un mensaje claro pero MLB sigue funcionando — mismo patrón
que el import opcional de requests en el resto del proyecto.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import nfl_data_py as _nfl
    import pandas as _pd
except ImportError:
    # Asignar None explícitamente en el except (no dejar las variables
    # sin vincular) para que el type checker pueda narrar correctamente
    # en los call sites: `if _nfl is None: raise` descarta la rama None.
    # Sin esta asignación, Pyright marca "possibly unbound" en cada uso.
    _nfl = None  # type: ignore[assignment]
    _pd  = None  # type: ignore[assignment]


# Directorio de caché en disco
_DEFAULT_CACHE_DIR = "output/cache/nfl"

# TTL en segundos por tipo de dato
_TTL_CURRENT_SEASON: int = 86_400      # 1 día — se actualiza los martes
_TTL_PAST_SEASON:    int = 31_536_000  # 1 año — datos inmutables
_TTL_INJURIES:       int = 43_200      # 12 horas — cambian en la semana

# Columnas mínimas de play-by-play que necesita el plugin.
# Cargar solo estas reduce la descarga de ~50 MB a ~8 MB.
_PBP_COLUMNS = [
    "game_id", "season", "week", "posteam", "defteam",
    "play_type", "epa", "wpa", "success", "yards_gained",
    "down", "ydstogo", "qtr", "score_differential",
    "pass", "rush", "touchdown", "interception", "fumble_lost",
]


class NFLDataSourceError(RuntimeError):
    """Error al acceder a los datos de nflverse."""


class NFLDataSource:
    """
    Acceso cacheado a los datasets de nfl_data_py.

    Una única instancia sirve a todos los fetchers del plugin NFL.
    El NFLPlugin la crea una vez y la inyecta en cada fetcher.

    Parámetros
    ----------
    cache_dir       -- Directorio para caché parquet en disco.
                      Default 'output/cache/nfl'.
    use_disk_cache  -- Si False, solo cachea en memoria (útil en tests).
                      Default True.
    current_season  -- Temporada considerada "actual" para efectos de
                      TTL. None = año calendario actual.
    """

    def __init__(
        self,
        cache_dir:      str  = _DEFAULT_CACHE_DIR,
        use_disk_cache: bool = True,
        current_season: int | None = None,
    ) -> None:
        if _nfl is None or _pd is None:
            raise NFLDataSourceError(
                "nfl_data_py no está instalado. El plugin NFL lo requiere.\n"
                "Instalar con: pip install nfl_data_py\n"
                "(El plugin MLB no se ve afectado — esta dependencia es "
                "exclusiva de NFL.)"
            )

        # Guardar los módulos como atributos de instancia tras validar
        # que no son None. Esto permite a Pyright narrar el tipo en todos
        # los usos posteriores: el chequeo `if _nfl is None: raise` de
        # arriba garantiza que a partir de aquí ambos están vinculados,
        # pero el type checker no puede propagar esa garantía a través
        # de closures (los lambda de los loaders) ni a otros métodos.
        # Con self._nfl / self._pd el narrowing ocurre una sola vez, aquí.
        self._nfl = _nfl
        self._pd  = _pd

        self._cache_dir      = Path(cache_dir)
        self._use_disk_cache = use_disk_cache
        self._current_season = current_season or _current_nfl_season()

        # Caché en memoria: key → DataFrame
        self._memory: dict[str, Any] = {}

        if self._use_disk_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets públicos ─────────────────────────────────────────────────────

    def load_schedules(self, seasons: list[int]):
        """
        Calendario de partidos: fechas, equipos, scores finales.

        Usado por:
            - NFLScheduleFetcher (Stage 1: get_events)
            - NFLRestFetcher (días de descanso entre partidos)
            - NFLH2HFetcher (historial de enfrentamientos)
            - NFLSettlementProvider (resultados finales)

        Columnas clave: game_id, season, week, gameday, gametime,
        away_team, home_team, away_score, home_score, result,
        overtime, div_game, roof, surface, temp, wind.

        Retorna
        -------
        pandas.DataFrame con todos los partidos de las temporadas dadas.
        """
        return self._load(
            key      = f"schedules_{_seasons_key(seasons)}",
            loader   = lambda: self._nfl.import_schedules(seasons),
            seasons  = seasons,
        )

    def load_pbp(
        self,
        seasons: list[int],
        columns: list[str] | None = None,
    ):
        """
        Play-by-play con EPA y success rate.

        El dataset más pesado (~50 MB por temporada completa). Por
        defecto carga solo las columnas que el plugin necesita, lo que
        reduce la descarga a ~8 MB.

        Usado por:
            - NFLTeamStatsFetcher (EPA/play ofensivo y defensivo)

        Columnas clave: game_id, posteam, defteam, epa, success,
        play_type, yards_gained.

        Parámetros
        ----------
        seasons  -- Temporadas a cargar.
        columns  -- Columnas específicas. None = _PBP_COLUMNS.

        Retorna
        -------
        pandas.DataFrame con las jugadas de las temporadas dadas.
        """
        cols = columns or _PBP_COLUMNS
        return self._load(
            key     = f"pbp_{_seasons_key(seasons)}",
            loader  = lambda: self._nfl.import_pbp_data(
                seasons, columns=cols, downcast=True, cache=False
            ),
            seasons = seasons,
        )

    def load_injuries(self, seasons: list[int]):
        """
        Injury report semanal.

        TTL más corto (12h) porque cambia durante la semana — aunque
        nflverse publica con ~24h de latencia, así que el dato más
        fresco disponible suele ser del martes anterior.

        Usado por:
            - NFLInjuryFetcher (estado del QB y titulares)

        Columnas clave: season, week, team, player, position,
        report_status ('Out', 'Doubtful', 'Questionable'),
        practice_status.

        Retorna
        -------
        pandas.DataFrame con los reportes de lesión.
        """
        return self._load(
            key     = f"injuries_{_seasons_key(seasons)}",
            loader  = lambda: self._nfl.import_injuries(seasons),
            seasons = seasons,
            ttl     = _TTL_INJURIES,
        )

    def load_rosters(self, seasons: list[int]):
        """
        Rosters semanales: quién está en el equipo y en qué posición.

        Usado por:
            - NFLInjuryFetcher (identificar el QB titular)

        Columnas clave: season, team, player_name, position,
        depth_chart_position, status.
        """
        return self._load(
            key     = f"rosters_{_seasons_key(seasons)}",
            loader  = lambda: self._nfl.import_seasonal_rosters(seasons),
            seasons = seasons,
        )

    def load_team_descriptions(self):
        """
        Metadatos de equipos: abreviaciones, nombres completos, colores.

        Usado para mapear entre las abreviaciones de nflverse ('KC')
        y los nombres completos de The Odds API ('Kansas City Chiefs').

        Dataset pequeño y estático — TTL largo.
        """
        return self._load(
            key     = "team_desc",
            loader  = lambda: self._nfl.import_team_desc(),
            seasons = [],
            ttl     = _TTL_PAST_SEASON,
        )

    # ── Utilidades de caché ───────────────────────────────────────────────────

    def clear_cache(self, memory_only: bool = False) -> None:
        """
        Limpia el caché.

        Parámetros
        ----------
        memory_only  -- Si True, solo limpia memoria y conserva los
                       parquet en disco. Útil para liberar RAM sin
                       forzar re-descargas.
        """
        self._memory.clear()
        if not memory_only and self._use_disk_cache:
            for f in self._cache_dir.glob("*.parquet"):
                try:
                    f.unlink()
                except OSError:
                    pass

    def cache_info(self) -> dict:
        """Estado del caché para diagnóstico."""
        disk_files = (
            list(self._cache_dir.glob("*.parquet"))
            if self._use_disk_cache and self._cache_dir.exists()
            else []
        )
        disk_bytes = sum(f.stat().st_size for f in disk_files)
        return {
            "memory_keys":   sorted(self._memory.keys()),
            "memory_count":  len(self._memory),
            "disk_files":    len(disk_files),
            "disk_mb":       round(disk_bytes / 1_048_576, 2),
            "cache_dir":     str(self._cache_dir),
            "current_season": self._current_season,
        }

    # ── Motor de caché ────────────────────────────────────────────────────────

    def _load(
        self,
        key:     str,
        loader,
        seasons: list[int],
        ttl:     int | None = None,
    ):
        """
        Carga un dataset con caché en dos niveles.

        Orden de resolución:
            1. Memoria (misma ejecución del pipeline)
            2. Disco parquet (entre ejecuciones, si no expiró el TTL)
            3. Descarga desde nflverse (última opción)

        Parámetros
        ----------
        key      -- Identificador único del dataset.
        loader   -- Callable sin argumentos que descarga el DataFrame.
        seasons  -- Temporadas incluidas. Determina el TTL por defecto.
        ttl      -- TTL explícito en segundos. None = derivar de seasons.
        """
        # ── Nivel 1: memoria ──────────────────────────────────────
        if key in self._memory:
            return self._memory[key]

        effective_ttl = ttl if ttl is not None else self._ttl_for(seasons)

        # ── Nivel 2: disco ────────────────────────────────────────
        if self._use_disk_cache:
            cached = self._read_parquet(key, effective_ttl)
            if cached is not None:
                self._memory[key] = cached
                return cached

        # ── Nivel 3: descarga ─────────────────────────────────────
        try:
            df = loader()
        except Exception as e:
            raise NFLDataSourceError(
                f"Fallo al descargar '{key}' desde nflverse: "
                f"{type(e).__name__}: {e}"
            ) from e

        self._memory[key] = df
        if self._use_disk_cache:
            self._write_parquet(key, df)

        return df

    def _ttl_for(self, seasons: list[int]) -> int:
        """
        TTL según si el dataset incluye la temporada actual.

        Datos de temporadas pasadas son inmutables — nunca expiran.
        Datos de la temporada en curso se actualizan cada martes.
        """
        if not seasons:
            return _TTL_PAST_SEASON
        return (
            _TTL_CURRENT_SEASON
            if self._current_season in seasons
            else _TTL_PAST_SEASON
        )

    def _parquet_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.parquet"

    def _read_parquet(self, key: str, ttl: int):
        """Lee del caché parquet si existe y no expiró."""
        path = self._parquet_path(key)
        if not path.exists():
            return None
        try:
            age = time.time() - path.stat().st_mtime
            if age > ttl:
                return None
            return self._pd.read_parquet(path)
        except Exception:
            # Parquet corrupto o pyarrow no disponible — re-descargar
            return None

    def _write_parquet(self, key: str, df) -> None:
        """Escribe el DataFrame al caché parquet."""
        try:
            df.to_parquet(self._parquet_path(key), index=False)
        except Exception:
            # El caché en disco es opcional — no abortar si falla
            pass

    def __repr__(self) -> str:
        return (
            f"NFLDataSource(season={self._current_season}, "
            f"memory={len(self._memory)} datasets, "
            f"disk={'on' if self._use_disk_cache else 'off'})"
        )


# ── Utilidades de módulo ──────────────────────────────────────────────────────

def _current_nfl_season() -> int:
    """
    Temporada NFL actual.

    La temporada NFL cruza el año calendario: la temporada 2026 va de
    septiembre 2026 a febrero 2027. nflverse la identifica por el año
    de inicio, así que en enero-febrero seguimos en la temporada del
    año anterior.
    """
    now = datetime.now(timezone.utc)
    # Enero y febrero pertenecen a la temporada del año previo
    return now.year - 1 if now.month <= 2 else now.year


def _seasons_key(seasons: list[int]) -> str:
    """Genera una key de caché estable desde una lista de temporadas."""
    if not seasons:
        return "all"
    return "_".join(str(s) for s in sorted(seasons))


def is_available() -> bool:
    """
    True si nfl_data_py está instalado y utilizable.

    Deriva del estado real de los imports en vez de una bandera
    separada — una bandera independiente podría desincronizarse del
    estado real si alguien modifica el bloque try/except sin
    actualizarla. Este patrón se aplicó también en los módulos MLB
    durante la auditoría de tipado.
    """
    return _nfl is not None and _pd is not None