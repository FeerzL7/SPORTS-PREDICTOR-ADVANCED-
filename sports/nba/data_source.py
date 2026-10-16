"""
sports/nba/data_source.py — Acceso cacheado a los datos de NBA.

OBJETIVO
--------
Capa de E/S pura para el plugin NBA. Una única instancia sirve a todos
los fetchers: el NBAPlugin la crea una vez y la inyecta.

Es el equivalente de `sports/nfl/data_source.py` (nflverse) y de
`sports/soccer/data_source.py` (football-data + Understat). Mismo
contrato de caché en dos niveles, misma separación de
responsabilidades, misma forma de `is_available()`.

ARQUITECTURA
------------
    NBADataSource
        ├── load_team_box(seasons)        → box score por equipo y partido
        ├── load_closing_lines()          → spread y total de cierre
        ├── load_legacy_odds_archive()    → segunda fuente de cuotas
        └── load_schedule()               → calendario maestro

    Resolución de cada carga:
        1. memoria   (misma ejecución del pipeline)
        2. disco     (entre ejecuciones, si no expiró el TTL)
        3. descarga  (última opción)

LÍMITE DE RESPONSABILIDAD — leer antes de extender
---------------------------------------------------
Este módulo **no aplica ninguna barrera temporal**. Devuelve los
datos crudos tal como vienen de la fuente, temporada completa
incluida.

La barrera la impone el provider (`NBADataProvider`), igual que
`NFLDataProvider._cutoff_week` y `SoccerDataProvider._as_of`. Mezclar
las dos responsabilidades es exactamente cómo se cuela el look-ahead:
en NFL el clima observado DURANTE el partido infló el ROI de totales
ocho puntos, y en soccer el xG del propio partido era el análogo.

Por la misma razón este módulo tampoco filtra la ventana de cierre.
Añade la columna derivada `lead_minutes` —un hecho aritmético sobre
los datos, no una decisión— y expone `CLOSE_WINDOW_MINUTES` para que
provider y backtest apliquen la MISMA regla desde una sola
definición. Quién filtra, y cuándo, es decisión del consumidor.

FUENTE
------
`sportsdataverse/hoopR-nba-data`, licencia CC BY 4.0. Resultados y
calendario derivados de ESPN con actualización diaria (07:00 UTC);
cuotas de cierre derivadas de The Odds API.

Elegida tras la auditoría de la tarea 12.0, que verificó sobre datos
reales: 7.836 partidos liquidables, cruce por `game_id` al 99,89%, y
dos fuentes de cuotas independientes que coinciden dentro de 1 punto
en el 95,5% de los spreads. Ver `scripts/audit_nba_sources.py`.

CONVENCIÓN DE TEMPORADA — la trampa de este módulo
---------------------------------------------------
La temporada se identifica por su año de **FIN**, porque así nombra
los archivos la fuente:

    team_box_2026.parquet  →  temporada 2025-26
    team_box_2021.parquet  →  temporada 2020-21

Es la convención OPUESTA a la de NFL, donde nflverse identifica la
temporada por el año de inicio. Confundirlas desplaza todo un año y
el error es silencioso: los datos cargan, solo son del año
equivocado. Usar `season_label()` al imprimir cualquier temporada.

**No derivar la temporada de la fecha.** Una primera versión de este
módulo la calculaba con un corte de mes (agosto en adelante → año
siguiente). Funciona para una temporada normal y falla en las de
COVID: la burbuja de 2019-20 se jugó entre julio y octubre de 2020,
así que el corte mandaba esos partidos a 2020-21. El síntoma fue un
cruce del 97,72% en vez del 99,89% de la auditoría — 170 partidos
perdidos sin un solo error en consola.

ESPN ya etiqueta cada partido con su `season` de forma autoritativa.
La temporada se **lee del dato**, nunca se calcula. `load_settled_games()`
la trae desde el box score por eso mismo.

RIESGOS CONOCIDOS
-----------------
  * `closing_lines_odds_api.parquet` es un backfill estático (un
    único commit, 2026-08-26) y ningún script de la fuente lo
    regenera. No crecerá más allá de 2025-26. Ver la tarea 12.6.
  * Las cuotas traen LÍNEAS, no precios. El moneyline no es
    backtesteable con esta fuente.
  * Las cuotas son un CONSENSO de 10-19 casas. No es la misma
    cantidad que el mejor precio que guarda `extract_best()` en
    producción; no deben compararse sin declarar cuál se usa.
  * ~1% de los snapshots son POSTERIORES al tip-off. `lead_minutes`
    sale negativo en esos casos y el consumidor debe descartarlos.

VENTAJAS
--------
  * Sin credenciales, sin rate limit, sin coste: archivos estáticos
    en GitHub. El backtest es reproducible indefinidamente.
  * Una sola clave (`game_id` de ESPN) cruza cuotas y resultados, lo
    que evita la reconciliación de nombres que en soccer costó 437
    líneas y produjo un bug real.

MEJORAS FUTURAS
---------------
  * Contrastar los resultados contra `nba_api` (la fuente documenta
    defectos de ESPN que corrige en compilación, hoopR #178/#140/#146).
  * Cuando exista el archivo propio de la tarea 12.6, añadir
    `load_own_closing_lines()` con precios, y con ello el moneyline.
"""

from __future__ import annotations

import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    import pandas as _pd
except ImportError:  # pragma: no cover
    _pd = None  # type: ignore[assignment]

try:
    import requests as _requests
except ImportError:  # pragma: no cover
    _requests = None  # type: ignore[assignment]


# ── Configuración de la fuente ────────────────────────────────────────────────

_RAW_BASE = (
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"
)

_PATH_CLOSING_LINES = "betting_lines/closing_lines_odds_api.parquet"
_PATH_LEGACY_ODDS   = "betting_lines/games-archive.json"
_PATH_SCHEDULE      = "schedules/nba_schedule_master.parquet"


def _path_team_box(season: int) -> str:
    return f"team_box/parquet/team_box_{season}.parquet"


_DEFAULT_CACHE_DIR = "output/cache/nba"

# TTL en segundos. Las temporadas cerradas son inmutables; la que está
# en curso la regenera la fuente cada día a las 07:00 UTC.
_TTL_PAST_SEASON    = 365 * 24 * 3600
_TTL_CURRENT_SEASON = 6 * 3600

# El backfill de cuotas no se regenera nunca. Un TTL corto solo
# provocaría descargas inútiles.
_TTL_STATIC = _TTL_PAST_SEASON

_HTTP_TIMEOUT = 120

# Primera temporada con box score utilizable en la fuente.
FIRST_BOX_SEASON = 2003

# Primera temporada con cuotas de cierre del consenso (la burbuja de
# 2019-20 aporta solo 6 partidos, así que la primera temporada
# completa es 2020-21).
FIRST_CLOSING_SEASON = 2021

# Ventana en la que un snapshot cuenta como cierre, en minutos antes
# del tip-off. Validada en la tarea 12.0 sobre 7.942 snapshots:
# dentro de ella caen 7.845 (98,78%); por debajo de 0 hay 76 snapshots
# POSTERIORES al tip-off, que son look-ahead puro.
CLOSE_WINDOW_MINUTES = (0.0, 60.0)

# ESPN codifica el tipo de partido en `season_type`.
SEASON_TYPE_PRESEASON    = 1
SEASON_TYPE_REGULAR      = 2
SEASON_TYPE_PLAYOFFS     = 3
SEASON_TYPE_PLAY_IN      = 5


class NBADataSourceError(RuntimeError):
    """Error al acceder a los datos de hoopR-nba-data."""


# ── Clase principal ──────────────────────────────────────────────────────────

class NBADataSource:
    """
    Acceso cacheado a los datasets de hoopR-nba-data.

    Parámetros
    ----------
    cache_dir       -- Directorio para caché en disco.
                      Default 'output/cache/nba'.
    use_disk_cache  -- Si False, solo cachea en memoria (útil en tests).
                      Default True.
    current_season  -- Temporada considerada "en curso" para efectos de
                      TTL, por año de FIN. None = derivar del reloj.
    """

    def __init__(
        self,
        cache_dir:      str  = _DEFAULT_CACHE_DIR,
        use_disk_cache: bool = True,
        current_season: int | None = None,
    ) -> None:
        if _pd is None or _requests is None:
            missing = []
            if _pd is None:
                missing.append("pandas")
            if _requests is None:
                missing.append("requests")
            raise NBADataSourceError(
                f"Faltan dependencias del plugin NBA: {', '.join(missing)}.\n"
                f"Instalar con: pip install {' '.join(missing)}\n"
                f"(Los plugins MLB, NFL y soccer no se ven afectados.)"
            )

        # Igual que en NFLDataSource: guardar los módulos como atributos
        # de instancia tras validar que no son None permite al type
        # checker estrechar el tipo una sola vez, aquí, en vez de en
        # cada uso dentro de los lambda de los loaders.
        self._pd       = _pd
        self._requests = _requests

        self._cache_dir      = Path(cache_dir)
        self._use_disk_cache = use_disk_cache
        self._current_season = current_season or current_nba_season()

        self._memory: dict[str, Any] = {}

        if self._use_disk_cache:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets públicos ────────────────────────────────────────────────────

    def load_team_box(self, seasons: list[int]):
        """
        Box score por equipo y partido. Dos filas por partido.

        Usado por:
            - NBAScheduleFetcher      (Stage 1: get_events)
            - NBATeamStatsFetcher     (ORtg, DRtg, Pace acumulados)
            - NBASettlementProvider   (resultados finales)

        Las temporadas se indican por año de FIN: `[2025, 2026]` son
        2024-25 y 2025-26. Ver la nota de convención en el encabezado
        del módulo.

        Columnas clave: game_id, season, season_type, game_date,
        team_id, team_abbreviation, team_home_away, team_score,
        opponent_team_score, field_goals_attempted,
        free_throws_attempted, offensive_rebounds, total_turnovers.

        Las cuatro últimas son las que estiman posesiones por la
        fórmula de Dean Oliver:

            POS ≈ FGA − OREB + TOV + 0.44 × FTA

        Retorna
        -------
        pandas.DataFrame con todas las temporadas concatenadas,
        `game_id` como int64 y `game_date` como datetime.
        """
        if not seasons:
            raise NBADataSourceError(
                "load_team_box() requiere al menos una temporada."
            )

        invalid = [s for s in seasons if s < FIRST_BOX_SEASON]
        if invalid:
            raise NBADataSourceError(
                f"Temporadas fuera de cobertura: {sorted(invalid)}. "
                f"La fuente empieza en {FIRST_BOX_SEASON} "
                f"({season_label(FIRST_BOX_SEASON)}). Recordar que la "
                f"temporada se nombra por su año de FIN."
            )

        frames = []
        for season in sorted(set(seasons)):
            frames.append(
                self._load(
                    key     = f"team_box_{season}",
                    loader  = lambda s=season: self._fetch_parquet(
                        _path_team_box(s)
                    ),
                    seasons = [season],
                )
            )

        combined = self._pd.concat(frames, ignore_index=True)
        return self._canonicalize_team_box(combined)

    def load_closing_lines(self):
        """
        Spread y total de CIERRE por partido, consenso de 10-19 casas.

        Fuente estática: no crece más allá de 2025-26. Cubre 2020-21 en
        adelante (`FIRST_CLOSING_SEASON`).

        Columnas de la fuente: game_id, game_spread, home_team_spread,
        home_favorite, over_under, home_point, n_books, commence_time,
        snapshot, match_method, spread_source.

        Convención de signo, verificada en la tarea 12.0:
        `home_team_spread` es POSITIVO cuando el local es favorito. El
        local cubre si `margen_real − home_team_spread > 0`.

        Columna añadida por este método:
            lead_minutes  -- minutos entre el snapshot y el tip-off.
                            Positivo = el snapshot precede al partido.
                            NEGATIVO = snapshot posterior al tip-off:
                            look-ahead, hay que descartarlo.

        NO añade `season`: la temporada se lee del box score, no se
        deriva de la fecha. Ver la nota de convención en el encabezado
        del módulo — derivarla rompe las temporadas de COVID.

        No se filtra nada. Para quedarse solo con cierres válidos:

            low, high = CLOSE_WINDOW_MINUTES
            df = df[(df.lead_minutes >= low) & (df.lead_minutes <= high)]

        o usar `load_settled_games()`, que ya aplica la ventana.

        Retorna
        -------
        pandas.DataFrame con `game_id` como int64, cruzable 1:1 contra
        `load_team_box()`.
        """
        df = self._load(
            key     = "closing_lines",
            loader  = lambda: self._fetch_parquet(_PATH_CLOSING_LINES),
            seasons = [],
            ttl     = _TTL_STATIC,
        )
        return self._annotate_closing_lines(df)

    def load_legacy_odds_archive(self):
        """
        Segunda fuente de cuotas, independiente de la anterior.

        Cubre 2017-18 → 2022-23, así que extiende el histórico tres
        temporadas hacia atrás. Su valor principal es el CONTRASTE: en
        los 2.532 partidos donde se solapa con el consenso, el spread
        coincide dentro de 1 punto en el 95,5% de los casos.

        Peculiaridades del formato, distintas de las del consenso:
            * `line` es la línea del FAVORITO, con signo negativo.
              Para pasarla a la convención de `home_team_spread`:
                  home_team_spread = −line  si el local es favorito
            * no trae `game_id`: el cruce es por fecha + abreviaturas
            * `game_over_under` viene como texto y puede ser nulo
            * no tiene snapshot, así que no hay `lead_minutes`: no se
              puede verificar a qué distancia del tip-off se tomó

        Esa última limitación es la razón de que sea la fuente
        SECUNDARIA. Sirve para contrastar, no como referencia.

        Retorna
        -------
        pandas.DataFrame con `line` y `archive_total` numéricos y
        `game_date` como datetime.
        """
        df = self._load(
            key     = "legacy_odds_archive",
            loader  = lambda: self._fetch_json_records(_PATH_LEGACY_ODDS),
            seasons = [],
            ttl     = _TTL_STATIC,
        )
        return self._canonicalize_legacy_odds(df)

    def load_schedule(self):
        """
        Calendario maestro, 2002 → presente.

        `load_team_box()` ya trae fecha y equipos de los partidos
        JUGADOS, así que para el backtest el calendario es redundante.
        Su uso es en producción: los partidos de HOY todavía no tienen
        box score.

        Retorna
        -------
        pandas.DataFrame con el calendario completo.
        """
        return self._load(
            key     = "schedule_master",
            loader  = lambda: self._fetch_parquet(_PATH_SCHEDULE),
            seasons = [self._current_season],
        )

    # ── Vista curada ─────────────────────────────────────────────────────────

    def load_settled_games(
        self,
        seasons:       list[int],
        season_types:  tuple[int, ...] = (SEASON_TYPE_REGULAR,),
        closing_window: bool = True,
    ):
        """
        Cuotas de cierre ya cruzadas con el resultado real. Una fila
        por partido.

        Por qué este cruce vive aquí y no en cada consumidor
        -----------------------------------------------------
        El cruce tiene dos trampas, y las dos cuestan silencio en vez
        de un error:

          1. **Convención de signo.** `home_team_spread` es positivo
             cuando el local es favorito. Invertirlo produce un
             backtest con el 50,75% de acierto en vez del 49,25% —
             plausible, y completamente falso.
          2. **Ventana de cierre.** 76 snapshots son posteriores al
             tip-off. Incluirlos es look-ahead directo: el mismo
             error que en NFL infló ocho puntos el ROI de totales.

        Tenerlo en un solo sitio es lo que evitó que soccer repitiera
        el bug de `_split_selection`. El backtest y la liquidación
        consumen la misma definición.

        Parámetros
        ----------
        seasons        -- Temporadas por año de FIN. La temporada que
                         se devuelve es la de ESPN, leída del box
                         score, no derivada de la fecha.
        season_types   -- Tipos de partido a conservar. Por defecto
                         solo temporada regular: los playoffs tienen
                         una dinámica distinta (rotaciones cortas,
                         series, ventaja de local más fuerte) y
                         mezclarlos contamina la calibración.
        closing_window -- Si True (defecto), descarta los snapshots
                         fuera de `CLOSE_WINDOW_MINUTES`. Ponerlo en
                         False solo tiene sentido para auditar la
                         propia ventana.

        Columnas añadidas
        -----------------
            margin         -- puntos del local menos puntos del visitante
            game_total     -- suma de puntos
            home_ats_edge  -- margin − home_team_spread.
                             > 0 el local cubre, = 0 push.
            total_edge     -- game_total − over_under.
                             > 0 sale el over, = 0 push.

        Retorna
        -------
        pandas.DataFrame listo para backtest y liquidación.
        """
        closing  = self.load_closing_lines()
        team_box = self.load_team_box(seasons)

        if closing_window:
            low, high = CLOSE_WINDOW_MINUTES
            closing = closing[
                (closing.lead_minutes >= low)
                & (closing.lead_minutes <= high)
            ]

        home = team_box[team_box.team_home_away == "home"][[
            "game_id",
            "game_date",
            "season",
            "season_type",
            "team_id",
            "team_abbreviation",
            "team_display_name",
            "opponent_team_id",
            "opponent_team_abbreviation",
            "team_score",
            "opponent_team_score",
        ]]

        merged = closing.merge(home, on="game_id", how="inner")

        if season_types:
            merged = merged[merged.season_type.isin(season_types)]

        merged = merged.copy()
        merged["margin"]        = (
            merged.team_score - merged.opponent_team_score
        )
        merged["game_total"]    = (
            merged.team_score + merged.opponent_team_score
        )
        merged["home_ats_edge"] = merged.margin - merged.home_team_spread
        merged["total_edge"]    = merged.game_total - merged.over_under

        return merged.sort_values(["game_date", "game_id"]).reset_index(
            drop=True
        )

    # ── Canonicalización ─────────────────────────────────────────────────────
    #
    # Los tipos se fijan en la frontera de E/S, no en cada consumidor.
    # Un `game_id` que llega como str en una fuente y como int64 en otra
    # produce un join vacío SIN ERROR — la clase de bug que en soccer
    # dejó el xG del Athletic Club sin cruzar con sus resultados durante
    # todo un bloque.

    def _canonicalize_team_box(self, df):
        out = df.copy()
        out["game_id"]   = out["game_id"].astype("int64")
        out["game_date"] = self._pd.to_datetime(out["game_date"])
        return out

    def _annotate_closing_lines(self, df):
        out = df.copy()
        out["game_id"] = out["game_id"].astype("int64")

        commence = self._pd.to_datetime(out["commence_time"], utc=True)
        snapshot = self._pd.to_datetime(out["snapshot"], utc=True)

        out["commence_time"] = commence
        out["snapshot"]      = snapshot
        out["lead_minutes"]  = (
            commence - snapshot
        ).dt.total_seconds() / 60.0
        return out

    def _canonicalize_legacy_odds(self, df):
        out = df.copy()
        out["game_date"]     = self._pd.to_datetime(out["game_date"])
        out["line"]          = self._pd.to_numeric(
            out["line"], errors="coerce"
        )
        out["archive_total"] = self._pd.to_numeric(
            out["game_over_under"], errors="coerce"
        )
        return out

    # ── Gestión de caché ─────────────────────────────────────────────────────

    def clear_cache(self, memory_only: bool = False) -> None:
        """Vacía el caché. `memory_only=True` conserva el disco."""
        self._memory.clear()
        if memory_only or not self._use_disk_cache:
            return
        for path in self._cache_dir.glob("*.parquet"):
            try:
                path.unlink()
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
            "memory_keys":    sorted(self._memory.keys()),
            "memory_count":   len(self._memory),
            "disk_files":     len(disk_files),
            "disk_mb":        round(disk_bytes / 1_048_576, 2),
            "cache_dir":      str(self._cache_dir),
            "current_season": self._current_season,
            "season_label":   season_label(self._current_season),
        }

    # ── Motor de caché ───────────────────────────────────────────────────────

    def _load(
        self,
        key:     str,
        loader:  Callable[[], Any],
        seasons: list[int],
        ttl:     int | None = None,
    ):
        """
        Carga un dataset con caché en dos niveles.

        Mismo contrato que `NFLDataSource._load`: memoria → disco →
        descarga, con el TTL derivado de si las temporadas pedidas
        incluyen la que está en curso.
        """
        if key in self._memory:
            return self._memory[key]

        effective_ttl = ttl if ttl is not None else self._ttl_for(seasons)

        if self._use_disk_cache:
            cached = self._read_parquet(key, effective_ttl)
            if cached is not None:
                self._memory[key] = cached
                return cached

        try:
            df = loader()
        except NBADataSourceError:
            raise
        except Exception as e:
            raise NBADataSourceError(
                f"Fallo al descargar '{key}' desde hoopR-nba-data: "
                f"{type(e).__name__}: {e}"
            ) from e

        self._memory[key] = df
        if self._use_disk_cache:
            self._write_parquet(key, df)

        return df

    def _ttl_for(self, seasons: list[int]) -> int:
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
        path = self._parquet_path(key)
        if not path.exists():
            return None
        try:
            age = time.time() - path.stat().st_mtime
            if age > ttl:
                return None
            return self._pd.read_parquet(path)
        except Exception:
            # Parquet corrupto o sin motor disponible — re-descargar.
            return None

    def _write_parquet(self, key: str, df) -> None:
        try:
            df.to_parquet(self._parquet_path(key), index=False)
        except Exception:
            # El caché en disco es opcional — no abortar si falla.
            pass

    # ── Descarga ─────────────────────────────────────────────────────────────

    def _fetch_bytes(self, relative: str) -> bytes:
        url      = f"{_RAW_BASE}/{relative}"
        response = self._requests.get(url, timeout=_HTTP_TIMEOUT)

        if response.status_code == 404:
            raise NBADataSourceError(
                f"La fuente no publica '{relative}' (HTTP 404). Si es un "
                f"box score, puede que esa temporada no exista todavía: "
                f"recordar que el archivo se nombra por el año de FIN de "
                f"la temporada."
            )
        response.raise_for_status()
        return response.content

    def _fetch_parquet(self, relative: str):
        raw = self._fetch_bytes(relative)
        try:
            return self._pd.read_parquet(io.BytesIO(raw))
        except ImportError as e:
            raise NBADataSourceError(
                "No hay motor de parquet disponible. Instalar uno:\n"
                "    pip install fastparquet\n"
                "Se prefiere fastparquet a pyarrow porque ya está "
                "presente en este entorno y no toca el equilibrio "
                "numpy/pandas que dejó la instalación de nfl-data-py."
            ) from e

    def _fetch_json_records(self, relative: str):
        raw = self._fetch_bytes(relative)
        records = json.loads(raw.decode("utf-8"))
        return self._pd.DataFrame(records)

    def __repr__(self) -> str:
        return (
            f"NBADataSource(season={self._current_season} "
            f"[{season_label(self._current_season)}], "
            f"memory={len(self._memory)} datasets, "
            f"disk={'on' if self._use_disk_cache else 'off'})"
        )


# ── Utilidades de módulo ─────────────────────────────────────────────────────

def current_nba_season() -> int:
    """
    Temporada NBA en curso, por año de FIN.

    ÚNICO USO LEGÍTIMO: elegir el TTL del caché. Nada más.

    La temporada cruza el año calendario: 2025-26 va de octubre de 2025
    a junio de 2026, y la fuente la llama 2026. El corte en agosto
    cubre la pretemporada y el hueco del draft sin dejar meses
    ambiguos.

    No usar para etiquetar datos. Este mismo corte aplicado a partidos
    reales rompe las temporadas de COVID —la burbuja de 2019-20 se
    jugó en agosto y octubre de 2020— y el fallo es silencioso. La
    temporada de un partido se lee de `team_box.season`. Para el TTL
    da igual: equivocarse de temporada ahí solo refresca el caché
    antes o después de lo necesario.
    """
    now = datetime.now(timezone.utc)
    return now.year + 1 if now.month >= 8 else now.year


def season_label(season: int) -> str:
    """
    Etiqueta legible desde el año de fin: 2026 → '2025-26'.

    Usar al imprimir cualquier temporada. El año desnudo es ambiguo
    para quien lea un informe, y la ambigüedad es justo la trampa que
    describe el encabezado del módulo.
    """
    return f"{season - 1}-{season % 100:02d}"


def is_available() -> bool:
    """
    True si el plugin NBA puede acceder a sus datos.

    Deriva del estado real de los imports en vez de una bandera
    separada, que podría desincronizarse si alguien modifica el bloque
    try/except sin actualizarla. Mismo patrón que en los módulos MLB y
    NFL tras la auditoría de tipado.
    """
    return _pd is not None and _requests is not None


__all__ = [
    "NBADataSource",
    "NBADataSourceError",
    "CLOSE_WINDOW_MINUTES",
    "FIRST_BOX_SEASON",
    "FIRST_CLOSING_SEASON",
    "SEASON_TYPE_PRESEASON",
    "SEASON_TYPE_REGULAR",
    "SEASON_TYPE_PLAYOFFS",
    "SEASON_TYPE_PLAY_IN",
    "current_nba_season",
    "season_label",
    "is_available",
]