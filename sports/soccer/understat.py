"""
sports/soccer/understat.py

Backend de xG desde Understat, sin dependencias pesadas.

Por qué no se usa la librería `soccerdata`
-------------------------------------------
La primera versión de este plugin dependía de `soccerdata`, que envuelve
Understat y otras fuentes. Instalarla en el entorno de producción rompió
el plugin NFL:

    nfl_data_py  exige  pandas < 2.0
    soccerdata   exige  pandas >= 2.0

El conflicto es irreconciliable: no pueden convivir en el mismo entorno.
Y `soccerdata` arrastraba además unos sesenta paquetes transitivos,
entre ellos Selenium y un navegador headless completo — para leer datos
que están en texto plano dentro del HTML.

Este módulo los extrae directamente. Necesita `requests`, que el Core ya
usa para The Odds API, más `json` y `re` de la stdlib. Nada más.

Cómo publica Understat sus datos
----------------------------------
CAMBIO (verificado 2026-10): la página de liga YA NO incrusta los datos.
Los carga después con una petición XHR a

    https://understat.com/getLeagueData/{liga}/{temporada}

que devuelve JSON con tres bloques: `teams` (historial partido a
partido por equipo, el mismo que antes traía `teamsData`), `players` y
`dates` (partidos con xG y el pronóstico 1X2 del propio Understat).

Hasta 2026 los datos iban dentro del HTML, en una variable JavaScript
con el JSON escapado en hexadecimal:

    var teamsData = JSON.parse('\\x7B\\x22...\\x7D');

Con ese cambio el cliente devolvía cero partidos SIN ERROR y el modelo
corría sin xG —su entrada principal— sin que nada lo advirtiera. Ahora
se pide el endpoint JSON y, si la respuesta fuera HTML, se intenta el
formato antiguo. Sigue siendo una fuente no oficial: todos los métodos
degradan a vacío en vez de lanzar.

Datos por partido, no solo agregados
--------------------------------------
`teamsData` trae el historial partido a partido de cada equipo, no un
resumen de temporada. Eso permite calcular ventanas móviles y separar
rendimiento en casa y fuera, que en fútbol importa mucho más que en
otros deportes: la ventaja de campo vale ~0.35 goles, y algunos equipos
la explotan bastante más que otros.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]


_BASE_URL = "https://understat.com/league"
_DATA_URL = "https://understat.com/getLeagueData"

# Navegador declarado en las peticiones.
#
# Understat es un sitio pequeño mantenido por una sola persona. Un
# User-Agent identificable y el caché agresivo de data_source.py son
# cortesía básica: sin ellos, un backtest de diez temporadas dispararía
# cincuenta peticiones cada vez que se ejecuta.
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; SportsPredictor/1.0; "
        "análisis estadístico)"
    ),
    # El endpoint solo responde JSON a peticiones marcadas como XHR,
    # que es como lo llama la propia página.
    "X-Requested-With": "XMLHttpRequest",
}

# Caché en disco: una temporada cerrada no cambia; la actual se
# revalida dos veces al día.
_TTL_CURRENT_SECONDS = 12 * 3600

# Variables JavaScript que contienen los datos.
_VAR_TEAMS   = "teamsData"
_VAR_PLAYERS = "playersData"
_VAR_DATES   = "datesData"


@dataclass(frozen=True)
class MatchXG:
    """
    xG de un equipo en un partido concreto.

    Campos
    ------
    team      -- Nombre del equipo según Understat.
    date      -- Fecha del partido, ISO.
    is_home   -- True si jugó en casa.
    opponent  -- Rival, cuando se puede determinar.
    xg / xga  -- Goles esperados a favor y en contra.
    npxg      -- xG sin penaltis. Más predictivo que el xG total:
                 un penalti vale ~0.76 de xG y su concesión depende
                 mucho más del azar que el resto del juego.
    npxga     -- npxG en contra.
    goals     -- Goles marcados.
    conceded  -- Goles encajados.
    result    -- 'w', 'd' o 'l'.
    """
    team:     str
    date:     str
    is_home:  bool
    opponent: str = ""
    xg:       float = 0.0
    xga:      float = 0.0
    npxg:     float = 0.0
    npxga:    float = 0.0
    goals:    int = 0
    conceded: int = 0
    result:   str = ""

    @property
    def xg_diff(self) -> float:
        """Diferencia de xG. El mejor indicador de dominio del partido."""
        return round(self.xg - self.xga, 3)

    @property
    def overperformance(self) -> float:
        """
        Goles marcados menos xG.

        Positivo indica finalización por encima de lo esperado, que
        históricamente revierte. El mercado reacciona a los goles antes
        que al xG, así que esta diferencia señala dónde puede haber
        valor.
        """
        return round(self.goals - self.xg, 3)


class UnderstatClient:
    """
    Cliente de Understat sin dependencias pesadas.

    Parámetros
    ----------
    timeout   -- Timeout HTTP en segundos.
    fetch_fn  -- Función de descarga inyectable, para tests. Recibe la
                 URL y devuelve el texto (JSON o HTML) o None. Sin ella
                 usa requests.
    cache_dir -- Directorio para cachear las respuestas. None: sin caché.
    current_season -- Temporada en curso; las anteriores no caducan.
    """

    def __init__(
        self,
        timeout:  int = 20,
        fetch_fn = None,
        cache_dir = None,
        current_season: int | None = None,
    ) -> None:
        self._timeout  = timeout
        self._fetch_fn = fetch_fn
        self._cache_dir = cache_dir
        self._current_season = current_season

    @staticmethod
    def is_available() -> bool:
        """
        True si se puede consultar Understat.

        Solo requiere `requests`, que ya es dependencia del Core. A
        diferencia de la versión con soccerdata, no hace falta instalar
        nada adicional ni existe riesgo de romper otros plugins por
        conflicto de versiones.
        """
        return _requests is not None

    # ── API pública ───────────────────────────────────────────────────────────

    def fetch_league_matches(
        self,
        understat_id: str,
        season:       int,
    ) -> list[MatchXG]:
        """
        Historial partido a partido de todos los equipos de una liga.

        Retorna lista vacía ante cualquier fallo: sin red, estructura
        cambiada o liga inexistente. La ausencia de xG NO es un error
        —el plugin funciona en tier PARTIAL— así que no se propaga
        como excepción.
        """
        text = self._fetch(understat_id, season)
        if not text:
            return []

        teams = self._teams_block(text)
        if not isinstance(teams, dict):
            return []

        return self._parse_teams(teams)

    @classmethod
    def _teams_block(cls, text: str):
        """
        Bloque de equipos, venga del endpoint JSON o del HTML antiguo.
        """
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            data = None
        if isinstance(data, dict) and isinstance(data.get("teams"), dict):
            return data["teams"]
        return cls._extract_json(text, _VAR_TEAMS)

    def fetch_team_totals(
        self,
        understat_id: str,
        season:       int,
    ) -> dict[str, dict]:
        """
        Agregados de temporada por equipo.

        Método de conveniencia: suma el historial partido a partido.
        Se mantiene separado porque el consumidor decide si quiere el
        detalle o el resumen, y agregar aquí evita que cada llamador
        reimplemente la misma suma.
        """
        matches = self.fetch_league_matches(understat_id, season)
        totals: dict[str, dict] = {}

        for match in matches:
            bucket = totals.setdefault(match.team, {
                "matches": 0, "xg_for": 0.0, "xg_against": 0.0,
                "npxg_for": 0.0, "npxg_against": 0.0,
                "goals_for": 0, "goals_against": 0,
            })
            bucket["matches"]       += 1
            bucket["xg_for"]        += match.xg
            bucket["xg_against"]    += match.xga
            bucket["npxg_for"]      += match.npxg
            bucket["npxg_against"]  += match.npxga
            bucket["goals_for"]     += match.goals
            bucket["goals_against"] += match.conceded

        for bucket in totals.values():
            for key in ("xg_for", "xg_against", "npxg_for", "npxg_against"):
                bucket[key] = round(bucket[key], 3)

        return totals

    # ── Descarga ──────────────────────────────────────────────────────────────

    def _fetch(self, understat_id: str, season: int) -> str | None:
        """Descarga los datos de la liga (endpoint JSON), con caché."""
        url = f"{_DATA_URL}/{understat_id}/{season}"

        if self._fetch_fn is not None:
            try:
                return self._fetch_fn(url)
            except Exception:
                return None

        cached = self._read_cache(understat_id, season)
        if cached is not None:
            return cached

        if _requests is None:
            return None

        try:
            response = _requests.get(url, headers=_HEADERS, timeout=self._timeout)
            if response.status_code != 200:
                return None
            text = response.text
        except Exception:
            return None

        # Solo se cachea una respuesta que trae datos: cachear un error o
        # una página vacía congelaría la ausencia de xG hasta que caduque.
        if self._teams_block(text):
            self._write_cache(understat_id, season, text)
        return text

    def _cache_path(self, understat_id: str, season: int):
        if not self._cache_dir:
            return None
        from pathlib import Path
        return Path(self._cache_dir) / f"understat_{understat_id}_{season}.json"

    def _read_cache(self, understat_id: str, season: int) -> str | None:
        import time
        path = self._cache_path(understat_id, season)
        if path is None or not path.exists():
            return None
        closed = self._current_season is not None and season < self._current_season
        if not closed and time.time() - path.stat().st_mtime > _TTL_CURRENT_SECONDS:
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _write_cache(self, understat_id: str, season: int, text: str) -> None:
        path = self._cache_path(understat_id, season)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError:
            pass   # el caché es una optimización

    # ── Extracción ────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(html: str, var_name: str):
        """
        Extrae y decodifica una variable JavaScript de la página.

        Understat escapa el JSON en hexadecimal dentro de una cadena
        JavaScript:

            var teamsData = JSON.parse('\\x7B\\x22id\\x22...');

        Deshacer el escapado con `unicode_escape` funciona porque las
        secuencias \\xNN son sintaxis válida tanto en JavaScript como en
        Python. El paso por latin-1 es necesario: `unicode_escape`
        opera sobre bytes, y codificar en UTF-8 corrompería los
        caracteres no ASCII de los nombres de equipo antes de
        decodificarlos.
        """
        pattern = rf"var\s+{var_name}\s*=\s*JSON\.parse\('(.*?)'\)"
        match = re.search(pattern, html, re.DOTALL)
        if not match:
            return None

        try:
            decoded = match.group(1).encode("latin-1").decode("unicode_escape")
            return json.loads(decoded)
        except Exception:
            return None

    @staticmethod
    def _parse_teams(teams: dict) -> list[MatchXG]:
        """
        Convierte el bloque `teamsData` en filas tipadas.

        Estructura de Understat:

            {
              "82": {
                "id": "82",
                "title": "Manchester City",
                "history": [
                  {"h_a": "h", "xG": 2.31, "xGA": 0.64, "npxG": 1.55,
                   "npxGA": 0.64, "scored": 3, "missed": 0,
                   "date": "2024-08-18 16:00:00", "result": "w", ...},
                  ...
                ]
              },
              ...
            }

        Los campos numéricos llegan a veces como cadena y a veces como
        número según la temporada, así que todos pasan por conversión
        segura en vez de asumir el tipo.
        """
        rows: list[MatchXG] = []

        for team_block in teams.values():
            if not isinstance(team_block, dict):
                continue

            name = _clean(team_block.get("title"))
            history = team_block.get("history")
            if not name or not isinstance(history, list):
                continue

            for entry in history:
                if not isinstance(entry, dict):
                    continue

                date = _parse_date(entry.get("date"))
                if not date:
                    continue

                rows.append(MatchXG(
                    team     = name,
                    date     = date,
                    is_home  = str(entry.get("h_a", "")).lower() == "h",
                    xg       = _safe_float(entry.get("xG")) or 0.0,
                    xga      = _safe_float(entry.get("xGA")) or 0.0,
                    npxg     = _safe_float(entry.get("npxG")) or 0.0,
                    npxga    = _safe_float(entry.get("npxGA")) or 0.0,
                    goals    = _safe_int(entry.get("scored")) or 0,
                    conceded = _safe_int(entry.get("missed")) or 0,
                    result   = _clean(entry.get("result")).lower(),
                ))

        return rows


# ── Utilidades ───────────────────────────────────────────────────────────────

def _parse_date(raw) -> str:
    """
    Normaliza la fecha de Understat a ISO.

    Llega como '2024-08-18 16:00:00'; solo interesa la parte de fecha,
    que ya está en orden ISO.
    """
    text = _clean(raw)
    if len(text) < 10:
        return ""
    candidate = text[:10]
    # Validación mínima de forma: YYYY-MM-DD
    if candidate[4] == "-" and candidate[7] == "-":
        return candidate
    return ""


def _clean(value) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("", "nan", "none", "null") else text


def _safe_float(value) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        result = float(text)
    except (ValueError, TypeError):
        return None
    return result if result == result else None


def _safe_int(value) -> int | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return None