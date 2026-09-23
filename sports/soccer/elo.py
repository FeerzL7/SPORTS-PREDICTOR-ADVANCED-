"""
sports/soccer/elo.py

Ratings Elo de clubelo.com, sin dependencias pesadas.

Por qué añadir Elo a un modelo que ya usa xG
----------------------------------------------
El diagnóstico de calibración señaló el problema exacto: el modelo está
bien calibrado —fiabilidad 0.0008— pero su RESOLUCIÓN es menor que la
del mercado.

    Modelo con xG   0.0212
    Modelo sin xG   0.0119
    Mercado         0.0205 - 0.0291

Resolución mide capacidad de DISCRIMINAR entre partidos. Un modelo
perfectamente calibrado que predijera siempre la media de liga tendría
resolución cero y sería inútil. El nuestro discrimina, pero un 27%
menos que el precio.

Subir la resolución exige información que el modelo no tiene hoy, y Elo
la aporta por tres vías que el xG no cubre:

    CALIDAD DEL RIVAL, automática
        El Elo se actualiza según contra quién se jugó. Nuestros
        índices de xG se comparan contra la media de liga, que no
        distingue haber enfrentado a los tres mejores de haber
        enfrentado a los tres peores.

    HISTORIAL LARGO
        El Elo arrastra años de resultados con decaimiento suave.
        Nuestra ventana es de 8-12 partidos, así que un equipo con mal
        arranque y buena historia queda infravalorado.

    COMPARABILIDAD ENTRE LIGAS
        El Elo sitúa a todos los clubes europeos en la misma escala.
        Nuestros índices son relativos a la media de SU liga, así que
        un 1.20 de la Ligue 1 y un 1.20 de la Premier no significan lo
        mismo.

Son señales parcialmente redundantes —ambas miden fuerza— pero solo
parcialmente. Si aportan información distinta, la combinación
discrimina mejor que cualquiera por separado. Si no, la medición lo
dirá y se descarta.

ALCANCE
---------
ClubElo cubre clubes EUROPEOS. Liga MX y Brasileirão quedan fuera, así
que esto ataca las cinco grandes —donde el modelo ya tenía mejor
resolución— y no las ligas latinoamericanas.

La barrera temporal
---------------------
El endpoint por fecha devuelve los ratings VIGENTES ese día, que
reflejan solo partidos anteriores. Eso lo hace utilizable en un
backtest walk-forward sin look-ahead: pedir 2024-03-15 da el estado del
ranking antes de la jornada de ese día.

Es una propiedad de la API, no un cuidado nuestro, pero conviene
dejarla escrita: si algún día cambiara, el backtest se inflaría sin que
nada fallara.
"""

from __future__ import annotations

import csv
import io
import math
import time
from dataclasses import dataclass
from pathlib import Path

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]


__all__ = ["ClubEloClient", "EloRating", "elo_to_supremacy"]


_BASE_URL = "http://api.clubelo.com"

# Cortesía con un servicio gratuito: los ratings de una fecha pasada no
# cambian nunca, así que se cachean indefinidamente.
_TTL_PAST    = 365 * 24 * 3600
_TTL_CURRENT = 12 * 3600

# ── Conversión Elo → ventaja en goles ────────────────────────────────────────
#
# Un Elo de referencia de 1500 y una escala de 400 puntos son la
# convención del sistema. La conversión a goles usa la relación
# empírica observada en fútbol: alrededor de 100 puntos de diferencia
# equivalen a unos 0.40 goles de ventaja.
#
# No es una constante universal: depende de la liga y del periodo. Se
# expone como parámetro para poder calibrarla con datos, que es lo que
# habría que hacer antes de operar.
_ELO_PER_GOAL = 250.0

# Cota de la ventaja derivada del Elo, en goles.
#
# Las mayores diferencias reales entre clubes europeos rondan los 600
# puntos —un grande de Champions contra un recién ascendido— que dan
# 2.4 goles. La cota deja margen sobre eso sin permitir valores
# absurdos si un rating llega corrupto.
_MAX_SUPREMACY = 3.0


@dataclass(frozen=True)
class EloRating:
    """
    Rating de un club en una fecha.

    Campos
    ------
    club    -- Nombre según ClubElo.
    country -- Código de país de dos letras.
    level   -- Nivel de la división (1 = primera).
    elo     -- Rating. La media de las grandes ligas ronda 1600-1750.
    rank    -- Puesto en el ranking mundial, si lo tiene.
    """
    club:    str
    country: str
    level:   int
    elo:     float
    rank:    int | None = None

    @property
    def is_top_division(self) -> bool:
        return self.level == 1


class ClubEloClient:
    """
    Cliente de clubelo.com.

    Parámetros
    ----------
    cache_dir -- Directorio del caché en disco.
    timeout   -- Timeout HTTP.
    fetch_fn  -- Función de descarga inyectable, para tests.
    """

    def __init__(
        self,
        cache_dir: str | None = "cache/elo",
        timeout:   int = 20,
        fetch_fn         = None,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._timeout = timeout
        self._fetch_fn = fetch_fn

        # Caché en memoria por fecha: una jornada consulta la misma
        # instantánea para los diez partidos.
        self._by_date: dict[str, dict[str, EloRating]] = {}

    @staticmethod
    def is_available() -> bool:
        """True si se puede consultar ClubElo. Solo requiere requests."""
        return _requests is not None

    # ── API pública ───────────────────────────────────────────────────────────

    def ratings_on(self, date: str) -> dict[str, EloRating]:
        """
        Ratings vigentes en una fecha, indexados por nombre normalizado.

        La normalización usa las mismas reglas que teams.py, así que
        las claves son comparables con los nombres canónicos del resto
        del plugin.

        Retorna dict vacío ante cualquier fallo: sin red, fecha
        inválida o formato cambiado. La ausencia de Elo no es un error
        —el modelo funciona sin él— así que no se propaga como
        excepción.
        """
        key = str(date)[:10]
        if key in self._by_date:
            return self._by_date[key]

        raw = self._fetch(key)
        ratings = self._parse(raw) if raw else {}
        self._by_date[key] = ratings
        return ratings

    def rating_for(
        self,
        team:    str,
        date:    str,
        comp_id: str = "",
    ) -> EloRating | None:
        """
        Rating de un equipo en una fecha.

        Cruza por nombre canónico, con la tabla de alias específica de
        ClubElo para los casos que la normalización no resuelve.
        """
        from sports.soccer.teams import canonical_team

        ratings = self.ratings_on(date)
        if not ratings:
            return None

        canon = canonical_team(team, comp_id)
        if not canon:
            return None

        direct = ratings.get(canon)
        if direct is not None:
            return direct

        alias = _CLUBELO_ALIASES.get(canon)
        return ratings.get(alias) if alias else None

    def difference(
        self,
        home:    str,
        away:    str,
        date:    str,
        comp_id: str = "",
    ) -> float | None:
        """
        Diferencia de Elo entre local y visitante.

        Positiva favorece al local. None si falta alguno de los dos:
        una diferencia con un rating ausente sería medio dato, peor que
        ninguno.
        """
        home_rating = self.rating_for(home, date, comp_id)
        away_rating = self.rating_for(away, date, comp_id)
        if home_rating is None or away_rating is None:
            return None
        return round(home_rating.elo - away_rating.elo, 2)

    def coverage(self, teams: list[str], date: str, comp_id: str = "") -> dict:
        """
        Cuántos equipos de una lista tienen rating.

        Diagnóstico para saber si merece la pena usar Elo en una
        competición: ClubElo cubre clubes europeos, así que en Liga MX
        o Brasileirão la cobertura será nula y conviene detectarlo
        antes de proyectar.
        """
        encontrados = [t for t in teams
                       if self.rating_for(t, date, comp_id) is not None]
        return {
            "total":     len(teams),
            "found":     len(encontrados),
            "missing":   sorted(set(teams) - set(encontrados)),
            "coverage":  round(len(encontrados) / len(teams), 4) if teams else 0.0,
        }

    def clear_cache(self) -> None:
        self._by_date.clear()

    # ── Descarga y parseo ─────────────────────────────────────────────────────

    def _fetch(self, date: str) -> str | None:
        """
        Descarga la instantánea de una fecha.

        El endpoint devuelve los ratings VIGENTES ese día, que reflejan
        solo partidos anteriores. Esa propiedad es la que hace el dato
        utilizable en walk-forward.
        """
        if self._fetch_fn is not None:
            try:
                return self._fetch_fn(f"{_BASE_URL}/{date}")
            except Exception:
                return None

        cached = self._read_cache(date)
        if cached is not None:
            return cached

        if _requests is None:
            return None

        try:
            response = _requests.get(f"{_BASE_URL}/{date}",
                                     timeout=self._timeout)
            if response.status_code != 200:
                return None
            text = response.text
        except Exception:
            return None

        self._write_cache(date, text)
        return text

    @staticmethod
    def _parse(raw: str) -> dict[str, EloRating]:
        """
        Convierte el CSV en ratings indexados por nombre canónico.

        Formato de ClubElo:

            Rank,Club,Country,Level,Elo,From,To
            1,Man City,ENG,1,2043.36,2024-11-04,2024-11-10

        El campo Rank viene vacío para equipos fuera del ranking
        principal, y Level indica la división.
        """
        from sports.soccer.teams import normalize_team

        ratings: dict[str, EloRating] = {}
        reader = csv.DictReader(io.StringIO(raw))

        for row in reader:
            club = (row.get("Club") or "").strip()
            elo = _safe_float(row.get("Elo"))
            if not club or elo is None:
                continue

            canon = normalize_team(club)
            if not canon:
                continue

            # Si un nombre aparece dos veces —puede ocurrir con clubes
            # homónimos de países distintos— se queda el de mayor Elo,
            # que en la práctica es el de primera división.
            existing = ratings.get(canon)
            if existing is not None and existing.elo >= elo:
                continue

            ratings[canon] = EloRating(
                club=club,
                country=(row.get("Country") or "").strip(),
                level=_safe_int(row.get("Level")) or 1,
                elo=elo,
                rank=_safe_int(row.get("Rank")),
            )

        return ratings

    # ── Caché ─────────────────────────────────────────────────────────────────

    def _read_cache(self, date: str) -> str | None:
        if self._cache_dir is None:
            return None
        path = self._cache_dir / f"elo_{date}.csv"
        if not path.exists():
            return None

        # Los ratings de una fecha pasada no cambian nunca.
        ttl = _TTL_CURRENT if _is_recent(date) else _TTL_PAST
        try:
            if time.time() - path.stat().st_mtime > ttl:
                return None
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _write_cache(self, date: str, content: str) -> None:
        if self._cache_dir is None:
            return
        try:
            (self._cache_dir / f"elo_{date}.csv").write_text(
                content, encoding="utf-8"
            )
        except Exception:
            pass


# ── Conversión a goles ───────────────────────────────────────────────────────

def elo_to_supremacy(
    elo_diff:     float,
    elo_per_goal: float = _ELO_PER_GOAL,
) -> float:
    """
    Convierte una diferencia de Elo en ventaja esperada de goles.

    La relación es aproximadamente lineal en el rango habitual: unos
    100 puntos de Elo equivalen a 0.40 goles de ventaja, lo que da un
    divisor de 250.

    No es una constante universal —depende de la liga y del periodo— y
    por eso se expone como parámetro. Calibrarla con datos propios es
    lo que habría que hacer antes de operar: el mismo error que
    cometimos con `rho` al tomarlo de la literatura sin ajustarlo.

    El resultado se acota: las mayores diferencias entre clubes
    europeos rondan los 600 puntos, que dan 2.4 goles. Un valor mayor
    indica rating corrupto, no un partido desequilibrado.
    """
    if elo_diff is None:
        return 0.0
    try:
        supremacy = float(elo_diff) / max(elo_per_goal, 1.0)
    except (ValueError, TypeError):
        return 0.0
    if supremacy != supremacy:   # NaN
        return 0.0
    return max(-_MAX_SUPREMACY, min(_MAX_SUPREMACY, supremacy))


def elo_win_probability(elo_diff: float, home_advantage: float = 65.0) -> float:
    """
    Probabilidad de victoria local según la fórmula estándar de Elo.

    No la usa el modelo de proyección —que necesita goles, no
    resultados— pero sirve como comprobación independiente: si la
    proyección de goles implica una probabilidad muy distinta de esta,
    hay algo mal en la conversión.

    `home_advantage` en puntos de Elo. ClubElo usa ~65, equivalente a
    unos 0.26 goles.
    """
    try:
        diff = float(elo_diff) + home_advantage
    except (ValueError, TypeError):
        return 0.5
    return 1.0 / (1.0 + math.pow(10.0, -diff / 400.0))


# ── Alias específicos de ClubElo ─────────────────────────────────────────────
#
# Casos donde la normalización no basta porque ClubElo usa un nombre
# genuinamente distinto del de football-data o Understat.
#
# Clave: forma canónica del plugin. Valor: forma normalizada de
# ClubElo.

_CLUBELO_ALIASES: dict[str, str] = {
    # Inglaterra
    "manchester city":        "man city",
    "manchester united":      "man united",
    "tottenham":              "tottenham",
    "wolverhampton wanderers": "wolves",
    "newcastle united":       "newcastle",
    "west bromwich albion":   "west brom",
    "nottingham forest":      "forest",
    "sheffield united":       "sheffield united",
    "brighton":               "brighton",
    # España
    "atletico madrid":        "atletico",
    "athletic":               "bilbao",
    "real sociedad":          "sociedad",
    "real betis":             "betis",
    "celta vigo":             "celta",
    "rayo vallecano":         "rayo",
    "real valladolid":        "valladolid",
    # Italia
    "ac milan":               "milan",
    "milan":                  "milan",
    "inter":                  "inter",
    "verona":                 "verona",
    # Alemania
    "borussia m gladbach":    "gladbach",
    "borussia dortmund":      "dortmund",
    "bayer leverkusen":       "leverkusen",
    "eintracht frankfurt":    "frankfurt",
    "rasenballsport leipzig": "rb leipzig",
    "cologne":                "koln",
    "hertha berlin":          "hertha",
    "mainz 05":               "mainz",
    "hamburger":              "hamburg",
    # Francia
    "paris saint germain":    "paris sg",
    "saint etienne":          "st etienne",
    "olympique marseille":    "marseille",
    "olympique lyonnais":     "lyon",
}


# ── Utilidades ───────────────────────────────────────────────────────────────

def _is_recent(date: str, days: int = 14) -> bool:
    """True si la fecha está dentro de los últimos `days` días."""
    from datetime import datetime, timezone
    try:
        parsed = datetime.strptime(date[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc
        )
    except (ValueError, TypeError):
        return True
    return (datetime.now(timezone.utc) - parsed).days <= days


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        result = float(str(value).strip())
    except (ValueError, TypeError):
        return None
    return result if result == result else None


def _safe_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).strip()))
    except (ValueError, TypeError):
        return None