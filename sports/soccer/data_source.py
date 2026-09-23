"""
sports/soccer/data_source.py

SoccerDataSource: acceso a datos de fútbol con dos backends.

Dos fuentes, dos propósitos
-----------------------------
    football-data.co.uk   Resultados históricos CON cuotas de cierre,
                          desde 1993, para las cinco grandes europeas.
                          CSV directo, sin scraping ni autenticación.

    Understat             xG por equipo y partido. Su cobertura es
                          exactamente la de las cinco grandes, que es
                          por lo que el alcance del plugin arranca ahí.

Las cuotas de cierre de football-data son lo que hace posible el
backtest, igual que las líneas de nflverse lo hicieron para NFL. Y son
mejores: incluye las de Pinnacle, que es el book más afilado del
mercado y por tanto el benchmark más exigente.

Ninguna de las dos fuentes necesita pandas
--------------------------------------------
La primera versión obtenía el xG con la librería `soccerdata`.
Instalarla rompió el plugin NFL en producción:

    nfl_data_py  exige  pandas < 2.0
    soccerdata   exige  pandas >= 2.0

Irreconciliable. Y `soccerdata` arrastraba unos sesenta paquetes
transitivos, Selenium incluido, para leer datos que están en texto
plano dentro del HTML.

Ahora el xG lo extrae sports/soccer/understat.py con `requests`, `json`
y `re`. Este módulo no importa pandas en absoluto, así que el plugin de
fútbol no puede romper al de NFL ni al revés.

Lección aplicada de NFL: sin DataFrames fuera de aquí
-------------------------------------------------------
En el plugin NFL, nfl_data_py devuelve DataFrames y schedule.py tenía
que convertirlos. De ahí salieron dos clases de problemas que costó
detectar:

    El boolean masking de pandas, que el type checker no modela y que
    hubo que erradicar módulo a módulo.

    Campos que se perdían en silencio — las líneas de mercado y el
    clima observado nunca llegaban a NFLGameInfo, y `getattr` con
    default devolvía None sin error.

Aquí eso no puede pasar: football-data sirve CSV plano, que el módulo
`csv` de la stdlib lee sin pandas. Esta fuente devuelve OBJETOS
TIPADOS, no DataFrames. El único punto donde aparece un DataFrame es la
conversión de Understat, y no sale de este módulo.

Tier efectivo frente a tier declarado
---------------------------------------
competitions.py declara las cinco grandes como TIER_FULL, que asume xG
disponible. Pero xG llega vía `soccerdata`, que es opcional.

Si no está instalado, esas competiciones son en la práctica PARTIAL: el
modelo funciona con goles y forma, notablemente menos predictivo.
`effective_tier()` reporta lo que REALMENTE hay, no lo que el catálogo
declara. Un sistema que afirmara tener xG sin tenerlo produciría
proyecciones confiadas sobre datos que no existen.

Caché en dos niveles
----------------------
    Memoria   Durante la ejecución. Una jornada de Premier consulta la
              misma temporada para los 10 partidos.

    Disco     Entre ejecuciones, en CSV. Las temporadas cerradas no
              cambian nunca, así que se cachean indefinidamente; la
              temporada en curso caduca en un día.

Sin la caché de disco, backtestear 10 temporadas × 5 ligas descargaría
50 ficheros en cada ejecución.
"""

from __future__ import annotations

import csv
import io
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sports.soccer.competitions import (
    Competition, TIER_FULL, TIER_PARTIAL, TIER_MINIMAL, get_competition,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Solo para la anotación de _understat. El import real ocurre
    # dentro de has_xg_backend(), de forma diferida, para que cargar
    # este módulo no arrastre el cliente de scraping cuando no hace
    # falta.
    from sports.soccer.understat import MatchXG, UnderstatClient

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]


class SoccerDataSourceError(RuntimeError):
    """Fallo irrecuperable al obtener datos de fútbol."""


_FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"

# Sección de ligas no europeas de football-data.
#
# Formato distinto al principal, y la diferencia importa:
#
#     Principal : un fichero por LIGA y TEMPORADA (E0.csv de 2425)
#                 columnas HomeTeam, FTHG, PSCH...
#
#     Extendida : un fichero por PAÍS con TODAS las temporadas
#                 (MEX.csv) y columnas Home, HG, PH, AvgCH...
#
# Tratarlas con el mismo parser produciría filas vacías en silencio,
# que es el modo de fallo que este plugin lleva corrigiendo desde el
# principio. Cada formato tiene su extractor.
_FOOTBALL_DATA_EXTRA = "https://www.football-data.co.uk/new"

# TTL del caché en disco, en segundos.
# Una temporada cerrada no cambia nunca; la actual se actualiza tras
# cada jornada, así que un día es suficiente.
_TTL_CLOSED_SEASON  = 365 * 24 * 3600
_TTL_CURRENT_SEASON = 24 * 3600


# ── Filas tipadas ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MatchRow:
    """
    Un partido con su resultado y las cuotas de cierre.

    Campos de resultado
    -------------------
    home_goals / away_goals  -- Goles al final del tiempo reglamentario.
                                None si el partido no se ha jugado.
    ht_home / ht_away        -- Goles al descanso. Permiten liquidar
                                mercados de primera parte, que en
                                fútbol sí son liquidables (a diferencia
                                de los H1 de NFL, donde no teníamos el
                                marcador parcial).

    Cuotas de cierre
    ----------------
    Se prefieren las de Pinnacle sobre las de Bet365 cuando ambas
    están: Pinnacle opera con margen mínimo y límites altos, así que su
    cierre es el mejor estimador del precio justo. Batirlo es el
    listón más exigente que existe.

    odds_home / draw / away  -- 1X2 de cierre.
    odds_over / under        -- Over/Under 2.5 goles de cierre.
    ah_line                  -- Hándicap asiático de cierre, desde la
                                perspectiva del local.
    ah_home / ah_away        -- Cuotas del hándicap asiático.
    odds_source              -- 'pinnacle' o 'bet365', para trazabilidad.
    """
    comp_id:    str
    season:     int
    date:       str
    home_team:  str
    away_team:  str

    home_goals: int | None = None
    away_goals: int | None = None
    ht_home:    int | None = None
    ht_away:    int | None = None

    # ── Cuotas de CIERRE ──────────────────────────────────────
    odds_home:  float | None = None
    odds_draw:  float | None = None
    odds_away:  float | None = None
    odds_over:  float | None = None
    odds_under: float | None = None
    ah_line:    float | None = None
    ah_home:    float | None = None
    ah_away:    float | None = None
    odds_source: str = ""

    # ── Cuotas de APERTURA ────────────────────────────────────
    #
    # football-data publica ambas. La diferencia entre ellas ES el
    # margen accesible: el cierre incorpora todo el dinero profesional
    # y toda la información pública, mientras que la apertura es el
    # precio inicial del book.
    #
    # El pipeline en producción opera con precios de apertura y media
    # semana, no de cierre, así que medir contra la apertura responde
    # una pregunta distinta y más relevante para decidir si el sistema
    # es operable.
    #
    # CORRECCIÓN: la versión anterior mezclaba ambos conjuntos sin
    # saberlo. Para Pinnacle tomaba el 1X2 de cierre (PSCH) pero el
    # total de apertura (P>2.5), porque esa columna no lleva la 'C'.
    # Los picks de total se estaban midiendo contra un benchmark más
    # blando que los de 1X2.
    open_home:  float | None = None
    open_draw:  float | None = None
    open_away:  float | None = None
    open_over:  float | None = None
    open_under: float | None = None
    open_source: str = ""

    @property
    def is_final(self) -> bool:
        """True si el partido tiene marcador registrado."""
        return self.home_goals is not None and self.away_goals is not None

    @property
    def total_goals(self) -> int | None:
        if not self.is_final:
            return None
        return (self.home_goals or 0) + (self.away_goals or 0)

    @property
    def result(self) -> str | None:
        """'H', 'D' o 'A' — local, empate o visitante."""
        if not self.is_final:
            return None
        h, a = self.home_goals or 0, self.away_goals or 0
        return "H" if h > a else ("A" if a > h else "D")

    @property
    def has_opening_odds(self) -> bool:
        """True si el 1X2 de apertura está completo."""
        return all(o is not None for o in
                   (self.open_home, self.open_draw, self.open_away))

    @property
    def closing_line_value(self) -> float | None:
        """
        Movimiento del 1X2 local, de apertura a cierre, en porcentaje.

        Positivo significa que la cuota SUBIÓ: el mercado se movió en
        contra del local. Es la magnitud que el CLV mide en producción,
        y tenerla aquí permite comprobar si el modelo anticipa el
        movimiento aunque no gane dinero contra el cierre.
        """
        if self.open_home is None or self.odds_home is None:
            return None
        if self.open_home <= 0:
            return None
        return round((self.odds_home / self.open_home - 1.0) * 100.0, 3)

    @property
    def has_closing_odds(self) -> bool:
        """
        True si hay cuotas 1X2 de cierre completas.

        Se exigen las tres: un 1X2 incompleto no permite calcular la
        probabilidad implícita sin vig, que es lo que el blending
        necesita.
        """
        return all(o is not None for o in
                   (self.odds_home, self.odds_draw, self.odds_away))

    @property
    def match_key(self) -> str:
        """Clave estable del partido, para cruzar con otras fuentes."""
        return f"{self.comp_id}_{self.date}_{self.home_team}_{self.away_team}"


@dataclass(frozen=True)
class TeamXG:
    """
    Métricas de xG acumuladas de un equipo en una temporada.

    Sobre npxG
    ----------
    npxG (non-penalty xG) excluye los penaltis. Importa porque un
    penalti vale ~0.76 de xG y su concesión depende mucho más del azar
    que el resto del juego: un equipo con muchos penaltis a favor tiene
    el xG inflado por un componente poco repetible.

    Para proyectar, npxG es más predictivo que xG total. Se conservan
    ambos: la diferencia entre ellos es informativa por sí misma.
    """
    comp_id:   str
    season:    int
    team:      str
    matches:   int = 0

    xg_for:      float = 0.0
    xg_against:  float = 0.0
    npxg_for:    float = 0.0
    npxg_against: float = 0.0
    goals_for:   int = 0
    goals_against: int = 0

    @property
    def xg_per_match(self) -> float:
        return self.xg_for / self.matches if self.matches else 0.0

    @property
    def xga_per_match(self) -> float:
        return self.xg_against / self.matches if self.matches else 0.0

    @property
    def npxg_per_match(self) -> float:
        return self.npxg_for / self.matches if self.matches else 0.0

    @property
    def npxga_per_match(self) -> float:
        return self.npxg_against / self.matches if self.matches else 0.0

    @property
    def xg_overperformance(self) -> float:
        """
        Goles marcados menos xG acumulado.

        Un valor muy positivo indica finalización por encima de lo
        esperado, que históricamente revierte: es el indicador clásico
        de que un equipo va a bajar su ritmo de goles. Un valor muy
        negativo sugiere lo contrario.

        El mercado suele reaccionar a los goles antes que al xG, así
        que esta diferencia es donde puede haber valor.
        """
        return round(self.goals_for - self.xg_for, 3)


# ── Fuente ───────────────────────────────────────────────────────────────────

class SoccerDataSource:
    """
    Acceso unificado a los datos de fútbol.

    Parámetros
    ----------
    cache_dir      -- Directorio del caché en disco. None lo desactiva,
                      lo que solo tiene sentido en tests.
    current_season -- Temporada en curso, para decidir el TTL. None la
                      deduce de la fecha.
    timeout        -- Timeout HTTP en segundos.
    """

    def __init__(
        self,
        cache_dir:      str | None = "cache/soccer",
        current_season: int | None = None,
        timeout:        int = 20,
    ) -> None:
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir:
            self._cache_dir.mkdir(parents=True, exist_ok=True)

        self._current_season = current_season or current_soccer_season()
        self._timeout = timeout

        # Caché en memoria
        self._matches: dict[tuple[str, int], list[MatchRow]] = {}
        self._xg: dict[tuple[str, int], list[TeamXG]] = {}
        self._match_xg: dict[tuple[str, int], list["MatchXG"]] = {}

        # Backend de xG, resuelto perezosamente.
        #
        # Se anota explícitamente en vez de dejar que el checker lo
        # infiera de las asignaciones: sin la anotación quedaría como
        # `UnderstatClient | None` deducido, y cada punto de uso
        # arrastraría esa indeterminación. Es el mismo criterio que
        # aplicamos a los componentes de NFLPlugin.
        self._understat: UnderstatClient | None = None
        self._understat_checked = False

    # ── Disponibilidad ────────────────────────────────────────────────────────

    @staticmethod
    def is_available() -> bool:
        """
        True si se puede obtener al menos resultados y cuotas.

        Solo requiere `requests`, que ya es dependencia del Core para
        The Odds API. El plugin de fútbol, a diferencia del de NFL, no
        necesita instalar nada para funcionar en tier PARTIAL.
        """
        return _requests is not None

    def has_xg_backend(self) -> bool:
        """
        True si el backend de xG está disponible.

        Tras sustituir `soccerdata` por el cliente propio, esto se
        reduce a comprobar `requests` — que el Core ya necesita para
        The Odds API. En la práctica, si el sistema funciona el xG
        está disponible.

        El método se conserva porque effective_tier() lo consulta y
        porque un fallo de red o un cambio en la estructura de
        Understat siguen dejando el plugin en tier PARTIAL. La
        distinción sigue siendo real; lo que cambió es que ya no
        depende de instalar nada.
        """
        if not self._understat_checked:
            self._understat_checked = True
            from sports.soccer.understat import UnderstatClient
            self._understat = (
                UnderstatClient(timeout=self._timeout)
                if UnderstatClient.is_available() else None
            )
        return self._understat is not None

    def effective_tier(self, comp: Competition | str) -> str:
        """
        Tier REAL de una competición, dado lo que está instalado.

        competitions.py declara las cinco grandes como TIER_FULL porque
        Understat las cubre. Pero si `soccerdata` no está instalado, no
        hay xG y en la práctica son PARTIAL.

        Reportar el tier declarado en ese caso haría que el provider
        asignara data_quality=1.0 a datos que no tienen la señal
        principal, y el modelo proyectaría con una confianza que no
        corresponde. El sistema debe declarar lo que tiene, no lo que
        podría tener.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None:
            return TIER_MINIMAL

        if not self.is_available():
            return TIER_MINIMAL

        if competition.tier == TIER_FULL and not self.has_xg_backend():
            return TIER_PARTIAL

        return competition.tier

    # ── Resultados y cuotas ───────────────────────────────────────────────────

    def load_matches(
        self,
        comp:   Competition | str,
        season: int,
    ) -> list[MatchRow]:
        """
        Partidos de una competición y temporada, con cuotas de cierre.

        Retorna lista vacía ante cualquier fallo: la fuente no está
        disponible, la competición no tiene código de football-data, o
        la descarga falla. Nunca lanza — el pipeline decide qué hacer
        con la ausencia de datos.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None or not competition.football_data_code:
            return []

        key = (competition.comp_id, season)
        if key in self._matches:
            return self._matches[key]

        raw = self._fetch_football_data(competition, season)
        if not raw:
            rows = []
        elif (competition.football_data_code or "").startswith("new:"):
            rows = self._parse_extra(raw, competition, season)
        else:
            rows = self._parse_football_data(raw, competition, season)

        self._matches[key] = rows
        return rows

    def _fetch_football_data(
        self,
        comp:   Competition,
        season: int,
    ) -> str | None:
        """
        Descarga el CSV de football-data.co.uk, con caché en disco.

        La URL sigue un patrón estable desde hace más de dos décadas:
            /mmz4281/{temporada}/{liga}.csv
        donde temporada es '2425' para 2024/25 y liga es 'E0' para
        Premier League.
        """
        code = comp.football_data_code
        if not code:
            return None

        # El código con prefijo 'new:' indica la sección extendida, que
        # sirve un fichero por PAÍS con todas las temporadas en vez de
        # uno por liga y temporada.
        if code.startswith("new:"):
            return self._fetch_extra(comp, season, code[4:])

        season_code = comp.football_data_season(season)
        cache_name = f"fd_{comp.comp_id}_{season}.csv"

        cached = self._read_cache(cache_name, season)
        if cached is not None:
            return cached

        if _requests is None:
            return None

        url = f"{_FOOTBALL_DATA_BASE}/{season_code}/{code}.csv"
        try:
            response = _requests.get(url, timeout=self._timeout)
            if response.status_code != 200:
                return None
            # Los CSV vienen en latin-1, no UTF-8: llevan nombres de
            # equipo con acentos desde los años noventa.
            text = response.content.decode("latin-1", errors="replace")
        except Exception:
            return None

        self._write_cache(cache_name, text)
        return text

    def _fetch_extra(
        self,
        comp:    Competition,
        season:  int,
        country: str,
    ) -> str | None:
        """
        Descarga el fichero de país de la sección extendida.

        Un solo fichero cubre todas las temporadas, así que se cachea
        por PAÍS y no por temporada. Un backtest de cinco temporadas de
        Liga MX hace una descarga, no cinco.

        El TTL usa la temporada actual porque el fichero se actualiza
        mientras haya alguna en curso.
        """
        cache_name = f"fdx_{country}.csv"

        cached = self._read_cache(cache_name, self._current_season)
        if cached is not None:
            return cached

        if _requests is None:
            return None

        url = f"{_FOOTBALL_DATA_EXTRA}/{country}.csv"
        try:
            response = _requests.get(url, timeout=self._timeout)
            if response.status_code != 200:
                return None
            text = response.content.decode("latin-1", errors="replace")
        except Exception:
            return None

        self._write_cache(cache_name, text)
        return text

    def _parse_extra(
        self,
        raw:    str,
        comp:   Competition,
        season: int,
    ) -> list[MatchRow]:
        """
        Convierte el formato extendido en MatchRow.

        Tres diferencias con el principal, todas con consecuencias:

        COLUMNAS         Home/Away en vez de HomeTeam/AwayTeam,
                         HG/AG en vez de FTHG/FTAG.

        TODAS LAS
        TEMPORADAS       El fichero las trae juntas, así que hay que
                         filtrar por la columna Season.

        FORMATO DE
        TEMPORADA        Varía según el calendario de la liga:

                             Brasileirão   '2023'      (año natural)
                             Liga MX       '2023/2024' (cruza el año)

                         Se aceptan ambos y se comparan contra el año
                         de inicio, que es la convención del plugin.

        FECHA            'dd/mm/yyyy', igual que el principal.
        """
        rows: list[MatchRow] = []
        target = str(season)
        reader = csv.DictReader(io.StringIO(raw))

        for record in reader:
            if not _season_matches(record.get("Season"), target):
                continue

            home = _clean(record.get("Home"))
            away = _clean(record.get("Away"))
            date = _parse_date(record.get("Date"))
            if not home or not away or not date:
                continue

            rows.append(MatchRow(
                comp_id    = comp.comp_id,
                season     = season,
                date       = date,
                home_team  = home,
                away_team  = away,
                home_goals = _safe_int(record.get("HG")),
                away_goals = _safe_int(record.get("AG")),
                **_extract_odds_extra(record),
            ))

        return rows

    def _parse_football_data(
        self,
        raw:    str,
        comp:   Competition,
        season: int,
    ) -> list[MatchRow]:
        """
        Convierte el CSV en MatchRow tipados.

        Sin pandas: el módulo csv de la stdlib basta y evita arrastrar
        la dependencia junto con su frontera de tipado.

        Sobre las columnas de cuotas
        ----------------------------
        football-data expone varias casas. Las columnas con 'C' tras el
        prefijo son las de CIERRE:

            PSCH / PSCD / PSCA     Pinnacle cierre
            B365CH / B365CD/ B365CA  Bet365 cierre
            PSH / PSD / PSA        Pinnacle apertura

        Se prefiere Pinnacle: margen mínimo y límites altos hacen de su
        cierre el mejor estimador del precio justo. Bet365 es el
        respaldo cuando Pinnacle falta, que ocurre en temporadas
        antiguas.
        """
        rows: list[MatchRow] = []
        reader = csv.DictReader(io.StringIO(raw))

        for record in reader:
            home = _clean(record.get("HomeTeam"))
            away = _clean(record.get("AwayTeam"))
            date = _parse_date(record.get("Date"))
            if not home or not away or not date:
                continue

            odds = _extract_odds(record)

            rows.append(MatchRow(
                comp_id    = comp.comp_id,
                season     = season,
                date       = date,
                home_team  = home,
                away_team  = away,
                home_goals = _safe_int(record.get("FTHG")),
                away_goals = _safe_int(record.get("FTAG")),
                ht_home    = _safe_int(record.get("HTHG")),
                ht_away    = _safe_int(record.get("HTAG")),
                **odds,
            ))

        return rows

    # ── xG ────────────────────────────────────────────────────────────────────

    def load_xg(
        self,
        comp:   Competition | str,
        season: int,
    ) -> list[TeamXG]:
        """
        Métricas de xG por equipo, desde Understat.

        Retorna lista vacía si el backend no está instalado. Esa
        ausencia NO es un error: el plugin funciona sin xG en tier
        PARTIAL, y effective_tier() lo refleja para que el provider
        ajuste data_quality en consecuencia.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None or not competition.understat_id:
            return []

        key = (competition.comp_id, season)
        if key in self._xg:
            return self._xg[key]

        rows = self._fetch_understat(competition, season)
        self._xg[key] = rows
        return rows

    def _fetch_understat(
        self,
        comp:   Competition,
        season: int,
    ) -> list[TeamXG]:
        """
        Obtiene xG vía el cliente propio y lo agrega por equipo.

        Ya no hay DataFrame en ningún punto: UnderstatClient devuelve
        MatchXG tipados y aquí solo se suman. Es la lección de NFL
        aplicada de forma completa —la frontera con pandas no existe
        porque pandas no interviene.
        """
        # El identificador se valida AQUÍ, no solo en el llamador.
        #
        # load_xg() ya lo comprueba, pero apoyarse en esa comprobación
        # es un contrato implícito: se rompe en cuanto alguien añade
        # otra ruta de llamada a este método privado. El type checker
        # lo señala con razón — no puede rastrear la validación entre
        # métodos, y tampoco debería tener que hacerlo.
        understat_id = comp.understat_id
        if not understat_id:
            return []

        if not self.has_xg_backend() or self._understat is None:
            return []

        try:
            totals = self._understat.fetch_team_totals(understat_id, season)
        except Exception:
            return []

        return [
            TeamXG(comp_id=comp.comp_id, season=season, team=team, **values)
            for team, values in sorted(totals.items())
        ]

    def load_match_xg(
        self,
        comp:   Competition | str,
        season: int,
    ) -> list["MatchXG"]:
        """
        Historial de xG partido a partido.

        Devuelve más detalle que load_xg(), que solo da agregados de
        temporada. El modelo de proyección lo necesita para calcular
        ventanas móviles y separar rendimiento en casa y fuera —una
        distinción que en fútbol pesa mucho: la ventaja de campo vale
        ~0.35 goles y algunos equipos la explotan bastante más que
        otros.
        """
        competition = (comp if isinstance(comp, Competition)
                       else get_competition(str(comp)))
        if competition is None or not competition.understat_id:
            return []

        key = (competition.comp_id, season)
        if key in self._match_xg:
            return self._match_xg[key]

        understat_id = competition.understat_id
        rows: list["MatchXG"] = []
        if understat_id and self.has_xg_backend() and self._understat is not None:
            try:
                rows = self._understat.fetch_league_matches(understat_id, season)
            except Exception:
                rows = []

        self._match_xg[key] = rows
        return rows

    # ── Caché en disco ────────────────────────────────────────────────────────

    def _cache_path(self, name: str) -> Path | None:
        return self._cache_dir / name if self._cache_dir else None

    def _read_cache(self, name: str, season: int) -> str | None:
        """
        Lee del caché si existe y no ha caducado.

        El TTL depende de si la temporada está cerrada: una temporada
        pasada no cambia nunca, así que revalidarla sería tráfico puro.
        """
        path = self._cache_path(name)
        if path is None or not path.exists():
            return None

        ttl = (_TTL_CURRENT_SEASON if season >= self._current_season
               else _TTL_CLOSED_SEASON)
        try:
            age = time.time() - path.stat().st_mtime
            if age > ttl:
                return None
            return path.read_text(encoding="utf-8")
        except Exception:
            return None

    def _write_cache(self, name: str, content: str) -> None:
        path = self._cache_path(name)
        if path is None:
            return
        try:
            path.write_text(content, encoding="utf-8")
        except Exception:
            pass   # el caché es una optimización, no un requisito

    def clear_cache(self, disk: bool = False) -> None:
        """Limpia el caché en memoria, y opcionalmente el de disco."""
        self._matches.clear()
        self._xg.clear()
        self._match_xg.clear()
        if disk and self._cache_dir and self._cache_dir.exists():
            for path in self._cache_dir.glob("*.csv"):
                try:
                    path.unlink()
                except Exception:
                    pass


# ── Extracción de cuotas ─────────────────────────────────────────────────────

# Columnas de cierre por casa, en orden de preferencia.
# Pinnacle primero: margen mínimo y límites altos hacen de su cierre el
# mejor estimador del precio justo, y por tanto el benchmark más
# exigente para el backtest.
_CLOSING_SOURCES: tuple[tuple[str, dict[str, str]], ...] = (
    ("pinnacle", {
        "home": "PSCH", "draw": "PSCD", "away": "PSCA",
        "over": "PC>2.5", "under": "PC<2.5",
        "ah_line": "AHCh", "ah_home": "PCAHH", "ah_away": "PCAHA",
    }),
    ("bet365", {
        "home": "B365CH", "draw": "B365CD", "away": "B365CA",
        "over": "B365C>2.5", "under": "B365C<2.5",
        "ah_line": "AHCh", "ah_home": "B365CAHH", "ah_away": "B365CAHA",
    }),
)

# Columnas del formato EXTENDIDO (ligas no europeas).
#
# Pinnacle aparece como PH/PD/PA y su cierre como PCH/PCD/PCA. Cuando
# falta, se usa la media del mercado (AvgH...), que es un benchmark
# algo más blando pero honesto: representa lo que un apostante
# encontraría sin buscar el mejor precio.
_EXTRA_CLOSING: tuple[tuple[str, dict[str, str]], ...] = (
    ("pinnacle", {
        "home": "PCH", "draw": "PCD", "away": "PCA",
        "over": "AvgC>2.5", "under": "AvgC<2.5",
    }),
    ("market_avg", {
        "home": "AvgCH", "draw": "AvgCD", "away": "AvgCA",
        "over": "AvgC>2.5", "under": "AvgC<2.5",
    }),
)

_EXTRA_OPENING: tuple[tuple[str, dict[str, str]], ...] = (
    ("pinnacle", {
        "home": "PH", "draw": "PD", "away": "PA",
        "over": "Avg>2.5", "under": "Avg<2.5",
    }),
    ("market_avg", {
        "home": "AvgH", "draw": "AvgD", "away": "AvgA",
        "over": "Avg>2.5", "under": "Avg<2.5",
    }),
)


_OPENING_SOURCES: tuple[tuple[str, dict[str, str]], ...] = (
    ("pinnacle", {
        "home": "PSH", "draw": "PSD", "away": "PSA",
        "over": "P>2.5", "under": "P<2.5",
    }),
    ("bet365", {
        "home": "B365H", "draw": "B365D", "away": "B365A",
        "over": "B365>2.5", "under": "B365<2.5",
    }),
)


def _season_matches(raw, target: str) -> bool:
    """
    True si la etiqueta de temporada corresponde al año de inicio.

    El formato varía según el calendario de la liga:

        Brasileirão   '2023'       año natural
        Liga MX       '2023/2024'  cruza el año

    El plugin identifica las temporadas por su AÑO DE INICIO, así que
    '2023/2024' y '2023' apuntan ambas a 2023. Aceptar solo una de las
    dos formas dejaría una de las dos ligas sin datos, en silencio.
    """
    text = _clean(raw)
    if not text:
        return False
    if text == target:
        return True
    # '2023/2024' o '2023/24'
    if "/" in text:
        return text.split("/")[0].strip() == target
    return False


def _extract_odds_extra(record: dict) -> dict:
    """
    Cuotas del formato extendido, apertura y cierre por separado.

    Mismo criterio que en el principal: se exige el 1X2 completo de
    cada conjunto, y Pinnacle tiene prioridad.

    Cuando Pinnacle falta se usa la media del mercado. Es un benchmark
    algo más blando —representa lo que encontraría alguien sin buscar
    el mejor precio— pero honesto, y en estas ligas Pinnacle no cotiza
    todos los partidos.
    """
    out: dict = {"odds_source": "", "open_source": ""}

    for source, cols in _EXTRA_CLOSING:
        home = _safe_float(record.get(cols["home"]))
        draw = _safe_float(record.get(cols["draw"]))
        away = _safe_float(record.get(cols["away"]))
        if home and draw and away:
            out.update({
                "odds_home": home, "odds_draw": draw, "odds_away": away,
                "odds_over":  _safe_float(record.get(cols["over"])),
                "odds_under": _safe_float(record.get(cols["under"])),
                "odds_source": source,
            })
            break

    for source, cols in _EXTRA_OPENING:
        home = _safe_float(record.get(cols["home"]))
        draw = _safe_float(record.get(cols["draw"]))
        away = _safe_float(record.get(cols["away"]))
        if home and draw and away:
            out.update({
                "open_home": home, "open_draw": draw, "open_away": away,
                "open_over":  _safe_float(record.get(cols["over"])),
                "open_under": _safe_float(record.get(cols["under"])),
                "open_source": source,
            })
            break

    return out


def _extract_odds(record: dict) -> dict:
    """
    Extrae las cuotas de apertura y de cierre por separado.

    Se exige el 1X2 completo de cada conjunto: una cuota suelta no
    permite calcular las probabilidades implícitas sin vig, que es lo
    que consume el blending.

    Pinnacle tiene prioridad sobre Bet365 en ambos: margen mínimo y
    límites altos hacen de su precio el mejor estimador del justo.

    Los mercados secundarios —over/under, hándicap— pueden faltar sin
    invalidar la fila: el 1X2 es el que define si el partido es
    operable.
    """
    out: dict = {"odds_source": "", "open_source": ""}

    for source, cols in _CLOSING_SOURCES:
        home = _safe_float(record.get(cols["home"]))
        draw = _safe_float(record.get(cols["draw"]))
        away = _safe_float(record.get(cols["away"]))
        if home and draw and away:
            out.update({
                "odds_home": home, "odds_draw": draw, "odds_away": away,
                "odds_over":  _safe_float(record.get(cols["over"])),
                "odds_under": _safe_float(record.get(cols["under"])),
                "ah_line":    _safe_float(record.get(cols.get("ah_line", ""))),
                "ah_home":    _safe_float(record.get(cols.get("ah_home", ""))),
                "ah_away":    _safe_float(record.get(cols.get("ah_away", ""))),
                "odds_source": source,
            })
            break

    for source, cols in _OPENING_SOURCES:
        home = _safe_float(record.get(cols["home"]))
        draw = _safe_float(record.get(cols["draw"]))
        away = _safe_float(record.get(cols["away"]))
        if home and draw and away:
            out.update({
                "open_home": home, "open_draw": draw, "open_away": away,
                "open_over":  _safe_float(record.get(cols["over"])),
                "open_under": _safe_float(record.get(cols["under"])),
                "open_source": source,
            })
            break

    return out


# ── Utilidades ───────────────────────────────────────────────────────────────

def current_soccer_season(today: str | None = None) -> int:
    """
    Temporada europea en curso, por su año de inicio.

    Las ligas europeas van de agosto a mayo, así que la temporada
    2024/25 se identifica como 2024 — la misma convención que Understat
    y FBref, y la que ya usamos para NFL.

    El corte se pone en julio: a partir de ese mes la pretemporada ya
    corresponde al curso siguiente.
    """
    if today:
        try:
            date = datetime.strptime(today[:10], "%Y-%m-%d")
        except ValueError:
            date = datetime.now(timezone.utc)
    else:
        date = datetime.now(timezone.utc)

    return date.year if date.month >= 7 else date.year - 1


def _parse_date(raw) -> str | None:
    """
    Normaliza la fecha de football-data a ISO.

    Esa fuente ha usado dos formatos a lo largo de los años: 'dd/mm/yy'
    en las temporadas antiguas y 'dd/mm/yyyy' en las recientes. El
    cambio ocurrió sin aviso a mitad del histórico, así que hay que
    soportar ambos o perder temporadas enteras en el backtest.
    """
    text = _clean(raw)
    if not text:
        return None

    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def _clean(value) -> str:
    """Normaliza un valor de texto del CSV."""
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in ("", "nan", "na") else text


def _safe_float(value) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        result = float(text)
    except (ValueError, TypeError):
        return None
    return result if result == result else None   # descarta NaN


def _safe_int(value) -> int | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return int(float(text))
    except (ValueError, TypeError):
        return None