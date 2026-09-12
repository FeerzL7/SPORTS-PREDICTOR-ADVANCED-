"""
sports/nfl/injuries.py

NFLInjuryFetcher: reporte de lesiones y su impacto en la proyección.

Por qué es el fetcher más importante del plugin NFL
-----------------------------------------------------
Una lesión de quarterback titular mueve el spread de mercado unos 7
puntos — el mayor swing por un solo jugador de cualquier deporte que
maneja este sistema. En MLB, la baja del abridor se compensa con el
bullpen y cambia la línea en ~0.3 carreras. En NFL no hay equivalente:
el QB suplente típico rinde muy por debajo del titular y no existe
mecanismo de sustitución gradual.

Por eso config/nfl.yaml asigna `qb_out_penalty: 7.0` y un techo
`max_injury_penalty: 10.0`: más allá de ese punto el partido es
demasiado impredecible y es preferible no apostarlo.

Tres problemas estructurales de esta fuente de datos
------------------------------------------------------

1. LATENCIA DE NFLVERSE
   nflverse publica los datos los martes tras la jornada, con ~24h de
   retraso sobre el último partido. El injury report que de verdad
   importa —el parte final del viernes antes de los partidos del
   domingo— puede no estar disponible cuando ejecutamos el pipeline.

   Estrategia: intentar la semana en curso; si no hay datos, caer a la
   semana anterior y marcar el reporte como `is_stale=True`. El
   NFLProjectionModel reduce `confidence` y el provider baja
   `data_quality` cuando eso ocurre.

   Nunca se inventa información: un reporte stale se declara como tal
   y el pipeline decide qué hacer con esa incertidumbre.

2. IDENTIFICAR TITULARES
   El injury report lista a todo el que aparece en el parte médico,
   incluidos jugadores de plantilla profunda que no iban a jugar de
   todos modos. Penalizar la baja de un WR5 igual que la de un WR1
   sería peor que no penalizar nada.

   Se usa `depth_chart_position` de los rosters de nflverse para
   distinguir titulares. Si esa información falta, se degrada a contar
   solo el primer jugador lesionado de cada grupo posicional — una
   heurística conservadora que evita inflar la penalización.

3. report_status NO ES SUFICIENTE POR SÍ SOLO
   La NFL exige declarar 'Out', 'Doubtful' o 'Questionable', pero esas
   etiquetas tienen tasas de participación muy distintas según el
   historial de práctica de la semana:

       Questionable + práctica completa el viernes  → juega ~95%
       Questionable + participación limitada        → juega ~70%
       Questionable + sin practicar en la semana    → juega ~30%

   Se combinan ambas señales para estimar la probabilidad real de
   ausencia en vez de aplicar un porcentaje fijo a todos los
   'Questionable'.

Qué NO hace este módulo
-------------------------
No estima la calidad del suplente. Un equipo con un QB2 competente
pierde menos que uno cuyo suplente nunca ha jugado, pero modelar eso
requiere datos de rendimiento individual que no están en el alcance
actual. La penalización es uniforme por posición, lo que sobreestima
el daño en equipos con buena profundidad y lo subestima en los demás.
Queda documentado como mejora futura.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sports.nfl.data_source import NFLDataSource, _current_nfl_season


# ── Estados del reporte oficial ──────────────────────────────────────────────

STATUS_OUT          = "Out"
STATUS_DOUBTFUL     = "Doubtful"
STATUS_QUESTIONABLE = "Questionable"
STATUS_IR           = "Injured Reserve"

# Estados que implican ausencia segura
_DEFINITELY_OUT = frozenset({STATUS_OUT, STATUS_IR})


# ── Grupos posicionales ──────────────────────────────────────────────────────
# Se agrupan por impacto esperado, no por nomenclatura de la liga.

POSITION_QB    = "QB"
POSITION_SKILL = "SKILL"   # RB, WR, TE — generadores de yardas
POSITION_OL    = "OL"      # T, G, C — protección y bloqueo
POSITION_DEF   = "DEF"     # toda la defensa
POSITION_OTHER = "OTHER"   # K, P, LS — impacto marginal

_POSITION_GROUPS: dict[str, str] = {
    "QB": POSITION_QB,
    "RB": POSITION_SKILL, "FB": POSITION_SKILL,
    "WR": POSITION_SKILL, "TE": POSITION_SKILL,
    "T":  POSITION_OL, "OT": POSITION_OL, "LT": POSITION_OL, "RT": POSITION_OL,
    "G":  POSITION_OL, "OG": POSITION_OL, "LG": POSITION_OL, "RG": POSITION_OL,
    "C":  POSITION_OL, "OL": POSITION_OL,
    "DE": POSITION_DEF, "DT": POSITION_DEF, "NT": POSITION_DEF,
    "LB": POSITION_DEF, "ILB": POSITION_DEF, "OLB": POSITION_DEF,
    "MLB": POSITION_DEF, "EDGE": POSITION_DEF,
    "CB": POSITION_DEF, "S": POSITION_DEF, "FS": POSITION_DEF,
    "SS": POSITION_DEF, "DB": POSITION_DEF,
    "K": POSITION_OTHER, "P": POSITION_OTHER, "LS": POSITION_OTHER,
}


# ── Probabilidad de ausencia según estado y práctica ─────────────────────────
#
# Tasas de participación observadas históricamente. La combinación de
# report_status con practice_status discrimina mucho mejor que el estado
# por sí solo: un 'Questionable' que practicó completo el viernes juega
# casi siempre, uno que no practicó en toda la semana casi nunca.
#
# Claves de practice_status en nflverse (abreviadas al matchear):
#   'Full Participation in Practice'     → 'full'
#   'Limited Participation in Practice'  → 'limited'
#   'Did Not Participate In Practice'    → 'dnp'
#
_MISS_PROBABILITY: dict[tuple[str, str], float] = {
    (STATUS_QUESTIONABLE, "full"):    0.05,
    (STATUS_QUESTIONABLE, "limited"): 0.30,
    (STATUS_QUESTIONABLE, "dnp"):     0.70,
    (STATUS_QUESTIONABLE, "unknown"): 0.35,   # = qb_questionable_pct del config

    (STATUS_DOUBTFUL, "full"):    0.50,
    (STATUS_DOUBTFUL, "limited"): 0.75,
    (STATUS_DOUBTFUL, "dnp"):     0.90,
    (STATUS_DOUBTFUL, "unknown"): 0.75,
}

# Penalizaciones por defecto si no hay ConfigLoader inyectado.
# Coinciden con los valores documentados en config/nfl.yaml.
_DEFAULT_QB_OUT       = 7.0
_DEFAULT_QB_QUESTION  = 0.35
_DEFAULT_SKILL_OUT    = 1.5
_DEFAULT_OL_OUT       = 0.8
_DEFAULT_MAX_PENALTY  = 10.0

# Penalización por titular defensivo ausente.
#
# NOTA: config/nfl.yaml no define este parámetro todavía. El valor por
# defecto es conservador (menor que skill_out_penalty) porque la defensa
# tiene 11 titulares y la baja de uno se reparte mejor que en ataque.
# Debería añadirse `nfl.injuries.def_out_penalty` al YAML para que sea
# calibrable como el resto.
_DEFAULT_DEF_OUT = 0.6

# Cuántos jugadores por grupo se consideran titulares cuando no hay
# depth chart disponible. Heurística conservadora.
_STARTERS_PER_GROUP: dict[str, int] = {
    POSITION_QB:    1,
    POSITION_SKILL: 3,   # RB1 + WR1/WR2 aproximadamente
    POSITION_OL:    5,
    POSITION_DEF:   11,
    POSITION_OTHER: 0,   # K/P/LS no penalizan
}


# ── Entradas del reporte ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class InjuryEntry:
    """
    Un jugador en el reporte de lesiones.

    Campos
    ------
    player          -- Nombre del jugador.
    position        -- Posición según nflverse ('QB', 'WR', 'CB'...).
    group           -- Grupo posicional normalizado (QB/SKILL/OL/DEF/OTHER).
    report_status   -- 'Out', 'Doubtful', 'Questionable' o None.
    practice_status -- Normalizado a 'full', 'limited', 'dnp' o 'unknown'.
    is_starter      -- True si figura como titular en el depth chart.
    """
    player:          str
    position:        str
    group:           str
    report_status:   str | None = None
    practice_status: str = "unknown"
    is_starter:      bool = False

    @property
    def miss_probability(self) -> float:
        """
        Probabilidad estimada de que el jugador NO juegue.

        'Out' e 'Injured Reserve' son certeza. Para 'Questionable' y
        'Doubtful' se combina el estado con el historial de práctica
        de la semana, que discrimina mucho mejor que el estado solo.
        """
        if self.report_status in _DEFINITELY_OUT:
            return 1.0
        if self.report_status is None:
            return 0.0
        key = (self.report_status, self.practice_status)
        return _MISS_PROBABILITY.get(
            key,
            _MISS_PROBABILITY.get((self.report_status, "unknown"), 0.0),
        )

    @property
    def is_out(self) -> bool:
        """True si la ausencia es segura."""
        return self.report_status in _DEFINITELY_OUT


@dataclass(frozen=True)
class InjuryReport:
    """
    Reporte de lesiones de un equipo para una semana concreta.

    Campos
    ------
    team        -- Abreviación del equipo.
    season      -- Temporada.
    week        -- Semana del reporte.
    entries     -- Jugadores en el parte médico.
    is_stale    -- True si los datos son de una semana anterior a la
                   solicitada. Ocurre por la latencia de nflverse
                   (publica los martes, ~24h tras la jornada).
    source_week -- Semana real de la que provienen los datos. Distinta
                   de `week` cuando is_stale=True.
    """
    team:        str
    season:      int
    week:        int
    entries:     list[InjuryEntry] = field(default_factory=list)
    is_stale:    bool = False
    source_week: int | None = None

    # ── Consultas sobre el QB ─────────────────────────────────────────────────

    @property
    def qb_entry(self) -> InjuryEntry | None:
        """
        Entrada del QB titular si está en el reporte.

        Si hay varios QB lesionados, devuelve el titular; si ninguno
        figura como titular, el de estado más grave.
        """
        qbs = [e for e in self.entries if e.group == POSITION_QB]
        if not qbs:
            return None
        starters = [e for e in qbs if e.is_starter]
        if starters:
            return starters[0]
        # Sin depth chart: el de mayor probabilidad de ausencia
        return max(qbs, key=lambda e: e.miss_probability)

    @property
    def qb_status(self) -> str | None:
        """Estado del QB titular. None si no está en el reporte."""
        entry = self.qb_entry
        return entry.report_status if entry else None

    @property
    def qb_out(self) -> bool:
        """True si el QB titular está descartado con certeza."""
        entry = self.qb_entry
        return entry.is_out if entry else False

    # ── Penalización agregada ─────────────────────────────────────────────────

    def total_penalty(
        self,
        qb_out_penalty:    float = _DEFAULT_QB_OUT,
        skill_out_penalty: float = _DEFAULT_SKILL_OUT,
        ol_out_penalty:    float = _DEFAULT_OL_OUT,
        def_out_penalty:   float = _DEFAULT_DEF_OUT,
        max_penalty:       float = _DEFAULT_MAX_PENALTY,
    ) -> float:
        """
        Penalización total en puntos por las lesiones del equipo.

        Cada jugador aporta `penalización_posicional × probabilidad_de_ausencia`.
        Un QB 'Out' aporta los 7 puntos completos; un QB 'Questionable'
        que practicó limitado aporta 7 × 0.30 = 2.1.

        Solo cuentan los titulares: la baja de un suplente profundo no
        cambia la proyección de forma medible y contarla inflaría la
        penalización de equipos que simplemente reportan más jugadores.

        El resultado se acota a `max_penalty`. Superar ese umbral
        significa que el partido tiene demasiada incertidumbre; el
        NFLProjectionModel lo refleja bajando `confidence`, y los
        filtros de EV harán el resto.
        """
        penalties = {
            POSITION_QB:    qb_out_penalty,
            POSITION_SKILL: skill_out_penalty,
            POSITION_OL:    ol_out_penalty,
            POSITION_DEF:   def_out_penalty,
            POSITION_OTHER: 0.0,
        }

        total = 0.0
        for entry in self.entries:
            if not entry.is_starter:
                continue
            base = penalties.get(entry.group, 0.0)
            total += base * entry.miss_probability

        return round(min(total, max_penalty), 3)

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    @property
    def starters_out(self) -> int:
        """Número de titulares con ausencia segura."""
        return sum(1 for e in self.entries if e.is_starter and e.is_out)

    @property
    def has_data(self) -> bool:
        """True si el reporte contiene información real."""
        return len(self.entries) > 0

    def by_group(self, group: str) -> list[InjuryEntry]:
        """Entradas de un grupo posicional concreto."""
        return [e for e in self.entries if e.group == group]

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "injury_qb_status":    self.qb_status,
            "injury_qb_out":       self.qb_out,
            "injury_starters_out": self.starters_out,
            "injury_total_listed": len(self.entries),
            "injury_is_stale":     self.is_stale,
            "injury_source_week":  self.source_week,
        }


# ── Fetcher principal ─────────────────────────────────────────────────────────

class NFLInjuryFetcher:
    """
    Obtiene el reporte de lesiones desde nflverse.

    Parámetros
    ----------
    data_source   -- NFLDataSource compartido. Si None, crea uno.
    season        -- Temporada. None = actual.
    config_loader -- ConfigLoader con nfl.yaml, para leer las
                     penalizaciones calibradas.
    max_stale_weeks -- Cuántas semanas hacia atrás buscar si la semana
                     solicitada no tiene datos. Default 1: solo la
                     inmediatamente anterior. Más allá, los datos son
                     demasiado antiguos para ser informativos.
    """

    def __init__(
        self,
        data_source:     NFLDataSource | None = None,
        season:          int | None = None,
        config_loader    = None,
        max_stale_weeks: int = 1,
    ) -> None:
        self._source   = data_source or NFLDataSource()
        self._season   = season or _current_nfl_season()
        self._config   = config_loader
        self._max_stale = max_stale_weeks

        # Caché: {week: {team: InjuryReport}}
        self._cache: dict[int, dict[str, InjuryReport]] = {}
        # Titulares por equipo desde el depth chart
        self._starters: dict[str, set[str]] | None = None

        # Penalizaciones desde el config
        self._qb_out    = self._cfg("nfl.injuries.qb_out_penalty",    _DEFAULT_QB_OUT)
        self._skill_out = self._cfg("nfl.injuries.skill_out_penalty", _DEFAULT_SKILL_OUT)
        self._ol_out    = self._cfg("nfl.injuries.ol_out_penalty",    _DEFAULT_OL_OUT)
        self._def_out   = self._cfg("nfl.injuries.def_out_penalty",   _DEFAULT_DEF_OUT)
        self._max_pen   = self._cfg("nfl.injuries.max_injury_penalty", _DEFAULT_MAX_PENALTY)

    # ── API pública ───────────────────────────────────────────────────────────

    def fetch(self, team: str, week: int) -> InjuryReport:
        """
        Reporte de lesiones de un equipo para una semana.

        Si la semana solicitada no tiene datos (latencia de nflverse),
        cae a semanas anteriores hasta `max_stale_weeks` y marca el
        resultado con `is_stale=True`.

        Nunca lanza: un equipo sin lesiones reportadas devuelve un
        reporte vacío, que es semánticamente correcto.
        """
        reports = self.fetch_all(week)
        return reports.get(team) or InjuryReport(
            team=team, season=self._season, week=week,
        )

    def fetch_all(self, week: int) -> dict[str, InjuryReport]:
        """
        Reportes de todos los equipos para una semana.

        Una sola pasada sobre el dataset produce los 32 reportes.
        """
        if week in self._cache:
            return self._cache[week]

        rows = self._load_rows()
        if not rows:
            self._cache[week] = {}
            return {}

        # ── Resolver la semana efectiva (latencia de nflverse) ────
        source_week = self._resolve_week(rows, week)
        if source_week is None:
            self._cache[week] = {}
            return {}

        is_stale = source_week != week
        starters = self._load_starters()

        by_team: dict[str, list[InjuryEntry]] = {}
        for row in rows:
            if row["week"] != source_week:
                continue
            team = row["team"]
            if not team:
                continue
            entry = self._build_entry(row, starters.get(team, set()))
            if entry is not None:
                by_team.setdefault(team, []).append(entry)

        reports = {
            team: InjuryReport(
                team        = team,
                season      = self._season,
                week        = week,
                entries     = entries,
                is_stale    = is_stale,
                source_week = source_week,
            )
            for team, entries in by_team.items()
        }

        self._cache[week] = reports
        return reports

    def penalty_for(self, team: str, week: int) -> float:
        """
        Penalización en puntos por lesiones, con los valores del config.

        Método de conveniencia para NFLProjectionModel: evita que el
        modelo tenga que conocer los parámetros de penalización.
        """
        report = self.fetch(team, week)
        return report.total_penalty(
            qb_out_penalty    = self._qb_out,
            skill_out_penalty = self._skill_out,
            ol_out_penalty    = self._ol_out,
            def_out_penalty   = self._def_out,
            max_penalty       = self._max_pen,
        )

    def clear_cache(self) -> None:
        """Limpia el caché de reportes."""
        self._cache.clear()
        self._starters = None

    # ── Carga de datos ────────────────────────────────────────────────────────

    def _load_rows(self) -> list[dict]:
        """
        Carga el dataset de lesiones como lista de dicts tipados.

        Igual que en schedule.py y team_stats.py, el DataFrame se
        convierte a estructuras Python en un único punto y toda la
        lógica posterior opera sobre ellas. Evita el boolean masking
        de pandas que el type checker no modela.
        """
        try:
            df = self._source.load_injuries([self._season])
        except Exception:
            return []

        if df is None or len(df) == 0:
            return []

        rows: list[dict] = []
        for r in df.itertuples(index=False):
            week = _safe_int(getattr(r, "week", None))
            if week is None:
                continue
            rows.append({
                "week":            week,
                "team":            _safe_str(getattr(r, "team", None)) or "",
                "player":          (
                    _safe_str(getattr(r, "full_name", None))
                    or _safe_str(getattr(r, "player_name", None))
                    or ""
                ),
                "position":        _safe_str(getattr(r, "position", None)) or "",
                "report_status":   _safe_str(getattr(r, "report_status", None)),
                "practice_status": _safe_str(getattr(r, "practice_status", None)),
            })
        return rows

    def _resolve_week(self, rows: list[dict], week: int) -> int | None:
        """
        Determina de qué semana tomar los datos.

        Prueba la semana solicitada; si no hay filas, retrocede hasta
        `max_stale_weeks`. Retorna None si no encuentra nada usable.

        Esta es la mitigación concreta de la latencia de nflverse
        documentada en data_source.py: los datos se publican los
        martes, así que el parte del viernes previo a los partidos del
        domingo puede no estar disponible al ejecutar el pipeline.
        """
        available = {r["week"] for r in rows}
        for candidate in range(week, week - self._max_stale - 1, -1):
            if candidate in available:
                return candidate
        return None

    def _load_starters(self) -> dict[str, set[str]]:
        """
        Titulares por equipo según el depth chart de los rosters.

        Si los rosters no están disponibles o no traen depth chart,
        retorna vacío y el fetcher degrada a la heurística de contar
        solo los primeros jugadores de cada grupo posicional.
        """
        if self._starters is not None:
            return self._starters

        try:
            df = self._source.load_rosters([self._season])
        except Exception:
            self._starters = {}
            return self._starters

        if df is None or len(df) == 0:
            self._starters = {}
            return self._starters

        starters: dict[str, set[str]] = {}
        for r in df.itertuples(index=False):
            team  = _safe_str(getattr(r, "team", None))
            name  = (
                _safe_str(getattr(r, "player_name", None))
                or _safe_str(getattr(r, "full_name", None))
            )
            depth = _safe_str(getattr(r, "depth_chart_position", None))
            if not team or not name:
                continue
            # depth_chart_position presente = figura en el depth chart.
            # nflverse no numera la profundidad de forma fiable en todas
            # las temporadas, así que se toma la presencia como señal.
            if depth:
                starters.setdefault(team, set()).add(_normalize_name(name))

        self._starters = starters
        return starters

    # ── Construcción de entradas ──────────────────────────────────────────────

    @staticmethod
    def _build_entry(row: dict, team_starters: set[str]) -> InjuryEntry | None:
        """Convierte una fila del dataset en InjuryEntry."""
        player = row["player"]
        if not player:
            return None

        position = row["position"].upper()
        group    = _POSITION_GROUPS.get(position, POSITION_OTHER)

        return InjuryEntry(
            player          = player,
            position        = position,
            group           = group,
            report_status   = _normalize_status(row["report_status"]),
            practice_status = _normalize_practice(row["practice_status"]),
            is_starter      = (
                _normalize_name(player) in team_starters if team_starters
                else _fallback_is_starter(group)
            ),
        )

    # ── Config ────────────────────────────────────────────────────────────────

    def _cfg(self, key: str, default: float) -> float:
        """Lee un valor del ConfigLoader con fallback documentado."""
        if self._config is None:
            return default
        try:
            val = self._config.get(key, default=default)
            return float(val) if val is not None else default
        except (ValueError, TypeError, AttributeError):
            return default


# ── Utilidades de módulo ──────────────────────────────────────────────────────

def _normalize_status(raw: str | None) -> str | None:
    """
    Normaliza el report_status a los valores canónicos.

    nflverse usa variantes de capitalización y a veces incluye
    'Injured Reserve' o 'IR' en el mismo campo.
    """
    if not raw:
        return None
    s = raw.strip().lower()
    if s in ("out", "o"):
        return STATUS_OUT
    if s in ("doubtful", "d"):
        return STATUS_DOUBTFUL
    if s in ("questionable", "q"):
        return STATUS_QUESTIONABLE
    if "injured reserve" in s or s == "ir":
        return STATUS_IR
    return None


def _normalize_practice(raw: str | None) -> str:
    """
    Normaliza el practice_status a 'full', 'limited', 'dnp' o 'unknown'.

    nflverse usa cadenas largas:
        'Full Participation in Practice'
        'Limited Participation in Practice'
        'Did Not Participate In Practice'
    """
    if not raw:
        return "unknown"
    s = raw.strip().lower()
    if "did not participate" in s or s == "dnp":
        return "dnp"
    if "limited" in s:
        return "limited"
    if "full" in s:
        return "full"
    return "unknown"


def _normalize_name(name: str) -> str:
    """
    Normaliza un nombre de jugador para matching entre datasets.

    nflverse NO garantiza que el mismo jugador aparezca con idéntica
    grafía en injuries y rosters. Variantes observadas:

        'Patrick Mahomes'  vs  'P.Mahomes'
        'Odell Beckham Jr.' vs 'Odell Beckham'
        'A.J. Brown'        vs  'AJ Brown'

    Si el matching falla, el QB titular queda marcado como suplente y
    se pierde la penalización de 7 puntos — el ajuste más importante
    del modelo NFL. Un fallo silencioso en este punto es peor que un
    dato ausente, porque el pipeline seguiría con una proyección
    confiada pero equivocada.

    La normalización reduce a: apellido + inicial del nombre, en
    minúsculas y sin puntuación. Eso sobrevive a las tres variantes
    de arriba manteniendo suficiente especificidad para no colisionar
    entre jugadores distintos del mismo equipo.
    """
    if not name:
        return ""
    cleaned = "".join(c for c in name if c.isalnum() or c.isspace())
    parts = cleaned.lower().split()
    if not parts:
        return ""
    # Descartar sufijos generacionales que aparecen de forma inconsistente
    if parts[-1] in ("jr", "sr", "ii", "iii", "iv", "v") and len(parts) > 1:
        parts = parts[:-1]
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]}{parts[-1]}"


def _fallback_is_starter(group: str) -> bool:
    """
    Heurística cuando no hay depth chart disponible.

    Sin información de titularidad, se asume titular a los QB (un
    equipo tiene uno solo y su ausencia siempre importa) y no titular
    al resto. Es deliberadamente conservador: prefiere subestimar la
    penalización a inflarla con lesiones de jugadores irrelevantes.
    """
    return group == POSITION_QB


def _is_nan(value) -> bool:
    """True si el valor es NaN."""
    try:
        return value != value
    except Exception:
        return False


def _safe_str(value) -> str | None:
    """Convierte a str, retornando None para NaN o vacíos."""
    if value is None or _is_nan(value):
        return None
    s = str(value).strip()
    return s if s and s.lower() != "nan" else None


def _safe_int(value) -> int | None:
    """Convierte a int de forma segura (NaN → None)."""
    if value is None or _is_nan(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None