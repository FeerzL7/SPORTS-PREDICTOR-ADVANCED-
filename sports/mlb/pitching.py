"""
sports/mlb/pitching.py

PitchingFetcher: estadísticas de lanzadores abridores probables para MLB.

Migrado de analysis/pitching.py del sistema MLB con tres correcciones
documentadas en SPORTS_PREDICTOR_ARCHITECTURE.md §12.3:

1. ProbablePitcher como dataclass inmutable — no dict.
   El sistema MLB retornaba dicts con keys como 'pitcher_era',
   'pitcher_confirmado'. Con dataclass tipada, el MLBDataProvider
   accede a pitcher.era, pitcher.is_bullpen_day sin KeyError.

2. is_bullpen_day como campo explícito — no inferencia implícita.
   El sistema MLB detectaba bullpen day como ausencia de pitcher.
   Aquí es un campo booleano documentado con semántica clara para
   el MLBProjectionModel: si is_bullpen_day=True, usar ERA del bullpen
   y aumentar la incertidumbre de la proyección.

3. Delega el fetch en StatcastFetcher — sin duplicar lógica.
   pitching.py no reimplementa fetch HTTP ni caché. Usa StatcastFetcher
   para todo el acceso a MLB Stats API, manteniendo separación de
   responsabilidades: pitching.py = parsing de abridores probables.

Posición en el plugin
----------------------
MLBDataProvider.enrich_event(event)
    ↓
PitchingFetcher.get_starters(game_pk)
    → (home_pitcher: ProbablePitcher, away_pitcher: ProbablePitcher)
    ↓
TeamFeatures(
    defense_index = pitching_fetcher.defense_index(pitcher),
    sport_metadata = {'era': pitcher.era, 'fip': pitcher.fip, ...}
)
"""

from __future__ import annotations

from dataclasses import dataclass

from sports.mlb.statcast import StatcastFetcher, _LEAGUE_ERA, _safe_float

try:
    import requests as _requests
except ImportError:
    _requests = None  # type: ignore[assignment]

# CORRECCIÓN (auditoría 2026-08): antes _requests solo se asignaba en la
# rama try, dejando la variable "possibly unbound" para el type checker
# en cualquier punto donde se usara tras el try/except (Pylance/pyright
# marcaba esto en cada uno de los ~10 archivos que repiten este patrón
# de dependencia opcional). Ahora _requests siempre está definida (como
# None si el import falla), y _REQUESTS_AVAILABLE se deriva de eso en
# vez de ser una bandera independiente que podía desincronizarse.
_REQUESTS_AVAILABLE = _requests is not None

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# ERA por defecto cuando el pitcher no tiene datos suficientes.
# Usar ERA de liga como fallback conservador.
_DEFAULT_ERA: float = _LEAGUE_ERA

# Innings mínimos para considerar una muestra de ERA confiable.
_MIN_IP_RELIABLE: float = 30.0


# ── Dataclass del abridor ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbablePitcher:
    """
    Estadísticas del abridor probable para un partido.

    Inmutable: representa el estado del abridor en el momento del fetch.

    Campos
    ------
    pitcher_id       -- ID en MLB Stats API. 0 si no está confirmado.
    name             -- Nombre completo del pitcher.
    hand             -- Mano de lanzamiento: 'R', 'L', o None.
    era              -- ERA de la temporada. None si no hay datos.
    era_recent       -- ERA de los últimos 30 días. None si no disponible.
    fip              -- FIP calculado. None si no hay componentes.
    whip             -- WHIP de la temporada.
    k9               -- Strikeouts por 9 innings.
    bb9              -- Walks por 9 innings.
    innings_pitched  -- IP de la temporada (sample size).
    is_confirmed     -- True si el abridor está oficialmente confirmado
                      por el equipo. False = probable pero no oficial.
    is_bullpen_day   -- True si no hay abridor probable anunciado.
                      El MLBProjectionModel debe usar ERA de bullpen
                      y aumentar la incertidumbre en la proyección.
    """
    pitcher_id:      int
    name:            str
    hand:            str | None    = None
    era:             float | None  = None
    era_recent:      float | None  = None
    fip:             float | None  = None
    whip:            float | None  = None
    k9:              float | None  = None
    bb9:             float | None  = None
    innings_pitched: float | None  = None
    is_confirmed:    bool          = False
    is_bullpen_day:  bool          = False

    @property
    def has_sufficient_sample(self) -> bool:
        """True si IP >= 30 (ERA estadísticamente confiable)."""
        return (
            self.innings_pitched is not None
            and self.innings_pitched >= _MIN_IP_RELIABLE
        )

    @property
    def effective_era(self) -> float:
        """
        ERA efectiva para proyección.

        Prioridad: era_recent > era > DEFAULT_ERA.
        era_recent es más predictivo para el partido próximo si
        el pitcher viene de buena o mala racha reciente.
        """
        if self.era_recent is not None:
            return self.era_recent
        if self.era is not None:
            return self.era
        return _DEFAULT_ERA

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "pitcher_id":      self.pitcher_id,
            "pitcher_name":    self.name,
            "pitcher_hand":    self.hand,
            "era":             self.era,
            "era_recent":      self.era_recent,
            "fip":             self.fip,
            "whip":            self.whip,
            "k9":              self.k9,
            "bb9":             self.bb9,
            "innings_pitched": self.innings_pitched,
            "is_confirmed":    self.is_confirmed,
            "is_bullpen_day":  self.is_bullpen_day,
        }


# ── Fetcher principal ─────────────────────────────────────────────────────────

class PitchingFetcher:
    """
    Obtiene estadísticas de abridores probables desde MLB Stats API.

    Usa StatcastFetcher internamente para todo el acceso HTTP y caché —
    no duplica lógica de fetch ni de caché.

    Parámetros
    ----------
    statcast_fetcher  -- StatcastFetcher ya configurado con config_loader
                        y cache_dir. Si None, crea uno con defaults.
    """

    def __init__(
        self,
        statcast_fetcher: StatcastFetcher | None = None,
    ) -> None:
        self._statcast = statcast_fetcher or StatcastFetcher()

    def get_starters(
        self,
        game_pk: int,
    ) -> tuple[ProbablePitcher, ProbablePitcher]:
        """
        Obtiene los abridores probables para un partido.

        Parámetros
        ----------
        game_pk  -- ID del partido en MLB Stats API.

        Retorna
        -------
        (home_pitcher, away_pitcher) — ambos ProbablePitcher.
        Si no hay pitcher confirmado, retorna un ProbablePitcher
        con is_bullpen_day=True.
        """
        raw = self._fetch_probable_pitchers(game_pk)
        if raw is None:
            return _bullpen_day(), _bullpen_day()

        home_raw = raw.get("home")
        away_raw = raw.get("away")

        home_pitcher = self._build_pitcher(home_raw)
        away_pitcher = self._build_pitcher(away_raw)

        return home_pitcher, away_pitcher

    def defense_index(self, pitcher: ProbablePitcher) -> float:
        """
        Calcula el defense_index del equipo desde el ERA del abridor.

        defense_index = ERA_LIGA / ERA_efectiva
        < 1.0: mejor que la liga (pitcher dominante)
        > 1.0: peor que la liga (pitcher débil)
        = 1.0: promedio de liga

        Si is_bullpen_day=True, usa ERA de liga (1.0) con incertidumbre
        implícita — el MLBProjectionModel añade varianza adicional.
        """
        if pitcher.is_bullpen_day:
            return 1.0
        era = pitcher.effective_era
        if era <= 0:
            return 1.0
        return round(_LEAGUE_ERA / era, 4)

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _build_pitcher(self, raw: dict | None) -> ProbablePitcher:
        """
        Construye un ProbablePitcher desde los datos raw del schedule.

        Si raw es None o el pitcher no tiene ID, retorna is_bullpen_day=True.
        """
        if raw is None:
            return _bullpen_day()

        pitcher_id = raw.get("id")
        if not pitcher_id:
            return _bullpen_day()

        pitcher_id = int(pitcher_id)
        name       = raw.get("fullName") or raw.get("name", "Unknown")
        hand       = raw.get("pitchHand", {}).get("code") if isinstance(raw.get("pitchHand"), dict) else raw.get("hand")
        confirmed  = raw.get("status", "") == "P"  # 'P' = Probable en MLB API

        # Obtener estadísticas desde StatcastFetcher
        statcast = self._statcast.fetch_pitcher_stats(pitcher_id)
        era_recent = self._statcast.fetch_pitcher_recent(pitcher_id, days=30)

        return ProbablePitcher(
            pitcher_id      = pitcher_id,
            name            = name,
            hand            = hand or statcast.hand,
            era             = statcast.era,
            era_recent      = era_recent,
            fip             = statcast.fip,
            whip            = statcast.whip,
            k9              = statcast.k9,
            bb9             = statcast.bb9,
            innings_pitched = statcast.innings_pitched,
            is_confirmed    = confirmed,
            is_bullpen_day  = False,
        )

    def _fetch_probable_pitchers(self, game_pk: int) -> dict | None:
        """
        Fetch de abridores probables desde MLB Stats API.

        Endpoint: /api/v1.1/game/{gamePk}/feed/live
        Retorna {'home': {...}, 'away': {...}} o None si falla.
        """
        if not _REQUESTS_AVAILABLE or _requests is None:
            return None

        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        try:
            resp = _requests.get(url, timeout=10)
            if not resp or resp.status_code != 200:
                return None

            data        = resp.json()
            game_data   = data.get("gameData", {})
            probable    = game_data.get("probablePitchers", {})

            home = probable.get("home")
            away = probable.get("away")

            # Si no están en probablePitchers, intentar desde boxscore
            if not home and not away:
                liveData   = data.get("liveData", {})
                boxscore   = liveData.get("boxscore", {})
                teams      = boxscore.get("teams", {})
                home_team  = teams.get("home", {})
                away_team  = teams.get("away", {})
                pitchers_h = home_team.get("pitchers", [])
                pitchers_a = away_team.get("pitchers", [])
                home = {"id": pitchers_h[0]} if pitchers_h else None
                away = {"id": pitchers_a[0]} if pitchers_a else None

            return {"home": home, "away": away}

        except Exception:
            return None


# ── Helpers de módulo ─────────────────────────────────────────────────────────

def _bullpen_day(name: str = "Bullpen") -> ProbablePitcher:
    """
    Retorna un ProbablePitcher de placeholder para bullpen day.

    pitcher_id=0 es la señal para el MLBProjectionModel de que
    debe usar ERA del bullpen y aumentar la incertidumbre.
    """
    return ProbablePitcher(
        pitcher_id     = 0,
        name           = name,
        is_bullpen_day = True,
        is_confirmed   = False,
    )