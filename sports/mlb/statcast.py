"""
sports/mlb/statcast.py

StatcastFetcher: estadísticas avanzadas de pitching y bateo para MLB.

Migrado de analysis/statcast.py del sistema MLB con cuatro correcciones
documentadas en SPORTS_PREDICTOR_ARCHITECTURE.md §12.3:

1. Sin pybaseball — solo MLB Stats API oficial.
   El sistema MLB usaba pybaseball (wrapper no oficial de Baseball Savant)
   que fallaba silenciosamente ante cambios del sitio. Aquí se usa
   exclusivamente la MLB Stats API (statsapi.mlb.com) que es el
   proveedor oficial y estable. xwOBA y Hard Hit% quedan como campos
   opcionales (None) hasta que MLB los exponga directamente via API.

2. Park factors desde config/mlb.yaml — no hardcodeados en Python.
   El sistema MLB tenía PARK_FACTORS = {'lad': 0.94, 'col': 1.36, ...}
   como constante en el código. Ahora viven en config/mlb.yaml bajo
   mlb.park_factors.{venue_id} — modificables sin tocar código.

3. FIP con fórmula exacta documentada.
   El sistema MLB usaba una aproximación. Aquí:
   FIP = (13×HR + 3×(BB+HBP) - 2×K) / IP + cFIP
   con cFIP = 3.10 (constante de liga MLB 2024-2026).

4. PitcherStatcast y BattingStatcast como dataclasses inmutables.
   El sistema MLB retornaba dicts — sin validación de tipos y con
   riesgo de KeyError. Dataclasses tipadas con defaults explícitos.

Posición en el plugin
----------------------
MLBDataProvider.enrich_event(event)
    ↓
StatcastFetcher.fetch_pitcher_stats(pitcher_id)  → PitcherStatcast
StatcastFetcher.fetch_batter_stats(team_id)       → BattingStatcast
StatcastFetcher.fetch_park_factor(venue_id)       → float
    ↓
TeamFeatures(sport_metadata={'era': ..., 'fip': ..., 'ops': ...})

El Core (core/) nunca importa desde sports/mlb/. Este módulo es
exclusivamente consumido por MLBDataProvider y MLBProjectionModel.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

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

# Base URL de la MLB Stats API (oficial, estable desde 2016)
_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Constante FIP de liga MLB (ajustada anualmente; valor 2024-2026)
_C_FIP: float = 3.10

# ERA de liga promedio MLB para normalización de defense_index
_LEAGUE_ERA: float = 4.20

# OPS de liga promedio MLB para normalización de offense_index
_LEAGUE_OPS: float = 0.720


# ── Dataclasses de estadísticas ────────────────────────────────────────────────

@dataclass(frozen=True)
class PitcherStatcast:
    """
    Estadísticas avanzadas del abridor para una temporada.

    Todos los campos son opcionales (None) si no están disponibles.
    El MLBProjectionModel maneja los None con fallbacks documentados.

    Campos
    ------
    pitcher_id       -- ID del jugador en MLB Stats API.
    season           -- Temporada (año).
    era              -- Earned Run Average (temporada completa).
    fip              -- Fielding Independent Pitching.
                       FIP = (13*HR + 3*(BB+HBP) - 2*K) / IP + cFIP
    whip             -- Walks + Hits per Inning Pitched.
    k9               -- Strikeouts por 9 innings.
    bb9              -- Walks por 9 innings.
    hr9              -- Home runs por 9 innings.
    innings_pitched  -- Innings lanzados (sample size).
                       >= 80 IP = "alta muestra" para H2 del audit.
    era_recent       -- ERA de los últimos 30 días (más predictivo).
    xwoba_against    -- Expected wOBA contra el pitcher (Baseball Savant).
                       None hasta que MLB Stats API lo exponga.
    hard_hit_pct     -- % de bateadores con exit velocity >= 95 mph.
                       None hasta que MLB Stats API lo exponga.
    hand             -- Mano del pitcher: 'R' o 'L'.
    """
    pitcher_id:     int
    season:         int
    era:            float | None = None
    fip:            float | None = None
    whip:           float | None = None
    k9:             float | None = None
    bb9:            float | None = None
    hr9:            float | None = None
    innings_pitched: float | None = None
    era_recent:     float | None = None
    xwoba_against:  float | None = None
    hard_hit_pct:   float | None = None
    hand:           str   | None = None

    @property
    def has_sufficient_sample(self) -> bool:
        """True si IP >= 80 (umbral de H2 del audit MLB)."""
        return (
            self.innings_pitched is not None
            and self.innings_pitched >= 80
        )

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "pitcher_id":      self.pitcher_id,
            "era":             self.era,
            "fip":             self.fip,
            "whip":            self.whip,
            "k9":              self.k9,
            "bb9":             self.bb9,
            "hr9":             self.hr9,
            "innings_pitched": self.innings_pitched,
            "era_recent":      self.era_recent,
            "hand":            self.hand,
        }


@dataclass(frozen=True)
class BattingStatcast:
    """
    Estadísticas avanzadas de bateo para un equipo en una temporada.

    Campos
    ------
    team_id     -- ID del equipo en MLB Stats API.
    season      -- Temporada (año).
    ops         -- On-base Plus Slugging (temporada).
    obp         -- On-base Percentage.
    slg         -- Slugging Percentage.
    avg         -- Batting Average.
    wrc_plus    -- wRC+ (normalizado a liga, 100=promedio). None si no disponible.
    xwoba       -- Expected wOBA (Baseball Savant). None por ahora.
    hard_hit_pct -- % de batazos >= 95 mph. None por ahora.
    ops_vs_rhp  -- OPS específico vs pitchers derechos.
    ops_vs_lhp  -- OPS específico vs pitchers zurdos.
    """
    team_id:      int
    season:       int
    ops:          float | None = None
    obp:          float | None = None
    slg:          float | None = None
    avg:          float | None = None
    wrc_plus:     float | None = None
    xwoba:        float | None = None
    hard_hit_pct: float | None = None
    ops_vs_rhp:   float | None = None
    ops_vs_lhp:   float | None = None

    def ops_vs_hand(self, hand: str | None) -> float | None:
        """
        Retorna el OPS específico contra la mano del abridor rival.

        Parámetros
        ----------
        hand  -- 'R' o 'L'. None → retorna ops general.
        """
        if hand == 'R' and self.ops_vs_rhp is not None:
            return self.ops_vs_rhp
        if hand == 'L' and self.ops_vs_lhp is not None:
            return self.ops_vs_lhp
        return self.ops

    def offense_index(self) -> float:
        """
        Índice de ofensiva normalizado a la liga.
        ops / OPS_LIGA. 1.0 = promedio de liga. > 1 = mejor que promedio.
        """
        if self.ops is None or self.ops <= 0:
            return 1.0
        return round(self.ops / _LEAGUE_OPS, 4)

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "ops":          self.ops,
            "obp":          self.obp,
            "slg":          self.slg,
            "avg":          self.avg,
            "wrc_plus":     self.wrc_plus,
            "ops_vs_rhp":   self.ops_vs_rhp,
            "ops_vs_lhp":   self.ops_vs_lhp,
        }


# ── Fórmulas derivadas ────────────────────────────────────────────────────────

def fip_from_components(
    hr:  float,
    bb:  float,
    hbp: float,
    k:   float,
    ip:  float,
    c_fip: float = _C_FIP,
) -> float | None:
    """
    Calcula FIP desde sus componentes individuales.

    Fórmula exacta (Tango, 2003):
        FIP = (13×HR + 3×(BB+HBP) - 2×K) / IP + cFIP

    Donde cFIP ≈ 3.10 es la constante de liga que ajusta FIP
    a la misma escala que ERA. Se recalcula cada temporada por
    Baseball Reference; 3.10 es el valor 2024-2026.

    Retorna None si IP <= 0 (división por cero).

    Parámetros
    ----------
    hr    -- Home runs permitidos.
    bb    -- Walks (bases por bolas).
    hbp   -- Hit by pitch.
    k     -- Strikeouts registrados.
    ip    -- Innings lanzados.
    c_fip -- Constante de liga. Default 3.10 (2024-2026).
    """
    if ip <= 0:
        return None
    return round((13 * hr + 3 * (bb + hbp) - 2 * k) / ip + c_fip, 3)


def defense_index_from_era(era: float | None) -> float:
    """
    Índice de defensa normalizado a la liga desde ERA del abridor.

    defense_index < 1.0 = mejor que la liga (menos carreras permitidas).
    defense_index > 1.0 = peor que la liga.

    Usa ERA del abridor como proxy principal de la defensa del equipo
    pitching — simplificación válida porque el abridor determina ~60%
    de las carreras permitidas en un partido MLB moderno.
    """
    if era is None or era <= 0:
        return 1.0
    return round(_LEAGUE_ERA / era, 4)


# ── Fetcher principal ─────────────────────────────────────────────────────────

class StatcastFetcher:
    """
    Obtiene estadísticas avanzadas de pitching y bateo desde MLB Stats API.

    Parámetros
    ----------
    config_loader  -- ConfigLoader con mlb.yaml cargado. Usado para
                     leer park_factors y parámetros de proyección.
    cache_dir      -- Directorio para caché JSON. Default 'output/cache'.
    cache_ttl_days -- TTL del caché en días. Default 1 (invalidar diario).
                     0 = sin caché (siempre fetch en vivo).
    timeout        -- Timeout HTTP en segundos. Default 10.
    """

    def __init__(
        self,
        config_loader=None,
        cache_dir:      str = "output/cache",
        cache_ttl_days: int = 1,
        timeout:        int = 10,
    ) -> None:
        self._config       = config_loader
        self._cache_dir    = cache_dir
        self._cache_ttl    = cache_ttl_days
        self._timeout      = timeout

    # ── Pitcher stats ─────────────────────────────────────────────────────────

    def fetch_pitcher_stats(
        self,
        pitcher_id: int,
        season:     int | None = None,
    ) -> PitcherStatcast:
        """
        Obtiene estadísticas de temporada del abridor desde MLB Stats API.

        Parámetros
        ----------
        pitcher_id  -- ID del jugador en MLB Stats API.
        season      -- Temporada. Default: temporada actual.

        Retorna
        -------
        PitcherStatcast con todos los campos disponibles.
        Si la API falla o el pitcher no tiene datos, retorna un
        PitcherStatcast con todos los campos numéricos en None.
        """
        season = season or _current_season()
        cache_key = f"pitcher_{pitcher_id}_{season}"

        cached = self._load_cache(cache_key)
        if cached is not None:
            return self._parse_pitcher(cached, pitcher_id, season)

        raw = self._fetch_player_stats(pitcher_id, season, group="pitching")
        if raw is None:
            return PitcherStatcast(pitcher_id=pitcher_id, season=season)

        self._save_cache(cache_key, raw)
        return self._parse_pitcher(raw, pitcher_id, season)

    def fetch_pitcher_recent(
        self,
        pitcher_id: int,
        days:       int = 30,
    ) -> float | None:
        """
        ERA del pitcher en los últimos N días (forma reciente).

        Más predictivo que la temporada completa para partidos próximos.
        Retorna None si no hay suficiente muestra reciente (< 15 IP).
        """
        season = _current_season()
        raw    = self._fetch_player_stats(
            pitcher_id, season, group="pitching",
            stats_type="lastXDays", days=days
        )
        if raw is None:
            return None

        stats = _extract_stats(raw)
        ip    = _safe_float(stats.get("inningsPitched"))
        if ip is None or ip < 15:
            return None

        era_str = stats.get("era")
        return _safe_float(era_str)

    # ── Batting stats ─────────────────────────────────────────────────────────

    def fetch_batter_stats(
        self,
        team_id: int,
        season:  int | None = None,
    ) -> BattingStatcast:
        """
        Obtiene estadísticas de bateo del equipo desde MLB Stats API.

        Incluye splits vsRHP/vsLHP si están disponibles.

        Parámetros
        ----------
        team_id  -- ID del equipo en MLB Stats API.
        season   -- Temporada. Default: actual.
        """
        season    = season or _current_season()
        cache_key = f"batting_{team_id}_{season}"

        cached = self._load_cache(cache_key)
        if cached is not None:
            return self._parse_batting(cached, team_id, season)

        raw = self._fetch_team_stats(team_id, season, group="hitting")
        if raw is None:
            return BattingStatcast(team_id=team_id, season=season)

        self._save_cache(cache_key, raw)
        return self._parse_batting(raw, team_id, season)

    # ── Park factors ──────────────────────────────────────────────────────────

    def fetch_park_factor(self, venue_id: str) -> float:
        """
        Retorna el park factor para el estadio dado.

        Lee desde config/mlb.yaml bajo mlb.park_factors.{venue_id}.
        Default 1.00 si no está configurado (estadio neutral).

        Valores de referencia:
            Coors Field (col):        1.35  (más carreras)
            Petco Park (sd):          0.91  (menos carreras)
            Wrigley Field (chc):      1.04  (ligeramente más)
            Yankee Stadium (nyy):     1.07
            Fenway Park (bos):        1.05
        """
        if self._config is None:
            return 1.00
        return float(
            self._config.get(
                f"mlb.park_factors.{venue_id.lower()}",
                default=self._config.get("mlb.park_factor_default", default=1.00),
            )
        )

    # ── Parsing ───────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_pitcher(
        raw: dict,
        pitcher_id: int,
        season: int,
    ) -> PitcherStatcast:
        """Convierte la respuesta raw de MLB Stats API a PitcherStatcast."""
        stats = _extract_stats(raw)
        ip    = _safe_float(stats.get("inningsPitched"))

        # Calcular FIP si tenemos los componentes.
        #
        # CORRECCIÓN DE TIPADO (auditoría 2026-08): `all(x is not None
        # for x in [...])` es seguro en runtime pero el type checker no
        # puede propagar ese narrowing a las variables individuales
        # (mismo patrón de falso positivo ya documentado en
        # core/bankroll/tracker.py y core/backtesting/engine.py para
        # list comprehensions). Con chequeos individuales encadenados
        # por `and`, pyright sí estrecha cada variable a `float`.
        fip = None
        hr  = _safe_float(stats.get("homeRuns"))
        bb  = _safe_float(stats.get("baseOnBalls"))
        hbp = _safe_float(stats.get("hitBatsmen"), default=0.0)
        k   = _safe_float(stats.get("strikeOuts"))
        if (
            hr is not None and bb is not None
            and hbp is not None and k is not None
            and ip and ip > 0
        ):
            fip = fip_from_components(hr, bb, hbp, k, ip)

        # K/9 y BB/9
        k9  = round(k  * 9 / ip, 3) if k  is not None and ip else None
        bb9 = round(bb * 9 / ip, 3) if bb is not None and ip else None
        hr9 = round(hr * 9 / ip, 3) if hr is not None and ip else None

        # Mano del pitcher desde info del jugador
        hand = raw.get("hand") or raw.get("pitchHand", {}).get("code")

        return PitcherStatcast(
            pitcher_id      = pitcher_id,
            season          = season,
            era             = _safe_float(stats.get("era")),
            fip             = fip,
            whip            = _safe_float(stats.get("whip")),
            k9              = k9,
            bb9             = bb9,
            hr9             = hr9,
            innings_pitched = ip,
            hand            = hand,
        )

    @staticmethod
    def _parse_batting(
        raw: dict,
        team_id: int,
        season: int,
    ) -> BattingStatcast:
        """Convierte la respuesta raw de MLB Stats API a BattingStatcast."""
        stats     = _extract_stats(raw)
        splits    = raw.get("splits", {})

        ops_vs_rhp = _safe_float(splits.get("opsVsRHP"))
        ops_vs_lhp = _safe_float(splits.get("opsVsLHP"))

        return BattingStatcast(
            team_id    = team_id,
            season     = season,
            ops        = _safe_float(stats.get("ops")),
            obp        = _safe_float(stats.get("obp")),
            slg        = _safe_float(stats.get("slg")),
            avg        = _safe_float(stats.get("avg")),
            ops_vs_rhp = ops_vs_rhp,
            ops_vs_lhp = ops_vs_lhp,
        )

    # ── HTTP helpers ──────────────────────────────────────────────────────────

    def _fetch_player_stats(
        self,
        player_id:  int,
        season:     int,
        group:      str = "pitching",
        stats_type: str = "season",
        days:       int = 30,
    ) -> dict | None:
        """Fetch de estadísticas de jugador desde MLB Stats API."""
        if not _REQUESTS_AVAILABLE or _requests is None:
            return None

        params: dict = {
            "hydrate": f"stats(group=[{group}],type=[{stats_type}],season={season})",
        }
        if stats_type == "lastXDays":
            params["hydrate"] = (
                f"stats(group=[{group}],type=[lastXDays],season={season},days={days})"
            )

        url = f"{_MLB_API_BASE}/people/{player_id}"
        try:
            resp = _requests.get(url, params=params, timeout=self._timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
            people = data.get("people", [])
            if not people:
                return None
            person = people[0]
            # Extraer hand info
            hand_code = person.get("pitchHand", {}).get("code")
            stats_list = person.get("stats", [])
            result = {"hand": hand_code, "stats_list": stats_list}
            return result
        except Exception:
            return None

    def _fetch_team_stats(
        self,
        team_id: int,
        season:  int,
        group:   str = "hitting",
    ) -> dict | None:
        """Fetch de estadísticas del equipo desde MLB Stats API."""
        if not _REQUESTS_AVAILABLE or _requests is None:
            return None

        url    = f"{_MLB_API_BASE}/teams/{team_id}/stats"
        params = {
            "stats":  "season",
            "group":  group,
            "season": season,
        }
        try:
            resp = _requests.get(url, params=params, timeout=self._timeout)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    # ── Caché ─────────────────────────────────────────────────────────────────

    def _cache_path(self, key: str) -> str:
        return os.path.join(self._cache_dir, f"{key}.json")

    def _load_cache(self, key: str) -> dict | None:
        """Carga caché si existe y no ha expirado."""
        if self._cache_ttl == 0:
            return None
        path = self._cache_path(key)
        if not os.path.exists(path):
            return None
        try:
            mtime = os.path.getmtime(path)
            age_days = (datetime.now(timezone.utc).timestamp() - mtime) / 86400
            if age_days > self._cache_ttl:
                return None
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

    def _save_cache(self, key: str, data: dict) -> None:
        """Guarda datos en caché JSON."""
        os.makedirs(self._cache_dir, exist_ok=True)
        try:
            with open(self._cache_path(key), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass  # Caché es opcional — no abortar si falla


# ── Utilidades ────────────────────────────────────────────────────────────────

def _current_season() -> int:
    """Temporada actual MLB (el año del calendario)."""
    return datetime.now(timezone.utc).year


def _safe_float(value: Any, default: float | None = None) -> float | None:
    """Convierte un valor de la API a float de forma segura."""
    if value is None:
        return default
    try:
        result = float(value)
        return None if result != result else result  # NaN check
    except (ValueError, TypeError):
        return default


def _extract_stats(raw: dict) -> dict:
    """
    Extrae el dict de estadísticas de la respuesta raw de MLB Stats API.

    La estructura de la API es:
        {stats_list: [{group: {displayName: 'pitching'}, splits: [{stat: {...}}]}]}
    """
    if raw is None:
        return {}
    stats_list = raw.get("stats_list", [])
    if not stats_list:
        # También puede venir como respuesta directa de team stats
        stats_groups = raw.get("stats", [])
        if stats_groups:
            splits = stats_groups[0].get("splits", [])
            if splits:
                return splits[0].get("stat", {})
        return {}
    for stat_group in stats_list:
        splits = stat_group.get("splits", [])
        if splits:
            return splits[0].get("stat", {})
    return {}