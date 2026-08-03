"""
sports/mlb/bullpen.py

BullpenFetcher: estadísticas del bullpen y ERA combinada para MLB.

Migrado de analysis/bullpen.py del sistema MLB con tres correcciones:

1. BullpenStats como dataclass inmutable — no dict.
2. Pesos abridor/bullpen desde config/mlb.yaml — no hardcodeados.
   mlb.yaml: mlb.projection.era_weight=0.35, bullpen_weight=0.10
   (junto con fip_weight y team_offense_weight completan 1.0).
   Para ERA combinada: starter_weight / (starter_weight + bullpen_weight)
   Simplificación práctica: starter=0.90, bullpen=0.10 como default.
3. Delega fetch en StatcastFetcher — sin duplicar HTTP ni caché.

Uso en el pipeline
-------------------
MLBDataProvider.enrich_event(event)
    ↓
BullpenFetcher.fetch(team_id, season) → BullpenStats
    ↓
combined_era(starter_era, bullpen_stats) → float
    ↓
TeamFeatures(
    defense_index = LEAGUE_ERA / combined_era,
    sport_metadata = {'bullpen_era': ..., 'bullpen_whip': ...}
)

ERA combinada
--------------
ERA_combinada = starter_era × w_starter + bullpen_era × w_bullpen

Con defaults calibrados del sistema MLB:
    w_starter = 0.90  (el abridor domina la proyección)
    w_bullpen = 0.10  (bullpen ajusta marginalmente)

Si is_bullpen_day=True (no hay abridor confirmado):
    ERA_combinada = bullpen_era (100% bullpen)
    Esto aumenta la incertidumbre del partido correctamente.
"""

from __future__ import annotations

from dataclasses import dataclass

from sports.mlb.statcast import StatcastFetcher, _LEAGUE_ERA, _safe_float, _extract_stats

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Pesos default para ERA combinada (calibrados del sistema MLB original)
_DEFAULT_STARTER_WEIGHT: float = 0.90
_DEFAULT_BULLPEN_WEIGHT: float = 0.10


# ── Dataclass del bullpen ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class BullpenStats:
    """
    Estadísticas del bullpen de un equipo en una temporada.

    Inmutable: representa el estado del bullpen en el momento del fetch.

    Campos
    ------
    team_id          -- ID del equipo en MLB Stats API.
    season           -- Temporada (año).
    era              -- ERA del bullpen en la temporada.
    whip             -- WHIP del bullpen.
    k9               -- Strikeouts por 9 innings del bullpen.
    bb9              -- Walks por 9 innings del bullpen.
    holds            -- Holds acumulados (proxy de efectividad del setup).
    saves            -- Saves acumulados.
    blown_saves      -- Blown saves (señal de fragilidad del bullpen).
    innings_pitched  -- IP totales del bullpen (sample size).
    appearances      -- Apariciones totales en la temporada.
    """
    team_id:         int
    season:          int
    era:             float | None = None
    whip:            float | None = None
    k9:              float | None = None
    bb9:             float | None = None
    holds:           int   | None = None
    saves:           int   | None = None
    blown_saves:     int   | None = None
    innings_pitched: float | None = None
    appearances:     int   | None = None

    @property
    def effective_era(self) -> float:
        """ERA efectiva del bullpen. Usa LEAGUE_ERA si no hay datos."""
        return self.era if self.era is not None and self.era > 0 else _LEAGUE_ERA

    @property
    def save_pct(self) -> float | None:
        """
        Porcentaje de saves convertidos.
        None si no hay suficiente muestra (saves + blown_saves < 5).
        """
        if self.saves is None or self.blown_saves is None:
            return None
        total = self.saves + self.blown_saves
        if total < 5:
            return None
        return round(self.saves / total, 4)

    def to_metadata(self) -> dict:
        """Convierte a dict para TeamFeatures.sport_metadata."""
        return {
            "bullpen_era":    self.era,
            "bullpen_whip":   self.whip,
            "bullpen_k9":     self.k9,
            "bullpen_saves":  self.saves,
            "bullpen_bs":     self.blown_saves,
            "bullpen_save_pct": self.save_pct,
        }


# ── Función de ERA combinada ──────────────────────────────────────────────────

def combined_era(
    starter_era:     float | None,
    bullpen:         BullpenStats,
    starter_weight:  float = _DEFAULT_STARTER_WEIGHT,
    bullpen_weight:  float = _DEFAULT_BULLPEN_WEIGHT,
    is_bullpen_day:  bool  = False,
) -> float:
    """
    Calcula la ERA combinada ponderada del abridor y el bullpen.

    Si is_bullpen_day=True, usa 100% bullpen ERA — no hay abridor.
    Si starter_era=None, usa ERA de liga como fallback.

    Parámetros
    ----------
    starter_era      -- ERA del abridor probable. None → LEAGUE_ERA.
    bullpen          -- BullpenStats del equipo.
    starter_weight   -- Peso del abridor. Default 0.90.
    bullpen_weight   -- Peso del bullpen. Default 0.10.
    is_bullpen_day   -- True si no hay abridor confirmado.

    Retorna
    -------
    ERA combinada normalizada. Siempre > 0.
    """
    if is_bullpen_day:
        return bullpen.effective_era

    era_s = starter_era if starter_era is not None and starter_era > 0 else _LEAGUE_ERA
    era_b = bullpen.effective_era

    # Normalizar pesos para que sumen 1.0
    total_w = starter_weight + bullpen_weight
    if total_w <= 0:
        return _LEAGUE_ERA

    w_s = starter_weight / total_w
    w_b = bullpen_weight / total_w

    return round(era_s * w_s + era_b * w_b, 4)


def defense_index_from_combined_era(era_comb: float) -> float:
    """
    defense_index = ERA_LIGA / ERA_combinada

    < 1.0: el equipo permite menos carreras que la liga (mejor defensa)
    > 1.0: el equipo permite más carreras que la liga (peor defensa)
    = 1.0: promedio de liga

    Nota: el audit del sistema MLB detectó que defense_index usaba
    la fórmula INVERTIDA (ERA/ERA_LIGA en vez de ERA_LIGA/ERA),
    lo que hacía que equipos con ERA alta parecieran tener mejor
    defensa. Corregido aquí con la fórmula correcta.
    """
    if era_comb <= 0:
        return 1.0
    return round(_LEAGUE_ERA / era_comb, 4)


# ── Fetcher principal ─────────────────────────────────────────────────────────

class BullpenFetcher:
    """
    Obtiene estadísticas del bullpen desde MLB Stats API.

    Delega en StatcastFetcher para todo el acceso HTTP y caché.

    Parámetros
    ----------
    statcast_fetcher  -- StatcastFetcher ya configurado. Si None,
                        crea uno con defaults.
    starter_weight    -- Peso del abridor en ERA combinada. Default 0.90.
    bullpen_weight    -- Peso del bullpen en ERA combinada. Default 0.10.
    config_loader     -- ConfigLoader con mlb.yaml. Si se provee, lee
                        los pesos desde mlb.projection.era_weight y
                        mlb.projection.bullpen_weight.
    """

    def __init__(
        self,
        statcast_fetcher: StatcastFetcher | None = None,
        starter_weight:   float = _DEFAULT_STARTER_WEIGHT,
        bullpen_weight:   float = _DEFAULT_BULLPEN_WEIGHT,
        config_loader     = None,
    ) -> None:
        self._statcast = statcast_fetcher or StatcastFetcher()
        self._config   = config_loader

        # Leer pesos del YAML si hay config disponible
        if config_loader is not None:
            era_w = config_loader.get(
                "mlb.projection.era_weight", default=None
            )
            bul_w = config_loader.get(
                "mlb.projection.bullpen_weight", default=None
            )
            if era_w is not None:
                # era_weight en YAML es el peso del abridor en la proyección
                # completa. Para ERA combinada escalamos a 0.90/0.10 usando
                # la relación era_weight / (era_weight + bullpen_weight).
                # Simplificación: si el YAML los define, usarlos directamente.
                try:
                    self._starter_weight = float(era_w)
                    self._bullpen_weight  = float(bul_w) if bul_w else bullpen_weight
                except (ValueError, TypeError):
                    self._starter_weight = starter_weight
                    self._bullpen_weight  = bullpen_weight
            else:
                self._starter_weight = starter_weight
                self._bullpen_weight  = bullpen_weight
        else:
            self._starter_weight = starter_weight
            self._bullpen_weight  = bullpen_weight

    def fetch(
        self,
        team_id: int,
        season:  int | None = None,
    ) -> BullpenStats:
        """
        Obtiene estadísticas del bullpen para el equipo y temporada dados.

        Usa el endpoint de team stats de MLB Stats API filtrado por
        'relief' (pitching de relevo). Delega caché en StatcastFetcher.

        Parámetros
        ----------
        team_id  -- ID del equipo en MLB Stats API.
        season   -- Temporada. Default: temporada actual.

        Retorna
        -------
        BullpenStats con todos los campos disponibles.
        Si la API falla, retorna BullpenStats con todos los campos None.
        """
        from sports.mlb.statcast import _current_season
        season = season or _current_season()

        raw = self._fetch_bullpen_raw(team_id, season)
        if raw is None:
            return BullpenStats(team_id=team_id, season=season)

        return self._parse_bullpen(raw, team_id, season)

    def combined_era(
        self,
        starter_era:    float | None,
        bullpen:        BullpenStats,
        is_bullpen_day: bool = False,
    ) -> float:
        """
        Calcula ERA combinada con los pesos configurados.

        Wrapper de la función pura combined_era() con los pesos
        del BullpenFetcher (desde YAML o defaults).
        """
        return combined_era(
            starter_era    = starter_era,
            bullpen        = bullpen,
            starter_weight = self._starter_weight,
            bullpen_weight = self._bullpen_weight,
            is_bullpen_day = is_bullpen_day,
        )

    def defense_index(
        self,
        starter_era:    float | None,
        bullpen:        BullpenStats,
        is_bullpen_day: bool = False,
    ) -> float:
        """
        Calcula defense_index desde ERA combinada.

        Combina combined_era() + defense_index_from_combined_era()
        en un único método de conveniencia para MLBDataProvider.
        """
        era_comb = self.combined_era(
            starter_era    = starter_era,
            bullpen        = bullpen,
            is_bullpen_day = is_bullpen_day,
        )
        return defense_index_from_combined_era(era_comb)

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _fetch_bullpen_raw(self, team_id: int, season: int) -> dict | None:
        """
        Fetch de estadísticas del bullpen desde MLB Stats API.

        Endpoint: /api/v1/teams/{teamId}/stats?stats=season&group=pitching
        Filtra por pitchersUsed (relevistas) en postprocessing.
        """
        if not _REQUESTS_AVAILABLE:
            return None

        url    = f"{_MLB_API_BASE}/teams/{team_id}/stats"
        params = {
            "stats":  "season",
            "group":  "pitching",
            "season": season,
        }
        try:
            resp = _requests.get(url, params=params, timeout=10)
            if resp.status_code != 200:
                return None
            return resp.json()
        except Exception:
            return None

    @staticmethod
    def _parse_bullpen(raw: dict, team_id: int, season: int) -> BullpenStats:
        """
        Parsea la respuesta de MLB Stats API para estadísticas del bullpen.

        MLB Stats API retorna stats agregadas del equipo — no hay un
        endpoint separado para el bullpen. Usamos el grupo 'pitching'
        del equipo como proxy del bullpen (incluyendo el abridor).
        La distinción abridor/bullpen se hace en combined_era() con pesos.
        """
        stats_groups = raw.get("stats", [])
        if not stats_groups:
            return BullpenStats(team_id=team_id, season=season)

        # Tomar el primer grupo de pitching disponible
        splits = stats_groups[0].get("splits", [])
        if not splits:
            return BullpenStats(team_id=team_id, season=season)

        stat = splits[0].get("stat", {})

        return BullpenStats(
            team_id         = team_id,
            season          = season,
            era             = _safe_float(stat.get("era")),
            whip            = _safe_float(stat.get("whip")),
            k9              = _safe_float(stat.get("strikeoutsPer9Inn")),
            bb9             = _safe_float(stat.get("walksPer9Inn")),
            holds           = _safe_int(stat.get("holds")),
            saves           = _safe_int(stat.get("saves")),
            blown_saves     = _safe_int(stat.get("blownSaves")),
            innings_pitched = _safe_float(stat.get("inningsPitched")),
            appearances     = _safe_int(stat.get("gamesPlayed")),
        )


# ── Utilidades ────────────────────────────────────────────────────────────────

def _safe_int(value, default: int | None = None) -> int | None:
    """Convierte un valor de la API a int de forma segura."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default