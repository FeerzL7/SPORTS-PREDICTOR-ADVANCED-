"""
sports/mlb/settlement.py

MLBSettlementProvider: liquidación de picks MLB por mercado.

Implementa dos protocolos simultáneamente:

1. core/pipeline/stage.py:SettlementProvider (versión "deportiva")
   - get_event_result(event) → dict | None
   - settle_pick(pick, event_result) → str

2. core/tracking/protocols.py:SettlementProvider (versión "financiera")
   - get_result(entry: BetLedgerEntry) → SettlementResult | None
   - get_closing_price(entry) → float | None

Reglas de settlement MLB por mercado
--------------------------------------
ML (Moneyline):
    home_score > away_score → local gana
    away_score > home_score → visitante gana
    Sin empate posible en MLB (extra innings hasta definición)
    Partido suspendido sin 5 innings → void

SPREAD (Runline, siempre ±1.5):
    Si selección es el favorito (-1.5):
        win  si margen_victoria > 1.5 (ganó por 2+ carreras)
        lose si margen_victoria <= 1.5 (ganó por 1 o perdió)
    Si selección es el underdog (+1.5):
        win  si perdió por 0 o ganó (margen > -1.5)
        lose si perdió por 2+ carreras (margen <= -1.5)
    Push imposible con línea .5

TOTAL (Over/Under):
    total_carreras > línea → over gana
    total_carreras < línea → under gana
    total_carreras == línea → push (solo posible con líneas enteras)

F5 (First 5 innings):
    Misma lógica que ML/TOTAL pero solo si el partido completó 5 innings.
    Usa linescore de la API para obtener el score al final del 5to inning.
    Si el partido se suspendió antes del 5to → void

Reglas especiales
------------------
- Partidos suspendidos por lluvia < 5 innings → void en todos los mercados
- Partidos suspendidos ≥ 5 innings → resultado oficial (se liquida normalmente)
- Extra innings → resultado oficial (no se anula)
- Double-header (juego 2 de 7 innings) → se liquida normalmente
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from core.contracts.event import Event
from core.contracts.ledger import BetLedgerEntry
from core.contracts.pick import CandidatePick
from core.tracking.protocols import SettlementResult

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

_MLB_API_BASE    = "https://statsapi.mlb.com/api/v1"
_MLB_API_BASE_11 = "https://statsapi.mlb.com/api/v1.1"

# Innings mínimos para que un partido tenga resultado oficial en MLB
_MIN_INNINGS_OFFICIAL: float = 5.0

# Runline estándar MLB (siempre ±1.5)
_MLB_RUNLINE: float = 1.5


# ── Resultado de partido ──────────────────────────────────────────────────────

@dataclass(frozen=True)
class MLBGameResult:
    """
    Resultado completo de un partido MLB.

    Inmutable: representa el resultado oficial del partido.

    Campos
    ------
    game_pk       -- ID del partido en MLB Stats API.
    status        -- 'final', 'suspended', 'postponed', 'in_progress'.
    home_score    -- Carreras del equipo local (juego completo).
    away_score    -- Carreras del equipo visitante.
    innings       -- Innings completados.
    is_official   -- True si el partido tiene resultado oficial
                   (≥ 5 innings completados o partido finalizado).
    home_team_id  -- ID del equipo local.
    away_team_id  -- ID del equipo visitante.
    f5_home_score -- Carreras del local al final del 5to inning.
                   None si el partido no llegó al 5to.
    f5_away_score -- Carreras del visitante al final del 5to inning.
    """
    game_pk:       int
    status:        str
    home_score:    float
    away_score:    float
    innings:       float
    is_official:   bool
    home_team_id:  int
    away_team_id:  int
    f5_home_score: float | None = None
    f5_away_score: float | None = None

    @property
    def total(self) -> float:
        """Total de carreras del partido."""
        return self.home_score + self.away_score

    @property
    def margin(self) -> float:
        """Diferencia: home_score - away_score."""
        return self.home_score - self.away_score

    @property
    def f5_total(self) -> float | None:
        """Total de carreras de los primeros 5 innings."""
        if self.f5_home_score is None or self.f5_away_score is None:
            return None
        return self.f5_home_score + self.f5_away_score

    def to_dict(self) -> dict:
        """Serializa a dict para stage.py:SettlementProvider.get_event_result()."""
        return {
            "game_pk":       self.game_pk,
            "status":        self.status,
            "home_score":    self.home_score,
            "away_score":    self.away_score,
            "innings":       self.innings,
            "is_official":   self.is_official,
            "home_team_id":  self.home_team_id,
            "away_team_id":  self.away_team_id,
            "f5_home_score": self.f5_home_score,
            "f5_away_score": self.f5_away_score,
        }

    @classmethod
    def from_dict(cls, d: dict) -> MLBGameResult:
        """Reconstruye desde dict (para usar en settle_pick)."""
        return cls(
            game_pk       = d.get("game_pk", 0),
            status        = d.get("status", "unknown"),
            home_score    = float(d.get("home_score", 0)),
            away_score    = float(d.get("away_score", 0)),
            innings       = float(d.get("innings", 0)),
            is_official   = bool(d.get("is_official", False)),
            home_team_id  = int(d.get("home_team_id", 0)),
            away_team_id  = int(d.get("away_team_id", 0)),
            f5_home_score = d.get("f5_home_score"),
            f5_away_score = d.get("f5_away_score"),
        )


# ── Proveedor principal ───────────────────────────────────────────────────────

class MLBSettlementProvider:
    """
    Liquida picks MLB determinando el resultado desde MLB Stats API.

    Implementa simultáneamente:
        - core/pipeline/stage.py:SettlementProvider (deportivo)
        - core/tracking/protocols.py:SettlementProvider (financiero)

    Parámetros
    ----------
    timeout          -- Timeout HTTP. Default 10s.
    closing_prices   -- Dict {entry_id: closing_price} para CLV.
                       El pipeline lo alimenta desde LineMovementDetector
                       antes de llamar settle. None = sin CLV tracking.
    """

    def __init__(
        self,
        timeout:        int  = 10,
        closing_prices: dict | None = None,
    ) -> None:
        self._timeout        = timeout
        self._closing_prices = closing_prices or {}

    # ── SettlementProvider (deportivo) ────────────────────────────────────────

    def get_event_result(self, event: Event) -> dict | None:
        """
        Obtiene el resultado del partido desde MLB Stats API.

        Retorna dict con MLBGameResult serializado, o None si:
        - El partido no ha finalizado aún
        - La API no está disponible
        - El game_pk no está en event.provider_ids
        """
        game_pk = self._extract_game_pk(event)
        if game_pk is None:
            return None

        result = self._fetch_game_result(game_pk)
        if result is None:
            return None

        # Solo retornar si el partido tiene resultado definitivo
        if result.status not in ("final", "suspended", "postponed"):
            return None

        return result.to_dict()

    def settle_pick(
        self,
        pick:         CandidatePick,
        event_result: dict,
    ) -> str:
        """
        Determina el resultado del pick dado el resultado del partido.

        Retorna: 'win', 'lose', 'null' (push), 'void', o 'pending'.
        """
        if not event_result:
            return "pending"

        game = MLBGameResult.from_dict(event_result)
        market = pick.market.upper()

        # Partido no oficial → void en todos los mercados
        if not game.is_official and game.status in ("suspended", "postponed"):
            return "void"

        if market == "ML":
            return self._settle_ml(pick, game)
        elif market in ("SPREAD", "RL"):
            return self._settle_spread(pick, game)
        elif market == "TOTAL":
            return self._settle_total(pick, game)
        elif market == "ML_F5":
            return self._settle_ml_f5(pick, game)
        elif market == "TOTAL_F5":
            return self._settle_total_f5(pick, game)
        else:
            # Mercado no soportado
            return "void"

    # ── SettlementProvider (financiero) ──────────────────────────────────────

    def get_result(self, entry: BetLedgerEntry) -> SettlementResult | None:
        """
        Determina el resultado de un BetLedgerEntry pendiente.

        Construye un Event mínimo desde entry para poder llamar a la API.
        Retorna None si el partido no ha terminado.
        """
        game_pk = self._extract_game_pk_from_entry(entry)
        if game_pk is None:
            return None

        game = self._fetch_game_result(game_pk)
        if game is None or game.status not in ("final", "suspended", "postponed"):
            return None

        # Construir un CandidatePick mínimo para reutilizar settle_pick
        result_str = self._settle_from_entry(entry, game)
        if result_str == "pending":
            return None

        closing = self._closing_prices.get(entry.entry_id)

        return SettlementResult(
            entry_id      = entry.entry_id,
            result        = result_str,
            settled_at    = datetime.now(timezone.utc).isoformat(),
            closing_price = closing,
            sport_context = game.to_dict(),
        )

    def get_closing_price(self, entry: BetLedgerEntry) -> float | None:
        """
        Retorna el precio de cierre registrado para este entry_id.

        Alimentado externamente por el pipeline desde LineMovementDetector
        antes de llamar a settle_pending().
        """
        return self._closing_prices.get(entry.entry_id)

    def register_closing_price(self, entry_id: str, price: float) -> None:
        """Registra el precio de cierre para CLV tracking."""
        self._closing_prices[entry_id] = price

    # ── Lógica de settlement por mercado ─────────────────────────────────────

    @staticmethod
    def _settle_ml(pick: CandidatePick, game: MLBGameResult) -> str:
        """
        Moneyline: ganador del partido completo.
        Sin empate posible en MLB (extra innings hasta definición).
        """
        if not game.is_official:
            return "void"

        # Determinar si la selección es home o away
        home_sel = _is_home_selection(pick, game)

        if home_sel is None:
            return "void"  # selección no reconocida

        home_won = game.home_score > game.away_score

        if home_sel:
            return "win" if home_won else "lose"
        else:
            return "win" if not home_won else "lose"

    @staticmethod
    def _settle_spread(pick: CandidatePick, game: MLBGameResult) -> str:
        """
        Runline (spread) MLB: siempre ±1.5.

        line negativa (-1.5) = favorito (debe ganar por 2+)
        line positiva (+1.5) = underdog (puede perder por 1 o ganar)
        Push imposible con línea .5
        """
        if not game.is_official:
            return "void"

        line = pick.line
        if line is None:
            line = -_MLB_RUNLINE  # default: favorito si no hay línea

        # Margen desde perspectiva de la selección
        home_sel = _is_home_selection(pick, game)
        if home_sel is None:
            return "void"

        # Margen del partido desde perspectiva de la selección
        if home_sel:
            effective_margin = game.margin  # home - away
        else:
            effective_margin = -game.margin  # away - home

        # La selección cubre si: effective_margin + line > 0
        # Con line=-1.5: necesita ganar por 2+ (effective_margin > 1.5)
        # Con line=+1.5: cubre si pierde por 1 o menos (effective_margin > -1.5)
        covers = effective_margin + line > 0

        return "win" if covers else "lose"

    @staticmethod
    def _settle_total(pick: CandidatePick, game: MLBGameResult) -> str:
        """
        Total Over/Under: suma de carreras vs la línea.

        Push posible solo con líneas enteras (ej: línea 9, total 9).
        """
        if not game.is_official:
            return "void"

        line = pick.line
        if line is None:
            return "void"

        total  = game.total
        sel    = pick.selection.lower()

        if total > line:
            return "win" if sel == "over" else "lose"
        elif total < line:
            return "win" if sel == "under" else "lose"
        else:
            return "null"  # push exacto

    @staticmethod
    def _settle_ml_f5(pick: CandidatePick, game: MLBGameResult) -> str:
        """
        ML primeros 5 innings.

        void si el partido no completó el 5to inning.
        """
        if game.f5_home_score is None or game.f5_away_score is None:
            return "void"

        home_sel = _is_home_selection(pick, game)
        if home_sel is None:
            return "void"

        h = game.f5_home_score
        a = game.f5_away_score

        if h == a:
            return "null"  # empate al final del 5to = push en F5

        home_won_f5 = h > a
        if home_sel:
            return "win" if home_won_f5 else "lose"
        else:
            return "win" if not home_won_f5 else "lose"

    @staticmethod
    def _settle_total_f5(pick: CandidatePick, game: MLBGameResult) -> str:
        """
        Total primeros 5 innings.

        void si el partido no completó el 5to inning.
        """
        f5_total = game.f5_total
        if f5_total is None:
            return "void"

        line  = pick.line
        if line is None:
            return "void"

        sel = pick.selection.lower()

        if f5_total > line:
            return "win" if sel == "over" else "lose"
        elif f5_total < line:
            return "win" if sel == "under" else "lose"
        else:
            return "null"

    # ── Settlement desde BetLedgerEntry ──────────────────────────────────────

    @staticmethod
    def _settle_from_entry(entry: BetLedgerEntry, game: MLBGameResult) -> str:
        """
        Determina resultado desde BetLedgerEntry (sin CandidatePick).

        Reconstruye la lógica de settlement solo con los campos del ledger:
        market, selection, line.
        """
        if not game.is_official and game.status in ("suspended", "postponed"):
            return "void"

        market = entry.market.upper()
        sel    = entry.selection.lower()

        if market == "ML":
            home_won = game.home_score > game.away_score
            # Determinar si la selección era el equipo local
            # desde el campo 'event' del ledger: "Away @ Home"
            home_won_sel = _selection_is_winner(sel, game, is_home=True)
            if home_won_sel is None:
                return "void"
            return "win" if home_won_sel == home_won else "lose"

        elif market in ("SPREAD", "RL"):
            line = entry.price  # La línea viene del campo price en este contexto
            # Aproximación: si price > 2.0 es el underdog (+1.5)
            line = 1.5 if entry.price > 2.0 else -1.5
            home_won  = game.home_score > game.away_score
            home_cover = game.margin > line if home_won else False
            is_home_sel = _selection_is_winner(sel, game, is_home=True)
            if is_home_sel is None:
                return "void"
            covers = (game.margin + line) > 0 if is_home_sel else (-game.margin + line) > 0
            return "win" if covers else "lose"

        elif market == "TOTAL":
            total  = game.total
            line   = _extract_line_from_selection(entry.selection)
            if line is None:
                return "void"
            if total > line:
                return "win" if "over" in sel else "lose"
            elif total < line:
                return "win" if "under" in sel else "lose"
            else:
                return "null"

        elif market == "ML_F5":
            # CORRECCIÓN (auditoría 2026-08): solo validaba
            # `f5_home_score is None`, dejando pasar el caso donde
            # `f5_away_score` sí es None (el partido no llegó al 5to
            # inning para el visitante en algún escenario de datos
            # parciales de la API) — `h > a` explotaba con
            # `TypeError: '>' not supported between 'float' and 'None'`.
            if game.f5_home_score is None or game.f5_away_score is None:
                return "void"
            h, a = game.f5_home_score, game.f5_away_score
            if h == a:
                return "null"
            return "win" if h > a else "lose"

        elif market == "TOTAL_F5":
            f5 = game.f5_total
            if f5 is None:
                return "void"
            line = _extract_line_from_selection(entry.selection)
            if line is None:
                return "void"
            if f5 > line:
                return "win" if "over" in sel else "lose"
            elif f5 < line:
                return "win" if "under" in sel else "lose"
            return "null"

        return "void"

    # ── Fetch desde MLB Stats API ─────────────────────────────────────────────

    def _fetch_game_result(self, game_pk: int) -> MLBGameResult | None:
        """
        Obtiene el resultado completo del partido desde MLB Stats API.

        Usa el endpoint live feed para obtener score y linescore completo.
        """
        if not _REQUESTS_AVAILABLE or _requests is None:
            return None

        url = f"{_MLB_API_BASE_11}/game/{game_pk}/feed/live"
        try:
            resp = _requests.get(url, timeout=self._timeout)
            if resp.status_code != 200:
                return None
            data = resp.json()
        except Exception:
            return None

        return self._parse_live_feed(game_pk, data)

    @staticmethod
    def _parse_live_feed(game_pk: int, data: dict) -> MLBGameResult | None:
        """Parsea la respuesta del live feed de MLB Stats API."""
        game_data = data.get("gameData", {})
        live_data = data.get("liveData", {})

        status_code = game_data.get("status", {}).get("abstractGameState", "")
        status = {
            "Final":       "final",
            "Live":        "in_progress",
            "Preview":     "scheduled",
            "Suspended":   "suspended",
            "Postponed":   "postponed",
        }.get(status_code, "unknown")

        # Equipos
        teams    = game_data.get("teams", {})
        home_id  = teams.get("home", {}).get("id", 0)
        away_id  = teams.get("away", {}).get("id", 0)

        # Score del partido completo
        linescore = live_data.get("linescore", {})
        home_score = float(linescore.get("teams", {}).get("home", {}).get("runs", 0))
        away_score = float(linescore.get("teams", {}).get("away", {}).get("runs", 0))
        innings    = float(linescore.get("currentInning", 0))
        inning_top = linescore.get("isTopInning", False)

        # Si estamos en el top de un inning, los innings completos son currentInning - 1
        completed_innings = innings if not inning_top else max(0, innings - 1)

        is_official = (
            status == "final" or
            (status == "suspended" and completed_innings >= _MIN_INNINGS_OFFICIAL)
        )

        # Score por inning para F5
        innings_data   = linescore.get("innings", [])
        f5_home, f5_away = _extract_f5_score(innings_data)

        return MLBGameResult(
            game_pk       = game_pk,
            status        = status,
            home_score    = home_score,
            away_score    = away_score,
            innings       = completed_innings,
            is_official   = is_official,
            home_team_id  = int(home_id),
            away_team_id  = int(away_id),
            f5_home_score = f5_home,
            f5_away_score = f5_away,
        )

    # ── Extracción de game_pk ─────────────────────────────────────────────────

    @staticmethod
    def _extract_game_pk(event: Event) -> int | None:
        """Extrae game_pk desde Event.provider_ids."""
        game_pk = event.provider_ids.get("mlb_game_pk") or \
                  event.provider_ids.get("game_pk")
        if game_pk is None:
            return None
        try:
            return int(game_pk)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _extract_game_pk_from_entry(entry: BetLedgerEntry) -> int | None:
        """
        Extrae game_pk desde BetLedgerEntry.

        El entry_id tiene formato: {event_id}_{market}_{selection}
        El event_id puede contener el game_pk si el pipeline lo incluyó.
        """
        # Intentar extraer desde event string "Away @ Home [game_pk=12345]"
        event_str = entry.event or ""
        if "game_pk=" in event_str:
            try:
                pk_str = event_str.split("game_pk=")[1].split("]")[0].strip()
                return int(pk_str)
            except (IndexError, ValueError):
                pass

        # Intentar desde entry_id (formato: {game_pk}_{market}_{selection})
        parts = entry.entry_id.split("_")
        if parts:
            try:
                return int(parts[0])
            except ValueError:
                pass

        return None


# ── Utilidades de módulo ──────────────────────────────────────────────────────

def _is_home_selection(
    pick: CandidatePick,
    game: MLBGameResult,
) -> bool | None:
    """
    Determina si la selección del pick corresponde al equipo local.

    Retorna True si es home, False si es away, None si no se puede determinar.
    """
    sel = pick.selection.lower().strip()

    # Para totales/props, la selección no es un equipo
    if sel in ("over", "under"):
        return None

    # Intentar matching por team_id en el event del pick
    event = pick.event
    if event:
        home_name = (event.home_team or "").lower()
        away_name = (event.away_team or "").lower()

        if sel in home_name or home_name in sel:
            return True
        if sel in away_name or away_name in sel:
            return False

    return None


def _selection_is_winner(
    sel: str,
    game: MLBGameResult,
    is_home: bool,
) -> bool | None:
    """Retorna si la selección ganó el partido."""
    home_won = game.home_score > game.away_score
    if is_home:
        return home_won
    return not home_won


def _extract_line_from_selection(selection: str) -> float | None:
    """
    Extrae la línea numérica de una selección como 'over 8.5' o 'under 9'.
    """
    parts = selection.lower().split()
    for part in parts:
        try:
            return float(part)
        except ValueError:
            continue
    return None


def _extract_f5_score(
    innings_data: list[dict],
) -> tuple[float | None, float | None]:
    """
    Extrae las carreras acumuladas al final del 5to inning.

    innings_data es la lista de innings del linescore.
    Cada inning tiene: num, home {runs}, away {runs}.
    """
    if len(innings_data) < 5:
        return None, None

    f5_home = sum(
        float(inn.get("home", {}).get("runs", 0) or 0)
        for inn in innings_data[:5]
    )
    f5_away = sum(
        float(inn.get("away", {}).get("runs", 0) or 0)
        for inn in innings_data[:5]
    )
    return f5_home, f5_away