"""
sports/nfl/settlement.py

NFLSettlementProvider: liquidación de picks NFL.

Implementa las DOS interfaces SettlementProvider del sistema:

    core/pipeline/stage.py       — "deportiva": score crudo y resolución
                                    del pick (Stage 10 del runner).
    core/tracking/protocols.py   — "financiera": recibe un
                                    BetLedgerEntry y produce un
                                    SettlementResult con CLV.

Es el mismo patrón que MLBSettlementProvider.

Tres reglas que el plugin MLB no necesitaba
---------------------------------------------

1. EMPATES
   La NFL permite empates tras un tiempo extra sin desempate (~0.4% de
   partidos). En MLB es imposible: se juegan entradas extra hasta que
   alguien gana.

   Tratamiento por mercado:
       ML      → PUSH. Las casas estadounidenses devuelven el stake en
                 moneyline cuando el partido termina empatado.
       SPREAD  → se resuelve normalmente; el empate solo importa si la
                 línea es 0 (pick'em), en cuyo caso también es push.
       TOTAL   → indiferente al empate; solo cuenta la suma.

2. PUSH EN SPREAD POR LÍNEAS ENTERAS
   Esta es la diferencia estructural. El runline de MLB es siempre
   ±1.5, así que el push es imposible por construcción. Las líneas de
   NFL se concentran en números enteros — y no por casualidad: 3 y 7
   son los valores de field goal y touchdown, los márgenes de victoria
   más frecuentes del deporte.

   Un -3 con victoria por exactamente 3 puntos devuelve el stake. Un
   modelo que lo contara como derrota subestimaría sistemáticamente el
   ROI de las apuestas a favoritos en números clave.

3. PUSH EN TOTAL POR LÍNEAS ENTERAS
   Mismo mecanismo. Los totales NFL enteros (44, 47) son habituales,
   a diferencia de MLB donde predominan los .5.

Tiempo extra
--------------
El OT cuenta para todos los mercados de partido completo. nflverse
expone el marcador final ya con OT incluido y una bandera `overtime`
que se conserva en el resultado para trazabilidad.

Mercados de primera mitad
---------------------------
Los mercados H1 (spreads_h1, totals_h1, h2h_h1) registrados en
MarketRegistry NO se pueden liquidar con los datos del schedule, que
solo trae el marcador final. Requerirían play-by-play filtrado por
cuarto. `settle_pick` los devuelve como 'void' de forma explícita en
vez de intentar una liquidación incorrecta con el marcador completo.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from core.contracts.event import Event
from core.contracts.ledger import BetLedgerEntry
from core.contracts.pick import CandidatePick
from core.tracking.protocols import SettlementResult

from sports.nfl.schedule import NFLGameInfo


# ── Dependencia inyectada ────────────────────────────────────────────────────

@runtime_checkable
class ScheduleSource(Protocol):
    """Interfaz mínima del calendario que este módulo consume."""

    def get_game_info(self, game_id: str) -> NFLGameInfo | None:
        """Metadatos y marcador de un partido, o None si no existe."""
        ...


# ── Resultados ───────────────────────────────────────────────────────────────

RESULT_WIN   = "win"
RESULT_LOSE  = "lose"
RESULT_PUSH  = "null"   # el Core usa 'null' para push/empate sin pérdida
RESULT_VOID  = "void"

# Estados del partido
STATUS_FINAL     = "final"
STATUS_SCHEDULED = "scheduled"
STATUS_UNKNOWN   = "unknown"

# Mercados de periodo que no se pueden liquidar desde el marcador final
_PERIOD_MARKETS = frozenset({
    "SPREAD_H1", "TOTAL_H1", "ML_H1",
    "spreads_h1", "totals_h1", "h2h_h1",
})

# Mercados de props de jugador: requieren box score individual
_PLAYER_PROP_PREFIXES = ("PLAYER_", "player_")


@dataclass(frozen=True)
class NFLGameResult:
    """
    Resultado de un partido NFL, normalizado para liquidación.

    Campos
    ------
    game_id     -- ID de nflverse.
    home_team   -- Abreviación del local.
    away_team   -- Abreviación del visitante.
    home_score  -- Puntos del local, OT incluido.
    away_score  -- Puntos del visitante, OT incluido.
    status      -- 'final', 'scheduled' o 'unknown'.
    overtime    -- True si el partido fue a tiempo extra.
    """
    game_id:    str
    home_team:  str
    away_team:  str
    home_score: float
    away_score: float
    status:     str = STATUS_FINAL
    overtime:   bool = False

    @property
    def is_final(self) -> bool:
        """True si el partido terminó y el marcador es definitivo."""
        return self.status == STATUS_FINAL

    @property
    def margin(self) -> float:
        """Margen desde la perspectiva del local (home - away)."""
        return self.home_score - self.away_score

    @property
    def total(self) -> float:
        """Suma de puntos del partido."""
        return self.home_score + self.away_score

    @property
    def is_tie(self) -> bool:
        """True si el partido terminó empatado tras OT."""
        return self.is_final and self.home_score == self.away_score

    def to_dict(self) -> dict:
        """Serializa para SettlementProvider.get_event_result()."""
        return {
            "game_id":    self.game_id,
            "home_team":  self.home_team,
            "away_team":  self.away_team,
            "home_score": self.home_score,
            "away_score": self.away_score,
            "status":     self.status,
            "overtime":   self.overtime,
            "margin":     self.margin,
            "total":      self.total,
            "is_tie":     self.is_tie,
        }

    @classmethod
    def from_dict(cls, data: dict) -> NFLGameResult | None:
        """Reconstruye desde el dict de get_event_result()."""
        if not isinstance(data, dict):
            return None
        home = _safe_float(data.get("home_score"))
        away = _safe_float(data.get("away_score"))
        if home is None or away is None:
            return None
        return cls(
            game_id    = str(data.get("game_id", "")),
            home_team  = str(data.get("home_team", "")),
            away_team  = str(data.get("away_team", "")),
            home_score = home,
            away_score = away,
            status     = str(data.get("status", STATUS_UNKNOWN)),
            overtime   = bool(data.get("overtime", False)),
        )


class NFLSettlementProvider:
    """
    Liquidación de picks NFL.

    Parámetros
    ----------
    schedule_fetcher -- Fuente del calendario con marcadores.
    closing_prices   -- Dict {entry_id: precio de cierre} para el CLV.
                        El pipeline lo alimenta desde
                        LineMovementDetector antes de liquidar.
    tie_voids_ml     -- Si True (por defecto), un empate anula el
                        moneyline devolviendo el stake. Es la práctica
                        estándar de las casas estadounidenses. Ponerlo
                        en False haría que el empate cuente como
                        derrota, que es la regla de algunos books
                        europeos con mercado a tres vías.
    """

    def __init__(
        self,
        schedule_fetcher: ScheduleSource,
        closing_prices:   dict | None = None,
        tie_voids_ml:     bool = True,
    ) -> None:
        self._schedule: ScheduleSource = schedule_fetcher
        self._closing_prices = closing_prices or {}
        self._tie_voids_ml   = tie_voids_ml

    # ── SettlementProvider "deportiva" ────────────────────────────────────────

    def get_event_result(self, event: Event) -> dict | None:
        """
        Resultado del partido, o None si aún no ha terminado.

        Retornar None (en vez de un dict con status='scheduled') es
        deliberado: el contrato del Core usa None para señalar
        "todavía no liquidable", y el runner salta el pick sin
        registrar un intento fallido.
        """
        game_id = self._extract_game_id(event)
        if not game_id:
            return None

        result = self._fetch_result(game_id)
        if result is None or not result.is_final:
            return None
        return result.to_dict()

    def settle_pick(self, pick: CandidatePick, event_result: dict) -> str:
        """
        Resuelve un pick contra el resultado del partido.

        Retorna 'win', 'lose', 'null' (push) o 'void'.

        Nunca lanza: ante datos incoherentes retorna 'void', que deja
        el stake intacto en el ledger. Anular es siempre preferible a
        registrar una ganancia o pérdida que no ocurrió.
        """
        game = NFLGameResult.from_dict(event_result)
        if game is None or not game.is_final:
            return RESULT_VOID

        market = (pick.market or "").upper()

        if market in _PERIOD_MARKETS or _is_player_prop(market):
            return RESULT_VOID

        selection = pick.selection or ""
        line = pick.line

        if market == "ML":
            return self._settle_moneyline(selection, game)
        if market == "SPREAD":
            return self._settle_spread(selection, line, game)
        if market == "TOTAL":
            return self._settle_total(selection, line, game)

        return RESULT_VOID

    # ── SettlementProvider "financiera" ───────────────────────────────────────

    def get_result(self, entry: BetLedgerEntry) -> SettlementResult | None:
        """
        Liquida un BetLedgerEntry y añade el CLV.

        Retorna None si el partido no ha terminado — el tracker deja
        el entry pendiente y reintentará en la siguiente ejecución.
        """
        game_id = self._extract_game_id_from_entry(entry)
        if not game_id:
            return None

        game = self._fetch_result(game_id)
        if game is None or not game.is_final:
            return None

        result = self._settle_from_entry(entry, game)

        return SettlementResult(
            entry_id      = entry.entry_id,
            result        = result,
            settled_at    = datetime.now(timezone.utc).isoformat(),
            closing_price = self.get_closing_price(entry),
            sport_context = game.to_dict(),
        )

    def get_closing_price(self, entry: BetLedgerEntry) -> float | None:
        """Precio de cierre registrado para este entry, si existe."""
        return self._closing_prices.get(entry.entry_id)

    def register_closing_price(self, entry_id: str, price: float) -> None:
        """Registra el precio de cierre para el cálculo de CLV."""
        self._closing_prices[entry_id] = price

    # ── Lógica por mercado ────────────────────────────────────────────────────

    def _settle_moneyline(self, selection: str, game: NFLGameResult) -> str:
        """
        Moneyline: gana quien anota más.

        El empate es PUSH por defecto. Las casas estadounidenses
        devuelven el stake en moneyline cuando el partido termina
        igualado tras OT — no existe un mercado a tres vías para NFL
        como sí lo hay en fútbol.
        """
        if game.is_tie:
            return RESULT_PUSH if self._tie_voids_ml else RESULT_LOSE

        side = _resolve_side(selection, game)
        if side is None:
            return RESULT_VOID

        home_won = game.margin > 0
        picked_home = side == "home"
        return RESULT_WIN if picked_home == home_won else RESULT_LOSE

    @staticmethod
    def _settle_spread(
        selection: str,
        line:      float | None,
        game:      NFLGameResult,
    ) -> str:
        """
        Spread: el margen ajustado por el handicap propio.

        `line` es el handicap de la SELECCIÓN, con la misma convención
        que MarketOdds.line y que NormalModel.spread_probability tras
        la corrección de la tarea 10.11: negativo para el favorito,
        positivo para el underdog.

            margen_ajustado = margen_de_la_selección + line

            > 0  → cubre        (win)
            < 0  → no cubre     (lose)
            = 0  → push         (null)

        El push solo ocurre con líneas ENTERAS, y en NFL eso dista de
        ser una rareza: las líneas se concentran en 3 y 7 porque son
        los márgenes de victoria más frecuentes del deporte. Contar
        esos empates como derrota subestimaría de forma sistemática el
        ROI de las apuestas a favoritos en números clave.
        """
        if line is None:
            return RESULT_VOID

        side = _resolve_side(selection, game)
        if side is None:
            return RESULT_VOID

        own_margin = game.margin if side == "home" else -game.margin
        adjusted = own_margin + line

        if adjusted > 0:
            return RESULT_WIN
        if adjusted < 0:
            return RESULT_LOSE
        return RESULT_PUSH

    @staticmethod
    def _settle_total(
        selection: str,
        line:      float | None,
        game:      NFLGameResult,
    ) -> str:
        """
        Total: la suma de puntos contra la línea.

        Push cuando la suma iguala exactamente una línea entera. Los
        totales NFL enteros (44, 47) son habituales, a diferencia de
        MLB donde predominan los .5 y el push es casi inexistente.
        """
        if line is None:
            return RESULT_VOID

        side = (selection or "").strip().lower()
        if side not in ("over", "under"):
            return RESULT_VOID

        total = game.total

        if total > line:
            return RESULT_WIN if side == "over" else RESULT_LOSE
        if total < line:
            return RESULT_WIN if side == "under" else RESULT_LOSE
        return RESULT_PUSH

    def _settle_from_entry(
        self,
        entry: BetLedgerEntry,
        game:  NFLGameResult,
    ) -> str:
        """
        Resuelve desde un BetLedgerEntry.

        El ledger guarda market, selection y line, que es todo lo
        necesario. Se reutiliza la misma lógica que settle_pick para
        garantizar que la liquidación deportiva y la financiera nunca
        divergen — si difirieran, el ROI registrado no correspondería
        a los picks reportados.
        """
        market = (entry.market or "").upper()

        if market in _PERIOD_MARKETS or _is_player_prop(market):
            return RESULT_VOID

        # El ledger guarda la selección con la línea embebida
        # ('over 47.0', 'LAC -3.5'), mientras que los métodos de
        # liquidación esperan la selección limpia. Separarlas aquí
        # mantiene ese formato de almacenamiento —legible en el CSV—
        # sin que la lógica de mercado tenga que conocerlo.
        selection, line = _split_selection(entry)

        if market == "ML":
            return self._settle_moneyline(selection, game)
        if market == "SPREAD":
            return self._settle_spread(selection, line, game)
        if market == "TOTAL":
            return self._settle_total(selection, line, game)

        return RESULT_VOID

    # ── Acceso al calendario ──────────────────────────────────────────────────

    def _fetch_result(self, game_id: str) -> NFLGameResult | None:
        """Obtiene y normaliza el resultado desde el calendario."""
        try:
            game = self._schedule.get_game_info(game_id)
        except Exception:
            return None

        if not isinstance(game, NFLGameInfo):
            return None

        home = _safe_float(game.home_score)
        away = _safe_float(game.away_score)

        # Sin marcador el partido no ha terminado, sea cual sea el
        # estado que declare el calendario.
        if home is None or away is None:
            return None

        return NFLGameResult(
            game_id    = game.game_id,
            home_team  = game.home_team,
            away_team  = game.away_team,
            home_score = home,
            away_score = away,
            status     = STATUS_FINAL,
            overtime   = bool(game.overtime),
        )

    @staticmethod
    def _extract_game_id(event: Event) -> str | None:
        """game_id desde provider_ids, con el event_id como respaldo."""
        provider_ids = event.provider_ids or {}
        game_id = provider_ids.get("nfl_game_id") or event.event_id
        return str(game_id) if game_id else None

    @staticmethod
    def _extract_game_id_from_entry(entry: BetLedgerEntry) -> str | None:
        """
        game_id desde un BetLedgerEntry.

        El entry_id se construye como '{event_id}_{market}_{selection}'
        y el event_id de NFL tiene formato '2026_03_KC_LAC' — que ya
        contiene guiones bajos. Por eso no se puede partir por el
        separador: se reconstruye tomando los cuatro primeros
        segmentos, que son temporada, semana y los dos equipos.
        """
        entry_id = entry.entry_id or ""
        parts = entry_id.split("_")
        if len(parts) >= 4:
            candidate = "_".join(parts[:4])
            # Validar la forma esperada: YYYY_WW_AWAY_HOME
            if parts[0].isdigit() and len(parts[0]) == 4 and parts[1].isdigit():
                return candidate
        return entry_id or None


# ── Utilidades de módulo ──────────────────────────────────────────────────────

def _resolve_side(selection: str, game: NFLGameResult) -> str | None:
    """
    Determina si la selección corresponde al local o al visitante.

    Acepta tanto la abreviación de nflverse ('KC') como el nombre
    completo que usa The Odds API ('Kansas City Chiefs'), porque el
    pick puede venir de cualquiera de las dos fuentes según en qué
    stage del pipeline se construyó.

    Retorna 'home', 'away' o None si no se puede determinar. None
    lleva a 'void', que es el comportamiento correcto: liquidar un
    pick cuyo equipo no se identifica sería adivinar.
    """
    sel = (selection or "").strip().lower()
    if not sel:
        return None

    home = (game.home_team or "").strip().lower()
    away = (game.away_team or "").strip().lower()

    if sel == home:
        return "home"
    if sel == away:
        return "away"

    # Coincidencia parcial para nombres completos: 'kansas city chiefs'
    # contiene la abreviación sólo por casualidad, así que se comprueba
    # en la dirección correcta — la abreviación dentro del nombre no es
    # fiable ('LA' aparece en 'Dallas'), pero el nombre conteniendo la
    # selección sí lo es cuando la selección es el nombre completo.
    if home and (sel.startswith(home) or home.startswith(sel)):
        return "home"
    if away and (sel.startswith(away) or away.startswith(sel)):
        return "away"

    return None


def _is_player_prop(market: str) -> bool:
    """True si el mercado es una prop de jugador."""
    return any(market.startswith(p) for p in _PLAYER_PROP_PREFIXES)


def _split_selection(entry: BetLedgerEntry) -> tuple[str, float | None]:
    """
    Separa la selección de la línea en un BetLedgerEntry.

    BetLedgerEntry no tiene campo `line` propio: el ledger guarda ambos
    datos juntos en `selection` porque así el CSV es legible sin
    consultar otra columna — 'over 47.0', 'LAC -3.5'.

    Esa conveniencia de almacenamiento no debe filtrarse a la lógica de
    mercado. _settle_total() compara la selección contra 'over'/'under'
    de forma exacta, así que recibir 'over 47.0' la anulaba: TODO pick
    de total se liquidaba como void y nunca registraba ROI en el
    ledger.

    (El spread se salvaba por casualidad: _resolve_side hace matching
    por prefijo, así que 'LAC -3.5' seguía resolviendo a 'LAC'. Un
    fallo que solo se manifestaba en uno de los tres mercados.)

    Retorna (selección_limpia, línea). La línea es None si no aparece
    embebida ni como atributo.
    """
    raw = (entry.selection or "").strip()

    # Atributo explícito si una versión futura lo añade
    line = _safe_float(getattr(entry, "line", None))

    # Separar el token numérico del resto. Se preserva el signo
    # pegándolo al número: '-3.5' debe leerse como -3.5, no como 3.5.
    tokens = raw.replace("+", " +").replace("-", " -").split()
    words: list[str] = []
    for token in tokens:
        value = _safe_float(token)
        if value is not None:
            if line is None:
                line = value
        else:
            words.append(token)

    return " ".join(words).strip(), line


def _is_nan(value) -> bool:
    try:
        return value != value
    except Exception:
        return False


def _safe_float(value) -> float | None:
    """Convierte a float de forma segura (NaN → None)."""
    if value is None or _is_nan(value):
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None