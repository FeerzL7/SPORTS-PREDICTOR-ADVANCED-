"""
sports/soccer/settlement.py

SoccerSettlementProvider: liquidación de picks de fútbol.

Implementa las DOS interfaces SettlementProvider del sistema, igual que
los plugins de MLB y NFL:

    core/pipeline/stage.py       "deportiva": resuelve un CandidatePick
                                  contra el resultado (Stage 10).
    core/tracking/protocols.py   "financiera": recibe un
                                  BetLedgerEntry y produce un
                                  SettlementResult con CLV.

Tres diferencias con el plugin NFL
------------------------------------

1. EL EMPATE ES UN RESULTADO, NO UN PUSH
   En NFL el empate ocurre en el 0.4% de partidos y las casas lo
   resuelven devolviendo el stake en moneyline.

   Aquí es el 25% de los partidos y la tercera opción del mercado
   principal. Un pick de empate GANA cuando el partido termina
   igualado, y pierde cuando no. Tratarlo como push destruiría el
   mercado donde el modelo busca más valor.

2. PUSH EN TOTALES ENTEROS, Y NO ES MARGINAL
   Los totales de fútbol se cotizan mayoritariamente en .5, pero las
   líneas enteras existen —2.0 y 3.0 sobre todo— y ahí el push es
   frecuente: la matriz de resultados asigna ~25% a que el partido
   termine con exactamente dos goles.

   Contarlo como derrota subestimaría el ROI de forma sistemática en
   todo pick de total con línea entera.

3. LOS MERCADOS DE PRIMERA PARTE SÍ SE LIQUIDAN
   SoccerMatchInfo lleva el marcador al descanso, que football-data
   publica. En NFL no teníamos el parcial y los mercados H1 devolvían
   'void' de forma explícita.

   Aquí se resuelven con normalidad. Es una capacidad real que el
   sistema tiene y que conviene no desperdiciar.

Qué NO se liquida
-------------------
Hándicap asiático. Las líneas de cuarto (-0.25, -0.75) dividen el
stake en dos apuestas y producen push parciales que el contrato del
ledger no representa: una entrada tendría que resolverse como "media
ganada, media devuelta". Devuelve 'void' de forma explícita hasta que
exista esa infraestructura.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from core.contracts.event import Event
from core.contracts.ledger import BetLedgerEntry
from core.contracts.pick import CandidatePick
from core.tracking.protocols import SettlementResult

from sports.soccer.schedule import SoccerMatchInfo
from sports.soccer.teams import canonical_team


__all__ = ["SoccerSettlementProvider", "SoccerMatchResult"]


# ── Dependencia inyectada ────────────────────────────────────────────────────

@runtime_checkable
class ScheduleSource(Protocol):
    """Interfaz mínima del calendario que este módulo consume."""

    def get_match_info(self, match_id: str) -> SoccerMatchInfo | None:
        ...


# ── Resultados ───────────────────────────────────────────────────────────────

RESULT_WIN  = "win"
RESULT_LOSE = "lose"
RESULT_PUSH = "null"    # el Core usa 'null' para devolución sin pérdida
RESULT_VOID = "void"

# Selecciones que designan el empate en el mercado 1X2.
#
# Se aceptan varias grafías porque el pick puede venir de The Odds API
# ("Draw"), del propio modelo ("draw") o del ledger, donde se guarda en
# el idioma de la interfaz.
_DRAW_SELECTIONS = frozenset({"draw", "x", "empate", "tie"})

# Mercados que requieren infraestructura de push parcial.
_ASIAN_HANDICAP_MARKETS = frozenset({"AH", "ASIAN_HANDICAP", "SPREAD"})


@dataclass(frozen=True)
class SoccerMatchResult:
    """
    Resultado de un partido, normalizado para liquidación.

    Lleva el marcador al descanso porque en fútbol los mercados de
    primera parte SÍ son liquidables, a diferencia de NFL.
    """
    match_id:   str
    comp_id:    str
    home:       str
    away:       str
    home_goals: int
    away_goals: int
    ht_home:    int | None = None
    ht_away:    int | None = None

    @property
    def total(self) -> int:
        return self.home_goals + self.away_goals

    @property
    def outcome(self) -> str:
        """'H', 'D' o 'A'."""
        if self.home_goals > self.away_goals:
            return "H"
        if self.away_goals > self.home_goals:
            return "A"
        return "D"

    @property
    def btts(self) -> bool:
        """True si ambos equipos marcaron."""
        return self.home_goals > 0 and self.away_goals > 0

    @property
    def has_halftime(self) -> bool:
        return self.ht_home is not None and self.ht_away is not None

    @property
    def ht_total(self) -> int | None:
        if not self.has_halftime:
            return None
        return (self.ht_home or 0) + (self.ht_away or 0)

    @property
    def ht_outcome(self) -> str | None:
        if not self.has_halftime:
            return None
        h, a = self.ht_home or 0, self.ht_away or 0
        return "H" if h > a else ("A" if a > h else "D")

    def to_dict(self) -> dict:
        return {
            "match_id":   self.match_id,
            "competition": self.comp_id,
            "home":       self.home,
            "away":       self.away,
            "home_goals": self.home_goals,
            "away_goals": self.away_goals,
            "total":      self.total,
            "outcome":    self.outcome,
            "btts":       self.btts,
            "ht_home":    self.ht_home,
            "ht_away":    self.ht_away,
            "ht_outcome": self.ht_outcome,
        }

    @classmethod
    def from_dict(cls, data: dict) -> SoccerMatchResult | None:
        if not isinstance(data, dict):
            return None
        home = _safe_int(data.get("home_goals"))
        away = _safe_int(data.get("away_goals"))
        if home is None or away is None:
            return None
        return cls(
            match_id=str(data.get("match_id", "")),
            comp_id=str(data.get("competition", "")),
            home=str(data.get("home", "")),
            away=str(data.get("away", "")),
            home_goals=home, away_goals=away,
            ht_home=_safe_int(data.get("ht_home")),
            ht_away=_safe_int(data.get("ht_away")),
        )


# ── Provider ─────────────────────────────────────────────────────────────────

class SoccerSettlementProvider:
    """
    Liquidación de picks de fútbol.

    Parámetros
    ----------
    schedule_fetcher -- Fuente de partidos con marcador.
    closing_prices   -- Dict {entry_id: precio de cierre} para el CLV.
                        El pipeline lo alimenta antes de liquidar.
    """

    def __init__(
        self,
        schedule_fetcher: ScheduleSource,
        closing_prices:   dict | None = None,
    ) -> None:
        self._schedule: ScheduleSource = schedule_fetcher
        self._closing_prices = dict(closing_prices or {})

    # ── Interfaz deportiva ────────────────────────────────────────────────────

    def get_event_result(self, event: Event) -> dict | None:
        """
        Resultado del partido, o None si aún no ha terminado.

        None significa "todavía no liquidable": el runner salta el pick
        sin registrar un intento fallido.
        """
        result = self._fetch(self._match_id(event))
        return result.to_dict() if result else None

    def settle_pick(self, pick: CandidatePick, event_result: dict) -> str:
        """
        Resuelve un pick contra el resultado.

        Retorna 'win', 'lose', 'null' (push) o 'void'.

        Nunca lanza: ante datos incoherentes devuelve 'void', que deja
        el stake intacto. Anular es siempre preferible a registrar una
        ganancia o pérdida que no ocurrió.
        """
        result = SoccerMatchResult.from_dict(event_result)
        if result is None:
            return RESULT_VOID

        market = (pick.market or "").upper()
        selection = pick.selection or ""
        line = pick.line

        return self._settle(market, selection, line, result)

    # ── Interfaz financiera ───────────────────────────────────────────────────

    def get_result(self, entry: BetLedgerEntry) -> SettlementResult | None:
        """
        Liquida un BetLedgerEntry y añade el CLV.

        None si el partido no ha terminado: el tracker deja el entry
        pendiente y reintentará.
        """
        result = self._fetch(self._match_id_from_entry(entry))
        if result is None:
            return None

        # El ledger guarda la línea embebida en la selección
        # ('over 2.5'), igual que en NFL. Separarlas aquí evita que la
        # lógica de mercado conozca ese formato de almacenamiento.
        selection, line = _split_selection(entry.selection)
        outcome = self._settle(
            (entry.market or "").upper(), selection, line, result
        )

        return SettlementResult(
            entry_id=entry.entry_id,
            result=outcome,
            settled_at=datetime.now(timezone.utc).isoformat(),
            closing_price=self.get_closing_price(entry),
            sport_context=result.to_dict(),
        )

    def get_closing_price(self, entry: BetLedgerEntry) -> float | None:
        return self._closing_prices.get(entry.entry_id)

    def register_closing_price(self, entry_id: str, price: float) -> None:
        self._closing_prices[entry_id] = price

    # ── Resolución por mercado ────────────────────────────────────────────────

    def _settle(
        self,
        market:    str,
        selection: str,
        line:      float | None,
        result:    SoccerMatchResult,
    ) -> str:
        """Despacha al resolutor del mercado."""
        if market in _ASIAN_HANDICAP_MARKETS:
            # Requiere infraestructura de push parcial. Ver la nota
            # del módulo.
            return RESULT_VOID

        if market in ("1X2", "ML"):
            return self._settle_1x2(selection, result.outcome, result)
        if market == "TOTAL":
            return self._settle_total(selection, line, result.total)
        if market == "BTTS":
            return self._settle_btts(selection, result.btts)

        # ── Mercados de primera parte ──────────────────────────────
        if market in ("1X2_H1", "ML_H1"):
            if not result.has_halftime:
                return RESULT_VOID
            return self._settle_1x2(selection, result.ht_outcome or "", result)
        if market == "TOTAL_H1":
            ht_total = result.ht_total
            if ht_total is None:
                return RESULT_VOID
            return self._settle_total(selection, line, ht_total)

        return RESULT_VOID

    @staticmethod
    def _settle_1x2(
        selection: str,
        outcome:   str,
        result:    SoccerMatchResult,
    ) -> str:
        """
        Mercado 1X2: local, empate o visitante.

        El empate es un RESULTADO, no un push. Un pick de empate gana
        cuando el partido termina igualado y pierde cuando no.

        Es la diferencia central con NFL, donde el empate ocurre en el
        0.4% de partidos y se resuelve devolviendo el stake. Aquí es el
        25% y el mercado donde los books aplican más margen —
        precisamente por eso es donde el modelo busca su valor.
        """
        side = _resolve_side(selection, result)
        if side is None:
            return RESULT_VOID
        return RESULT_WIN if side == outcome else RESULT_LOSE

    @staticmethod
    def _settle_total(
        selection: str,
        line:      float | None,
        total:     int,
    ) -> str:
        """
        Over/under de goles.

        Push cuando el total iguala exactamente una línea entera. En
        fútbol eso dista de ser marginal: la distribución de goles
        concentra alrededor del 25% de los partidos en exactamente dos,
        que es la línea entera más cotizada.

        Contarlo como derrota subestimaría el ROI de forma sistemática
        en todo pick de total con línea entera.
        """
        if line is None:
            return RESULT_VOID

        side = (selection or "").strip().lower()
        if side not in ("over", "under"):
            return RESULT_VOID

        if total > line:
            return RESULT_WIN if side == "over" else RESULT_LOSE
        if total < line:
            return RESULT_WIN if side == "under" else RESULT_LOSE
        return RESULT_PUSH

    @staticmethod
    def _settle_btts(selection: str, both_scored: bool) -> str:
        """
        Ambos equipos marcan.

        No tiene push: o ambos marcaron o no. Acepta las grafías de The
        Odds API ('Yes'/'No') y las del modelo ('yes'/'no').
        """
        side = (selection or "").strip().lower()
        if side in ("yes", "si", "sí", "true", "btts"):
            return RESULT_WIN if both_scored else RESULT_LOSE
        if side in ("no", "false", "nobtts"):
            return RESULT_WIN if not both_scored else RESULT_LOSE
        return RESULT_VOID

    # ── Acceso al calendario ──────────────────────────────────────────────────

    def _fetch(self, match_id: str | None) -> SoccerMatchResult | None:
        """Obtiene y normaliza el resultado."""
        if not match_id:
            return None

        try:
            match = self._schedule.get_match_info(match_id)
        except Exception:
            return None

        if not isinstance(match, SoccerMatchInfo) or not match.is_final:
            return None
        if match.home_goals is None or match.away_goals is None:
            return None

        return SoccerMatchResult(
            match_id=match.match_id,
            comp_id=match.comp_id,
            home=match.home,
            away=match.away,
            home_goals=int(match.home_goals),
            away_goals=int(match.away_goals),
            ht_home=match.ht_home,
            ht_away=match.ht_away,
        )

    @staticmethod
    def _match_id(event: Event) -> str | None:
        provider_ids = event.provider_ids or {}
        match_id = provider_ids.get("match_id") or event.event_id
        return str(match_id) if match_id else None

    @staticmethod
    def _match_id_from_entry(entry: BetLedgerEntry) -> str | None:
        """
        match_id desde un BetLedgerEntry.

        El entry_id se construye como '{match_id}_{market}_{selection}'
        y el match_id de fútbol tiene formato
        'epl_2024-11-09_manchester-city_arsenal' — cuatro segmentos
        separados por guion bajo, con guiones normales dentro de los
        nombres.

        Por eso se reconstruye tomando los cuatro primeros segmentos en
        vez de partir por el separador, que aparece también dentro del
        propio id.
        """
        entry_id = entry.entry_id or ""
        parts = entry_id.split("_")
        if len(parts) >= 4:
            # Validar la forma: comp_YYYY-MM-DD_local_visitante
            fecha = parts[1]
            if len(fecha) == 10 and fecha[4] == "-" and fecha[7] == "-":
                return "_".join(parts[:4])
        return entry_id or None


# ── Utilidades ───────────────────────────────────────────────────────────────

def _resolve_side(selection: str, result: SoccerMatchResult) -> str | None:
    """
    Traduce una selección de 1X2 a 'H', 'D' o 'A'.

    Acepta el nombre del equipo en cualquiera de las grafías que
    manejan las fuentes —'Man City' de football-data, 'Manchester City'
    de The Odds API— gracias a la reconciliación de teams.py.

    Retorna None si no se puede determinar, lo que lleva a 'void':
    liquidar un pick cuyo equipo no se identifica sería adivinar.
    """
    text = (selection or "").strip().lower()
    if not text:
        return None

    if text in _DRAW_SELECTIONS:
        return "D"

    canon = canonical_team(selection, result.comp_id)
    if canon and canon == result.home:
        return "H"
    if canon and canon == result.away:
        return "A"

    return None


def _split_selection(raw: str) -> tuple[str, float | None]:
    """
    Separa 'over 2.5' en ('over', 2.5), preservando el signo.

    El ledger guarda la línea embebida en la selección para que el CSV
    sea legible sin cruzar columnas. Esa conveniencia de almacenamiento
    no debe filtrarse a la lógica de mercado: _settle_total compara la
    selección contra 'over'/'under' de forma exacta, así que recibir
    'over 2.5' la anularía.

    Es el mismo fallo que apareció en el plugin NFL, donde TODO pick de
    total se liquidaba como void por la vía financiera y ninguno
    registraba ROI.
    """
    text = (raw or "").strip()
    line: float | None = None
    words: list[str] = []

    for token in text.replace("+", " +").replace("-", " -").split():
        value = _safe_float(token)
        if value is None:
            words.append(token)
        elif line is None:
            line = value

    return " ".join(words).strip(), line


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