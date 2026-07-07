"""
core/odds/normalizer.py

OddsNormalizer: convierte RawOddsEvent (JSON crudo de The Odds API)
a list[MarketOdds] tipados y normalizados.

Migrado de analysis/markets.py del sistema MLB con tres correcciones
arquitectónicas documentadas en MLB_EDGE_AUDIT.md:

1. Elimina fuzzy matching por nombre de equipo.
   El sistema MLB usaba SequenceMatcher para unir partidos del schedule
   con eventos de la API. Frágil: "NY Yankees" vs "New York Yankees"
   fallaba silenciosamente. Aquí el sport plugin provee event_id
   directamente (Event.provider_ids['odds_api']) y el normalizer
   busca por ID exacto — sin ambigüedad posible.

2. Estrategia de mejor precio desacoplada del mercado.
   El sistema MLB hardcodeaba lógica de consenso de línea (totals) y
   handicap de referencia ±1.5 (spreads) dentro del parser. Aquí
   BestPriceStrategy es un Protocol — el sport plugin puede registrar
   su propia estrategia sin modificar el normalizer.

3. preferred_line como parámetro explícito.
   El normalizer no sabe que el runline de MLB es ±1.5 ni que el
   spread estándar de NBA es ±4.5. El sport plugin especifica qué
   línea prefiere. Si None, se elige la línea de mayor consenso
   entre los bookmakers disponibles.

Separación de responsabilidades
---------------------------------
OddsNormalizer hace SOLO:
    - Iterar bookmakers y markets del RawOddsEvent
    - Seleccionar el mejor precio por outcome
    - Construir objetos MarketOdds tipados
    - Normalizar nombres de mercado (h2h → ML, spreads → SPREAD, etc.)

NO hace:
    - HTTP (client.py)
    - Cálculo no-vig (no_vig.py)
    - Cálculo de EV/edge/Kelly (value/)
    - Matching de nombres de equipos (responsabilidad del sport plugin)
    - Decisiones de qué mercados solicitar (responsabilidad del pipeline)

Normalización de nombres de mercado
-------------------------------------
The Odds API usa nombres en minúsculas específicos de su implementación.
El contrato interno MarketOdds.market usa nombres convencionales en
mayúsculas que son independientes de la API:

    API key       →  MarketOdds.market
    ─────────────────────────────────
    h2h           →  ML
    spreads       →  SPREAD
    totals        →  TOTAL
    h2h_3_way     →  1X2
    team_totals   →  TEAM_TOTAL
    (otros)       →  key.upper()  (preservar el key original en mayúsculas)

Esta traducción ocurre en UN lugar (MARKET_NAME_MAP) — no dispersa
por el código ni requiere conocimiento de la API en los consumidores.

Uso típico
-----------
    from core.odds.normalizer import OddsNormalizer

    normalizer = OddsNormalizer()
    market_odds = normalizer.extract_best(
        raw_event=raw_event,          # RawOddsEvent de client.py
        markets=['h2h', 'totals', 'spreads'],
        preferred_line=-1.5,          # MLB runline estándar
    )
    # market_odds = [
    #   MarketOdds(market='ML', selection='New York Yankees', price=1.95, ...),
    #   MarketOdds(market='ML', selection='Boston Red Sox',   price=1.95, ...),
    #   MarketOdds(market='TOTAL', selection='over',  line=8.5, price=1.91, ...),
    #   MarketOdds(market='TOTAL', selection='under', line=8.5, price=1.91, ...),
    #   MarketOdds(market='SPREAD', selection='New York Yankees', line=-1.5, ...),
    #   MarketOdds(market='SPREAD', selection='Boston Red Sox',   line=+1.5, ...),
    # ]
"""

from __future__ import annotations

from collections import defaultdict
from typing import Protocol, runtime_checkable

from core.contracts.market_odds import MarketOdds
from core.odds.client import RawOddsEvent


# ── Normalización de nombres de mercado ───────────────────────────────────────

# Mapa de API key → nombre convencional interno.
# Permite que el resto del sistema use nombres estables independientes
# de la API de odds utilizada.
MARKET_NAME_MAP: dict[str, str] = {
    "h2h":          "ML",
    "spreads":      "SPREAD",
    "totals":       "TOTAL",
    "h2h_3_way":   "1X2",
    "team_totals":  "TEAM_TOTAL",
    # Mercados de innings MLB — se preservan con su key original en mayúsculas
    # si no están en este mapa, via la lógica de fallback en normalize_market_name()
}


def normalize_market_name(api_key: str) -> str:
    """
    Convierte un market key de The Odds API al nombre convencional
    interno del sistema.

    Si el key no está en MARKET_NAME_MAP, retorna api_key.upper().
    Esto preserva mercados desconocidos de forma legible sin forzar
    un mapeo exhaustivo que requeriría mantenimiento constante.
    """
    return MARKET_NAME_MAP.get(api_key.lower(), api_key.upper())


# ── Protocolo de estrategia de mejor precio ───────────────────────────────────

@runtime_checkable
class BestPriceStrategy(Protocol):
    """
    Protocolo para estrategias de selección de mejor precio.

    Permite desacoplar la lógica de "qué precio usar" del normalizer.
    El sport plugin puede registrar su propia estrategia para mercados
    con semántica especial (ej. handicap asiático de Soccer, donde
    el spread puede ser fraccionario y la línea de consenso es distinta).

    Contrato:
        select_best(outcomes, preferred_line) → list[dict]
        donde cada dict tiene: selection, price, line, bookmaker,
        timestamp, market_key.
    """

    def select_best(
        self,
        outcomes: list[dict],
        preferred_line: float | None,
    ) -> list[dict]:
        """
        Recibe la lista plana de todos los outcomes de todos los
        bookmakers para un mercado y retorna los outcomes seleccionados
        (uno por selección para extract_best, todos para extract_all).
        """
        ...


# ── Estrategias de mejor precio ───────────────────────────────────────────────

class SimpleBestPrice:
    """
    Estrategia simple: máximo precio por selección.

    Válida para mercados donde la selección no tiene línea (ML, 1X2,
    outright) o donde la línea es irrelevante para la selección de
    mejor precio.

    Para cada selección única, retorna el outcome con el precio más
    alto entre todos los bookmakers disponibles.
    """

    def select_best(
        self,
        outcomes: list[dict],
        preferred_line: float | None = None,
    ) -> list[dict]:
        """
        Agrupa por selection, retorna el mejor precio de cada grupo.
        preferred_line se ignora en esta estrategia.
        """
        best: dict[str, dict] = {}
        for outcome in outcomes:
            sel   = outcome.get("selection", "")
            price = outcome.get("price", 0.0)
            if not sel or price <= 1.0:
                continue
            if sel not in best or price > best[sel]["price"]:
                best[sel] = outcome
        return list(best.values())


class ConsensusTotalPrice:
    """
    Estrategia para mercados de totales (Over/Under).

    Elige la línea de mayor consenso entre bookmakers, luego retorna
    el mejor precio para Over y Under en esa línea.

    "Consenso" = línea que aparece en el mayor número de bookmakers.
    Cuando hay empate, se prefiere la línea con menor overround
    (menor vig = mercado más eficiente).

    Si preferred_line está especificado, se usa esa línea en vez de
    la de mayor consenso — útil para sport plugins que quieren
    comparar específicamente contra una línea histórica de referencia.
    """

    def select_best(
        self,
        outcomes: list[dict],
        preferred_line: float | None = None,
    ) -> list[dict]:
        """
        1. Agrupar por línea → contar bookmakers que la ofrecen
        2. Si preferred_line → usar esa línea
           Si no → elegir línea de mayor consenso (menor vig en empate)
        3. Retornar mejor precio para over y under en la línea elegida
        """
        # Agrupar outcomes por línea
        by_line: dict[float, dict] = {}
        for outcome in outcomes:
            line  = outcome.get("line")
            price = outcome.get("price", 0.0)
            sel   = outcome.get("selection", "").lower()
            if line is None or price <= 1.0 or sel not in ("over", "under"):
                continue

            line = float(line)
            if line not in by_line:
                by_line[line] = {
                    "over_prices":  [],
                    "under_prices": [],
                    "bookmaker_count": 0,
                }
            entry = by_line[line]
            if sel == "over":
                entry["over_prices"].append(outcome)
            else:
                entry["under_prices"].append(outcome)
            entry["bookmaker_count"] += 1

        if not by_line:
            return []

        # Seleccionar la línea objetivo
        if preferred_line is not None and preferred_line in by_line:
            target_line = preferred_line
        else:
            # Línea de mayor consenso; en empate, menor overround
            def line_score(line: float) -> tuple:
                entry = by_line[line]
                best_over  = max((o["price"] for o in entry["over_prices"]),  default=0.0)
                best_under = max((o["price"] for o in entry["under_prices"]), default=0.0)
                overround  = (1/best_over + 1/best_under) if best_over > 1 and best_under > 1 else 99.0
                return (-entry["bookmaker_count"], overround)

            target_line = min(by_line.keys(), key=line_score)

        entry = by_line[target_line]
        result = []

        # Mejor precio para over
        over_outcomes = entry["over_prices"]
        if over_outcomes:
            best_over = max(over_outcomes, key=lambda o: o["price"])
            result.append(best_over)

        # Mejor precio para under
        under_outcomes = entry["under_prices"]
        if under_outcomes:
            best_under = max(under_outcomes, key=lambda o: o["price"])
            result.append(best_under)

        return result


class BestSpreadPrice:
    """
    Estrategia para mercados de spread/handicap.

    Cuando preferred_line está especificado, busca el handicap más
    cercano a ese valor. Cuando no está especificado, usa el spread
    de mayor consenso entre bookmakers.

    Retorna un outcome por lado (home y away) para el spread elegido.

    El sport plugin especifica preferred_line:
        MLB: preferred_line=-1.5 (runline estándar)
        NBA: preferred_line=-4.5 (spread típico)
        NFL: preferred_line=-3.0 (field goal típico)
        Soccer: preferred_line=None (consenso, puede ser fraccionario)
    """

    def select_best(
        self,
        outcomes: list[dict],
        preferred_line: float | None = None,
    ) -> list[dict]:
        """
        1. Agrupar por (selection, line) → mejor precio por par
        2. Elegir el line más cercano a preferred_line (si especificado)
           o el de mayor consenso
        3. Retornar un outcome por selection para el line elegido
        """
        # Indexar outcomes por (selection, line)
        by_sel_line: dict[tuple[str, float], list[dict]] = defaultdict(list)
        selections_seen: set[str] = set()

        for outcome in outcomes:
            sel   = outcome.get("selection", "")
            line  = outcome.get("line")
            price = outcome.get("price", 0.0)
            if not sel or line is None or price <= 1.0:
                continue
            key = (sel, float(line))
            by_sel_line[key].append(outcome)
            selections_seen.add(sel)

        if not by_sel_line:
            return []

        # Líneas disponibles (únicas, independiente de selection)
        available_lines = sorted({line for _, line in by_sel_line.keys()})
        if not available_lines:
            return []

        # Elegir target_line
        if preferred_line is not None:
            # La línea de mercado más cercana al preferred_line
            target_line = min(
                available_lines,
                key=lambda l: abs(abs(l) - abs(preferred_line))
            )
        else:
            # Línea de mayor consenso: cuenta outcomes totales por línea
            line_counts: dict[float, int] = defaultdict(int)
            for (_, line), outs in by_sel_line.items():
                line_counts[line] += len(outs)
            target_line = max(line_counts, key=lambda l: line_counts[l])

        # Retornar mejor precio por selection para target_line
        result = []
        for sel in selections_seen:
            key = (sel, target_line)
            candidates = by_sel_line.get(key, [])
            if not candidates:
                continue
            best = max(candidates, key=lambda o: o["price"])
            result.append(best)

        return result


# ── Motor principal ───────────────────────────────────────────────────────────

# Estrategias por defecto por API market key.
# El sport plugin puede sobreescribir estas estrategias pasando
# strategies={"totals": MiEstrategiaCustom()} a OddsNormalizer.
_DEFAULT_STRATEGIES: dict[str, BestPriceStrategy] = {
    "h2h":        SimpleBestPrice(),
    "h2h_3_way": SimpleBestPrice(),
    "spreads":    BestSpreadPrice(),
    "totals":     ConsensusTotalPrice(),
}


class OddsNormalizer:
    """
    Convierte RawOddsEvent a list[MarketOdds] tipados y normalizados.

    No contiene lógica deportiva. No sabe qué es un runline, un
    carrera, ni un park factor. Sólo entiende la estructura JSON
    de The Odds API y el contrato MarketOdds.

    Parámetros
    ----------
    strategies  -- Mapa de API market key → BestPriceStrategy.
                  Si None, usa _DEFAULT_STRATEGIES. El sport plugin
                  puede sobreescribir estrategias para mercados con
                  semántica especial.
    """

    def __init__(
        self,
        strategies: dict[str, BestPriceStrategy] | None = None,
    ) -> None:
        self._strategies = strategies or dict(_DEFAULT_STRATEGIES)

    # ── API pública ────────────────────────────────────────────────────────────

    def extract_best(
        self,
        raw_event: RawOddsEvent,
        markets: list[str],
        preferred_line: float | None = None,
    ) -> list[MarketOdds]:
        """
        Extrae el mejor precio disponible por selección para los
        mercados solicitados de un RawOddsEvent.

        "Mejor precio" = máxima cuota disponible entre bookmakers para
        esa selección en ese mercado. El consumidor (ValueEngine) se
        beneficia del mejor precio disponible, no del promedio.

        Parámetros
        ----------
        raw_event       -- Evento crudo de OddsAPIClient.
        markets          -- Lista de API market keys a normalizar.
                           Ej: ['h2h', 'totals', 'spreads'].
                           Solo se procesan los markets que tengan
                           datos en raw_event.bookmakers.
        preferred_line   -- Línea de referencia para mercados de spread
                           y totals. El sport plugin especifica el
                           valor estándar de su deporte:
                               MLB: -1.5 (runline)
                               NBA: -4.5 (spread típico)
                               None: usar línea de mayor consenso
                           No afecta mercados sin línea (h2h, 1X2).

        Retorna
        -------
        list[MarketOdds] ordenada por mercado y selección. Cada
        elemento tiene price > 1.0 (invariante garantizado por
        MarketOdds.__post_init__). Los outcomes inválidos (price ≤ 1.0,
        selección vacía) se descartan silenciosamente.
        """
        result: list[MarketOdds] = []

        for api_key in markets:
            outcomes = self._collect_outcomes(raw_event, api_key)
            if not outcomes:
                continue

            strategy  = self._strategies.get(api_key.lower(), SimpleBestPrice())
            selected  = strategy.select_best(outcomes, preferred_line)
            market_name = normalize_market_name(api_key)

            for outcome in selected:
                market_odds = self._to_market_odds(
                    outcome=outcome,
                    event_id=raw_event.event_id,
                    market_name=market_name,
                )
                if market_odds is not None:
                    result.append(market_odds)

        return result

    def extract_all(
        self,
        raw_event: RawOddsEvent,
        markets: list[str],
    ) -> list[MarketOdds]:
        """
        Extrae TODOS los precios disponibles (no solo el mejor) para
        análisis de CLV y detección de line movement.

        A diferencia de extract_best(), no aplica estrategia de
        selección — retorna un MarketOdds por (selection, line,
        bookmaker) para cada combinación disponible.

        Útil para:
            - LineMovementDetector: comparar apertura vs cierre
            - CLV calculation: precio al momento del pick vs precio
              de cierre del mercado
            - Análisis de dispersión de cuotas entre bookmakers
        """
        result: list[MarketOdds] = []

        for api_key in markets:
            outcomes = self._collect_outcomes(raw_event, api_key)
            market_name = normalize_market_name(api_key)

            for outcome in outcomes:
                market_odds = self._to_market_odds(
                    outcome=outcome,
                    event_id=raw_event.event_id,
                    market_name=market_name,
                )
                if market_odds is not None:
                    result.append(market_odds)

        return result

    @staticmethod
    def find_by_event_id(
        raw_events: list[RawOddsEvent],
        event_id: str,
    ) -> RawOddsEvent | None:
        """
        Busca un RawOddsEvent por ID exacto.

        Reemplaza el fuzzy matching por nombre de equipo del sistema
        MLB. El sport plugin debe proveer el event_id de The Odds API
        en Event.provider_ids['odds_api'].

        Retorna None si no se encuentra — el pipeline decide si abortar
        o continuar sin cuotas para ese evento.
        """
        for event in raw_events:
            if event.event_id == event_id:
                return event
        return None

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _collect_outcomes(
        self,
        raw_event: RawOddsEvent,
        api_key: str,
    ) -> list[dict]:
        """
        Recopila todos los outcomes de todos los bookmakers para un
        market key específico, con metadatos de procedencia.

        Retorna lista plana de dicts normalizados:
            selection, price, line, bookmaker, timestamp, market_key.
        """
        outcomes: list[dict] = []
        api_key_lower = api_key.lower()

        for bm in raw_event.bookmakers:
            bm_name     = bm.get("title") or bm.get("key", "unknown")
            bm_markets  = bm.get("markets", [])

            for market in bm_markets:
                if market.get("key", "").lower() != api_key_lower:
                    continue

                last_update = market.get("last_update", "")

                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    name  = outcome.get("name", "")
                    point = outcome.get("point")

                    if price is None or not name:
                        continue

                    try:
                        price = float(price)
                    except (ValueError, TypeError):
                        continue

                    if price <= 1.0:
                        continue

                    outcomes.append({
                        "selection":  name,
                        "price":      price,
                        "line":       float(point) if point is not None else None,
                        "bookmaker":  bm_name,
                        "timestamp":  last_update,
                        "market_key": api_key,
                    })

        return outcomes

    @staticmethod
    def _to_market_odds(
        outcome: dict,
        event_id: str,
        market_name: str,
    ) -> MarketOdds | None:
        """
        Convierte un outcome dict a un objeto MarketOdds.

        Retorna None si la construcción falla (price inválido,
        campos requeridos faltantes). El caller descarta silenciosamente
        los None — no aborta el procesamiento del evento completo.
        """
        try:
            return MarketOdds(
                event_id  = event_id,
                market    = market_name,
                selection = outcome["selection"],
                line      = outcome.get("line"),
                price     = outcome["price"],
                bookmaker = outcome.get("bookmaker", "unknown"),
                timestamp = outcome.get("timestamp", ""),
            )
        except (ValueError, KeyError, TypeError):
            return None