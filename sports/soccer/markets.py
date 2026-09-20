"""
sports/soccer/markets.py

SoccerMarketDefinitions: catálogo de mercados de fútbol.
Implementa core/pipeline/stage.py:MarketDefinitions.

Consume el MarketRegistry
---------------------------
Igual que el plugin NFL, y por la misma razón: el catálogo vive en
core/odds/market_registry.py y duplicarlo obligaría a editar dos sitios
cada vez que cambia.

Los mercados de fútbol se registraron allí en la tarea 11.3, incluida
la corrección de retirar `h2h_3_way` —redundante, porque el `h2h` de
fútbol YA devuelve el empate— y de eliminar `draw_no_bet` y
`double_chance`, cuyo margen del book los hace peores vehículos que el
1X2 para expresar la misma opinión.

La línea preferida: aquí SÍ existe
------------------------------------
En NFL `get_preferred_line('SPREAD')` devuelve None porque el spread es
variable, de -1 a -14 según el emparejamiento: no hay un valor que
tenga sentido para todos los partidos.

En fútbol el total de 2.5 es el estándar de las cinco grandes. Casi
todos los partidos se cotizan ahí, y OddsNormalizer puede seleccionar
esa línea con certeza. Devolverla concentra la comparación en un punto
consistente entre partidos, en vez de mezclar totales de 2.0, 2.5 y 3.0
que no son el mismo mercado.

El 1X2 no lleva línea, así que devuelve None por naturaleza, no por la
misma razón que el spread de NFL.

Traducción de nombres de mercado
----------------------------------
`market_name_overrides()` expone el mapeo que OddsNormalizer necesita.

El `h2h` de fútbol devuelve TRES resultados; el de MLB, NFL y NBA
devuelve dos. Con el mapa global del Core se normalizaría a 'ML' y el
pipeline buscaría filters.ML —umbrales de dos vías aplicados a uno de
tres— donde el empate, con su ~27% de probabilidad, quedaría
descartado por cualquier min_prob razonable. Y es justamente el mercado
donde los books aplican más margen.

Riesgo de push por línea entera
---------------------------------
Los totales de fútbol se cotizan mayoritariamente en .5, pero las
líneas enteras existen y ahí el push es frecuente: la distribución de
goles concentra ~25% de los partidos en exactamente dos.

`has_push_risk()` lo expone porque cambia la estructura del EV: una
apuesta con 25% de devolución no es comparable a una sin push aunque
acierte con la misma frecuencia — el stake efectivamente arriesgado es
menor y el retorno esperado, distinto.
"""

from __future__ import annotations

from core.odds.market_registry import MarketRegistry, default_registry


__all__ = ["SoccerMarketDefinitions"]


# ── Línea preferida por mercado ──────────────────────────────────────────────
#
# 2.5 goles es el estándar de las cinco grandes: la media de la
# competición más goleadora (Bundesliga, 3.10) y la menos (La Liga,
# 2.53) quedan a ambos lados, así que el mercado se reparte de forma
# razonablemente equilibrada en todas.
_PREFERRED_TOTAL_LINE = 2.5

# Líneas de total con riesgo de push. Se listan las enteras que el
# mercado cotiza realmente; por encima de 4 son residuales.
_PUSH_RISK_LINES = frozenset({1.0, 2.0, 3.0, 4.0})

# Traducción de claves de la API a nombres internos, específica de
# fútbol. Ver la nota del módulo.
_MARKET_NAME_OVERRIDES: dict[str, str] = {
    "h2h": "1X2",
}

# Mercados que el settlement no puede resolver todavía.
#
# `spreads` es el hándicap asiático en fútbol. Sus líneas de cuarto
# (-0.25, -0.75) dividen el stake en dos apuestas y producen push
# parciales que el contrato del ledger no representa, así que
# SoccerSettlementProvider las devuelve como 'void'.
#
# Pedirlas gastaría créditos de API para generar picks anulados — el
# mismo criterio que llevó a excluir los mercados de periodo en NFL.
_HANDICAP_KEYS = frozenset({"spreads", "alternate_spreads"})


class SoccerMarketDefinitions:
    """
    Catálogo de mercados de fútbol.

    Parámetros
    ----------
    registry         -- MarketRegistry a consultar. None usa el del
                        Core, que ya incluye los mercados de fútbol
                        desde la tarea 11.3.
    include_handicap -- Si True, incluye el hándicap asiático. Default
                        False: el settlement no lo resuelve todavía.
    include_btts     -- Si False, excluye BTTS. Útil para operar solo
                        los dos mercados principales mientras se
                        acumula historial propio.
    """

    sport_id: str = "soccer"

    def __init__(
        self,
        registry:         MarketRegistry | None = None,
        include_handicap: bool = False,
        include_btts:     bool = True,
    ) -> None:
        self._registry = registry or default_registry()
        self._include_handicap = include_handicap
        self._include_btts = include_btts

    # ── MarketDefinitions Protocol ────────────────────────────────────────────

    def get_core_markets(self) -> list[str]:
        """
        Mercados CORE: una request cubre todos los partidos.

        Para fútbol quedan `h2h` (que ES el 1X2) y `totals`. El
        hándicap asiático se excluye por defecto porque el settlement
        no lo resuelve.

        Nota de coste: The Odds API cobra por mercado y región. Pasar
        de tres mercados a dos ahorra un tercio de cada petición, y con
        cinco competiciones activas eso se multiplica por cinco.
        """
        markets = self._registry.get_core_markets(self.sport_id)
        return [m for m in markets if self._allowed(m)]

    def get_extended_markets(self) -> list[str]:
        """
        Mercados EXTENDED: endpoint por evento, mayor coste.

        BTTS es el único que el plugin opera de esta lista. Los
        alternate_* y team_totals se registran en el Core pero no
        aportan a un modelo que ya deriva todos los mercados de una
        matriz de resultados: pedirlos sería pagar por información que
        el modelo ya tiene.
        """
        markets = self._registry.get_extended_markets(self.sport_id)
        return [m for m in markets if self._allowed(m) and self._operated(m)]

    def get_preferred_line(self, market: str) -> float | None:
        """
        Línea preferida de un mercado.

        TOTAL devuelve 2.5, que es el estándar de las cinco grandes.
        A diferencia de NFL —donde el spread variable obligaba a
        devolver None— aquí OddsNormalizer puede seleccionar esa línea
        con certeza en casi todos los partidos.

        Concentrar la comparación en un punto consistente importa: un
        total de 2.0 y otro de 3.0 no son el mismo mercado, y mezclar
        sus EV en el mismo filtro compararía cosas distintas.

        1X2 y BTTS no llevan línea, así que devuelven None por
        naturaleza.
        """
        key = (market or "").strip().upper()
        if key in ("TOTAL", "TOTALS", "OU"):
            return _PREFERRED_TOTAL_LINE
        return None

    # ── Extensiones específicas de fútbol ─────────────────────────────────────

    @staticmethod
    def market_name_overrides() -> dict[str, str]:
        """
        Traducción de claves de la API a nombres internos.

        El plugin lo pasa a OddsNormalizer, que lo aplica sobre el mapa
        global del Core.

        Sin esto, el `h2h` de fútbol —que devuelve TRES resultados— se
        normalizaría a 'ML' y el pipeline buscaría filters.ML: umbrales
        calibrados para mercados de dos vías. El empate, con su ~27% de
        probabilidad, quedaría descartado por cualquier min_prob
        razonable.
        """
        return dict(_MARKET_NAME_OVERRIDES)

    @staticmethod
    def has_push_risk(line: float | None) -> bool:
        """
        True si la línea de total puede terminar en push.

        Ocurre con líneas enteras, y en fútbol no es un caso residual:
        la distribución de goles concentra ~25% de los partidos en
        exactamente dos, que es la línea entera más cotizada.

        Importa para el EV: una apuesta con 25% de probabilidad de
        devolución no es comparable a una sin push aunque acierte con
        la misma frecuencia. El stake efectivamente arriesgado es menor
        y el retorno esperado, distinto.
        """
        if line is None:
            return False
        try:
            return float(line) in _PUSH_RISK_LINES
        except (ValueError, TypeError):
            return False

    @staticmethod
    def is_standard_line(line: float | None) -> bool:
        """True si es la línea de referencia del mercado de totales."""
        if line is None:
            return False
        try:
            return float(line) == _PREFERRED_TOTAL_LINE
        except (ValueError, TypeError):
            return False

    # ── Filtros internos ──────────────────────────────────────────────────────

    def _allowed(self, api_key: str) -> bool:
        """Falso si la política del plugin excluye el mercado."""
        if not self._include_handicap and api_key in _HANDICAP_KEYS:
            return False
        if not self._include_btts and api_key == "btts":
            return False
        return True

    @staticmethod
    def _operated(api_key: str) -> bool:
        """
        Falso para mercados que el plugin no explota.

        Los alternate_* y team_totals no aportan a un modelo que deriva
        todos los mercados de una matriz de resultados: si el modelo ya
        conoce P(total = k) para todo k, pedir cotizaciones de líneas
        alternativas no le da información nueva — solo cuesta créditos.

        Se dejan registrados en el Core porque otros deportes sí los
        usan y porque un modelo futuro podría aprovecharlos.
        """
        return api_key == "btts"

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Resumen del catálogo, para logs."""
        core = self.get_core_markets()
        ext = self.get_extended_markets()
        return {
            "sport":             self.sport_id,
            "core_markets":      core,
            "extended_markets":  ext,
            "n_core":            len(core),
            "n_extended":        len(ext),
            "include_handicap":  self._include_handicap,
            "include_btts":      self._include_btts,
            "name_overrides":    self.market_name_overrides(),
            "preferred_lines":   {
                m: self.get_preferred_line(m) for m in ("1X2", "TOTAL", "BTTS")
            },
        }