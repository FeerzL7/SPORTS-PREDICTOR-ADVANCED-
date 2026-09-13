"""
sports/nfl/markets.py

NFLMarketDefinitions: catálogo de mercados NFL para el pipeline.
Implementa core/pipeline/stage.py:MarketDefinitions.

Diferencia con el plugin MLB: consume el MarketRegistry
---------------------------------------------------------
MLBMarketDefinitions mantiene sus listas de mercados hardcodeadas y no
consulta core/odds/market_registry.py. Funciona, pero duplica una
información que el Core ya centraliza y obliga a editar dos sitios cada
vez que cambia el catálogo.

Este módulo consume el registry directamente. Los mercados NFL se
registraron allí en la tarea 10.2, incluida la corrección de la clave
`sports` para que use el sport_id interno ('nfl') en vez del
identificador de The Odds API — sin esa corrección,
`get_extended_markets('nfl')` habría devuelto una lista vacía y el
plugin no habría podido pedir ni una sola prop.

La línea preferida: por qué NFL devuelve None en SPREAD
---------------------------------------------------------
En MLB `get_preferred_line('SPREAD')` devuelve -1.5 porque el runline
es una línea FIJA: todos los partidos se cotizan al mismo handicap y
OddsNormalizer puede seleccionar esa línea con certeza.

En NFL el spread es variable — va de -1 a -14 según el emparejamiento.
No existe una "línea preferida" que tenga sentido para todos los
partidos, así que devolver un número concreto haría que
OddsNormalizer descartara las cuotas de casi todos los eventos.

Devolver None indica "usa la línea de mayor consenso entre casas",
que es el comportamiento correcto para un mercado de línea variable.

Números clave: lo que sí es específico de NFL
-----------------------------------------------
Aunque no haya línea preferida, la distribución de márgenes de victoria
en NFL está fuertemente concentrada en dos valores:

    Margen      Frecuencia aproximada
      3            ~15%
      7            ~9%
      6            ~6%
      10           ~6%
      4            ~5%

El 3 y el 7 son los valores de field goal y touchdown, y esa
concentración tiene una consecuencia práctica: cruzar uno de esos
números vale mucho más que medio punto en cualquier otra parte de la
distribución. Pasar de -3 a -3.5 elimina la posibilidad de push en el
margen más frecuente del deporte; pasar de -5 a -5.5 apenas cambia
nada.

`is_key_number()` y `key_number_penalty()` exponen esa información
para que el detector de movimiento de línea y el modelo de proyección
puedan ponderarla. No forman parte del Protocol — son extensiones
específicas del plugin que el Core no necesita conocer.
"""

from __future__ import annotations

from core.odds.market_registry import MarketRegistry, default_registry


# ── Números clave de NFL ─────────────────────────────────────────────────────
#
# Frecuencia histórica de cada margen de victoria. Los valores son
# aproximados y estables entre temporadas: derivan de la estructura de
# puntuación del deporte (3 por field goal, 7 por touchdown con
# conversión), no de la composición concreta de la liga.
_MARGIN_FREQUENCY: dict[int, float] = {
    3:  0.150,
    7:  0.090,
    6:  0.060,
    10: 0.058,
    4:  0.050,
    1:  0.043,
    14: 0.042,
    17: 0.030,
    13: 0.029,
    2:  0.020,
}

# Márgenes que concentran suficiente masa para considerarse clave.
# Cruzarlos con la línea cambia el resultado de un porcentaje
# apreciable de partidos.
_KEY_NUMBERS = frozenset({3, 7})

# Umbral de frecuencia para la categoría secundaria.
_SECONDARY_KEY_THRESHOLD: float = 0.05


class NFLMarketDefinitions:
    """
    Catálogo de mercados NFL.

    Parámetros
    ----------
    registry         -- MarketRegistry a consultar. None usa el
                        registro por defecto del Core, que ya incluye
                        los mercados NFL desde la tarea 10.2.
    include_props    -- Si False, excluye las props de jugador de los
                        mercados EXTENDED. Útil cuando el plan de la
                        API no las cubre o para ahorrar créditos: las
                        props se piden por evento, no en la request
                        principal.
    include_periods  -- Si False, excluye los mercados de primera mitad.
                        Por defecto False porque NFLSettlementProvider
                        no puede liquidarlos con el marcador final —
                        pedirlos gastaría créditos en picks que
                        terminarían en 'void'.
    """

    sport_id: str = "nfl"

    def __init__(
        self,
        registry:        MarketRegistry | None = None,
        include_props:   bool = True,
        include_periods: bool = False,
    ) -> None:
        self._registry        = registry or default_registry()
        self._include_props   = include_props
        self._include_periods = include_periods

    # ── MarketDefinitions Protocol ────────────────────────────────────────────

    def get_core_markets(self) -> list[str]:
        """
        Mercados CORE: una sola request cubre todos los partidos.

        Para NFL son los tres universales — h2h, totals y spreads.
        El registry los filtra por sport_id, así que h2h_3_way (que se
        restringió a soccer y hockey en la tarea 10.2) no aparece aquí
        y no desperdicia espacio en la request.
        """
        return self._registry.get_core_markets(self.sport_id)

    def get_extended_markets(self) -> list[str]:
        """
        Mercados EXTENDED: endpoint por evento, mayor costo.

        Se filtran según los flags del constructor. Excluir los
        mercados de periodo por defecto es deliberado: pedirlos
        consumiría créditos para generar picks que
        NFLSettlementProvider liquida como 'void' por no poder
        resolverlos con el marcador final.
        """
        markets = self._registry.get_extended_markets(self.sport_id)

        if not self._include_props:
            markets = [m for m in markets if not _is_player_prop(m)]

        if not self._include_periods:
            markets = [m for m in markets if not _is_period_market(m)]

        return markets

    def get_preferred_line(self, market: str) -> float | None:
        """
        Línea preferida para un mercado. None en todos los casos de NFL.

        En MLB este método devuelve -1.5 para SPREAD porque el runline
        es fijo y OddsNormalizer puede seleccionar esa línea con
        certeza en cualquier partido.

        El spread de NFL es variable — de -1 a -14 según el
        emparejamiento — así que no existe un valor que tenga sentido
        para todos los eventos. Devolver uno concreto haría que
        OddsNormalizer descartara las cuotas de casi todos los
        partidos por no encontrar esa línea.

        None indica "usar la línea de mayor consenso entre casas", que
        es el comportamiento correcto para líneas variables. Lo mismo
        aplica a los totales, que van de 35 a 55.
        """
        return None

    # ── Extensiones específicas de NFL ────────────────────────────────────────
    #
    # No forman parte del Protocol: el Core no necesita conocer los
    # números clave. Los consumen el detector de movimiento de línea y
    # el modelo de proyección del propio plugin.

    @staticmethod
    def is_key_number(line: float) -> bool:
        """
        True si la línea coincide con un número clave de NFL.

        El 3 y el 7 concentran el 24% de todos los márgenes de victoria
        del deporte, porque son los valores de field goal y touchdown.

        Importa para el movimiento de línea: un desplazamiento de -2.5
        a -3.5 atraviesa el margen más frecuente y cambia el resultado
        de aproximadamente el 15% de los partidos. El mismo medio punto
        entre -5.5 y -6.5 apenas afecta al 6%.
        """
        return abs(line) in _KEY_NUMBERS

    @staticmethod
    def margin_frequency(margin: float) -> float:
        """
        Frecuencia histórica aproximada de un margen de victoria.

        Devuelve 0.0 para márgenes no tabulados, que son los poco
        frecuentes. La suma de los tabulados no llega a 1.0 por
        diseño: solo se listan los que concentran masa apreciable.
        """
        try:
            return _MARGIN_FREQUENCY.get(int(abs(margin)), 0.0)
        except (ValueError, TypeError):
            return 0.0

    @classmethod
    def key_number_penalty(cls, from_line: float, to_line: float) -> float:
        """
        Masa de probabilidad atravesada al mover la línea.

        Suma la frecuencia de todos los márgenes que quedan entre las
        dos líneas. Cuantifica el coste real de aceptar un movimiento
        adverso: mover de -2.5 a -3.5 cruza el margen 3 y cuesta un 15%
        de los partidos, mientras que de -5.5 a -6.5 cruza el 6 y
        cuesta un 6%.

        El ValueEngine puede usarlo para decidir si un movimiento de
        línea invalida un pick: perder medio punto no es igual de grave
        en todas las zonas de la distribución.
        """
        lo, hi = sorted((abs(from_line), abs(to_line)))
        return round(
            sum(
                freq for margin, freq in _MARGIN_FREQUENCY.items()
                if lo < margin < hi
            ),
            4,
        )

    @classmethod
    def key_numbers(cls) -> list[int]:
        """Números clave de NFL, ordenados por frecuencia descendente."""
        return sorted(
            _KEY_NUMBERS,
            key=lambda n: _MARGIN_FREQUENCY.get(n, 0.0),
            reverse=True,
        )

    @classmethod
    def secondary_key_numbers(cls) -> list[int]:
        """
        Márgenes con frecuencia apreciable sin llegar a clave.

        Entre el 5% y el umbral de los números clave. Son relevantes
        para el movimiento de línea aunque no justifiquen un
        tratamiento especial en la proyección.
        """
        return sorted(
            n for n, f in _MARGIN_FREQUENCY.items()
            if n not in _KEY_NUMBERS and f >= _SECONDARY_KEY_THRESHOLD
        )

    # ── Diagnóstico ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Resumen del catálogo, para logs y depuración."""
        core = self.get_core_markets()
        ext  = self.get_extended_markets()
        return {
            "sport":            self.sport_id,
            "core_markets":     core,
            "extended_markets": ext,
            "n_core":           len(core),
            "n_extended":       len(ext),
            "include_props":    self._include_props,
            "include_periods":  self._include_periods,
            "key_numbers":      self.key_numbers(),
            "preferred_lines":  {
                m: self.get_preferred_line(m)
                for m in ("SPREAD", "TOTAL", "ML")
            },
        }


# ── Utilidades de módulo ──────────────────────────────────────────────────────

def _is_player_prop(api_key: str) -> bool:
    """True si el mercado es una prop de jugador."""
    return api_key.lower().startswith("player_")


def _is_period_market(api_key: str) -> bool:
    """True si el mercado corresponde a un periodo del partido."""
    key = api_key.lower()
    return key.endswith("_h1") or key.endswith("_h2") or "_q" in key