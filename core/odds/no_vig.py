"""
core/odds/no_vig.py

Normalización no-vig para mercados de apuestas con N selecciones.

Contexto
---------
Toda cuota decimal lleva vig (margen de la casa) incorporado. Una cuota
de 1.91 en un mercado binario perfectamente simétrico implica una
probabilidad del 52.36% para cada lado — pero las dos probabilidades
suman 104.71%, no 100%. Ese 4.71% es el overround: lo que la casa
retiene en expectativa.

Usar la probabilidad implícita bruta (1/price) como "probabilidad de
mercado" en el cálculo de EV produce EV artificialmente inflado, porque
se compara una probabilidad del modelo contra una probabilidad de mercado
que también contiene vig. El sistema MLB-PREDICTOR-ADVANCED cometió este
error de forma sistemática, generando picks con EV aparente de +40 que en
producción resultaron en ROI de -21.56% (documentado en MLB_EDGE_AUDIT.md).

La corrección es normalizar: dividir cada probabilidad implícita entre la
suma de todas las probabilidades implícitas del mismo mercado. El resultado
son probabilidades que suman exactamente 1.0 — la estimación del mercado
"sin casa".

Fórmula (idéntica para 2 o N selecciones)
-------------------------------------------
    p_i_implied  = 1 / price_i
    overround    = Σ p_i_implied
    p_i_no_vig   = p_i_implied / overround

Propiedades garantizadas post-normalización:
    - Σ p_i_no_vig = 1.0  (exacto dentro de precisión float)
    - 0 < p_i_no_vig < 1  para todo i (garantizado porque price > 1.0
      es invariante de MarketOdds, y overround > 0 siempre que exista
      al menos una selección)

Interfaz pública
-----------------
    no_vig_probabilities(selections) → list[MarketOdds]
        Función principal. Retorna nuevas instancias con no_vig_prob
        poblado. Orden de entrada preservado (índice i entrada = índice
        i salida), para que el consumidor pueda usar zip() directamente.

    implied_overround(selections) → float
        Overround del mercado (suma de probabilidades implícitas brutas).
        Métrica de mercado independiente, útil para filtrar mercados con
        vig excesivo antes de calcular EV.

Diseño: por qué retorna list[MarketOdds] y no dict
-----------------------------------------------------
MarketOdds es frozen=True — inmutabilidad del hecho de mercado. En vez
de mutar ni retornar un mapping separado {selection: float} que podría
desincronizarse, se usa dataclasses.replace() para crear nuevas instancias
con todos los campos iguales excepto no_vig_prob. El consumidor recibe
objetos completos y tipados, coherentes con el contrato original.

Diseño: validación de homogeneidad
-------------------------------------
Una list[MarketOdds] puede mezclar accidentalmente selecciones de mercados
distintos (ML home + TOTAL over del mismo evento) o de eventos distintos.
El no-vig calculado sobre un grupo mezclado produciría probabilidades sin
sentido que se propagarían silenciosamente al EV. _validate_selections()
verifica homogeneidad de event_id y market antes de calcular — fail fast
en el punto de entrada, no tres stages después.

Diseño: caso N=1 (outright con cotización parcial)
----------------------------------------------------
Un mercado de golf outright puede tener 150 jugadores pero solo N<<150
cargados en el snapshot actual. El no-vig calculado sobre un subconjunto
es matemáticamente correcto (las N probs suman 1.0) pero estadísticamente
engañoso (la probabilidad de cada jugador queda inflada por la ausencia
de los demás). Este módulo retorna el resultado sin error — es
responsabilidad del motor de valor decidir si filtrar mercados con
cobertura insuficiente, no de la función matemática que calcula no-vig.
"""

from __future__ import annotations

from dataclasses import replace

from core.contracts.market_odds import MarketOdds


# ── Validación interna ────────────────────────────────────────────────────────

def _validate_selections(selections: list[MarketOdds]) -> None:
    """
    Verifica que la lista de selecciones sea válida para calcular no-vig:

    1. No vacía — una lista vacía no tiene overround calculable.
    2. Todas comparten el mismo event_id — mezzclar eventos distintos
       produce probabilidades sin significado estadístico.
    3. Todas comparten el mismo market — mezclar ML con TOTAL del mismo
       evento es el error más común en parseo de APIs de odds.

    price > 1.0 NO se valida aquí porque es un invariante del contrato
    MarketOdds.__post_init__: si un objeto MarketOdds existe, su precio
    ya fue validado en construcción. Documentarlo explícitamente evita
    una doble validación redundante.

    Raises
    ------
    ValueError
        Con descripción detallada de qué campo viola la homogeneidad,
        incluyendo los valores conflictivos encontrados — para que el
        mensaje sea directamente accionable en debugging.
    """
    if not selections:
        raise ValueError(
            "no_vig_probabilities() recibió una lista vacía. "
            "Se necesita al menos una selección para calcular el overround."
        )

    first = selections[0]

    event_ids = {s.event_id for s in selections}
    if len(event_ids) > 1:
        raise ValueError(
            f"Selecciones de eventos distintos mezcladas en el mismo "
            f"cálculo no-vig. event_ids encontrados: {sorted(event_ids)}. "
            f"Cada llamada a no_vig_probabilities() debe contener "
            f"selecciones de un único event_id."
        )

    markets = {s.market for s in selections}
    if len(markets) > 1:
        raise ValueError(
            f"Selecciones de mercados distintos mezcladas en el mismo "
            f"cálculo no-vig para event_id='{first.event_id}'. "
            f"Mercados encontrados: {sorted(markets)}. "
            f"Cada llamada a no_vig_probabilities() debe contener "
            f"selecciones de un único market (ML, TOTAL, SPREAD, 1X2, etc.)."
        )


# ── API pública ───────────────────────────────────────────────────────────────

def no_vig_probabilities(
    selections: list[MarketOdds],
) -> list[MarketOdds]:
    """
    Calcula la probabilidad no-vig para cada selección de un mercado y
    retorna nuevas instancias de MarketOdds con no_vig_prob poblado.

    El orden de la lista de retorno es idéntico al de entrada: el
    elemento i de la salida corresponde al elemento i de la entrada.
    Esto permite al consumidor usar zip(original, resultado) directamente
    sin búsqueda por selection.

    Parámetros
    ----------
    selections
        Lista de MarketOdds pertenecientes al MISMO event_id y MISMO
        market. Puede contener 2 selecciones (mercado binario: ML, RL,
        TOTAL) o N selecciones (Soccer 1X2, Golf outright, etc.).
        El orden se preserva en la salida.

    Retorna
    -------
    list[MarketOdds]
        Nuevas instancias (frozen dataclass) con no_vig_prob calculado.
        Todos los demás campos son idénticos a los originales.
        La suma de todos los no_vig_prob es 1.0 (dentro de precisión
        float estándar).

    Raises
    ------
    ValueError
        Si la lista está vacía, o si las selecciones mezclan event_id
        o market distintos. Ver _validate_selections().

    Ejemplo
    -------
    >>> over  = MarketOdds(event_id='e1', market='TOTAL', selection='over',
    ...                    line=8.5, price=1.91, bookmaker='Pinnacle',
    ...                    timestamp='2026-07-01T18:00:00Z')
    >>> under = MarketOdds(event_id='e1', market='TOTAL', selection='under',
    ...                    line=8.5, price=1.91, bookmaker='Pinnacle',
    ...                    timestamp='2026-07-01T18:00:00Z')
    >>> result = no_vig_probabilities([over, under])
    >>> result[0].no_vig_prob  # 0.5 exacto para cuotas simétricas
    0.5
    >>> result[1].no_vig_prob
    0.5
    >>> sum(r.no_vig_prob for r in result)
    1.0
    """
    _validate_selections(selections)

    # implied_prob ya está calculado por MarketOdds.__post_init__
    # como round(1/price, 4). Sumamos directamente sin recalcular.
    overround = sum(s.implied_prob for s in selections)  # type: ignore[misc]

    # overround > 0 está garantizado porque:
    #   - selections no está vacía (validado arriba)
    #   - cada implied_prob = 1/price > 0 porque price > 1.0 (invariante)
    # La guarda defensiva existe para documentar la garantía, no porque
    # pueda ocurrir en práctica con el contrato actual.
    if overround <= 0:
        raise ValueError(
            f"Overround calculado es {overround} — no positivo. "
            f"Esto no debería ocurrir si todos los MarketOdds tienen "
            f"price > 1.0 (invariante del contrato). Revisar los datos."
        )

    result = []
    for sel in selections:
        no_vig = round(sel.implied_prob / overround, 6)  # type: ignore[operator]
        result.append(replace(sel, no_vig_prob=no_vig))

    return result


def implied_overround(selections: list[MarketOdds]) -> float:
    """
    Calcula el overround (margen de la casa) de un mercado.

    overround = Σ (1/price_i) para todas las selecciones del mercado.

    Un mercado "justo" sin vig tendría overround = 1.0. En la práctica:
        - Mercados binarios típicos (ML, TOTAL): 1.04 – 1.08
        - Mercados de 3 vías (1X2 soccer): 1.06 – 1.12
        - Mercados outright (golf, tennis): 1.15 – 1.40+

    El overround es una métrica de mercado independiente del cálculo
    no-vig. El motor de valor puede usarla para filtrar mercados con
    vig excesivo (ej. rechazar si overround > 1.10) antes de calcular
    EV — un filtro que el sistema MLB original no tenía y que explica
    parte del underperformance en picks con cuotas muy cerradas.

    Parámetros
    ----------
    selections
        Lista de MarketOdds del mismo event_id y market. Las mismas
        restricciones de homogeneidad que no_vig_probabilities().

    Retorna
    -------
    float
        Suma de probabilidades implícitas brutas. Redondeado a 6
        decimales para consistencia con no_vig_prob.

    Raises
    ------
    ValueError
        Mismas condiciones que _validate_selections().

    Ejemplo
    -------
    >>> # Mercado binario típico con vig simétrico del 4.7%
    >>> over  = MarketOdds(..., price=1.91, ...)
    >>> under = MarketOdds(..., price=1.91, ...)
    >>> implied_overround([over, under])
    1.047120  # 1/1.91 + 1/1.91 = 0.52356 + 0.52356
    """
    _validate_selections(selections)
    return round(sum(s.implied_prob for s in selections), 6)  # type: ignore[misc]