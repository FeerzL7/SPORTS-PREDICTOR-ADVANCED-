"""
tests/sports/test_soccer.py

Suite del plugin de fútbol.

Qué cubre y por qué estas tres áreas
--------------------------------------
No se testea todo el plugin: se testea donde un fallo sería SILENCIOSO,
que es donde los tests aportan más que la revisión de código.

    LIQUIDACIÓN
        Un settlement equivocado produce un ledger con ROI incorrecto
        sin lanzar ninguna excepción. El fútbol tiene tres reglas que
        NFL no necesita —el empate como resultado, el push en línea
        entera y los mercados de primera parte— y las tres pueden
        fallar sin ruido.

    MATRIZ DE RESULTADOS
        Todos los mercados se derivan de ella. Un error en la
        corrección Dixon-Coles desplaza el empate, que es el 25% de los
        partidos y donde los books aplican más margen. La primera
        versión combinaba dos mecanismos que cuentan el mismo efecto y
        empujaba el empate al 30.2%, fuera del rango real.

    RECONCILIACIÓN DE NOMBRES
        'Man City' de football-data y 'Manchester City' de Understat
        deben converger. Si no lo hacen, el xG queda vacío y el modelo
        proyecta con la media de liga CREYENDO que tiene datos. Es el
        mismo fallo que en NFL dejó al QB titular marcado como suplente
        por no cruzar 'P.Mahomes' con 'Patrick Mahomes'.

Lo que NO se testea aquí: que las fuentes respondan (eso es
integración) ni que el modelo tenga ventaja (eso es el backtest).
"""

import pytest

from core.contracts.event import Event, EventStatus
from core.contracts.ledger import TERMINAL_RESULTS, BetLedgerEntry
from core.contracts.pick import CandidatePick
from core.pipeline.stage import SettlementProvider as SportSettlement
from core.tracking.protocols import SettlementProvider as FinancialSettlement

from sports.soccer.dixon_coles import build_score_matrix
from sports.soccer.h2h import is_derby
from sports.soccer.schedule import SoccerMatchInfo, build_match_id
from sports.soccer.settlement import (
    RESULT_LOSE, RESULT_PUSH, RESULT_VOID, RESULT_WIN,
    SoccerSettlementProvider,
)
from sports.soccer.teams import canonical_team, normalize_team, same_team


MATCH_ID = "epl_2024-11-09_manchester-city_arsenal"


# ── Infraestructura ──────────────────────────────────────────────────────────

class StubSchedule:
    """
    Calendario mínimo.

    El parámetro se llama `match_id` y no `mid` porque un Protocol de
    Python incluye los nombres en su contrato. Un doble que use otro
    nombre funciona en runtime —Python no los comprueba en llamadas
    posicionales— pero no satisface el contrato que dice satisfacer, y
    valida menos de lo que aparenta.
    """

    def __init__(self, match: SoccerMatchInfo | None) -> None:
        self._match = match

    def get_match_info(self, match_id: str) -> SoccerMatchInfo | None:
        return self._match if self._match and match_id == MATCH_ID else None


def match(home_goals=2, away_goals=1, ht_home=1, ht_away=0):
    return SoccerMatchInfo(
        match_id=MATCH_ID, comp_id="epl", season=2024, date="2024-11-09",
        home="manchester city", away="arsenal",
        home_display="Man City", away_display="Arsenal",
        home_goals=home_goals, away_goals=away_goals,
        ht_home=ht_home, ht_away=ht_away,
    )


def event():
    return Event(
        event_id=MATCH_ID, sport="soccer", league="Premier League",
        season_start=2024, season_end=2025, date="2024-11-09",
        start_time="2024-11-09T12:00:00Z",
        home_team_id="manchester city", away_team_id="arsenal",
        home_team="Man City", away_team="Arsenal",
        venue_id="manchester city", venue_name="",
        status=EventStatus.FINAL,
        provider_ids={"match_id": MATCH_ID, "competition": "epl"},
    )


def pick(market, selection, line=None, price=2.10):
    return CandidatePick(
        event=event(), market=market, selection=selection, line=line,
        price=price, model_prob_raw=0.50, market_prob=0.47,
        blended_prob=0.48,
    )


def ledger_entry(market, selection, price=2.10):
    """
    Entrada del ledger con la selección tal como se almacena.

    El CSV guarda la línea embebida ('over 2.5') para que sea legible
    sin cruzar columnas. El provider debe separarlas al liquidar.
    """
    return BetLedgerEntry(
        entry_id=f"{MATCH_ID}_{market}_{selection}".replace(" ", "_"),
        sport="soccer", league="Premier League", date="2024-11-09",
        event="Arsenal @ Man City", market=market, selection=selection,
        price=price, model_prob=0.50, ev=5.0, stake_pct=1,
        stake_amount=10.0, bankroll_before=1000.0, result="pending",
        model_version="soccer-v1.0.0", created_at="2024-11-09T10:00:00Z",
    )


def provider_for(home_goals, away_goals, **kwargs):
    return SoccerSettlementProvider(
        schedule_fetcher=StubSchedule(match(home_goals, away_goals, **kwargs))
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def city_wins():
    """Man City 2 - 1 Arsenal. Total 3, ambos marcaron, 1-0 al descanso."""
    return provider_for(2, 1)


@pytest.fixture
def city_result(city_wins):
    return city_wins.get_event_result(event())


@pytest.fixture
def drawn():
    """Empate 1-1. Total 2, ambos marcaron."""
    return provider_for(1, 1)


# ── Contratos ────────────────────────────────────────────────────────────────

class TestProtocols:

    def test_implements_sport_protocol(self, city_wins):
        assert isinstance(city_wins, SportSettlement)

    def test_implements_financial_protocol(self, city_wins):
        assert isinstance(city_wins, FinancialSettlement)


# ── El empate como resultado ─────────────────────────────────────────────────

class TestDraw:
    """
    La diferencia central con NFL.

    Allí el empate ocurre en el 0.4% de partidos y las casas devuelven
    el stake. Aquí es el 25% y la tercera opción del mercado principal:
    un pick de empate GANA cuando el partido termina igualado.

    Tratarlo como push destruiría el mercado donde el modelo busca más
    valor, que es precisamente donde los books aplican más margen.
    """

    def test_draw_pick_wins_on_draw(self, drawn):
        result = drawn.get_event_result(event())
        assert result["outcome"] == "D"
        assert drawn.settle_pick(pick("1X2", "draw"), result) == RESULT_WIN

    def test_home_pick_loses_on_draw(self, drawn):
        result = drawn.get_event_result(event())
        assert drawn.settle_pick(pick("1X2", "Man City"), result) == RESULT_LOSE

    def test_draw_pick_loses_when_decided(self, city_wins, city_result):
        assert city_wins.settle_pick(pick("1X2", "draw"), city_result) == RESULT_LOSE

    def test_draw_is_never_push(self, drawn, city_wins, city_result):
        """El empate nunca devuelve el stake, gane o pierda."""
        drawn_result = drawn.get_event_result(event())
        assert drawn.settle_pick(pick("1X2", "draw"), drawn_result) != RESULT_PUSH
        assert city_wins.settle_pick(pick("1X2", "draw"), city_result) != RESULT_PUSH

    @pytest.mark.parametrize("spelling", ["draw", "Draw", "X", "x", "empate", "tie"])
    def test_accepts_draw_spellings(self, drawn, spelling):
        """
        El pick puede venir de The Odds API, del modelo o del ledger,
        cada uno con su grafía.
        """
        result = drawn.get_event_result(event())
        assert drawn.settle_pick(pick("1X2", spelling), result) == RESULT_WIN


# ── 1X2 ──────────────────────────────────────────────────────────────────────

class TestMoneyline:

    def test_home_wins(self, city_wins, city_result):
        assert city_wins.settle_pick(pick("1X2", "Man City"), city_result) == RESULT_WIN

    def test_away_loses(self, city_wins, city_result):
        assert city_wins.settle_pick(pick("1X2", "Arsenal"), city_result) == RESULT_LOSE

    @pytest.mark.parametrize("spelling", [
        "Man City", "Manchester City", "manchester city", "MAN CITY",
    ])
    def test_team_spellings_resolve(self, city_wins, city_result, spelling):
        """
        La reconciliación de teams.py hace que las grafías de las tres
        fuentes converjan al mismo equipo.
        """
        assert city_wins.settle_pick(pick("1X2", spelling), city_result) == RESULT_WIN

    def test_unknown_team_voids(self, city_wins, city_result):
        """Liquidar un pick cuyo equipo no se identifica sería adivinar."""
        assert city_wins.settle_pick(
            pick("1X2", "Equipo Inexistente"), city_result
        ) == RESULT_VOID

    def test_exactly_one_side_wins(self, city_wins, city_result):
        """Las tres selecciones del 1X2 son mutuamente excluyentes."""
        outcomes = [
            city_wins.settle_pick(pick("1X2", s), city_result)
            for s in ("Man City", "draw", "Arsenal")
        ]
        assert outcomes.count(RESULT_WIN) == 1
        assert outcomes.count(RESULT_LOSE) == 2


# ── Totales ──────────────────────────────────────────────────────────────────

class TestTotals:

    @pytest.mark.parametrize("selection,line,expected", [
        ("over",  2.5, RESULT_WIN),    # total 3
        ("under", 2.5, RESULT_LOSE),
        ("over",  3.5, RESULT_LOSE),
        ("under", 3.5, RESULT_WIN),
        ("over",  1.5, RESULT_WIN),
    ])
    def test_over_under(self, city_wins, city_result, selection, line, expected):
        assert city_wins.settle_pick(
            pick("TOTAL", selection, line), city_result
        ) == expected

    @pytest.mark.parametrize("selection", ["over", "under"])
    def test_push_on_integer_line(self, city_wins, city_result, selection):
        """
        Total exacto contra línea entera: push en ambos lados.

        En fútbol no es marginal: la distribución de goles concentra
        ~25% de los partidos en exactamente dos, que es la línea entera
        más cotizada. Contarlo como derrota subestimaría el ROI de
        forma sistemática.
        """
        assert city_wins.settle_pick(
            pick("TOTAL", selection, 3.0), city_result
        ) == RESULT_PUSH

    def test_push_on_most_common_line(self, drawn):
        """Un partido de dos goles con línea 2.0 es el caso frecuente."""
        result = drawn.get_event_result(event())
        assert drawn.settle_pick(pick("TOTAL", "over", 2.0), result) == RESULT_PUSH

    @pytest.mark.parametrize("line", [0.5, 1.5, 2.5, 3.5, 4.5])
    def test_half_lines_never_push(self, city_wins, city_result, line):
        """Las líneas .5 no pueden hacer push: es su razón de ser."""
        assert city_wins.settle_pick(
            pick("TOTAL", "over", line), city_result
        ) in (RESULT_WIN, RESULT_LOSE)

    def test_missing_line_voids(self, city_wins, city_result):
        assert city_wins.settle_pick(
            pick("TOTAL", "over", None), city_result
        ) == RESULT_VOID

    def test_invalid_selection_voids(self, city_wins, city_result):
        assert city_wins.settle_pick(
            pick("TOTAL", "Man City", 2.5), city_result
        ) == RESULT_VOID


# ── BTTS ─────────────────────────────────────────────────────────────────────

class TestBTTS:

    def test_both_scored(self, city_wins, city_result):
        assert city_wins.settle_pick(pick("BTTS", "yes"), city_result) == RESULT_WIN
        assert city_wins.settle_pick(pick("BTTS", "no"), city_result) == RESULT_LOSE

    def test_clean_sheet(self):
        provider = provider_for(2, 0)
        result = provider.get_event_result(event())
        assert provider.settle_pick(pick("BTTS", "yes"), result) == RESULT_LOSE
        assert provider.settle_pick(pick("BTTS", "no"), result) == RESULT_WIN

    def test_goalless_draw(self):
        provider = provider_for(0, 0)
        result = provider.get_event_result(event())
        assert provider.settle_pick(pick("BTTS", "yes"), result) == RESULT_LOSE
        assert provider.settle_pick(pick("1X2", "draw"), result) == RESULT_WIN

    @pytest.mark.parametrize("spelling", ["yes", "Yes", "sí", "si", "true"])
    def test_yes_spellings(self, city_wins, city_result, spelling):
        assert city_wins.settle_pick(pick("BTTS", spelling), city_result) == RESULT_WIN

    def test_btts_has_no_push(self, city_wins, city_result):
        """O ambos marcaron o no: no hay tercer caso."""
        for selection in ("yes", "no"):
            assert city_wins.settle_pick(
                pick("BTTS", selection), city_result
            ) != RESULT_PUSH


# ── Primera parte ────────────────────────────────────────────────────────────

class TestFirstHalf:
    """
    Los mercados de primera parte SÍ se liquidan en fútbol.

    football-data publica el marcador al descanso. En el plugin NFL no
    teníamos el parcial y los mercados H1 devolvían 'void' de forma
    explícita — una capacidad que aquí sí existe y conviene no
    desperdiciar.
    """

    def test_first_half_1x2(self, city_wins, city_result):
        # Al descanso: 1-0
        assert city_wins.settle_pick(pick("1X2_H1", "Man City"), city_result) == RESULT_WIN
        assert city_wins.settle_pick(pick("1X2_H1", "draw"), city_result) == RESULT_LOSE

    def test_first_half_total(self, city_wins, city_result):
        # Al descanso: total 1
        assert city_wins.settle_pick(pick("TOTAL_H1", "under", 1.5), city_result) == RESULT_WIN
        assert city_wins.settle_pick(pick("TOTAL_H1", "over", 1.5), city_result) == RESULT_LOSE

    def test_first_half_differs_from_full_time(self, city_wins, city_result):
        """
        El resultado al descanso puede diferir del final.

        Aquí el local gana ambos, pero el total no: 1 al descanso y 3
        al final. Un settlement que confundiera ambos daría el mismo
        resultado en las dos líneas.
        """
        ht = city_wins.settle_pick(pick("TOTAL_H1", "over", 2.5), city_result)
        ft = city_wins.settle_pick(pick("TOTAL", "over", 2.5), city_result)
        assert ht == RESULT_LOSE and ft == RESULT_WIN

    def test_no_halftime_voids(self):
        provider = provider_for(2, 1, ht_home=None, ht_away=None)
        result = provider.get_event_result(event())
        assert provider.settle_pick(pick("1X2_H1", "Man City"), result) == RESULT_VOID
        assert provider.settle_pick(pick("TOTAL_H1", "over", 1.5), result) == RESULT_VOID


# ── Mercados no soportados ───────────────────────────────────────────────────

class TestUnsupported:

    @pytest.mark.parametrize("market", ["AH", "ASIAN_HANDICAP", "SPREAD"])
    def test_asian_handicap_voids(self, city_wins, city_result, market):
        """
        Las líneas de cuarto (-0.25, -0.75) dividen el stake en dos
        apuestas y producen push parciales que el contrato del ledger
        no representa.
        """
        assert city_wins.settle_pick(
            pick(market, "Man City", -0.75), city_result
        ) == RESULT_VOID

    def test_unknown_market_voids(self, city_wins, city_result):
        assert city_wins.settle_pick(
            pick("MERCADO_INVENTADO", "Man City"), city_result
        ) == RESULT_VOID


# ── Robustez ─────────────────────────────────────────────────────────────────

class TestRobustness:

    def test_unplayed_match_returns_none(self):
        unplayed = SoccerMatchInfo(
            match_id=MATCH_ID, comp_id="epl", season=2024, date="2024-11-09",
            home="manchester city", away="arsenal",
        )
        provider = SoccerSettlementProvider(schedule_fetcher=StubSchedule(unplayed))
        assert provider.get_event_result(event()) is None

    def test_unknown_match_returns_none(self, city_wins):
        ghost = event()
        object.__setattr__(ghost, "event_id", "no_existe")
        object.__setattr__(ghost, "provider_ids", {"match_id": "no_existe"})
        assert city_wins.get_event_result(ghost) is None

    def test_schedule_failure_returns_none(self):
        class Failing:
            def get_match_info(self, match_id: str):
                raise RuntimeError("fuente caída")

        provider = SoccerSettlementProvider(schedule_fetcher=Failing())
        assert provider.get_event_result(event()) is None

    @pytest.mark.parametrize("corrupt", [{}, {"home_goals": None}, {"x": 1}, None])
    def test_corrupt_result_voids(self, city_wins, corrupt):
        """Anular es preferible a registrar un resultado que no ocurrió."""
        assert city_wins.settle_pick(pick("1X2", "Man City"), corrupt) == RESULT_VOID

    def test_all_outcomes_terminal(self, city_wins, city_result):
        combos = [
            ("1X2", "Man City", None), ("1X2", "draw", None),
            ("TOTAL", "over", 3.0), ("TOTAL", "over", 2.5),
            ("BTTS", "yes", None), ("AH", "Man City", -0.5),
        ]
        outcomes = {
            city_wins.settle_pick(pick(m, s, l), city_result)
            for m, s, l in combos
        }
        assert outcomes.issubset(TERMINAL_RESULTS)


# ── Interfaz financiera ──────────────────────────────────────────────────────

class TestFinancialInterface:

    def test_settles_ledger_entry(self, city_wins):
        result = city_wins.get_result(ledger_entry("1X2", "Man City"))
        assert result is not None and result.result == RESULT_WIN

    def test_closing_price_enables_clv(self, city_wins):
        entry = ledger_entry("1X2", "Man City")
        city_wins.register_closing_price(entry.entry_id, 1.95)
        result = city_wins.get_result(entry)
        assert result.closing_price == 1.95
        clv = result.clv(pick_price=entry.price)
        assert clv is not None and clv > 0

    def test_sport_context_carries_result(self, city_wins):
        result = city_wins.get_result(ledger_entry("1X2", "Man City"))
        assert result.sport_context["outcome"] == "H"
        assert result.sport_context["total"] == 3


class TestInterfaceCoherence:
    """
    Las dos interfaces deben coincidir siempre.

    Si divergieran, el ROI del ledger no correspondería a los picks que
    el pipeline reportó. Este test detectó en el plugin NFL que TODO
    pick de total se liquidaba como void por la vía financiera, porque
    el ledger guarda 'over 47.0' y la lógica esperaba 'over' a secas.
    """

    @pytest.mark.parametrize("market,selection,line,ledger_selection", [
        ("1X2",   "Man City", None, "Man City"),
        ("1X2",   "Arsenal",  None, "Arsenal"),
        ("1X2",   "draw",     None, "draw"),
        ("TOTAL", "over",     2.5,  "over 2.5"),
        ("TOTAL", "over",     3.0,  "over 3.0"),      # push
        ("TOTAL", "under",    3.5,  "under 3.5"),
        ("BTTS",  "yes",      None, "yes"),
        ("BTTS",  "no",       None, "no"),
    ])
    def test_sport_and_financial_agree(
        self, city_wins, city_result, market, selection, line, ledger_selection
    ):
        sport = city_wins.settle_pick(pick(market, selection, line), city_result)
        financial = city_wins.get_result(
            ledger_entry(market, ledger_selection)
        ).result
        assert sport == financial, (
            f"{market} {ledger_selection}: deportiva={sport} "
            f"vs financiera={financial}"
        )

    def test_total_push_not_voided_by_parsing(self, city_wins):
        """Regresión del bug de NFL: la línea embebida en la selección."""
        result = city_wins.get_result(ledger_entry("TOTAL", "over 3.0"))
        assert result.result == RESULT_PUSH


# ── Matriz de resultados ─────────────────────────────────────────────────────

class TestScoreMatrix:
    """
    Todos los mercados se derivan de una sola matriz.

    Calcularlos por separado no garantizaría coherencia: podría salir
    un P(BTTS) mayor que P(over 1.5), que es imposible.
    """

    def test_normalized(self):
        matrix = build_score_matrix(1.53, 1.25)
        assert abs(sum(sum(row) for row in matrix.grid) - 1.0) < 1e-9

    def test_outcomes_sum_to_one(self):
        outcomes = build_score_matrix(1.53, 1.25).outcome_probabilities()
        assert abs(sum(outcomes.values()) - 1.0) < 1e-5

    def test_draw_in_european_range(self):
        """
        Un partido equilibrado debe dar el empate en el rango observado.

        No se exige a TODOS los partidos: el ~25% es la media de liga,
        y un grande contra un recién ascendido debe tener el empate
        bajo. Exigirle 22% sería pedirle al modelo que ignore la
        diferencia de nivel.
        """
        draw = build_score_matrix(1.45, 1.35).outcome_probabilities()["draw"]
        assert 0.22 <= draw <= 0.30

    def test_draw_decreases_with_imbalance(self):
        """El invariante que sí aplica a todos los partidos."""
        balanced = build_score_matrix(1.40, 1.40).outcome_probabilities()["draw"]
        skewed = build_score_matrix(2.60, 0.70).outcome_probabilities()["draw"]
        assert skewed < balanced

    def test_dixon_coles_increases_draws(self):
        """
        La corrección redistribuye hacia los empates de marcador bajo.

        CONSERVA la masa total de marcadores bajos: sube 0-0 y 1-1,
        baja 1-0 y 0-1. Lo que cambia es el reparto entre empates y
        victorias mínimas, no el total.
        """
        without = build_score_matrix(1.5, 1.2, rho=0.0, lambda_3=0.0)
        with_dc = build_score_matrix(1.5, 1.2, rho=-0.05, lambda_3=0.0)
        assert (with_dc.outcome_probabilities()["draw"]
                > without.outcome_probabilities()["draw"])
        assert abs(with_dc.low_score_mass - without.low_score_mass) < 1e-6

    def test_dixon_coles_only_touches_low_scores(self):
        without = build_score_matrix(1.5, 1.2, rho=0.0, lambda_3=0.0)
        with_dc = build_score_matrix(1.5, 1.2, rho=-0.05, lambda_3=0.0)
        for home in range(2, 6):
            for away in range(2, 6):
                assert abs(with_dc.grid[home][away] - without.grid[home][away]) < 0.01

    def test_markets_are_coherent(self):
        """P(BTTS) nunca puede superar P(over 1.5)."""
        matrix = build_score_matrix(1.53, 1.25)
        assert matrix.btts() <= matrix.total_over(1.5) + 1e-6
        assert abs(matrix.total_over(0.5) - (1 - matrix.exact_score(0, 0))) < 1e-4

    def test_integer_line_has_push_mass(self):
        matrix = build_score_matrix(1.40, 1.20)
        over, under = matrix.total_over(2.0), matrix.total_under(2.0)
        push = matrix.total_push(2.0)
        assert push > 0.15
        assert abs(over + under + push - 1.0) < 1e-5

    def test_half_line_has_no_push(self):
        assert build_score_matrix(1.53, 1.25).total_push(2.5) == 0.0

    @pytest.mark.parametrize("home,away", [
        (0.0, 0.0), (99.0, 99.0), (-5.0, 2.0), (None, None),
    ])
    def test_extreme_lambdas_produce_valid_matrix(self, home, away):
        matrix = build_score_matrix(home, away)
        assert abs(sum(sum(r) for r in matrix.grid) - 1.0) < 1e-6
        assert abs(sum(matrix.outcome_probabilities().values()) - 1.0) < 1e-5


# ── Reconciliación de nombres ────────────────────────────────────────────────

class TestTeamReconciliation:
    """
    Sin esto, cruzar el xG de Understat con los resultados de
    football-data falla en SILENCIO: el equipo no aparece, sus métricas
    quedan vacías y el modelo proyecta con la media de liga creyendo
    que tiene datos.
    """

    @pytest.mark.parametrize("raw,expected", [
        ("Atlético Madrid", "atletico madrid"),
        ("FC Barcelona",    "barcelona"),
        ("VfB Stuttgart",   "stuttgart"),
        ("Real Madrid CF",  "real madrid"),
        ("Nott'm Forest",   "nott m forest"),
    ])
    def test_normalization(self, raw, expected):
        assert normalize_team(raw) == expected

    @pytest.mark.parametrize("football_data,understat,comp", [
        ("Man City",      "Manchester City",     "epl"),
        ("Man United",    "Manchester United",   "epl"),
        ("Nott'm Forest", "Nottingham Forest",   "epl"),
        ("Ath Madrid",    "Atletico Madrid",     "laliga"),
        ("Ath Bilbao",    "Athletic Club",       "laliga"),
        ("M'gladbach",    "Borussia M.Gladbach", "bundesliga"),
        ("Paris SG",      "Paris Saint Germain", "ligue1"),
        ("Milan",         "AC Milan",            "seriea"),
    ])
    def test_sources_converge(self, football_data, understat, comp):
        assert same_team(football_data, understat, comp)

    def test_similar_names_do_not_collide(self):
        """
        El caso que descarta la coincidencia difusa.

        'Ath Madrid' y 'Ath Bilbao' comparten prefijo y son clubes
        distintos. Emparejarlos daría métricas plausibles pero del
        rival equivocado, y nada en el sistema lo detectaría.
        """
        assert canonical_team("Ath Madrid", "laliga") != canonical_team("Ath Bilbao", "laliga")
        assert canonical_team("Sheffield United", "epl") != canonical_team("Sheffield Weds", "epl")

    def test_empty_never_matches(self):
        assert not same_team("", "Arsenal", "epl")
        assert not same_team("", "", "epl")

    def test_match_id_stable_across_spellings(self):
        """
        Un id inestable rompería la liquidación: el pick del ledger
        apuntaría a un partido que el settlement no encuentra.
        """
        a = build_match_id("epl", "2024-11-09", "Man City", "Arsenal")
        b = build_match_id("epl", "2024-11-09", "Manchester City", "Arsenal")
        assert a == b == "epl_2024-11-09_manchester-city_arsenal"


# ── Derbis ───────────────────────────────────────────────────────────────────

class TestDerbies:

    @pytest.mark.parametrize("home,away,comp", [
        ("Arsenal", "Tottenham", "epl"),
        ("Everton", "Liverpool", "epl"),
        ("Ath Madrid", "Real Madrid", "laliga"),
        ("Milan", "Inter", "seriea"),
        ("Dortmund", "Schalke 04", "bundesliga"),
        ("Lyon", "St Etienne", "ligue1"),
    ])
    def test_known_derbies(self, home, away, comp):
        assert is_derby(home, away, comp)
        assert is_derby(away, home, comp), "la detección debe ser simétrica"

    def test_distant_rivalry_is_not_a_derby(self):
        """
        El efecto modelable viene de la proximidad geográfica —afición
        visitante, desplazamiento nulo— no de la rivalidad histórica.
        Madrid y Barcelona distan 600 km.
        """
        assert not is_derby("Real Madrid", "Barcelona", "laliga")

    def test_same_team_is_not_a_derby(self):
        assert not is_derby("Arsenal", "Arsenal", "epl")