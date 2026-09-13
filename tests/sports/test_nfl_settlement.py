"""
tests/sports/test_nfl_settlement.py

Suite de liquidación NFL: empates, pushes en números clave y tiempo extra.

Qué cubre y por qué
---------------------
NFLSettlementProvider tiene tres reglas que el equivalente de MLB no
necesita, y las tres pueden fallar silenciosamente — produciendo un
ledger con ROI incorrecto sin lanzar ninguna excepción:

    EMPATES        MLB no puede empatar (se juegan entradas extra hasta
                   definir). NFL empata en ~0.4% de partidos tras OT, y
                   las casas estadounidenses lo resuelven como push en
                   moneyline.

    PUSH EN SPREAD El runline de MLB es siempre ±1.5, así que el push
                   es imposible por construcción. Las líneas de NFL se
                   concentran en 3 y 7 — los valores de field goal y
                   touchdown — que son los márgenes de victoria más
                   frecuentes del deporte. Contarlos como derrota
                   subestimaría el ROI de forma sistemática.

    PUSH EN TOTAL  Los totales enteros (44, 47) son habituales en NFL,
                   a diferencia de MLB donde predominan los .5.

Además se verifica la COHERENCIA entre las dos interfaces que el
provider implementa: la deportiva (settle_pick, desde el runner) y la
financiera (get_result, desde el tracker). Si divergieran, el ROI
registrado no correspondería a los picks reportados — y ese test es
justamente el que detectó en la tarea 10.12 que todo pick de TOTAL se
liquidaba como void por la vía financiera.
"""

import pytest

from core.contracts.event import Event, EventStatus
from core.contracts.ledger import BetLedgerEntry, TERMINAL_RESULTS
from core.contracts.pick import CandidatePick
from core.pipeline.stage import SettlementProvider as SportSettlement
from core.tracking.protocols import SettlementProvider as FinancialSettlement

from sports.nfl.schedule import NFLGameInfo
from sports.nfl.settlement import (
    NFLSettlementProvider,
    RESULT_WIN, RESULT_LOSE, RESULT_PUSH, RESULT_VOID,
)


# ── Infraestructura ──────────────────────────────────────────────────────────

GAME_ID = "2026_10_KC_LAC"


class StubSchedule:
    """Calendario mínimo: solo lo que el settlement consulta."""

    def __init__(self, games):
        self._games = {g.game_id: g for g in games}

    def get_game_info(self, game_id):
        return self._games.get(game_id)


def game(home="LAC", away="KC", home_score=None, away_score=None, overtime=False):
    """Partido con el marcador dado. Sin marcador = aún no jugado."""
    return NFLGameInfo(
        game_id=GAME_ID, season=2026, week=10, game_type="REG",
        gameday="2026-11-08", home_team=home, away_team=away,
        home_score=home_score, away_score=away_score, overtime=overtime,
    )


def event(home="LAC", away="KC"):
    return Event(
        event_id=GAME_ID, sport="nfl", league="NFL",
        season_start=2026, season_end=2027,
        date="2026-11-08", start_time="2026-11-08T18:00:00Z",
        home_team_id=home, away_team_id=away,
        home_team=home, away_team=away,
        venue_id=home, venue_name="",
        status=EventStatus.FINAL,
        provider_ids={"nfl_game_id": GAME_ID},
    )


def pick(market, selection, line=None, price=1.91):
    return CandidatePick(
        event=event(), market=market, selection=selection, line=line,
        price=price, model_prob_raw=0.55, market_prob=0.52, blended_prob=0.54,
    )


def ledger_entry(market, selection, price=1.91):
    """
    Entry con la selección tal como la guarda el ledger.

    El CSV almacena la línea embebida en `selection` ('over 47.0',
    'LAC -3.5') para que sea legible sin cruzar columnas. El provider
    debe separarlas al liquidar.
    """
    return BetLedgerEntry(
        entry_id=f"{GAME_ID}_{market}_{selection}".replace(" ", "_"),
        sport="nfl", league="NFL", date="2026-11-08",
        event="KC @ LAC", market=market, selection=selection,
        price=price, model_prob=0.55, ev=5.0,
        stake_pct=1, stake_amount=10.0, bankroll_before=1000.0,
        result="pending", model_version="nfl-v1.0.0",
        created_at="2026-11-08T14:00:00Z",
    )


def provider_for(home_score, away_score, overtime=False, **kwargs):
    """Provider cuyo calendario tiene un partido con ese marcador."""
    return NFLSettlementProvider(
        schedule_fetcher=StubSchedule([
            game(home_score=home_score, away_score=away_score, overtime=overtime)
        ]),
        **kwargs,
    )


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def lac_by_7():
    """LAC 27 - KC 20. Margen +7, total 47 — ambos números clave."""
    return provider_for(home_score=27, away_score=20)


@pytest.fixture
def result_lac_by_7(lac_by_7):
    return lac_by_7.get_event_result(event())


@pytest.fixture
def tied():
    """Empate 23-23 tras tiempo extra."""
    return provider_for(home_score=23, away_score=23, overtime=True)


# ── Contratos ────────────────────────────────────────────────────────────────

class TestProtocols:
    """El provider implementa las dos interfaces SettlementProvider."""

    def test_implements_sport_protocol(self, lac_by_7):
        assert isinstance(lac_by_7, SportSettlement)

    def test_implements_financial_protocol(self, lac_by_7):
        assert isinstance(lac_by_7, FinancialSettlement)


# ── Disponibilidad del resultado ─────────────────────────────────────────────

class TestEventResult:

    def test_unplayed_game_returns_none(self):
        """
        Un partido sin marcador devuelve None, no un dict con status.

        El contrato del Core usa None para señalar 'todavía no
        liquidable'. Así el runner salta el pick sin registrar un
        intento fallido.
        """
        p = NFLSettlementProvider(schedule_fetcher=StubSchedule([game()]))
        assert p.get_event_result(event()) is None

    def test_unknown_game_returns_none(self, lac_by_7):
        ghost = event()
        object.__setattr__(ghost, "event_id", "NO_EXISTE")
        object.__setattr__(ghost, "provider_ids", {"nfl_game_id": "NO_EXISTE"})
        assert lac_by_7.get_event_result(ghost) is None

    def test_final_game_returns_scores(self, result_lac_by_7):
        assert result_lac_by_7["home_score"] == 27
        assert result_lac_by_7["away_score"] == 20
        assert result_lac_by_7["margin"] == 7
        assert result_lac_by_7["total"] == 47

    def test_schedule_failure_returns_none(self):
        class Failing:
            def get_game_info(self, game_id):
                raise RuntimeError("fuente caída")

        p = NFLSettlementProvider(schedule_fetcher=Failing())
        assert p.get_event_result(event()) is None


# ── Moneyline y empates ──────────────────────────────────────────────────────

class TestMoneyline:

    def test_winner_wins(self, lac_by_7, result_lac_by_7):
        assert lac_by_7.settle_pick(pick("ML", "LAC"), result_lac_by_7) == RESULT_WIN

    def test_loser_loses(self, lac_by_7, result_lac_by_7):
        assert lac_by_7.settle_pick(pick("ML", "KC"), result_lac_by_7) == RESULT_LOSE

    def test_tie_is_push_on_both_sides(self, tied):
        """
        El empate devuelve el stake en moneyline.

        Es la práctica estándar de las casas estadounidenses: no existe
        mercado a tres vías para NFL como sí lo hay en fútbol, así que
        el empate no puede resolverse como victoria de nadie.
        """
        result = tied.get_event_result(event())
        assert result["is_tie"] is True
        assert tied.settle_pick(pick("ML", "LAC"), result) == RESULT_PUSH
        assert tied.settle_pick(pick("ML", "KC"), result) == RESULT_PUSH

    def test_tie_counts_as_loss_when_configured(self):
        """
        tie_voids_ml=False implementa la regla europea.

        Algunos books con mercado a tres vías cuentan el empate como
        derrota del moneyline a dos vías.
        """
        p = provider_for(23, 23, overtime=True, tie_voids_ml=False)
        result = p.get_event_result(event())
        assert p.settle_pick(pick("ML", "LAC"), result) == RESULT_LOSE

    def test_overtime_flag_preserved(self, tied):
        assert tied.get_event_result(event())["overtime"] is True

    def test_overtime_winner_settles_normally(self):
        """El marcador de nflverse ya incluye el tiempo extra."""
        p = provider_for(30, 27, overtime=True)
        result = p.get_event_result(event())
        assert result["overtime"] is True
        assert p.settle_pick(pick("ML", "LAC"), result) == RESULT_WIN


# ── Spread ───────────────────────────────────────────────────────────────────

class TestSpread:

    @pytest.mark.parametrize("selection,line,expected", [
        # LAC gana por 7
        ("LAC", -3.5, RESULT_WIN),    # cubre: 7 - 3.5 = +3.5
        ("LAC", -6.5, RESULT_WIN),    # cubre por medio punto
        ("LAC", -7.5, RESULT_LOSE),   # no cubre: 7 - 7.5 = -0.5
        ("LAC", -10.5, RESULT_LOSE),
        ("KC",   3.5, RESULT_LOSE),   # el otro lado, complementario
        ("KC",   7.5, RESULT_WIN),
        ("KC",  10.5, RESULT_WIN),
    ])
    def test_cover_and_no_cover(self, lac_by_7, result_lac_by_7,
                                selection, line, expected):
        assert lac_by_7.settle_pick(
            pick("SPREAD", selection, line), result_lac_by_7
        ) == expected

    @pytest.mark.parametrize("selection,line", [("LAC", -7.0), ("KC", 7.0)])
    def test_push_on_exact_key_number_seven(self, lac_by_7, result_lac_by_7,
                                            selection, line):
        """
        Victoria por exactamente 7 con línea -7: push en ambos lados.

        El 7 concentra ~9% de todos los márgenes de victoria de NFL
        porque es el valor de un touchdown con conversión. Contarlo
        como derrota subestimaría el ROI de forma sistemática.
        """
        assert lac_by_7.settle_pick(
            pick("SPREAD", selection, line), result_lac_by_7
        ) == RESULT_PUSH

    def test_push_on_key_number_three(self):
        """
        El 3 es el número clave más frecuente: ~15% de los márgenes.

        Es el valor de un field goal, el desenlace más común de un
        partido igualado.
        """
        p = provider_for(24, 21)
        result = p.get_event_result(event())
        assert p.settle_pick(pick("SPREAD", "LAC", -3.0), result) == RESULT_PUSH
        assert p.settle_pick(pick("SPREAD", "KC", 3.0), result) == RESULT_PUSH

    @pytest.mark.parametrize("line", [-2.5, -3.5, -6.5, -7.5])
    def test_half_point_lines_never_push(self, line):
        """
        Las líneas .5 no pueden producir push: es su razón de ser.

        Por eso el runline de MLB es siempre ±1.5 y el push allí es
        imposible por construcción.
        """
        p = provider_for(24, 21)
        result = p.get_event_result(event())
        assert p.settle_pick(pick("SPREAD", "LAC", line), result) in (
            RESULT_WIN, RESULT_LOSE
        )

    def test_pickem_with_tie_is_push(self, tied):
        """Línea 0 con empate: nadie cubre."""
        result = tied.get_event_result(event())
        assert tied.settle_pick(pick("SPREAD", "LAC", 0.0), result) == RESULT_PUSH

    def test_missing_line_voids(self, lac_by_7, result_lac_by_7):
        assert lac_by_7.settle_pick(
            pick("SPREAD", "LAC", None), result_lac_by_7
        ) == RESULT_VOID

    def test_both_sides_are_complementary(self, lac_by_7, result_lac_by_7):
        """
        Las líneas opuestas del mismo mercado dan resultados opuestos.

        Si el local es -3.5, el visitante es +3.5. Nunca pueden ganar
        ni perder ambos salvo en push.
        """
        for line in (-1.5, -3.5, -6.5, -10.5):
            home = lac_by_7.settle_pick(pick("SPREAD", "LAC", line), result_lac_by_7)
            away = lac_by_7.settle_pick(pick("SPREAD", "KC", -line), result_lac_by_7)
            assert {home, away} == {RESULT_WIN, RESULT_LOSE}


# ── Total ────────────────────────────────────────────────────────────────────

class TestTotal:

    @pytest.mark.parametrize("selection,line,expected", [
        # Total del partido: 47
        ("over",  44.5, RESULT_WIN),
        ("over",  46.5, RESULT_WIN),
        ("over",  47.5, RESULT_LOSE),
        ("over",  50.5, RESULT_LOSE),
        ("under", 44.5, RESULT_LOSE),
        ("under", 47.5, RESULT_WIN),
        ("under", 50.5, RESULT_WIN),
    ])
    def test_over_under(self, lac_by_7, result_lac_by_7, selection, line, expected):
        assert lac_by_7.settle_pick(
            pick("TOTAL", selection, line), result_lac_by_7
        ) == expected

    @pytest.mark.parametrize("selection", ["over", "under"])
    def test_push_on_integer_total(self, lac_by_7, result_lac_by_7, selection):
        """
        Total exacto contra línea entera: push.

        Los totales NFL enteros (44, 47) son habituales, a diferencia
        de MLB donde predominan los .5.
        """
        assert lac_by_7.settle_pick(
            pick("TOTAL", selection, 47.0), result_lac_by_7
        ) == RESULT_PUSH

    def test_missing_line_voids(self, lac_by_7, result_lac_by_7):
        assert lac_by_7.settle_pick(
            pick("TOTAL", "over", None), result_lac_by_7
        ) == RESULT_VOID

    def test_invalid_selection_voids(self, lac_by_7, result_lac_by_7):
        assert lac_by_7.settle_pick(
            pick("TOTAL", "LAC", 47.5), result_lac_by_7
        ) == RESULT_VOID


# ── Mercados no liquidables ──────────────────────────────────────────────────

class TestUnsupportedMarkets:
    """
    Los mercados que el marcador final no permite resolver devuelven
    void de forma explícita, en vez de intentar una liquidación
    incorrecta. Void deja el stake intacto, que es preferible a
    registrar una ganancia o pérdida que no ocurrió.
    """

    @pytest.mark.parametrize("market", ["SPREAD_H1", "TOTAL_H1", "ML_H1"])
    def test_period_markets_void(self, lac_by_7, result_lac_by_7, market):
        assert lac_by_7.settle_pick(
            pick(market, "LAC", -3.5), result_lac_by_7
        ) == RESULT_VOID

    @pytest.mark.parametrize("market", [
        "PLAYER_PASS_YDS", "PLAYER_ANYTIME_TD", "player_rush_yds",
    ])
    def test_player_props_void(self, lac_by_7, result_lac_by_7, market):
        assert lac_by_7.settle_pick(
            pick(market, "P.Mahomes", 250.5), result_lac_by_7
        ) == RESULT_VOID

    def test_unknown_market_voids(self, lac_by_7, result_lac_by_7):
        assert lac_by_7.settle_pick(
            pick("MERCADO_INVENTADO", "LAC", 1.0), result_lac_by_7
        ) == RESULT_VOID


# ── Robustez ─────────────────────────────────────────────────────────────────

class TestRobustness:

    def test_unknown_selection_voids(self, lac_by_7, result_lac_by_7):
        """
        Una selección que no corresponde a ningún equipo se anula.

        Liquidar un pick cuyo equipo no se identifica sería adivinar.
        """
        assert lac_by_7.settle_pick(
            pick("ML", "EQUIPO_INEXISTENTE"), result_lac_by_7
        ) == RESULT_VOID

    def test_full_team_names_resolve(self):
        """
        Acepta nombres completos además de abreviaciones.

        El pick puede venir de The Odds API ('Los Angeles Chargers') o
        de nflverse ('LAC') según en qué stage se construyó.
        """
        p = NFLSettlementProvider(schedule_fetcher=StubSchedule([
            game(home="Los Angeles Chargers", away="Kansas City Chiefs",
                 home_score=27, away_score=20)
        ]))
        result = p.get_event_result(event(home="Los Angeles Chargers",
                                          away="Kansas City Chiefs"))
        assert p.settle_pick(
            pick("ML", "Los Angeles Chargers"), result
        ) == RESULT_WIN

    @pytest.mark.parametrize("corrupt", [
        {}, {"home_score": None}, {"foo": "bar"}, None,
    ])
    def test_corrupt_result_voids(self, lac_by_7, corrupt):
        """Un resultado corrupto no debe producir win ni lose."""
        assert lac_by_7.settle_pick(pick("ML", "LAC"), corrupt) == RESULT_VOID

    def test_all_outcomes_are_terminal(self, lac_by_7, result_lac_by_7):
        """Todo resultado emitido pertenece a TERMINAL_RESULTS."""
        combos = [
            ("ML", "LAC", None), ("ML", "KC", None),
            ("SPREAD", "LAC", -7.0), ("SPREAD", "LAC", -3.5),
            ("TOTAL", "over", 47.0), ("TOTAL", "over", 44.5),
            ("SPREAD_H1", "LAC", -3.5),
        ]
        outcomes = {
            lac_by_7.settle_pick(pick(m, s, l), result_lac_by_7)
            for m, s, l in combos
        }
        assert outcomes.issubset(TERMINAL_RESULTS)


# ── Interfaz financiera ──────────────────────────────────────────────────────

class TestFinancialInterface:

    def test_settles_ledger_entry(self, lac_by_7):
        entry = ledger_entry("SPREAD", "LAC -3.5")
        result = lac_by_7.get_result(entry)
        assert result is not None
        assert result.entry_id == entry.entry_id
        assert result.result == RESULT_WIN

    def test_unplayed_game_returns_none(self):
        p = NFLSettlementProvider(schedule_fetcher=StubSchedule([game()]))
        assert p.get_result(ledger_entry("ML", "LAC")) is None

    def test_closing_price_enables_clv(self, lac_by_7):
        entry = ledger_entry("SPREAD", "LAC -3.5")
        lac_by_7.register_closing_price(entry.entry_id, 1.83)

        result = lac_by_7.get_result(entry)
        assert result.closing_price == 1.83

        clv = result.clv(pick_price=entry.price)
        assert clv is not None and clv > 0, (
            "coger 1.91 y ver cerrar en 1.83 es CLV positivo"
        )

    def test_no_closing_price_gives_none_clv(self, lac_by_7):
        result = lac_by_7.get_result(ledger_entry("ML", "LAC"))
        assert result.closing_price is None
        assert result.clv(pick_price=1.91) is None

    def test_sport_context_carries_scores(self, lac_by_7):
        result = lac_by_7.get_result(ledger_entry("ML", "LAC"))
        assert result.sport_context["home_score"] == 27
        assert result.sport_context["total"] == 47


# ── Coherencia entre interfaces ──────────────────────────────────────────────

class TestInterfaceCoherence:
    """
    Las dos interfaces deben coincidir siempre.

    Si divergieran, el ROI registrado en el ledger no correspondería a
    los picks reportados por el pipeline. Este test detectó en la tarea
    10.12 que todo pick de TOTAL se liquidaba como void por la vía
    financiera: el ledger guarda 'over 47.0' con la línea embebida,
    mientras que la lógica de mercado esperaba 'over' a secas.

    El spread se salvaba por casualidad — la resolución de equipo hace
    matching por prefijo, así que 'LAC -3.5' seguía identificando a LAC.
    Un fallo que solo se manifestaba en uno de los tres mercados.
    """

    @pytest.mark.parametrize("market,selection,line,ledger_selection", [
        ("ML",     "LAC",  None,  "LAC"),
        ("ML",     "KC",   None,  "KC"),
        ("SPREAD", "LAC",  -7.0,  "LAC -7.0"),     # push
        ("SPREAD", "LAC",  -3.5,  "LAC -3.5"),     # win
        ("SPREAD", "LAC", -10.5,  "LAC -10.5"),    # lose
        ("SPREAD", "KC",    7.0,  "KC +7.0"),      # push, otro lado
        ("SPREAD", "KC",    3.5,  "KC +3.5"),      # lose
        ("TOTAL",  "over",  47.0, "over 47.0"),    # push
        ("TOTAL",  "over",  44.5, "over 44.5"),    # win
        ("TOTAL",  "under", 50.5, "under 50.5"),   # win
        ("TOTAL",  "under", 44.5, "under 44.5"),   # lose
    ])
    def test_sport_and_financial_agree(self, lac_by_7, result_lac_by_7,
                                       market, selection, line,
                                       ledger_selection):
        sport = lac_by_7.settle_pick(pick(market, selection, line),
                                     result_lac_by_7)
        financial = lac_by_7.get_result(
            ledger_entry(market, ledger_selection)
        ).result
        assert sport == financial, (
            f"{market} {ledger_selection}: "
            f"deportiva={sport} vs financiera={financial}"
        )

    def test_total_push_not_voided_by_ledger_parsing(self, lac_by_7):
        """
        Regresión del bug de la tarea 10.12.

        Antes del fix, la vía financiera devolvía void para todo TOTAL
        porque comparaba 'over 47.0' contra 'over'. En producción eso
        habría significado que ningún pick de total registrara ROI.
        """
        result = lac_by_7.get_result(ledger_entry("TOTAL", "over 47.0"))
        assert result.result == RESULT_PUSH
        assert result.result != RESULT_VOID