"""
sports/soccer/projections.py

SoccerProjectionModel: convierte TeamFeatures en Projection.
Implementa core/pipeline/stage.py:ProjectionModel.

La formulación: Maher multiplicativo
--------------------------------------
    λ_local     = ataque_local × (1 / defensa_visitante) × media_liga_local
    λ_visitante = ataque_visitante × (1 / defensa_local) × media_liga_visitante

Tres decisiones dentro de esas dos líneas:

MULTIPLICATIVO, a diferencia de NFL
    En el plugin NFL se rechazó el producto de índices y se usó un
    promedio ponderado: el EPA promedia ~0.00, puede ser negativo, y con
    rango 0.60-1.40 dos errores del 10% arrastran un 21%.

    Aquí el producto es correcto. El xG es estrictamente positivo,
    promedia ~1.4, y esta es la formulación de Maher (1982) y
    Dixon-Coles (1997) — un modelo estudiado durante cuatro décadas.
    Apartarse de él sin razón específica sería cambiar rigor por
    originalidad.

INVERSIÓN DEL ÍNDICE DEFENSIVO
    team_stats.py define defense_index con la convención del Core:
    mayor es mejor. Un equipo con 1.5 concede 1/1.5 de la media.

    En la fórmula de Maher el factor defensivo entra como "goles
    concedidos relativos", así que hay que invertirlo. Se hace AQUÍ, en
    un punto explícito y documentado, en vez de romper la convención
    del Core en team_stats.py — donde BlendingEngine y los filtros
    asumen que un índice alto significa mejor rendimiento.

LA VENTAJA DE CAMPO YA ESTÁ DENTRO
    Las medias de liga vienen separadas por localía: league.home_goals
    es lo que anotan los locales, league.away_goals lo que anotan los
    visitantes. La diferencia ES la ventaja de campo.

    Sumarla otra vez la contaría dos veces. Es un error fácil de
    cometer porque soccer.yaml tiene un parámetro
    `home_advantage_default` — que existe solo como respaldo cuando no
    hay medias observadas.

Ajustes aditivos, en goles
----------------------------
Sobre la base multiplicativa se aplican correcciones en goles, no
multiplicadores:

    Derbi        reduce la ventaja del local (~-0.12). El visitante
                 lleva afición y el desplazamiento es nulo.
    Congestión   penaliza a quien viene de jugar con poco descanso.

Son transferencias concretas de goles, no reescalados: un equipo con
tres días de descanso pierde una cantidad parecida de rendimiento
independientemente de si proyectaba 1.2 o 2.4 goles.

Los mercados salen de UNA matriz
----------------------------------
El 1X2, el over/under y el BTTS se derivan de la misma matriz de
resultados construida por dixon_coles.py. Calcularlos por separado no
garantizaría coherencia entre ellos: podría salir un P(BTTS) mayor que
P(over 1.5), que es imposible.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.contracts.features import TeamFeatures
from core.contracts.projection import Projection

from sports.soccer.dixon_coles import ScoreMatrix, build_score_matrix


_MODEL_VERSION = "soccer-v1.0.0"

# ── Parámetros de la distribución ────────────────────────────────────────────
#
# Respaldos cuando soccer.yaml no los define. Se declaran AQUÍ y no se
# importan de dixon_coles.py, aunque ese módulo tenga constantes
# homónimas.
#
# La razón es de propiedad: estos son los valores por defecto DEL
# MODELO DE PROYECCIÓN. dixon_coles.py tiene los suyos para cuando se
# le llama sin parámetros, que es un caso distinto —un script de
# análisis, un test— y no tiene por qué coincidir.
#
# Importarlos de allí acoplaba este módulo a los internos de otro: si
# alguien recalibra el default del constructor de matrices, cambiaría
# en silencio el comportamiento del modelo de proyección aunque
# soccer.yaml no se hubiera tocado. Y el type checker lo señalaba con
# razón: eran símbolos con guion bajo importados desde fuera.
_DEFAULT_RHO      = -0.05
_DEFAULT_LAMBDA_3 = 0.0
_DEFAULT_MAX_GOALS = 10

# ── Valores por defecto ──────────────────────────────────────────────────────
#
# Medias de las cinco grandes cuando el contexto no las aporta. Que el
# local promedie más que el visitante ES la ventaja de campo, y por eso
# no hay que sumarla aparte.
_DEFAULT_LEAGUE_HOME = 1.50
_DEFAULT_LEAGUE_AWAY = 1.18

# Cotas de la media esperada por equipo. Red de seguridad: ningún
# equipo de las cinco grandes proyecta fuera de este rango.
_LAMBDA_MIN = 0.35
_LAMBDA_MAX = 3.50

# ── Penalizaciones de confianza ──────────────────────────────────────────────
_CONF_NO_XG              = 0.20   # sin xG el modelo pierde su señal principal
_CONF_SHORT_SAMPLE       = 0.15
_CONF_EARLY_SEASON       = 0.10
_CONF_CONGESTION_UNKNOWN = 0.08   # cobertura parcial del calendario
_CONF_MIN                = 0.10


@dataclass(frozen=True)
class _Lambdas:
    """Medias esperadas y el desglose de cómo se llegó a ellas."""
    home: float
    away: float

    base_home: float = 0.0
    base_away: float = 0.0
    attack_home: float = 1.0
    attack_away: float = 1.0
    defense_factor_home: float = 1.0   # 1 / defensa_visitante
    defense_factor_away: float = 1.0   # 1 / defensa_local
    league_home: float = _DEFAULT_LEAGUE_HOME
    league_away: float = _DEFAULT_LEAGUE_AWAY
    derby_adjustment: float = 0.0
    congestion_home: float = 0.0
    congestion_away: float = 0.0


class SoccerProjectionModel:
    """
    Modelo de proyección de goles para fútbol.

    Parámetros
    ----------
    config_loader -- ConfigLoader con soccer.yaml.
    """

    def __init__(self, config_loader = None) -> None:
        self._config = config_loader

        self._rho = self._cfg("soccer.dixon_coles.rho", _DEFAULT_RHO)
        self._lambda_3 = self._cfg(
            "simulation.soccer.bivariate_poisson.lambda_3", _DEFAULT_LAMBDA_3
        )
        self._max_goals = int(self._cfg(
            "simulation.soccer.bivariate_poisson.max_goals", _DEFAULT_MAX_GOALS
        ))
        self._lambda_min = self._cfg("simulation.soccer.lambda_min", _LAMBDA_MIN)
        self._lambda_max = self._cfg("simulation.soccer.lambda_max", _LAMBDA_MAX)

    def model_version(self) -> str:
        return _MODEL_VERSION

    # ── ProjectionModel Protocol ──────────────────────────────────────────────

    def project(
        self,
        home_features: TeamFeatures,
        away_features: TeamFeatures,
        context:       dict,
    ) -> Projection:
        """
        Proyecta los goles esperados de ambos equipos.

        Nunca lanza: ante datos ausentes cae a las medias de liga y lo
        refleja bajando `confidence`. Un pipeline que aborta por falta
        de un dato secundario es peor que uno que proyecta con
        incertidumbre declarada.
        """
        home_meta = home_features.sport_metadata or {}
        away_meta = away_features.sport_metadata or {}
        ctx = context or {}

        lambdas = self._compute_lambdas(
            home_features, away_features, home_meta, away_meta, ctx
        )

        matrix = build_score_matrix(
            lambda_home=lambdas.home,
            lambda_away=lambdas.away,
            rho=self._rho,
            lambda_3=self._lambda_3,
            max_goals=self._max_goals,
        )

        outcomes = matrix.outcome_probabilities()
        confidence = self._confidence(
            home_features, away_features, home_meta, away_meta, ctx
        )

        return Projection(
            event_id       = str(ctx.get("match_id") or ctx.get("event_id") or ""),
            sport          = "soccer",
            expected_home  = round(lambdas.home, 3),
            expected_away  = round(lambdas.away, 3),
            expected_total = round(lambdas.home + lambdas.away, 3),
            home_win_prob  = outcomes["home"],
            away_win_prob  = outcomes["away"],
            draw_prob      = outcomes["draw"],
            distribution   = "bivariate_poisson",
            distribution_params = {
                # Parámetros que la capa de probabilidad necesita para
                # reconstruir exactamente esta matriz. Sin ellos usaría
                # sus propios valores y las probabilidades del pick
                # diferirían de las que generaron la proyección.
                "lambda_home": round(lambdas.home, 4),
                "lambda_away": round(lambdas.away, 4),
                "rho":         self._rho,
                "lambda_3":    self._lambda_3,
                "max_goals":   self._max_goals,
            },
            confidence     = round(confidence, 4),
            model_version  = _MODEL_VERSION,
            model_inputs   = self._trace(lambdas, matrix, ctx),
        )

    # ── Cálculo de las medias ─────────────────────────────────────────────────

    def _compute_lambdas(
        self,
        home:      TeamFeatures,
        away:      TeamFeatures,
        home_meta: dict,
        away_meta: dict,
        ctx:       dict,
    ) -> _Lambdas:
        """
        Aplica la fórmula de Maher más los ajustes situacionales.

        Orden: base multiplicativa, ajustes aditivos, cotas.
        """
        # Cadena de respaldo: contexto, metadata del equipo, default.
        #
        # Se usa _num_or y no `or` por coherencia con score_matrix,
        # aunque aquí un cero sería igualmente inválido: una media de
        # liga de 0.0 goles no describe ninguna competición real, y
        # tratarla como ausencia es lo correcto. La diferencia es que
        # con _num_or la intención queda explícita en vez de depender
        # de que el cero resulte ser inválido.
        league_home = _num_or(
            ctx.get("league_home_goals"),
            _num_or(home_meta.get("league_home_goals"), _DEFAULT_LEAGUE_HOME),
        )
        league_away = _num_or(
            ctx.get("league_away_goals"),
            _num_or(home_meta.get("league_away_goals"), _DEFAULT_LEAGUE_AWAY),
        )
        # Una media no positiva no describe ninguna liga: se descarta.
        if league_home <= 0:
            league_home = _DEFAULT_LEAGUE_HOME
        if league_away <= 0:
            league_away = _DEFAULT_LEAGUE_AWAY

        attack_home = _index(home_meta.get("attack_index"), home.offense_index)
        attack_away = _index(away_meta.get("attack_index"), away.offense_index)

        # Inversión del índice defensivo. Ver la nota del módulo.
        defense_home = _index(home_meta.get("defense_index"), home.defense_index)
        defense_away = _index(away_meta.get("defense_index"), away.defense_index)
        factor_vs_away_defense = 1.0 / defense_away if defense_away > 0 else 1.0
        factor_vs_home_defense = 1.0 / defense_home if defense_home > 0 else 1.0

        base_home = attack_home * factor_vs_away_defense * league_home
        base_away = attack_away * factor_vs_home_defense * league_away

        # ── Ajustes aditivos ───────────────────────────────────────
        derby = _num_or(ctx.get("derby_adjustment"), 0.0)
        cong_home, cong_away = self._congestion(home_meta, away_meta, ctx)

        lam_home = base_home + derby + cong_home
        lam_away = base_away + cong_away

        return _Lambdas(
            home=_clamp(lam_home, self._lambda_min, self._lambda_max),
            away=_clamp(lam_away, self._lambda_min, self._lambda_max),
            base_home=base_home, base_away=base_away,
            attack_home=attack_home, attack_away=attack_away,
            defense_factor_home=factor_vs_away_defense,
            defense_factor_away=factor_vs_home_defense,
            league_home=league_home, league_away=league_away,
            derby_adjustment=derby,
            congestion_home=cong_home, congestion_away=cong_away,
        )

    @staticmethod
    def _congestion(
        home_meta: dict,
        away_meta: dict,
        ctx:       dict,
    ) -> tuple[float, float]:
        """
        Penalizaciones individuales de congestión, en goles.

        Se prefieren las penalizaciones POR EQUIPO cuando están en su
        sport_metadata, porque el diferencial del contexto no permite
        recuperarlas: un diferencial de -0.12 puede venir de un local
        penalizado o de un visitante fresco, y eso cambia el TOTAL
        proyectado aunque no cambie el margen.

        Sin ellas se reparte el diferencial a medias en direcciones
        opuestas: preserva el margen, que es lo que el diferencial
        mide, y deja el total sin alterar — el supuesto neutro cuando
        no se sabe de dónde viene la diferencia.
        """
        home_penalty = _num(home_meta.get("congestion_penalty"))
        away_penalty = _num(away_meta.get("congestion_penalty"))

        if home_penalty is not None or away_penalty is not None:
            return (home_penalty or 0.0), (away_penalty or 0.0)

        differential = _num_or(ctx.get("congestion_differential"), 0.0)
        if differential == 0.0:
            return 0.0, 0.0

        half = differential / 2.0
        return half, -half

    # ── Confianza ─────────────────────────────────────────────────────────────

    def _confidence(
        self,
        home:      TeamFeatures,
        away:      TeamFeatures,
        home_meta: dict,
        away_meta: dict,
        ctx:       dict,
    ) -> float:
        """
        Score de confianza en [0.10, 1.0].

        Se descuenta por factores concretos de incertidumbre, no por
        una estimación global. Cada descuento corresponde a una
        carencia identificable en los datos.
        """
        confidence = 1.0

        # Sin xG el modelo pierde su señal principal: la correlación
        # con resultados futuros baja de ~0.60 a ~0.40.
        if not home_meta.get("xg_available", True):
            confidence -= _CONF_NO_XG
        if not away_meta.get("xg_available", True):
            confidence -= _CONF_NO_XG

        if not home.has_sufficient_sample:
            confidence -= _CONF_SHORT_SAMPLE
        if not away.has_sufficient_sample:
            confidence -= _CONF_SHORT_SAMPLE

        # Calidad de datos declarada por el provider
        dq_home = _num(home.data_quality)
        dq_away = _num(away.data_quality)
        dq = ((dq_home if dq_home is not None else 1.0)
              + (dq_away if dq_away is not None else 1.0)) / 2.0
        confidence *= dq

        if ctx.get("is_early_season"):
            confidence -= _CONF_EARLY_SEASON

        # Congestión incompleta: el sistema solo ve partidos de liga,
        # así que un equipo con competición europea aparece más
        # descansado de lo que está.
        if (home_meta.get("congestion_incomplete")
                or away_meta.get("congestion_incomplete")):
            confidence -= _CONF_CONGESTION_UNKNOWN

        return max(_CONF_MIN, min(1.0, confidence))

    # ── Trazabilidad ──────────────────────────────────────────────────────────

    @staticmethod
    def _trace(lambdas: _Lambdas, matrix: ScoreMatrix, ctx: dict) -> dict:
        """
        Desglose completo del cálculo, para auditoría.

        Permite reconstruir a mano cómo se llegó a cada λ, que es lo
        que hace falta cuando un pick sale raro y hay que decidir si el
        problema está en los datos, en los índices o en los ajustes.
        """
        best_h, best_a, best_p = matrix.most_likely_score()
        return {
            # Base multiplicativa
            "attack_home":  round(lambdas.attack_home, 4),
            "attack_away":  round(lambdas.attack_away, 4),
            "defense_factor_home": round(lambdas.defense_factor_home, 4),
            "defense_factor_away": round(lambdas.defense_factor_away, 4),
            "league_home_goals": round(lambdas.league_home, 4),
            "league_away_goals": round(lambdas.league_away, 4),
            "base_home":    round(lambdas.base_home, 4),
            "base_away":    round(lambdas.base_away, 4),
            # Ajustes aditivos
            "adj_derby":       lambdas.derby_adjustment,
            "adj_congestion_home": lambdas.congestion_home,
            "adj_congestion_away": lambdas.congestion_away,
            # Distribución
            "rho":          matrix.rho,
            "lambda_3":     matrix.lambda_3,
            "most_likely_score": f"{best_h}-{best_a}",
            "most_likely_prob":  round(best_p, 4),
            # Mercados derivados de la misma matriz
            "btts":         matrix.btts(),
            "over_2_5":     matrix.total_over(2.5),
            "low_score_mass": matrix.low_score_mass,
            # Contexto relevante
            "is_derby":     bool(ctx.get("is_derby", False)),
            "matchday":     ctx.get("matchday"),
            "competition":  ctx.get("competition"),
        }

    # ── Mercados auxiliares ───────────────────────────────────────────────────

    def score_matrix(
        self,
        projection: Projection,
    ) -> ScoreMatrix:
        """
        Reconstruye la matriz de una proyección.

        Necesaria para derivar mercados que el contrato Projection no
        lleva: BTTS, over/under en líneas distintas de 2.5, hándicap.

        Usa los parámetros de `distribution_params`, así que la matriz
        es EXACTAMENTE la que produjo las probabilidades de la
        proyección. Reconstruirla con otros parámetros daría mercados
        incoherentes entre sí.
        """
        params = projection.distribution_params or {}
        return build_score_matrix(
            lambda_home=_num_or(params.get("lambda_home"),
                                projection.expected_home),
            lambda_away=_num_or(params.get("lambda_away"),
                                projection.expected_away),
            rho=_num_or(params.get("rho"), self._rho),
            lambda_3=_num_or(params.get("lambda_3"), self._lambda_3),
            max_goals=int(_num_or(params.get("max_goals"), self._max_goals)),
        )

    # ── Config ────────────────────────────────────────────────────────────────

    def _cfg(self, key: str, default: float) -> float:
        if self._config is None:
            return default
        try:
            value = self._config.get(key, default=default)
            return float(value) if value is not None else default
        except (ValueError, TypeError, AttributeError):
            return default


# ── Utilidades ───────────────────────────────────────────────────────────────

def _index(primary, fallback) -> float:
    """
    Índice de ataque o defensa, con respaldo.

    Se prefiere el de sport_metadata —calculado por team_stats.py con
    shrinkage y separación por localía— sobre el campo genérico de
    TeamFeatures. Ambos deberían coincidir, pero el metadata es el
    origen y el campo del contrato una copia.

    Un índice de 0 o negativo no es utilizable en una fórmula
    multiplicativa, así que cae a 1.0: neutro, no cero.
    """
    value = _num(primary)
    if value is None or value <= 0:
        value = _num(fallback)
    if value is None or value <= 0:
        return 1.0
    return value


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _num_or(value, fallback: float) -> float:
    """
    Convierte a float, con respaldo cuando el valor está AUSENTE.

    Por qué existe en vez de usar `_num(x) or fallback`
    ----------------------------------------------------
    El operador `or` trata el 0.0 como ausencia, y aquí el cero es un
    valor legítimo:

        rho = 0.0       desactiva la corrección Dixon-Coles
        lambda_3 = 0.0  desactiva el componente bivariado (es el
                        default actual)

    Con `or`, una proyección calculada con λ₃=0.0 se reconstruiría con
    el λ₃ del config. Los mercados derivados —BTTS, over/under— dejarían
    de coincidir con el 1X2 de la proyección, que es exactamente la
    incoherencia que este módulo existe para evitar.

    La distinción es entre "el parámetro no está" y "el parámetro vale
    cero", y solo un chequeo explícito contra None la respeta.
    """
    parsed = _num(value)
    return parsed if parsed is not None else fallback


def _num(value) -> float | None:
    """Convierte a float tratando None y NaN como ausencia."""
    if value is None:
        return None
    try:
        result = float(value)
    except (ValueError, TypeError):
        return None
    return result if result == result else None