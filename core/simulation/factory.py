"""
core/simulation/factory.py

DistributionFactory: resuelve qué ProbabilityModel usar según deporte,
mercado y, opcionalmente, una preferencia explícita de la Projection.

Lógica de resolución documentada en SPORTS_PREDICTOR_ARCHITECTURE.md
§8.2:

    DistributionFactory.get_model(sport, market, projection)
        ├─ sport='mlb'    → PoissonModel(max_score=20)
        ├─ sport='soccer' → BivariatePoissonModel()
        ├─ sport='nhl'    → PoissonNBModel()  # Neg. Binomial (futuro)
        ├─ sport='nba'    → SkellamModel() para spread
                            NormalModel() para total
        ├─ sport='nfl'    → NormalModel()
        ├─ sport='tennis' → BradleyTerryModel()
        └─ sport='golf'   → NormalStrokeModel()  # futuro

Precedencia de resolución (dos niveles)
-----------------------------------------
1. Projection.distribution, si está poblado y coincide con un modelo
   registrado: tiene prioridad absoluta. Permite que un ProjectionModel
   sobreescriba la elección estándar para una proyección puntual — caso
   documentado explícitamente en la arquitectura: "El sport plugin
   puede sobreescribir la distribución via Projection.distribution".
   Ejemplo: NHL detecta overdispersión en una proyección específica y
   fija distribution='neg_binomial' en vez del PoissonModel por defecto
   para ese partido.

2. Tabla (sport, market) → modelo: resolución estándar cuando no hay
   override, o cuando Projection.distribution no coincide con ningún
   modelo registrado (se ignora silenciosamente el valor no reconocido
   y se cae a la tabla — distinto de un (sport, market) sin entrada en
   absoluto, que sí es un error).

Sin fallback silencioso a Poisson
-----------------------------------
Si (sport, market) no tiene modelo registrado, get_model() lanza
ValueError explícito. Mismo principio aplicado en config_loader.py
para sport.yaml faltante: un deporte sin modelo de probabilidad
calibrado no debe ejecutarse usando un default genérico — un
PoissonModel aplicado a golf (strokes, scoring inverso, sin
"victoria" binaria simple) produciría probabilidades sin sentido
de forma silenciosa, exactamente el tipo de fallo que el principio
"fail fast" de este proyecto existe para prevenir.

Caché de instancias
---------------------
Los modelos no tienen estado mutable entre llamadas a sus métodos
(PoissonModel(max_score=20) es la misma instancia útil para cualquier
Projection del mismo sport/market). Se cachean por clave
(sport, market, config_hash) para evitar reinstanciar en cada pick
evaluado durante un pipeline con cientos de eventos.

Parámetros desde ConfigLoader
--------------------------------
Si se provee un ConfigLoader (ej. load_config(sport='nba')), los
parámetros de cada modelo (max_score, default_sigma, rho) se leen de
YAML bajo la key simulation.{modelo}.{parametro}. Sin ConfigLoader,
se usan los defaults documentados en cada módulo individual.

Uso típico:
    from core.simulation.factory import DistributionFactory
    from core.utils.config_loader import load_config

    factory = DistributionFactory(config=load_config(sport='nba'))

    model_spread = factory.get_model('nba', 'SPREAD', projection)
    model_total  = factory.get_model('nba', 'TOTAL', projection)
    # model_spread es SkellamModel, model_total es NormalModel —
    # mismo deporte, mercado distinto, modelo distinto.
"""

from __future__ import annotations

from typing import Callable

from core.contracts import Projection
from core.simulation.bivariate_poisson import BivariatePoissonModel
from core.simulation.bradley_terry import BradleyTerryModel
from core.simulation.normal import NormalModel
from core.simulation.poisson import PoissonModel
from core.simulation.protocols import ProbabilityModel
from core.simulation.skellam import SkellamModel

# Comodín usado en la tabla de resolución para deportes donde un único
# modelo cubre los tres mercados (ML, SPREAD, TOTAL) sin distinción.
_ANY_MARKET = "ALL"


class DistributionFactory:
    """
    Resuelve y cachea instancias de ProbabilityModel según deporte,
    mercado y configuración.

    Parámetros
    ----------
    config   -- ConfigLoader opcional (core.utils.config_loader). Si se
               provee, los parámetros de cada modelo (max_score, sigma,
               rho) se leen de YAML. Si es None, se usan los defaults
               documentados en cada módulo de core/simulation/.
    """

    def __init__(self, config=None) -> None:
        self.config = config
        self._cache: dict[tuple[str, str], ProbabilityModel] = {}

        # Tabla de resolución (sport, market) → factory function.
        # market='ALL' es comodín: aplica a cualquier mercado consultado
        # para ese sport cuando no hay entrada más específica.
        #
        # IMPORTANTE: entradas comentadas (nhl/neg_binomial, golf/normal_stroke)
        # son modelos planeados en el roadmap pero NO implementados aún
        # (NegBinomialModel, NormalStrokeModel). Intentar resolver esos
        # (sport, market) hoy produce ValueError explícito — comportamiento
        # correcto: no hay fallback silencioso a un modelo no calibrado.
        self._registry: dict[tuple[str, str], Callable[[], ProbabilityModel]] = {
            ('mlb', _ANY_MARKET):    lambda: PoissonModel(max_score=self._get('mlb', 'poisson.max_score', 20)),
            ('soccer', _ANY_MARKET): lambda: BivariatePoissonModel(
                rho=self._get('soccer', 'bivariate.rho', -0.10),
                max_score=self._get('soccer', 'bivariate.max_score', 10),
            ),
            ('nfl', _ANY_MARKET):    lambda: NormalModel(
                default_sigma=self._get('nfl', 'normal.default_sigma', 10.0)
            ),
            ('nba', 'SPREAD'):       lambda: SkellamModel(
                max_diff=self._get('nba', 'skellam.max_diff', 60)
            ),
            ('nba', 'ML'):           lambda: SkellamModel(
                max_diff=self._get('nba', 'skellam.max_diff', 60)
            ),
            ('nba', 'TOTAL'):        lambda: NormalModel(
                default_sigma=self._get('nba', 'normal.default_sigma', 11.0)
            ),
            ('tennis', _ANY_MARKET): lambda: BradleyTerryModel(),

            # Distribuciones registrables vía Projection.distribution
            # override, independientemente del (sport, market) consultado.
            # No tienen entrada de tabla propia — solo se alcanzan si una
            # Projection trae distribution='poisson' explícitamente para
            # un sport que no es 'mlb' (ej. NHL con muestra suficiente
            # para Poisson puro en vez de Neg. Binomial).
        }

        # Mapa de Projection.distribution → factory function, para
        # resolución por override explícito (precedencia 1, ver
        # docstring del módulo). Reutiliza las mismas factory functions
        # cuando aplica, evitando duplicar la lógica de parámetros.
        self._distribution_override: dict[str, Callable[[], ProbabilityModel]] = {
            'poisson':           lambda: PoissonModel(max_score=20),
            'bivariate_poisson': lambda: BivariatePoissonModel(),
            'normal':            lambda: NormalModel(),
            'skellam':           lambda: SkellamModel(),
            'bradley_terry':     lambda: BradleyTerryModel(),
        }

    # ── Acceso a configuración con fallback ────────────────────────────────────

    def _get(self, sport: str, key: str, default):
        """
        Lee config.get(f"simulation.{sport}.{key}", default) si hay
        ConfigLoader; retorna default directamente si no.
        """
        if self.config is None:
            return default
        return self.config.get(f"simulation.{sport}.{key}", default=default)

    # ── Punto de entrada principal ─────────────────────────────────────────────

    def get_model(
        self,
        sport: str,
        market: str,
        projection: Projection | None = None,
    ) -> ProbabilityModel:
        """
        Resuelve el ProbabilityModel correcto para (sport, market),
        honrando Projection.distribution como override si se provee
        y coincide con un modelo registrado.

        Parámetros
        ----------
        sport       -- Identificador del deporte ('mlb', 'nba', etc.),
                      en minúsculas, igual que Event.sport.
        market       -- Mercado consultado ('ML', 'SPREAD', 'TOTAL'),
                      en mayúsculas. Determina qué modelo usar cuando
                      el deporte tiene resolución distinta por mercado
                      (NBA: SkellamModel para SPREAD/ML, NormalModel
                      para TOTAL).
        projection    -- Projection opcional. Si projection.distribution
                      está poblado y coincide con un modelo registrado
                      en _distribution_override, tiene prioridad sobre
                      la tabla (sport, market).

        Raises
        ------
        ValueError
            Si no hay modelo registrado para (sport, market) ni override
            válido en projection.distribution. Sin fallback silencioso.
        """
        sport = sport.lower()
        market = market.upper()

        # ── Precedencia 1: override explícito de Projection.distribution ──
        if projection is not None and projection.distribution:
            override_key = projection.distribution
            if override_key in self._distribution_override:
                cache_key = ('__override__', override_key)
                if cache_key not in self._cache:
                    self._cache[cache_key] = self._distribution_override[override_key]()
                return self._cache[cache_key]

        # ── Precedencia 2: tabla (sport, market) ───────────────────────────
        cache_key = (sport, market)
        if cache_key in self._cache:
            return self._cache[cache_key]

        factory_fn = self._registry.get((sport, market))
        if factory_fn is None:
            factory_fn = self._registry.get((sport, _ANY_MARKET))

        if factory_fn is None:
            registered = sorted(set(s for s, _ in self._registry.keys()))
            raise ValueError(
                f"No hay ProbabilityModel registrado para sport='{sport}', "
                f"market='{market}'. Deportes con modelo registrado: "
                f"{registered}. Si '{sport}' está planeado en el roadmap "
                f"pero su modelo aún no está implementado (ej. "
                f"NegBinomialModel para NHL, NormalStrokeModel para golf), "
                f"esto es el comportamiento correcto: no hay fallback "
                f"silencioso a un modelo no calibrado para este deporte."
            )

        model = factory_fn()
        self._cache[cache_key] = model
        return model

    def clear_cache(self) -> None:
        """
        Limpia las instancias cacheadas. Útil en tests, o si el
        ConfigLoader subyacente cambió y se necesita reinstanciar
        modelos con parámetros nuevos.
        """
        self._cache.clear()