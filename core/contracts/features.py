"""
core/contracts/features.py

Abstracción del conocimiento deportivo de un equipo para un evento.

TeamFeatures es el contrato que separa el Core de cualquier dominio
deportivo: el Core ve offense_index=1.18, no sabe si ese número viene
de OPS (MLB), ORtg (NBA) o xG (Soccer). Cada sport plugin es responsable
de poblar este contrato a partir de sus propias fuentes de datos.

Origen del diseño — protección contra el bug documentado en
CRITICAL_FINDINGS_VALIDATION.md (hallazgos F1, F2, F5):

    El sistema MLB original permitía que 'runs_last_5' y
    'runs_recientes_lista' coexistieran de forma inconsistente
    (runs_last_5=4.5 con runs_recientes_lista=[]) sin que ningún
    mecanismo lo detectara. El fallback quedaba invisible para
    cualquier consumidor del dato.

    TeamFeatures resuelve esto estructuralmente: recent_avg se
    recalcula SIEMPRE desde recent_scores en __post_init__. Es
    imposible que ambos campos se contradigan.

Uso típico:
    features = TeamFeatures(
        team_id        = "yankees",
        team_name      = "New York Yankees",
        expected_score = 4.8,
        offense_index  = 1.18,
        defense_index  = 0.95,
        recent_scores  = [3.0, 7.0, 4.0, 2.0, 8.0, 5.0, 1.0, 6.0, 4.0, 3.0],
        recent_avg     = 0.0,   # se ignora — recalculado en __post_init__
        recent_n       = 0,     # se ignora — recalculado en __post_init__
        venue_factor   = 1.34,
        sample_size    = 120,
        data_quality   = 1.0,
        missing_fields = [],
        sport_metadata = {"era": 3.5, "fip": 3.2, "ops": 0.745},
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field


# Umbral mínimo de partidos recientes para considerar el dato estadísticamente
# utilizable en modelos de regresión/ensemble. Consistente con
# MIN_JUEGOS_PARA_REGRESION=5 del ensemble.py del sistema MLB original.
MIN_RECENT_SAMPLE = 5

# Umbral mínimo de data_quality para considerar el dato confiable.
MIN_DATA_QUALITY = 0.5

# data_quality aplicado cuando no hay forma reciente disponible en absoluto.
# Penaliza la confianza del dato sin anularla — otros campos (offense_index,
# defense_index) pueden seguir siendo válidos aunque falte la forma reciente.
NO_RECENT_DATA_QUALITY_PENALTY = 0.5


@dataclass
class TeamFeatures:
    """
    Resumen estadístico de un equipo para un evento específico.

    Mutable a propósito (a diferencia de Event): se construye
    progresivamente dentro de SportDataProvider.enrich_event() a
    medida que se obtienen distintos datos (forma reciente, splits,
    venue factor). La validación e invariantes se aplican en
    __post_init__ y se re-ejecutan si se llama a recompute() tras
    mutar recent_scores manualmente.

    Campos
    ------
    Identidad:
        team_id        -- ID canónico del equipo (igual a Event.home_team_id
                          o Event.away_team_id).
        team_name       -- Nombre para display.

    Puntuación esperada:
        expected_score  -- Puntuación esperada por partido en la unidad del
                          deporte (runs, goals, points). Lo produce el
                          ProjectionModel del sport plugin, no este contrato.

    Índices relativos a liga (1.0 = promedio de liga):
        offense_index   -- Fuerza ofensiva normalizada.
                          MLB: OPS/OPS_LIGA. NBA: ORtg/ORtg_LIGA.
                          Soccer: xG/xG_LIGA.
        defense_index   -- Fuerza defensiva normalizada (rival que enfrenta
                          o capacidad propia, según defina el plugin).
                          MLB: ERA/ERA_LIGA. NBA: DRtg/DRtg_LIGA.

    Forma reciente:
        recent_scores   -- Lista de puntuaciones reales de los últimos N
                          partidos. Sucesor directo de runs_recientes_lista.
                          Lista vacía es válida y debe pasarse explícitamente
                          cuando no hay datos — NUNCA debe inventarse un
                          valor placeholder.
        recent_avg      -- Media de recent_scores. CALCULADO AUTOMÁTICAMENTE
                          en __post_init__; cualquier valor pasado al
                          constructor es IGNORADO Y SOBRESCRITO. Sucesor
                          directo de runs_last_5.
        recent_n        -- len(recent_scores). CALCULADO AUTOMÁTICAMENTE,
                          igual que recent_avg.

    Contexto de venue:
        venue_factor    -- Factor de ventaja de localía/condiciones.
                          1.0 = neutral. Park factor MLB, court factor NBA,
                          surface adjustment Tennis. Default 1.0 — todo
                          evento tiene *algún* venue factor, aunque sea
                          neutral; nunca debe ser None.

    Calidad del dato:
        sample_size     -- Tamaño de la muestra que respalda offense_index/
                          defense_index. IP para MLB, juegos para otros
                          deportes. Unidad definida por cada sport plugin.
        data_quality    -- Confianza en el dato, 0.0 a 1.0. Ajustado
                          automáticamente a la baja en __post_init__ si
                          recent_scores está vacía (ver
                          NO_RECENT_DATA_QUALITY_PENALTY). El plugin puede
                          fijar un valor inicial, pero el contrato lo
                          corrige si los datos no lo respaldan.
        missing_fields  -- Lista de nombres de campos que usaron fallback.
                          Se añade 'recent_scores' automáticamente si la
                          lista llega vacía. El plugin añade sus propios
                          nombres antes de construir el objeto.

    Metadatos opacos:
        sport_metadata  -- Dict de datos específicos del deporte, sin
                          interpretación por parte del Core.
                          MLB: {'era': 3.5, 'fip': 3.2, 'ops': 0.745}.
                          Soccer: {'xg': 1.8, 'possession': 54.2}.
                          NINGÚN código del Core debe depender de claves
                          específicas de este dict — es responsabilidad
                          exclusiva del sport plugin que lo produce y lo
                          consume.

    Propiedades derivadas
    ----------------------
        has_sufficient_sample -- True si recent_n >= MIN_RECENT_SAMPLE
                                Y data_quality >= MIN_DATA_QUALITY.
                                Usado por EnsembleModel para decidir si
                                aplica regresión lineal sobre recent_scores
                                o cae a proyección Poisson pura.
    """

    # ── Identidad ──────────────────────────────────────────────────────────────
    team_id:   str
    team_name: str

    # ── Puntuación esperada ────────────────────────────────────────────────────
    expected_score: float

    # ── Índices relativos a liga ───────────────────────────────────────────────
    offense_index: float
    defense_index: float

    # ── Forma reciente ──────────────────────────────────────────────────────────
    # recent_avg y recent_n se recalculan SIEMPRE en __post_init__.
    # Los valores pasados al constructor para estos dos campos son ignorados.
    recent_scores: list[float]
    recent_avg:    float = field(default=0.0)
    recent_n:      int   = field(default=0)

    # ── Contexto de venue ──────────────────────────────────────────────────────
    venue_factor: float = 1.0

    # ── Calidad del dato ───────────────────────────────────────────────────────
    sample_size:    int             = 0
    data_quality:   float           = 1.0
    missing_fields: list[str]       = field(default_factory=list)

    # ── Metadatos opacos ───────────────────────────────────────────────────────
    sport_metadata: dict = field(default_factory=dict)

    # ── Validación e invariantes ──────────────────────────────────────────────

    def __post_init__(self) -> None:
        """
        Aplica las invariantes estructurales del contrato.

        Se ejecuta automáticamente al construir el objeto. Si se muta
        recent_scores manualmente después de la construcción, llamar a
        recompute() para volver a aplicar estas invariantes.
        """
        self._recompute_recent_form()
        self._apply_data_quality_penalty()

    def _recompute_recent_form(self) -> None:
        """
        recent_avg y recent_n se derivan EXCLUSIVAMENTE de recent_scores.

        Esto hace estructuralmente imposible el bug documentado en
        CRITICAL_FINDINGS_VALIDATION.md F1/F2: no puede existir un
        TeamFeatures con recent_avg=4.5 y recent_scores=[] de forma
        inconsistente, porque recent_avg ignora cualquier valor recibido
        y se calcula directamente desde la lista real.
        """
        self.recent_n = len(self.recent_scores)
        if self.recent_n > 0:
            self.recent_avg = round(sum(self.recent_scores) / self.recent_n, 3)
        else:
            self.recent_avg = 0.0

    def _apply_data_quality_penalty(self) -> None:
        """
        Penaliza data_quality y registra el campo faltante si no hay
        forma reciente disponible. El fallback queda visible en la
        estructura del objeto, no enmascarado por un valor que parece
        válido (a diferencia del RPG_LIGA=4.5 silencioso del sistema MLB).
        """
        if self.recent_n == 0:
            self.data_quality = min(self.data_quality, NO_RECENT_DATA_QUALITY_PENALTY)
            if "recent_scores" not in self.missing_fields:
                self.missing_fields.append("recent_scores")

    def recompute(self) -> None:
        """
        Reaplica las invariantes del contrato.

        Llamar explícitamente si recent_scores se muta después de la
        construcción inicial (por ejemplo, un sport plugin que añade
        partidos a la lista de forma incremental).
        """
        self.__post_init__()

    # ── Propiedades derivadas ──────────────────────────────────────────────────

    @property
    def has_sufficient_sample(self) -> bool:
        """
        True si la forma reciente es estadísticamente utilizable.

        Usado por EnsembleModel (core/simulation/ensemble.py) para decidir
        entre aplicar regresión lineal sobre recent_scores o usar
        proyección Poisson pura. Umbral consistente con
        MIN_JUEGOS_PARA_REGRESION=5 del ensemble.py del sistema MLB
        original.
        """
        return self.recent_n >= MIN_RECENT_SAMPLE and self.data_quality >= MIN_DATA_QUALITY