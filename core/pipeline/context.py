"""
core/pipeline/context.py

Re-exporta PipelineContext desde core/pipeline/stage.py.

Por qué PipelineContext vive en stage.py y no aquí
-----------------------------------------------------
La arquitectura especifica context.py como archivo separado, pero
PipelineContext depende estructuralmente de los protocolos del pipeline
(su forma refleja exactamente qué produce cada stage) y un archivo
separado crearía un import circular:

    context.py → importa Event, TeamFeatures, Projection, CandidatePick
    stage.py   → importa PipelineContext desde context.py
    PipelineStage.run() → tiene PipelineContext en su firma

Al mantener PipelineContext en stage.py junto a los protocolos que lo
producen y consumen, el contrato completo del pipeline es legible en
un solo lugar. Este archivo cumple la separación de nombres del spec
permitiendo que el caller importe desde el path esperado:

    # Ambas formas son equivalentes:
    from core.pipeline.context import PipelineContext
    from core.pipeline.stage   import PipelineContext

Uso típico
-----------
    from core.pipeline.context import PipelineContext

    context = PipelineContext(sport='mlb', date='2026-07-15')

    # Stage 1: SportDataProvider
    context.events = data_provider.get_events(context.date)

    # Stage 2: enrich
    for event in context.events:
        home_f, away_f = data_provider.enrich_event(event)
        context.enriched[event.event_id] = (home_f, away_f)

    # Stage 3: project
    for event_id, (home_f, away_f) in context.enriched.items():
        ctx_event = data_provider.get_context(
            context.events[0]  # o buscar por event_id
        )
        context.projections[event_id] = proj_model.project(
            home_f, away_f, ctx_event
        )

    # ... stages 4-9 ...

    print(context.summary())
    # [mlb/2026-07-15] eventos=12 candidatos=8 activos=3 errores=0
"""

from core.pipeline.stage import PipelineContext

__all__ = ["PipelineContext"]