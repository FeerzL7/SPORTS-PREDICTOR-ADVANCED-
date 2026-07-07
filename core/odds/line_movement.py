"""
core/odds/line_movement.py

LineMovementDetector: detecta movimiento de línea comparando snapshots
de cuotas en distintos momentos del día.

Migrado de data/line_movement.py del sistema MLB con cuatro correcciones
arquitectónicas documentadas en MLB_EDGE_AUDIT.md:

1. SnapshotStore como Protocol — persistencia desacoplada del detector.
   El sistema MLB guardaba snapshots directamente en disco dentro de
   guardar_snapshot_diario(). Aquí JsonSnapshotStore es una
   implementación intercambiable — tests usan InMemorySnapshotStore,
   producción puede escalar a Redis/S3 sin tocar LineMovementDetector.

2. MovementSignal con magnitude — no solo booleans mov_confirma/mov_contradice.
   mov_confirma/mov_contradice perdían la intensidad del movimiento.
   Un juice shift de 0.02 es ruido; uno de 0.15 es señal sharp.
   magnitude permite a StakingStrategy ajustar stake proporcionalmente.

3. Thresholds por (sport, market) en YAML — no constantes hardcodeadas.
   UMBRAL_ML_MOVE=0.06 y UMBRAL_JU_MOVE=0.04 del sistema MLB son
   empíricos para baseball. Para NBA u otros deportes pueden ser ruido.

4. annotate_pick() no muta el pick directamente — añade razones vía
   add_reason() y retorna. La decisión de desactivar el pick pertenece
   a RiskManager (Stage 9), no al detector de movimiento.

Roles en el pipeline
----------------------
Stage 5 (Odds Ingestion):
    detector.snapshot(market_odds, event_id, sport, date)
    → Guarda el estado actual como punto de referencia temporal.
    → Si es la primera ejecución del día, este será el snapshot de "apertura".

Stage 7 (Line Movement Adjustment):
    signals = detector.analyze(current_odds, event_id, sport, date)
    → Compara current_odds vs snapshot de apertura del día.
    → Retorna list[MovementSignal] — vacía si no hay movimiento o no hay snapshot previo.
    pick = detector.annotate_pick(pick, signals)
    → Añade MovementSignal.to_reason() al trail del pick.
    → RiskManager leerá pick.reasons para decidir si desactivar.

Señales detectadas
-------------------
LINE_MOVE:
    La línea de totales/spread cambió entre apertura y estado actual.
    Indica que el mercado actualizó su estimación del total/handicap.
    Ejemplo: TOTAL 8.5 → 9.0 (dinero en Over forzó la línea hacia arriba).

JUICE_SHIFT:
    La línea no cambió pero los precios se movieron asimétricamente.
    Ejemplo: Over 1.91/Under 1.91 → Over 1.82/Under 2.00.
    Indica que el dinero entró en Under sin cambiar la línea aún.

ML_MOVE:
    La cuota de ML cambió significativamente para un equipo.
    Ejemplo: home 2.10 → 1.90 (dinero en home acortó la cuota).

NO_MOVEMENT:
    Sin señal detectada para este (event_id, market, selection).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import date as Date
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Protocol, runtime_checkable

from core.contracts.market_odds import MarketOdds
from core.contracts.pick import CandidatePick


# ── Tipos de movimiento ────────────────────────────────────────────────────────

class MovementType(Enum):
    """
    Clasificación del tipo de movimiento de línea detectado.

    LINE_MOVE:   La línea numérica cambió (total, spread).
    JUICE_SHIFT: Los precios cambiaron con la línea estable.
    ML_MOVE:     El precio de ML cambió significativamente.
    NO_MOVEMENT: Sin señal detectable en el umbral configurado.
    """
    LINE_MOVE   = auto()
    JUICE_SHIFT = auto()
    ML_MOVE     = auto()
    NO_MOVEMENT = auto()


class MovementDirection(Enum):
    """
    Dirección del movimiento relativa a la selección detectada.

    CONFIRMS:     El movimiento va en la misma dirección que el pick.
    CONTRADICTS:  El movimiento va en la dirección opuesta al pick.
    NEUTRAL:      El movimiento no tiene dirección clara o no aplica.
    """
    CONFIRMS    = "confirms"
    CONTRADICTS = "contradicts"
    NEUTRAL     = "neutral"


# ── Snapshot de cuotas ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OddsSnapshot:
    """
    Estado inmutable de las cuotas de un evento en un instante.

    Inmutable: representa un hecho histórico. Una vez guardado,
    el snapshot no se modifica — cada ejecución del pipeline genera
    un nuevo snapshot que se añade al historial del día.

    Campos
    ------
    event_id   -- ID del evento en The Odds API.
    sport       -- Identificador del deporte.
    timestamp   -- Momento exacto del snapshot en ISO-8601 UTC.
    odds        -- Estado de las cuotas en ese instante.
                  list[MarketOdds] con no_vig_prob posiblemente None
                  (el snapshot se toma antes de no_vig_probabilities()).
    snapshot_date -- Fecha del snapshot (para agrupar por día).
    """
    event_id:      str
    sport:         str
    timestamp:     str
    odds:          tuple[MarketOdds, ...]  # tuple para inmutabilidad
    snapshot_date: str  # YYYY-MM-DD

    @classmethod
    def create(
        cls,
        event_id: str,
        sport: str,
        odds: list[MarketOdds],
        snapshot_date: str | None = None,
    ) -> OddsSnapshot:
        """
        Factory method — crea un OddsSnapshot con timestamp UTC actual.
        """
        now = datetime.now(timezone.utc).isoformat()
        today = snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return cls(
            event_id=event_id,
            sport=sport,
            timestamp=now,
            odds=tuple(odds),
            snapshot_date=today,
        )

    def get_odds_map(self) -> dict[tuple[str, str, str | None], MarketOdds]:
        """
        Indexa las cuotas por (market, selection, line_str) para
        comparación O(1) entre snapshots.

        line_str es str(line) o None — permite distinguir líneas
        distintas del mismo mercado/selección.
        """
        return {
            (o.market, o.selection, str(o.line) if o.line is not None else None): o
            for o in self.odds
        }


# ── Señal de movimiento ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class MovementSignal:
    """
    Señal de movimiento detectada para una selección específica.

    Inmutable: representa un hecho detectado en el análisis.

    Campos
    ------
    event_id     -- ID del evento.
    market        -- Mercado donde ocurrió el movimiento ('ML', 'TOTAL', etc.).
    selection     -- Selección específica ('over', 'New York Yankees', etc.).
    movement_type -- Tipo de movimiento detectado.
    direction     -- Si el movimiento confirma o contradice la dirección
                    de la selección (para uso por annotate_pick).
    magnitude     -- Tamaño del movimiento. Para LINE_MOVE: diferencia
                    absoluta de línea. Para JUICE_SHIFT/ML_MOVE: diferencia
                    absoluta de precio. Permite a StakingStrategy ajustar
                    stake proporcionalmente.
    from_price    -- Precio en el snapshot de apertura.
    to_price      -- Precio en el snapshot actual.
    from_line     -- Línea en apertura (None para ML).
    to_line       -- Línea actual (None para ML).
    detail        -- Descripción legible del movimiento para reasons trail.
    """
    event_id:      str
    market:        str
    selection:     str
    movement_type: MovementType
    direction:     MovementDirection
    magnitude:     float
    from_price:    float | None
    to_price:      float | None
    from_line:     float | None
    to_line:       float | None
    detail:        str

    def to_reason(self) -> str:
        """
        Serialización para CandidatePick.add_reason().
        """
        dir_symbol = {
            MovementDirection.CONFIRMS:    "✓",
            MovementDirection.CONTRADICTS: "✗",
            MovementDirection.NEUTRAL:     "~",
        }[self.direction]

        return (
            f"MOVEMENT[{dir_symbol}] {self.movement_type.name} "
            f"{self.market}/{self.selection}: {self.detail} "
            f"(magnitude={self.magnitude:.3f})"
        )

    @property
    def confirms(self) -> bool:
        return self.direction == MovementDirection.CONFIRMS

    @property
    def contradicts(self) -> bool:
        return self.direction == MovementDirection.CONTRADICTS


# ── Thresholds de detección ───────────────────────────────────────────────────

@dataclass(frozen=True)
class MovementThresholds:
    """
    Umbrales de detección para un (sport, market) específico.

    Todos los valores son mínimos — movimientos por debajo del umbral
    se clasifican como NO_MOVEMENT (ruido de mercado normal).

    Campos
    ------
    ml_move_threshold    -- Cambio mínimo en precio de ML para considerarlo
                           señal significativa. Sistema MLB: 0.06.
    juice_shift_threshold -- Cambio mínimo asimétrico en juice con línea
                           estable. Sistema MLB: 0.04.
    line_move_threshold   -- Cambio mínimo en la línea numérica. Para
                           totales con líneas en 0.5, el mínimo es 0.5
                           (un medio punto). Para spreads enteros, puede
                           ser 1.0.
    """
    ml_move_threshold:     float = 0.06
    juice_shift_threshold: float = 0.04
    line_move_threshold:   float = 0.25  # cualquier cambio de línea es señal


# ── Protocolo de persistencia de snapshots ────────────────────────────────────

@runtime_checkable
class SnapshotStore(Protocol):
    """
    Interfaz de persistencia para snapshots de cuotas.

    Permite sustituir la implementación de disco local (JsonSnapshotStore)
    por Redis, S3 u otro backend sin modificar LineMovementDetector.
    """

    def save(self, snapshot: OddsSnapshot) -> None:
        """Persiste un snapshot."""
        ...

    def load_latest(
        self,
        event_id: str,
        sport: str,
        snapshot_date: str,
    ) -> OddsSnapshot | None:
        """
        Carga el snapshot más reciente del día para un evento.
        Retorna None si no hay snapshot previo.
        """
        ...

    def load_opening(
        self,
        event_id: str,
        sport: str,
        snapshot_date: str,
    ) -> OddsSnapshot | None:
        """
        Carga el primer snapshot del día (apertura) para un evento.
        Retorna None si no hay snapshot previo.
        """
        ...


# ── Implementación: JSON en disco ─────────────────────────────────────────────

class JsonSnapshotStore:
    """
    Implementación de SnapshotStore que persiste snapshots en JSON local.

    Compatible con el formato del sistema MLB (output/line_snapshots/).
    Los snapshots se guardan en {base_dir}/{YYYY-MM-DD}/{sport}/{event_id}.json

    La ruta incluye sport para evitar colisiones entre eventos de distintos
    deportes con el mismo event_id (posible si se usa el mismo provider
    para múltiples deportes).

    Parámetros
    ----------
    base_dir  -- Directorio raíz para los snapshots.
               Default: 'output/line_snapshots'
    """

    def __init__(self, base_dir: str = "output/line_snapshots") -> None:
        self._base_dir = base_dir

    def _path(self, event_id: str, sport: str, snapshot_date: str) -> str:
        return os.path.join(
            self._base_dir, snapshot_date, sport, f"{event_id}.json"
        )

    def save(self, snapshot: OddsSnapshot) -> None:
        path = self._path(snapshot.event_id, snapshot.sport, snapshot.snapshot_date)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Cargar snapshots existentes del día para este evento
        existing: list[dict] = []
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, OSError):
                existing = []

        # Serializar el nuevo snapshot
        snapshot_dict = {
            "event_id":      snapshot.event_id,
            "sport":         snapshot.sport,
            "timestamp":     snapshot.timestamp,
            "snapshot_date": snapshot.snapshot_date,
            "odds": [
                {
                    "event_id":  o.event_id,
                    "market":    o.market,
                    "selection": o.selection,
                    "line":      o.line,
                    "price":     o.price,
                    "bookmaker": o.bookmaker,
                    "timestamp": o.timestamp,
                }
                for o in snapshot.odds
            ],
        }
        existing.append(snapshot_dict)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)

    def load_opening(
        self,
        event_id: str,
        sport: str,
        snapshot_date: str,
    ) -> OddsSnapshot | None:
        """Carga el primer snapshot del día (índice 0)."""
        snapshots = self._load_all(event_id, sport, snapshot_date)
        if not snapshots:
            return None
        return snapshots[0]

    def load_latest(
        self,
        event_id: str,
        sport: str,
        snapshot_date: str,
    ) -> OddsSnapshot | None:
        """Carga el snapshot más reciente del día (índice -1)."""
        snapshots = self._load_all(event_id, sport, snapshot_date)
        if not snapshots:
            return None
        return snapshots[-1]

    def _load_all(
        self,
        event_id: str,
        sport: str,
        snapshot_date: str,
    ) -> list[OddsSnapshot]:
        path = self._path(event_id, sport, snapshot_date)
        if not os.path.exists(path):
            return []
        try:
            with open(path, encoding="utf-8") as f:
                raw_list = json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

        result = []
        for raw in raw_list:
            try:
                odds = tuple(
                    MarketOdds(
                        event_id=o["event_id"],
                        market=o["market"],
                        selection=o["selection"],
                        line=o.get("line"),
                        price=o["price"],
                        bookmaker=o.get("bookmaker", "unknown"),
                        timestamp=o.get("timestamp", ""),
                    )
                    for o in raw.get("odds", [])
                )
                result.append(OddsSnapshot(
                    event_id=raw["event_id"],
                    sport=raw["sport"],
                    timestamp=raw["timestamp"],
                    odds=odds,
                    snapshot_date=raw["snapshot_date"],
                ))
            except (KeyError, ValueError):
                continue
        return result


class InMemorySnapshotStore:
    """
    Implementación en memoria para tests y entornos sin disco.

    No persiste entre instancias — útil para tests unitarios que
    necesitan control total sobre el estado de los snapshots.
    """

    def __init__(self) -> None:
        # key: (event_id, sport, snapshot_date) → lista ordenada de snapshots
        self._store: dict[tuple[str, str, str], list[OddsSnapshot]] = {}

    def save(self, snapshot: OddsSnapshot) -> None:
        key = (snapshot.event_id, snapshot.sport, snapshot.snapshot_date)
        if key not in self._store:
            self._store[key] = []
        self._store[key].append(snapshot)

    def load_opening(
        self,
        event_id: str,
        sport: str,
        snapshot_date: str,
    ) -> OddsSnapshot | None:
        key = (event_id, sport, snapshot_date)
        snapshots = self._store.get(key, [])
        return snapshots[0] if snapshots else None

    def load_latest(
        self,
        event_id: str,
        sport: str,
        snapshot_date: str,
    ) -> OddsSnapshot | None:
        key = (event_id, sport, snapshot_date)
        snapshots = self._store.get(key, [])
        return snapshots[-1] if snapshots else None

    def clear(self) -> None:
        self._store.clear()


# ── Motor principal ───────────────────────────────────────────────────────────

class LineMovementDetector:
    """
    Detecta movimiento de línea comparando snapshots de cuotas.

    Parámetros
    ----------
    store       -- SnapshotStore para persistir y recuperar snapshots.
                  Default: JsonSnapshotStore() (compatible con sistema MLB).
    thresholds  -- MovementThresholds con umbrales de detección.
                  Si None, usa los defaults documentados.
    config      -- ConfigLoader opcional para leer thresholds desde YAML.
                  Si se provee, sobreescribe thresholds con valores del YAML.
    """

    def __init__(
        self,
        store: SnapshotStore | None = None,
        thresholds: MovementThresholds | None = None,
        config=None,
    ) -> None:
        self._store = store or JsonSnapshotStore()
        self._config = config
        self._thresholds = self._load_thresholds(thresholds)

    def _load_thresholds(
        self,
        default: MovementThresholds | None,
    ) -> MovementThresholds:
        base = default or MovementThresholds()
        if self._config is None:
            return base
        return MovementThresholds(
            ml_move_threshold=float(self._config.get(
                "line_movement.ml_move_threshold",
                default=base.ml_move_threshold,
            )),
            juice_shift_threshold=float(self._config.get(
                "line_movement.juice_shift_threshold",
                default=base.juice_shift_threshold,
            )),
            line_move_threshold=float(self._config.get(
                "line_movement.line_move_threshold",
                default=base.line_move_threshold,
            )),
        )

    # ── Stage 5: guardar snapshot ─────────────────────────────────────────────

    def snapshot(
        self,
        odds: list[MarketOdds],
        event_id: str,
        sport: str,
        snapshot_date: str | None = None,
    ) -> OddsSnapshot:
        """
        Guarda el estado actual de las cuotas como snapshot.

        Llamar en Stage 5, inmediatamente después de OddsNormalizer,
        antes de ValueEngine. Si es la primera ejecución del día,
        este snapshot se convierte en la "apertura" de referencia.

        Parámetros
        ----------
        odds           -- Cuotas actuales del evento (list[MarketOdds]).
        event_id       -- ID del evento en The Odds API.
        sport          -- Identificador del deporte.
        snapshot_date  -- Fecha del snapshot en YYYY-MM-DD.
                         Default: hoy UTC.

        Retorna
        -------
        OddsSnapshot guardado (útil para tests que necesitan
        el objeto sin cargarlo desde el store).
        """
        snap = OddsSnapshot.create(
            event_id=event_id,
            sport=sport,
            odds=odds,
            snapshot_date=snapshot_date,
        )
        self._store.save(snap)
        return snap

    # ── Stage 7: analizar movimiento ──────────────────────────────────────────

    def analyze(
        self,
        current_odds: list[MarketOdds],
        event_id: str,
        sport: str,
        snapshot_date: str | None = None,
    ) -> list[MovementSignal]:
        """
        Compara el estado actual de las cuotas contra el snapshot de
        apertura del día y retorna las señales de movimiento detectadas.

        Si no hay snapshot de apertura (primera ejecución del día),
        retorna lista vacía sin error — el pipeline continúa sin
        información de movimiento.

        Parámetros
        ----------
        current_odds   -- Estado actual de las cuotas del evento.
        event_id       -- ID del evento.
        sport          -- Identificador del deporte.
        snapshot_date  -- Fecha a comparar. Default: hoy UTC.

        Retorna
        -------
        list[MovementSignal] — vacía si no hay snapshot previo o
        no se detecta movimiento significativo.
        """
        today = snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        opening = self._store.load_opening(event_id, sport, today)

        if opening is None:
            return []

        opening_map = opening.get_odds_map()
        signals: list[MovementSignal] = []

        # Agrupar cuotas actuales por mercado para detección de juice shift
        current_by_market: dict[str, list[MarketOdds]] = {}
        for odds in current_odds:
            current_by_market.setdefault(odds.market, []).append(odds)

        opening_by_market: dict[str, list[MarketOdds]] = {}
        for odds in opening.odds:
            opening_by_market.setdefault(odds.market, []).append(odds)

        # Analizar cada cuota actual contra su apertura
        for current in current_odds:
            key = (current.market, current.selection,
                   str(current.line) if current.line is not None else None)
            opening_odds = opening_map.get(key)

            if opening_odds is None:
                # La selección no existía en apertura — puede ser LINE_MOVE
                # (la línea cambió y ahora existe en una posición distinta)
                signal = self._detect_line_move_new(
                    current=current,
                    opening_market_odds=opening_by_market.get(current.market, []),
                )
                if signal:
                    signals.append(signal)
                continue

            # Comparar precio actual vs apertura para esta selección/línea
            signal = self._detect_price_move(opening_odds, current)
            if signal:
                signals.append(signal)

        # Detectar juice shift (línea estable, precios asimétricos)
        for market, current_group in current_by_market.items():
            opening_group = opening_by_market.get(market, [])
            juice_signal = self._detect_juice_shift(
                market=market,
                event_id=event_id,
                current_group=current_group,
                opening_group=opening_group,
            )
            if juice_signal:
                signals.append(juice_signal)

        return signals

    # ── Stage 7: anotar picks con señales de movimiento ──────────────────────

    def annotate_pick(
        self,
        pick: CandidatePick,
        signals: list[MovementSignal],
    ) -> CandidatePick:
        """
        Añade señales de movimiento relevantes al trail de razones
        del pick.

        Solo anota señales relevantes para la selección del pick
        (mismo market y selection). No modifica active — esa decisión
        pertenece a RiskManager (Stage 9).

        Si ninguna señal es relevante, retorna el pick sin modificación.
        """
        relevant = [
            s for s in signals
            if s.market == pick.market and s.selection == pick.selection
        ]

        for signal in relevant:
            pick.add_reason(signal.to_reason())

        return pick

    # ── Detección de señales individuales ────────────────────────────────────

    def _detect_price_move(
        self,
        opening: MarketOdds,
        current: MarketOdds,
    ) -> MovementSignal | None:
        """
        Detecta ML_MOVE: cambio de precio para la misma selección/línea.
        Solo aplica para mercados sin línea (ML) — para totals/spreads
        el cambio de precio con misma línea se detecta vía juice shift.
        """
        if current.market not in ("ML", "1X2"):
            return None  # juice shift cubre los mercados con línea

        price_diff = abs(current.price - opening.price)
        if price_diff < self._thresholds.ml_move_threshold:
            return None

        # Precio bajó → dinero en esta selección (cuota se acortó)
        # Precio subió → dinero en la selección opuesta
        direction = (
            MovementDirection.CONFIRMS
            if current.price < opening.price  # cuota acortada = dinero aquí
            else MovementDirection.CONTRADICTS
        )

        detail = (
            f"{current.selection}: {opening.price:.2f} → {current.price:.2f} "
            f"({'acortó' if direction == MovementDirection.CONFIRMS else 'alargó'})"
        )

        return MovementSignal(
            event_id=current.event_id,
            market=current.market,
            selection=current.selection,
            movement_type=MovementType.ML_MOVE,
            direction=direction,
            magnitude=price_diff,
            from_price=opening.price,
            to_price=current.price,
            from_line=None,
            to_line=None,
            detail=detail,
        )

    def _detect_line_move_new(
        self,
        current: MarketOdds,
        opening_market_odds: list[MarketOdds],
    ) -> MovementSignal | None:
        """
        Detecta LINE_MOVE cuando una selección existe en el estado actual
        pero su línea no estaba en el snapshot de apertura.

        Ejemplo: TOTAL estaba en 8.5, ahora está en 9.0. La selección
        'over' con line=9.0 no existe en apertura (que tenía line=8.5).

        Busca la misma selection en apertura con cualquier línea para
        calcular el delta.
        """
        if current.line is None:
            return None  # Sin línea no hay LINE_MOVE

        # Buscar la misma selección en apertura con cualquier línea
        opening_same_selection = [
            o for o in opening_market_odds
            if o.selection == current.selection and o.line is not None
        ]
        if not opening_same_selection:
            return None  # Selección completamente nueva, sin referencia

        # Tomar la apertura más cercana en línea
        opening = min(
            opening_same_selection,
            key=lambda o: abs(o.line - current.line)  # type: ignore
        )
        line_diff = abs(current.line - opening.line)  # type: ignore

        if line_diff < self._thresholds.line_move_threshold:
            return None

        # Línea subió en totales → dinero en Over forzó la línea hacia arriba
        direction = MovementDirection.NEUTRAL
        if current.market == "TOTAL":
            if current.line > opening.line and current.selection.lower() == "over":  # type: ignore
                direction = MovementDirection.CONTRADICTS  # mercado empujó línea contra over
            elif current.line < opening.line and current.selection.lower() == "under":  # type: ignore
                direction = MovementDirection.CONTRADICTS

        detail = (
            f"Línea movió: {opening.line} → {current.line} "
            f"(∆{line_diff:+.1f})"
        )

        return MovementSignal(
            event_id=current.event_id,
            market=current.market,
            selection=current.selection,
            movement_type=MovementType.LINE_MOVE,
            direction=direction,
            magnitude=line_diff,
            from_price=opening.price,
            to_price=current.price,
            from_line=opening.line,
            to_line=current.line,
            detail=detail,
        )

    def _detect_juice_shift(
        self,
        market: str,
        event_id: str,
        current_group: list[MarketOdds],
        opening_group: list[MarketOdds],
    ) -> MovementSignal | None:
        """
        Detecta JUICE_SHIFT: los precios se movieron asimétricamente
        con la misma línea.

        Solo aplica para mercados binarios (TOTAL, SPREAD) donde hay
        exactamente dos lados. Para mercados N-arios (1X2, outright)
        el juice shift es más complejo y se omite en esta versión.

        El juice shift es la señal más sutil y más informativa:
        indica que el dinero inteligente (sharp) está entrando en
        un lado sin que la línea haya reaccionado aún.
        """
        if market not in ("TOTAL", "SPREAD"):
            return None

        # Necesitamos exactamente 2 selecciones en cada snapshot
        if len(current_group) != 2 or len(opening_group) != 2:
            return None

        # Verificar que las líneas son iguales (mismo mercado/línea)
        current_lines = {o.line for o in current_group}
        opening_lines = {o.line for o in opening_group}
        if current_lines != opening_lines:
            return None  # Línea cambió — es LINE_MOVE, no JUICE_SHIFT

        # Construir mapping selection → precio
        current_prices = {o.selection.lower(): o for o in current_group}
        opening_prices = {o.selection.lower(): o for o in opening_group}

        # Detectar movimiento asimétrico: un lado sube, el otro baja
        max_shift = 0.0
        shifted_selection: str | None = None
        shifted_direction = MovementDirection.NEUTRAL

        for sel, current_odds in current_prices.items():
            opening_odds = opening_prices.get(sel)
            if opening_odds is None:
                continue
            shift = current_odds.price - opening_odds.price
            if abs(shift) > max_shift:
                max_shift = abs(shift)
                shifted_selection = current_odds.selection
                # Precio bajó en esta selección → dinero entrando aquí
                shifted_direction = (
                    MovementDirection.CONFIRMS if shift < 0
                    else MovementDirection.CONTRADICTS
                )

        if max_shift < self._thresholds.juice_shift_threshold:
            return None
        if shifted_selection is None:
            return None

        # Resumir el movimiento de precios
        price_summary = ", ".join(
            f"{sel}: {opening_prices[sel].price:.2f}→{curr.price:.2f}"
            for sel, curr in current_prices.items()
            if sel in opening_prices
        )

        return MovementSignal(
            event_id=event_id,
            market=market,
            selection=shifted_selection,
            movement_type=MovementType.JUICE_SHIFT,
            direction=shifted_direction,
            magnitude=max_shift,
            from_price=None,
            to_price=None,
            from_line=next(iter(opening_lines), None),
            to_line=next(iter(current_lines), None),
            detail=f"Juice shift ({price_summary})",
        )