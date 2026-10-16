"""
core/odds/book_archive.py — Archivo de precios por casa de apuestas.

OBJETIVO
--------
Persistir TODOS los precios que ya descargamos, con la dimensión de
casa intacta, para que puedan analizarse después.

Hoy esa información se descarga y se tira. El pipeline pide a The Odds
API los mercados de cada evento —y la respuesta trae el precio de cada
casa— pero `OddsNormalizer.extract_best()` colapsa esa dimensión y
conserva solo el mejor precio por selección. El resto se descarta sin
haberse guardado nunca.

Eso importa por dos razones concretas:

  1. **Coste cero.** Los precios por casa vienen en la misma respuesta
     HTTP que ya pagamos en créditos. Persistirlos no cuesta un solo
     crédito adicional.
  2. **Irrecuperable.** No hay proveedor barato de histórico de precios
     por casa de NBA. Al contrario que las líneas de consenso —que The
     Odds API conserva y revende— lo que no capturemos hoy no se puede
     comprar mañana.

POR QUÉ ES UN MÓDULO APARTE Y NO UNA MODIFICACIÓN
--------------------------------------------------
La tentación era redirigir el snapshot que el Stage 5 ya guarda:
cambiar `extract_best()` por `extract_all()` en la llamada a
`LineMovementDetector.snapshot()`. Habría sido una línea.

Rompe el Stage 7. `OddsSnapshot.get_odds_map()` indexa por la clave
`(market, selection, line)`, que con precios de varias casas deja de
ser única: el diccionario se queda con la última casa que entre, de
forma arbitraria y silenciosa. La detección de movimiento compararía
snapshots formados por casas distintas y produciría señales falsas.

De ahí la decisión: el archivo de casas es un consumidor NUEVO del
mismo dato crudo, en paralelo al snapshot existente, que no se toca.
Compatibilidad hacia atrás total.

ARQUITECTURA
------------
    Stage 5 (runner)
        raw_event  ──┬──► normalizer.extract_best()  ──► market_odds
                     │        (sin cambios: alimenta value engine
                     │         y el snapshot de line movement)
                     │
                     └──► normalizer.extract_all()   ──► BookArchive
                              (nuevo: todas las casas, a disco)

    BookArchive            decide QUÉ se archiva y con qué metadatos
        └── BookArchiveStore (Protocol)
                ├── CsvBookArchiveStore    producción
                └── InMemoryBookArchiveStore  tests

    Mismo patrón que `SnapshotStore` en line_movement.py y
    `CsvLedgerStore` en bankroll/tracker.py: un Protocol con una
    implementación de disco y otra en memoria.

LÍMITE DE RESPONSABILIDAD
-------------------------
Este módulo **solo captura**. No resuelve la línea de cierre, no
deduplica, no calcula consenso ni CLV. Eso es el consolidador de la
tarea 12.6, que leerá este archivo.

La razón de separarlo: capturar tiene fecha de caducidad y consolidar
no. Lo que no se escriba hoy se pierde; lo que se consolide puede
reprocesarse las veces que haga falta sobre los mismos datos crudos.

FORMATO
-------
CSV de solo-añadir, un archivo por deporte y mes:

    output/book_archive/{sport}/{YYYY-MM}.csv

CSV y no parquet por tres razones: `core/` no importa pandas (solo
`yaml`, en un archivo) y meterlo aquí sería deuda nueva; añadir al
final de un CSV es seguro ante una caída, mientras que reescribir un
parquet no lo es; y el proyecto ya usa CSV para el ledger, así que es
consistente.

Volumen estimado para NBA: ~3 mercados × 2 selecciones × ~12 casas =
~72 filas por evento y captura. Con 10 partidos al día y 3 capturas
diarias son ~65.000 filas al mes, en torno a 8 MB. Despreciable.

DUPLICADOS
----------
Se toleran deliberadamente. Dos ejecuciones del pipeline el mismo día
producen dos capturas del mismo evento, y eso es justo lo que se
quiere: la secuencia de capturas ES el movimiento de línea.

Una re-ejecución accidental añade filas casi idénticas, distinguibles
por `captured_at`. Deduplicar aquí exigiría leer el archivo entero en
cada escritura y convertiría una operación de solo-añadir en una de
lectura-y-reescritura. El consolidador deduplica por la clave
(event_id, market, selection, line, bookmaker, captured_at).

RIESGOS
-------
  * Un fallo de disco pierde la captura. Por eso `capture()` nunca
    lanza hacia el pipeline: devuelve 0 y registra el motivo. Perder
    una captura no debe tumbar una ejecución que iba a producir picks.
  * El archivo crece sin límite. La rotación mensual lo acota por
    archivo, pero nadie purga. Es aceptable: son los datos que no se
    pueden recomprar.
  * `commence_time` llega como texto de la API. Si no se puede
    interpretar, `lead_minutes` queda vacío en vez de romper la fila:
    el precio se conserva igual, que es lo que importa.

MEJORAS FUTURAS
---------------
  * Rotación a parquet comprimido desde un script de mantenimiento en
    `scripts/`, donde pandas sí está disponible.
  * Capturar también `last_update` por casa, que The Odds API provee y
    `MarketOdds` no modela todavía. Permitiría distinguir un precio
    rancio de uno recién movido.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Protocol, Sequence


# ── Esquema del archivo ───────────────────────────────────────────────────────
#
# El orden es el orden de las columnas en el CSV. Añadir campos nuevos
# SOLO al final: un archivo escrito con el esquema viejo debe seguir
# siendo legible, y el consolidador lee por nombre de columna.

FIELDNAMES: tuple[str, ...] = (
    "captured_at",      # ISO-8601 UTC — cuándo lo guardamos nosotros
    "sport",
    "event_id",
    "commence_time",    # ISO-8601 UTC — tal como lo da la API
    "lead_minutes",     # minutos entre la captura y el tip-off
    "market",
    "selection",
    "line",
    "price",
    "bookmaker",
    "odds_timestamp",   # timestamp que trae el propio MarketOdds
)

_DEFAULT_BASE_DIR = "output/book_archive"


# ── Protocol de almacenamiento ────────────────────────────────────────────────

class BookArchiveStore(Protocol):
    """
    Destino de las filas del archivo.

    Mismo patrón que `SnapshotStore`: el Protocol permite sustituir
    disco por memoria en los tests sin tocar la lógica de captura.
    """

    def append(self, sport: str, period: str, rows: Sequence[dict]) -> int:
        """
        Añade `rows` al archivo de `sport` en el periodo `period`
        (formato 'YYYY-MM'). Retorna cuántas filas se escribieron.
        """
        ...


class CsvBookArchiveStore:
    """
    Implementación de disco: un CSV de solo-añadir por deporte y mes.

    Parámetros
    ----------
    base_dir  -- Directorio raíz. Default 'output/book_archive'.
    """

    def __init__(self, base_dir: str = _DEFAULT_BASE_DIR) -> None:
        self._base_dir = base_dir

    def path_for(self, sport: str, period: str) -> str:
        return os.path.join(self._base_dir, sport, f"{period}.csv")

    def append(self, sport: str, period: str, rows: Sequence[dict]) -> int:
        if not rows:
            return 0

        path = self.path_for(sport, period)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # El encabezado se escribe solo la primera vez. Comprobar la
        # existencia ANTES de abrir en modo 'a' es necesario: abrir ya
        # crea el archivo vacío.
        write_header = not os.path.exists(path)

        with open(path, "a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=FIELDNAMES,
                extrasaction="ignore",
            )
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        return len(rows)


class InMemoryBookArchiveStore:
    """
    Implementación en memoria para tests.

    Conserva las filas por (sport, period) para poder afirmar sobre lo
    que se habría escrito, sin tocar disco.
    """

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str], list[dict]] = {}

    def append(self, sport: str, period: str, rows: Sequence[dict]) -> int:
        if not rows:
            return 0
        self.rows.setdefault((sport, period), []).extend(rows)
        return len(rows)

    def all_rows(self) -> list[dict]:
        out: list[dict] = []
        for value in self.rows.values():
            out.extend(value)
        return out

    def clear(self) -> None:
        self.rows.clear()


# ── Resultado de una captura ──────────────────────────────────────────────────

@dataclass(frozen=True)
class CaptureResult:
    """
    Qué ocurrió en una captura. Inmutable: es un hecho consumado.

    Campos
    ------
    rows_written -- Filas efectivamente escritas.
    events       -- Eventos distintos capturados.
    bookmakers   -- Casas distintas vistas.
    errors       -- Motivos por los que algo no se escribió. El
                   pipeline los agrega a su propio informe; una
                   captura fallida nunca aborta la ejecución.
    """

    rows_written: int = 0
    events:       int = 0
    bookmakers:   int = 0
    errors:       tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        base = (
            f"book_archive: {self.rows_written} filas, "
            f"{self.events} eventos, {self.bookmakers} casas"
        )
        if self.errors:
            base += f", {len(self.errors)} errores"
        return base


# ── Captura ───────────────────────────────────────────────────────────────────

class BookArchive:
    """
    Decide qué se archiva y con qué metadatos.

    Parámetros
    ----------
    store    -- Destino de las filas (BookArchiveStore).
    enabled  -- Si False, `capture()` no hace nada y devuelve un
               CaptureResult vacío. Permite desactivar el archivo
               desde configuración sin quitar el cableado.

    Por qué la firma es de primitivos y no de `Event`
    --------------------------------------------------
    `capture()` recibe event_id, sport y commence_time sueltos en vez
    del contrato `Event`. Así puede llamarlo también
    `scripts/capture_closing.py`, que trabaja con entradas del ledger y
    no construye Events, y los tests no necesitan fabricar un Event
    completo para probar una sola fila.
    """

    def __init__(
        self,
        store:   BookArchiveStore,
        enabled: bool = True,
    ) -> None:
        self._store   = store
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    def capture(
        self,
        *,
        sport:         str,
        event_id:      str,
        commence_time: str | None,
        odds:          Iterable,          # Iterable[MarketOdds]
        captured_at:   datetime | None = None,
    ) -> CaptureResult:
        """
        Archiva todos los precios de un evento.

        `odds` debe venir de `OddsNormalizer.extract_all()`, que
        devuelve un MarketOdds por (selection, line, bookmaker). Si se
        le pasa la salida de `extract_best()` el archivo queda correcto
        pero pobre: una sola casa por selección, que es exactamente la
        información que este módulo existe para no perder.

        Nunca lanza. Un fallo de disco devuelve un CaptureResult con
        `errors` poblado: perder una captura no debe tumbar una
        ejecución que iba a producir picks.
        """
        if not self._enabled:
            return CaptureResult()

        moment = captured_at or datetime.now(timezone.utc)
        lead   = _lead_minutes(commence_time, moment)

        rows: list[dict] = []
        books: set[str] = set()

        for quote in odds:
            bookmaker = getattr(quote, "bookmaker", None) or "unknown"
            books.add(bookmaker)
            rows.append({
                "captured_at":    _iso(moment),
                "sport":          sport,
                "event_id":       event_id,
                "commence_time":  commence_time or "",
                "lead_minutes":   "" if lead is None else f"{lead:.2f}",
                "market":         getattr(quote, "market", ""),
                "selection":      getattr(quote, "selection", ""),
                "line":           _blank_if_none(getattr(quote, "line", None)),
                "price":          getattr(quote, "price", ""),
                "bookmaker":      bookmaker,
                "odds_timestamp": getattr(quote, "timestamp", "") or "",
            })

        if not rows:
            return CaptureResult(events=0)

        period = _period_of(moment)
        try:
            written = self._store.append(sport, period, rows)
        except Exception as error:
            return CaptureResult(
                errors=(
                    f"book_archive: no se pudo escribir {len(rows)} filas de "
                    f"event_id='{event_id}' ({type(error).__name__}: {error}). "
                    f"La captura se pierde; la ejecución continúa.",
                ),
            )

        return CaptureResult(
            rows_written=written,
            events=1,
            bookmakers=len(books),
        )

    def capture_many(
        self,
        *,
        sport:       str,
        events:      Sequence[tuple[str, str | None, Iterable]],
        captured_at: datetime | None = None,
    ) -> CaptureResult:
        """
        Archiva varios eventos y agrega los resultados.

        `events` es una secuencia de (event_id, commence_time, odds).
        Se fija un único `captured_at` para todo el lote: las cuotas
        provienen de la misma respuesta HTTP, así que compartir el
        instante de captura es más fiel que tomar el reloj por evento.
        """
        moment = captured_at or datetime.now(timezone.utc)
        total_rows = 0
        total_events = 0
        books: set[str] = set()
        errors: list[str] = []

        for event_id, commence_time, odds in events:
            materialized = list(odds)
            result = self.capture(
                sport=sport,
                event_id=event_id,
                commence_time=commence_time,
                odds=materialized,
                captured_at=moment,
            )
            total_rows += result.rows_written
            total_events += result.events
            errors.extend(result.errors)
            for quote in materialized:
                book = getattr(quote, "bookmaker", None) or "unknown"
                books.add(book)

        return CaptureResult(
            rows_written=total_rows,
            events=total_events,
            bookmakers=len(books),
            errors=tuple(errors),
        )


# ── Utilidades de módulo ──────────────────────────────────────────────────────

def _iso(moment: datetime) -> str:
    """ISO-8601 en UTC con sufijo Z, consistente con el resto del Core."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (
        moment.astimezone(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _period_of(moment: datetime) -> str:
    """Periodo de rotación: 'YYYY-MM'."""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).strftime("%Y-%m")


def _lead_minutes(commence_time: str | None, moment: datetime) -> float | None:
    """
    Minutos entre la captura y el tip-off. Positivo = la captura
    precede al partido.

    Devuelve None si `commence_time` no se puede interpretar. Esa
    decisión es deliberada: un `commence_time` ilegible no debe impedir
    que se guarde el precio, que es el dato irrecuperable. El
    consolidador puede recomputar el lead cruzando contra el
    calendario.
    """
    parsed = parse_iso8601(commence_time)
    if parsed is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return (parsed - moment).total_seconds() / 60.0


def parse_iso8601(value: str | None) -> datetime | None:
    """
    Interpreta un timestamp ISO-8601, tolerando el sufijo 'Z'.

    `datetime.fromisoformat` no acepta 'Z' antes de Python 3.11, y The
    Odds API lo usa siempre. Se normaliza antes de delegar para no
    depender de la versión del intérprete.
    """
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _blank_if_none(value) -> str:
    """
    Celda vacía para None, en vez de la cadena 'None'.

    Importa en el consolidador: un moneyline no tiene línea, y la
    cadena literal 'None' obligaría a cada lector a tratarla como caso
    especial. Una celda vacía la interpreta como nulo cualquier lector
    de CSV.
    """
    return "" if value is None else str(value)


def build_default_archive(
    base_dir: str = _DEFAULT_BASE_DIR,
    enabled:  bool = True,
) -> BookArchive:
    """
    Fábrica de conveniencia para el cableado del runner.

    Mantiene el `CsvBookArchiveStore` como detalle de construcción, de
    forma que cambiar el almacenamiento no obligue a tocar el runner.
    """
    return BookArchive(store=CsvBookArchiveStore(base_dir), enabled=enabled)


__all__ = [
    "FIELDNAMES",
    "BookArchive",
    "BookArchiveStore",
    "CsvBookArchiveStore",
    "InMemoryBookArchiveStore",
    "CaptureResult",
    "build_default_archive",
    "parse_iso8601",
]