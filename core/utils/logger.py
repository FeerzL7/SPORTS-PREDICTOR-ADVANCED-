"""
core/utils/logger.py

Logging estructurado, agnóstico al deporte.

Migrado desde utils/logger.py del sistema MLB-PREDICTOR-ADVANCED
"sin cambios funcionales" (ver SPORTS_PREDICTOR_ROADMAP.md, tarea
C1.4.9). La única generalización aplicada es paramétrica, no de
comportamiento: el sistema original tenía el nombre del logger y el
prefijo del archivo hardcodeados al string "mlb"
(logging.getLogger("mlb"), logs/mlb_YYYY-MM-DD.log), lo cual es
funcionalmente un acoplamiento a un único deporte — incompatible con
el principio rector de SPORTS_PREDICTOR_ARCHITECTURE.md de que el
Core no puede conocer conceptos deportivos específicos.

Aquí logger_name es un parámetro con default "sports_predictor".
Llamando configurar(logger_name="mlb") se reproduce exactamente el
comportamiento original (logs/mlb_YYYY-MM-DD.log, logger interno
"mlb"), preservando compatibilidad total con los logs históricos ya
existentes del sistema MLB cuando se migre el plugin en Fase 2. Otros
sport plugins (soccer, nba, etc.) usan su propio logger_name sin
colisionar logs entre deportes corriendo en la misma sesión de
Python — caso real en backtesting multi-sport (Fase 3 del roadmap).

Niveles usados en el proyecto (idéntico al sistema MLB original):
    DEBUG   → datos internos de cálculo (proyecciones, features por evento)
    INFO    → flujo principal del pipeline (inicio, stages, picks finales)
    WARNING → datos faltantes, fallbacks activados, APIs con respuesta parcial
    ERROR   → fallos que impiden completar un stage (sin cuotas, sin eventos)

Salida (idéntica al sistema MLB original):
    - Consola: nivel INFO y superior, formato compacto con color
    - Archivo: nivel DEBUG y superior, formato completo con timestamp
      Ruta: logs/{logger_name}_YYYY-MM-DD.log (uno por día, rotación
      automática a medianoche, 30 días de retención)

Uso típico:
    from core.utils.logger import configurar, get

    log = configurar(logger_name="mlb")   # primera línea de run_daily.py
    log.info("Iniciando pipeline MLB...")

    # en cualquier otro módulo del mismo proceso:
    log = get(logger_name="mlb")
    log.debug("proj_home=4.8 proj_away=3.9")
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler

# ── Colores para consola (ANSI) ───────────────────────────────────────────────
_RESET  = "\x1b[0m"
_BOLD   = "\x1b[1m"
_COLORS = {
    'DEBUG':    "\x1b[36m",    # cyan
    'INFO':     "\x1b[32m",    # verde
    'WARNING':  "\x1b[33m",    # amarillo
    'ERROR':    "\x1b[31m",    # rojo
    'CRITICAL': "\x1b[35m",    # magenta
}

# Nombre por defecto del logger y del prefijo de archivo cuando el
# llamador no especifica logger_name. "mlb" reproduce exactamente el
# comportamiento del sistema original; cualquier otro sport plugin
# debe pasar su propio nombre explícitamente.
DEFAULT_LOGGER_NAME = "sports_predictor"


class _ColorFormatter(logging.Formatter):
    """Formatter con colores ANSI para la consola."""

    FMT = "{color}{level:<8}{reset} {message}"

    def format(self, record: logging.LogRecord) -> str:
        color = _COLORS.get(record.levelname, "")
        level = record.levelname
        # Recortar módulo largo para que el log sea legible
        record.module_short = record.module[:20]
        msg = super().format(record)
        return self.FMT.format(
            color=color, level=level, reset=_RESET, message=record.getMessage()
        )


class _FileFormatter(logging.Formatter):
    """Formatter para archivo: timestamp ISO + nivel + módulo + mensaje."""

    def format(self, record: logging.LogRecord) -> str:
        ts  = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S")
        lvl = record.levelname
        mod = f"{record.module}.{record.funcName}"[:35]
        msg = record.getMessage()

        base = f"{ts} | {lvl:<8} | {mod:<35} | {msg}"

        # Adjuntar excepción si existe
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configurar(
    logger_name: str = DEFAULT_LOGGER_NAME,
    nivel_consola: str = "INFO",
    nivel_archivo: str = "DEBUG",
    directorio_logs: str = "logs",
) -> logging.Logger:
    """
    Configura y devuelve el logger raíz de un sport plugin o del Core.

    Llamar una sola vez al inicio del entry point (scripts/run_daily.py
    --sport <logger_name>). Llamadas posteriores con el mismo
    logger_name retornan el logger ya configurado sin duplicar
    handlers (protección idéntica a la del sistema MLB original).

    Parámetros
    ----------
    logger_name      -- Determina tanto el nombre del logger interno
                       (logging.getLogger(logger_name)) como el
                       prefijo del archivo de log
                       (logs/{logger_name}_YYYY-MM-DD.log). Usar el
                       sport_id del plugin (ej. "mlb", "soccer", "nba")
                       para que logs de distintos deportes corriendo
                       en la misma sesión de Python no colisionen.
                       Default "sports_predictor" para uso genérico
                       del Core fuera del contexto de un sport
                       específico (ej. scripts de migración).
    nivel_consola     -- Nivel mínimo mostrado en consola.
    nivel_archivo      -- Nivel mínimo escrito en archivo.
    directorio_logs     -- Carpeta donde se crean los archivos rotados.
    """
    logger = logging.getLogger(logger_name)
    if logger.handlers:
        # Ya configurado (ej: doble import en tests, o segunda llamada
        # con el mismo logger_name dentro de la misma sesión)
        return logger

    logger.setLevel(logging.DEBUG)

    # ── Handler de consola ────────────────────────────────────────────────────
    stream = sys.stdout
    if hasattr(stream, "reconfigure"):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace") # type: ignore
        except Exception:
            pass

    ch = logging.StreamHandler(stream)
    ch.setLevel(getattr(logging, nivel_consola.upper(), logging.INFO))
    ch.setFormatter(_ColorFormatter())
    logger.addHandler(ch)

    # ── Handler de archivo con rotación diaria ────────────────────────────────
    os.makedirs(directorio_logs, exist_ok=True)
    hoy       = datetime.now().strftime("%Y-%m-%d")
    log_path  = os.path.join(directorio_logs, f"{logger_name}_{hoy}.log")

    fh = TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",       # rota a medianoche
        interval=1,
        backupCount=30,        # conserva los últimos 30 días
        encoding="utf-8",
    )
    fh.setLevel(getattr(logging, nivel_archivo.upper(), logging.DEBUG))
    fh.setFormatter(_FileFormatter())
    logger.addHandler(fh)

    # Evitar que los mensajes suban al logger raíz de Python
    logger.propagate = False

    logger.info(f"Logger iniciado → {log_path}")
    return logger


def get(logger_name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """
    Devuelve el logger ya configurado para logger_name.

    El logger_name debe coincidir con el usado en configurar() para
    obtener la instancia correcta. Si dos sport plugins llaman get()
    sin especificar logger_name, ambos reciben el mismo logger
    "sports_predictor" por defecto — pasar logger_name explícitamente
    es responsabilidad de cada sport plugin para evitar mezclar logs
    de deportes distintos en la misma sesión.
    """
    return logging.getLogger(logger_name)