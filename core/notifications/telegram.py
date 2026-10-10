"""
core/notifications/telegram.py

TelegramNotifier: Stage 12 del pipeline — envío de picks y resumen
diario via Telegram Bot API.

Migrado de notifications/telegram.py del sistema MLB con dos
correcciones documentadas en SPORTS_PREDICTOR_ARCHITECTURE.md §1112:

1. Formato de pick genérico — opera sobre CandidatePick tipado.
   El sistema MLB usaba formatear_pick(partido: dict) con keys
   MLB-específicos (pitcher_local, carreras_proyectadas, mejor_pick).
   Aquí format_pick(pick: CandidatePick) usa solo campos del contrato
   tipado — funciona para cualquier deporte sin modificación.

2. chat_ids como lista — soporte para múltiples canales.
   El sistema MLB tenía un único CHAT_ID. Aquí TelegramConfig.chat_ids
   es list[str] — el operador puede enviar a su chat personal, un canal
   público y un grupo premium simultáneamente.

Posición en el pipeline
------------------------
Stage 12 — Output & Notification (el último stage).
    PipelineResult.active_picks + roi_summary
        ↓
    TelegramNotifier.send_picks()
        → Mensaje por pick + mensaje de resumen

Stage 12 es completamente no-bloqueante: si Telegram falla (API caída,
rate limit, token inválido), el pipeline retorna PipelineResult sin
error fatal — los picks ya están en el ledger y son la fuente de verdad.

NotificationChannel como Protocol
-----------------------------------
TelegramNotifier implementa NotificationChannel — la misma interfaz
que futuros notificadores (Discord, email, SMS). El runner llama
channel.send_picks() sin saber qué canal es.

Rate limiting
--------------
Telegram permite 30 msg/seg por bot pero 1 msg/seg por chat.
TelegramConfig.rate_limit_delay=0.1s garantiza que el bot no excede
el límite incluso enviando a múltiples chats simultáneamente.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from core.contracts.pick import CandidatePick

try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False

_TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/{method}"


# ── Protocol ──────────────────────────────────────────────────────────────────

@runtime_checkable
class NotificationChannel(Protocol):
    """
    Interfaz de canal de notificación.

    Implementada por TelegramNotifier, DiscordNotifier, etc.
    El runner llama send_picks() sin saber qué canal usa.
    """

    def send_picks(
        self,
        picks:       list[CandidatePick],
        roi_summary: dict,
        date:        str,
    ) -> "NotificationResult":
        """
        Envía todos los picks activos del día y el resumen de ROI.

        Nunca lanza excepción — errores se registran en
        NotificationResult.failures.
        """
        ...

    def send_message(
        self,
        text:    str,
        chat_id: str | None = None,
    ) -> bool:
        """
        Envía un mensaje de texto libre.

        Retorna True si se envió a al menos un chat.
        """
        ...


# ── Configuración ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TelegramConfig:
    """
    Configuración inmutable del TelegramNotifier.

    Campos
    ------
    bot_token         -- Token del bot de Telegram.
                        Formato: '123456789:ABCdef...'
                        NUNCA hardcodear — leer de .env o secrets.yaml.
    chat_ids          -- Lista de chat IDs a notificar.
                        Puede ser chat personal ('-123456'),
                        canal ('@mi_canal') o grupo ('-987654321').
    parse_mode        -- 'HTML' o 'MarkdownV2'. Default 'HTML'.
                        El formateador de picks usa HTML por defecto.
    rate_limit_delay  -- Segundos entre mensajes al mismo chat.
                        Default 0.1s — respeta el límite de Telegram
                        (1 msg/seg/chat) con margen de seguridad.
    timeout_seconds   -- Timeout HTTP por request. Default 10s.
    disable_preview   -- Desactiva preview de links en mensajes.
    """
    bot_token:        str
    chat_ids:         list[str]
    parse_mode:       str   = "HTML"
    rate_limit_delay: float = 0.1
    timeout_seconds:  int   = 10
    disable_preview:  bool  = True

    def __post_init__(self) -> None:
        if not self.bot_token or not self.bot_token.strip():
            raise ValueError(
                "TelegramConfig.bot_token no puede estar vacío. "
                "Definir TELEGRAM_BOT_TOKEN en .env o secrets.yaml."
            )
        if not self.chat_ids:
            raise ValueError(
                "TelegramConfig.chat_ids no puede estar vacío. "
                "Definir al menos un TELEGRAM_CHAT_ID."
            )

    def redacted_token(self) -> str:
        """Token con solo los primeros 8 caracteres visibles."""
        return self.bot_token[:8] + "****" if len(self.bot_token) > 8 else "****"


# ── Resultado de notificación ─────────────────────────────────────────────────

@dataclass(frozen=True)
class NotificationResult:
    """
    Resultado inmutable de un ciclo de notificación.

    Campos
    ------
    success          -- True si al menos un mensaje se envió.
    messages_sent    -- Número total de mensajes enviados con éxito.
    failures         -- Lista de errores (chat_id + motivo).
    chat_ids_reached -- Lista de chat IDs que recibieron los mensajes.
    """
    success:          bool
    messages_sent:    int
    failures:         list[str]
    chat_ids_reached: list[str]

    def summary(self) -> str:
        return (
            f"Telegram: {self.messages_sent} mensajes enviados | "
            f"chats={len(self.chat_ids_reached)} | "
            f"errores={len(self.failures)}"
        )


# ── Formateadores ─────────────────────────────────────────────────────────────

def format_pick(pick: CandidatePick, index: int = 1) -> str:
    """
    Formatea un CandidatePick como mensaje HTML para Telegram.

    Completamente agnóstico al deporte — usa solo campos del contrato
    CandidatePick. El deporte se muestra como pick.event.sport.upper().

    Formato:
        🎯 PICK #1 — MLB | TOTAL
        ━━━━━━━━━━━━━━━━━━━━
        🏟 BOS @ NYY
        📌 Over 8.5
        💰 Cuota: 1.91
        📊 EV: +8.25% | Edge: 0.0700
        🎲 Prob modelo: 57.5%
        💼 Stake: 2% del bankroll
        ━━━━━━━━━━━━━━━━━━━━
    """
    sport    = pick.event.sport.upper()
    market   = pick.market
    away     = pick.event.away_team
    home     = pick.event.home_team
    sel      = pick.selection
    price    = pick.price
    ev       = pick.ev
    edge     = pick.edge
    prob     = pick.blended_prob
    stake    = pick.stake_pct
    line_str = f" {pick.line}" if pick.line is not None else ""

    # Emoji de tendencia por EV
    ev_emoji = "🔥" if ev >= 15 else ("✅" if ev >= 8 else "⚠️")

    lines = [
        f"🎯 <b>PICK #{index}</b> — {sport} | {market}",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🏟 {away} @ {home}",
        f"📌 {sel}{line_str}",
        f"💰 Cuota: <b>{price}</b>",
        f"{ev_emoji} EV: <b>{ev:+.2f}%</b> | Edge: {edge:.4f}",
        f"🎲 Prob modelo: {prob:.1%}",
        f"💼 Stake: <b>{stake}%</b> del bankroll",
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(lines)


def format_roi_summary(summary: dict, date: str) -> str:
    """
    Formatea el resumen de ROI diario como mensaje HTML para Telegram.

    Parámetros
    ----------
    summary  -- Dict retornado por ROITracker.roi_summary().
    date     -- Fecha de la ejecución en 'YYYY-MM-DD'.
    """
    sport     = summary.get("sport", "all").upper()
    n         = summary.get("total_apuestas", 0)
    wins      = summary.get("wins", 0)
    losses    = summary.get("losses", 0)
    roi       = summary.get("roi", 0.0)
    bankroll  = summary.get("bankroll", 0.0)
    pending   = summary.get("pendientes", 0)
    clv_mean  = summary.get("clv_mean")
    drawdown  = summary.get("max_drawdown", 0.0)

    hit_rate  = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
    roi_emoji = "📈" if roi >= 0 else "📉"
    clv_str   = f"\n📐 CLV medio: <b>{clv_mean:+.2f}%</b>" if clv_mean is not None else ""

    lines = [
        f"📊 <b>RESUMEN {sport} — {date}</b>",
        "━━━━━━━━━━━━━━━━━━━━",
        f"🏆 Resultados: {wins}W / {losses}L",
        f"🎯 Hit Rate: {hit_rate:.1f}%",
        f"{roi_emoji} ROI: <b>{roi:+.2f}%</b>",
        f"🏦 Bankroll: <b>{bankroll:.2f}</b>",
        f"📉 Max Drawdown: {drawdown:.2f}%",
        f"⏳ Pendientes: {pending}",
        clv_str,
        "━━━━━━━━━━━━━━━━━━━━",
    ]
    return "\n".join(l for l in lines if l)


def format_no_picks(date: str, sport: str = "") -> str:
    """Mensaje cuando no hay picks activos para el día."""
    sport_str = f" {sport.upper()}" if sport else ""
    return (
        f"🔕 <b>Sin picks{sport_str} para {date}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"El sistema no encontró valor suficiente hoy.\n"
        f"Los filtros de EV y riesgo no aprobaron candidatos."
    )


# ── TelegramNotifier ──────────────────────────────────────────────────────────

class TelegramNotifier:
    """
    Notificador via Telegram Bot API.

    Implementa NotificationChannel. Stage 12 del pipeline diario.

    Completamente no-bloqueante: si Telegram falla, el pipeline
    no se ve afectado — los picks ya están en el ledger.

    Parámetros
    ----------
    config  -- TelegramConfig con token y chat IDs.
    """

    def __init__(self, config: TelegramConfig) -> None:
        self._config = config

    # ── API pública ────────────────────────────────────────────────────────────

    def send_picks(
        self,
        picks:       list[CandidatePick],
        roi_summary: dict,
        date:        str,
    ) -> NotificationResult:
        """
        Envía los picks activos del día y el resumen de ROI.

        Flujo:
        1. Si no hay picks → envía mensaje de "sin picks hoy"
        2. Por cada pick activo → envía mensaje formateado
        3. Envía resumen de ROI al final

        Nunca lanza excepción. Errores se acumulan en NotificationResult.
        """
        messages_sent    = 0
        failures:  list[str] = []
        reached:   list[str] = []

        if not picks:
            sport = roi_summary.get("sport", "")
            text  = format_no_picks(date, sport)
            ok, errs, rchd = self._broadcast(text)
            return NotificationResult(
                success          = ok > 0,
                messages_sent    = ok,
                failures         = errs,
                chat_ids_reached = rchd,
            )

        # Enviar un mensaje por pick
        for i, pick in enumerate(picks, start=1):
            text = format_pick(pick, index=i)
            ok, errs, _ = self._broadcast(text)
            messages_sent += ok
            failures.extend(errs)
            if ok > 0:
                time.sleep(self._config.rate_limit_delay)

        # Enviar resumen de ROI
        summary_text = format_roi_summary(roi_summary, date)
        ok, errs, rchd = self._broadcast(summary_text)
        messages_sent += ok
        failures.extend(errs)
        reached = rchd

        return NotificationResult(
            success          = messages_sent > 0,
            messages_sent    = messages_sent,
            failures         = failures,
            chat_ids_reached = reached,
        )

    def send_message(
        self,
        text:    str,
        chat_id: str | None = None,
    ) -> bool:
        """
        Envía un mensaje de texto libre.

        Parámetros
        ----------
        text     -- Texto del mensaje (puede incluir HTML).
        chat_id  -- Chat específico. None → enviar a todos los chat_ids.

        Retorna True si se envió a al menos un chat.
        """
        if chat_id:
            return self._send_to_chat(text, chat_id)
        ok, _, _ = self._broadcast(text)
        return ok > 0

    def send_roi_summary(
        self,
        summary: dict,
        date:    str,
    ) -> NotificationResult:
        """
        Envía solo el resumen de ROI (sin picks individuales).
        Útil para reportes periódicos o cuando se quiere el resumen
        independientemente del ciclo de picks.
        """
        text = format_roi_summary(summary, date)
        ok, errs, reached = self._broadcast(text)
        return NotificationResult(
            success          = ok > 0,
            messages_sent    = ok,
            failures         = errs,
            chat_ids_reached = reached,
        )

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _broadcast(
        self,
        text: str,
    ) -> tuple[int, list[str], list[str]]:
        """
        Envía el texto a todos los chat_ids configurados.

        Retorna (n_exitosos, lista_errores, chats_alcanzados).
        """
        sent    = 0
        errors: list[str] = []
        reached: list[str] = []

        for chat_id in self._config.chat_ids:
            ok = self._send_to_chat(text, chat_id)
            if ok:
                sent += 1
                reached.append(chat_id)
            else:
                errors.append(f"chat_id={chat_id}: envío fallido")
            time.sleep(self._config.rate_limit_delay)

        return sent, errors, reached

    def _send_to_chat(self, text: str, chat_id: str) -> bool:
        """
        Envía un mensaje a un chat específico via Telegram Bot API.

        Retorna True si el status code es 2xx.
        Nunca lanza excepción — errores de red retornan False.
        """
        if not _REQUESTS_AVAILABLE:
            return False

        url     = _TELEGRAM_API_BASE.format(
            token  = self._config.bot_token,
            method = "sendMessage",
        )
        payload = {
            "chat_id":                  chat_id,
            "text":                     text,
            "parse_mode":               self._config.parse_mode,
            "disable_web_page_preview": self._config.disable_preview,
        }

        try:
            response = _requests.post(
                url,
                json    = payload,
                timeout = self._config.timeout_seconds,
            )
            return 200 <= response.status_code < 300
        except Exception:
            return False

    # ── Factory desde ConfigLoader ─────────────────────────────────────────────

    @classmethod
    def from_config(cls, config_loader=None) -> "TelegramNotifier | None":
        """
        Construye un TelegramNotifier desde el entorno o el config.

        Orden de búsqueda de credenciales
        ----------------------------------
        1. Variables de entorno (.env) — donde viven de verdad
        2. ConfigLoader — por si alguien las pone en el YAML

        CORRECCIÓN: la versión anterior solo consultaba el
        ConfigLoader, que lee ÚNICAMENTE archivos YAML. Las
        credenciales de Telegram viven en .env, así que
        `config.get("TELEGRAM_BOT_TOKEN")` buscaba una clave YAML
        inexistente, devolvía None, y las notificaciones se saltaban
        en silencio — sin un mensaje que dijera por qué.

        Un secreto en el YAML además acabaría versionado en git, que
        es exactamente lo que .env evita. Por eso el entorno tiene
        prioridad.

        Variables reconocidas
        ----------------------
            TELEGRAM_BOT_TOKEN   token del bot, de @BotFather
            TELEGRAM_CHAT_IDS    ids separados por coma
            TELEGRAM_CHAT_ID     alias de una sola id, por compatibilidad
                                 con el sistema MLB anterior
            TELEGRAM_TOKEN       alias del token, mismo motivo

        Retorna None si faltan credenciales. Para saber POR QUÉ, usar
        `diagnose()`, que distingue "no configurado" de "configurado
        pero inválido".
        """
        token, chat_ids = cls._read_credentials(config_loader)
        if not token or not chat_ids:
            return None

        try:
            cfg = TelegramConfig(
                bot_token  = token,
                chat_ids   = chat_ids,
                parse_mode = cls._cfg(config_loader, "telegram.parse_mode",
                                      "HTML"),
                rate_limit_delay = float(
                    cls._cfg(config_loader, "telegram.rate_limit_delay", 0.1)
                ),
            )
            return cls(config=cfg)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _read_credentials(config_loader) -> tuple[str, list[str]]:
        """
        Lee token y chat_ids del entorno, con el config como respaldo.

        Carga el .env si python-dotenv está disponible: los scripts
        que llaman al notificador no siempre lo han hecho ya, y
        depender de ello convertía el orden de imports en parte del
        contrato.
        """
        import os

        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        def buscar(*claves: str) -> str:
            for clave in claves:
                valor = os.environ.get(clave, "").strip()
                if valor:
                    return valor
            if config_loader is not None:
                for clave in claves:
                    try:
                        valor = config_loader.get(clave, default=None)
                    except Exception:
                        continue
                    if valor:
                        return str(valor).strip()
            return ""

        token = buscar("TELEGRAM_BOT_TOKEN", "TELEGRAM_TOKEN")
        raw_ids = buscar("TELEGRAM_CHAT_IDS", "TELEGRAM_CHAT_ID")

        chat_ids = [cid.strip() for cid in raw_ids.split(",") if cid.strip()]
        return token, chat_ids

    @staticmethod
    def _cfg(config_loader, path: str, default):
        """Lee del config con respaldo, tolerando que sea None."""
        if config_loader is None:
            return default
        try:
            value = config_loader.get(path, default=default)
        except Exception:
            return default
        return value if value is not None else default

    @classmethod
    def diagnose(cls, config_loader=None) -> dict:
        """
        Estado de la configuración de Telegram, sin enviar nada.

        Distingue los casos que `from_config()` colapsa en un mismo
        None:

            sin token          nunca se configuró
            token sin chat     falta a quién enviar
            token con formato
            sospechoso         probablemente copiado a medias

        Existe porque el fallo original era invisible: el pipeline
        imprimía su resumen sin mencionar que las notificaciones no se
        habían intentado siquiera.

        El token se devuelve siempre redactado.
        """
        token, chat_ids = cls._read_credentials(config_loader)

        problemas: list[str] = []

        if not token:
            problemas.append(
                "Falta TELEGRAM_BOT_TOKEN. Se obtiene hablando con "
                "@BotFather en Telegram: /newbot y copiar el token."
            )
        elif ":" not in token or len(token) < 40:
            problemas.append(
                f"TELEGRAM_BOT_TOKEN tiene formato sospechoso "
                f"({len(token)} caracteres). Un token válido es como "
                f"'123456789:AAH...' — unos 46 caracteres con dos "
                f"puntos separando el id numérico del secreto."
            )

        if not chat_ids:
            problemas.append(
                "Falta TELEGRAM_CHAT_IDS. Para obtenerlo: escribir al "
                "bot desde el chat destino y abrir "
                "https://api.telegram.org/bot<TOKEN>/getUpdates — el "
                "campo chat.id es el valor. Los grupos llevan signo "
                "negativo."
            )

        return {
            "configured":     bool(token and chat_ids and not problemas),
            "has_token":      bool(token),
            "token_redacted": (token[:8] + "****") if len(token) > 8 else "",
            "chat_ids":       chat_ids,
            "n_chats":        len(chat_ids),
            "requests_available": _REQUESTS_AVAILABLE,
            "problems":       problemas,
        }

    def verify_connection(self) -> tuple[bool, str]:
        """
        Comprueba credenciales contra la API, sin enviar mensajes.

        Llama a getMe, que devuelve los datos del bot si el token es
        válido. Es el paso previo a cualquier envío: distingue un token
        incorrecto de un chat_id incorrecto, que producen el mismo
        síntoma —nada llega— por causas distintas.

        Retorna (ok, descripción).
        """
        if not _REQUESTS_AVAILABLE:
            return False, "requests no está instalado"

        url = _TELEGRAM_API_BASE.format(
            token=self._config.bot_token, method="getMe"
        )
        try:
            response = _requests.get(url, timeout=self._config.timeout_seconds)
        except Exception as e:
            return False, f"Sin conexión con api.telegram.org: {type(e).__name__}"

        if response.status_code == 401:
            return False, ("Token rechazado (401). Verificar "
                           "TELEGRAM_BOT_TOKEN con @BotFather.")
        if response.status_code != 200:
            return False, f"La API respondió {response.status_code}"

        try:
            data = response.json()
        except Exception:
            return False, "Respuesta de la API ilegible"

        if not data.get("ok"):
            return False, f"API: {data.get('description', 'error desconocido')}"

        bot = data.get("result", {})
        return True, f"@{bot.get('username', '?')} ({bot.get('first_name', '')})"