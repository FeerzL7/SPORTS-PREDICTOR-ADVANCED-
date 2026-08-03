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
    def from_config(cls, config_loader) -> "TelegramNotifier | None":
        """
        Construye un TelegramNotifier desde un ConfigLoader.

        Lee TELEGRAM_BOT_TOKEN y TELEGRAM_CHAT_IDS del ConfigLoader.
        Retorna None si las credenciales no están configuradas —
        el pipeline puede continuar sin notificaciones.

        TELEGRAM_CHAT_IDS puede ser un string con IDs separados por
        coma: '-123456789,-987654321,@mi_canal'
        """
        token = config_loader.get("TELEGRAM_BOT_TOKEN", default=None)
        if not token:
            return None

        raw_ids = config_loader.get("TELEGRAM_CHAT_IDS", default="")
        if not raw_ids:
            return None

        chat_ids = [cid.strip() for cid in str(raw_ids).split(",") if cid.strip()]
        if not chat_ids:
            return None

        try:
            cfg = TelegramConfig(
                bot_token = token,
                chat_ids  = chat_ids,
                parse_mode = config_loader.get(
                    "telegram.parse_mode", default="HTML"
                ),
                rate_limit_delay = float(config_loader.get(
                    "telegram.rate_limit_delay", default=0.1
                )),
            )
            return cls(config=cfg)
        except ValueError:
            return None
         