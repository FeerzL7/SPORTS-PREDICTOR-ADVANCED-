#!/usr/bin/env python3
"""
scripts/test_telegram.py

Verifica el circuito completo de notificaciones, paso a paso.

Por qué existe
---------------
Las notificaciones fallaban en silencio. `TelegramNotifier.from_config()`
leía las credenciales del ConfigLoader, que solo consulta archivos YAML,
mientras que el token y los chat_ids viven en .env. Devolvía None, el
pipeline seguía su curso, y el informe no mencionaba que ni siquiera se
había intentado enviar nada.

Un canal de notificación que falla callado es peor que no tenerlo: uno
cree estar recibiendo picks y no los recibe.

Este script recorre la cadena y se detiene en el primer eslabón roto:

    1. ¿Están las credenciales?
    2. ¿Tienen formato válido?
    3. ¿Acepta la API el token?
    4. ¿Llega un mensaje de prueba a cada chat?
    5. ¿Se formatea bien un pick real?

Uso
----
    python scripts/test_telegram.py              # diagnóstico + envío
    python scripts/test_telegram.py --check      # sin enviar nada
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


def _paso(n: int, titulo: str) -> None:
    print()
    print(f"  [{n}] {titulo}")
    print("  " + "─" * 60)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verifica el circuito de notificaciones Telegram"
    )
    parser.add_argument("--check", action="store_true",
                        help="Solo diagnosticar, sin enviar mensajes")
    args = parser.parse_args()

    print("=" * 66)
    print("  VERIFICACIÓN DE TELEGRAM")
    print("=" * 66)

    from core.notifications.telegram import TelegramNotifier

    # ── 1. Credenciales ────────────────────────────────────────────
    _paso(1, "¿Están las credenciales?")

    estado = TelegramNotifier.diagnose()

    print(f"  Token      : {'sí' if estado['has_token'] else 'NO'}"
          f"{'  ' + estado['token_redacted'] if estado['has_token'] else ''}")
    print(f"  Chats      : {estado['n_chats']}")
    for cid in estado["chat_ids"]:
        print(f"               {cid}")
    print(f"  requests   : {'sí' if estado['requests_available'] else 'NO'}")

    if estado["problems"]:
        print()
        for p in estado["problems"]:
            print(f"  ❌ {p}")
        print()
        print("  Definir en el archivo .env de la raíz del proyecto:")
        print()
        print("      TELEGRAM_BOT_TOKEN=123456789:AAH...")
        print("      TELEGRAM_CHAT_IDS=-1001234567890")
        print()
        return 1

    print()
    print("  ✅ Credenciales presentes y con formato plausible")

    # ── 2. El notificador se construye ─────────────────────────────
    _paso(2, "¿Se construye el notificador?")

    notifier = TelegramNotifier.from_config()
    if notifier is None:
        print("  ❌ from_config() devolvió None pese al diagnóstico correcto.")
        print("     Revisar TelegramConfig.__post_init__.")
        return 1
    print("  ✅ TelegramNotifier construido")

    # ── 3. La API acepta el token ──────────────────────────────────
    _paso(3, "¿Acepta la API el token?")

    ok, detalle = notifier.verify_connection()
    if not ok:
        print(f"  ❌ {detalle}")
        return 1
    print(f"  ✅ Bot verificado: {detalle}")

    if args.check:
        print()
        print("=" * 66)
        print("  CREDENCIALES CORRECTAS")
        print("=" * 66)
        print()
        print("  Ejecutar sin --check para enviar un mensaje de prueba.")
        print()
        return 0

    # ── 4. Mensaje de prueba ───────────────────────────────────────
    _paso(4, "¿Llega un mensaje de prueba?")

    from datetime import datetime, timezone
    ahora = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    mensaje = (
        "🔧 <b>Prueba de notificaciones</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        f"Sports Predictor Advanced\n"
        f"{ahora}\n\n"
        "Si lees esto, el canal funciona."
    )

    enviado = notifier.send_message(mensaje)
    if not enviado:
        print("  ❌ El envío falló pese a que el token es válido.")
        print()
        print("     Causa habitual: el chat_id es incorrecto, o el bot")
        print("     nunca ha sido invitado a ese chat. Un bot no puede")
        print("     escribir a un chat donde no esté.")
        print()
        print("     Comprobar:")
        print("       • En un chat personal: escribirle al bot primero")
        print("       • En un grupo: añadir el bot como miembro")
        print("       • En un canal: añadirlo como administrador")
        return 1

    print(f"  ✅ Mensaje enviado a {estado['n_chats']} chat(s)")
    print("     Revisar Telegram para confirmar que llegó.")

    # ── 5. Formato de un pick real ─────────────────────────────────
    _paso(5, "¿Se formatea bien un pick?")

    from core.contracts.event import Event, EventStatus
    from core.contracts.pick import CandidatePick
    from core.notifications.telegram import format_pick

    evento = Event(
        event_id="prueba", sport="mlb", league="MLB",
        season_start=2026, season_end=2026,
        date="2026-09-27", start_time="2026-09-27T23:05:00Z",
        home_team_id="1", away_team_id="2",
        home_team="Los Angeles Dodgers", away_team="San Diego Padres",
        venue_id="v", venue_name="Dodger Stadium",
        status=EventStatus.SCHEDULED, provider_ids={},
    )

    pick = CandidatePick(
        event=evento, market="TOTAL", selection="over", line=7.5,
        price=2.05, model_prob_raw=0.58, market_prob=0.4878,
        blended_prob=0.5165, model_prob_capped=0.58,
        kelly_fraction=0.021, stake_pct=1, active=True,
    )

    texto = format_pick(pick)
    print("  Vista previa:")
    print()
    for linea in texto.splitlines():
        print(f"      {linea}")
    print()

    enviado = notifier.send_message(
        "📋 <b>Ejemplo de pick</b> (datos de prueba)\n\n" + texto
    )
    print(f"  {'✅' if enviado else '⚠️ '} Pick de ejemplo "
          f"{'enviado' if enviado else 'no enviado'}")

    print()
    print("=" * 66)
    print("  CIRCUITO COMPLETO VERIFICADO")
    print("=" * 66)
    print()
    print("  El pipeline enviará picks automáticamente:")
    print()
    print("      python scripts/run_daily.py --sport mlb")
    print()
    print("  Con --dry-run NO se envía nada, por diseño: un ensayo no")
    print("  debe generar notificaciones que parezcan picks reales.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())