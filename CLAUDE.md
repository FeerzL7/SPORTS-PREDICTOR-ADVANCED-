# SPORTS-PREDICTOR-ADVANCED

Sistema de predicción deportiva y paper trading de apuestas (MLB, NFL, NBA,
fútbol). Corre un pipeline diario, registra los picks en ledgers y mide el
CLV contra la cuota de cierre.

## Stack

- Python 3.12, sin framework. Es la versión validada: con versiones más
  nuevas nfl_data_py no instala (ver `scripts/check_environment.py`).
- Dependencias: `requirements.txt` (sistema) y `requirements-dev.txt`
  (+ pytest). nfl_data_py se instala aparte con `--no-deps`; las
  instrucciones están en `requirements.txt`.
- Cuotas de The Odds API (plan gratuito: 500 créditos al mes). Avisos por
  Telegram.

## Comandos

- Tests: `python -m pytest -q` (desde la raíz)
- Diagnóstico del entorno: `python scripts/check_environment.py`
- Pipeline de un deporte: `python scripts/run_daily.py --sport mlb`
  - `--dry-run` no toca el ledger; `--date AAAA-MM-DD`; `--no-notify`
- Ciclo de paper trading: `python scripts/daily_cycle.py picks` y
  `python scripts/daily_cycle.py closing` (`closing --plan` decide sin
  ejecutar)
- `main.py` está vacío: el punto de entrada real es `scripts/run_daily.py`.

## Arquitectura

- `core/`: agnóstico al deporte. Contratos (`core/contracts/`), pipeline
  (`core/pipeline/`), cuotas, valor, riesgo, bankroll, tracking,
  backtesting y notificaciones.
- `sports/<deporte>/`: un plugin por deporte. `plugin.py` implementa
  `SportPlugin` de `core/pipeline/stage.py`.
- Dependencia en un solo sentido: `plugins → stage.py → contracts → core`.
  El core nunca importa de `sports/` ni conoce conceptos deportivos
  (pitcher, xG, park factor). Romper esto es una violación de arquitectura.
- Configuración: `config/base.yaml`, sobreescrito por `config/<deporte>.yaml`.
  Rutas con la forma `sección.subsección.parámetro`.
- `scripts/`: puntos de entrada, backtests, auditorías de fuentes y captura
  de cierres.
- `output/` (ledgers, cachés, archivo de precios) y `cache/` (datos
  descargados) los generan los scripts: no editarlos a mano.

## Reglas que no se rompen

- Secretos (ODDS_API_KEY, Telegram) solo en `.env` o `config/secrets.yaml`,
  ambos fuera de git. Nunca en `config/*.yaml` ni en los logs.
- Toda llamada que pide cuotas gasta créditos. Antes de agregar una, revisar
  la disciplina de créditos de `scripts/daily_cycle.py` (`/events` y
  `/sports` son gratis).
- Los mercados sin evidencia van a SOLO-PAPEL: se registran para medir,
  nunca se operan ni se notifican (`tests/core/test_paper_only.py`).
- NBA es solo de captura: su modelo dio no-go.
- Las dependencias de un deporte se importan de forma opcional dentro del
  plugin (`try/except ImportError`), para que un paquete faltante no rompa
  los demás deportes.
- Para probar el pipeline sin tocar el ledger, usar `--dry-run`.

## Tests

- pytest en `tests/core/` y `tests/sports/`, archivos `test_*.py`.
- No hay paquete instalable: los tests agregan la raíz al `sys.path`.
- CI: `.github/workflows/tests.yml` corre la suite en Windows con
  Python 3.12, sin nfl_data_py.

## Convenciones

- Todo en español: comentarios, docstrings y commits.
- Commits estilo Conventional Commits con el área como scope:
  `feat(soccer): ...`, `fix(ciclo): ...`, `chore: ...`.
- Ramas en kebab-case (`nba-precio-manana`, `correcciones-ciclo`); se
  integran a `main` por PR.
- Los docstrings de módulo explican el porqué: objetivo, decisiones y
  el motivo de cada fix.

## Reglas de ECC

`.claude/rules/ecc/common/` y `.claude/rules/ecc/python/` son las reglas
genéricas del plugin ECC. Si alguna choca con este archivo (idioma, formato
de commits, `print` como salida normal de los scripts de consola), manda
este archivo.
