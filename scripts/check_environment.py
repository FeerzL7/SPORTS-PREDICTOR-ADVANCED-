#!/usr/bin/env python3
"""
scripts/check_environment.py

Verifica que el entorno está completo, plugin por plugin.

Para qué
---------
Al migrar de máquina o reinstalar Python, los fallos no aparecen todos
a la vez: un paquete que falta se manifiesta al ejecutar el deporte que
lo usa, semanas después.

Este script recorre las tres capas —intérprete, paquetes, plugins— y
dice qué funciona, qué falta y con qué comando se arregla.

Cada comprobación es independiente: que MLB falle no impide saber si
fútbol está bien.

Uso
----
    python scripts/check_environment.py
    python scripts/check_environment.py --verbose
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_OK   = "✅"
_FAIL = "❌"
_WARN = "⚠️ "


def _section(titulo: str) -> None:
    print()
    print(f"  {titulo}")
    print("  " + "─" * 62)


def check_python() -> list[str]:
    """
    Versión del intérprete.

    3.12 es la configuración validada del proyecto. Versiones
    superiores rompen nfl_data_py, cuyo pin de pandas<2.0 no tiene
    binarios precompilados para Python nuevo.
    """
    problemas: list[str] = []
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"

    _section("INTÉRPRETE")
    if (v.major, v.minor) == (3, 12):
        print(f"  {_OK} Python {version}  (versión validada del proyecto)")
    elif (v.major, v.minor) > (3, 12):
        print(f"  {_WARN} Python {version}")
        print(f"       nfl_data_py exige pandas<2.0, que no tiene binarios")
        print(f"       precompilados para esta versión. Fútbol y MLB no se")
        print(f"       ven afectados.")
        problemas.append("python_nuevo")
    else:
        print(f"  {_WARN} Python {version}  (el proyecto usa 3.12)")
        problemas.append("python_viejo")

    return problemas


def check_packages(verbose: bool) -> list[str]:
    """
    Paquetes externos, agrupados por para qué sirven.

    Se separan los del núcleo —sin ellos no arranca nada— de los que
    solo afectan a un deporte.
    """
    faltan: list[str] = []

    grupos = [
        ("Núcleo — sin esto no arranca nada", True, [
            ("requests", "HTTP para todas las fuentes"),
            ("yaml",     "carga de config/*.yaml"),
        ]),
        ("Configuración y tests", False, [
            ("dotenv", "lee ODDS_API_KEY del .env"),
            ("pytest", "suite de tests"),
        ]),
        ("Solo NFL", False, [
            ("nfl_data_py", "datos de nflverse"),
            ("pandas",      "lo arrastra nfl_data_py"),
        ]),
    ]

    for titulo, critico, paquetes in grupos:
        _section(f"PAQUETES — {titulo}")
        for modulo, para_que in paquetes:
            try:
                mod = importlib.import_module(modulo)
                version = getattr(mod, "__version__", "")
                extra = f"  v{version}" if version else ""
                print(f"  {_OK} {modulo:14s}{extra:12s} {para_que}")

                # pandas: el pin de nfl_data_py resultó conservador.
                #
                # La librería declara pandas<2.0 y numpy<2.0, pero esa
                # combinación no se puede instalar en Python 3.12 —
                # pandas 1.5.3 no tiene binarios precompilados y falla
                # al compilar sin Visual Studio.
                #
                # Verificado con pandas 2.3.3 y numpy 2.5.3: las cuatro
                # funciones que el plugin usa responden bien, incluido
                # import_pbp_data con 49.492 jugadas, que es de donde
                # sale el EPA.
                #
                # Solo se avisa de pandas 3.x, donde sí hay cambios de
                # API sin verificar.
                if modulo == "pandas" and version:
                    mayor = version.split(".")[0]
                    if mayor >= "3":
                        print(f"       {_WARN} pandas {version} sin verificar "
                              f"con nfl_data_py")
                        print(f"          Probado hasta 2.x. Comprobar con:")
                        print(f"          python -c \"import nfl_data_py as n; "
                              f"print(len(n.import_schedules([2024])))\"")
                        faltan.append("pandas_version")
            except ImportError:
                marca = _FAIL if critico else _WARN
                print(f"  {marca} {modulo:14s}{'':12s} {para_que}  — NO INSTALADO")
                faltan.append(modulo)

    return faltan


def check_plugins(verbose: bool) -> list[str]:
    """
    Cada plugin, por separado.

    Se comprueban dos cosas distintas: que el módulo IMPORTE —lo que
    revela dependencias rotas dentro del propio proyecto— y que su
    is_available() diga que puede operar.

    La diferencia importa: un plugin puede importar y no estar
    disponible (le falta una librería externa), o fallar al importar
    por un problema del código, que es más grave.
    """
    problemas: list[str] = []

    plugins = [
        ("soccer", "sports.soccer.plugin", "SoccerPlugin"),
        ("nfl",    "sports.nfl.plugin",    "NFLPlugin"),
        ("mlb",    "sports.mlb.plugin",    "MLBPlugin"),
    ]

    _section("PLUGINS")
    for nombre, ruta, clase in plugins:
        try:
            mod = importlib.import_module(ruta)
        except Exception as e:
            print(f"  {_FAIL} {nombre:8s} NO IMPORTA")
            print(f"       {type(e).__name__}: {e}")
            problemas.append(f"{nombre}_import")
            continue

        cls = getattr(mod, clase, None)
        if cls is None:
            print(f"  {_FAIL} {nombre:8s} el módulo no expone {clase}")
            problemas.append(f"{nombre}_clase")
            continue

        # Un plugin sin is_available() no es lo mismo que uno que
        # falla: el primero es una interfaz incompleta —se arregla en
        # el código— y el segundo, datos que faltan.
        if not hasattr(cls, "is_available"):
            print(f"  {_WARN} {nombre:8s} importa, pero no declara is_available()")
            print(f"       Interfaz incompleta: los otros plugins sí lo tienen.")
            problemas.append(f"{nombre}_interfaz")
            continue

        try:
            disponible = cls.is_available()
        except Exception as e:
            print(f"  {_WARN} {nombre:8s} importa, pero is_available() falló")
            print(f"       {type(e).__name__}: {e}")
            problemas.append(f"{nombre}_disponible")
            continue

        marca = _OK if disponible else _WARN
        estado = "operativo" if disponible else "importa, pero sin datos"
        print(f"  {marca} {nombre:8s} {estado}")
        if not disponible:
            problemas.append(f"{nombre}_datos")

    return problemas


def check_modules() -> list[str]:
    """
    Módulos internos que otros consumen.

    Un import roto DENTRO del proyecto es distinto de un paquete que
    falta: no se arregla instalando nada, sino corrigiendo el código o
    restaurando un archivo.
    """
    problemas: list[str] = []

    internos = [
        ("core.odds.client",          "OddsAPIClient"),
        ("core.odds.normalizer",      "OddsNormalizer"),
        ("core.value.engine",         "ValueEngine"),
        ("core.bankroll.tracker",     "CsvLedgerStore"),
        ("core.tracking.roi_tracker", "ROITracker"),
        ("core.risk.manager",         "RiskManager"),
        ("core.evaluation.clv",       "CLVTracker"),
        ("core.pipeline.runner",      "PipelineRunner"),
        ("core.utils.h2h_base",       "compute_h2h"),
    ]

    _section("MÓDULOS INTERNOS DEL CORE")
    for ruta, simbolo in internos:
        try:
            mod = importlib.import_module(ruta)
        except Exception as e:
            print(f"  {_FAIL} {ruta:28s} {type(e).__name__}")
            problemas.append(ruta)
            continue

        if hasattr(mod, simbolo):
            print(f"  {_OK} {ruta:28s} {simbolo}")
        else:
            exporta = sorted(n for n in dir(mod)
                             if n[0].isupper() and not n.startswith("_"))[:4]
            print(f"  {_FAIL} {ruta:28s} falta {simbolo}")
            print(f"       exporta: {', '.join(exporta) or '(nada)'}")
            problemas.append(f"{ruta}.{simbolo}")

    return problemas


def check_config() -> list[str]:
    """Archivos de configuración y la clave de API."""
    problemas: list[str] = []

    _section("CONFIGURACIÓN")

    for nombre in ("base.yaml", "mlb.yaml", "nfl.yaml", "soccer.yaml"):
        ruta = _PROJECT_ROOT / "config" / nombre
        if ruta.exists():
            print(f"  {_OK} config/{nombre}")
        else:
            print(f"  {_FAIL} config/{nombre}  — NO EXISTE")
            problemas.append(nombre)

    env = _PROJECT_ROOT / ".env"
    if env.exists():
        try:
            contenido = env.read_text(encoding="utf-8")
            tiene = "ODDS_API_KEY" in contenido and len(
                contenido.split("ODDS_API_KEY=")[-1].strip()
            ) > 5
        except Exception:
            tiene = False
        if tiene:
            print(f"  {_OK} .env con ODDS_API_KEY")
        else:
            print(f"  {_WARN} .env existe pero ODDS_API_KEY parece vacía")
            problemas.append("odds_key")
    else:
        print(f"  {_WARN} .env  — NO EXISTE")
        print(f"       Necesario para pedir cuotas en vivo. Los backtests")
        print(f"       funcionan sin él: usan cuotas históricas.")
        problemas.append("env")

    return problemas


def _remedios(faltan: list[str], plugins: list[str],
              internos: list[str], config: list[str]) -> None:
    """Qué hacer con cada problema, en orden de prioridad."""
    print()
    print("=" * 66)
    print("  QUÉ HACER")
    print("=" * 66)

    hay_algo = False

    # ── Paquetes del núcleo ────────────────────────────────────
    nucleo = [p for p in faltan if p in ("requests", "yaml")]
    if nucleo:
        hay_algo = True
        print()
        print("  1. PAQUETES DEL NÚCLEO (bloquea todo)")
        print()
        print('     pip install "requests>=2.31,<3.0" "pyyaml>=6.0,<7.0"')

    opcionales = [p for p in faltan if p in ("dotenv", "pytest")]
    if opcionales:
        hay_algo = True
        nombres = {"dotenv": "python-dotenv", "pytest": "pytest"}
        print()
        print("  2. CONFIGURACIÓN Y TESTS")
        print()
        print("     pip install " + " ".join(nombres[p] for p in opcionales))

    # ── NFL ────────────────────────────────────────────────────
    if "nfl_data_py" in faltan:
        hay_algo = True
        print()
        print("  3. PLUGIN NFL")
        print()
        print("     pandas 1.5.3 no tiene binarios precompilados para")
        print("     Python 3.12, así que pip intenta compilarlo desde")
        print("     fuente y falla sin el compilador de Visual Studio.")
        print()
        print("     Opción rápida — saltarse el pin:")
        print()
        print('         pip install "pandas>=2.0,<3.0"')
        print('         pip install --no-deps "nfl-data-py>=0.3.0,<0.4.0"')
        print('         pip install appdirs fastparquet python-snappy')
        print()
        print("     Y comprobar que descarga de verdad:")
        print()
        print('         python -c "import nfl_data_py as n; '
              'print(len(n.import_schedules([2024])))"')
        print()
        print("     Debe imprimir ~285. Si falla, la vía limpia es migrar")
        print("     a nflreadpy, que solo afecta a sports/nfl/data_source.py")

    # ── Módulos internos ───────────────────────────────────────
    if internos:
        hay_algo = True
        print()
        print("  4. MÓDULOS INTERNOS ROTOS")
        print()
        print("     Esto NO se arregla instalando nada: falta un archivo")
        print("     del proyecto o hay un símbolo que cambió de nombre.")
        print()
        for p in internos:
            print(f"       {p}")
        print()
        print("     Restaurar esos archivos desde el repositorio.")

    interfaz = [p for p in plugins if p.endswith("_interfaz")]
    if interfaz:
        hay_algo = True
        print()
        print("  5. INTERFAZ INCOMPLETA")
        print()
        print("     Estos plugins importan bien pero no declaran")
        print("     is_available(). No falta ningún paquete: falta el")
        print("     método en el código.")
        print()
        for p in interfaz:
            print(f"       {p.replace('_interfaz', '')}")

    if config:
        hay_algo = True
        print()
        print("  6. CONFIGURACIÓN")
        if "env" in config or "odds_key" in config:
            print()
            print("     Crear .env en la raíz con:")
            print()
            print("         ODDS_API_KEY=tu_clave")

    # ── Plugins rotos ──────────────────────────────────────────
    #
    # Se trata aparte porque no se arregla instalando nada. Y el
    # verificador NO debe decir "nada que hacer" cuando un plugin
    # falla: la primera versión lo hacía, porque solo contemplaba las
    # categorías de problema que había previsto.
    rotos = [p for p in plugins
             if p.endswith(("_import", "_clase", "_disponible"))]
    if rotos:
        hay_algo = True
        print()
        print("  PLUGINS ROTOS")
        print()
        for p in rotos:
            nombre = p.rsplit("_", 1)[0]
            causa = p.rsplit("_", 1)[1]
            print(f"     {nombre}")
            if causa == "clase":
                print(f"       El módulo importa pero no contiene su clase.")
                print(f"       Causa habitual: el archivo en disco es de OTRO")
                print(f"       plugin. Los tres se llaman plugin.py, así que")
                print(f"       es fácil copiarlos a la carpeta equivocada.")
                print()
                print(f"       Comprobar con:")
                print(f'         python -c "import sports.{nombre}.plugin as p; '
                      f'print([n for n in dir(p) if \'Plugin\' in n])"')
            elif causa == "import":
                print(f"       El módulo no importa: falta un archivo del")
                print(f"       plugin o una de sus dependencias internas.")
            else:
                print(f"       is_available() lanzó una excepción.")
            print()

    if not hay_algo:
        print()
        print("  Nada que hacer: el entorno está completo.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verifica el entorno del proyecto"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 66)
    print("  VERIFICACIÓN DEL ENTORNO")
    print("=" * 66)
    print(f"  Proyecto: {_PROJECT_ROOT}")

    check_python()
    faltan = check_packages(args.verbose)
    internos = check_modules()
    plugins = check_plugins(args.verbose)
    config = check_config()

    # ── Resumen ────────────────────────────────────────────────
    print()
    print("=" * 66)
    print("  RESUMEN")
    print("=" * 66)
    print()

    nucleo_ok = not any(p in faltan for p in ("requests", "yaml"))
    soccer_ok = not any(p.startswith("soccer") for p in plugins)
    nfl_ok    = not any(p.startswith("nfl") for p in plugins)
    mlb_ok    = not any(p.startswith("mlb") for p in plugins)

    print(f"  {_OK if nucleo_ok else _FAIL} Núcleo")
    print(f"  {_OK if soccer_ok else _FAIL} Fútbol   (backtests, dry run, tests)")
    print(f"  {_OK if mlb_ok else _FAIL} MLB")
    print(f"  {_OK if nfl_ok else _FAIL} NFL")

    if soccer_ok and not nfl_ok:
        print()
        print("  El plugin de fútbol no depende de NFL: puedes seguir")
        print("  trabajando en él mientras resuelves pandas.")

    _remedios(faltan, plugins, internos, config)

    print()
    return 0 if (nucleo_ok and soccer_ok and mlb_ok and nfl_ok) else 1


if __name__ == "__main__":
    sys.exit(main())