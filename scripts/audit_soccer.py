#!/usr/bin/env python3
"""
scripts/audit_soccer.py

Auditoría estática del plugin de fútbol.

Qué verifica y por qué
------------------------
No es un test de comportamiento —de eso se encargan los dry runs y la
suite de pytest— sino de INVARIANTES ESTRUCTURALES: propiedades que
deben cumplirse en todo el plugin y cuyo incumplimiento produce fallos
silenciosos.

Cada invariante está aquí porque su violación causó un bug real en este
proyecto o en el plugin de NFL:

    SIN PANDAS
        El plugin NFL arrastró boolean masking de pandas por varios
        módulos. El type checker no lo modela, así que
        `df[df["col"] == x]` pasa la revisión y falla en ejecución.
        Además, instalar `soccerdata` —que exige pandas>=2.0— rompió el
        plugin NFL, que exige pandas<2.0.

    SIN IMPORTS PRIVADOS ENTRE MÓDULOS
        `from otro_modulo import _CONSTANTE` acopla a los internos de
        otro módulo. Si alguien recalibra ese valor, cambia el
        comportamiento de un consumidor que no debería depender de él.
        Ocurrió con los defaults de dixon_coles en projections.

    DEPENDENCIA sports → core, NUNCA AL REVÉS
        El Core no puede importar de un plugin. Si lo hiciera, añadir
        un deporte obligaría a tocar el Core.

    BARRERA TEMPORAL DECLARADA
        Los fetchers que consultan historial deben exigir la fecha de
        corte sin valor por defecto. Un parámetro opcional invita a
        omitirlo, y el modo de fallo es un backtest inflado que no se
        reproduce en producción.

    PROTOCOLS EN LAS DEPENDENCIAS INYECTADAS
        Declarar la interfaz mínima que cada módulo consume documenta
        la dependencia real y evita que el type checker propague
        `Unknown`.

Uso
----
    python scripts/audit_soccer.py
    python scripts/audit_soccer.py --verbose
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

_PLUGIN_DIR = _PROJECT_ROOT / "sports" / "soccer"

# Módulos que deben exigir fecha de corte sin default.
_TEMPORAL_BARRIERS: dict[str, list[tuple[str, str]]] = {
    "team_stats.py": [("fetch", "as_of_date"), ("fetch_all", "as_of_date")],
    "h2h.py":        [("get_h2h", "as_of_date")],
    "congestion.py": [("fetch", "match_date")],
}


@dataclass
class Finding:
    module: str
    line:   int
    kind:   str
    detail: str


@dataclass
class AuditReport:
    modules:  int = 0
    checks:   int = 0
    findings: list[Finding] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.findings

    def add(self, module: str, line: int, kind: str, detail: str) -> None:
        self.findings.append(Finding(module, line, kind, detail))


# ── Invariantes ──────────────────────────────────────────────────────────────

def check_no_pandas(tree: ast.Module, module: str, report: AuditReport) -> None:
    """
    Ningún módulo importa pandas, ni a nivel de módulo ni diferido.

    El plugin se diseñó para leer CSV plano con la stdlib y JSON de
    Understat con `re` y `json`. Introducir pandas reabriría el
    conflicto de versiones con nfl_data_py.
    """
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]

        for name in names:
            if name.split(".")[0] in ("pandas", "numpy", "soccerdata"):
                report.add(module, node.lineno, "PANDAS",
                           f"importa '{name}'")


def check_no_boolean_masking(tree: ast.Module, module: str,
                             report: AuditReport) -> None:
    """
    Ningún subíndice con máscara booleana.

    El patrón de pandas es `df[df["c"] == x]` —slice de tipo Compare—
    o `df[(a == 1) | (b == 2)]` —BinOp CON Compare dentro.

    La indexación aritmética de listas Python (`p1[h - k]`) también usa
    BinOp y NO es masking: solo se marca el BinOp que contiene un
    Compare. Esa distinción hizo falta cuando el check original dio un
    falso positivo sobre el Poisson bivariado.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        s = node.slice
        if isinstance(s, ast.Compare):
            report.add(module, node.lineno, "MASKING", ast.unparse(node)[:60])
        elif isinstance(s, ast.BinOp):
            if any(isinstance(x, ast.Compare) for x in ast.walk(s)):
                report.add(module, node.lineno, "MASKING",
                           ast.unparse(node)[:60])


def check_no_private_imports(tree: ast.Module, module: str,
                             report: AuditReport) -> None:
    """
    Ningún import de símbolos privados de otro módulo del proyecto.

    Un `from x import _CONST` acopla a los internos de x. Ocurrió en
    projections.py, que importaba los defaults de dixon_coles: si
    alguien recalibraba ese valor, cambiaba el comportamiento del
    modelo sin tocar soccer.yaml.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if not node.module.startswith(("sports.", "core.")):
            continue
        for alias in node.names:
            if alias.name.startswith("_"):
                report.add(module, node.lineno, "PRIVADO",
                           f"{node.module}.{alias.name}")


def check_dependency_direction(tree: ast.Module, module: str,
                               report: AuditReport) -> None:
    """
    Un módulo de fútbol no importa de otro plugin deportivo.

    Compartir código entre plugins significa que pertenece al Core. Un
    import cruzado haría que cambiar MLB rompiera fútbol.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or not node.module:
            continue
        if node.module.startswith(("sports.mlb", "sports.nfl", "sports.nba")):
            report.add(module, node.lineno, "CRUZADO", node.module)


def check_temporal_barriers(tree: ast.Module, module: str,
                            report: AuditReport) -> None:
    """
    Los métodos que consultan historial exigen la fecha de corte.

    Sin valor por defecto: omitirla debe ser un error de programación
    visible, no un fallo silencioso que contamina el backtest con datos
    del futuro.
    """
    expected = _TEMPORAL_BARRIERS.get(module)
    if not expected:
        return

    found: dict[str, ast.FunctionDef] = {
        n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
    }

    for method, param in expected:
        fn = found.get(method)
        if fn is None:
            report.add(module, 0, "BARRERA", f"falta el método {method}()")
            continue

        args = fn.args.args
        defaults = fn.args.defaults
        # Los defaults se alinean por la derecha
        con_default = {a.arg for a in args[len(args) - len(defaults):]}

        if param not in {a.arg for a in args}:
            report.add(module, fn.lineno, "BARRERA",
                       f"{method}() sin parámetro '{param}'")
        elif param in con_default:
            report.add(module, fn.lineno, "BARRERA",
                       f"{method}() tiene '{param}' con valor por defecto")


def check_protocols(tree: ast.Module, module: str,
                    report: AuditReport) -> None:
    """
    Los módulos con dependencias inyectadas declaran su Protocol.

    Solo informativo: se cuenta, no se exige. Un módulo puede recibir
    una dependencia ya tipada por otra vía.
    """
    return


# ── Ejecución ────────────────────────────────────────────────────────────────

_CHECKS = (
    check_no_pandas,
    check_no_boolean_masking,
    check_no_private_imports,
    check_dependency_direction,
    check_temporal_barriers,
)


def audit(verbose: bool = False) -> AuditReport:
    report = AuditReport()

    if not _PLUGIN_DIR.exists():
        print(f"ERROR: no existe {_PLUGIN_DIR}")
        return report

    for path in sorted(_PLUGIN_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue

        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError as e:
            report.add(path.name, e.lineno or 0, "SINTAXIS", str(e))
            continue

        report.modules += 1
        for check in _CHECKS:
            check(tree, path.name, report)
            report.checks += 1

        if verbose:
            print(f"  {path.name:20s} {len(path.read_text(encoding='utf-8').splitlines()):4d} líneas")

    return report


def _protocol_count() -> dict[str, list[str]]:
    """Protocols declarados por módulo, para el informe."""
    result: dict[str, list[str]] = {}
    for path in sorted(_PLUGIN_DIR.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        names = [
            n.name for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef)
            and any("Protocol" in ast.unparse(b) for b in n.bases)
        ]
        if names:
            result[path.name] = names
    return result


def _external_deps() -> set[str]:
    """Dependencias externas del plugin."""
    stdlib = {
        "__future__", "csv", "io", "json", "math", "re", "time",
        "dataclasses", "datetime", "pathlib", "typing", "unicodedata",
        "collections", "itertools", "functools",
    }
    deps: set[str] = set()
    for path in _PLUGIN_DIR.glob("*.py"):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                deps |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                deps.add(node.module.split(".")[0])
    return deps - stdlib - {"sports", "core"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auditoría estática del plugin de fútbol"
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("=" * 66)
    print("  AUDITORÍA DEL PLUGIN DE FÚTBOL")
    print("=" * 66)
    print()

    if args.verbose:
        print("  Módulos:")

    report = audit(verbose=args.verbose)

    if args.verbose:
        print()

    print(f"  Módulos auditados : {report.modules}")
    print(f"  Comprobaciones    : {report.checks}")
    print(f"  Hallazgos         : {len(report.findings)}")
    print()

    if report.findings:
        print("  HALLAZGOS:")
        for f in report.findings:
            print(f"    {f.module}:L{f.line}  [{f.kind}] {f.detail}")
        print()
    else:
        print("  ✅ Todos los invariantes se cumplen.")
        print()

    # ── Contexto ───────────────────────────────────────────────────
    deps = _external_deps()
    print(f"  Dependencias externas: {sorted(deps) or '(ninguna)'}")
    if deps == {"requests"}:
        print("    requests ya es requisito del Core para The Odds API,")
        print("    así que el plugin no añade ninguna dependencia nueva.")
    print()

    protocols = _protocol_count()
    total = sum(len(v) for v in protocols.values())
    print(f"  Protocols declarados: {total} en {len(protocols)} módulos")
    for module, names in protocols.items():
        print(f"    {module:20s} {', '.join(names)}")
    print()

    print(f"  Barreras temporales verificadas:")
    for module, methods in _TEMPORAL_BARRIERS.items():
        for method, param in methods:
            print(f"    {module:20s} {method}({param}=...) sin default")
    print()

    print("=" * 66)
    print(f"  RESULTADO: {'✅ LIMPIO' if report.ok else '❌ ' + str(len(report.findings)) + ' HALLAZGOS'}")
    print("=" * 66)

    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())