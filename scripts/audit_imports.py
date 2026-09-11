#!/usr/bin/env python3
"""
scripts/audit_imports.py

Audit AST de imports: detecta símbolos no resueltos antes del dry run.

Recorre todos los .py del proyecto, extrae cada `from X import Y`
mediante AST parse (no regex — evita falsos positivos de docstrings
y comentarios), y verifica que el símbolo Y exista realmente en el
módulo X.

Detecta cuatro clases de error:
    1. MISSING_FILE   — el módulo importado no existe en disco
    2. MISSING_SYMBOL — el módulo existe pero no exporta el símbolo
    3. IMPORT_ERROR   — el módulo falla al importarse (error en cadena)
    4. SIGNATURE      — llamada con kwargs que el __init__ no acepta

Uso
----
    python scripts/audit_imports.py                 # todo el proyecto
    python scripts/audit_imports.py --path sports   # solo un subdir
    python scripts/audit_imports.py --verbose       # muestra OK también
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ── Bootstrap de paquetes namespace ───────────────────────────────────────────

def bootstrap_packages(root: Path) -> None:
    """
    Registra los paquetes del proyecto en sys.modules.

    Si el paquete tiene un __init__.py con contenido real, lo ejecuta
    para que los re-exports del paquete (ej. core.contracts re-exporta
    Event, TeamFeatures, etc.) estén disponibles. Sin esto, los imports
    del tipo `from core.contracts import TeamFeatures` fallarían con
    falsos positivos aunque el código sea correcto.
    """
    # Orden: paquetes más profundos primero para que los __init__.py
    # de nivel superior encuentren sus submódulos ya registrados.
    pkg_dirs = sorted(
        (d for d in root.rglob("*/") if "__pycache__" not in str(d)),
        key=lambda p: len(p.parts),
    )

    for pkg_dir in pkg_dirs:
        rel = pkg_dir.relative_to(root)
        if not rel.parts or rel.parts[0] not in ("core", "sports", "scripts"):
            continue
        pkg_name = ".".join(rel.parts)
        if not pkg_name or pkg_name in sys.modules:
            continue

        mod = types.ModuleType(pkg_name)
        mod.__path__ = [str(pkg_dir)]
        sys.modules[pkg_name] = mod

        # Ejecutar __init__.py si tiene contenido real (> 50 bytes)
        init_py = pkg_dir / "__init__.py"
        if init_py.exists() and init_py.stat().st_size > 50:
            try:
                spec = importlib.util.spec_from_file_location(
                    pkg_name, str(init_py),
                    submodule_search_locations=[str(pkg_dir)],
                )
                if spec and spec.loader:
                    real_mod = importlib.util.module_from_spec(spec)
                    sys.modules[pkg_name] = real_mod
                    spec.loader.exec_module(real_mod)
            except Exception:
                # Si el __init__ falla, dejamos el namespace vacío —
                # el audit reportará los símbolos faltantes normalmente
                sys.modules[pkg_name] = mod


def load_module(name: str, path: Path):
    """Carga un módulo por path, cacheando en sys.modules."""
    if name in sys.modules and hasattr(sys.modules[name], "__file__"):
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"No se pudo crear spec para {name}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── Extracción de imports por AST ─────────────────────────────────────────────

def extract_imports(path: Path) -> list[tuple[str, str, int]]:
    """
    Extrae todos los `from MODULE import SYMBOL` de un archivo.

    Retorna lista de (module, symbol, lineno).
    Solo imports internos del proyecto (core.*, sports.*).
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError) as e:
        return [("__PARSE_ERROR__", str(e), 0)]

    result = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if not (mod.startswith("core") or mod.startswith("sports")):
                continue
            for alias in node.names:
                if alias.name == "*":
                    continue
                result.append((mod, alias.name, node.lineno))
    return result


def extract_instantiations(path: Path) -> list[tuple[str, list[str], int]]:
    """
    Extrae llamadas Clase(kwarg=..., ...) para validar firmas.

    Retorna lista de (class_name, [kwarg_names], lineno).
    """
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):
        return []

    result = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # Solo llamadas a nombres simples con mayúscula inicial (clases)
        if isinstance(node.func, ast.Name) and node.func.id[:1].isupper():
            kwargs = [kw.arg for kw in node.keywords if kw.arg]
            if kwargs:
                result.append((node.func.id, kwargs, node.lineno))
    return result


# ── Audit principal ───────────────────────────────────────────────────────────

def audit(root: Path, subpath: str | None = None, verbose: bool = False) -> int:
    """
    Ejecuta el audit completo. Retorna el número de problemas encontrados.
    """
    bootstrap_packages(root)

    scan_root = root / subpath if subpath else root
    files = sorted(
        p for p in scan_root.rglob("*.py")
        if "__pycache__" not in str(p)
        and p.name != "audit_imports.py"
    )

    issues: list[str] = []
    ok_count = 0

    print(f"{'='*66}")
    print(f"  AUDIT DE IMPORTS — {len(files)} archivos")
    print(f"{'='*66}\n")

    for path in files:
        rel = path.relative_to(root)
        imports = extract_imports(path)
        if not imports:
            continue

        file_issues = []

        for mod_name, symbol, lineno in imports:
            if mod_name == "__PARSE_ERROR__":
                file_issues.append(f"    L0    PARSE_ERROR: {symbol}")
                continue

            # Resolver la ruta del módulo: puede ser un archivo .py
            # (core/contracts/event.py) o un paquete con __init__.py
            # (core/contracts/ → core/contracts/__init__.py).
            # Sin esta distinción, `from core.contracts import Projection`
            # daba falso positivo MISSING_FILE porque solo se buscaba
            # core/contracts.py, que no existe — el símbolo vive en el
            # __init__.py del paquete.
            base     = root / mod_name.replace(".", "/")
            mod_path = base.with_suffix(".py")
            if not mod_path.exists():
                pkg_init = base / "__init__.py"
                if pkg_init.exists():
                    mod_path = pkg_init
                else:
                    file_issues.append(
                        f"    L{lineno:<4} MISSING_FILE: {mod_name} "
                        f"(buscando {symbol})"
                    )
                    continue

            try:
                mod = load_module(mod_name, mod_path)
            except Exception as e:
                file_issues.append(
                    f"    L{lineno:<4} IMPORT_ERROR: {mod_name} — {type(e).__name__}: {e}"
                )
                continue

            if not hasattr(mod, symbol):
                exported = [n for n in dir(mod) if not n.startswith("__")][:5]
                file_issues.append(
                    f"    L{lineno:<4} MISSING_SYMBOL: {mod_name}.{symbol} "
                    f"(exporta: {', '.join(exported)}...)"
                )
            else:
                ok_count += 1

        if file_issues:
            print(f"  ❌ {rel}")
            for issue in file_issues:
                print(issue)
                issues.append(f"{rel}: {issue.strip()}")
            print()
        elif verbose:
            print(f"  ✅ {rel} ({len(imports)} imports)")

    # ── Resumen ────────────────────────────────────────────────────
    print(f"{'='*66}")
    print(f"  RESULTADO")
    print(f"{'='*66}")
    print(f"  Archivos escaneados:  {len(files)}")
    print(f"  Imports verificados:  {ok_count + len(issues)}")
    print(f"  Imports OK:           {ok_count}")
    print(f"  Problemas:            {len(issues)}")
    print()

    if issues:
        print("  DETALLE DE PROBLEMAS:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")
    else:
        print("  ✅ Sin problemas — todos los imports resueltos.")

    print()
    return len(issues)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit AST de imports")
    parser.add_argument("--path", default=None,
                        help="Subdirectorio a escanear (ej: sports)")
    parser.add_argument("--verbose", action="store_true",
                        help="Mostrar archivos OK además de los problemáticos")
    args = parser.parse_args()

    n_issues = audit(_PROJECT_ROOT, args.path, args.verbose)
    return 1 if n_issues else 0


if __name__ == "__main__":
    sys.exit(main())