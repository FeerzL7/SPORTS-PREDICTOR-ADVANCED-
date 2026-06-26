
#core/utils/config_loader.py


from __future__ import annotations

import os
from typing import Any

import yaml


# Directorio base donde se buscan base.yaml y {sport}.yaml.
DEFAULT_CONFIG_DIR = "config"


class ConfigLoader:
    """
    Configuración del sistema como resultado del merge base + sport.

    Inmutable tras la carga: el mismo objeto se reutiliza en toda la
    ejecución del pipeline sin releer disco (cache en memoria).
    """

    def __init__(self, data: dict[str, Any], sport: str | None) -> None:
        self._data = data
        self.sport = sport  # None si se cargó solo base.yaml

    # ── Acceso por dot-notation ────────────────────────────────────────────────

    def get(self, path: str, default: Any = None) -> Any:
        """
        Acceso a un valor anidado usando notación de puntos.

        Ejemplo:
            config.get("value.blending.ML.model_weight")
            config.get("filters.MLB.TOTAL.min_ev", default=15)

        Retorna default si cualquier segmento de la ruta no existe,
        en vez de lanzar KeyError — permite que el código consumidor
        especifique un fallback explícito y documentado sin necesidad
        de try/except dispersos por todo el codebase.
        """
        parts = path.split(".")
        node: Any = self._data
        for part in parts:
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def raw(self, path: str) -> dict[str, Any]:
        """
        Retorna un sub-árbol de la configuración como dict raw.

        Útil cuando el consumidor necesita iterar sobre un conjunto
        de claves (ej. todos los mercados de un deporte) en vez de
        acceder a un valor escalar específico.

        Retorna {} si la ruta no existe.
        """
        result = self.get(path, default={})
        return result if isinstance(result, dict) else {}

    def require(self, path: str) -> Any:
        """
        Como get(), pero lanza KeyError si la ruta no existe o el
        valor es None. Usar cuando un parámetro es absolutamente
        necesario para que el módulo consumidor funcione correctamente
        — fail early en inicialización, no en runtime.
        """
        value = self.get(path, default=None)
        if value is None:
            raise KeyError(
                f"Parámetro de configuración requerido no encontrado: "
                f"'{path}'. Verificar config/base.yaml y "
                f"config/{self.sport}.yaml (si aplica)."
            )
        return value


# ── Merge recursivo ────────────────────────────────────────────────────────────

def _deep_merge(base: dict, override: dict) -> dict:
    """
    Merge recursivo de dos dicts: override tiene prioridad en
    conflictos, pero no elimina claves de base que no aparezcan en
    override.

    Ejemplo:
        base     = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99}}
        result   = {"a": {"x": 1, "y": 99}, "b": 3}

    Esto permite que mlb.yaml solo defina los parámetros que difieren
    de los defaults del Core, sin repetir toda la estructura de
    base.yaml.
    """
    result = dict(base)
    for key, override_value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(override_value, dict)
        ):
            result[key] = _deep_merge(result[key], override_value)
        else:
            result[key] = override_value
    return result


# ── Carga desde disco ─────────────────────────────────────────────────────────

def _read_yaml(path: str) -> dict[str, Any]:
    """Lee y parsea un archivo YAML. Retorna {} si el archivo está vacío."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


# Cache en memoria: (sport, base_dir) → ConfigLoader
# Evita releer y parsear los mismos YAMLs en cada llamada a
# load_config() dentro de la misma ejecución del pipeline.
_cache: dict[tuple[str | None, str], ConfigLoader] = {}


def load_config(
    sport: str | None = None,
    base_dir: str = DEFAULT_CONFIG_DIR,
    force_reload: bool = False,
) -> ConfigLoader:
    """
    Carga la configuración del sistema para un sport específico.

    Parámetros
    ----------
    sport        -- Identificador del deporte ('mlb', 'soccer', 'nba',
                   etc.). Si es None, solo se carga base.yaml — útil
                   para tests del Core y scripts de migración que no
                   pertenecen a ningún sport específico.
    base_dir      -- Directorio donde se buscan base.yaml y
                   {sport}.yaml. Default 'config'.
    force_reload   -- Si True, ignora el cache en memoria y recarga
                   desde disco. Útil en tests que modifican los
                   archivos YAML entre llamadas.

    Raises
    ------
    FileNotFoundError
        Si base.yaml no existe (siempre requerido).
        Si se especifica sport y config/{sport}.yaml no existe.
        Un sport sin su YAML de configuración no puede ejecutarse —
        usaría parámetros del Core no calibrados para ese deporte.
    """
    cache_key = (sport, base_dir)
    if not force_reload and cache_key in _cache:
        return _cache[cache_key]

    # ── Cargar base.yaml (siempre requerido) ──────────────────────────────────
    base_path = os.path.join(base_dir, "base.yaml")
    if not os.path.exists(base_path):
        raise FileNotFoundError(
            f"Archivo de configuración base no encontrado: '{base_path}'. "
            f"config/base.yaml es un prerrequisito del sistema — debe "
            f"existir antes de ejecutar cualquier pipeline."
        )
    merged = _read_yaml(base_path)

    # ── Cargar {sport}.yaml y hacer deep merge si se especificó sport ─────────
    if sport is not None:
        sport_path = os.path.join(base_dir, f"{sport}.yaml")
        if not os.path.exists(sport_path):
            raise FileNotFoundError(
                f"Archivo de configuración de sport no encontrado: "
                f"'{sport_path}'. Cada sport plugin requiere su propio "
                f"YAML de configuración con los parámetros calibrados "
                f"para ese deporte. Ejecutar con parámetros del Core "
                f"(base.yaml) sin el YAML del sport contaminaría picks "
                f"reales con valores no calibrados — ver decisión de "
                f"diseño en SPORTS_PREDICTOR_ROADMAP.md, tarea 7.4."
            )
        sport_data = _read_yaml(sport_path)
        merged = _deep_merge(merged, sport_data)

    loader = ConfigLoader(data=merged, sport=sport)
    _cache[cache_key] = loader
    return loader

