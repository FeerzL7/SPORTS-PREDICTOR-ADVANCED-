#core/utils/cache.py

from __future__ import annotations

import json
import os
import time
from typing import Any


# Directorio base donde se almacenan todos los namespaces de caché.
# Cada namespace resuelve a {DEFAULT_BASE_DIR}/{namespace}.json,
# centralizando la convención de nombres de archivo que en el sistema
# MLB original estaba dispersa (output/offense_cache.json,
# output/adv_pitching_cache.json, output/park_factors_cache.json).
DEFAULT_BASE_DIR = "output/cache"


class CacheManager:
    """
    Caché de dos niveles para un namespace específico.

    Nivel 1 (memoria): dict en el proceso actual, evita I/O repetido
    dentro de la misma ejecución del pipeline.
    Nivel 2 (disco): JSON persistente entre ejecuciones, con TTL
    basado en mtime del archivo.

    Ambos niveles están siempre activos. Cada instancia gestiona un
    namespace independiente. Para datos relacionados pero
    conceptualmente distintos (ej. statcast pitching vs batting),
    usar dos instancias con namespaces distintos.

    Atributos
    ---------
    namespace   -- Identificador del caché. Resuelve a
                  {base_dir}/{namespace}.json. Usar nombres que
                  incluyan el sport para evitar colisiones entre
                  plugins (ej. 'mlb_offense', no solo 'offense').
    ttl_hours    -- Horas antes de que el namespace se considere
                  expirado. 6 para datos intradiarios (offense),
                  24 para datos de baja frecuencia de cambio
                  (statcast, park_factors).
    base_dir      -- Directorio raíz. Default DEFAULT_BASE_DIR.
    """

    def __init__(
        self,
        namespace: str,
        ttl_hours: float,
        base_dir: str = DEFAULT_BASE_DIR,
    ) -> None:
        self.namespace = namespace
        self.ttl_hours = ttl_hours
        self.base_dir = base_dir
        self._path = os.path.join(base_dir, f"{namespace}.json")
        self._memory: dict[str, Any] | None = None  # None = no cargado aún

    # ── TTL ────────────────────────────────────────────────────────────────────

    def _file_age_hours(self) -> float | None:
        """Antigüedad del archivo en horas, o None si no existe."""
        if not os.path.exists(self._path):
            return None
        return (time.time() - os.path.getmtime(self._path)) / 3600

    def is_fresh(self) -> bool:
        """
        True si el archivo existe y su antigüedad es menor a
        ttl_hours. Replica el patrón de park_factors.py original
        (_cache_vigente): evalúa el TTL del namespace completo antes
        de decidir si recalcular, en vez de verificar clave por clave.
        """
        age = self._file_age_hours()
        return age is not None and age < self.ttl_hours

    # ── Carga perezosa ─────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> dict[str, Any]:
        """
        Carga el namespace desde disco a memoria si aún no se ha
        cargado, y si el archivo no expiró. Si expiró o está
        corrupto, trata el caché como vacío sin propagar excepción
        — un caché roto no debe interrumpir el pipeline.
        """
        if self._memory is not None:
            return self._memory

        if self.is_fresh():
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._memory = json.load(f)
            except (OSError, json.JSONDecodeError):
                self._memory = {}
        else:
            self._memory = {}

        return self._memory # type: ignore

    # ── Operaciones por clave ──────────────────────────────────────────────────

    def get(self, key: str) -> Any | None:
        """
        Retorna el valor cacheado para key, o None si la clave no
        existe o el namespace expiró.

        NUNCA evalúa si el valor retornado es semánticamente válido.
        Ver nota de diseño en el docstring del módulo (hallazgo F4).
        """
        return self._ensure_loaded().get(key)

    def set(self, key: str, value: Any) -> None:
        """
        Almacena value bajo key en memoria y en disco.

        Reescribe el archivo completo del namespace en cada llamada
        — mismo comportamiento que _guardar_en_cache() del sistema
        MLB original. Aceptable para los volúmenes actuales (decenas
        de claves); si un sport plugin futuro con namespaces grandes
        demuestra esto como cuello de botella real, ese es el momento
        de optimizar, no antes.
        """
        memory = self._ensure_loaded()
        memory[key] = value
        self._write_to_disk(memory)

    def invalidate(self, key: str) -> None:
        """Elimina key del caché (memoria y disco). No falla si no existe."""
        memory = self._ensure_loaded()
        if key in memory:
            del memory[key]
            self._write_to_disk(memory)

    # ── Operaciones de namespace completo ──────────────────────────────────────

    def get_all(self) -> dict[str, Any]:
        """
        Retorna el namespace completo como dict. Útil para el patrón
        park_factors.py: cargar todos los valores de una vez tras
        confirmar is_fresh().
        """
        return dict(self._ensure_loaded())

    def set_all(self, values: dict[str, Any]) -> None:
        """
        Reemplaza el namespace completo con values. Útil para el
        patrón park_factors.py / statcast.py: recalcular y persistir
        todo el conjunto de datos en una sola operación.
        """
        self._memory = dict(values)
        self._write_to_disk(self._memory)

    def clear(self) -> None:
        """
        Limpia el namespace completo: memoria y archivo de disco.
        Equivalente a borrar manualmente el archivo de caché (paso
        de remediación documentado en SPORTS_PREDICTOR_ROADMAP.md,
        tarea F0.1.4 para el bug de offense_cache.json).
        """
        self._memory = {}
        if os.path.exists(self._path):
            os.remove(self._path)

    # ── Persistencia ───────────────────────────────────────────────────────────

    def _write_to_disk(self, memory: dict[str, Any]) -> None:
        os.makedirs(self.base_dir, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2, ensure_ascii=False)
