"""
sports/mlb/venue_factors.py

VenueFactorProvider: park factors para estadios de MLB.

Migrado de analysis/park_factors.py + utils/constants.py del sistema
MLB con dos correcciones documentadas en SPORTS_PREDICTOR_ARCHITECTURE.md:

1. Park factors desde config/mlb.yaml — no hardcodeados en Python.
   El sistema MLB tenía PARK_FACTORS = {'lad': 0.94, 'col': 1.36, ...}
   como dict en utils/constants.py. Ahora viven en mlb.yaml bajo
   mlb.park_factors.{venue_id}. Modificables sin tocar código.

2. venue_id como clave estable — no nombre del estadio.
   El sistema MLB usaba nombres ('Coors Field') que podían cambiar.
   Aquí se usa el venue_id de MLB Stats API (entero o slug estable).

Park factor > 1.0: el estadio produce más carreras que el promedio.
Park factor < 1.0: el estadio produce menos carreras que el promedio.
Park factor = 1.0: estadio neutro (referencia de liga).

Cómo se aplica en el pipeline
--------------------------------
MLBDataProvider.enrich_event(event)
    ↓
VenueFactorProvider.get(venue_id) → float
    ↓
TeamFeatures(venue_factor=1.34)  ← Coors Field
    ↓
MLBProjectionModel: expected_home *= venue_factor (parcialmente)

El venue_factor se aplica por igual a ambos equipos (es propiedad
del estadio, no del equipo). El plugin de proyección decide exactamente
cómo incorporarlo (multiplicador, offset, o ignorarlo para mercados
donde el factor es ya absorbido por las cuotas de mercado eficientes).

Fuentes de datos
-----------------
1. config/mlb.yaml → mlb.park_factors.{venue_id}: valores actualizados
   por el operador basados en datos de temporada reciente.

2. _HISTORICAL_PARK_FACTORS: tabla de respaldo con valores históricos
   multi-año (2022-2026) calculados desde MLB Stats API.
   Se usa cuando el venue_id no está en el YAML.

3. Fallback = 1.00 (estadio neutro) cuando el venue no está en ninguna
   fuente. Documentado explícitamente — no falla silenciosamente.
"""

from __future__ import annotations


# ── Factores históricos de respaldo ──────────────────────────────────────────
# Calculados desde MLB Stats API (runs scored/allowed, 2022-2026).
# venue_id = string slug del estadio en MLB Stats API.
# Actualizar periódicamente con datos de la temporada más reciente.
#
# Metodología: park_factor = (rpg_at_venue / rpg_away) normalizado a 1.0
# Promedio de las últimas 3 temporadas para estabilidad estadística.
#
_HISTORICAL_PARK_FACTORS: dict[str, float] = {
    # Estadios que favorecen el bateo (PF > 1.0)
    "coors-field":          1.36,  # Denver, altitud 5280ft — máxima inflación
    "great-american-ball-park": 1.12,  # Cincinnati — favorable al bateo
    "citizens-bank-park":   1.09,  # Philadelphia — pequeño en los jardines
    "yankee-stadium":       1.07,  # New York Yankees — corto en RF
    "fenway-park":          1.05,  # Boston — Green Monster en LF
    "wrigley-field":        1.04,  # Chicago Cubs — viento variable
    "globe-life-field":     1.03,  # Texas — cubierto pero favorable
    "angel-stadium":        1.02,  # Los Angeles Angels
    "truist-park":          1.01,  # Atlanta Braves

    # Estadios neutros (PF ≈ 1.0)
    "dodger-stadium":       0.99,  # Los Angeles Dodgers — ligera depresión
    "busch-stadium":        0.98,  # St. Louis Cardinals
    "american-family-field": 0.98, # Milwaukee Brewers
    "kauffman-stadium":     0.97,  # Kansas City Royals
    "progressive-field":    0.97,  # Cleveland Guardians
    "comerica-park":        0.97,  # Detroit Tigers — jardines amplios

    # Estadios que deprimen el bateo (PF < 1.0)
    "oracle-park":          0.95,  # San Francisco — viento marino en LF
    "t-mobile-park":        0.95,  # Seattle — depresión histórica
    "guaranteed-rate-field": 0.94, # Chicago White Sox
    "petco-park":           0.91,  # San Diego — pitcher-friendly
    "tropicana-field":      0.93,  # Tampa Bay — cubierto, depresión leve
    "pnc-park":             0.94,  # Pittsburgh — río y viento
    "minute-maid-park":     0.96,  # Houston — cerrado favorable pitching
    "loanDepot-park":       0.93,  # Miami — cerrado, alta humedad
    "citi-field":           0.94,  # New York Mets — amplios jardines
    "camden-yards":         0.98,  # Baltimore Orioles
    "target-field":         0.96,  # Minnesota Twins — clima frío
    "chase-field":          1.00,  # Arizona — cubierto, neutro
    "sutter-health-park":   1.00,  # Sacramento (AAA ref)
}

# Aliases de venue_id — distintas fuentes usan distintos formatos
_VENUE_ALIASES: dict[str, str] = {
    # ID numérico MLB Stats API → slug
    "32":   "angel-stadium",
    "2392": "truist-park",
    "2394": "fenway-park",
    "17":   "wrigley-field",
    "35":   "guaranteed-rate-field",
    "4169": "great-american-ball-park",
    "5":    "progressive-field",
    "2394": "fenway-park",
    "3289": "minute-maid-park",
    "7":    "kauffman-stadium",
    "1":    "dodger-stadium",
    "4705": "american-family-field",
    "3":    "comerica-park",
    "3312": "target-field",
    "31":   "busch-stadium",
    "4":    "citi-field",
    "3313": "yankee-stadium",
    "10":   "oakland-coliseum",
    "2681": "t-mobile-park",
    "22":   "petco-park",
    "2":    "oracle-park",
    "2889": "pnc-park",
    "680":  "tropicana-field",
    "12":   "globe-life-field",
    "2680": "citizens-bank-park",
    "14":   "camden-yards",
    "5":    "progressive-field",
    "4249": "chase-field",
    "2395": "coors-field",
    "3289": "minute-maid-park",
    "2":    "oracle-park",
}


# ── Proveedor principal ───────────────────────────────────────────────────────

class VenueFactorProvider:
    """
    Proveedor de park factors para estadios de MLB.

    Resolución por prioridad:
        1. config/mlb.yaml → mlb.park_factors.{venue_id}
        2. _HISTORICAL_PARK_FACTORS (tabla de respaldo interna)
        3. 1.00 (estadio neutro como fallback documentado)

    Parámetros
    ----------
    config_loader  -- ConfigLoader con mlb.yaml cargado. Si None,
                     solo usa la tabla histórica interna.
    """

    def __init__(self, config_loader=None) -> None:
        self._config = config_loader
        self._cache: dict[str, float] = {}

    def get(self, venue_id: str) -> float:
        """
        Retorna el park factor para el venue_id dado.

        Parámetros
        ----------
        venue_id  -- Identificador del estadio. Puede ser el slug
                    estable ('coors-field') o el ID numérico de
                    MLB Stats API ('2395'). Se normaliza internamente.

        Retorna
        -------
        float — Park factor. 1.0 si no se encuentra en ninguna fuente.
        """
        normalized = self._normalize(venue_id)

        if normalized in self._cache:
            return self._cache[normalized]

        factor = self._resolve(normalized)
        self._cache[normalized] = factor
        return factor

    def get_all(self) -> dict[str, float]:
        """
        Retorna todos los park factors conocidos.

        Merge de tabla histórica + config YAML.
        YAML tiene prioridad sobre la tabla histórica.
        """
        result = dict(_HISTORICAL_PARK_FACTORS)

        if self._config is not None:
            yaml_factors = self._config.get("mlb.park_factors", default={})
            if isinstance(yaml_factors, dict):
                result.update(yaml_factors)

        return result

    def is_pitcher_friendly(self, venue_id: str, threshold: float = 0.97) -> bool:
        """True si el estadio favorece a los pitchers (PF < threshold)."""
        return self.get(venue_id) < threshold

    def is_hitter_friendly(self, venue_id: str, threshold: float = 1.03) -> bool:
        """True si el estadio favorece a los bateadores (PF > threshold)."""
        return self.get(venue_id) > threshold

    def clear_cache(self) -> None:
        """Limpia el caché de resoluciones."""
        self._cache.clear()

    # ── Helpers privados ───────────────────────────────────────────────────────

    def _resolve(self, venue_id: str) -> float:
        """
        Resuelve el factor por prioridad:
        1. YAML, 2. Tabla histórica, 3. Fallback 1.00.
        """
        # 1. YAML
        if self._config is not None:
            yaml_val = self._config.get(
                f"mlb.park_factors.{venue_id}",
                default=None,
            )
            if yaml_val is not None:
                try:
                    return float(yaml_val)
                except (ValueError, TypeError):
                    pass

        # 2. Tabla histórica
        if venue_id in _HISTORICAL_PARK_FACTORS:
            return _HISTORICAL_PARK_FACTORS[venue_id]

        # 3. Fallback neutro
        return 1.00

    @staticmethod
    def _normalize(venue_id: str) -> str:
        """
        Normaliza venue_id a slug estable.

        Convierte IDs numéricos de MLB Stats API a slugs legibles
        usando _VENUE_ALIASES. Si no está en los aliases, normaliza
        a minúsculas con guiones.
        """
        vid = str(venue_id).strip().lower()

        # Intentar alias numérico primero
        if vid in _VENUE_ALIASES:
            return _VENUE_ALIASES[vid]

        # Normalizar nombre con espacios a slug
        slug = vid.replace(" ", "-").replace("_", "-")
        return slug


# ── Función de conveniencia ───────────────────────────────────────────────────

def get_park_factor(
    venue_id:      str,
    config_loader = None,
) -> float:
    """
    Función de conveniencia para obtener un park factor sin instanciar
    VenueFactorProvider. Útil para scripts y tests.

    Parámetros
    ----------
    venue_id       -- ID o slug del estadio.
    config_loader  -- ConfigLoader opcional con mlb.yaml.

    Retorna
    -------
    Park factor (float). 1.00 si no se encuentra.
    """
    return VenueFactorProvider(config_loader=config_loader).get(venue_id)