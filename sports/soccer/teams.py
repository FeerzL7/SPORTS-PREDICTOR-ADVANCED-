"""
sports/soccer/teams.py

Reconciliación de nombres de equipo entre fuentes.

El problema
------------
Las tres fuentes del plugin nombran al mismo equipo de forma distinta:

    football-data.co.uk   "Man City"      "Ath Madrid"     "M'gladbach"
    Understat             "Manchester City"  "Atletico Madrid"  "Borussia M.Gladbach"
    The Odds API          "Manchester City"  "Atlético Madrid"  "Borussia Mönchengladbach"

Sin reconciliación, cruzar el xG de Understat con los resultados y
cuotas de football-data falla en SILENCIO: el equipo no aparece, sus
métricas quedan vacías y el modelo proyecta con la media de liga
creyendo que tiene datos.

Es la misma clase de fallo que apareció en el plugin NFL con los
nombres del injury report: allí `"P.Mahomes"` no cruzaba con
`"Patrick Mahomes"` y el QB titular quedaba marcado como suplente,
perdiendo los 7 puntos de penalización más importantes del modelo. Se
detectó por casualidad al validar. Aquí se resuelve de entrada.

Estrategia en dos capas
-------------------------
1. NORMALIZACIÓN AGRESIVA
   Minúsculas, sin acentos, sin sufijos societarios (FC, CF, AFC, SV,
   AC...), sin puntuación, espacios colapsados. Eso ya reconcilia la
   mayoría: "Atlético Madrid" y "Atletico Madrid" convergen, igual que
   "FC Barcelona" y "Barcelona".

2. ALIAS EXPLÍCITOS
   Para lo que la normalización no puede salvar, porque son nombres
   genuinamente distintos y no variantes ortográficas:
   "Man City" ≠ "manchester city" por ninguna regla mecánica.

Por qué no se usa coincidencia difusa
---------------------------------------
Un matcher por similitud resolvería muchos casos, pero produce falsos
positivos silenciosos que son peores que un fallo: en la Premier
conviven Sheffield United y Sheffield Wednesday; en España, Athletic
Club y Atlético Madrid. Emparejar mal dos equipos daría métricas
plausibles pero del rival equivocado, y nada en el sistema lo
detectaría.

La tabla es más larga de mantener, pero un equipo sin alias falla de
forma visible en vez de cruzarse con otro.
"""

from __future__ import annotations

import unicodedata


# ── Sufijos y prefijos societarios ───────────────────────────────────────────
#
# Se eliminan porque las fuentes los incluyen de forma inconsistente:
# football-data escribe "Ein Frankfurt" y Understat "Eintracht
# Frankfurt", pero también "FC Koln" frente a "FC Cologne".
_NOISE_TOKENS = frozenset({
    "fc", "cf", "afc", "sc", "ac", "as", "ss", "us", "rc", "rcd",
    "sv", "tsv", "vfb", "vfl", "fsv", "bsc", "sd", "ud", "cd",
    "calcio", "club", "cp", "sad", "aas", "ogc", "losc", "sco",
})


def normalize_team(name: str) -> str:
    """
    Reduce un nombre de equipo a su forma canónica comparable.

    Aplica, en orden:
        1. Minúsculas
        2. Eliminación de acentos (NFD + descarte de diacríticos)
        3. Sustitución de puntuación por espacios
        4. Eliminación de sufijos societarios
        5. Colapso de espacios

    Ejemplos:
        "Atlético Madrid"      → "atletico madrid"
        "FC Barcelona"         → "barcelona"
        "Borussia M.Gladbach"  → "borussia m gladbach"
        "Nott'm Forest"        → "nott m forest"

    Los dos últimos siguen sin coincidir con su contraparte: para eso
    está la tabla de alias.
    """
    if not name:
        return ""

    # Acentos: descomponer y descartar los diacríticos
    decomposed = unicodedata.normalize("NFD", str(name))
    stripped = "".join(
        ch for ch in decomposed if unicodedata.category(ch) != "Mn"
    )

    # Puntuación a espacio, resto a minúsculas
    cleaned = "".join(
        ch.lower() if (ch.isalnum() or ch.isspace()) else " "
        for ch in stripped
    )

    tokens = [t for t in cleaned.split() if t and t not in _NOISE_TOKENS]
    return " ".join(tokens)


# ── Alias por competición ────────────────────────────────────────────────────
#
# Clave: forma normalizada tal como la escribe football-data.co.uk.
# Valor: forma normalizada canónica, que coincide con Understat.
#
# Se agrupan por competición porque el mismo texto puede designar
# equipos distintos en ligas distintas: "Valencia" es un club español,
# pero también existe un Valencia en otras competiciones.

_ALIASES: dict[str, dict[str, str]] = {

    # ── Premier League ───────────────────────────────────────────
    # football-data abrevia mucho más en Inglaterra que en el resto.
    "epl": {
        "man city":         "manchester city",
        "man united":       "manchester united",
        "nott m forest":    "nottingham forest",
        "wolves":           "wolverhampton wanderers",
        "newcastle":        "newcastle united",
        "leicester":        "leicester city",
        "norwich":          "norwich city",
        "leeds":            "leeds united",
        "west brom":        "west bromwich albion",
        "west ham":         "west ham united",
        "cardiff":          "cardiff city",
        "hull":             "hull city",
        "stoke":            "stoke city",
        "swansea":          "swansea city",
        "huddersfield":     "huddersfield town",
        "ipswich":          "ipswich town",
        "luton":            "luton town",
        "sheffield weds":   "sheffield wednesday",
        "sheffield united": "sheffield united",
        "qpr":              "queens park rangers",
        "birmingham":       "birmingham city",
        "blackburn":        "blackburn rovers",
        "bolton":           "bolton wanderers",
        "wigan":            "wigan athletic",
        "brighton":         "brighton",
        "bournemouth":      "bournemouth",
        "tottenham":        "tottenham",
    },

    # ── La Liga ──────────────────────────────────────────────────
    # El caso crítico: "Ath Madrid" y "Ath Bilbao" comparten prefijo
    # pero son clubes distintos. Una coincidencia difusa los
    # confundiría; la tabla no.
    "laliga": {
        "ath madrid":   "atletico madrid",
        "ath bilbao":   "athletic club",
        "sociedad":     "real sociedad",
        "betis":        "real betis",
        "espanol":      "espanyol",
        "vallecano":    "rayo vallecano",
        "celta":        "celta vigo",
        "la coruna":    "deportivo la coruna",
        "valladolid":   "real valladolid",
        "sp gijon":     "sporting gijon",
        "almeria":      "almeria",
        "cadiz":        "cadiz",
        "villarreal":   "villarreal",
        "getafe":       "getafe",
        "mallorca":     "mallorca",
        "osasuna":      "osasuna",
        "girona":       "girona",
        "las palmas":   "las palmas",
        "leganes":      "leganes",
    },

    # ── Serie A ──────────────────────────────────────────────────
    "seriea": {
        "milan":       "ac milan",
        "inter":       "inter",
        "roma":        "roma",
        "lazio":       "lazio",
        "napoli":      "napoli",
        "juventus":    "juventus",
        "verona":      "verona",
        "hellas":      "verona",
        "spal":        "spal",
        "chievo":      "chievo",
        "parma":       "parma",
        "sassuolo":    "sassuolo",
        "atalanta":    "atalanta",
        "fiorentina":  "fiorentina",
        "bologna":     "bologna",
        "torino":      "torino",
        "udinese":     "udinese",
        "sampdoria":   "sampdoria",
        "genoa":       "genoa",
        "cagliari":    "cagliari",
        "empoli":      "empoli",
        "lecce":       "lecce",
        "monza":       "monza",
        "como":        "como",
        "venezia":     "venezia",
    },

    # ── Bundesliga ───────────────────────────────────────────────
    # Aquí la divergencia es sistemática: football-data abrevia los
    # nombres compuestos que Understat escribe completos.
    "bundesliga": {
        "m gladbach":     "borussia m gladbach",
        "dortmund":       "borussia dortmund",
        "leverkusen":     "bayer leverkusen",
        "ein frankfurt":  "eintracht frankfurt",
        "koln":           "cologne",
        "hertha":         "hertha berlin",
        "stuttgart":      "stuttgart",
        "bayern munich":  "bayern munich",
        "rb leipzig":     "rasenballsport leipzig",
        "mainz":          "mainz 05",
        "schalke 04":     "schalke 04",
        "werder bremen":  "werder bremen",
        "union berlin":   "union berlin",
        "hoffenheim":     "hoffenheim",
        "wolfsburg":      "wolfsburg",
        "freiburg":       "freiburg",
        "augsburg":       "augsburg",
        "bochum":         "bochum",
        "darmstadt":      "darmstadt",
        "heidenheim":     "heidenheim",
        "st pauli":       "st pauli",
        "holstein kiel":  "holstein kiel",
        "hamburg":        "hamburger sv",
        "greuther furth": "greuther furth",
        "paderborn":      "paderborn",
        "bielefeld":      "arminia bielefeld",
    },

    # ── Ligue 1 ──────────────────────────────────────────────────
    "ligue1": {
        "paris sg":      "paris saint germain",
        "paris fc":      "paris fc",
        "st etienne":    "saint etienne",
        "marseille":     "marseille",
        "lyon":          "lyon",
        "monaco":        "monaco",
        "lille":         "lille",
        "nice":          "nice",
        "rennes":        "rennes",
        "montpellier":   "montpellier",
        "nantes":        "nantes",
        "strasbourg":    "strasbourg",
        "reims":         "reims",
        "lens":          "lens",
        "brest":         "brest",
        "toulouse":      "toulouse",
        "angers":        "angers",
        "auxerre":       "auxerre",
        "le havre":      "le havre",
        "metz":          "metz",
        "bordeaux":      "bordeaux",
        "lorient":       "lorient",
        "clermont":      "clermont foot",
        "troyes":        "troyes",
        "ajaccio":       "ajaccio",
    },
}


# ── API pública ──────────────────────────────────────────────────────────────

def canonical_team(name: str, comp_id: str = "") -> str:
    """
    Nombre canónico de un equipo, comparable entre fuentes.

    Parámetros
    ----------
    name    -- Nombre tal como lo escribe la fuente.
    comp_id -- Competición. Los alias se agrupan por liga porque el
               mismo texto puede designar equipos distintos en
               competiciones distintas.

    Retorna la forma normalizada, con el alias aplicado si existe.
    Un nombre sin alias devuelve simplemente su normalización, que es
    lo correcto para la mayoría de equipos: solo los que las fuentes
    escriben de forma genuinamente distinta necesitan entrada.
    """
    normalized = normalize_team(name)
    if not normalized:
        return ""

    aliases = _ALIASES.get(str(comp_id).strip().lower(), {})
    return aliases.get(normalized, normalized)


def same_team(name_a: str, name_b: str, comp_id: str = "") -> bool:
    """True si ambos nombres designan al mismo equipo."""
    canon_a = canonical_team(name_a, comp_id)
    return bool(canon_a) and canon_a == canonical_team(name_b, comp_id)


def display_name(name: str) -> str:
    """
    Nombre legible, conservando la grafía de la fuente.

    La forma canónica sirve para comparar, no para mostrar: nadie
    quiere leer "borussia m gladbach" en un pick. Esta función se
    limita a limpiar espacios, dejando el nombre tal como llegó.
    """
    return " ".join(str(name or "").split())


def known_aliases(comp_id: str) -> dict[str, str]:
    """Tabla de alias de una competición, para diagnóstico."""
    return dict(_ALIASES.get(str(comp_id).strip().lower(), {}))


def alias_coverage() -> dict[str, int]:
    """Número de alias declarados por competición."""
    return {comp: len(table) for comp, table in _ALIASES.items()}


def diagnose(
    names_a:  list[str],
    names_b:  list[str],
    comp_id:  str,
) -> dict:
    """
    Compara dos listas de nombres y reporta los que no cruzan.

    Pensado para ejecutarse tras cargar una temporada nueva: si una
    fuente cambia la grafía de un equipo, o asciende uno sin alias,
    esto lo detecta antes de que sus métricas queden vacías en
    silencio.

    Retorna
    -------
    dict con las claves:
        matched    -- Equipos presentes en ambas listas.
        only_in_a  -- Solo en la primera (típicamente football-data).
        only_in_b  -- Solo en la segunda (típicamente Understat).
        coverage   -- Fracción de la primera lista que cruzó.
    """
    canon_a = {canonical_team(n, comp_id): n for n in names_a if n}
    canon_b = {canonical_team(n, comp_id): n for n in names_b if n}

    common = set(canon_a) & set(canon_b)
    only_a = set(canon_a) - set(canon_b)
    only_b = set(canon_b) - set(canon_a)

    return {
        "matched":   sorted(common),
        "only_in_a": sorted(canon_a[k] for k in only_a),
        "only_in_b": sorted(canon_b[k] for k in only_b),
        "coverage":  round(len(common) / len(canon_a), 4) if canon_a else 0.0,
    }