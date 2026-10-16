#!/usr/bin/env python3
"""
audit_nba_sources.py — Auditoría de las fuentes de datos para el plugin NBA.

OBJETIVO
--------
Antes de escribir una sola línea de `sports/nba/`, demostrar con datos
reales que las tres piezas que el Bloque 12 necesita existen, cruzan
entre sí y tienen calidad suficiente:

    1. RESULTADOS      — box score por equipo y partido, con fecha.
    2. CUOTAS DE CIERRE — spread y total de cierre, histórico.
    3. EFICIENCIA       — ORtg/DRtg/Pace derivables con barrera temporal.

La lección del Bloque 11 (soccer) fue que construimos 7.594 líneas y
descubrimos en la última tarea que el modelo no batía al mercado. Esa
medición solo necesitaba calendario, estadísticas, un modelo y cuotas
históricas. Este script verifica que esos insumos están disponibles —
y mide, de paso, cuán lejos está un modelo ingenuo de la línea de
cierre. Ese número decide si el Bloque 12 se construye o no.

ARQUITECTURA
------------
Script independiente, sin dependencias del paquete `core/` ni de
`sports/`. Se ejecuta antes de que exista el plugin, así que no puede
importar nada de él. Solo stdlib + pandas + requests.

    descargar (con caché en disco)
        ↓
    validar cada fuente por separado
        ↓
    cruzar cuotas ↔ resultados por game_id
        ↓
    contrastar las dos fuentes de cuotas entre sí
        ↓
    medir sigma del mercado y sigma de un modelo ingenuo
        ↓
    veredicto go / no-go

FUENTE
------
`sportsdataverse/hoopR-nba-data`, licencia CC BY 4.0. Datos derivados
de ESPN (box scores, calendario) y de The Odds API (consenso de cierre,
backfill del mantenedor). Se descarga vía raw.githubusercontent.com:
no requiere credenciales ni cuenta.

RIESGOS CONOCIDOS
-----------------
  * `closing_lines_odds_api.parquet` es un backfill estático (un solo
    commit, 2026-08-26). No se actualiza con el workflow diario del
    repo. Cubre hasta el final de 2025-26; para 2026-27 en adelante
    hay que capturar el cierre nosotros mismos.
  * Ninguna de las dos fuentes de cuotas trae PRECIOS, solo líneas.
    El backtest de spread/total es viable asumiendo el -110 estándar;
    el moneyline NO es backtesteable con estos datos.
  * ~1% de los snapshots de cierre son POSTERIORES al tip-off. Se
    excluyen explícitamente: es exactamente el look-ahead que infló
    el ROI de totales en NFL ocho puntos.

USO
---
    python scripts/audit_nba_sources.py
    python scripts/audit_nba_sources.py --refresh      # ignora la caché
    python scripts/audit_nba_sources.py --cache-dir X  # otra ruta

Código de salida 0 si todas las comprobaciones pasan, 1 si alguna falla.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    print("ERROR: falta pandas.  pip install pandas", file=sys.stderr)
    raise SystemExit(1)

try:
    import requests
except ImportError:  # pragma: no cover
    print("ERROR: falta requests.  pip install requests", file=sys.stderr)
    raise SystemExit(1)


# ── Configuración de la fuente ────────────────────────────────────────────────

_RAW_BASE = (
    "https://raw.githubusercontent.com/sportsdataverse/hoopR-nba-data/main/nba"
)

# Temporadas de box score a descargar. El nombre del archivo usa el año de
# FIN de temporada: team_box_2021.parquet es la temporada 2020-21.
_BOX_SEASONS = tuple(range(2018, 2027))

_CLOSING_LINES = "betting_lines/closing_lines_odds_api.parquet"
_GAMES_ARCHIVE = "betting_lines/games-archive.json"

# Ventana en la que un snapshot cuenta como "cierre". Fuera de ella el dato
# se descarta: negativo = posterior al tip-off (look-ahead), >60 min = la
# línea todavía se movió después.
_CLOSE_WINDOW_MIN = (0.0, 60.0)

# Partidos previos mínimos para que un rating acumulado sea utilizable.
_MIN_PRIOR_GAMES = 15

# Coeficiente de posesiones de Dean Oliver para tiros libres.
_FT_POSSESSION_WEIGHT = 0.44

_SEASON_TYPE_REGULAR = 2


# ── Utilidades de presentación ────────────────────────────────────────────────

def _header(text: str) -> None:
    print()
    print("=" * 70)
    print(f"  {text}")
    print("=" * 70)


def _section(text: str) -> None:
    print()
    print(f"  {text}")
    print("  " + "─" * 66)


@dataclass
class Check:
    """Una comprobación con veredicto, para el resumen final."""

    name: str
    passed: bool
    detail: str


_checks: list[Check] = []


def _check(name: str, passed: bool, detail: str) -> bool:
    _checks.append(Check(name, passed, detail))
    mark = "OK  " if passed else "FALLA"
    print(f"    [{mark}] {name}: {detail}")
    return passed


# ── Descarga con caché ────────────────────────────────────────────────────────

def _download(relative: str, cache_dir: Path, refresh: bool) -> Path:
    """
    Descarga `relative` desde el repo y lo deja en la caché local.

    La caché es deliberadamente simple: si el archivo existe y no se pidió
    --refresh, se reutiliza. Estos datasets son inmutables salvo el box
    score de la temporada en curso, que se refresca con --refresh.
    """
    destination = cache_dir / Path(relative).name
    if destination.exists() and not refresh:
        size_mb = destination.stat().st_size / 1e6
        print(f"    cache  {destination.name:<40} {size_mb:6.1f} MB")
        return destination

    url = f"{_RAW_BASE}/{relative}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)
    size_mb = len(response.content) / 1e6
    print(f"    baja   {destination.name:<40} {size_mb:6.1f} MB")
    return destination


def _read_parquet(path: Path) -> pd.DataFrame:
    """
    Lee parquet con el motor que esté disponible.

    No forzamos pyarrow: en esta máquina numpy/pandas quedaron en un
    equilibrio delicado tras la instalación de nfl-data-py, y fastparquet
    ya está presente. Instalar pyarrow a ciegas podría romperlo.
    """
    last_error: Exception | None = None
    for engine in ("pyarrow", "fastparquet"):
        try:
            return pd.read_parquet(path, engine=engine)
        except ImportError as error:
            last_error = error
        except Exception as error:  # motor presente pero el archivo falló
            raise RuntimeError(f"{path.name}: {error}") from error
    raise RuntimeError(
        f"No hay motor de parquet disponible para {path.name}. "
        f"Instala uno:  pip install fastparquet   (ultimo error: {last_error})"
    )


# ── Carga de las tres fuentes ─────────────────────────────────────────────────

def load_sources(cache_dir: Path, refresh: bool) -> tuple[pd.DataFrame, ...]:
    _section("Descarga")
    closing_path = _download(_CLOSING_LINES, cache_dir, refresh)
    archive_path = _download(_GAMES_ARCHIVE, cache_dir, refresh)

    box_frames: list[pd.DataFrame] = []
    for season in _BOX_SEASONS:
        try:
            box_path = _download(
                f"team_box/parquet/team_box_{season}.parquet", cache_dir, refresh
            )
        except requests.HTTPError:
            print(f"    omite  team_box_{season}.parquet (no publicado)")
            continue
        box_frames.append(_read_parquet(box_path))

    if not box_frames:
        raise RuntimeError("No se pudo descargar ningún box score.")

    closing = _read_parquet(closing_path)
    archive = pd.DataFrame(json.loads(archive_path.read_text(encoding="utf-8")))
    team_box = pd.concat(box_frames, ignore_index=True)
    return closing, archive, team_box


# ── 1. Resultados ─────────────────────────────────────────────────────────────

def audit_team_box(team_box: pd.DataFrame) -> pd.DataFrame:
    _header("FUENTE 1 — RESULTADOS Y BOX SCORE  (ESPN vía hoopR)")

    team_box = team_box.copy()
    team_box["game_date"] = pd.to_datetime(team_box["game_date"])
    team_box["game_id"] = team_box["game_id"].astype("int64")

    print()
    print(f"    filas            : {len(team_box):,}")
    print(f"    partidos únicos  : {team_box.game_id.nunique():,}")
    print(
        f"    rango de fechas  : {team_box.game_date.min().date()} "
        f"→ {team_box.game_date.max().date()}"
    )

    required = [
        "field_goals_attempted",
        "free_throws_attempted",
        "offensive_rebounds",
        "total_turnovers",
        "team_score",
        "opponent_team_score",
        "team_home_away",
        "season_type",
    ]
    missing = [column for column in required if column not in team_box.columns]

    print()
    _check(
        "columnas para estimar posesiones",
        not missing,
        "presentes" if not missing else f"faltan {missing}",
    )

    # Dos filas por partido (local y visitante) es la forma esperada.
    rows_per_game = len(team_box) / max(team_box.game_id.nunique(), 1)
    _check(
        "dos filas por partido",
        1.98 <= rows_per_game <= 2.02,
        f"{rows_per_game:.3f} filas/partido",
    )

    regular = team_box[team_box.season_type == _SEASON_TYPE_REGULAR]
    _check(
        "distinción temporada regular / playoffs",
        0 < len(regular) < len(team_box),
        f"{len(regular):,} de {len(team_box):,} filas son temporada regular",
    )
    return team_box


# ── 2. Cuotas de cierre ───────────────────────────────────────────────────────

def audit_closing_lines(closing: pd.DataFrame) -> pd.DataFrame:
    _header("FUENTE 2 — CUOTAS DE CIERRE  (The Odds API, consenso)")

    closing = closing.copy()
    closing["commence_time"] = pd.to_datetime(closing["commence_time"], utc=True)
    closing["snapshot"] = pd.to_datetime(closing["snapshot"], utc=True)
    # Año de fin de temporada: agosto en adelante pertenece a la siguiente.
    closing["season"] = closing.commence_time.dt.year + (
        closing.commence_time.dt.month >= 8
    ).astype(int)

    lead = (closing.commence_time - closing.snapshot).dt.total_seconds() / 60.0
    closing["lead_minutes"] = lead

    print()
    print(
        f"    rango            : {closing.commence_time.min().date()} "
        f"→ {closing.commence_time.max().date()}"
    )
    print(f"    partidos         : {len(closing):,}")

    print()
    print("    por temporada (año de fin):")
    per_season = closing.groupby("season").agg(
        partidos=("game_id", "size"),
        casas_media=("n_books", "mean"),
        con_total=("over_under", lambda s: s.notna().mean()),
    )
    for season, row in per_season.iterrows():
        print(
            f"      {season}   {int(row.partidos):>5} partidos   "
            f"{row.casas_media:>5.1f} casas   "
            f"total presente {row.con_total * 100:5.1f}%"
        )

    low, high = _CLOSE_WINDOW_MIN
    after_tipoff = int((lead < low).sum())
    too_early = int((lead > high).sum())
    valid = int(((lead >= low) & (lead <= high)).sum())

    print()
    print("    ventana del snapshot respecto al tip-off:")
    print(f"      posterior al tip-off (LOOK-AHEAD) : {after_tipoff:>5}")
    print(f"      más de {high:.0f} min antes            : {too_early:>5}")
    print(f"      cierre válido [{low:.0f}, {high:.0f}] min        : {valid:>5}")
    print(f"      mediana de antelación             : {lead.median():.1f} min")

    print()
    _check(
        "identificador cruzable con resultados",
        "game_id" in closing.columns and closing.game_id.duplicated().sum() == 0,
        f"game_id único en {len(closing):,} filas",
    )
    _check(
        "cierre válido en la gran mayoría",
        valid / len(closing) > 0.95,
        f"{valid / len(closing) * 100:.2f}% dentro de la ventana",
    )
    _check(
        "precios por casa",
        False,
        "AUSENTES — solo líneas. Moneyline NO backtesteable; "
        "spread/total asumen -110",
    )
    return closing


# ── 3. Contraste entre las dos fuentes de cuotas ──────────────────────────────

def cross_validate_odds(
    closing: pd.DataFrame, archive: pd.DataFrame, team_box: pd.DataFrame
) -> None:
    _header("CONTRASTE — dos fuentes de cuotas independientes")

    archive = archive.copy()
    archive["game_date"] = pd.to_datetime(archive["game_date"])
    archive["line"] = pd.to_numeric(archive["line"], errors="coerce")
    archive["archive_total"] = pd.to_numeric(
        archive["game_over_under"], errors="coerce"
    )

    print()
    print(
        f"    games-archive.json: {len(archive):,} partidos, "
        f"{archive.game_date.min().date()} → {archive.game_date.max().date()}"
    )

    home = team_box[team_box.team_home_away == "home"][
        ["game_id", "game_date", "team_abbreviation", "opponent_team_abbreviation"]
    ]

    def key(frame: pd.DataFrame, date: str, home_col: str, away_col: str) -> pd.Series:
        return (
            pd.to_datetime(frame[date]).dt.strftime("%Y-%m-%d")
            + "|"
            + frame[home_col].astype(str)
            + "|"
            + frame[away_col].astype(str)
        )

    left = closing.merge(home, on="game_id", how="inner")
    left["key"] = key(
        left, "game_date", "team_abbreviation", "opponent_team_abbreviation"
    )
    archive["key"] = key(
        archive, "game_date", "home_team_abbrev", "visit_team_abbrev"
    )

    overlap = left.merge(
        archive[["key", "line", "archive_total"]], on="key", how="inner"
    )

    if overlap.empty:
        _check("solape entre fuentes", False, "ninguna fila en común")
        return

    # `line` del archivo es la línea del favorito (negativa). `home_team_spread`
    # del consenso es positiva cuando el local es favorito. De ahí el signo.
    spread_gap = (overlap.home_team_spread - (-overlap.line)).abs()
    total_gap = (overlap.over_under - overlap.archive_total).abs()

    print()
    print(f"    partidos en común: {len(overlap):,}")
    print(
        f"      spread  idéntico {(spread_gap == 0).mean() * 100:5.1f}%   "
        f"≤1 pt {(spread_gap <= 1).mean() * 100:5.1f}%   "
        f"discrepancia media {spread_gap.mean():.2f} pts"
    )
    print(
        f"      total   idéntico {(total_gap == 0).mean() * 100:5.1f}%   "
        f"≤1 pt {(total_gap <= 1).mean() * 100:5.1f}%   "
        f"discrepancia media {total_gap.mean():.2f} pts"
    )

    print()
    _check(
        "dos fuentes se corroboran",
        (spread_gap <= 1).mean() > 0.90,
        f"{(spread_gap <= 1).mean() * 100:.1f}% de los spreads coinciden "
        f"dentro de 1 punto",
    )


# ── 4. Cruce cuotas ↔ resultados y calidad de la línea ────────────────────────

def audit_settlement(closing: pd.DataFrame, team_box: pd.DataFrame) -> pd.DataFrame:
    _header("CRUCE — ¿se pueden liquidar las apuestas?")

    low, high = _CLOSE_WINDOW_MIN
    usable = closing[
        (closing.lead_minutes >= low) & (closing.lead_minutes <= high)
    ].copy()

    home = team_box[team_box.team_home_away == "home"][
        [
            "game_id",
            "game_date",
            "season",
            "season_type",
            "team_display_name",
            "team_score",
            "opponent_team_score",
        ]
    ]
    joined = usable.merge(home, on="game_id", how="inner", suffixes=("", "_box"))

    print()
    print(
        f"    cuotas con cierre válido : {len(usable):,}\n"
        f"    cruzan con resultados    : {len(joined):,} "
        f"({len(joined) / len(usable) * 100:.2f}%)"
    )

    joined["margin"] = joined.team_score - joined.opponent_team_score
    joined["game_total"] = joined.team_score + joined.opponent_team_score
    joined["ats_edge"] = joined.margin - joined.home_team_spread
    joined["total_edge"] = joined.game_total - joined.over_under

    print()
    print("    liquidación:")
    print(
        f"      el local cubre      : {(joined.ats_edge > 0).mean() * 100:5.2f}%   "
        f"(push exacto: {int((joined.ats_edge == 0).sum())})"
    )
    print(
        f"      sale el over        : {(joined.total_edge > 0).mean() * 100:5.2f}%   "
        f"(push exacto: {int((joined.total_edge == 0).sum())})"
    )

    print()
    print("    calidad de la línea de cierre — esto es lo que hay que batir:")
    print(f"      sigma(margen − spread) = {joined.ats_edge.std():6.2f} pts")
    print(f"      sigma(total − O/U)     = {joined.total_edge.std():6.2f} pts")

    print()
    _check(
        "cruce por game_id",
        len(joined) / len(usable) > 0.95,
        f"{len(joined) / len(usable) * 100:.2f}% de las cuotas encuentran resultado",
    )
    # Un mercado eficiente reparte los lados casi 50/50. Una desviación
    # grande señalaría un error de signo nuestro, no una ineficiencia real.
    cover = (joined.ats_edge > 0).mean()
    _check(
        "línea sin sesgo detectable",
        0.46 <= cover <= 0.54,
        f"el local cubre {cover * 100:.2f}% — consistente con mercado eficiente",
    )
    return joined


# ── 5. Eficiencia derivada y señal del modelo ingenuo ─────────────────────────

def audit_efficiency(team_box: pd.DataFrame, settled: pd.DataFrame) -> None:
    _header("FUENTE 3 — EFICIENCIA DERIVADA, CON BARRERA TEMPORAL")

    regular = team_box[team_box.season_type == _SEASON_TYPE_REGULAR].copy()
    regular["possessions"] = (
        regular.field_goals_attempted
        - regular.offensive_rebounds
        + regular.total_turnovers
        + _FT_POSSESSION_WEIGHT * regular.free_throws_attempted
    )
    regular = regular.sort_values(["season", "game_date", "game_id"])

    # BARRERA TEMPORAL: el acumulado resta la fila actual, así que un rating
    # nunca incluye el partido que está prediciendo. Misma garantía que
    # NFLDataProvider._cutoff_week y SoccerDataProvider._as_of.
    grouped = regular.groupby(["season", "team_id"])
    for column in ("team_score", "opponent_team_score", "possessions"):
        regular[f"{column}_prior"] = grouped[column].cumsum() - regular[column]
    regular["prior_games"] = grouped.cumcount()

    prior_possessions = regular.possessions_prior.replace(0, pd.NA)
    regular["ortg"] = 100 * regular.team_score_prior / prior_possessions
    regular["drtg"] = 100 * regular.opponent_team_score_prior / prior_possessions
    regular["pace"] = regular.possessions_prior / regular.prior_games.replace(0, pd.NA)

    mature = regular[regular.prior_games >= 20]
    print()
    print("    valores de cordura (equipos con ≥20 partidos previos):")
    for label, series in (
        ("ORtg", mature.ortg),
        ("DRtg", mature.drtg),
        ("Pace", mature.pace),
    ):
        p05, p50, p95 = series.quantile([0.05, 0.50, 0.95])
        print(f"      {label}   p5={p05:6.1f}   mediana={p50:6.1f}   p95={p95:6.1f}")

    _check(
        "ORtg en rango realista (100-125)",
        100 <= mature.ortg.median() <= 125,
        f"mediana {mature.ortg.median():.1f} pts por 100 posesiones",
    )
    _check(
        "Pace en rango realista (90-110)",
        90 <= mature.pace.median() <= 110,
        f"mediana {mature.pace.median():.1f} posesiones por partido",
    )

    # ── Señal: ¿discrimina un modelo sin ningún ajuste? ──────────────────
    home = regular[regular.team_home_away == "home"].set_index("game_id")
    away = regular[regular.team_home_away == "away"].set_index("game_id")

    frame = pd.DataFrame(
        {
            "margin": home.team_score - home.opponent_team_score,
            "game_total": home.team_score + home.opponent_team_score,
            "net_home": home.ortg - home.drtg,
            "net_away": away.ortg - away.drtg,
            "ortg_home": home.ortg,
            "ortg_away": away.ortg,
            "drtg_home": home.drtg,
            "drtg_away": away.drtg,
            "pace_home": home.pace,
            "pace_away": away.pace,
            "games_home": home.prior_games,
            "games_away": away.prior_games,
        }
    ).dropna()
    frame = frame[
        (frame.games_home >= _MIN_PRIOR_GAMES)
        & (frame.games_away >= _MIN_PRIOR_GAMES)
    ]

    frame["net_diff"] = frame.net_home - frame.net_away
    expected_pace = (frame.pace_home + frame.pace_away) / 2
    frame["expected_total"] = (
        expected_pace
        * (
            (frame.ortg_home + frame.drtg_away) / 2
            + (frame.ortg_away + frame.drtg_home) / 2
        )
        / 100
    )

    _section(
        f"Señal del modelo ingenuo  (n={len(frame):,}, ≥{_MIN_PRIOR_GAMES} "
        f"partidos previos)"
    )
    print(
        f"    corr(diferencial de net rating, margen real) = "
        f"{frame.net_diff.corr(frame.margin):+.3f}"
    )
    print(
        f"    corr(total esperado, total real)             = "
        f"{frame.expected_total.corr(frame.game_total):+.3f}"
    )

    slope, intercept = _fit_line(frame.net_diff, frame.margin)
    residual = frame.margin - (slope * frame.net_diff + intercept)
    naive_sigma = float(residual.std())
    market_sigma = float(settled.ats_edge.std())

    print()
    print(
        f"    margen ≈ {slope:.3f} × net_diff + {intercept:.2f}"
        f"   ← la constante es la ventaja de local medida"
    )
    print(f"    sigma del modelo ingenuo  = {naive_sigma:6.2f} pts")
    print(f"    sigma de la línea cierre  = {market_sigma:6.2f} pts")
    print(
        f"    brecha a cerrar           = {naive_sigma - market_sigma:6.2f} pts "
        f"({(naive_sigma / market_sigma - 1) * 100:+.1f}%)"
    )

    print()
    _check(
        "los ratings llevan señal",
        abs(frame.net_diff.corr(frame.margin)) > 0.25,
        f"correlación {frame.net_diff.corr(frame.margin):+.3f} con el margen",
    )
    # Un modelo sin un solo ajuste que ya quede a <15% del cierre significa
    # que el margen de mejora está en el modelado, no en los datos. Más de
    # eso y la señal disponible probablemente no alcanza.
    _check(
        "brecha abordable con modelado",
        naive_sigma / market_sigma < 1.15,
        f"el modelo ingenuo está {(naive_sigma / market_sigma - 1) * 100:.1f}% "
        f"por detrás del cierre",
    )


def _fit_line(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """
    Regresión lineal simple sin depender de numpy.polyfit.

    Mantiene el script sobre pandas puro, que es la única dependencia
    pesada que ya está garantizada en el entorno.
    """
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    covariance = float(((x - x_mean) * (y - y_mean)).sum())
    variance = float(((x - x_mean) ** 2).sum())
    slope = covariance / variance if variance else 0.0
    return slope, y_mean - slope * x_mean


# ── Veredicto ─────────────────────────────────────────────────────────────────

def verdict() -> int:
    _header("VEREDICTO")

    failed = [check for check in _checks if not check.passed]
    print()
    print(f"    comprobaciones : {len(_checks) - len(failed)}/{len(_checks)} OK")

    if not failed:
        print()
        print("    Las tres fuentes existen, cruzan y tienen calidad suficiente.")
        print("    El Bloque 12 puede construirse sobre ellas.")
        return 0

    print()
    for check in failed:
        print(f"    FALLA  {check.name}: {check.detail}")

    # La ausencia de precios es una limitación conocida y documentada, no un
    # fallo del entorno. No debe tumbar el código de salida por sí sola.
    blocking = [check for check in failed if check.name != "precios por casa"]
    if not blocking:
        print()
        print("    Única falla: no hay precios. Es una limitación conocida —")
        print("    el backtest de spread/total sigue siendo viable al -110.")
        return 0

    print()
    print("    Hay fallas que bloquean. Revisar antes de construir el plugin.")
    return 1


# ── Entrada ───────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audita las fuentes de datos para el plugin NBA."
    )
    parser.add_argument(
        "--cache-dir",
        default="data/nba/raw",
        help="Dónde cachear las descargas (por defecto data/nba/raw).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Vuelve a descargar aunque exista caché.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    _header("AUDITORÍA DE FUENTES — NBA  (tarea 12.0)")
    print()
    print("  Fuente : sportsdataverse/hoopR-nba-data  (CC BY 4.0)")
    print("  Origen : ESPN (resultados) + The Odds API (consenso de cierre)")
    print(f"  Caché  : {cache_dir.resolve()}")

    try:
        closing_raw, archive, team_box_raw = load_sources(cache_dir, args.refresh)
    except Exception as error:
        print()
        print(f"  ERROR descargando las fuentes: {error}")
        return 1

    team_box = audit_team_box(team_box_raw)
    closing = audit_closing_lines(closing_raw)
    cross_validate_odds(closing, archive, team_box)
    settled = audit_settlement(closing, team_box)
    audit_efficiency(team_box, settled)
    return verdict()


if __name__ == "__main__":
    raise SystemExit(main())