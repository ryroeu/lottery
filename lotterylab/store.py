"""Data store — immutable raw snapshots, derived canonical cache.

Source CSVs live under ``data/raw/<game>/<snapshot>.csv`` and are NEVER overwritten
(the old scripts downloaded and clobbered the committed file in place, destroying
provenance). ``load_canonical`` reads the newest snapshot through the game's
adapter, filters to the current matrix era, validates, and returns a tidy frame.
"""

from __future__ import annotations

import csv as _csv
import datetime as _dt
import glob
import hashlib
import io
import os
import zipfile

import pandas as pd
import requests

from . import adapters
from .games import get
from .schema import Draw, draws_to_frame
from .validate import InvalidTicket, validate_ticket

# Repo root = two levels up from this file (lotterylab/store.py -> repo/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
CACHE_DIR = os.path.join(ROOT, "data", "cache")

# Project-wide floor: only ever use draws from 2018-01-01 onward. Combined with
# each game's matrix-change date (whichever is later) so the uniform
# "2018 to present" cut never re-introduces a stale main or special-ball pool.
MIN_DATE = _dt.date(2018, 1, 1)

_UA = {"User-Agent": "Mozilla/5.0 (compatible; lotterylab/0.1)"}
SNAPSHOT_PARSE_ERRORS = (
    KeyError,
    OSError,
    TypeError,
    UnicodeDecodeError,
    ValueError,
    pd.errors.EmptyDataError,
    pd.errors.ParserError,
)

# Single-CSV download URLs. fetch_raw writes a new timestamped snapshot and never
# touches existing ones. (EuroMillions is multi-source — see EUROMILLIONS_FDJ_SOURCES.)
FETCH_URLS = {
    "powerball": "https://data.ny.gov/api/views/d6yy-54nr/rows.csv?accessType=DOWNLOAD",
    "megamillions": "https://data.ny.gov/api/views/5xaw-6ayf/rows.csv?accessType=DOWNLOAD",
}

# EuroMillions: FDJ (the French operator) publishes the official history split by
# rule-era (the star pool changed over the years). All eras are served through the
# draw-info API; the current-era document is updated in place. Together these cover
# 2016-09 to present, which the 2018 floor trims to 2018+.
EUROMILLIONS_FDJ_SOURCES = [
    (
        "https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/"
        "documentations/1a2b3c4d-9876-4562-b3fc-2c963f66afc6"
    ),
    (
        "https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/"
        "documentations/1a2b3c4d-9876-4562-b3fc-2c963f66afd6"
    ),
    (
        "https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/"
        "documentations/1a2b3c4d-9876-4562-b3fc-2c963f66afe6"
    ),
]

# EuroDreams launched 2023-11 with one stable matrix, so FDJ serves it as a single
# live file via the draw-info API (boule_1..6 + numero_dream).
EURODREAMS_FDJ_SOURCES = [
    (
        "https://www.sto.api.fdj.fr/anonymous/service-draw-info/v3/"
        "documentations/1a2b3c4d-9876-4562-b3fc-2c963f66afa5"
    ),
]


def all_snapshots(game: str) -> list[str]:
    """All raw snapshots for a game, newest filename first."""
    return sorted(glob.glob(os.path.join(RAW_DIR, game, "*.csv")), reverse=True)


def newest_snapshot(game: str) -> str | None:
    """Newest raw snapshot path for a game, if one exists."""
    snaps = all_snapshots(game)
    return snaps[0] if snaps else None


def _count_parsable_draws(game: str, path: str) -> int:
    """How many draws this snapshot yields through the adapter (0 if unparsable)."""
    try:
        return len(adapters.parse(game, pd.read_csv(path)))
    except SNAPSHOT_PARSE_ERRORS:
        return 0


def _parse_fdj(content: bytes, n_main: int, special_cols: list[str]) -> list[dict]:
    """Parse one FDJ draw file (zip-of-CSV or raw CSV) to draw dicts.

    FDJ format: semicolon-delimited, latin-1, columns ``boule_1..n_main`` plus the
    given ``special_cols`` (etoile_1/etoile_2 for EuroMillions, numero_dream for
    EuroDreams) and ``date_de_tirage`` as DD/MM/YYYY. ``index_col=False`` because
    some files carry a trailing ';' that would otherwise make pandas adopt the first
    column as the index and silently shift every column left. Returns [] if the
    expected columns are absent.
    """
    raw = content
    if content[:2] == b"PK":  # zip magic
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            raw = archive.read(archive.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), sep=";", encoding="latin-1", index_col=False)
    main_cols = [f"boule_{i}" for i in range(1, n_main + 1)]
    need = ["date_de_tirage"] + main_cols + special_cols
    if not all(c in df.columns for c in need):
        return []
    rows = []
    for _, r in df.iterrows():
        try:
            date = _dt.datetime.strptime(
                str(r["date_de_tirage"]).strip(), "%d/%m/%Y"
            ).date()
            main = sorted(int(r[c]) for c in main_cols)
            special = sorted(int(r[c]) for c in special_cols)
        except (ValueError, TypeError):
            continue
        rows.append(
            {
                "date": date,
                "main": main,
                "special": special,
                "draw": str(r.get("annee_numero_de_tirage", "")).strip(),
            }
        )
    return rows


def _download_fdj(sources: list[str], n_main: int, special_cols: list[str]) -> dict:
    """Download FDJ sources and merge their draws, de-duped by draw date."""
    by_date: dict[_dt.date, dict] = {}
    for url in sources:
        resp = requests.get(url, headers=_UA, timeout=60)
        resp.raise_for_status()
        rows = _parse_fdj(resp.content, n_main, special_cols)
        if not rows:
            raise ValueError(
                f"FDJ source produced no usable draws; its format may have changed: {url}"
            )
        for row in rows:
            by_date[row["date"]] = row
    return by_date


def _write_fdj_snapshot(
    game: str, today: _dt.date, header: list[str], by_date: dict
) -> str:
    """Write merged FDJ draws to one combined snapshot in the game's adapter layout."""
    if not by_date:
        raise ValueError(
            f"FDJ {game} fetch produced no draws — the FDJ sources may have changed."
        )
    rows = sorted(by_date.values(), key=lambda r: r["date"])
    buffer = io.StringIO(newline="")
    writer = _csv.writer(buffer)
    writer.writerow(header)
    for row in rows:
        writer.writerow(
            [row["date"].isoformat(), *row["main"], *row["special"], row["draw"]]
        )
    content = buffer.getvalue().encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()[:8]
    timestamp = _dt.datetime.now().strftime("%H%M%S")
    os.makedirs(os.path.join(RAW_DIR, game), exist_ok=True)
    filename = f"{today.isoformat()}__zz-{timestamp}__fdj-{digest}.csv"
    path = os.path.join(RAW_DIR, game, filename)
    try:
        with open(path, "xb") as output:
            output.write(content)
    except FileExistsError:
        # Same content fetched within the same second: return the immutable file.
        pass
    return path


def fetch_euromillions(*, today: _dt.date | None = None) -> str:
    """Download FDJ's multi-era EuroMillions history, merge, write one snapshot.

    Written in the national-lottery column layout so the existing euromillions
    adapter reads it with no special-casing (the trailing draw_id column is ignored
    by games whose adapter doesn't use it).
    """
    today = today or _dt.date.today()
    by_date = _download_fdj(EUROMILLIONS_FDJ_SOURCES, 5, ["etoile_1", "etoile_2"])
    header = [
        "DrawDate",
        "Ball 1",
        "Ball 2",
        "Ball 3",
        "Ball 4",
        "Ball 5",
        "Lucky Star 1",
        "Lucky Star 2",
        "DrawNumber",
    ]
    return _write_fdj_snapshot("euromillions", today, header, by_date)


def fetch_eurodreams(*, today: _dt.date | None = None) -> str:
    """Download FDJ's EuroDreams history and write one snapshot in Kaggle layout."""
    today = today or _dt.date.today()
    by_date = _download_fdj(EURODREAMS_FDJ_SOURCES, 6, ["numero_dream"])
    # trailing "DrawNumber" matches the writer's draw_id column; the adapter reads
    # by name and ignores it.
    header = [
        "Date",
        "Number 1",
        "Number 2",
        "Number 3",
        "Number 4",
        "Number 5",
        "Number 6",
        "Dream Number",
        "DrawNumber",
    ]
    return _write_fdj_snapshot("eurodreams", today, header, by_date)


def fetch_raw(
    game: str, *, today: _dt.date | None = None, validate: bool = True
) -> str:
    """Download a fresh snapshot to data/raw/<game>/. Returns the new path.

    Network side-effect; not exercised by the offline test suite. With
    ``validate`` (default), the result must parse to >= 1 draw or it is discarded
    and an error is raised — so a changed/expired source can never leave a garbage
    snapshot behind (as the old EuroMillions CSV endpoint did when it switched to
    an XML latest-draw-only feed).
    """
    today = today or _dt.date.today()
    newly_written = True
    if game == "euromillions":
        path = fetch_euromillions(today=today)
    elif game == "eurodreams":
        path = fetch_eurodreams(today=today)
    elif game in FETCH_URLS:
        resp = requests.get(FETCH_URLS[game], headers=_UA, timeout=60)
        resp.raise_for_status()
        digest = hashlib.sha256(resp.content).hexdigest()[:8]
        os.makedirs(os.path.join(RAW_DIR, game), exist_ok=True)
        path = os.path.join(RAW_DIR, game, f"{today.isoformat()}__{digest}.csv")
        newly_written = not os.path.exists(path)
        if newly_written:
            with open(path, "wb") as f:
                f.write(resp.content)
    else:
        raise ValueError(f"No fetch source configured for {game!r}.")

    if validate and _count_parsable_draws(game, path) == 0:
        if newly_written and os.path.exists(path):
            os.remove(path)
        raise ValueError(
            f"Fetched {game!r} data did not parse to any draws — the source format "
            f"may have changed. Snapshot discarded."
        )
    return path


def load_canonical(
    game: str,
    *,
    modern_only: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """Load the newest snapshot as a validated, tidy, chronologically-sorted frame.

    ``modern_only`` filters to draws on/after the 2018 floor / current matrix and
    drops any row not valid under the current matrix (a safety net for mixed-era
    files). Snapshots are tried newest-first; a snapshot that fails to parse or
    yields zero draws is skipped, so one bad fetch never breaks the pipeline.
    """
    spec = get(game)
    snapshots = all_snapshots(game)
    if not snapshots:
        raise FileNotFoundError(
            f"No raw snapshot for {game!r} in {os.path.join(RAW_DIR, game)}. "
            f"Run fetch_raw({game!r}) or add a CSV."
        )

    # Effective cutoff = the later of the 2018 floor and either pool's last change.
    cutoff = None
    if modern_only:
        cutoff = MIN_DATE
        if spec.matrix_since and spec.matrix_since > cutoff:
            cutoff = spec.matrix_since

    for idx, path in enumerate(snapshots):
        try:
            draws = adapters.parse(game, pd.read_csv(path))
        except SNAPSHOT_PARSE_ERRORS:
            draws = []

        kept: list[Draw] = []
        dropped_era = dropped_invalid = 0
        for d in draws:
            if cutoff and d.date < cutoff:
                dropped_era += 1
                continue
            try:
                validate_ticket(d.main, d.special, spec)
            except InvalidTicket:
                dropped_invalid += 1
                continue
            kept.append(d)

        if not kept:
            continue  # unparsable / empty snapshot — fall back to an older one

        if verbose:
            note = (
                "" if idx == 0 else f" [fell back past {idx} newer broken snapshot(s)]"
            )
            print(
                f"[{game}] loaded {len(kept)} draws from {os.path.basename(path)}"
                f"{note} (dropped {dropped_era} pre-{cutoff} + "
                f"{dropped_invalid} invalid-under-current-matrix)"
            )
        return draws_to_frame(kept, spec)

    raise ValueError(
        f"No usable snapshot for {game!r}: all {len(snapshots)} file(s) failed to "
        f"parse into draws. The source format may have changed."
    )


def write_cache(game: str, df: pd.DataFrame) -> str:
    """Write a derived canonical cache CSV and return its path."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{game}.csv")
    df.to_csv(path, index=False)
    return path
