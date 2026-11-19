"""TMDB enrichment: look up every Netflix title on TMDB and save one row of details per title.

Runs after ingestion in the pipeline, and is the only step that talks to TMDB:
    ingest -> tmdb (new titles only) -> prepare -> train -> ...

Output: data/external/tmdb_titles.parquet, one row per ``title_key`` (built with
``features.add_title_key``, so it joins straight onto the weekly Netflix rows).

Matching rules (from notebooks/02_tmdb_exploration.ipynb)
----------------------------------------------------------
Search TMDB by title, then pick a result, trying in order:
  1. exact title (ignoring punctuation, spaces and case), released before it first charted,
     in the list's language -> films and new TV shows (Season 1, Limited Series, specials):
     the most recent such release (remakes share titles), ignoring results under 25% as
     popular as the top one (obscure shorts with famous titles); later TV seasons: the most popular
  2. the same, but in another language -> low confidence
  3. any result released before it first charted, in the list's language -> the most popular
  4. exact title with an unknown release date, in the list's language -> low confidence
  Anything else is rejected (left unmatched) rather than guessed.
"Released before" allows up to 14 days of slack, because TMDB's date is often the
home-country premiere, which can differ from Netflix's global release by a few days.
"The list's language": Netflix splits its lists by original language, so a title on an
English list should be an English-language title on TMDB (and vice versa). This stops
e.g. the live-action ONE PIECE (English list) matching the Japanese anime.

For a new show (Season 1, Limited Series, special), a loose match (rule 3) that came out
more than a year before the title charted is kept but marked low confidence: Netflix
originals are usually new, and an old show with a similar name is usually a different show.

TV titles with no season (stand-up specials, one-off events) that TMDB doesn't list as
TV are searched again as films, because TMDB files most stand-up specials as films.

Manual fixes
------------
data/external/tmdb_overrides.csv (committed to Git, not DVC) fixes matches by hand:
one row per title_key, with the right tmdb_id, or a blank tmdb_id for "no TMDB match".
Overridden titles get match_confidence "manual" and are re-looked-up on every run.

Runtime check
-------------
``runtime_ratio`` = TMDB runtime / Netflix runtime, and ``runtime_mismatch`` = 1 when
they differ 2x or more. A mismatch means "don't use TMDB's runtime in place of Netflix's":
either the match is wrong, or the show releases weekly, so Netflix's runtime only counts
the episodes out so far while TMDB counts the whole season.

Seasons
-------
TV details come from the TMDB season. The season number is read from ``season_title``
("Season 2", "Stranger Things 4", "Limited Series"...). "Part N" and other named labels
are matched against TMDB's season names; if no name matches, season details stay blank
(by design: guessing by air date can be confidently wrong). Weekly dated episodes
(e.g. "Raw: April 13, 2026") are skipped: they're live episodes, not seasons.

What's saved
------------
Only stable facts: genres (one shared list for films and TV), a few keyword flags,
US age-rating group, TV type, Netflix-original flag, language, runtime, episode
count, release date. Popularity, ratings and vote counts are NEVER saved: they're
today's values, so as features they would leak the future into past weeks.

Every TMDB response is cached under data/external/tmdb_cache/, so reruns and weekly
updates only call TMDB for titles it hasn't seen.

Usage (from the repo root; needs TMDB_API_KEY in .env):
    python -m src.tmdb --limit 50          # trial run on 50 titles
    python -m src.tmdb                     # look up every title not already in the table
    python -m src.tmdb --retry-unmatched   # also retry titles that failed before
    python -m src.tmdb --rebuild           # re-match everything (after changing the rules)

This product uses the TMDB API but is not endorsed or certified by TMDB.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa
import requests
from dotenv import load_dotenv

from src.features import add_title_key
from src.ingest import REPO_ROOT, latest_snapshot, load_raw

logger = logging.getLogger(__name__)

EXTERNAL_DIR = REPO_ROOT / "data" / "external"
TITLES_PATH = EXTERNAL_DIR / "tmdb_titles.parquet"
OVERRIDES_PATH = EXTERNAL_DIR / "tmdb_overrides.csv"
CACHE_DIR = EXTERNAL_DIR / "tmdb_cache"
BASE_URL = "https://api.themoviedb.org/3"

RELEASE_TOLERANCE_DAYS = 14  # TMDB date may be a few days after Netflix's global release
RUNTIME_MISMATCH_RATIO = 2.0  # TMDB and Netflix runtimes this far apart are flagged
# Season patterns that mean "a new show": pick the most recent exact-title show, not the most popular
NEW_SHOW_PATTERNS = {"Limited series", "Same as show title", "no season title"}
NEW_SHOW_MAX_LOOSE_AGE_DAYS = 365  # a loose match for a new show older than this is low confidence
MIN_RELATIVE_POPULARITY = 0.25  # "most recent" ignores results under 25% as popular as the top one
RULE_EXACT = "1 exact title, released before"
RULE_EXACT_OTHER_LANGUAGE = "2 exact title, other language"
RULE_LOOSE = "3 released before"
RULE_DATE_UNKNOWN = "4 exact title, date unknown"
MAX_RUNTIME_HOURS = 1_000  # sanity limit for one title's total runtime

# --- Shared genre list -------------------------------------------------------
# TMDB uses different genre names for films and TV; map both onto one list.
GENRE_MAP = {
    "Action": "action_adventure",
    "Adventure": "action_adventure",
    "Action & Adventure": "action_adventure",
    "Western": "action_adventure",
    "Animation": "animation",
    "Comedy": "comedy",
    "Crime": "crime",
    "Documentary": "documentary",
    "News": "documentary",
    "Drama": "drama",
    "Soap": "drama",
    "Family": "family_kids",
    "Kids": "family_kids",
    "Fantasy": "fantasy_scifi",
    "Science Fiction": "fantasy_scifi",
    "Sci-Fi & Fantasy": "fantasy_scifi",
    "History": "history_war",
    "War": "history_war",
    "War & Politics": "history_war",
    "Horror": "horror",
    "Music": "music",
    "Mystery": "mystery",
    "Romance": "romance",
    "Thriller": "thriller",
    "Reality": "reality_talk",
    "Talk": "reality_talk",
}
GENRES = sorted(set(GENRE_MAP.values()))
GENRE_COLUMNS = [f"genre_{g}" for g in GENRES]

# --- Keyword flags -----------------------------------------------------------
# TMDB has thousands of keywords; only a few become yes/no flags.
KEYWORD_FLAGS = {
    "kw_christmas": lambda k: "christmas" in k or k == "holiday",
    "kw_anime": lambda k: k == "anime",
    "kw_true_story": lambda k: k in {"based on true story", "true crime", "biography"},
    "kw_based_on_book": lambda k: k in {"based on novel or book", "based on book"},
}
KEYWORD_COLUMNS = list(KEYWORD_FLAGS)

# --- US age ratings -> groups ------------------------------------------------
RATING_GROUPS = {
    "G": "kids",
    "TV-Y": "kids",
    "TV-Y7": "kids",
    "TV-Y7-FV": "kids",
    "TV-G": "kids",
    "PG": "general",
    "TV-PG": "general",
    "PG-13": "teen",
    "TV-14": "teen",
    "R": "mature",
    "NC-17": "mature",
    "TV-MA": "mature",
}

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


# =============================================================================
# Season titles
# =============================================================================


def parse_season(show_title: str, season_title) -> tuple[int | None, str]:
    """Return (season_number, pattern). season_number is None when the title alone can't tell."""
    if pd.isna(season_title) or str(season_title).strip() in ("", "N/A"):
        return None, "no season title"
    s = str(season_title)
    rest = s[len(show_title) :].strip(" :-") if s.startswith(show_title) else s

    if re.search(rf"\b(?:{MONTHS}) \d{{1,2}}, \d{{4}}", s):
        return None, "Dated episode"
    patterns = [
        (r"\bSeason (\d+)", "Season N"),
        (r"\bPart (\d+)", "Part N"),
        (r"\b(?:Volume|Vol\.) ?(\d+)", "Volume N"),
        (r"\bBook (\d+)", "Book N"),
        (r"\bChapter (\d+)", "Chapter N"),
    ]
    for regex, name in patterns:
        match = re.search(regex, s, flags=re.IGNORECASE)
        if match:
            if name == "Part N":
                return None, name  # parts are usually halves of one TMDB season: match by name instead
            return int(match.group(1)), name
    if re.search(r"Limited Series|Miniseries|Mini-Series", s, flags=re.IGNORECASE):
        return 1, "Limited series"
    if rest == "":
        return 1, "Same as show title"
    if re.fullmatch(r"\d{1,2}", rest):
        return int(rest), "Show name + number"  # e.g. "Stranger Things 4"
    return None, "other"


def season_label(show_title: str, season_title) -> str:
    """The part of the season title after the show's name, e.g. "The Final Season Part 3"."""
    if pd.isna(season_title):
        return ""
    s = str(season_title)
    return s[len(show_title) :].strip(" :-") if s.startswith(show_title) else s


# =============================================================================
# Matching
# =============================================================================


def normalize(title) -> str:
    """Lowercase and drop punctuation and spaces: "Spider-Man" == "Spiderman", "&" == "and"."""
    if not isinstance(title, str):
        return ""
    return re.sub(r"[^\w]", "", title.casefold().replace("&", "and"))


def same_title(a, b) -> bool:
    return normalize(a) != "" and normalize(a) == normalize(b)


def candidates_from_search(result: dict | None, is_tv: bool, first_week: pd.Timestamp) -> pd.DataFrame:
    """Search results as a table, with days between release and first charting (negative = after)."""
    rows = []
    for r in (result or {}).get("results", []):
        date = pd.to_datetime(r.get("first_air_date" if is_tv else "release_date"), errors="coerce")
        rows.append(
            {
                "tmdb_id": r["id"],
                "name": r.get("name") or r.get("title"),
                "original_name": r.get("original_name") or r.get("original_title"),
                "date": date,
                "days_before_charting": (first_week - date).days if pd.notna(date) else np.nan,
                "original_language": r.get("original_language"),
                "popularity": r.get("popularity") or 0.0,  # used to choose between results only
            }
        )
    return pd.DataFrame(rows)


def is_new_show(season_number, season_pattern) -> bool:
    """Season 1, a limited series or a special: the show is probably new, not an old famous one."""
    return season_pattern in NEW_SHOW_PATTERNS or (
        season_pattern in ("Season N", "Show name + number") and season_number == 1
    )


def pick_match(
    show_title: str,
    is_tv: bool,
    cands: pd.DataFrame,
    english: bool | None = None,
    prefer_recent: bool = False,
) -> tuple[pd.Series | None, str, str | None]:
    """Choose a search result. Returns (row or None, rule, confidence).

    ``english``: True/False when the title is on an English/non-English list (None = don't check).
    ``prefer_recent``: for TV, break ties between exact titles by recency instead of popularity.
    """
    if cands.empty:
        return None, "no results", None
    c = cands.copy()
    c["exact"] = [
        same_title(show_title, n) or same_title(show_title, o)
        for n, o in zip(c["name"], c["original_name"], strict=True)
    ]
    days = c["days_before_charting"]
    c["released_before"] = days.notna() & (days >= -RELEASE_TOLERANCE_DAYS)
    lang = c["original_language"] if "original_language" in c else pd.Series(None, index=c.index, dtype=object)
    known = lang.notna() & (lang != "")
    same = ~known | ((lang == "en") == english) if english is not None else pd.Series(True, index=c.index)

    recent, popular = ("days_before_charting", True), ("popularity", False)
    exact_order = recent if (not is_tv or prefer_recent) else popular
    steps = [
        (c["exact"] & c["released_before"] & same, RULE_EXACT, exact_order, "high"),
        (c["exact"] & c["released_before"], RULE_EXACT_OTHER_LANGUAGE, exact_order, "low"),
        (c["released_before"] & same, RULE_LOOSE, popular, "medium"),
        (c["exact"] & days.isna() & same, RULE_DATE_UNKNOWN, popular, "low"),
    ]
    for mask, rule, (column, ascending), confidence in steps:
        if mask.any():
            pool = c[mask]
            if column == "days_before_charting":
                # "Most recent" only among real contenders: TMDB has many obscure short films that
                # share a famous title (e.g. a 3-minute "Prometheus" from 2021)
                pool = pool[pool["popularity"] >= MIN_RELATIVE_POPULARITY * pool["popularity"].max()]
            best = pool.sort_values(column, ascending=ascending).iloc[0]
            if rule == RULE_LOOSE and is_tv and prefer_recent and best["days_before_charting"] > (
                NEW_SHOW_MAX_LOOSE_AGE_DAYS
            ):
                return best, f"{RULE_LOOSE} (old show for a new title)", "low"
            return best, rule, confidence
    if c["released_before"].any():
        return None, "rejected: only results in another language", None
    return None, "rejected: only results released after charting", None


def match_season_by_name(label: str, seasons: list[dict]) -> int | None:
    """Season number whose TMDB name matches the label (either contains the other), if exactly one does."""
    target = normalize(label)
    if not target:
        return None
    named = [(s["season_number"], normalize(s.get("name"))) for s in seasons if s.get("season_number", 0) > 0]
    exact = [n for n, name in named if name and name == target]
    if len(exact) == 1:
        return exact[0]
    partial = [n for n, name in named if name and (name in target or target in name)]
    return partial[0] if len(partial) == 1 else None


# =============================================================================
# TMDB client
# =============================================================================


class TMDBClient:
    """Small TMDB client: caches every response on disk, retries rate limits, works from several threads."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_dir: Path = CACHE_DIR,
        session_factory: Callable[[], requests.Session] = requests.Session,
        pause: float = 0.05,
        max_retries: int = 3,
    ):
        load_dotenv(REPO_ROOT / ".env")
        api_key = api_key or os.getenv("TMDB_API_KEY")
        if not api_key:
            raise RuntimeError("TMDB_API_KEY not set: add it to .env at the repo root")
        self._bearer = api_key.startswith("eyJ")  # long "API Read Access Token" vs short "API Key"
        self._api_key = api_key
        self._session_factory = session_factory
        self._local = threading.local()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pause = pause
        self.max_retries = max_retries
        self.stats = {"network": 0, "cache": 0, "cache_write_failures": 0}
        self._lock = threading.Lock()
        self._request_locks: dict[str, threading.Lock] = {}
        for leftover in self.cache_dir.glob("*.tmp"):  # half-saved files from an interrupted run
            leftover.unlink(missing_ok=True)

    def _request_lock(self, cache_file: Path) -> threading.Lock:
        # One lock per distinct request: if two workers need the same response (e.g. several seasons
        # of the same show), the second waits and then reads the first one's cached copy.
        with self._lock:
            return self._request_locks.setdefault(cache_file.name, threading.Lock())

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):  # one session per thread
            session = self._session_factory()
            if self._bearer:
                session.headers["Authorization"] = f"Bearer {self._api_key}"
            self._local.session = session
        return self._local.session

    def _cache_file(self, path: str, params: dict) -> Path:
        request_id = path + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        readable = re.sub(r"[^A-Za-z0-9]+", "_", request_id)[:80]
        return self.cache_dir / f"{readable}_{hashlib.md5(request_id.encode()).hexdigest()[:12]}.json"

    def _count(self, key: str) -> None:
        with self._lock:
            self.stats[key] += 1

    def _read_cache(self, cache_file: Path) -> tuple[bool, dict | None]:
        """(found, data). A cache file that can't be read is treated as missing, never as an error."""
        try:
            return True, json.loads(cache_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False, None

    def _write_cache(self, cache_file: Path, data: dict | None) -> None:
        """Save a response. The cache only saves time, so a failed save is logged and skipped, never fatal.

        Windows refuses to replace a file that another program (OneDrive, antivirus) has open,
        so the rename is retried a few times first.
        """
        tmp = cache_file.with_suffix(f".{threading.get_ident()}.tmp")
        try:
            tmp.write_text(json.dumps(data), encoding="utf-8")
            for attempt in range(5):
                try:
                    tmp.replace(cache_file)  # write-then-rename, so a half-written file is never read
                    return
                except PermissionError:
                    time.sleep(0.1 * (attempt + 1))
            raise PermissionError(f"could not save {cache_file.name}")
        except OSError as err:
            self._count("cache_write_failures")
            logger.warning("Cache not saved (the lookup still worked): %s", err)
            tmp.unlink(missing_ok=True)

    def get(self, path: str, **params) -> dict | None:
        """GET a TMDB endpoint. Returns None for 404 (e.g. a season that doesn't exist)."""
        cache_file = self._cache_file(path, params)
        with self._request_lock(cache_file):
            if cache_file.exists():
                found, data = self._read_cache(cache_file)
                if found:
                    self._count("cache")
                    return data
            data = self._fetch(path, params)
            self._write_cache(cache_file, data)
            return data

    def _fetch(self, path: str, params: dict) -> dict | None:
        query = {**params} if self._bearer else {**params, "api_key": self._api_key}
        for attempt in range(self.max_retries + 1):
            response = self._session().get(f"{BASE_URL}{path}", params=query, timeout=30)
            self._count("network")
            time.sleep(self.pause)
            if response.status_code == 429 and attempt < self.max_retries:  # rate limited: wait and retry
                time.sleep(float(response.headers.get("Retry-After", 2)))
                continue
            if response.status_code >= 500 and attempt < self.max_retries:
                time.sleep(2**attempt)
                continue
            break

        if response.status_code == 404:
            return None
        response.raise_for_status()  # 401 = bad key: fail loudly
        return response.json()

    def check(self) -> None:
        """Fail fast on a bad key or no connection, before looking up thousands of titles."""
        self.get("/configuration")


# =============================================================================
# Extracting details
# =============================================================================


def _genre_flags(details: dict) -> dict:
    found = {GENRE_MAP[g["name"]] for g in details.get("genres", []) if g.get("name") in GENRE_MAP}
    return {f"genre_{g}": int(g in found) for g in GENRES}


def _keyword_flags(details: dict, is_tv: bool) -> dict:
    block = details.get("keywords") or {}
    names = [k.get("name", "").casefold() for k in block.get("results" if is_tv else "keywords", [])]
    return {col: int(any(test(k) for k in names)) for col, test in KEYWORD_FLAGS.items()}


def _us_rating(details: dict, is_tv: bool) -> str | None:
    if is_tv:
        for r in (details.get("content_ratings") or {}).get("results", []):
            if r.get("iso_3166_1") == "US" and r.get("rating"):
                return r["rating"]
    else:
        for r in (details.get("release_dates") or {}).get("results", []):
            if r.get("iso_3166_1") == "US":
                for d in r.get("release_dates", []):
                    if d.get("certification"):
                        return d["certification"]
    return None


def _is_netflix_original(details: dict, is_tv: bool) -> int:
    companies = details.get("networks" if is_tv else "production_companies", [])
    return int(any("netflix" in (c.get("name") or "").casefold() for c in companies))


def season_runtime(season: dict) -> tuple[float | None, int, int, float | None]:
    """(total hours, episode count, episodes missing a runtime, median episode minutes).

    Missing episode runtimes are filled with the median of the others; if more than half
    are missing, the total is left blank rather than guessed.
    """
    runtimes = [ep.get("runtime") for ep in season.get("episodes", [])]
    known = [r for r in runtimes if r]
    missing = len(runtimes) - len(known)
    if not known or missing > len(runtimes) / 2:
        return None, len(runtimes), missing, (float(np.median(known)) if known else None)
    median = float(np.median(known))
    total_minutes = sum(known) + missing * median
    return round(total_minutes / 60, 3), len(runtimes), missing, median


def _age_group(days) -> str:
    if days is None or pd.isna(days):
        return "unknown"
    if days < 60:
        return "new"
    return "recent" if days <= 365 else "catalog"


# =============================================================================
# Looking up one title
# =============================================================================


def _empty_record(title: pd.Series) -> dict:
    return {
        "title_key": title["title_key"],
        "category": title["category"],
        "show_title": title["show_title"],
        "season_title": title["season_title"],
        "is_tv": bool(title["is_tv"]),
        "first_week": title["first_week"],
        "netflix_runtime_h": title["netflix_runtime_h"],
        "lookup_status": None,
        "match_rule": None,
        "match_confidence": None,
        "tmdb_id": pd.NA,
        "tmdb_name": None,
        "original_language": None,
        "tmdb_type": None,
        "rating_group": "unknown",
        "is_netflix_original": pd.NA,
        **dict.fromkeys(GENRE_COLUMNS, pd.NA),
        **dict.fromkeys(KEYWORD_COLUMNS, pd.NA),
        "season_pattern": None,
        "season_number": pd.NA,
        "season_resolved": False,
        "episode_count": pd.NA,
        "episodes_missing_runtime": pd.NA,
        "avg_episode_minutes": np.nan,
        "tmdb_runtime_h": np.nan,
        "runtime_ratio": np.nan,
        "runtime_mismatch": pd.NA,
        "release_date": pd.NaT,
        "age_days_at_charting": pd.NA,
        "age_group": "unknown",
        "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _search(client: TMDBClient, kind: str, show: str, first_week, english: bool, prefer_recent: bool):
    """Search TMDB films or TV and pick a match; retry once without punctuation if nothing fits."""
    as_tv = kind == "tv"
    search = client.get(f"/search/{kind}", query=show, include_adult="false", language="en-US")
    best, rule, confidence = pick_match(
        show, as_tv, candidates_from_search(search, as_tv, first_week), english, prefer_recent
    )
    if best is None and re.search(r"[^\w\s]", show):
        # e.g. "Monsters vs. Aliens" -> "Monsters vs Aliens"
        plain = " ".join(re.sub(r"[^\w\s]", " ", show).split())
        search = client.get(f"/search/{kind}", query=plain, include_adult="false", language="en-US")
        best, rule, confidence = pick_match(
            show, as_tv, candidates_from_search(search, as_tv, first_week), english, prefer_recent
        )
    return best, rule, confidence


def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, dict]:
    """Hand-made fixes: {title_key: {"tmdb_id": int or None, "kind": "tv"/"movie", "note": str}}.

    A blank tmdb_id means "this title has no TMDB match". ``kind`` defaults to the title's
    own type (TV list -> tv); set it to "movie" for a TV special that TMDB files as a film.
    """
    path = Path(path)
    if not path.exists():
        return {}
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8", comment="#")
    missing = {"title_key", "tmdb_id"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if df["title_key"].duplicated().any():
        raise ValueError(f"{path} lists a title_key twice: {df.loc[df['title_key'].duplicated(), 'title_key'].tolist()}")
    overrides = {}
    for r in df.to_dict("records"):
        kind = (r.get("kind") or "").strip().lower() or None
        if kind not in (None, "tv", "movie"):
            raise ValueError(f"{path}: kind must be tv, movie or blank, got {kind!r} for {r['title_key']!r}")
        tmdb_id = r["tmdb_id"].strip()
        overrides[r["title_key"].strip()] = {
            "tmdb_id": int(tmdb_id) if tmdb_id else None,
            "kind": kind,
            "note": (r.get("note") or "").strip(),
        }
    return overrides


def lookup_title(client: TMDBClient, title: pd.Series, override: dict | None = None) -> dict:
    """Look up one title on TMDB and return its row for the titles table. Never raises for one bad title."""
    rec = _empty_record(title)
    is_tv, show = bool(title["is_tv"]), title["show_title"]
    english = "(English)" in str(title["category"])  # Netflix's lists are split by original language
    prefer_recent = False

    if is_tv:
        rec["season_number"], rec["season_pattern"] = parse_season(show, title["season_title"])
        if rec["season_pattern"] == "Dated episode":
            rec.update(lookup_status="skipped", match_rule="skipped: dated weekly episode")
            return rec
        prefer_recent = is_new_show(rec["season_number"], rec["season_pattern"])
        if rec["season_number"] is None:
            rec["season_number"] = pd.NA

    try:
        kind = "tv" if is_tv else "movie"
        if override is not None:
            if override["tmdb_id"] is None:
                rec.update(lookup_status="no_match", match_rule="override: no TMDB match")
                return rec
            kind = override["kind"] or kind
            best = pd.Series({"tmdb_id": override["tmdb_id"], "name": None, "date": pd.NaT})
            rule, confidence = "override", "manual"
        else:
            best, rule, confidence = _search(client, kind, show, title["first_week"], english, prefer_recent)
        if best is None and is_tv and rec["season_pattern"] == "no season title":
            # Stand-up and other one-off TV specials are usually filed on TMDB as films
            film_best, film_rule, film_confidence = _search(client, "movie", show, title["first_week"], english, False)
            if film_best is not None:
                kind, best, rule, confidence = "movie", film_best, f"{film_rule} (found as a film)", film_confidence
        rec["match_rule"] = rule
        if best is None:
            rec["lookup_status"] = "no_match"
            return rec

        as_tv = kind == "tv"
        tmdb_id = int(best["tmdb_id"])
        extras = "keywords,content_ratings" if as_tv else "keywords,release_dates"
        details = client.get(f"/{kind}/{tmdb_id}", append_to_response=extras, language="en-US")
        if details is None and override is not None:
            raise ValueError(f"override tmdb_id {tmdb_id} not found on TMDB as a {kind} (wrong id or kind?)")
        details = details or {}
        rating = _us_rating(details, as_tv)
        rec.update(
            lookup_status="matched",
            match_confidence=confidence,
            tmdb_id=tmdb_id,
            tmdb_name=details.get("name" if as_tv else "title") or best["name"],
            original_language=details.get("original_language"),
            tmdb_type=details.get("type") if as_tv else "Film",
            rating_group=RATING_GROUPS.get(rating, "unknown"),
            is_netflix_original=_is_netflix_original(details, as_tv),
            **_genre_flags(details),
            **_keyword_flags(details, as_tv),
        )

        if as_tv:
            number = rec["season_number"]
            if pd.isna(number):
                label = season_label(show, title["season_title"])
                number = match_season_by_name(label, details.get("seasons", []))
            if number is not None and not pd.isna(number):
                season = client.get(f"/tv/{tmdb_id}/season/{int(number)}", language="en-US")
                if season:
                    total, episodes, missing, avg = season_runtime(season)
                    rec.update(
                        season_number=int(number),
                        season_resolved=True,
                        episode_count=episodes,
                        episodes_missing_runtime=missing,
                        avg_episode_minutes=avg if avg is not None else np.nan,
                        tmdb_runtime_h=total if total is not None else np.nan,
                        release_date=pd.to_datetime(season.get("air_date"), errors="coerce"),
                    )
        else:
            minutes = details.get("runtime")
            rec["tmdb_runtime_h"] = round(minutes / 60, 3) if minutes else np.nan
            rec["release_date"] = pd.to_datetime(details.get("release_date"), errors="coerce")
            if pd.isna(rec["release_date"]):
                rec["release_date"] = best["date"]

        if pd.notna(rec["release_date"]):
            days = (title["first_week"] - rec["release_date"]).days
            if days >= -RELEASE_TOLERANCE_DAYS:  # small negatives = date conventions, not errors
                rec["age_days_at_charting"] = max(days, 0)
        rec["age_group"] = _age_group(rec["age_days_at_charting"])
        return rec

    except requests.HTTPError as err:
        if err.response is not None and err.response.status_code in (401, 403):
            raise  # bad key: stop the whole run
        rec.update(lookup_status="error", match_rule=f"error: {err}")
        return rec
    except (requests.RequestException, ValueError, KeyError) as err:
        rec.update(lookup_status="error", match_rule=f"error: {type(err).__name__}: {err}")
        return rec


# =============================================================================
# The titles table
# =============================================================================


def titles_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per title_key, with the week it first charted and Netflix's largest runtime."""
    df = add_title_key(raw)
    out = (
        df.groupby("title_key")
        .agg(
            category=("category", "first"),
            show_title=("show_title", "first"),
            season_title=("season_title", "first"),
            first_week=("week", "min"),
            netflix_runtime_h=("runtime", lambda s: s[s > 0].max()),  # full season once all episodes are out
        )
        .reset_index()
    )
    out["is_tv"] = out["category"].str.startswith("TV")
    return out


_nullable_flag = pa.Column("Int64", pa.Check.isin([0, 1]), nullable=True)
titles_schema = pa.DataFrameSchema(
    {
        "title_key": pa.Column(str, unique=True),
        "first_week": pa.Column("datetime64[ns]", coerce=True),
        "lookup_status": pa.Column(str, pa.Check.isin(["matched", "no_match", "skipped", "error"])),
        "match_confidence": pa.Column(str, pa.Check.isin(["high", "medium", "low", "manual"]), nullable=True),
        "tmdb_id": pa.Column("Int64", nullable=True),
        "rating_group": pa.Column(str, pa.Check.isin([*sorted(set(RATING_GROUPS.values())), "unknown"])),
        "is_netflix_original": _nullable_flag,
        **dict.fromkeys(GENRE_COLUMNS + KEYWORD_COLUMNS, _nullable_flag),
        "season_number": pa.Column("Int64", pa.Check.ge(1), nullable=True),
        "episode_count": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        # Long telenovelas can be 300+ hours on TMDB; 1,000 still catches unit mistakes (minutes saved as hours)
        "tmdb_runtime_h": pa.Column(
            float, pa.Check.in_range(0, MAX_RUNTIME_HOURS, include_min=False), nullable=True
        ),
        "runtime_mismatch": _nullable_flag,
        "age_days_at_charting": pa.Column("Int64", pa.Check.ge(0), nullable=True),
        "age_group": pa.Column(str, pa.Check.isin(["new", "recent", "catalog", "unknown"])),
    },
    checks=[
        pa.Check(
            lambda df: df.loc[df["lookup_status"] == "matched", "tmdb_id"].notna().all(),
            name="matched_titles_have_tmdb_id",
        )
    ],
    strict=False,
)

_INT_COLUMNS = [
    "tmdb_id",
    "is_netflix_original",
    *GENRE_COLUMNS,
    *KEYWORD_COLUMNS,
    "season_number",
    "episode_count",
    "episodes_missing_runtime",
    "age_days_at_charting",
    "runtime_mismatch",
]


TABLE_COLUMNS = list(
    _empty_record(
        pd.Series(
            {"title_key": "", "category": "", "show_title": "", "season_title": None, "is_tv": False,
             "first_week": pd.NaT, "netflix_runtime_h": np.nan}
        )
    )
)


def _to_table(records: list[dict]) -> pd.DataFrame:
    """Records -> table with consistent column types (nullable integers, floats, dates)."""
    table = pd.DataFrame(records, columns=TABLE_COLUMNS)
    for col in _INT_COLUMNS:
        table[col] = pd.array(table[col].astype("Float64"), dtype="Int64")
    for col in ["avg_episode_minutes", "tmdb_runtime_h", "netflix_runtime_h"]:
        table[col] = pd.to_numeric(table[col], errors="coerce").astype(float)
    # Recomputed for every row, so tables saved before these columns existed get them too
    ratio = table["tmdb_runtime_h"] / table["netflix_runtime_h"]
    table["runtime_ratio"] = ratio.round(3)
    far_apart = (ratio >= RUNTIME_MISMATCH_RATIO) | (ratio <= 1 / RUNTIME_MISMATCH_RATIO)
    table["runtime_mismatch"] = pd.array(far_apart.astype(float).where(ratio.notna()), dtype="Float64").astype("Int64")
    table["release_date"] = pd.to_datetime(table["release_date"], errors="coerce")
    table["first_week"] = pd.to_datetime(table["first_week"])
    table["season_resolved"] = table["season_resolved"].astype(bool)
    return table


def update_titles_table(
    raw: pd.DataFrame,
    existing: pd.DataFrame | None = None,
    client: TMDBClient | None = None,
    limit: int | None = None,
    retry_unmatched: bool = False,
    workers: int = 8,
    validate: bool = True,
    overrides: dict[str, dict] | None = None,
) -> pd.DataFrame:
    """Look up titles not yet in ``existing`` (and, optionally, earlier failures); return the full table.

    Titles in ``overrides`` (see ``load_overrides``) are always looked up again, so an edit
    to the overrides file takes effect on the next run.
    ``validate=False`` returns the table without the schema check, so the caller can save it
    for inspection before validating (see ``save_table``).
    """
    overrides = overrides or {}
    titles = titles_from_raw(raw)
    unknown = sorted(set(overrides) - set(titles["title_key"]))
    if unknown:
        logger.warning("Overrides for titles not in the Netflix data (typo in title_key?): %s", unknown)
    keep = existing if existing is not None else pd.DataFrame(columns=["title_key", "lookup_status"])
    if retry_unmatched:
        keep = keep[keep["lookup_status"].isin(["matched", "skipped"])]
    keep = keep[~keep["title_key"].isin(overrides)]
    todo = titles[~titles["title_key"].isin(keep["title_key"])]
    if limit is not None:
        todo = todo.head(limit)

    client = client or TMDBClient()
    if len(todo):
        client.check()
    logger.info("Looking up %d titles (%d already in the table)", len(todo), len(keep))

    records: list[dict] = []
    rows = [row for _, row in todo.iterrows()]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, rec in enumerate(pool.map(lambda r: lookup_title(client, r, overrides.get(r["title_key"])), rows), start=1):
            records.append(rec)
            if i % 250 == 0:
                logger.info("  %d / %d titles looked up", i, len(rows))

    combined = [*keep.reindex(columns=TABLE_COLUMNS).to_dict("records"), *records] if len(keep) else records
    table = _to_table(combined).sort_values("title_key").reset_index(drop=True)
    return titles_schema.validate(table, lazy=True) if validate else table


def save_table(table: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Validate, then save. If validation fails, save the table next to ``out`` as ``*.failed.parquet``
    first, so a long run's results are never lost, then raise the validation error."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        validated = titles_schema.validate(table, lazy=True)
    except pa.errors.SchemaErrors:
        failed = out.with_name(out.stem + ".failed.parquet")
        table.to_parquet(failed, index=False)
        logger.error("Validation failed: the unvalidated table was saved to %s for inspection", failed)
        raise
    validated.to_parquet(out, index=False)
    out.with_name(out.stem + ".failed.parquet").unlink(missing_ok=True)  # clear an old failed copy
    return validated


def summarize(table: pd.DataFrame) -> dict:
    """Match rate, coverage and the TMDB-vs-Netflix runtime check, for the logs, README or a slide."""
    looked_up = table[table["lookup_status"] != "skipped"]
    matched = table["lookup_status"] == "matched"
    tv = table["is_tv"]

    # Runtime cross-check: titles with both runtimes, excluding split releases (TMDB counts the whole season)
    both = table[
        matched
        & table["tmdb_runtime_h"].notna()
        & table["netflix_runtime_h"].notna()
        & ~table["season_pattern"].isin(["Part N", "Volume N"])
    ]
    ratio = both["tmdb_runtime_h"] / both["netflix_runtime_h"]
    return {
        "titles": len(table),
        "skipped_dated_episodes": int((table["lookup_status"] == "skipped").sum()),
        "match_rate": round(float((looked_up["lookup_status"] == "matched").mean()), 4) if len(looked_up) else None,
        "match_rate_by_list": {
            k: round(float(v), 4)
            for k, v in looked_up.groupby("category")["lookup_status"].apply(lambda s: (s == "matched").mean()).items()
        },
        "confidence": {k: int(v) for k, v in table.loc[matched, "match_confidence"].value_counts().items()},
        "tv_season_resolved_rate": round(float(table.loc[matched & tv, "season_resolved"].mean()), 4)
        if (matched & tv).any()
        else None,
        "runtime_coverage": round(float(table["tmdb_runtime_h"].notna().mean()), 4) if len(table) else None,
        "runtime_check": {
            "titles_compared": len(both),
            "median_ratio_tmdb_to_netflix": round(float(ratio.median()), 3) if len(both) else None,
            "share_within_15pct": round(float(((ratio - 1).abs() <= 0.15).mean()), 3) if len(both) else None,
            "flagged_2x_apart": int(table["runtime_mismatch"].fillna(0).sum()),
        },
        "tv_specials_found_as_films": int(table["match_rule"].fillna("").str.endswith("(found as a film)").sum()),
        # Worth a look by hand: very long runtimes are usually real (telenovelas), sometimes a wrong season
        "longest_runtimes": [
            {"title": t, "tmdb_runtime_h": round(float(r), 1), "episodes": None if pd.isna(e) else int(e)}
            for t, r, e in table.nlargest(5, "tmdb_runtime_h")[["title_key", "tmdb_runtime_h", "episode_count"]]
            .itertuples(index=False)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Look up Netflix titles on TMDB and save data/external/tmdb_titles.parquet"
    )
    parser.add_argument("--limit", type=int, help="Only look up this many new titles (for a trial run)")
    parser.add_argument("--retry-unmatched", action="store_true", help="Also retry titles that failed before")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Re-match every title, e.g. after changing the matching rules. Cached responses are reused, "
        "so only lookups that weren't needed before go to TMDB.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Parallel requests (default 8)")
    parser.add_argument("--out", type=Path, default=TITLES_PATH)
    parser.add_argument("--overrides", type=Path, default=OVERRIDES_PATH, help="Hand-made match fixes (CSV)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.getenv("TMDB_CACHE_DIR", CACHE_DIR)),
        help="Where to save TMDB responses (default data/external/tmdb_cache, or TMDB_CACHE_DIR in .env). "
        "Pointing this outside a OneDrive/Dropbox folder avoids syncing thousands of small files.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raw = load_raw(latest_snapshot())
    existing = pd.read_parquet(args.out) if args.out.exists() and not args.rebuild else None
    client = TMDBClient(cache_dir=args.cache_dir)
    table = update_titles_table(
        raw,
        existing,
        client,
        limit=args.limit,
        retry_unmatched=args.retry_unmatched,
        workers=args.workers,
        validate=False,
        overrides=load_overrides(args.overrides),
    )
    table = save_table(table, args.out)
    print(f"Saved {len(table):,} titles to {args.out}")
    print(f"TMDB requests: {client.stats['network']:,} from the network, {client.stats['cache']:,} from the cache")
    if client.stats["cache_write_failures"]:
        print(f"Responses that couldn't be cached (looked up fine, will be re-fetched next run): "
              f"{client.stats['cache_write_failures']:,}")
    print(json.dumps(summarize(table), indent=2))


if __name__ == "__main__":
    main()
