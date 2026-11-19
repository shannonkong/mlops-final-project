"""Tests for src/tmdb.py. A fake TMDB stands in for the real API, so no key or network is needed."""

import pathlib
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import requests
from pandera.errors import SchemaErrors

from src import tmdb

# =============================================================================
# Fake TMDB
# =============================================================================


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


class FakeTMDB:
    """Routes requests to canned data, and records every call."""

    def __init__(self):
        self.searches = {}  # (kind, query) -> list of results
        self.details = {}  # "/movie/1" or "/tv/1" -> dict
        self.seasons = {}  # "/tv/1/season/2" -> dict
        self.fail_queries = set()  # queries that raise a connection error
        self.scripted = []  # responses to return first, in order (e.g. a 429)
        self.calls = []

    def session(self):
        fake = self

        class Session:
            def __init__(self):
                self.headers = {}

            def get(self, url, params=None, timeout=None):
                path = url.removeprefix(tmdb.BASE_URL)
                fake.calls.append((path, dict(params or {}), dict(self.headers)))
                if fake.scripted:
                    return fake.scripted.pop(0)
                if path == "/configuration":
                    return FakeResponse(200, {"images": {}})
                if path.startswith("/search/"):
                    kind, query = path.split("/")[2], params["query"]
                    if query in fake.fail_queries:
                        raise requests.ConnectionError("connection reset")
                    return FakeResponse(200, {"results": fake.searches.get((kind, query), [])})
                if path in fake.seasons:
                    return FakeResponse(200, fake.seasons[path])
                if path in fake.details:
                    return FakeResponse(200, fake.details[path])
                return FakeResponse(404, {"status_message": "not found"})

        return Session()

    def search_calls(self):
        return [c for c in self.calls if c[0].startswith("/search/")]


def film(id_, title, date, popularity=10.0, lang=None):
    return {"id": id_, "title": title, "original_title": title, "release_date": date, "popularity": popularity,
            "original_language": lang}


def show(id_, name, date, popularity=10.0, original=None, lang=None):
    return {"id": id_, "name": name, "original_name": original or name, "first_air_date": date,
            "popularity": popularity, "original_language": lang}


@pytest.fixture
def fake():
    f = FakeTMDB()
    # Film: the real film plus a behind-the-scenes special with similar words
    f.searches[("movie", "Spider-Man: No Way Home")] = [
        film(634649, "Spider-Man: No Way Home", "2021-12-15", 39.3),
        film(961651, "Spider-Man: All Roads Lead to No Way Home", "2022-05-03", 2.3),
    ]
    f.details["/movie/634649"] = {
        "title": "Spider-Man: No Way Home", "runtime": 148, "release_date": "2021-12-15",
        "original_language": "en",
        "genres": [{"name": "Action"}, {"name": "Adventure"}, {"name": "Science Fiction"}],
        "keywords": {"keywords": [{"name": "superhero"}, {"name": "based on comic"}]},
        "release_dates": {"results": [{"iso_3166_1": "US", "release_dates": [{"certification": "PG-13"}]}]},
        "production_companies": [{"name": "Columbia Pictures"}],
    }
    # Film found only after removing punctuation from the search
    f.searches[("movie", "Monsters vs Aliens")] = [film(15512, "Monsters vs Aliens", "2009-03-19", 25.0)]
    f.details["/movie/15512"] = {
        "title": "Monsters vs Aliens", "runtime": 94, "release_date": "2009-03-19", "original_language": "en",
        "genres": [{"name": "Animation"}, {"name": "Family"}], "keywords": {"keywords": []},
        "release_dates": {"results": [{"iso_3166_1": "US", "release_dates": [{"certification": "PG"}]}]},
        "production_companies": [{"name": "DreamWorks Animation"}],
    }
    # TV, plain "Season N", Netflix original, with one episode missing a runtime
    f.searches[("tv", "Wednesday")] = [
        show(119051, "Wednesday", "2022-11-23", 100.0),
        show(999, "Wednesday", "1990-01-01", 1.0),
    ]
    f.details["/tv/119051"] = {
        "name": "Wednesday", "type": "Scripted", "original_language": "en",
        "genres": [{"name": "Sci-Fi & Fantasy"}, {"name": "Mystery"}, {"name": "Comedy"}],
        "keywords": {"results": [{"name": "boarding school"}]},
        "content_ratings": {"results": [{"iso_3166_1": "US", "rating": "TV-14"}]},
        "networks": [{"name": "Netflix"}],
        "seasons": [{"season_number": 1, "name": "Season 1"}, {"season_number": 2, "name": "Season 2"}],
    }
    f.seasons["/tv/119051/season/1"] = {
        "air_date": "2022-11-23",
        "episodes": [{"runtime": r} for r in [60, 50, 48, 52, 46, 48, 50, None]],
    }
    # TV, "Part N": resolved by matching TMDB's season name
    f.searches[("tv", "Attack on Titan")] = [show(1429, "Attack on Titan", "2013-04-07", 80.0, "進撃の巨人")]
    f.details["/tv/1429"] = {
        "name": "Attack on Titan", "type": "Scripted", "original_language": "ja",
        "genres": [{"name": "Animation"}, {"name": "Action & Adventure"}],
        "keywords": {"results": [{"name": "anime"}]},
        "content_ratings": {"results": [{"iso_3166_1": "US", "rating": "TV-MA"}]},
        "networks": [{"name": "NHK"}],
        "seasons": [
            {"season_number": 0, "name": "Specials"},
            {"season_number": 1, "name": "Season 1"},
            {"season_number": 4, "name": "The Final Season"},
        ],
    }
    f.seasons["/tv/1429/season/4"] = {"air_date": "2020-12-07", "episodes": [{"runtime": 24}] * 30}
    # TV, named season that matches TMDB's season name exactly
    f.searches[("tv", "Ancient Apocalypse")] = [show(215803, "Ancient Apocalypse", "2022-11-11", 20.0)]
    f.details["/tv/215803"] = {
        "name": "Ancient Apocalypse", "type": "Documentary", "original_language": "en",
        "genres": [{"name": "Documentary"}], "keywords": {"results": []},
        "content_ratings": {"results": []}, "networks": [{"name": "Netflix"}],
        "seasons": [{"season_number": 1, "name": "Season 1"}, {"season_number": 2, "name": "The Americas"}],
    }
    f.seasons["/tv/215803/season/2"] = {"air_date": "2024-10-16", "episodes": [{"runtime": 30}] * 6}
    # TV, named season that TMDB doesn't name the same way: show matched, season left blank
    f.searches[("tv", "Demon Slayer: Kimetsu no Yaiba")] = [
        show(85937, "Demon Slayer: Kimetsu no Yaiba", "2019-04-06", 90.0)
    ]
    f.details["/tv/85937"] = {
        "name": "Demon Slayer: Kimetsu no Yaiba", "type": "Scripted", "original_language": "ja",
        "genres": [{"name": "Animation"}], "keywords": {"results": [{"name": "anime"}]},
        "content_ratings": {"results": []}, "networks": [],
        "seasons": [{"season_number": 1, "name": "Season 1"}, {"season_number": 2, "name": "Season 2"}],
    }
    # Film whose only exact match came out years after it charted: rejected, not guessed
    f.searches[("movie", "Future Remake")] = [film(5, "Future Remake", "2030-01-01", 50.0)]
    # Film that makes the connection fail
    f.fail_queries.add("Broken Title")
    return f


@pytest.fixture
def client(fake, tmp_path):
    return tmdb.TMDBClient(api_key="testkey", cache_dir=tmp_path / "cache", session_factory=fake.session, pause=0)


def raw_rows(rows):
    """Minimal weekly rows: (week, category, show_title, season_title, runtime)."""
    return pd.DataFrame(rows, columns=["week", "category", "show_title", "season_title", "runtime"]).assign(
        week=lambda d: pd.to_datetime(d["week"])
    )


@pytest.fixture
def raw():
    nan = np.nan
    return raw_rows(
        [
            ("2022-07-17", "Films (English)", "Spider-Man: No Way Home", nan, nan),
            ("2026-07-26", "Films (English)", "Monsters vs. Aliens", nan, 1.57),
            ("2022-11-27", "TV (English)", "Wednesday", "Wednesday: Season 1", nan),
            ("2023-07-02", "TV (English)", "Wednesday", "Wednesday: Season 1", 6.6),
            ("2023-11-12", "TV (Non-English)", "Attack on Titan", "Attack on Titan: The Final Season Part 3", 1.5),
            ("2024-10-20", "TV (English)", "Ancient Apocalypse", "Ancient Apocalypse: The Americas", 3.0),
            ("2023-06-25", "TV (Non-English)", "Demon Slayer: Kimetsu no Yaiba",
             "Demon Slayer: Kimetsu no Yaiba: Swordsmith Village Arc", 5.0),
            ("2026-04-19", "TV (English)", "Raw", "Raw: April 13, 2026", 3.0),
            ("2024-01-07", "Films (English)", "Unknown Indie Film", nan, 1.5),
            ("2024-01-07", "Films (Non-English)", "Broken Title", nan, 1.7),
            ("2024-01-07", "Films (English)", "Future Remake", nan, 2.0),
        ]
    )


@pytest.fixture
def table(raw, client):
    return tmdb.update_titles_table(raw, client=client, workers=2)


def row(table, show_title):
    match = table[table["show_title"] == show_title]
    assert len(match) == 1
    return match.iloc[0]


# =============================================================================
# Season titles
# =============================================================================


@pytest.mark.parametrize(
    "show_title, season_title, expected",
    [
        ("Wednesday", "Wednesday: Season 1", (1, "Season N")),
        ("Bridgerton", "Bridgerton: Season 3 Part 2", (3, "Season N")),
        ("Attack on Titan", "Attack on Titan: The Final Season Part 3", (None, "Part N")),
        ("Stranger Things", "Stranger Things 4", (4, "Show name + number")),
        ("Adolescence", "Adolescence: Limited Series", (1, "Limited series")),
        ("Ms. Rachel", "Ms. Rachel", (1, "Same as show title")),
        ("Ancient Apocalypse", "Ancient Apocalypse: The Americas", (None, "other")),
        ("Raw", "Raw: April 13, 2026", (None, "Dated episode")),
        ("Some Film", np.nan, (None, "no season title")),
    ],
)
def test_parse_season(show_title, season_title, expected):
    assert tmdb.parse_season(show_title, season_title) == expected


def test_season_label():
    assert tmdb.season_label("Attack on Titan", "Attack on Titan: The Final Season Part 3") == "The Final Season Part 3"


def test_match_season_by_name_both_directions_and_ambiguity():
    seasons = [{"season_number": 1, "name": "Season 1"}, {"season_number": 4, "name": "The Final Season"}]
    assert tmdb.match_season_by_name("The Final Season Part 3", seasons) == 4  # TMDB name inside label
    assert tmdb.match_season_by_name("Final Season", seasons) == 4  # label inside TMDB name
    assert tmdb.match_season_by_name("Hashira Training Arc", seasons) is None
    two = [{"season_number": 1, "name": "Arc One"}, {"season_number": 2, "name": "Arc Two"}]
    assert tmdb.match_season_by_name("Arc", two) is None  # ambiguous: don't guess


def test_specials_are_never_matched():
    assert tmdb.match_season_by_name("Specials", [{"season_number": 0, "name": "Specials"}]) is None


# =============================================================================
# Matching
# =============================================================================


@pytest.mark.parametrize(
    "a, b", [("Spiderman No Way Home", "Spider-Man: No Way Home"), ("Monsters vs. Aliens", "Monsters vs Aliens"),
             ("Fast & Furious", "Fast and Furious")]
)
def test_same_title_ignores_punctuation_spaces_and_case(a, b):
    assert tmdb.same_title(a, b)


def test_same_title_empty_is_never_a_match():
    assert not tmdb.same_title("", "")
    assert not tmdb.same_title(None, None)


def cands(*rows):
    return pd.DataFrame(rows, columns=["tmdb_id", "name", "original_name", "days_before_charting", "popularity"])


def test_exact_title_beats_related_special():
    c = cands((1, "Spider-Man: No Way Home", "Spider-Man: No Way Home", 214, 39.3),
              (2, "Spider-Man: All Roads Lead to No Way Home", "x", 75, 2.3))
    best, rule, confidence = tmdb.pick_match("Spider-Man: No Way Home", False, c)
    assert best["tmdb_id"] == 1 and rule.startswith("1") and confidence == "high"


def test_films_prefer_most_recent_remake():
    c = cands((1, "The Little Mermaid", "The Little Mermaid", 12500, 20.0),
              (2, "The Little Mermaid", "The Little Mermaid", 100, 15.0))
    assert tmdb.pick_match("The Little Mermaid", False, c)[0]["tmdb_id"] == 2


def test_tv_prefers_most_popular_show():
    c = cands((1, "Manifest", "Manifest", 3000, 5.0), (2, "Manifest", "Manifest", 1200, 60.0))
    assert tmdb.pick_match("Manifest", True, c)[0]["tmdb_id"] == 2


def test_without_exact_title_most_popular_wins():
    c = cands((1, "Monsters vs Aliens: Creature Features", "x", 4316, 1.0), (2, "Monsters Vs Aliens!", "y", 6000, 25.0))
    best, rule, confidence = tmdb.pick_match("Monsters & Aliens", False, c)
    assert best["tmdb_id"] == 2 and rule == tmdb.RULE_LOOSE and confidence == "medium"


def test_small_negative_age_is_accepted():
    c = cands((1, "Global Premiere", "Global Premiere", -5, 10.0))  # TMDB date 5 days after Netflix release
    assert tmdb.pick_match("Global Premiere", False, c)[2] == "high"


def test_released_long_after_charting_is_rejected():
    c = cands((1, "Future Remake", "Future Remake", -2000, 50.0))
    best, rule, _ = tmdb.pick_match("Future Remake", False, c)
    assert best is None and rule.startswith("rejected")


def test_exact_title_with_unknown_date_is_low_confidence():
    c = cands((1, "Mystery Doc", "Mystery Doc", np.nan, 3.0))
    assert tmdb.pick_match("Mystery Doc", False, c)[2] == "low"


def test_no_results():
    assert tmdb.pick_match("Anything", False, cands()) == (None, "no results", None)


def lang_cands(*rows):
    return pd.DataFrame(
        rows, columns=["tmdb_id", "name", "original_name", "days_before_charting", "popularity", "original_language"]
    )


def test_same_language_beats_more_popular_other_language():
    # ONE PIECE on the English TV list: the live-action show, not the far more popular Japanese anime
    c = lang_cands((1, "One Piece", "ワンピース", 8800, 300.0, "ja"), (2, "One Piece", "One Piece", 5, 80.0, "en"))
    best, _, confidence = tmdb.pick_match("ONE PIECE", True, c, english=True)
    assert best["tmdb_id"] == 2 and confidence == "high"


def test_loose_match_in_another_language_is_rejected():
    # "Indian 2" (non-English film list) once matched "The Second Best Exotic Marigold Hotel"
    c = lang_cands((1, "The Second Best Exotic Marigold Hotel", "x", 3400, 20.0, "en"))
    best, rule, _ = tmdb.pick_match("Indian 2", False, c, english=False)
    assert best is None and rule == "rejected: only results in another language"


def test_exact_title_in_another_language_is_low_confidence():
    c = lang_cands((1, "Rare Doc", "Rare Doc", 100, 3.0, "fr"))
    best, rule, confidence = tmdb.pick_match("Rare Doc", False, c, english=True)
    assert best["tmdb_id"] == 1 and rule == tmdb.RULE_EXACT_OTHER_LANGUAGE and confidence == "low"


def test_exact_title_in_another_language_beats_a_loose_match():
    # "The Passion of the Christ" (Aramaic on TMDB) once lost to a 1903 film with a similar name
    c = lang_cands((1, "The Passion of the Christ", "x", 7700, 30.0, "en"),
                   (2, "The Life and Passion of Jesus Christ", "y", 44000, 2.0, "fr"))
    best, rule, _ = tmdb.pick_match("The Passion of the Christ", False, c, english=False)
    assert best["tmdb_id"] == 1 and rule == tmdb.RULE_EXACT_OTHER_LANGUAGE


def test_most_recent_ignores_obscure_shorts_with_the_same_title():
    # A 3-minute "Prometheus" (2021) on TMDB, vs. the 2012 film
    c = cands((1, "Prometheus", "Prometheus", 3600, 12.0), (2, "Prometheus", "Prometheus", 1600, 0.6))
    assert tmdb.pick_match("Prometheus", False, c)[0]["tmdb_id"] == 1
    # ...same for new TV shows
    assert tmdb.pick_match("Prometheus", True, c, prefer_recent=True)[0]["tmdb_id"] == 1


def test_missing_language_is_not_held_against_a_result():
    c = lang_cands((1, "Mystery Show", "Mystery Show", 100, 3.0, None))
    assert tmdb.pick_match("Mystery Show", True, c, english=False)[2] == "high"


def test_new_tv_show_prefers_most_recent_exact_title():
    # "Celebrity: Limited Series" is the 2023 Korean show, not a famous 2003 telenovela
    c = cands((1, "Celebrity", "Celebridade", 7300, 40.0), (2, "Celebrity", "셀러브리티", 10, 15.0))
    assert tmdb.pick_match("Celebrity", True, c, prefer_recent=True)[0]["tmdb_id"] == 2
    assert tmdb.pick_match("Celebrity", True, c)[0]["tmdb_id"] == 1  # later seasons still use popularity


@pytest.mark.parametrize(
    "number, pattern, expected",
    [(1, "Season N", True), (1, "Show name + number", True), (2, "Season N", False), (1, "Limited series", True),
     (1, "Same as show title", True), (None, "no season title", True), (None, "Part N", False),
     (3, "Volume N", False), (None, "other", False)],
)
def test_is_new_show(number, pattern, expected):
    assert tmdb.is_new_show(number, pattern) is expected


def test_season_runtime_fills_one_missing_episode_with_median():
    total, episodes, missing, avg = tmdb.season_runtime({"episodes": [{"runtime": 50}, {"runtime": 50}, {}]})
    assert (total, episodes, missing, avg) == (2.5, 3, 1, 50.0)


def test_season_runtime_blank_when_most_episodes_missing():
    total, _, missing, _ = tmdb.season_runtime({"episodes": [{"runtime": 50}, {}, {}]})
    assert total is None and missing == 2


# =============================================================================
# Client
# =============================================================================


def test_responses_are_cached(client, fake):
    first = client.get("/search/tv", query="Wednesday")
    second = client.get("/search/tv", query="Wednesday")
    assert first == second
    assert client.stats == {"network": 1, "cache": 1, "cache_write_failures": 0}


def test_404_returns_none_and_is_cached(client, fake):
    assert client.get("/tv/1/season/99") is None
    assert client.get("/tv/1/season/99") is None
    assert client.stats["network"] == 1


def test_short_key_sent_as_parameter(client, fake):
    client.get("/configuration")
    _, params, headers = fake.calls[-1]
    assert params["api_key"] == "testkey" and "Authorization" not in headers


def test_long_token_sent_as_header(fake, tmp_path):
    c = tmdb.TMDBClient(api_key="eyJlongtoken", cache_dir=tmp_path, session_factory=fake.session, pause=0)
    c.get("/configuration")
    _, params, headers = fake.calls[-1]
    assert "api_key" not in params and headers["Authorization"] == "Bearer eyJlongtoken"


def test_rate_limit_is_retried(client, fake):
    fake.scripted.append(FakeResponse(429, {}, headers={"Retry-After": "0"}))
    assert client.get("/configuration") == {"images": {}}
    assert client.stats["network"] == 2


def test_bad_key_stops_the_run(raw, fake, tmp_path):
    fake.scripted.append(FakeResponse(401, {"status_message": "Invalid API key"}))
    c = tmdb.TMDBClient(api_key="wrong", cache_dir=tmp_path, session_factory=fake.session, pause=0)
    with pytest.raises(requests.HTTPError):
        tmdb.update_titles_table(raw, client=c)


def test_identical_parallel_requests_call_tmdb_once(fake, tmp_path):
    """Several seasons of one show need the same search: only one worker should fetch it."""
    original = fake.session

    def slow_session():
        session = original()
        get = session.get

        def slow_get(*args, **kwargs):
            time.sleep(0.05)  # make the requests overlap
            return get(*args, **kwargs)

        session.get = slow_get
        return session

    c = tmdb.TMDBClient(api_key="k", cache_dir=tmp_path, session_factory=slow_session, pause=0)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: c.get("/search/tv", query="Wednesday"), range(8)))
    assert all(r == results[0] for r in results)
    assert c.stats["network"] == 1 and c.stats["cache"] == 7


def test_cache_save_retried_when_windows_locks_the_file(client, monkeypatch):
    real_replace, failures = pathlib.Path.replace, []

    def locked_twice(self, target):
        if len(failures) < 2:  # e.g. OneDrive or antivirus briefly holding the file
            failures.append(1)
            raise PermissionError(5, "Access is denied")
        return real_replace(self, target)

    monkeypatch.setattr(pathlib.Path, "replace", locked_twice)
    monkeypatch.setattr(tmdb.time, "sleep", lambda s: None)
    assert client.get("/search/tv", query="Wednesday")["results"]
    assert client.stats["cache_write_failures"] == 0
    assert client.get("/search/tv", query="Wednesday") is not None and client.stats["cache"] == 1


def test_cache_save_failure_never_stops_the_lookup(client, monkeypatch):
    def always_locked(self, target):
        raise PermissionError(5, "Access is denied")

    monkeypatch.setattr(pathlib.Path, "replace", always_locked)
    monkeypatch.setattr(tmdb.time, "sleep", lambda s: None)
    data = client.get("/search/tv", query="Wednesday")
    assert data["results"]  # the lookup itself still worked
    assert client.stats["cache_write_failures"] == 1
    assert not list(client.cache_dir.glob("*.tmp"))  # nothing half-saved left behind


def test_unreadable_cache_file_is_fetched_again(client, fake):
    client.get("/configuration")
    next(client.cache_dir.glob("*.json")).write_text("{not json", encoding="utf-8")
    assert client.get("/configuration") == {"images": {}}
    assert client.stats["network"] == 2


def test_leftover_temp_files_are_cleaned_up(fake, tmp_path):
    (tmp_path / "old_request.1234.tmp").write_text("partial")
    tmdb.TMDBClient(api_key="k", cache_dir=tmp_path, session_factory=fake.session, pause=0)
    assert not list(tmp_path.glob("*.tmp"))


def test_missing_key_is_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.delenv("TMDB_API_KEY", raising=False)
    monkeypatch.setattr(tmdb, "load_dotenv", lambda *args, **kwargs: None)  # ignore a local .env
    with pytest.raises(RuntimeError, match="TMDB_API_KEY"):
        tmdb.TMDBClient(cache_dir=tmp_path)


# =============================================================================
# The titles table, end to end
# =============================================================================


def test_one_row_per_title(table, raw):
    assert len(table) == 10  # 11 weekly rows, Wednesday Season 1 charted twice
    assert table["title_key"].is_unique


def test_film_details(table):
    r = row(table, "Spider-Man: No Way Home")
    assert r["lookup_status"] == "matched" and r["tmdb_id"] == 634649 and r["match_confidence"] == "high"
    assert r["genre_action_adventure"] == 1 and r["genre_fantasy_scifi"] == 1 and r["genre_drama"] == 0
    assert r["rating_group"] == "teen" and r["is_netflix_original"] == 0 and r["tmdb_type"] == "Film"
    assert r["tmdb_runtime_h"] == pytest.approx(148 / 60, abs=1e-3)
    assert r["age_days_at_charting"] == 214 and r["age_group"] == "recent"


def test_search_retried_without_punctuation(table, fake):
    r = row(table, "Monsters vs. Aliens")
    assert r["tmdb_id"] == 15512 and r["genre_family_kids"] == 1 and r["genre_animation"] == 1
    assert r["age_group"] == "catalog"
    queries = [params["query"] for _, params, _ in fake.search_calls()]
    assert "Monsters vs. Aliens" in queries and "Monsters vs Aliens" in queries


def test_tv_season_details(table):
    r = row(table, "Wednesday")
    assert r["tmdb_id"] == 119051  # the 2022 show, not the 1990 one
    assert r["season_resolved"] and r["season_number"] == 1 and r["episode_count"] == 8
    assert r["episodes_missing_runtime"] == 1
    assert r["tmdb_runtime_h"] == pytest.approx((354 + 50) / 60, abs=1e-3)  # missing episode filled with median 50
    assert r["is_netflix_original"] == 1 and r["rating_group"] == "teen" and r["tmdb_type"] == "Scripted"
    assert r["netflix_runtime_h"] == 6.6  # largest Netflix runtime across its weeks
    assert r["age_days_at_charting"] == 4 and r["age_group"] == "new"


def test_part_title_resolved_by_season_name(table):
    r = row(table, "Attack on Titan")
    assert r["season_pattern"] == "Part N" and r["season_number"] == 4 and r["episode_count"] == 30
    assert r["kw_anime"] == 1 and r["rating_group"] == "mature" and r["original_language"] == "ja"


def test_named_season_resolved(table):
    r = row(table, "Ancient Apocalypse")
    assert r["season_number"] == 2 and r["tmdb_type"] == "Documentary"


def test_unmatched_season_name_leaves_season_blank(table):
    r = row(table, "Demon Slayer: Kimetsu no Yaiba")
    assert r["lookup_status"] == "matched" and r["kw_anime"] == 1  # show-level details still filled
    assert not r["season_resolved"] and pd.isna(r["season_number"]) and pd.isna(r["tmdb_runtime_h"])


def test_dated_episodes_are_skipped_without_calling_tmdb(table, fake):
    r = row(table, "Raw")
    assert r["lookup_status"] == "skipped"
    assert not any(params.get("query") == "Raw" for _, params, _ in fake.search_calls())


def test_unmatched_and_rejected_titles(table):
    assert row(table, "Unknown Indie Film")["lookup_status"] == "no_match"
    future = row(table, "Future Remake")
    assert future["lookup_status"] == "no_match" and future["match_rule"].startswith("rejected")
    assert pd.isna(future["tmdb_id"]) and pd.isna(future["genre_drama"])


def test_one_failing_title_does_not_stop_the_others(table):
    assert row(table, "Broken Title")["lookup_status"] == "error"
    assert (table["lookup_status"] == "matched").sum() == 6


def test_runtime_mismatch_flag(table):
    aot = row(table, "Attack on Titan")  # TMDB counts the whole 30-episode season, Netflix only one part
    assert aot["runtime_ratio"] == pytest.approx(12.0 / 1.5) and aot["runtime_mismatch"] == 1
    assert row(table, "Wednesday")["runtime_mismatch"] == 0
    assert pd.isna(row(table, "Spider-Man: No Way Home")["runtime_mismatch"])  # no Netflix runtime to compare


def test_runtime_flag_added_to_tables_saved_without_it(table):
    old = table.drop(columns=["runtime_ratio", "runtime_mismatch"])
    rebuilt = tmdb._to_table(old.reindex(columns=tmdb.TABLE_COLUMNS).to_dict("records"))
    assert rebuilt["runtime_mismatch"].tolist() == table["runtime_mismatch"].tolist()


def test_new_show_matched_by_language_and_recency(fake, client):
    fake.searches[("tv", "ONE PIECE")] = [
        show(37854, "One Piece", "1999-10-20", 300.0, "ワンピース", lang="ja"),
        show(111110, "ONE PIECE", "2023-08-31", 80.0, lang="en"),
    ]
    fake.details["/tv/111110"] = {"name": "ONE PIECE", "type": "Scripted", "original_language": "en",
                                  "genres": [], "networks": [{"name": "Netflix"}],
                                  "seasons": [{"season_number": 1, "name": "Season 1"}]}
    fake.seasons["/tv/111110/season/1"] = {"air_date": "2023-08-31", "episodes": [{"runtime": 60}] * 8}
    raw = raw_rows([("2023-09-03", "TV (English)", "ONE PIECE", "ONE PIECE: Season 1", 7.5)])
    r = row(tmdb.update_titles_table(raw, client=client), "ONE PIECE")
    assert r["tmdb_id"] == 111110 and r["episode_count"] == 8 and r["runtime_mismatch"] == 0


def test_tv_special_found_as_a_film(fake, client):
    # Netflix lists stand-up as TV; TMDB files it as a film
    fake.searches[("movie", "Ali Wong: Single Lady")] = [film(1200, "Ali Wong: Single Lady", "2024-05-21", lang="en")]
    fake.details["/movie/1200"] = {"title": "Ali Wong: Single Lady", "runtime": 66, "release_date": "2024-05-21",
                                   "original_language": "en", "genres": [{"name": "Comedy"}],
                                   "production_companies": [{"name": "Netflix"}]}
    raw = raw_rows([("2024-05-26", "TV (English)", "Ali Wong: Single Lady", np.nan, 1.1)])
    table = tmdb.update_titles_table(raw, client=client)
    r = row(table, "Ali Wong: Single Lady")
    assert r["lookup_status"] == "matched" and r["tmdb_type"] == "Film" and r["match_rule"].endswith("(found as a film)")
    assert r["tmdb_runtime_h"] == pytest.approx(1.1) and r["genre_comedy"] == 1 and r["is_netflix_original"] == 1
    assert not r["season_resolved"]
    assert tmdb.summarize(table)["tv_specials_found_as_films"] == 1


def test_seasons_are_never_searched_as_films(table, fake):
    # Demon Slayer's season didn't resolve, but the show matched: no film search for TV with a season
    assert not any(path == "/search/movie" and params["query"].startswith("Demon Slayer")
                   for path, params, _ in fake.search_calls())


def test_old_loose_match_for_new_show_is_low_confidence():
    # "Class: Season 1" (2023) once matched "Mr. Kinpachi in Class 3B" (1979)
    c = cands((1, "Mr. Kinpachi in Class 3B", "x", 15800, 5.0))
    best, rule, confidence = tmdb.pick_match("Class", True, c, prefer_recent=True)
    assert best["tmdb_id"] == 1 and confidence == "low" and "old show" in rule
    assert tmdb.pick_match("Class", True, c)[2] == "medium"  # later seasons: unchanged


# --- Manual overrides --------------------------------------------------------


def write_overrides(path, rows):
    lines = ["# comment lines are ignored", "title_key,tmdb_id,kind,note", *rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_load_overrides(tmp_path):
    path = write_overrides(tmp_path / "o.csv", ["TV (English) | A | N/A,123,movie,special", "Films (English) | B | N/A,,,"])
    o = tmdb.load_overrides(path)
    assert o["TV (English) | A | N/A"] == {"tmdb_id": 123, "kind": "movie", "note": "special"}
    assert o["Films (English) | B | N/A"]["tmdb_id"] is None
    assert tmdb.load_overrides(tmp_path / "missing.csv") == {}


@pytest.mark.parametrize(
    "rows, message",
    [(["X | Y | N/A,1,,", "X | Y | N/A,2,,"], "twice"), (["X | Y | N/A,1,film,"], "kind")],
)
def test_bad_overrides_file_is_a_clear_error(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        tmdb.load_overrides(write_overrides(tmp_path / "o.csv", rows))


def test_override_replaces_the_automatic_match(table, raw, fake, client):
    key = row(table, "Wednesday")["title_key"]
    fake.details["/tv/999"] = {"name": "Wednesday (1990)", "type": "Scripted", "genres": [{"name": "Drama"}],
                               "seasons": [{"season_number": 1, "name": "Season 1"}]}
    fake.seasons["/tv/999/season/1"] = {"air_date": "1990-01-01", "episodes": [{"runtime": 30}] * 4}
    overrides = {key: {"tmdb_id": 999, "kind": None, "note": ""}}
    updated = tmdb.update_titles_table(raw, existing=table, client=client, overrides=overrides)
    r = row(updated, "Wednesday")  # already in the table, but looked up again because it's overridden
    assert r["tmdb_id"] == 999 and r["match_confidence"] == "manual" and r["match_rule"] == "override"
    assert r["episode_count"] == 4  # season details still resolved as usual
    assert row(updated, "Spider-Man: No Way Home")["tmdb_id"] == 634649  # others kept as they were


def test_blank_override_removes_a_wrong_match(table, raw, client):
    key = row(table, "Spider-Man: No Way Home")["title_key"]
    overrides = {key: {"tmdb_id": None, "kind": None, "note": ""}}
    r = row(tmdb.update_titles_table(raw, existing=table, client=client, overrides=overrides), "Spider-Man: No Way Home")
    assert r["lookup_status"] == "no_match" and r["match_rule"] == "override: no TMDB match"
    assert pd.isna(r["tmdb_id"]) and pd.isna(r["genre_drama"])


def test_override_with_a_wrong_id_is_reported(raw, client):
    key = "Films (English) | Spider-Man: No Way Home | N/A"
    overrides = {key: {"tmdb_id": 424242, "kind": None, "note": ""}}
    r = row(tmdb.update_titles_table(raw, client=client, overrides=overrides), "Spider-Man: No Way Home")
    assert r["lookup_status"] == "error" and "424242" in r["match_rule"]


def test_override_for_unknown_title_is_warned_about(raw, client, caplog):
    tmdb.update_titles_table(raw, client=client, overrides={"TV (English) | Typo | N/A": {"tmdb_id": 1, "kind": None}})
    assert "Typo" in caplog.text


def test_repo_overrides_file_loads():
    assert all(" | " in key for key in tmdb.load_overrides())


def test_popularity_is_never_saved(table):
    assert not any("popular" in c or "vote" in c for c in table.columns)


def test_table_survives_a_parquet_round_trip(table, tmp_path):
    path = tmp_path / "tmdb_titles.parquet"
    table.to_parquet(path, index=False)
    tmdb.titles_schema.validate(pd.read_parquet(path), lazy=True)


# =============================================================================
# Weekly updates
# =============================================================================


def test_only_new_titles_are_looked_up(table, raw, fake, tmp_path):
    new_week = raw_rows([("2024-02-04", "TV (English)", "Wednesday", "Wednesday: Season 2", 7.0)])
    fake.searches[("tv", "Wednesday")]  # same show, new season
    fake.seasons["/tv/119051/season/2"] = {"air_date": "2024-01-30", "episodes": [{"runtime": 50}] * 8}
    fresh = tmdb.TMDBClient(api_key="testkey", cache_dir=tmp_path / "other_cache",
                            session_factory=fake.session, pause=0)
    before = len(fake.search_calls())
    updated = tmdb.update_titles_table(pd.concat([raw, new_week]), existing=table, client=fresh)
    assert len(updated) == 11
    assert len(fake.search_calls()) - before == 1  # only the new season was searched
    assert row(updated[updated["season_title"] == "Wednesday: Season 2"], "Wednesday")["season_number"] == 2


def test_retry_unmatched_looks_up_failures_again(table, raw, fake, client):
    fake.fail_queries.clear()
    fake.searches[("movie", "Broken Title")] = [film(77, "Broken Title", "2023-12-01", 5.0)]
    fake.details["/movie/77"] = {"title": "Broken Title", "runtime": 100, "genres": [{"name": "Drama"}]}
    updated = tmdb.update_titles_table(raw, existing=table, client=client, retry_unmatched=True)
    assert row(updated, "Broken Title")["lookup_status"] == "matched"
    assert row(updated, "Spider-Man: No Way Home")["tmdb_id"] == 634649  # kept as it was


def test_limit_for_trial_runs(raw, client):
    assert len(tmdb.update_titles_table(raw, client=client, limit=3)) == 3


# =============================================================================
# Schema and summary
# =============================================================================


def test_schema_catches_duplicate_titles(table):
    with pytest.raises(SchemaErrors):
        tmdb.titles_schema.validate(pd.concat([table, table.head(1)]), lazy=True)


def test_schema_catches_matched_title_without_id(table):
    broken = table.copy()
    broken.loc[broken["lookup_status"] == "matched", "tmdb_id"] = pd.NA
    with pytest.raises(SchemaErrors):
        tmdb.titles_schema.validate(broken, lazy=True)


def test_long_telenovela_runtime_is_accepted(table):
    long = table.copy()
    long.loc[0, "tmdb_runtime_h"] = 322.0  # e.g. a single TMDB season of ~335 hour-long episodes
    tmdb.titles_schema.validate(long, lazy=True)


def test_impossible_runtime_is_still_caught(table):
    bad = table.copy()
    bad.loc[0, "tmdb_runtime_h"] = 3000.0  # e.g. minutes saved as hours
    with pytest.raises(SchemaErrors):
        tmdb.titles_schema.validate(bad, lazy=True)


def test_failed_validation_saves_the_table_for_inspection(table, tmp_path):
    out = tmp_path / "tmdb_titles.parquet"
    bad = table.copy()
    bad.loc[0, "tmdb_runtime_h"] = 3000.0
    with pytest.raises(SchemaErrors):
        tmdb.save_table(bad, out)
    assert not out.exists()  # the real table is never overwritten with bad data
    assert len(pd.read_parquet(tmp_path / "tmdb_titles.failed.parquet")) == len(table)


def test_successful_save_clears_an_old_failed_copy(table, tmp_path):
    out = tmp_path / "tmdb_titles.parquet"
    (tmp_path / "tmdb_titles.failed.parquet").write_bytes(b"old")
    tmdb.save_table(table, out)
    assert out.exists() and not (tmp_path / "tmdb_titles.failed.parquet").exists()


def test_unvalidated_table_can_be_returned(raw, client):
    table = tmdb.update_titles_table(raw, client=client, validate=False)
    assert len(table) == 10


def test_summary(table):
    s = tmdb.summarize(table)
    assert s["titles"] == 10 and s["skipped_dated_episodes"] == 1
    assert s["match_rate"] == pytest.approx(6 / 9, abs=1e-3)  # skipped titles aren't counted
    assert s["confidence"] == {"high": 6}
    # Runtime check compares Wednesday, Ancient Apocalypse and Monsters vs Aliens; Attack on Titan (a Part) is excluded
    assert s["runtime_check"]["titles_compared"] == 3
    assert s["runtime_check"]["flagged_2x_apart"] == 1  # Attack on Titan (whole season vs one part)
    assert s["longest_runtimes"][0]["title"].startswith("TV (Non-English) | Attack on Titan")  # 30 x 24 min
    assert s["longest_runtimes"][0]["episodes"] == 30
