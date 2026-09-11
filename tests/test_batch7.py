"""
The matching bench (Batch 7).

Nothing about the matching policy is decided until there is something to
score it against. `yargvid bench` re-ranks the stored `candidates` table
offline under a named policy and reports, per class of ground truth, how
often the known-right video wins. It downloads nothing and writes nothing
but its own CSV.

Ground truth is what a person said:

- approval: `review = 'keep'` on a pipeline pick
- override: `match_note` starts `MANUAL:` and a video is set - the person
  chose the video, whether or not they have confirmed the timing yet.
  A hand pick that is also approved is an override, not an approval:
  the two classes partition the labelled set so a policy cannot be
  scored twice for one song.

A labelled song's known video is `reachable` when it is among the stored
candidates and passes the fingerprint gate, `gated` when it is stored but
was never heard or failed the gate, and `absent` when the search never
returned it. Only reachable songs can be won, so win rate is over them.

`--songs <file>` on `match` and `sync` makes the file the selection: one
song_dir per line, exact match, `#` lines and blank lines ignored, unknown
paths reported and skipped. `match --songs` re-matches regardless of
status but never touches a hand pick, and a changed pick requeues the
later stages the way `set` does. `sync --songs` applies the recheck
rules: an unchanged answer keeps the approval, a moved one clears it.

`view_count` comes back with every search result and Batch 8 needs it, so
it is stored on the candidate. Six more penalty terms name things that are
not the music video.

    pytest -q tests/test_batch7.py
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yargvid import audio as au
from yargvid import bench
from yargvid import cli
from yargvid import encode as enc
from yargvid import fingerprint as fp
from yargvid import match as mt
from yargvid import sync as sy
from yargvid.db import Database

GATE = fp.ACCEPT_SCORE + 100.0    # comfortably heard
COV = fp.ACCEPT_COVERAGE + 0.5


# ----------------------------------------------------------------- helpers ---

@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


def song(db, path, artist="Artist", title="Song", **cols):
    db.add_song(Path(path), artist, title, 200.0)
    if cols:
        db.update(Path(path), **cols)
    return Path(path)


def row(db, path):
    return db.conn.execute(
        "SELECT * FROM songs WHERE song_dir = ?", (str(path),)
    ).fetchone()


def cand(video_id, title, uploader="Artist", score=GATE, coverage=COV):
    return mt.Candidate(video_id=video_id, title=title, uploader=uploader,
                        duration=200.0, score=score, coverage=coverage)


def dump(db):
    """Every row of both tables, for a writes-nothing check."""
    return [tuple(r) for r in db.conn.execute(
        "SELECT * FROM songs ORDER BY song_dir")] + [
        tuple(r) for r in db.conn.execute(
            "SELECT * FROM candidates ORDER BY song_dir, video_id")]


# ------------------------------------------------------------ view_count -----

def test_candidates_table_has_view_count(db):
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(candidates)")}
    assert "view_count" in cols


def test_existing_database_gains_view_count(tmp_path):
    # A database made before the column existed must be migrated on open,
    # the way the songs columns are.
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE songs (song_dir TEXT PRIMARY KEY, match_status TEXT, "
        "sync_status TEXT, encode_status TEXT);"
        "CREATE TABLE candidates (song_dir TEXT, video_id TEXT, title TEXT, "
        "uploader TEXT, duration REAL, score REAL, coverage REAL, "
        "PRIMARY KEY (song_dir, video_id));"
    )
    conn.commit()
    conn.close()
    d = Database(path)
    try:
        cols = {r[1] for r in d.conn.execute("PRAGMA table_info(candidates)")}
    finally:
        d.close()
    assert "view_count" in cols


def test_save_candidates_stores_view_count(db):
    s = song(db, "/lib/a")
    c = cand("aaaaaaaaaaa", "Song (Official Video)")
    c.view_count = 1234
    db.save_candidates(s, [c])
    got = db.conn.execute(
        "SELECT view_count FROM candidates WHERE video_id = 'aaaaaaaaaaa'"
    ).fetchone()[0]
    assert got == 1234


def test_search_candidates_reads_view_count(monkeypatch):
    lines = [
        json.dumps({"id": "aaaaaaaaaaa", "title": "Song (Official Video)",
                    "uploader": "Artist", "duration": 200,
                    "view_count": 4567}),
        json.dumps({"id": "bbbbbbbbbbb", "title": "Song (Lyric Video)",
                    "uploader": "Artist", "duration": 201}),
    ]
    monkeypatch.setattr(mt, "_run", lambda cmd, timeout=600: (
        subprocess.CompletedProcess(cmd, 0, "\n".join(lines), "")))
    got = {c.video_id: c for c in mt.search_candidates("Artist", "Song", 200.0)}
    assert got["aaaaaaaaaaa"].view_count == 4567
    assert got["bbbbbbbbbbb"].view_count is None


# --------------------------------------------------------- penalty terms -----

@pytest.mark.parametrize("term", [
    "behind the curtain", "montage", "rb2", "rb3", "rb4", "rbn",
])
def test_new_penalty_terms_lower_the_title_score(term):
    plain = mt.title_score("Artist - Song (Official Video)")
    assert mt.title_score(f"Artist - Song (Official Video) {term}") < plain


# ------------------------------------------------------------- the bench -----

@pytest.fixture
def bench_db(db):
    """
    Eight songs with stored candidates, covering every label and outcome.

    a  approval, known video reachable, current policy picks it     -> win
    b  approval, known is the lyric video, a stranger's 'official'
       outranks it under the current rule                            -> lost
    c  override, known reachable and best                            -> win
    d  override, known never returned by the search                  -> absent
    e  override that is also approved; known stored but never heard  -> gated
    f  unlabelled, the policy would pick something other than stored
    g  dropped by the user (MANUAL, no video)                        -> unlabelled
    h  saved for later                                               -> unlabelled
    """
    a = song(db, "/lib/a", match_status="ok", video_id="aaaaaaaaaaa",
             match_note="Song (Official Video) [Artist]", review="keep")
    db.save_candidates(a, [
        cand("aaaaaaaaaaa", "Song (Official Video)"),
        cand("aaaaaaaaaa2", "Song (Lyric Video)", score=GATE + 50),
    ])
    b = song(db, "/lib/b", match_status="ok", video_id="bbbbbbbbbbb",
             match_note="Song (Lyric Video) [Artist]", review="keep")
    db.save_candidates(b, [
        cand("bbbbbbbbbbb", "Song (Lyric Video)"),
        cand("bbbbbbbbbb2", "Song (Official Video)", uploader="Stranger"),
    ])
    c = song(db, "/lib/c", match_status="ok", video_id="ccccccccccc",
             match_note="MANUAL: Song (Official Music Video) [Artist]")
    db.save_candidates(c, [
        cand("ccccccccccc", "Song (Official Music Video)"),
        cand("cccccccccc2", "Song (Official Video)", uploader="Stranger"),
    ])
    d = song(db, "/lib/d", match_status="ok", video_id="ddddddddddd",
             match_note="MANUAL: Song [Artist]")
    db.save_candidates(d, [
        cand("dddddddddd2", "Song (Lyric Video)"),
    ])
    e = song(db, "/lib/e", match_status="ok", video_id="eeeeeeeeeee",
             match_note="MANUAL: Song (Official Video) [Artist]", review="keep")
    db.save_candidates(e, [
        cand("eeeeeeeeeee", "Song (Official Video)", score=0.0, coverage=0.0),
        cand("eeeeeeeeee2", "Song (Lyric Video)"),
    ])
    f = song(db, "/lib/f", match_status="ok", video_id="fffffffffff",
             match_note="Song (Lyric Video) [Artist]")
    db.save_candidates(f, [
        cand("fffffffffff", "Song (Lyric Video)"),
        cand("ffffffffff2", "Song (Official Video)"),
    ])
    g = song(db, "/lib/g", match_status="skipped", video_id=None,
             match_note="MANUAL: no video wanted")
    db.save_candidates(g, [cand("ggggggggggg", "Song (Official Video)")])
    h = song(db, "/lib/h", match_status="ok", video_id="hhhhhhhhhhh",
             match_note="Song (Official Video) [Artist]", review="later")
    db.save_candidates(h, [cand("hhhhhhhhhhh", "Song (Official Video)")])
    return db


def _by_dir(rows):
    return {r["song_dir"]: r for r in rows}


def test_bench_labels_approvals_and_overrides_separately(bench_db):
    got = _by_dir(bench.run(bench_db, "current"))
    assert {k: v["label"] for k, v in got.items()} == {
        "/lib/a": "approval", "/lib/b": "approval",
        "/lib/c": "override", "/lib/d": "override", "/lib/e": "override",
        "/lib/f": "", "/lib/g": "", "/lib/h": "",
    }


def test_bench_current_policy_is_the_pick_best_rule(bench_db):
    # Highest title preference among gate passers, score breaking ties -
    # exactly what pick_best applies. A candidate that was never heard
    # (score 0) can never be the pick.
    got = _by_dir(bench.run(bench_db, "current"))
    assert got["/lib/a"]["policy_pick"] == "aaaaaaaaaaa"
    assert got["/lib/b"]["policy_pick"] == "bbbbbbbbbb2"
    assert got["/lib/c"]["policy_pick"] == "ccccccccccc"
    assert got["/lib/e"]["policy_pick"] == "eeeeeeeeee2"
    assert got["/lib/f"]["policy_pick"] == "ffffffffff2"


def test_bench_reports_reachability_and_wins(bench_db):
    got = _by_dir(bench.run(bench_db, "current"))
    assert got["/lib/a"]["known_status"] == "reachable"
    assert got["/lib/b"]["known_status"] == "reachable"
    assert got["/lib/c"]["known_status"] == "reachable"
    assert got["/lib/d"]["known_status"] == "absent"
    assert got["/lib/e"]["known_status"] == "gated"
    assert got["/lib/a"]["win"] is True
    assert got["/lib/b"]["win"] is False
    assert got["/lib/c"]["win"] is True
    assert got["/lib/d"]["win"] is False
    assert got["/lib/e"]["win"] is False
    assert got["/lib/f"]["win"] is None       # unlabelled: nothing to win


def test_bench_summary_scores_each_class_on_its_own(bench_db):
    s = bench.summary(bench.run(bench_db, "current"))
    assert s["approval"] == dict(labelled=2, reachable=2, gated=0,
                                 absent=0, win=1)
    assert s["override"] == dict(labelled=3, reachable=1, gated=1,
                                 absent=1, win=1)
    # f is stored as the lyric video, the policy would take the official
    # one; g has no stored pick; h agrees with the policy.
    assert s["unlabelled"] == dict(songs=3, differs=1)


def test_bench_writes_a_csv_named_by_policy(bench_db, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--db", str(bench_db.path), "bench"]) == 0
    out = tmp_path / "yargvid_bench_current.csv"
    assert out.exists()
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    for col in ("song_dir", "label", "known_video", "known_status",
                "policy_pick", "current_video", "win"):
        assert col in header
    assert len(out.read_text(encoding="utf-8").splitlines()) == 9  # 8 songs


def test_bench_refuses_to_overwrite_without_force(bench_db, tmp_path,
                                                  monkeypatch):
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "yargvid_bench_current.csv"
    out.write_text("keep me", encoding="utf-8")
    cli.main(["--db", str(bench_db.path), "bench"])
    assert out.read_text(encoding="utf-8") == "keep me"
    cli.main(["--db", str(bench_db.path), "bench", "--force"])
    assert out.read_text(encoding="utf-8") != "keep me"


def test_bench_writes_nothing_and_downloads_nothing(bench_db, tmp_path,
                                                    monkeypatch):
    monkeypatch.chdir(tmp_path)

    def no_network(*a, **k):
        raise AssertionError("bench must not run yt-dlp")
    monkeypatch.setattr(mt, "_run", no_network)
    before = dump(bench_db)
    cli.main(["--db", str(bench_db.path), "bench"])
    assert dump(bench_db) == before


# ---------------------------------------------------------------- --songs ----

@pytest.fixture
def stubbed_match(monkeypatch):
    """cmd_match without ffmpeg or yt-dlp: the winner is whatever is set."""
    monkeypatch.setattr(au, "find_stems", lambda d: [Path("x.ogg")])
    monkeypatch.setattr(au, "mix_stems",
                        lambda stems, sr=fp.SR: np.ones(sr, np.float32))
    state = SimpleNamespace(winner=None, seen=[])

    def pick(chart, artist, title, work, cookies=None, sleep=0.0):
        state.seen.append(title)
        w = state.winner
        return w, [w] if w else [], "ok" if w else "no audio match"
    monkeypatch.setattr(mt, "pick_best", pick)
    return state


def _match_args(songs_file, **extra):
    base = dict(songs=str(songs_file), redo=False, limit=None, sample=False,
                work="./.yargvid_work", cookies=None, sleep=0.0, gate=None)
    base.update(extra)
    return SimpleNamespace(**base)


def test_match_songs_is_the_selection(db, stubbed_match, tmp_path):
    listed = song(db, "/lib/listed", title="Listed", match_status="ok",
                  video_id="aaaaaaaaaaa", match_score=5.0)
    song(db, "/lib/pending", title="Pending")
    manual = song(db, "/lib/manual", title="Manual", match_status="ok",
                  video_id="mmmmmmmmmmm", match_score=5.0,
                  match_note="MANUAL: Song [Artist]")
    f = tmp_path / "songs.txt"
    f.write_text("# a comment\n\n/lib/listed\n/lib/manual\n/lib/not-there\n",
                 encoding="utf-8")
    stubbed_match.winner = cand("aaaaaaaaaaa", "Song (Official Video)")
    stubbed_match.winner.score = 999.0

    cli.cmd_match(_match_args(f), db)

    assert stubbed_match.seen == ["Listed"]            # not Pending, not Manual
    assert row(db, listed)["match_score"] == 999.0
    assert row(db, manual)["match_score"] == 5.0
    assert row(db, Path("/lib/pending"))["match_status"] == "pending"


def test_match_songs_same_pick_keeps_the_download(db, stubbed_match, tmp_path):
    s = song(db, "/lib/s", match_status="ok", video_id="aaaaaaaaaaa",
             match_score=5.0, download_status="ok",
             source_path="/lib/s/video.src.mkv", sync_status="ok",
             offset_ms=-100.0, review="keep")
    f = tmp_path / "songs.txt"
    f.write_text("/lib/s\n", encoding="utf-8")
    stubbed_match.winner = cand("aaaaaaaaaaa", "Song (Official Video)")
    stubbed_match.winner.score = 999.0

    cli.cmd_match(_match_args(f), db)

    r = row(db, s)
    assert r["match_score"] == 999.0                   # it was re-matched
    assert r["download_status"] == "ok"
    assert r["source_path"] == "/lib/s/video.src.mkv"
    assert r["sync_status"] == "ok" and r["review"] == "keep"


def test_match_songs_changed_pick_requeues_like_set(db, stubbed_match, tmp_path):
    s = song(db, "/lib/s", match_status="ok", video_id="aaaaaaaaaaa",
             download_status="ok", source_path="/lib/s/video.src.mkv",
             sync_status="ok", offset_ms=-100.0, spread_ms=3.0,
             fp_score=600.0, motion=0.5, windows=7, video_seconds=210.0,
             review="keep", encode_status="ok", ini_status="ok")
    f = tmp_path / "songs.txt"
    f.write_text("/lib/s\n", encoding="utf-8")
    stubbed_match.winner = cand("bbbbbbbbbbb", "Song (Official Video)")

    cli.cmd_match(_match_args(f), db)

    r = row(db, s)
    assert r["video_id"] == "bbbbbbbbbbb" and r["match_status"] == "ok"
    assert r["download_status"] == "pending" and r["source_path"] is None
    assert r["sync_status"] == "pending" and r["offset_ms"] is None
    assert r["fp_score"] is None and r["motion"] is None
    assert r["windows"] is None and r["video_seconds"] is None
    assert r["review"] is None
    assert r["encode_status"] == "pending" and r["ini_status"] == "pending"


@pytest.fixture
def stubbed_sync(monkeypatch, tmp_path):
    """cmd_sync without ffmpeg: audio decodes to ones, motion is 0.5."""
    src = tmp_path / "video.src.mkv"
    src.write_bytes(b"x")
    monkeypatch.setattr(au, "find_stems", lambda d: [Path("x.ogg")])
    monkeypatch.setattr(au, "mix_stems",
                        lambda stems, sr=fp.SR: np.ones(sr, np.float32))
    monkeypatch.setattr(au, "decode_mono",
                        lambda p, sr=fp.SR, max_seconds=None: np.ones(sr, np.float32))
    monkeypatch.setattr(au, "duration_of", lambda p: 100.0)
    monkeypatch.setattr(enc, "motion_score", lambda p: 0.5)
    return src


def _sync_args(songs_file, **extra):
    base = dict(songs=str(songs_file), recheck=False, min_offset=0.0,
                skip_reviewed=False, limit=None, sample=False)
    base.update(extra)
    return SimpleNamespace(**base)


def _synced(db, path, src, **extra):
    return song(db, path, match_status="ok", download_status="ok",
                source_path=str(src), sync_status="ok", offset_ms=-1000.0,
                spread_ms=3.0, motion=0.5, review="keep", encode_status="ok",
                ini_status="ok", **extra)


def test_sync_songs_is_the_selection(db, stubbed_sync, tmp_path, monkeypatch):
    listed = song(db, "/lib/listed", match_status="ok", download_status="ok",
                  source_path=str(stubbed_sync))
    other = song(db, "/lib/other", match_status="ok", download_status="ok",
                 source_path=str(stubbed_sync))
    f = tmp_path / "songs.txt"
    f.write_text("/lib/listed\n", encoding="utf-8")
    monkeypatch.setattr(sy, "estimate", lambda *a, **k: sy.SyncResult(
        "ok", offset_ms=-500.0, spread_ms=3.0, fp_score=80.0))

    cli.cmd_sync(_sync_args(f), db)

    assert row(db, listed)["sync_status"] == "ok"
    assert row(db, listed)["offset_ms"] == -500.0
    assert row(db, other)["sync_status"] == "pending"


def test_sync_songs_unchanged_answer_keeps_the_approval(
        db, stubbed_sync, tmp_path, monkeypatch):
    s = _synced(db, "/lib/s", stubbed_sync)
    f = tmp_path / "songs.txt"
    f.write_text("/lib/s\n", encoding="utf-8")
    monkeypatch.setattr(sy, "estimate", lambda *a, **k: sy.SyncResult(
        "ok", offset_ms=-1010.0, spread_ms=4.0, fp_score=80.0,
        windows=6, windows_total=7))

    cli.cmd_sync(_sync_args(f), db)

    r = row(db, s)
    assert r["spread_ms"] == 4.0 and r["windows"] == 6   # it was re-measured
    assert r["review"] == "keep" and r["encode_status"] == "ok"


def test_sync_songs_moved_answer_clears_the_approval(
        db, stubbed_sync, tmp_path, monkeypatch):
    s = _synced(db, "/lib/s", stubbed_sync)
    f = tmp_path / "songs.txt"
    f.write_text("/lib/s\n", encoding="utf-8")
    monkeypatch.setattr(sy, "estimate", lambda *a, **k: sy.SyncResult(
        "ok", offset_ms=-1500.0, spread_ms=3.0, fp_score=80.0))

    cli.cmd_sync(_sync_args(f), db)

    r = row(db, s)
    assert r["offset_ms"] == -1500.0
    assert r["review"] is None and r["encode_status"] == "pending"


def test_songs_flag_is_registered_on_match_and_sync(db, monkeypatch):
    # argparse defaults are state rules: the flag exists on both commands
    # and is off unless given.
    seen = {}
    monkeypatch.setattr(cli, "cmd_match", lambda a, d: seen.setdefault("m", a))
    monkeypatch.setattr(cli, "cmd_sync", lambda a, d: seen.setdefault("s", a))
    cli.main(["--db", str(db.path), "match"])
    cli.main(["--db", str(db.path), "sync", "--songs", "list.txt"])
    assert seen["m"].songs is None
    assert seen["s"].songs == "list.txt"
