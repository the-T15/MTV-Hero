"""
Batch 9 - encode settings.

Nothing in the library has ever been encoded, so every number `encode` runs
with is untested against a real output file. This batch pins the settings the
first run will use and the two state rules around them. It is small on code
and large on measurement; the measurement is not a test, it is a run.

    M1  `cmd_encode` passes `bitrate_cap` and `max_fps` into `EncodeSettings`.
        The dataclass default cap moves from 2M to 4M so that at 1080p
        `-crf 31` governs and the cap is a ceiling, not the operating point.
    M2  `encode` gains `--bitrate-cap` (default 4M) and `--max-fps`
        (default 30.0).
    M4  `encode_many` takes `keep_source` and forwards it to `encode_one`; the
        preview path runs through the pool, so `--workers` applies to both.
    M5  A successful encode writes `encode_note=None`, not "".
    M6  The full encode writes `video_start_time` and sets `ini_status='ok'`
        in the same update as `encode_status='ok'`. A failed encode leaves
        `ini_status` at pending. The preview path writes the ini and sets
        `ini_status='ok'` too, and leaves `encode_status` at pending so the
        full run still happens. `cmd_ini` stays as the repair command.
    M7  `--skip-static` is the default; `--include-static` turns it off. A
        song with no motion value is encoded, not skipped.

Deterministic bookkeeping only: `build_command` is read as a list, ffmpeg is
never run, and `encode_many` is replaced by a fake that calls `on_done`.
Every path is a `Path`; row lookups compare against `str(Path(...))`.

Run from the repository root:  pytest -q
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from yargvid import cli
from yargvid import encode as enc
from yargvid.db import Database


# ----------------------------------------------------------------- helpers ---

@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


@pytest.fixture
def no_ffmpeg(monkeypatch):
    monkeypatch.setattr(enc, "_supports_fps_mode", lambda: True)


def song(db, path, **cols):
    """A synced, encodable song folder with a song.ini and a source video."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "song.ini").write_text("[song]\nname = X\n", encoding="utf-8")
    src = path / "video.mp4"
    src.write_bytes(b"x")
    db.add_song(path, "Artist", path.name, 100.0)
    base = dict(match_status="ok", download_status="ok", sync_status="ok",
                offset_ms=1234.0, motion=0.5, source_path=str(src))
    base.update(cols)
    db.update(path, **base)
    return path


def row(db, path):
    return db.conn.execute(
        "SELECT * FROM songs WHERE song_dir = ?", (str(path),)
    ).fetchone()


def ini_text(path):
    return (Path(path) / "song.ini").read_text(encoding="utf-8")


def encode_args(**over):
    """The namespace `encode` produces with no flags, before Batch 9's."""
    a = dict(limit=None, sample=False, skip_existing=False, reviewed=False,
             skip_static=True, height=1080, crf=31, cpu_used=3, threads=2,
             workers=None, preview=False, preview_height=480,
             bitrate_cap=None, max_fps=None)
    a.update(over)
    return SimpleNamespace(**a)


class FakePool:
    """Stands in for `encode_many`: records the call, reports each job."""

    def __init__(self):
        self.failing = set()      # folder names that report a failed encode
        self.calls = []

    def __call__(self, jobs, settings, workers=None, on_done=None, **kw):
        self.calls.append(dict(jobs=list(jobs), settings=settings,
                               workers=workers, kw=kw))
        results = {}
        for _src, d in jobs:
            results[d] = ((False, "boom") if d.name in self.failing
                          else (True, ""))
            if on_done:
                on_done(d, results[d])
        return results


@pytest.fixture
def pool(monkeypatch):
    fake = FakePool()
    monkeypatch.setattr(enc, "encode_many", fake)
    return fake


def parsed(monkeypatch, db, argv):
    """Run `main` far enough to parse, capture the encode namespace."""
    seen = {}
    monkeypatch.setattr(cli, "cmd_encode", lambda args, db_: seen.update(vars(args)))
    assert cli.main(["--db", str(db.path), *argv]) == 0
    return SimpleNamespace(**seen)


# ------------------------------------------------------------- M2 flags -----

def test_M2_encode_defaults_to_the_good_tier_and_the_source_fps(db,
                                                                monkeypatch):
    # Batch 11: neither flag carries a value any more. The ceiling comes from
    # --quality, so an explicit 4M can be told from the absence of the flag,
    # and there is no frame-rate cap at all unless one is asked for.
    a = parsed(monkeypatch, db, ["encode"])
    assert a.bitrate_cap is None
    assert cli.encode_settings(a).bitrate_cap == "4M"
    assert a.max_fps is None


def test_M2_encode_cap_and_fps_flags_are_parsed(db, monkeypatch):
    a = parsed(monkeypatch, db,
               ["encode", "--bitrate-cap", "6M", "--max-fps", "24"])
    assert a.bitrate_cap == "6M"
    assert a.max_fps == 24.0
    assert isinstance(a.max_fps, float)


# ------------------------------------------------------------- M1 settings --

def test_M1_cmd_encode_passes_cap_and_fps_into_settings(db, pool):
    song(db, db.path.parent / "s")
    cli.cmd_encode(encode_args(bitrate_cap="3M", max_fps=24.0), db)
    settings = pool.calls[0]["settings"]
    assert settings.bitrate_cap == "3M"
    assert settings.max_fps == 24.0
    # The four it already passed still arrive.
    assert (settings.height, settings.crf, settings.cpu_used,
            settings.threads_per_job) == (1080, 31, 3, 2)


def test_M1_default_settings_emit_crf_and_a_4M_ceiling_never_b_v_0(no_ffmpeg):
    cmd = enc.build_command(Path("in.mp4"), Path("out.webm"),
                            enc.EncodeSettings(), source_fps=30.0)
    assert cmd[cmd.index("-crf") + 1] == "31"
    assert cmd[cmd.index("-b:v") + 1] == "4M"
    assert enc.EncodeSettings().bitrate_cap == "4M"
    # NOTES: `-b:v 0` is a VP9 idiom; on VP8 it silently means 256 kbit/s.
    assert cmd[cmd.index("-b:v") + 1] != "0"


# ------------------------------------------------------------- M4 pool ------

def test_M4_encode_many_forwards_keep_source(monkeypatch, tmp_path):
    seen = {}

    def fake_one(src, song_dir, settings, keep_source=False):
        seen[song_dir] = keep_source
        return True, ""

    monkeypatch.setattr(enc, "encode_one", fake_one)
    d = tmp_path / "s"
    enc.encode_many([(d / "video.mp4", d)], enc.EncodeSettings(), workers=1,
                    keep_source=True)
    assert seen == {d: True}
    enc.encode_many([(d / "video.mp4", d)], enc.EncodeSettings(), workers=1)
    assert seen == {d: False}


def test_M4_preview_runs_through_the_pool_with_workers(db, pool, monkeypatch):
    monkeypatch.setattr(enc, "encode_one",
                        lambda *a, **k: pytest.fail("preview bypassed the pool"))
    song(db, db.path.parent / "s")
    cli.cmd_encode(encode_args(preview=True, workers=3, bitrate_cap="6M"), db)
    call = pool.calls[0]
    assert call["workers"] == 3
    assert call["kw"].get("keep_source") is True
    s = call["settings"]
    assert (s.height, s.cpu_used, s.bitrate_cap) == (480, 5, "800k")


# ------------------------------------------------------------- M5 note ------

def test_M5_success_writes_encode_note_none(db, pool):
    s = song(db, db.path.parent / "s")
    cli.cmd_encode(encode_args(), db)
    r = row(db, s)
    assert r["encode_status"] == "ok"
    assert r["encode_note"] is None


# ------------------------------------------------------------- M6 ini -------

def test_M6_full_encode_writes_ini_on_success_only(db, pool):
    lib = db.path.parent
    good = song(db, lib / "good", offset_ms=-2067.0)
    bad = song(db, lib / "bad")
    pool.failing.add("bad")
    cli.cmd_encode(encode_args(), db)

    r = row(db, good)
    assert r["encode_status"] == "ok" and r["ini_status"] == "ok"
    assert "video_start_time = -2067" in ini_text(good)

    r = row(db, bad)
    assert r["encode_status"] == "failed" and r["encode_note"] == "boom"
    assert r["ini_status"] == "pending"
    assert "video_start_time" not in ini_text(bad)

    # Nothing is left for `ini` to repair after a successful encode.
    assert db.pending("ini") == []


def test_M6_preview_writes_ini_ok_and_leaves_encode_pending(db, pool):
    s = song(db, db.path.parent / "s", offset_ms=500.0)
    cli.cmd_encode(encode_args(preview=True), db)
    r = row(db, s)
    assert r["ini_status"] == "ok"
    assert "video_start_time = 500" in ini_text(s)
    # A preview is not the encode; the full run must still pick it up.
    assert r["encode_status"] == "pending"


def test_M6_an_unwritable_ini_leaves_the_repair_to_ini_and_the_run_going(
        db, pool, monkeypatch):
    """A song.ini that cannot be written must not abandon the rest of the run.

    `progress` is called from `encode_many`'s result loop, which sits outside
    that loop's own try/except and inside the pool's `with` block. An
    exception there cancels nothing: every queued song still encodes and still
    loses its source, and none of them is ever recorded. The encode is on
    disk, so the row stays 'ok' and `ini` - the repair command - owes it the
    one file that could not be written.
    """
    lib = db.path.parent
    locked = song(db, lib / "a_locked", offset_ms=100.0)
    later = song(db, lib / "b_later", offset_ms=200.0)

    real = cli.write_video_start_time

    def refuse_one(song_dir, value, backup=True):
        if Path(song_dir).name == "a_locked":
            raise PermissionError(5, "Access is denied")
        return real(song_dir, value, backup)

    monkeypatch.setattr(cli, "write_video_start_time", refuse_one)
    cli.cmd_encode(encode_args(), db)

    r = row(db, locked)
    assert r["encode_status"] == "ok"        # the webm is on disk
    assert r["ini_status"] == "pending"      # ... and only the ini is owed
    assert "video_start_time" not in ini_text(locked)
    assert [x["song_dir"] for x in db.pending("ini")] == [str(locked)]

    # The song queued behind it was still recorded.
    r = row(db, later)
    assert r["encode_status"] == "ok" and r["ini_status"] == "ok"
    assert "video_start_time = 200" in ini_text(later)


# ------------------------------------------------------------- M7 static ----

def test_M7_skip_static_is_the_default(db, monkeypatch):
    assert parsed(monkeypatch, db, ["encode"]).skip_static is True


def test_M7_include_static_turns_it_off(db, monkeypatch):
    assert parsed(monkeypatch, db,
                  ["encode", "--include-static"]).skip_static is False


def test_M7_default_run_skips_static_and_keeps_unmeasured(db, pool):
    lib = db.path.parent
    still = song(db, lib / "still", motion=0.01)
    unmeasured = song(db, lib / "unmeasured", motion=None)
    footage = song(db, lib / "footage", motion=0.9)
    assert cli.main(["--db", str(db.path), "encode"]) == 0
    dirs = {d for _src, d in pool.calls[0]["jobs"]}
    assert dirs == {unmeasured, footage}
    assert row(db, still)["encode_status"] == "pending"
    assert row(db, still)["ini_status"] == "pending"
