"""
Batch 10b - a fast estimate.

`estimate` measured its rate by encoding three whole songs, which on the real
library meant waiting as long as three encodes for a number that is a
per-second figure. Three changes, all inside `estimate`'s measurement; the
printed line and everything Batch 10 pinned stay as they are.

    Q1  A measurement encodes a slice, not the song. `EncodeSettings.clip`
        is `(start_seconds, length_seconds) | None`; `build_command` puts
        `-ss <start>` before `-i` and `-t <length>` after it, and nothing
        else changes. `encode_many` accepts a job as `(src, song_dir)` or
        `(src, song_dir, settings)`, the third element replacing the call's
        settings for that job, so one pool call can carry a different clip
        per song. `measure_rate` submits every sample in ONE call, taking
        `SAMPLE_SECONDS` (20.0) from the middle of each song - the whole
        song when it is shorter - and divides the bytes written by the
        seconds actually encoded. `clip` is not part of `rate_key`.
    Q2  A measurement is remembered. `Database.get_rate(key)` and
        `Database.set_rate(key, bps)` keep measured rates in a `rates` table
        keyed exactly as `rate_key`, so a setting is measured once per
        database, not once per process. `estimate` checks the in-process
        table, then the database, and stores what it measures in both.
        Song rows are still never written.
    Q3  A first answer needs no measurement. `encode.TYPICAL_RATES` maps
        (codec, height, effective crf) to a typical bits-per-second figure,
        seeded from the 2026-09-17 measurements on the real library (vp8
        1080p crf 31: 2.9 Mbit/s; h264_nvenc 1080p cq 23: 3.4 Mbit/s).
        `typical_rate(settings)` looks a settings object up there. With no
        stored rate, `estimate` prints the typical figure at once, labelled
        `typical`, and says `estimate --measure` gives a measured one; a
        stored rate prints as `measured`; with neither typical nor stored it
        measures. `estimate --measure` always measures and stores. The flag
        is `estimate`'s only; `encode` does not take it.

Nothing shells out: `encode_many` is faked, `build_command` is read as a
list. Every path is a `Path`.

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


@pytest.fixture(autouse=True)
def fresh_tables(monkeypatch):
    monkeypatch.setattr(enc, "RATE_TABLE", {})
    monkeypatch.setattr(enc, "_supports_fps_mode", lambda: True)


@pytest.fixture
def no_typical(monkeypatch):
    monkeypatch.setattr(enc, "TYPICAL_RATES", {})


def song(db, path, **cols):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "song.ini").write_text("[song]\nname = X\n", encoding="utf-8")
    src = path / "video.mp4.src"
    src.write_bytes(b"x")
    db.add_song(path, "Artist", path.name, 100.0)
    base = dict(match_status="ok", download_status="ok", sync_status="ok",
                offset_ms=1234.0, motion=0.5, source_path=str(src),
                video_seconds=200.0)
    base.update(cols)
    db.update(path, **base)
    return path


def row(db, path):
    return db.conn.execute(
        "SELECT * FROM songs WHERE song_dir = ?", (str(path),)
    ).fetchone()


def estimate_args(**over):
    a = dict(limit=None, sample=False, skip_existing=False, reviewed=False,
             skip_static=True, height=1080, crf=None, cpu_used=3, threads=2,
             workers=None, preview=False, preview_height=480,
             bitrate_cap=None, max_fps=None, codec="vp8", fps=None,
             size_lock=None, measure=False)
    a.update(over)
    return SimpleNamespace(**a)


def parsed(monkeypatch, db, argv, cmd):
    seen = {}
    monkeypatch.setattr(cli, cmd, lambda args, db_: seen.update(vars(args)))
    assert cli.main(["--db", str(db.path), *argv]) == 0
    return SimpleNamespace(**seen)


def three_songs(db):
    return [song(db, db.path.parent / n) for n in ("a", "b", "c")]   # 600 s


def never_measure(monkeypatch):
    monkeypatch.setattr(enc, "measure_rate",
                        lambda *a, **k: pytest.fail("measured"))


# ------------------------------------------------------------- Q1 slice -----

def test_Q1_clip_puts_ss_before_input_and_t_after_it():
    assert enc.EncodeSettings().clip is None
    assert enc.SAMPLE_SECONDS == 20.0
    src, dst = Path("in.mp4"), Path("out.webm")
    plain = enc.build_command(src, dst, enc.EncodeSettings(), 30.0)
    assert "-ss" not in plain and "-t" not in plain

    cmd = enc.build_command(src, dst, enc.EncodeSettings(clip=(90.0, 20.0)),
                            30.0)
    i = cmd.index("-i")
    assert cmd.index("-ss") < i and float(cmd[cmd.index("-ss") + 1]) == 90.0
    assert cmd.index("-t") > i and float(cmd[cmd.index("-t") + 1]) == 20.0
    # Only the two flags were added.
    assert [x for x in cmd if x not in ("-ss", "90.000", "-t", "20.000")] == \
        [x for x in plain]


def test_Q1_clip_is_not_part_of_the_rate_key():
    assert (enc.rate_key(enc.EncodeSettings(clip=(90.0, 20.0)))
            == enc.rate_key(enc.EncodeSettings()))


def test_Q1_encode_many_takes_per_job_settings(monkeypatch, tmp_path):
    seen = {}

    def fake_one(src, song_dir, settings, keep_source=False):
        seen[song_dir] = settings
        return True, ""

    monkeypatch.setattr(enc, "encode_one", fake_one)
    a, b = tmp_path / "a", tmp_path / "b"
    base = enc.EncodeSettings()
    special = enc.EncodeSettings(clip=(1.0, 2.0))
    enc.encode_many([(a / "v", a), (b / "v", b, special)], base, workers=1)
    assert seen[a] is base and seen[b] is special


def test_Q1_measure_rate_encodes_middle_slices_in_one_pool_call(db,
                                                                monkeypatch):
    lib = db.path.parent
    long = row(db, song(db, lib / "long"))                       # 200 s
    short = row(db, song(db, lib / "short", video_seconds=12.0))
    calls = []

    def fake_many(jobs, settings, workers=None, on_done=None, **kw):
        calls.append(list(jobs))
        out = {}
        for job in jobs:
            d = job[1]
            each = job[2] if len(job) > 2 else settings
            enc.output_path(d, each).write_bytes(b"x" * 4_000)
            out[d] = (True, "")
        return out

    monkeypatch.setattr(enc, "encode_many", fake_many)
    bps = enc.measure_rate([long, short], enc.EncodeSettings(), workers=2)
    assert len(calls) == 1 and len(calls[0]) == 2       # parallel, not serial
    clips = {job[1].name: job[2].clip for job in calls[0]}
    assert clips == {"long": (90.0, 20.0),              # middle 20 s of 200
                     "short": (0.0, 12.0)}              # the whole short song
    # Bytes over seconds ENCODED (20 + 12), not over the songs' 212.
    assert bps == pytest.approx(8 * 8_000 / 32.0)
    # Per-job settings differ from the caller's only by clip.
    for job in calls[0]:
        assert (job[2].codec, job[2].height) == ("vp8", 1080)


# ------------------------------------------------------------- Q2 remember --

def test_Q2_rates_persist_in_the_database(tmp_path):
    key = enc.rate_key(enc.EncodeSettings(codec="h264", height=720))
    d = Database(tmp_path / "r.sqlite")
    assert d.get_rate(key) is None
    d.set_rate(key, 1_500_000.0)
    assert d.get_rate(key) == 1_500_000.0
    d.set_rate(key, 1_600_000.0)
    assert d.get_rate(key) == 1_600_000.0               # overwrite, not add
    assert d.get_rate(enc.rate_key(enc.EncodeSettings())) is None
    d.close()

    d = Database(tmp_path / "r.sqlite")                 # survives reopening
    assert d.get_rate(key) == 1_600_000.0
    d.close()


def test_Q2_estimate_stores_what_it_measures(db, monkeypatch, no_typical,
                                             capsys):
    three_songs(db)
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: 2_000_000.0)
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert "~ 0.15 GB (max 0.30 GB)" in out and "measured" in out
    key = enc.rate_key(enc.EncodeSettings())
    assert db.get_rate(key) == 2_000_000.0
    assert enc.RATE_TABLE[key] == 2_000_000.0


def test_Q2_a_stored_rate_is_used_without_measuring(db, monkeypatch, capsys):
    three_songs(db)
    db.set_rate(enc.rate_key(enc.EncodeSettings()), 2_000_000.0)
    never_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert "~ 0.15 GB (max 0.30 GB)" in out
    assert "measured" in out and "typical" not in out


def test_Q2_song_rows_are_still_never_written(db, monkeypatch, no_typical):
    s = three_songs(db)[0]
    before = tuple(row(db, s))
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: 2_000_000.0)
    cli.cmd_estimate(estimate_args(), db)
    assert tuple(row(db, s)) == before
    assert sorted(p.name for p in s.iterdir()) == ["song.ini", "video.mp4.src"]


# ------------------------------------------------------------- Q3 typical ---

def test_Q3_typical_rates_are_seeded_from_the_library_measurements():
    assert enc.TYPICAL_RATES[("vp8", 1080, 31)] == pytest.approx(2.4e6)
    assert enc.TYPICAL_RATES[("h264_nvenc", 1080, 23)] == pytest.approx(3.4e6)
    assert enc.typical_rate(enc.EncodeSettings()) == pytest.approx(2.4e6)
    assert enc.typical_rate(enc.EncodeSettings(crf=31)) == pytest.approx(2.4e6)
    assert enc.typical_rate(enc.EncodeSettings(crf=40)) is None


def test_Q3_a_typical_rate_answers_at_once_and_says_so(db, monkeypatch,
                                                       capsys):
    three_songs(db)
    monkeypatch.setattr(enc, "TYPICAL_RATES", {("vp8", 1080, 31): 2_000_000.0})
    never_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert "~ 0.15 GB (max 0.30 GB)" in out
    assert "typical" in out and "--measure" in out
    assert db.get_rate(enc.rate_key(enc.EncodeSettings())) is None


def test_Q3_a_stored_rate_beats_a_typical_one(db, monkeypatch, capsys):
    three_songs(db)
    monkeypatch.setattr(enc, "TYPICAL_RATES", {("vp8", 1080, 31): 4_000_000.0})
    db.set_rate(enc.rate_key(enc.EncodeSettings()), 2_000_000.0)
    never_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(), db)
    assert "~ 0.15 GB (max 0.30 GB)" in capsys.readouterr().out


def test_Q3_no_typical_and_no_stored_rate_measures(db, monkeypatch, capsys):
    three_songs(db)
    monkeypatch.setattr(enc, "TYPICAL_RATES", {})
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: 2_000_000.0)
    cli.cmd_estimate(estimate_args(crf=40), db)
    assert "~ 0.15 GB (max 0.30 GB)" in capsys.readouterr().out


def test_Q3_measure_flag_measures_over_anything_stored_or_typical(
        db, monkeypatch, capsys):
    three_songs(db)
    monkeypatch.setattr(enc, "TYPICAL_RATES", {("vp8", 1080, 31): 4_000_000.0})
    key = enc.rate_key(enc.EncodeSettings())
    db.set_rate(key, 3_000_000.0)
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: 2_000_000.0)
    cli.cmd_estimate(estimate_args(measure=True), db)
    assert "~ 0.15 GB (max 0.30 GB)" in capsys.readouterr().out
    assert db.get_rate(key) == 2_000_000.0


def test_Q3_measure_is_estimates_flag_only(db, monkeypatch):
    a = parsed(monkeypatch, db, ["estimate"], "cmd_estimate")
    assert a.measure is False
    a = parsed(monkeypatch, db, ["estimate", "--measure"], "cmd_estimate")
    assert a.measure is True
    with pytest.raises(SystemExit):
        cli.main(["--db", str(db.path), "encode", "--measure"])


def test_Q3_size_lock_still_needs_no_rate_at_all(db, monkeypatch, capsys):
    three_songs(db)
    never_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(size_lock="2M"), db)
    out = capsys.readouterr().out
    assert "~ 0.15 GB (max 0.15 GB)" in out
    assert "typical" not in out
    assert db.get_rate(enc.rate_key(enc.EncodeSettings(size_lock="2M"))) is None
