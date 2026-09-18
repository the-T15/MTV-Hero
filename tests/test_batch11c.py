"""
Batch 11c - one sample for every measurement.

`estimate --measure` drew three random songs per call, and a different three
each time. On the real library that put "better" above "best" in one round
and moved every tier by 25-35% between rounds, because busy footage costs
more bits than calm footage and the draw decided which the tier got. The
numbers could not be compared, so the tiers could not be ranked from them.

    S1  The sample is a fixed set. `encode.sample_songs(rows, n)` returns
        the `n` rows whose song folder name hashes lowest (SHA-1 of the
        folder name, ties broken by the full path), in that order, whatever
        order the rows arrive in. Every settings row measured on one
        library therefore sees the same songs, and a re-run sees them
        again. `encode.SAMPLE_SONGS` is 50, the default `n`. Approving one
        more song can swap at most one member of the sample.
    S2  `estimate` measures that sample. The random draw is gone;
        `estimate --sample-size N` sets `n` (default `SAMPLE_SONGS`, must be
        positive, `estimate`'s flag only). The "Measuring ..." line names up
        to five of the songs and counts the rest, and says the set is the
        same every run.
    S3  A measurement reports its rate and its time. After measuring,
        `estimate` prints the measured rate in Mbit/s (`2.00 Mbit/s` for
        2e6), how long the sample encode took, and the time a full run of
        every song to be encoded would take at that speed - wall time of
        the sample scaled by seconds of video in the run over seconds of
        video in the sample. `encode.sample_seconds(rows)` is the seconds a
        sample of those rows encodes (`SAMPLE_SECONDS` or the whole song if
        shorter), `encode.projected_seconds(wall, sampled, total)` is the
        scaling, and `encode.duration_text(seconds)` is how both are
        printed: `45 s` below 90 s, `5 min` below 90 min, `1.5 h` above.
        Nothing about the time is printed when nothing was measured.

Nothing shells out: `measure_rate` is faked and the clock is faked. Every
path is a `Path`.

Run from the repository root:  pytest -q
"""

from __future__ import annotations

import random
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


def rows_of(db, paths):
    return [db.conn.execute("SELECT * FROM songs WHERE song_dir = ?",
                            (str(p),)).fetchone() for p in paths]


def fake_rows(names, parent="lib"):
    """Rows for `sample_songs`, which reads only `song_dir`."""
    return [{"song_dir": str(Path(parent) / n), "video_seconds": 200.0}
            for n in names]


def estimate_args(**over):
    a = dict(limit=None, sample=False, skip_existing=False, reviewed=False,
             skip_static=True, height=1080, crf=None, cpu_used=3, threads=2,
             workers=None, preview=False, preview_height=480,
             bitrate_cap=None, max_fps=None, codec="vp8", fps=None,
             size_lock=None, quality=None, measure=False,
             sample_size=enc.SAMPLE_SONGS)
    a.update(over)
    return SimpleNamespace(**a)


def parsed(monkeypatch, db, argv, cmd):
    seen = {}
    monkeypatch.setattr(cli, cmd, lambda args, db_: seen.update(vars(args)))
    assert cli.main(["--db", str(db.path), *argv]) == 0
    return SimpleNamespace(**seen)


def record_measure(monkeypatch, bps=2_000_000.0):
    """Fake `measure_rate` that records the rows it was handed."""
    sampled = []

    def measure(rows, settings, workers=None):
        sampled.append(list(rows))
        return bps

    monkeypatch.setattr(enc, "measure_rate", measure)
    return sampled


def fake_clock(monkeypatch, *readings):
    """`time.perf_counter` in `cli` returns these readings in turn."""
    it = iter(readings)
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(it))


def names(rows):
    return [Path(r["song_dir"]).name for r in rows]


# ------------------------------------------------------------- S1 fixed -----

def test_S1_fifty_is_the_default_sample():
    assert enc.SAMPLE_SONGS == 50


def test_S1_sample_songs_is_the_same_set_in_the_same_order_every_call():
    rows = fake_rows(f"song{i:02d}" for i in range(60))
    first = names(enc.sample_songs(rows, 50))
    assert len(first) == 50 and len(set(first)) == 50
    shuffled = list(rows)
    random.Random(7).shuffle(shuffled)
    assert names(enc.sample_songs(shuffled, 50)) == first
    assert names(enc.sample_songs(list(reversed(rows)), 50)) == first


def test_S1_sample_songs_takes_all_when_there_are_fewer_than_n():
    rows = fake_rows(["a", "b", "c"])
    assert sorted(names(enc.sample_songs(rows, 50))) == ["a", "b", "c"]
    assert sorted(names(enc.sample_songs(rows))) == ["a", "b", "c"]
    assert enc.sample_songs([], 50) == []
    assert len(enc.sample_songs(rows, 2)) == 2


def test_S1_sample_songs_defaults_to_sample_songs_constant():
    rows = fake_rows(f"song{i:02d}" for i in range(80))
    assert len(enc.sample_songs(rows)) == enc.SAMPLE_SONGS


def test_S1_the_pick_is_keyed_on_the_folder_name_not_the_library_path():
    here = fake_rows((f"song{i:02d}" for i in range(30)), parent="C:/lib")
    there = fake_rows((f"song{i:02d}" for i in range(30)), parent="D:/other")
    assert names(enc.sample_songs(here, 10)) == names(enc.sample_songs(there, 10))
    # And it is not the first n in name order - a hash, not a sort.
    assert names(enc.sample_songs(here, 10)) != sorted(names(here))[:10]


def test_S1_the_pick_is_a_hash_of_the_folder_name():
    # Pinned so a re-run on the real library picks what the last one did.
    picked = names(enc.sample_songs(fake_rows(["Alpha", "Beta", "Gamma",
                                               "Delta", "Epsilon"]), 2))
    assert picked == ["Alpha", "Epsilon"]


def test_S1_two_folders_with_one_name_both_stay_in_a_fixed_order():
    rows = fake_rows(["same"], parent="A") + fake_rows(["same"], parent="B")
    picked = enc.sample_songs(rows, 2)
    assert [r["song_dir"] for r in picked] == [str(Path("A") / "same"),
                                              str(Path("B") / "same")]
    assert [r["song_dir"] for r in enc.sample_songs(rows[::-1], 2)] == \
        [r["song_dir"] for r in picked]


def test_S1_approving_one_more_song_swaps_at_most_one_member():
    rows = fake_rows(f"song{i:03d}" for i in range(190))
    before = set(names(enc.sample_songs(rows, 50)))
    for extra in ("New Song", "Another", "Zzz Top"):
        after = set(names(enc.sample_songs(rows + fake_rows([extra]), 50)))
        assert len(after) == 50
        assert len(before - after) <= 1 and len(after - before) <= 1


# ------------------------------------------------------------- S2 estimate --

def test_S2_estimate_measures_the_fixed_sample_after_the_filters(
        db, monkeypatch, no_typical):
    lib = db.path.parent
    stills = [song(db, lib / f"still{i:02d}", motion=0.01) for i in range(10)]
    footage = [song(db, lib / f"footage{i:02d}") for i in range(60)]
    sampled = record_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(), db)
    assert len(sampled) == 1 and len(sampled[0]) == 50
    got = [Path(r["song_dir"]) for r in sampled[0]]
    assert set(got) <= set(footage) and not set(got) & set(stills)
    # Exactly `sample_songs` over the rows the run would encode, in order.
    assert got == [Path(r["song_dir"])
                   for r in enc.sample_songs(rows_of(db, footage), 50)]


def test_S2_two_measure_runs_pick_the_same_songs(db, monkeypatch):
    lib = db.path.parent
    for i in range(20):
        song(db, lib / f"s{i:02d}")
    sampled = record_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(measure=True, sample_size=8), db)
    cli.cmd_estimate(estimate_args(measure=True, sample_size=8), db)
    assert len(sampled) == 2
    assert names(sampled[0]) == names(sampled[1]) and len(sampled[0]) == 8
    # A different quality on the same library sees the same eight.
    cli.cmd_estimate(estimate_args(measure=True, sample_size=8, crf=12), db)
    assert names(sampled[2]) == names(sampled[0])


def test_S2_estimate_does_not_draw_at_random(db, monkeypatch, no_typical):
    lib = db.path.parent
    for i in range(6):
        song(db, lib / f"s{i}")
    monkeypatch.setattr(random, "sample",
                        lambda *a, **k: pytest.fail("random draw"))
    monkeypatch.setattr(random, "shuffle",
                        lambda *a, **k: pytest.fail("random draw"))
    record_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(sample_size=3), db)


def test_S2_sample_size_flag(db, monkeypatch):
    a = parsed(monkeypatch, db, ["estimate"], "cmd_estimate")
    assert a.sample_size == enc.SAMPLE_SONGS == 50
    a = parsed(monkeypatch, db, ["estimate", "--sample-size", "8"],
               "cmd_estimate")
    assert a.sample_size == 8
    a = parsed(monkeypatch, db, ["estimate", "--measure", "--sample-size", "8",
                                 "--quality", "best"], "cmd_estimate")
    assert (a.sample_size, a.measure, a.quality) == (8, True, "best")
    for bad in ("0", "-3", "eight"):
        with pytest.raises(SystemExit):
            cli.main(["--db", str(db.path), "estimate", "--sample-size", bad])
    with pytest.raises(SystemExit):
        cli.main(["--db", str(db.path), "encode", "--sample-size", "8"])


def test_S2_estimate_honours_the_sample_size(db, monkeypatch, no_typical):
    lib = db.path.parent
    all_rows = rows_of(db, [song(db, lib / f"s{i:02d}") for i in range(12)])
    sampled = record_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(sample_size=4), db)
    assert names(sampled[0]) == names(enc.sample_songs(all_rows, 4))
    cli.cmd_estimate(estimate_args(sample_size=4, measure=True), db)
    assert names(sampled[1]) == names(sampled[0])


def test_S2_an_args_object_without_sample_size_still_works(db, monkeypatch,
                                                            no_typical):
    # The GUI and older callers build args by hand; the default applies.
    lib = db.path.parent
    for i in range(3):
        song(db, lib / f"s{i}")
    args = estimate_args()
    del args.sample_size
    sampled = record_measure(monkeypatch)
    cli.cmd_estimate(args, db)
    assert len(sampled[0]) == 3


def test_S2_the_measuring_line_names_five_and_counts_the_rest(
        db, monkeypatch, no_typical, capsys):
    lib = db.path.parent
    all_rows = rows_of(db, [song(db, lib / f"song{i:02d}") for i in range(8)])
    record_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(sample_size=8), db)
    out = capsys.readouterr().out
    picked = names(enc.sample_songs(all_rows, 8))
    assert f"{enc.SAMPLE_SECONDS:.0f} s from each of 8 songs" in out
    for n in picked[:5]:
        assert n in out
    for n in picked[5:]:
        assert n not in out
    assert "and 3 more" in out
    assert "same" in out.lower() and "every run" in out.lower()


def test_S2_a_small_sample_is_named_in_full(db, monkeypatch, no_typical,
                                            capsys):
    lib = db.path.parent
    for n in ("a", "b", "c"):
        song(db, lib / n)
    record_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert "3 songs" in out and "more" not in out


# ------------------------------------------------------------- S3 report ----

def test_S3_duration_text():
    assert enc.duration_text(0.0) == "0 s"
    assert enc.duration_text(45.0) == "45 s"
    assert enc.duration_text(89.9) == "90 s"
    assert enc.duration_text(90.0) == "2 min"
    assert enc.duration_text(300.0) == "5 min"
    assert enc.duration_text(89 * 60.0) == "89 min"
    assert enc.duration_text(90 * 60.0) == "1.5 h"
    assert enc.duration_text(5400.0) == "1.5 h"
    assert enc.duration_text(4 * 3600.0) == "4.0 h"


def test_S3_sample_seconds_is_what_measure_rate_encodes():
    rows = [{"song_dir": "lib/long", "video_seconds": 200.0},
            {"song_dir": "lib/short", "video_seconds": 12.0},
            {"song_dir": "lib/none", "video_seconds": None},
            {"song_dir": "lib/zero", "video_seconds": 0.0}]
    assert enc.sample_seconds(rows) == pytest.approx(enc.SAMPLE_SECONDS + 12.0)
    assert enc.sample_seconds([]) == 0.0


def test_S3_projected_seconds_scales_the_wall_time_by_video_seconds():
    # 30 s of wall time to encode 32 s of video; 212 s of video to do.
    assert enc.projected_seconds(30.0, 32.0, 212.0) == \
        pytest.approx(30.0 * 212.0 / 32.0)
    assert enc.projected_seconds(30.0, 600.0, 600.0) == pytest.approx(30.0)
    assert enc.projected_seconds(30.0, 0.0, 600.0) == 0.0


def test_S3_estimate_prints_the_measured_rate_and_time(db, monkeypatch,
                                                       no_typical, capsys):
    lib = db.path.parent
    for n in ("a", "b", "c"):
        song(db, lib / n)                       # 3 x 200 s = 600 s of video
    record_measure(monkeypatch, 2_000_000.0)
    fake_clock(monkeypatch, 100.0, 130.0)       # the sample took 30 s
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert "~ 0.15 GB (max 0.30 GB)" in out and "[measured]" in out
    assert "2.00 Mbit/s" in out
    assert "3 songs" in out
    # 30 s for 60 s of video; 600 s of video is 300 s: "5 min".
    assert "30 s" in out
    assert "5 min" in out


def test_S3_the_projection_covers_every_song_in_the_run(db, monkeypatch,
                                                        no_typical, capsys):
    lib = db.path.parent
    for i in range(12):
        song(db, lib / f"s{i:02d}", video_seconds=300.0)   # 3600 s of video
    record_measure(monkeypatch, 2_000_000.0)
    fake_clock(monkeypatch, 0.0, 40.0)          # 40 s for 4 x 20 s of video
    cli.cmd_estimate(estimate_args(sample_size=4), db)
    out = capsys.readouterr().out
    assert "40 s" in out
    assert enc.duration_text(40.0 * 3600.0 / 80.0) in out    # 30 min


def test_S3_measure_prints_rate_and_time_when_a_rate_was_stored(
        db, monkeypatch, capsys):
    lib = db.path.parent
    for n in ("a", "b", "c"):
        song(db, lib / n)
    db.set_rate(enc.rate_key(enc.EncodeSettings()), 3_000_000.0)
    record_measure(monkeypatch, 2_000_000.0)
    fake_clock(monkeypatch, 0.0, 30.0)
    cli.cmd_estimate(estimate_args(measure=True), db)
    out = capsys.readouterr().out
    assert "2.00 Mbit/s" in out and "5 min" in out
    assert db.get_rate(enc.rate_key(enc.EncodeSettings())) == 2_000_000.0


def test_S3_nothing_about_time_is_printed_without_a_measurement(
        db, monkeypatch, capsys):
    lib = db.path.parent
    for n in ("a", "b", "c"):
        song(db, lib / n)
    monkeypatch.setattr(enc, "measure_rate",
                        lambda *a, **k: pytest.fail("measured"))
    fake_clock(monkeypatch, *([0.0] * 8))
    # Typical figure.
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert "Mbit/s" not in out and "took" not in out
    # Stored figure.
    db.set_rate(enc.rate_key(enc.EncodeSettings()), 3_000_000.0)
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert "Mbit/s" not in out and "took" not in out
    # Size lock.
    cli.cmd_estimate(estimate_args(size_lock="2M"), db)
    out = capsys.readouterr().out
    assert "Mbit/s" not in out and "took" not in out


def test_S3_a_failed_sample_prints_no_rate_or_time(db, monkeypatch,
                                                   no_typical, capsys):
    lib = db.path.parent
    for n in ("a", "b", "c"):
        song(db, lib / n)
    record_measure(monkeypatch, 0.0)
    fake_clock(monkeypatch, 0.0, 30.0)
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert "produced nothing" in out
    assert "Mbit/s" not in out and "took" not in out
