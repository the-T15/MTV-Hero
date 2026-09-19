"""
Batch 13 - the tier per song, and the floor.

Batch 11 made `--height` a ceiling and gave the encoder four quality tiers;
Batch 12 recorded what each source is. Nothing joined the two: `encode
--quality better` spent the same bits on a 268-high source as on a 2160
one. Five things, in the order the letters below take them.

    T   THE TIER FOLLOWS THE SOURCE. `encode.TIER_SOURCE_HEIGHT` (1080,
        moved from cli.py) is the source height the typed tier is FOR, and
        `encode.tier_for_source(quality, source_height)` returns a tier
        NAME: the typed tier at 1080p, one step down the ladder below it,
        one step up above it, stopping at `good` and `super`. No typed tier
        is `good`. A song with no recorded size takes the typed tier as it
        is - an unknown source is not a reason to encode something quietly
        cheaper. `cmd_encode` builds one settings object per row and hands
        the pool `(src, song_dir, settings)` jobs; a typed `--crf` or
        `--bitrate-cap` holds for every song and only the halves the tier
        supplied move. `--preview` and `--size-lock` runs read no tier and
        hand every song the run's settings unchanged. The rule is on by
        default: it is what `--quality` now means.

    E   ESTIMATE SUMS PER TIER. Songs are grouped by the settings they
        would really encode under (`rate_key` of the per-song settings),
        each group gets its own bits-per-second figure the way one run did
        before - process table, database, typical, sample - and the total
        is the sum. One group prints exactly as before; several print a
        line each (tier, songs, hours, ~ GB, max, label) and then one
        blended `~ X GB (max Y GB)`, because "how much disk" wants one
        number. `--measure` measures each group on its own `sample_songs`
        pick and stores each key.

    L   THE LINES SAY THE MIX. `cli.tier_mix_line(rows, quality)` is one
        sentence on how many songs take each tier at that typed quality,
        naming songs with no recorded size as a gap; `encode` and
        `estimate` print it for the run and `videos` for the approved
        songs. `cli.source_size_line` keeps counting sources above 1080p
        but no longer credits source height with deciding the tier: it
        decides the height ceiling, and a bigger source comes down to it
        with detail to spare.

    F   THE FLOOR. `encode.MIN_HEIGHT` is 720. A source below it is
        enlarged until it fits 1280x720 with its shape kept, then padded
        with black to 16:9; at and above it the never-enlarge rule stands.
        It is an EXPRESSION in `video_filter` - the scale box becomes
        min(W, max(iw, 1280)) x min(H, max(ih, 720)) - never a number
        probed off the source, so `rate_key` still carries no source
        dimension. The ceiling wins where the two meet: a 480p preview is
        still 480p. Jack's decision, 2026-09-19: a 268-high background
        shipped at 640x360 for the game to stretch is worse than 1280x720
        with bars.

    S   `set` CLEARS WHAT `match` CLEARS. `cmd_set` left `source_height`,
        `source_max_height` and `video_seconds` behind for the new pick,
        and `--upgrade` reads a stale size as a reason to re-fetch. The
        two field lists become one: `cmd_set` requeues through
        `_requeue_after_match` and then writes only the match columns.

Nothing here reaches YouTube, and ffmpeg is used only by the F tests that
are marked for it: `encode_many` is a fake pool, `match.fetch_metadata` is
faked, `measure_rate` is faked. Every path is a `Path`. `estimate` may not
print the word "more" on a small sample (`test_batch11c.py` S2).

Older tests adjusted by this batch, all mechanically: the fake pools in
`test_batch9.py`, `test_batch10.py`, `test_batch11.py` and `test_batch12.py`
unpack jobs as pairs and now read `job[1]`; `test_batch11.py`'s S1-S4 pins
of the 640x480 and 320x240 outputs move to the floored sizes.

Run from the repository root:  pytest -q
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from yargvid import cli
from yargvid import encode as enc
from yargvid import match as mt
from yargvid.db import Database


# ----------------------------------------------------------------- helpers ---

needs_ffmpeg = pytest.mark.skipif(
    not (enc.have("ffmpeg") and enc.have("ffprobe")),
    reason="the filter chain is only pinned by what ffmpeg makes of it",
)


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


@pytest.fixture(autouse=True)
def fresh_tables(monkeypatch):
    monkeypatch.setattr(enc, "RATE_TABLE", {})
    monkeypatch.setattr(enc, "_supports_fps_mode", lambda: True)


def song(db, path, **cols):
    """A downloaded, synced, approved, unencoded song with a file on disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "song.ini").write_text("[song]\nname = X\n", encoding="utf-8")
    src = path / "video.src.mkv"
    src.write_bytes(b"x")
    db.add_song(path, "Artist", path.name, 100.0)
    base = dict(match_status="ok", video_id="aaaaaaaaaaa",
                download_status="ok", source_path=str(src),
                sync_status="ok", offset_ms=1234.0, motion=0.5,
                video_seconds=200.0, review="keep")
    base.update(cols)
    db.update(path, **base)
    return path


def row(db, path):
    return db.conn.execute(
        "SELECT * FROM songs WHERE song_dir = ?", (str(path),)).fetchone()


def rows_of(db, *paths):
    return [row(db, p) for p in paths]


def encode_args(**over):
    a = dict(limit=None, sample=False, skip_existing=False, reviewed=False,
             skip_static=True, height=1080, crf=None, cpu_used=3, threads=2,
             workers=None, preview=False, preview_height=480,
             bitrate_cap=None, max_fps=None, codec="vp8", fps=None,
             size_lock=None, quality=None, keep_source=False)
    a.update(over)
    return SimpleNamespace(**a)


def estimate_args(**over):
    a = encode_args(measure=False, sample_size=enc.SAMPLE_SONGS)
    return SimpleNamespace(**{**vars(a), **over})


class FakePool:
    """Stands in for `encode_many`: records the call, reports each job."""

    def __init__(self):
        self.calls = []

    def __call__(self, jobs, settings, workers=None, on_done=None, **kw):
        self.calls.append(dict(jobs=list(jobs), settings=settings, kw=kw))
        results = {}
        for job in jobs:
            d = job[1]
            results[d] = (True, "")
            if on_done:
                on_done(d, results[d])
        return results


@pytest.fixture
def pool(monkeypatch):
    fake = FakePool()
    monkeypatch.setattr(enc, "encode_many", fake)
    monkeypatch.setattr(enc, "resolve_codec", lambda s, say=print: s)
    return fake


def job_settings(pool):
    """{song folder name: the settings that job was handed}."""
    call = pool.calls[0]
    return {job[1].name: job[2] for job in call["jobs"]}


def never_measure(monkeypatch):
    monkeypatch.setattr(enc, "measure_rate",
                        lambda *a, **k: pytest.fail("measured"))


def three_tiers(db, lib):
    """One song below the threshold, one at it, one above."""
    return (song(db, lib / "low", source_height=720),
            song(db, lib / "mid", source_height=1080),
            song(db, lib / "high", source_height=2160))


# ============================================ T  the tier per song ==========

def test_T1_the_threshold_moved_into_encode():
    assert enc.TIER_SOURCE_HEIGHT == 1080
    # cli reads encode's number rather than keeping one of its own.
    assert getattr(cli, "TIER_SOURCE_HEIGHT", 1080) == enc.TIER_SOURCE_HEIGHT


@pytest.mark.parametrize("quality,below,at,above", [
    ("good", "good", "good", "better"),
    ("better", "good", "better", "best"),
    ("best", "better", "best", "super"),
    ("super", "best", "super", "super"),
    (None, "good", "good", "better"),           # nothing typed is good
])
def test_T2_the_typed_tier_is_for_1080p_a_step_each_way_from_it(
        quality, below, at, above):
    assert enc.tier_for_source(quality, 720) == below
    assert enc.tier_for_source(quality, 1080) == at
    assert enc.tier_for_source(quality, 2160) == above
    # The step is on the threshold, not on some band around it.
    assert enc.tier_for_source(quality, 1079) == below
    assert enc.tier_for_source(quality, 1081) == above
    # The ladder stops at its ends rather than falling off them.
    assert enc.tier_for_source(quality, 268) == below
    assert enc.tier_for_source(quality, 4320) == above


def test_T2_an_unknown_source_takes_the_typed_tier():
    for quality in enc.QUALITY_ORDER:
        assert enc.tier_for_source(quality, None) == quality
    assert enc.tier_for_source(None, None) == enc.DEFAULT_QUALITY


def test_T3_cmd_encode_hands_the_pool_one_settings_per_song(db, tmp_path,
                                                            pool):
    three_tiers(db, tmp_path)
    cli.cmd_encode(encode_args(quality="better"), db)
    got = job_settings(pool)
    assert {k: (s.crf, s.bitrate_cap) for k, s in got.items()} == {
        "low": (31, "4M"),      # good
        "mid": (24, "6M"),      # better, as typed
        "high": (18, "8M"),     # best
    }
    # Only the tier's two numbers move; everything else is the run's.
    run = pool.calls[0]["settings"]
    for s in got.values():
        assert replace(s, crf=run.crf, bitrate_cap=run.bitrate_cap) == run


def test_T3_the_step_follows_the_codec_row(db, tmp_path, pool):
    three_tiers(db, tmp_path)
    cli.cmd_encode(encode_args(codec="h264", quality="best"), db)
    got = job_settings(pool)
    assert (got["low"].crf, got["low"].bitrate_cap) == (20, "6M")     # better
    assert (got["mid"].crf, got["mid"].bitrate_cap) == (17, "8M")     # best
    assert (got["high"].crf, got["high"].bitrate_cap) == (14, "12M")  # super
    assert {s.codec for s in got.values()} == {"h264"}


def test_T4_a_song_with_no_recorded_size_takes_the_typed_tier(db, tmp_path,
                                                              pool):
    song(db, tmp_path / "untagged")
    song(db, tmp_path / "zero", source_height=0)      # never written, too
    cli.cmd_encode(encode_args(quality="best"), db)
    got = job_settings(pool)
    assert (got["untagged"].crf, got["untagged"].bitrate_cap) == (18, "8M")
    assert (got["zero"].crf, got["zero"].bitrate_cap) == (18, "8M")


def test_T5_a_typed_crf_or_cap_holds_for_every_song(db, tmp_path, pool):
    three_tiers(db, tmp_path)
    # Kept sources, so the same three songs are still pending for run two.
    cli.cmd_encode(encode_args(quality="better", crf=20, keep_source=True),
                   db)
    got = job_settings(pool)
    assert {s.crf for s in got.values()} == {20}
    # ... and the other half still follows the source.
    assert {k: s.bitrate_cap for k, s in got.items()} == {
        "low": "4M", "mid": "6M", "high": "8M"}

    pool.calls.clear()
    cli.cmd_encode(encode_args(quality="better", bitrate_cap="5M",
                               keep_source=True), db)
    got = job_settings(pool)
    assert {s.bitrate_cap for s in got.values()} == {"5M"}
    assert {k: s.crf for k, s in got.items()} == {
        "low": 31, "mid": 24, "high": 18}


def test_T6_a_preview_reads_no_tier(db, tmp_path, pool):
    three_tiers(db, tmp_path)
    cli.cmd_encode(encode_args(preview=True, quality="best"), db)
    run = pool.calls[0]["settings"]
    assert run.height == 480
    assert all(s == run for s in job_settings(pool).values())


def test_T6_a_size_lock_reads_no_tier(db, tmp_path, pool):
    three_tiers(db, tmp_path)
    cli.cmd_encode(encode_args(size_lock="2500k"), db)
    run = pool.calls[0]["settings"]
    assert run.size_lock == "2500k"
    assert all(s == run for s in job_settings(pool).values())


def test_T7_encode_says_the_mix_before_it_starts(db, tmp_path, pool, capsys):
    three_tiers(db, tmp_path)
    song(db, tmp_path / "high2", source_height=1440)
    cli.cmd_encode(encode_args(quality="better"), db)
    out = capsys.readouterr().out
    assert cli.tier_mix_line(
        rows_of(db, tmp_path / "high", tmp_path / "high2", tmp_path / "low",
                tmp_path / "mid"), "better") in out
    assert "2 at best" in out and "1 at better" in out and "1 at good" in out


# ============================================ E  estimate sums per tier =====

def test_E1_one_group_prints_exactly_as_before(db, tmp_path, monkeypatch,
                                               capsys):
    for n in ("a", "b", "c"):
        song(db, tmp_path / n, source_height=1080)              # 600 s
    monkeypatch.setattr(enc, "TYPICAL_RATES", {("vp8", 1080, 31): 2_000_000.0})
    never_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert "~ 0.15 GB (max 0.30 GB) [typical" in out
    # No per-tier breakdown for a run that has only one tier in it.
    assert out.count(" GB (max ") == 1


def test_E2_several_groups_print_a_line_each_and_the_sum(db, tmp_path,
                                                         monkeypatch,
                                                         capsys):
    three_tiers(db, tmp_path)                                    # 200 s each
    monkeypatch.setattr(enc, "TYPICAL_RATES", {
        ("vp8", 1080, 31): 2_000_000.0,      # good   -> 0.05 GB of 200 s
        ("vp8", 1080, 24): 4_000_000.0,      # better -> 0.10 GB
        ("vp8", 1080, 18): 8_000_000.0,      # best   -> 0.20 GB
    })
    never_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(quality="better"), db)
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if " GB (max " in ln]
    assert len(lines) == 4
    # One line per tier, each with its songs, its estimate and its ceiling.
    by_tier = {ln.split(":")[0].strip(): ln for ln in lines[:3]}
    assert set(by_tier) == {"good", "better", "best"}
    assert "1 songs" in by_tier["good"] or "1 song" in by_tier["good"]
    assert "~ 0.05 GB (max 0.10 GB)" in by_tier["good"]
    assert "~ 0.10 GB (max 0.15 GB)" in by_tier["better"]
    assert "~ 0.20 GB (max 0.20 GB)" in by_tier["best"]
    assert all("typical" in ln for ln in lines[:3])
    # Then the sum, and the sum of the ceilings, as the last line.
    assert lines[3].startswith("~ 0.35 GB (max 0.45 GB)")


def test_E2_each_group_has_its_own_source_of_figure(db, tmp_path, monkeypatch,
                                                    capsys):
    three_tiers(db, tmp_path)
    # best is measured in this database, better is typical, good is neither.
    best = enc.EncodeSettings(crf=18, bitrate_cap="8M")
    db.set_rate(enc.rate_key(best), 8_000_000.0)
    monkeypatch.setattr(enc, "TYPICAL_RATES", {("vp8", 1080, 24): 4_000_000.0})
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: 2_000_000.0)
    cli.cmd_estimate(estimate_args(quality="better"), db)
    out = capsys.readouterr().out
    lines = {ln.split(":")[0].strip(): ln
             for ln in out.splitlines() if " GB (max " in ln}
    assert "[measured]" in lines["best"]
    assert "typical" in lines["better"]
    assert "[measured]" in lines["good"]        # sampled, then stored
    assert "no typical rate is published" in out.lower()
    assert db.get_rate(enc.rate_key(enc.EncodeSettings())) == 2_000_000.0
    assert "~ 0.35 GB (max 0.45 GB)" in out


def test_E3_measure_samples_every_group_on_its_own_songs(db, tmp_path,
                                                         monkeypatch, capsys):
    lib = tmp_path
    lows = [song(db, lib / f"low{i}", source_height=720) for i in range(3)]
    highs = [song(db, lib / f"high{i}", source_height=2160) for i in range(2)]
    sampled = []

    def fake_measure(rows, settings, workers=None):
        sampled.append((settings.crf, sorted(Path(r["song_dir"]).name
                                             for r in rows)))
        return 2_000_000.0

    monkeypatch.setattr(enc, "measure_rate", fake_measure)
    cli.cmd_estimate(estimate_args(quality="better", measure=True), db)
    assert sorted(sampled) == [
        (18, sorted(p.name for p in highs)),
        (31, sorted(p.name for p in lows)),
    ]
    for crf, cap in ((31, "4M"), (18, "8M")):
        key = enc.rate_key(enc.EncodeSettings(crf=crf, bitrate_cap=cap))
        assert db.get_rate(key) == 2_000_000.0
    out = capsys.readouterr().out
    assert out.count("Mbit/s") == 2
    assert "~ 0.25 GB (max 0.70 GB)" in out       # 1000 s at 2 Mbit/s


def test_E4_a_typed_crf_collapses_the_groups(db, tmp_path, monkeypatch,
                                             capsys):
    three_tiers(db, tmp_path)
    monkeypatch.setattr(enc, "TYPICAL_RATES", {})
    calls = []
    monkeypatch.setattr(enc, "measure_rate",
                        lambda rows, *a, **k: calls.append(len(list(rows)))
                        or 2_000_000.0)
    cli.cmd_estimate(estimate_args(crf=20, bitrate_cap="5M"), db)
    assert calls == [3]                           # one recipe, one sample
    out = capsys.readouterr().out
    assert out.count(" GB (max ") == 1


def test_E5_size_lock_and_preview_are_one_group(db, tmp_path, monkeypatch,
                                                capsys):
    three_tiers(db, tmp_path)
    never_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(size_lock="2M"), db)
    out = capsys.readouterr().out
    assert "~ 0.15 GB (max 0.15 GB)" in out
    assert out.count(" GB (max ") == 1

    monkeypatch.setattr(enc, "TYPICAL_RATES", {})
    db.set_rate(enc.rate_key(cli.encode_settings(
        estimate_args(preview=True, quality="best"))), 500_000.0)
    cli.cmd_estimate(estimate_args(preview=True, quality="best"), db)
    out = capsys.readouterr().out
    assert out.count(" GB (max ") == 1 and "[measured]" in out


def test_E6_estimate_still_writes_no_song_row(db, tmp_path, monkeypatch):
    paths = three_tiers(db, tmp_path)
    before = [tuple(r) for r in rows_of(db, *paths)]
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: 2_000_000.0)
    cli.cmd_estimate(estimate_args(quality="better", measure=True), db)
    assert [tuple(r) for r in rows_of(db, *paths)] == before


# ============================================ L  the lines ==================

def test_L1_the_mix_line_counts_songs_per_tier(db, tmp_path):
    a, b, c = three_tiers(db, tmp_path)
    d = song(db, tmp_path / "d", source_height=2160)
    line = cli.tier_mix_line(rows_of(db, a, b, c, d), "better")
    assert "2 at best" in line and "1 at better" in line and "1 at good" in line
    assert "tag-sources" not in line
    assert "more" not in line
    # Tiers with nobody in them are not named.
    line = cli.tier_mix_line(rows_of(db, a), "better")
    assert "1 at good" in line and "best" not in line and "better" in line


def test_L1_the_mix_line_names_the_gap_and_the_default(db, tmp_path):
    a = song(db, tmp_path / "a", source_height=2160)
    b = song(db, tmp_path / "b")
    line = cli.tier_mix_line(rows_of(db, a, b), None)
    assert "1 at better" in line and "1 at good" in line
    assert "tag-sources" in line and "1 with no recorded size" in line
    assert cli.tier_mix_line([], "best") is None


def test_L2_estimate_and_videos_print_the_mix(db, tmp_path, capsys,
                                              monkeypatch):
    a, b, c = three_tiers(db, tmp_path)
    song(db, tmp_path / "unapproved", source_height=2160, review=None)
    never_measure(monkeypatch)
    cli.cmd_estimate(estimate_args(quality="best", reviewed=True), db)
    assert cli.tier_mix_line(rows_of(db, a, b, c), "best") in \
        capsys.readouterr().out

    monkeypatch.setattr(enc, "find_output", lambda d: None)
    cli.cmd_videos(SimpleNamespace(lengths=False, mark=False, out=None,
                                   quiet=True), db)
    assert cli.tier_mix_line(rows_of(db, a, b, c), None) in \
        capsys.readouterr().out


def test_L3_the_size_line_credits_the_ceiling_not_the_tier(db, tmp_path):
    a, b, c = three_tiers(db, tmp_path)
    line = cli.source_size_line(rows_of(db, a, b, c))
    assert "1 of 3" in line and "1080" in line
    assert "tier" not in line.lower() and "good" not in line
    assert "ceiling" in line


# ============================================ F  the floor ==================

def vf_of(settings, source_fps=30.0):
    cmd = enc.build_command(Path("in.mp4"), Path("out"), settings, source_fps)
    return cmd[cmd.index("-vf") + 1]


def test_F1_the_floor_is_an_expression_in_the_scale_box():
    assert enc.MIN_HEIGHT == 720
    assert enc.box_width(enc.MIN_HEIGHT) == 1280
    vf = vf_of(enc.EncodeSettings())
    assert r"scale=w=min(1920\,max(iw\,1280)):h=min(1080\,max(ih\,720))" in vf
    assert "force_original_aspect_ratio=decrease" in vf
    # The pad is untouched: 16:9 at the output height, each side max'd.
    assert r"pad=w=max(iw\,round(ih*16/9/2)*2):h=max(ih\,round(iw*9/16/2)*2)" \
        in vf
    # Still no number taken off the source anywhere in the chain.
    assert vf == vf_of(enc.EncodeSettings())
    assert (enc.rate_key(enc.EncodeSettings())
            == enc.rate_key(enc.EncodeSettings()))


def test_F1_the_ceiling_wins_where_the_two_meet():
    vf = vf_of(enc.EncodeSettings(height=480))
    assert r"scale=w=min(854\,max(iw\,1280)):h=min(480\,max(ih\,720))" in vf
    vf = vf_of(enc.EncodeSettings(height=720))
    assert r"scale=w=min(1280\,max(iw\,1280)):h=min(720\,max(ih\,720))" in vf


# --------------------------------------------------------- real ffmpeg -------

@pytest.fixture(scope="session")
def sources(tmp_path_factory):
    root = tmp_path_factory.mktemp("b13src")
    made: dict[str, Path] = {}

    def make(size: str) -> Path:
        dst = made.get(size)
        if dst is None:
            dst = root / f"{size}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-nostdin",
                 "-f", "lavfi", "-i", f"testsrc2=s={size}:r=10:d=1",
                 "-pix_fmt", "yuv420p", "-c:v", "libx264",
                 "-preset", "ultrafast", str(dst)],
                check=True, capture_output=True,
            )
            made[size] = dst
        return dst

    return make


def dims_of(path: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])


def run_encode(src: Path, out_dir: Path, **over) -> tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    settings = enc.EncodeSettings(codec="h264", preset="ultrafast",
                                  threads_per_job=2, **over)
    ok, err = enc.encode_one(src, out_dir, settings, keep_source=True)
    assert ok, err
    return dims_of(enc.output_path(out_dir, settings))


@needs_ffmpeg
@pytest.mark.parametrize("size,expected", [
    ("640x268", (1280, 720)),         # the Alexisonfire shape, letterboxed
    ("640x360", (1280, 720)),         # 16:9 below the floor: brought up
    ("640x480", (1280, 720)),         # 4:3 below the floor: pillarboxed
    ("1280x720", (1280, 720)),        # at the floor: left alone
    ("1024x576", (1280, 720)),        # 576 is below the floor: brought up
    ("1920x804", (1920, 1080)),       # a widescreen crop of 1080 footage
    ("1920x1080", (1920, 1080)),      # at the ceiling
    ("3840x2160", (1920, 1080)),      # above it: reduced
    ("3000x500", (1920, 1080)),       # wider than the box, shorter than it
])
def test_F2_below_the_floor_is_brought_up_to_it_with_bars(sources, tmp_path,
                                                          size, expected):
    assert run_encode(sources(size), tmp_path / size, height=1080) == expected


@needs_ffmpeg
def test_F2_the_picture_keeps_its_shape_inside_the_bars(sources, tmp_path):
    out = tmp_path / "a"
    assert run_encode(sources("640x268"), out, height=1080) == (1280, 720)
    written = enc.output_path(out, enc.EncodeSettings(codec="h264"))

    def brightest(crop):
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-nostdin", "-i", str(written),
             "-vf", f"crop={crop},format=gray", "-frames:v", "1",
             "-f", "rawvideo", "-"],
            capture_output=True, check=True,
        ).stdout
        assert raw
        return max(raw)

    # 640x268 scaled by 2 is 1280x536, so the bars are 92 high each.
    assert brightest("1280:80:0:0") == 0          # top bar
    assert brightest("1280:80:0:640") == 0        # bottom bar
    assert brightest("1280:100:0:310") > 0        # picture


@needs_ffmpeg
def test_F3_a_preview_stops_at_its_own_ceiling(sources, tmp_path):
    assert run_encode(sources("320x240"), tmp_path / "p",
                      height=480) == (854, 480)
    assert run_encode(sources("640x268"), tmp_path / "q",
                      height=480) == (854, 480)


# ============================================ S  set clears what match does =

@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setattr(mt, "fetch_metadata",
                        lambda vid, cookies=None: {"title": "T", "uploader": "U"})


def test_S1_set_clears_the_source_sizes_and_the_length(db, tmp_path,
                                                       no_network):
    s = song(db, tmp_path / "a", source_height=1080, source_max_height=2160,
             video_seconds=200.0, fp_score=80.0, dominance=3.0, windows=7)
    out = cli.cmd_set(SimpleNamespace(pattern=str(s), url="bbbbbbbbbbb",
                                      cookies=None), db)
    assert out == "set"
    r = row(db, s)
    assert r["video_id"] == "bbbbbbbbbbb" and r["match_status"] == "ok"
    assert r["match_note"].startswith("MANUAL:")
    for col in ("source_height", "source_max_height", "video_seconds",
                "source_path", "offset_ms", "review", "fp_score",
                "dominance", "windows", "motion"):
        assert r[col] is None, col
    for stage in ("download", "sync", "encode", "ini"):
        assert r[f"{stage}_status"] == "pending", stage
    assert not (tmp_path / "a" / "video.src.mkv").exists()


def test_S1_set_and_match_clear_the_same_columns(db, tmp_path, no_network):
    """Whatever `_requeue_after_match` blanks, `set` blanks too."""
    cols = dict(source_height=1080, source_max_height=2160,
                video_seconds=200.0, fp_score=80.0, dominance=3.0, windows=7,
                spread_ms=5.0, drift_ppm=1.0, sync_note="n", encode_note="e")
    a = song(db, tmp_path / "a", **cols)
    b = song(db, tmp_path / "b", **cols)
    cli._requeue_after_match(db, a, row(db, a))
    cli.cmd_set(SimpleNamespace(pattern=str(b), url="bbbbbbbbbbb",
                                cookies=None), db)
    ra, rb = row(db, a), row(db, b)
    # The match columns are `set`'s own to fill; everything else it blanks.
    blanked = {k for k in ra.keys()
               if ra[k] is None and not k.startswith("match_")}
    assert {k for k in rb.keys() if rb[k] is None} >= blanked
    assert "source_height" in blanked and "video_seconds" in blanked
