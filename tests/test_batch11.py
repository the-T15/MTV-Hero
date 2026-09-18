"""
Batch 11 - quality tiers.

Two things the encode step gets wrong today and one it is missing.

    S   IT ENLARGES SMALL VIDEOS. `build_command` scales to a fixed WxH box
        and then pads to the same fixed box, so a 720p source is blown up to
        1080p: more bits and roughly double the encode time for detail that
        is not in the file, when the game scales whatever it is given to the
        screen anyway. (The comment above the filter says "pad if smaller",
        which is not what that chain does.) `--height` becomes a CEILING.
        The scale box is min(W, iw) x min(H, ih) with
        force_original_aspect_ratio=decrease, and the pad box is the 16:9
        box at the OUTPUT height rather than a fixed 1920x1080. Even
        dimensions throughout, SAR stays 1.

            at --height 1080   1280x720  -> 1280x720
                               1920x1080 -> 1920x1080
                               3840x2160 -> 1920x1080
                               640x480   -> 854x480    pillarboxed
                               2560x1080 -> 1920x1080  letterboxed
            at --height 720    1920x1080 -> 1280x720

        The 480p preview path follows the same rule.

    F   IT CAPS THE FRAME RATE AT 30. Both the YARG and the Clone Hero wikis
        say to keep the source rate; the cap was a cost default, not a
        compatibility rule. `--max-fps` becomes opt-in - `EncodeSettings.
        max_fps` and the flag both default to None, meaning no cap. A 25 fps
        source encodes at 25, a 60 fps source at 60, and at 30 only when
        `--max-fps 30` is asked for. Constant frame rate and the `fps=` in
        the filter chain are unchanged.

    K   so `rate_key` needs a new stand-in source rate. It used to build its
        probe command with `max_fps`, which worked only while `max_fps` was
        always a number. With no cap as the default, "--max-fps 30" and "no
        cap at all" would both resolve to 30.0 and share one row of the
        rates table. `KEY_SOURCE_FPS` is above any real source rate, so "no
        cap" keys as itself and every explicit cap keys as the cap.

    Q   THERE IS ONE QUALITY SETTING FOR EVERYONE. `--quality
        {good,better,best,super}` on `encode` and `estimate`, default good.
        The numbers live in ONE table, `encode.QUALITY_TIERS`, keyed by
        (tier, codec) and holding (crf, bitrate_cap):

            vp8   good 31/4M  better 24/6M  best 18/8M  super 12/12M
            h264  good 23/4M  better 20/6M  best 17/8M  super 14/12M

        `good` is today's default unchanged, so `QUALITY_TIERS[("good", n)]`
        is `(CODECS[n].crf, "4M")` for every codec row. The three hardware
        rows take H.264's numbers - their -cq / -qp_* / -global_quality are
        the same scale. An explicit `--crf` or `--bitrate-cap` overrides the
        tier's corresponding value and leaves the other one alone, which is
        why both flags now default to None: a default of "4M" cannot be told
        apart from someone typing 4M.

    T   `TYPICAL_RATES` gains the vp8 tiers, the measured `good` figure
        scaled by the ratio of the ceilings - 4.4, 5.8 and 8.7 Mbit/s
        against good's 2.9 - labelled `[typical]` exactly as the existing
        seeds are. The H.264 tiers above good get no seed, and `estimate`
        says there is nothing published for those settings and points at
        `--measure` rather than going quiet.

Tiers change bits, not resolution: every tier is still capped by `--height`,
and an upper tier only adds visible quality when the SOURCE is above 1080p,
which is Batch 12's job.

The scaling and frame-rate tests encode two-second synthetic sources with
real ffmpeg and ffprobe the result, because the whole question is what comes
out the far end of a filter chain. They skip when ffmpeg is absent; every
other test here is a pure function or a parsed namespace and shells out to
nothing.

Run from the repository root:  pytest -q
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from yargvid import cli
from yargvid import encode as enc
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


def encode_args(**over):
    """The namespace `encode` produces with no flags, Batch 11 defaults."""
    a = dict(limit=None, sample=False, skip_existing=False, reviewed=False,
             skip_static=True, height=1080, crf=None, cpu_used=3, threads=2,
             workers=None, preview=False, preview_height=480,
             bitrate_cap=None, max_fps=None, codec="vp8", fps=None,
             size_lock=None, quality=None)
    a.update(over)
    return SimpleNamespace(**a)


def estimate_args(**over):
    return encode_args(measure=False, **over)


class FakePool:
    def __init__(self):
        self.calls = []

    def __call__(self, jobs, settings, workers=None, on_done=None, **kw):
        self.calls.append(dict(jobs=list(jobs), settings=settings, kw=kw))
        results = {}
        for _src, d in jobs:
            results[d] = (True, "")
            if on_done:
                on_done(d, results[d])
        return results


@pytest.fixture
def pool(monkeypatch):
    fake = FakePool()
    monkeypatch.setattr(enc, "encode_many", fake)
    return fake


def parsed(monkeypatch, db, argv, cmd="cmd_encode"):
    seen = {}
    monkeypatch.setattr(cli, cmd, lambda args, db_: seen.update(vars(args)))
    assert cli.main(["--db", str(db.path), *argv]) == 0
    return SimpleNamespace(**seen)


def vf_of(settings, source_fps=30.0):
    cmd = enc.build_command(Path("in.mp4"), Path("out"), settings, source_fps)
    return cmd[cmd.index("-vf") + 1]


# --------------------------------------------------------- real ffmpeg -------

@pytest.fixture(scope="session")
def sources(tmp_path_factory):
    """
    Two-second synthetic sources, built once and shared.

    `make("3840x2160")` returns the path, generating it on first use so that
    a test that never asks for 4K never pays for it.
    """
    root = tmp_path_factory.mktemp("b11src")
    made: dict[tuple[str, int], Path] = {}

    def make(size: str, rate: int = 10) -> Path:
        dst = made.get((size, rate))
        if dst is None:
            dst = root / f"{size}@{rate}.mp4"
            subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-nostdin",
                 "-f", "lavfi", "-i", f"testsrc2=s={size}:r={rate}:d=2",
                 "-pix_fmt", "yuv420p", "-c:v", "libx264",
                 "-preset", "ultrafast", str(dst)],
                check=True, capture_output=True,
            )
            made[(size, rate)] = dst
        return dst

    return make


def probe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries",
         "stream=width,height,sample_aspect_ratio,avg_frame_rate",
         "-of", "json", str(path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    ).stdout
    return json.loads(out)["streams"][0]


def run_encode(src: Path, out_dir: Path, **over) -> dict:
    """Encode `src` into a folder of its own and probe what came out."""
    out_dir.mkdir(parents=True, exist_ok=True)
    # libx264 at ultrafast: the filter chain is what is under test and it is
    # the same string whatever encoder consumes it, so the test pays for the
    # cheapest one there is.
    settings = enc.EncodeSettings(
        codec="h264", preset="ultrafast", threads_per_job=2, **over
    )
    ok, err = enc.encode_one(src, out_dir, settings, keep_source=True)
    assert ok, err
    return probe(enc.output_path(out_dir, settings))


def dims(info: dict) -> tuple[int, int]:
    return int(info["width"]), int(info["height"])


def frame_rate(info: dict) -> float:
    num, den = info["avg_frame_rate"].split("/")
    return float(num) / float(den)


# ==================================================== S  never enlarge =======

def test_S1_height_is_a_ceiling_and_the_box_is_a_min_against_the_source():
    # 1920 for 1080, 1280 for 720, 854 for 480 - even, and the width the
    # pad expression rounds to, so the two cannot disagree.
    assert (enc.box_width(1080), enc.box_width(720), enc.box_width(480),
            enc.box_width(360)) == (1920, 1280, 854, 640)
    assert all(enc.box_width(h) % 2 == 0 for h in range(120, 2161, 2))

    vf = vf_of(enc.EncodeSettings())
    # The box is the ceiling met against the source, not a target to reach.
    assert r"min(1920\,iw)" in vf and r"min(1080\,ih)" in vf
    assert "force_original_aspect_ratio=decrease" in vf
    # ... and the pad is no longer the same fixed box.
    assert "pad=1920:1080" not in vf and "scale=1920:1080" not in vf
    assert "setsar=1" in vf

    # The chain stays resolution-independent: it may not carry numbers taken
    # off the source, or every source size becomes its own rates row.
    assert vf == vf_of(enc.EncodeSettings())


def test_S1_the_comment_that_described_the_old_box_is_gone():
    # It said "Downscale if larger, pad if smaller", which is not what a
    # fixed-box scale-then-pad does to a 720p source.
    text = Path(enc.__file__).read_text(encoding="utf-8")
    assert "Downscale if larger, pad if smaller." not in text


@needs_ffmpeg
@pytest.mark.parametrize("size,expected", [
    ("1280x720", (1280, 720)),        # smaller than the ceiling: left alone
    ("1920x1080", (1920, 1080)),      # exactly the ceiling
    ("3840x2160", (1920, 1080)),      # larger: reduced to it
    ("640x480", (854, 480)),          # 4:3, pillarboxed at ITS OWN height
    ("2560x1080", (1920, 1080)),      # ultrawide, letterboxed
])
def test_S2_nothing_is_enlarged_at_height_1080(sources, tmp_path, size,
                                               expected):
    info = run_encode(sources(size), tmp_path / size, height=1080)
    assert dims(info) == expected
    assert info.get("sample_aspect_ratio") in (None, "1:1")
    src_w, src_h = (int(x) for x in size.split("x"))
    # "Never enlarge" is about the picture, not the frame: padding may make
    # the frame wider than the source, but never both wider and taller.
    w, h = dims(info)
    assert not (w > src_w and h > src_h)


@needs_ffmpeg
def test_S3_the_pad_box_is_at_the_output_height_not_1920x1080(sources,
                                                              tmp_path):
    # A 1080p source under a lower ceiling comes down to it.
    assert dims(run_encode(sources("1920x1080"), tmp_path / "a",
                           height=720)) == (1280, 720)
    # ... and a 4:3 source is padded to 16:9 at 480, not blown up to 1080.
    out = tmp_path / "b"
    assert dims(run_encode(sources("640x480"), out, height=1080)) == (854, 480)

    # The bars are real black and the picture is still in the middle of them.
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
    assert brightest("100:480:0:0") == 0        # left bar
    assert brightest("100:480:754:0") == 0      # right bar
    assert brightest("100:480:400:0") > 0       # picture


@needs_ffmpeg
def test_S4_the_preview_path_follows_the_same_rule(sources, tmp_path):
    # A preview is the real encode at a low ceiling, so it must not enlarge
    # either - and it keeps the source.
    src = sources("320x240")
    out = tmp_path / "p"
    info = run_encode(src, out, height=480)
    assert dims(info) == (426, 240)             # 4:3 padded to 16:9 at 240
    assert src.exists()


@needs_ffmpeg
def test_S4_the_rule_is_the_filter_not_the_codec(sources, tmp_path):
    # Same chain, VP8: the box belongs to -vf, not to any encoder row.
    out = tmp_path / "v"
    out.mkdir()
    s = enc.EncodeSettings(codec="vp8", height=480, cpu_used=5)
    ok, err = enc.encode_one(sources("320x240"), out, s, keep_source=True)
    assert ok, err
    assert dims(probe(enc.output_path(out, s))) == (426, 240)


# ================================================ F  keep the source rate ====

def test_F1_max_fps_is_off_by_default():
    assert enc.EncodeSettings().max_fps is None


@pytest.mark.parametrize("fps,cap,source,expected", [
    (None, None, 25.0, 25.0),      # a 25 fps source encodes at 25
    (None, None, 60.0, 60.0),      # ... and a 60 fps source at 60
    (None, None, 23.976, 23.976),  # ... and film rate is not rounded to 30
    (None, 30.0, 60.0, 30.0),      # --max-fps 30 still caps
    (None, 30.0, 25.0, 25.0),      # ... and never raises a slower source
    (60.0, 24.0, 25.0, 60.0),      # --fps beats --max-fps, as it did
    (None, None, None, 30.0),      # an unreadable source falls back to 30
    (None, 24.0, None, 24.0),      # ... or to the cap when there is one
])
def test_F1_output_rate_keeps_the_source_rate_unless_capped(fps, cap, source,
                                                            expected):
    s = enc.EncodeSettings(fps=fps, max_fps=cap)
    assert enc.output_rate(s, source) == pytest.approx(expected)


def test_F1_the_flag_defaults_to_no_cap_on_both_subcommands(db, monkeypatch):
    assert parsed(monkeypatch, db, ["encode"]).max_fps is None
    assert parsed(monkeypatch, db, ["estimate"],
                  cmd="cmd_estimate").max_fps is None
    # Naming it still caps.
    assert parsed(monkeypatch, db, ["encode", "--max-fps", "30"]).max_fps \
        == 30.0


def test_F1_the_absent_cap_reaches_the_encode_pool(db, pool):
    # Through the real parser, not a namespace written out by hand: the
    # defaults ARE the thing under test, so a helper that supplies them
    # would be testing itself.
    song(db, db.path.parent / "s")
    assert cli.main(["--db", str(db.path), "encode"]) == 0
    s = pool.calls[0]["settings"]
    assert s.max_fps is None                    # no cap
    assert s.bitrate_cap == "4M"                # ... and good's ceiling


@needs_ffmpeg
@pytest.mark.parametrize("rate,cap,expected", [
    (25, None, 25.0),
    (60, None, 60.0),
    (60, 30.0, 30.0),
])
def test_F2_the_source_rate_survives_the_encode(sources, tmp_path, rate, cap,
                                                expected):
    src = sources("320x240", rate)
    info = run_encode(src, tmp_path / f"{rate}-{cap}", height=480,
                      max_fps=cap)
    assert frame_rate(info) == pytest.approx(expected, rel=0.02)


# ================================================== K  the key still parts ===

def test_K1_the_key_tells_a_cap_from_no_cap():
    # The stand-in has to be above any real source rate, or a fast source
    # would be capped by the probe itself.
    assert enc.KEY_SOURCE_FPS > 240

    no_cap = enc.rate_key(enc.EncodeSettings())
    at_30 = enc.rate_key(enc.EncodeSettings(max_fps=30.0))
    at_24 = enc.rate_key(enc.EncodeSettings(max_fps=24.0))
    forced = enc.rate_key(enc.EncodeSettings(fps=60.0))
    assert len({no_cap, at_30, at_24, forced}) == 4

    # Still Batch 10c's rule: the key IS the command, nothing invented.
    s = enc.EncodeSettings()
    cmd = enc.build_command(Path("in"), Path("out"), s, enc.KEY_SOURCE_FPS)
    assert set(no_cap) <= set(cmd)


def test_K2_every_tier_is_its_own_row(db):
    keys = {}
    for tier in enc.QUALITY_ORDER:
        crf, cap = enc.tier_of(tier, "vp8")
        keys[tier] = enc.rate_key(
            enc.EncodeSettings(crf=crf, bitrate_cap=cap))
    assert len(set(keys.values())) == len(enc.QUALITY_ORDER)

    db.set_rate(keys["best"], 5_800_000.0)
    assert db.get_rate(keys["best"]) == 5_800_000.0
    assert db.get_rate(keys["good"]) is None


# ======================================================= Q  the tiers ========

def test_Q1_the_tier_table_holds_the_numbers_once():
    assert enc.QUALITY_ORDER == ("good", "better", "best", "super")
    assert enc.DEFAULT_QUALITY == "good"

    for tier, expected in (("good", (31, "4M")), ("better", (24, "6M")),
                           ("best", (18, "8M")), ("super", (12, "12M"))):
        assert enc.QUALITY_TIERS[(tier, "vp8")] == expected

    for tier, expected in (("good", (23, "4M")), ("better", (20, "6M")),
                           ("best", (17, "8M")), ("super", (14, "12M"))):
        for codec in ("h264", "h264_nvenc", "h264_amf", "h264_qsv"):
            assert enc.QUALITY_TIERS[(tier, codec)] == expected

    # Every codec row, every tier - no key can be missing.
    assert set(enc.QUALITY_TIERS) == {
        (t, c) for t in enc.QUALITY_ORDER for c in enc.CODECS
    }
    # `good` is today's default, unchanged, so the row's own number and the
    # table cannot drift apart without this failing.
    for name, codec in enc.CODECS.items():
        assert enc.QUALITY_TIERS[("good", name)] == (codec.crf, "4M")


def test_Q2_quality_is_a_flag_on_encode_and_estimate(db, monkeypatch):
    # Unset at the flag so that --size-lock can refuse an explicit tier,
    # and `good` once resolved - the same shape --bitrate-cap takes.
    for sub_, cmd in (("encode", "cmd_encode"), ("estimate", "cmd_estimate")):
        a = parsed(monkeypatch, db, [sub_], cmd=cmd)
        assert a.quality is None
        s = cli.encode_settings(a)
        assert (enc.effective_crf(s), s.bitrate_cap) == (31, "4M")
    assert parsed(monkeypatch, db,
                  ["encode", "--quality", "best"]).quality == "best"
    assert parsed(monkeypatch, db, ["estimate", "--quality", "super"],
                  cmd="cmd_estimate").quality == "super"

    with pytest.raises(SystemExit) as e:
        cli.main(["--db", str(db.path), "encode", "--quality", "ultra"])
    assert e.value.code == 2


@pytest.mark.parametrize("codec,tier,crf,cap", [
    ("vp8", "good", 31, "4M"),
    ("vp8", "better", 24, "6M"),
    ("vp8", "best", 18, "8M"),
    ("vp8", "super", 12, "12M"),
    ("h264", "best", 17, "8M"),
    ("h264_nvenc", "super", 14, "12M"),
])
def test_Q3_a_tier_sets_both_numbers(codec, tier, crf, cap):
    s = cli.encode_settings(encode_args(codec=codec, quality=tier))
    assert (enc.effective_crf(s), s.bitrate_cap) == (crf, cap)


def test_Q4_an_explicit_flag_overrides_only_its_own_half():
    # --crf keeps the tier's ceiling.
    s = cli.encode_settings(encode_args(quality="super", crf=40))
    assert (enc.effective_crf(s), s.bitrate_cap) == (40, "12M")
    # --bitrate-cap keeps the tier's quality number.
    s = cli.encode_settings(encode_args(quality="super", bitrate_cap="2M"))
    assert (enc.effective_crf(s), s.bitrate_cap) == (12, "2M")
    # Both named, both honoured.
    s = cli.encode_settings(encode_args(quality="best", crf=9,
                                        bitrate_cap="3M"))
    assert (enc.effective_crf(s), s.bitrate_cap) == (9, "3M")
    # An args namespace from before the flag existed still works.
    old = encode_args()
    del old.quality
    assert cli.encode_settings(old).bitrate_cap == "4M"


def test_Q5_the_tier_reaches_the_ffmpeg_command(db, pool):
    song(db, db.path.parent / "s")
    cli.cmd_encode(encode_args(quality="best"), db)
    s = pool.calls[0]["settings"]
    cmd = enc.build_command(Path("in.mp4"), Path("out.webm"), s, 30.0)
    assert cmd[cmd.index("-crf") + 1] == "18"
    assert cmd[cmd.index("-b:v") + 1] == "8M"
    # NOTES: `-b:v 0` is a VP9 idiom; on VP8 it silently means 256 kbit/s.
    assert cmd[cmd.index("-b:v") + 1] != "0"


def test_Q6_size_lock_refuses_a_typed_tier_but_not_the_default_one(db,
                                                                   monkeypatch):
    # A lock sets the bitrate and lets the quality fall where it may. A
    # tier sets a quality number AND a ceiling, and build_command under a
    # lock reads neither - so `--quality super --size-lock 2500k` asks for
    # two different things and quietly gets the lock, which is the exact
    # case the existing --crf check was written for.
    for extra in (["--crf", "20"], ["--quality", "super"]):
        with pytest.raises(SystemExit) as e:
            cli.main(["--db", str(db.path), "encode",
                      "--size-lock", "2500k", *extra])
        assert e.value.code == 2

    # The default tier is not a request, so a lock on its own still runs.
    a = parsed(monkeypatch, db, ["encode", "--size-lock", "2500k"])
    assert a.size_lock == "2500k" and a.crf is None and a.quality is None


def test_Q7_height_is_documented_as_a_ceiling(db, monkeypatch, capsys):
    for sub in ("encode", "estimate"):
        with pytest.raises(SystemExit):
            cli.main(["--db", str(db.path), sub, "--help"])
        text = capsys.readouterr().out.lower()
        assert "--quality" in text
        # The last mention is the one in the option list; the first is the
        # usage line, which argparse wraps without any help text at all.
        assert "ceiling" in text.rsplit("--height", 1)[1][:400]


# ================================================ T  what it is going to cost =

def test_T1_typical_rates_carry_the_vp8_tiers():
    for crf, bps in ((31, 2.4e6), (24, 3.22e6), (18, 3.84e6), (12, 4.81e6)):
        assert enc.TYPICAL_RATES[("vp8", 1080, crf)] == pytest.approx(bps)
    assert enc.TYPICAL_RATES[("h264_nvenc", 1080, 23)] == pytest.approx(3.4e6)
    # Nothing is published for the H.264 tiers above `good`.
    for crf in (20, 17, 14):
        for codec in ("h264", "h264_nvenc", "h264_amf", "h264_qsv"):
            assert ("h264" if codec == "h264" else codec, 1080,
                    crf) not in enc.TYPICAL_RATES

    # A tier looked up through settings, not by hand.
    s = cli.encode_settings(encode_args(quality="best"))
    assert enc.typical_rate(s) == pytest.approx(3.84e6)


def test_T2_a_tier_changes_the_estimate(db, capsys):
    for n in ("a", "b", "c"):
        song(db, db.path.parent / n)            # 3 x 200 s = 600 s

    cli.cmd_estimate(estimate_args(), db)
    good = capsys.readouterr().out
    cli.cmd_estimate(estimate_args(quality="better"), db)
    better = capsys.readouterr().out

    # 2.4 Mbit/s x 600 s = 0.18 GB under a 4M ceiling; 3.22 under 6M = 0.24.
    assert "~ 0.18 GB (max 0.30 GB)" in good
    assert "~ 0.24 GB (max 0.45 GB)" in better
    assert "typical" in good and "typical" in better


def test_T3_estimate_says_when_nothing_is_published(db, capsys, monkeypatch):
    song(db, db.path.parent / "a")
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: 3_000_000.0)
    cli.cmd_estimate(estimate_args(codec="h264", quality="best"), db)
    out = capsys.readouterr().out.lower()
    # Batch 10b's rule stands - with nothing typical and nothing stored it
    # measures. What it must not do is go quiet about which of the four
    # tiers has a published figure behind it and which does not.
    assert "no typical rate is published" in out
    assert "--measure" in out
    assert "gb (max" in out and "[measured]" in out

    # `good` on the same codec has no such line: it is the seeded one.
    enc.RATE_TABLE.clear()
    cli.cmd_estimate(estimate_args(codec="h264_nvenc"), db)
    assert "no typical rate is published" not in capsys.readouterr().out
