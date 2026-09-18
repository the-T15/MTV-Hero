"""
Batch 11b - a sane source frame rate.

Batch 11 made "keep the source's own frame rate" the default. That put the
probe on the critical path: whatever `source_frame_rate` reads is what the
encode runs at, and nothing above it checks the number. Two things are
wrong with what it reads.

    P   IT READS THE WRONG FIELD. `r_frame_rate` is ffprobe's guess at the
        stream's timebase-derived rate, and on mkv - which is what yt-dlp
        writes - it can be `1000/1` when the container gives no better
        answer. `avg_frame_rate` is the rate the frames actually arrive at.
        `source_frame_rate` reads `avg_frame_rate` first and falls back to
        `r_frame_rate` only when the average is absent or unusable (`0/0`,
        a zero denominator, not a fraction).

    B   IT BELIEVES ANY NUMBER. A 1000/1 rate read from a real file would
        have run the encode at 1000 fps with no cap set - a file forty
        times the size, for nothing. `MAX_SOURCE_FPS = 120.0` is the
        highest rate the probe will pass on as a source rate; anything
        above it, zero, or negative is treated as unreadable. An
        unreadable rate is `None`, and `output_rate` already turns `None`
        into 30 (or the cap, when there is one) - that fallback is Batch
        11's and is not touched here.

        The bound is a bound on what a FILE claims, not on what a person
        asks for: `--fps` and `--max-fps` are not checked against it.

The 190 approved sources were all probed on 2026-09-18 and none of them
trips either rule, so this is a guard, not a repair. It has to land before
the first `encode --reviewed` all the same, because that run deletes the
sources and there is no second try.

Every test here is a pure function with `au.probe` replaced by a dict, or
`encode_one` driven with a fake `subprocess.run` that records the command
instead of running it. Nothing shells out to ffmpeg or ffprobe.

Run from the repository root:  pytest -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yargvid import encode as enc


# ----------------------------------------------------------------- helpers ---

@pytest.fixture(autouse=True)
def fresh_tables(monkeypatch):
    monkeypatch.setattr(enc, "RATE_TABLE", {})
    monkeypatch.setattr(enc, "_supports_fps_mode", lambda: True)


def probed(monkeypatch, *streams):
    """Make `au.probe` answer with these streams for any path."""
    info = {"streams": list(streams), "format": {"duration": "200.0"}}
    monkeypatch.setattr(enc.au, "probe", lambda path: info)


def video(**fields):
    return {"codec_type": "video", **fields}


def audio(**fields):
    return {"codec_type": "audio", **fields}


def fps_in(cmd: list[str]) -> str:
    """The `fps=` value inside the -vf chain of an ffmpeg command."""
    vf = cmd[cmd.index("-vf") + 1]
    for part in vf.split(","):
        if part.startswith("fps="):
            return part[len("fps="):]
    raise AssertionError(f"no fps= in {vf!r}")


# ============================================ P  the right field, in order ===

def test_P1_avg_frame_rate_wins_over_r_frame_rate(monkeypatch):
    # The mkv case that started this: a nominal 1000/1 beside a real 29.97.
    probed(monkeypatch, video(r_frame_rate="1000/1",
                              avg_frame_rate="30000/1001"))
    assert enc.source_frame_rate(Path("x.mkv")) == pytest.approx(29.97, rel=1e-3)


@pytest.mark.parametrize("avg", [None, "", "0/0", "30/0", "x/y", "30"])
def test_P2_an_unusable_average_falls_back_to_r_frame_rate(monkeypatch, avg):
    fields = {"r_frame_rate": "25/1"}
    if avg is not None:
        fields["avg_frame_rate"] = avg
    probed(monkeypatch, video(**fields))
    assert enc.source_frame_rate(Path("x.mkv")) == pytest.approx(25.0)


def test_P3_only_the_first_video_stream_counts(monkeypatch):
    # Audio streams carry the same fields and must not be read; a second
    # video stream (a cover-art picture) must not override the first.
    probed(monkeypatch,
           audio(r_frame_rate="0/0", avg_frame_rate="0/0"),
           video(r_frame_rate="24/1", avg_frame_rate="24/1"),
           video(r_frame_rate="90000/1", avg_frame_rate="0/0"))
    assert enc.source_frame_rate(Path("x.mkv")) == pytest.approx(24.0)


@pytest.mark.parametrize("text,expected", [
    ("26979/1124", 24.002),     # Jefferson Airplane - White Rabbit, mkv
    ("29953/1000", 29.953),     # Alexisonfire, mkv
    ("30000/1001", 29.970),     # NTSC
    ("24000/1001", 23.976),     # film
    ("25/1", 25.0),
    ("50/1", 50.0),
    ("60/1", 60.0),
])
def test_P4_real_library_shapes_come_through(monkeypatch, text, expected):
    probed(monkeypatch, video(r_frame_rate=text, avg_frame_rate=text))
    assert enc.source_frame_rate(Path("x.mkv")) == pytest.approx(expected,
                                                                 abs=1e-3)


# ==================================================== B  a bounded answer ===

def test_B1_the_bound_is_named_and_is_120():
    assert enc.MAX_SOURCE_FPS == 120.0


@pytest.mark.parametrize("text", ["1000/1", "121/1", "90000/1", "0/1",
                                  "-30/1"])
def test_B2_a_rate_outside_the_bound_is_unreadable(monkeypatch, text):
    # Both fields say the same nonsense: nothing sane to fall back to.
    probed(monkeypatch, video(r_frame_rate=text, avg_frame_rate=text))
    assert enc.source_frame_rate(Path("x.mkv")) is None


def test_B2_exactly_120_is_still_a_source_rate(monkeypatch):
    probed(monkeypatch, video(r_frame_rate="120/1", avg_frame_rate="120/1"))
    assert enc.source_frame_rate(Path("x.mkv")) == pytest.approx(120.0)


def test_B3_a_nonsense_average_falls_back_to_a_sane_nominal(monkeypatch):
    # The order is avg then r, but a candidate outside the bound is skipped,
    # not returned: the first SANE reading wins.
    probed(monkeypatch, video(r_frame_rate="30/1", avg_frame_rate="1000/1"))
    assert enc.source_frame_rate(Path("x.mkv")) == pytest.approx(30.0)


def test_B4_nothing_to_read_is_none(monkeypatch):
    for streams in ([], [audio(r_frame_rate="0/0")], [video()]):
        probed(monkeypatch, *streams)
        assert enc.source_frame_rate(Path("x.mkv")) is None
    monkeypatch.setattr(enc.au, "probe", lambda path: {})
    assert enc.source_frame_rate(Path("x.mkv")) is None


def test_B5_the_bound_is_on_the_file_not_the_person():
    # `--fps` and `--max-fps` are what someone typed; `output_rate` passes
    # them through as it did in Batch 11.
    assert enc.output_rate(enc.EncodeSettings(fps=240.0), 30.0) == 240.0
    assert enc.output_rate(enc.EncodeSettings(max_fps=240.0), None) == 240.0


# ======================================== E  what reaches the ffmpeg line ===

@pytest.fixture
def ffmpeg(monkeypatch):
    """A `subprocess.run` that records the command and fakes a good encode."""
    calls: list[list[str]] = []

    class Done:
        returncode = 0
        stderr = ""

    def fake(cmd, **kwargs):
        calls.append(list(cmd))
        Path(cmd[-1]).write_bytes(b"webm")
        return Done()

    monkeypatch.setattr(enc.subprocess, "run", fake)
    return calls


def test_E1_a_1000_fps_source_encodes_at_the_fallback_not_at_1000(
        tmp_path, monkeypatch, ffmpeg):
    probed(monkeypatch, video(r_frame_rate="1000/1", avg_frame_rate="0/0"))
    src = tmp_path / "video.src.mkv"
    src.write_bytes(b"x")
    ok, note = enc.encode_one(src, tmp_path, enc.EncodeSettings(),
                              keep_source=True)
    assert ok, note
    assert fps_in(ffmpeg[0]) == "30.000000"
    assert "1000" not in fps_in(ffmpeg[0])


def test_E2_the_real_rate_reaches_the_command_when_the_nominal_is_nonsense(
        tmp_path, monkeypatch, ffmpeg):
    probed(monkeypatch, video(r_frame_rate="1000/1",
                              avg_frame_rate="30000/1001"))
    src = tmp_path / "video.src.mkv"
    src.write_bytes(b"x")
    ok, note = enc.encode_one(src, tmp_path, enc.EncodeSettings(),
                              keep_source=True)
    assert ok, note
    assert fps_in(ffmpeg[0]) == "29.970030"


def test_E3_a_cap_still_caps_a_nonsense_source(tmp_path, monkeypatch,
                                                ffmpeg):
    # Unreadable falls back to the cap, as Batch 11 pinned it - so with
    # `--max-fps 24` a 1000/1 file runs at 24, not 30 and not 1000.
    probed(monkeypatch, video(r_frame_rate="1000/1"))
    src = tmp_path / "video.src.mkv"
    src.write_bytes(b"x")
    ok, note = enc.encode_one(src, tmp_path, enc.EncodeSettings(max_fps=24.0),
                              keep_source=True)
    assert ok, note
    assert fps_in(ffmpeg[0]) == "24.000000"
