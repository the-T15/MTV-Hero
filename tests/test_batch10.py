"""
Batch 10 - codec, hardware encoders, size estimate.

Everything in the pipeline assumes one output file, `video.webm`, written by
one encoder, `libvpx`. This batch makes the codec a choice, adds the hardware
encoders that make a 1,423-song run finish, and gives the run a size estimate
before it starts. It must land before the first `encode --reviewed`: that run
deletes the sources and a second codec pass would have nothing to work from.

    N1  `--codec {vp8,h264,h264_nvenc,h264_amf,h264_qsv}` selects a row in
        `encode.CODECS`. A row (`encode.Codec`) carries: `encoder` (the
        ffmpeg `-c:v` name), `quality_flags` (the flag(s) that take the
        quality number; `-crf` for libvpx/libx264, `-cq` for nvenc),
        `crf` (the default quality number, 31 for vp8, 23 for the H.264
        rows), `container` (ffmpeg `-f`), `extension` (`.webm`/`.mp4`),
        `hardware`, `fallback` (the software row a failed hardware row falls
        back to) and `preview` (the fields `replace(...)` applies for
        `--preview`; every row caps the preview at `800k`). The default
        codec stays `vp8`. `EncodeSettings` gains `codec="vp8"`,
        `preset=None` (the row's default preset when None) and
        `crf=None` (the row's default when None); `--crf` defaults to None
        for the same reason. VP8 output is unchanged: `-crf 31 -b:v 4M`
        plus the libvpx flags. H.264 rows emit `-pix_fmt yuv420p`,
        `-movflags +faststart`, `-f mp4`, the cap as `-maxrate`, and none
        of the libvpx-only flags.
    N2  `--fps N` forces the output rate (`EncodeSettings.fps`). With both
        `--fps` and `--max-fps`, `--fps` wins; with neither, the source rate
        is kept under the 30.0 cap as now.
    N3  `check_encoder(name)` asks the ffmpeg build for the encoder AND runs
        a one-frame encode with it; listing proves nothing about the driver.
        `resolve_codec(settings, say=print)` returns settings unchanged for a
        software row, and for a hardware row that fails either check returns
        the row's `fallback` and prints one line saying so. `cmd_encode`
        resolves once per run, never per song. `doctor` prints one line per
        hardware row; a missing hardware encoder is not a missing tool, and
        `h264_amf` / `h264_qsv` are labelled untested.
    N4  `output_name(settings)`, `output_path(song_dir, settings)` and
        `find_output(song_dir)` (whichever codec's file is present) replace
        every `video.webm` literal in `encode_one`, `cmd_encode
        --skip-existing`, `cmd_videos` and `review.py`. Outside `encode.py`
        the literal may survive only in docstrings.
    N5  A successful encode deletes the other codec's output from the folder.
        YARG loads the wrong file when two are present (YARG #1331).
    N6  `--size-lock <bitrate>` is two-pass target-bitrate mode: exact size,
        quality varies per song. `passes(settings)` is 2 for a software row
        under size lock, 1 otherwise (hardware rows encode one pass at the
        target). Pass 1 writes to `os.devnull` with `-f null`; both passes
        share a `-passlogfile` inside the song folder that is removed after
        the encode. `--size-lock` with `--crf` is an argparse error.
    N7  `estimate` is its own subcommand with `encode`'s flags. It reads the
        rows `encode` would run (`cli.encode_rows`), sums `video_seconds`,
        and prints `~ N GB (max M GB)`: the maximum is the bitrate cap times
        the summed seconds; the estimate is the measured bits-per-second for
        `rate_key(settings)`, which covers every setting that changes the
        bits (Batch 10c), times it. `parse_bitrate` accepts exactly what
        ffmpeg means by k/K/M/G and both bitrate flags are validated through
        it at parse time. The
        measurement encodes a 3-song sample with `keep_source=True` into a
        temp folder, never a song folder, fills `encode.RATE_TABLE`, and is
        skipped when the table already has the key or under `--size-lock`.
        It writes nothing to the database or the library.
    P1  The parked `--limit` finding: `encode_rows` applies `--skip-existing`,
        `--reviewed` and `--skip-static` BEFORE `--limit` and before the
        `--sample` draw, so `--limit 2` over three static folders followed by
        two real ones encodes the two real ones, not nothing.

`build_command` is read as a list and ffmpeg is never run; the encoder checks
run against a faked `subprocess.run`, never a real GPU; `encode_many` is
replaced by a fake that calls `on_done`. Every path is a `Path`; row lookups
compare against `str(Path(...))`.

Run from the repository root:  pytest -q
"""

from __future__ import annotations

import ast
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from yargvid import cli
from yargvid import encode as enc
from yargvid import review as rv
from yargvid.db import Database

PKG = Path(enc.__file__).parent
HARDWARE = ("h264_nvenc", "h264_amf", "h264_qsv")


# ----------------------------------------------------------------- helpers ---

@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


@pytest.fixture
def no_ffmpeg(monkeypatch):
    monkeypatch.setattr(enc, "_supports_fps_mode", lambda: True)


@pytest.fixture(autouse=True)
def empty_rate_table(monkeypatch):
    monkeypatch.setattr(enc, "RATE_TABLE", {}, raising=False)
    monkeypatch.setattr(enc, "TYPICAL_RATES", {}, raising=False)


def song(db, path, **cols):
    """A synced, encodable song folder with a song.ini and a source video."""
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


def encode_args(**over):
    """The namespace `encode` produces with no flags."""
    a = dict(limit=None, sample=False, skip_existing=False, reviewed=False,
             skip_static=True, height=1080, crf=None, cpu_used=3, threads=2,
             workers=None, preview=False, preview_height=480,
             bitrate_cap=None, max_fps=None, codec="vp8", fps=None,
             size_lock=None)
    a.update(over)
    return SimpleNamespace(**a)


class FakePool:
    """Stands in for `encode_many`: records the call, reports each job."""

    def __init__(self):
        self.failing = set()
        self.calls = []

    def __call__(self, jobs, settings, workers=None, on_done=None, **kw):
        self.calls.append(dict(jobs=list(jobs), settings=settings,
                               workers=workers, kw=kw))
        results = {}
        for job in jobs:
            d = job[1]
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


@pytest.fixture
def hw_ok(monkeypatch):
    """Every hardware check passes; nothing shells out."""
    monkeypatch.setattr(enc, "check_encoder", lambda name: (True, "ok"))


def parsed(monkeypatch, db, argv, cmd="cmd_encode"):
    """Run `main` far enough to parse, capture the subcommand's namespace."""
    seen = {}
    monkeypatch.setattr(cli, cmd, lambda args, db_: seen.update(vars(args)))
    assert cli.main(["--db", str(db.path), *argv]) == 0
    return SimpleNamespace(**seen)


def cmd_for(codec, **over):
    s = enc.EncodeSettings(codec=codec, **over)
    return enc.build_command(Path("in.mp4"), Path("out"), s, source_fps=30.0), s


class FakeRun:
    """`subprocess.run` for encode_one: records commands, writes the output."""

    def __init__(self, size=10):
        self.cmds = []
        self.size = size

    def __call__(self, cmd, **kw):
        self.cmds.append(list(cmd))
        dst = Path(cmd[-1])
        if cmd[-1] != os.devnull:
            dst.write_bytes(b"x" * self.size)
        return SimpleNamespace(returncode=0, stdout="", stderr="")


# ------------------------------------------------------------- N1 table -----

def test_N1_table_has_the_five_rows_and_vp8_is_the_default():
    assert set(enc.CODECS) == {"vp8", "h264", *HARDWARE}
    assert enc.EncodeSettings().codec == "vp8"
    for name, c in enc.CODECS.items():
        assert c.hardware == (name in HARDWARE)
        assert c.preview.get("bitrate_cap") == "800k"
    assert enc.CODECS["vp8"].crf == 31
    assert all(enc.CODECS[n].crf == 23 for n in ("h264", *HARDWARE))
    assert all(enc.CODECS[n].fallback == "h264" for n in HARDWARE)
    assert enc.CODECS["vp8"].extension == ".webm"
    assert all(enc.CODECS[n].extension == ".mp4" for n in ("h264", *HARDWARE))


def test_N1_codec_flag_is_parsed_and_validated(db, monkeypatch):
    assert parsed(monkeypatch, db, ["encode"]).codec == "vp8"
    a = parsed(monkeypatch, db, ["encode", "--codec", "h264"])
    assert a.codec == "h264"
    assert a.crf is None                       # the row's default applies
    with pytest.raises(SystemExit):
        cli.main(["--db", str(db.path), "encode", "--codec", "av1"])


def test_N1_vp8_command_is_unchanged(no_ffmpeg):
    cmd, _ = cmd_for("vp8")
    assert cmd[cmd.index("-c:v") + 1] == "libvpx"
    assert cmd[cmd.index("-crf") + 1] == "31"
    assert cmd[cmd.index("-b:v") + 1] == "4M"
    assert cmd[cmd.index("-f") + 1] == "webm"
    for flag in ("-qmin", "-qmax", "-deadline", "-cpu-used", "-auto-alt-ref"):
        assert flag in cmd
    assert "-maxrate" not in cmd and "-movflags" not in cmd


def test_N1_h264_command_is_x264_in_mp4(no_ffmpeg):
    cmd, _ = cmd_for("h264")
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-crf") + 1] == "23"
    assert cmd[cmd.index("-preset") + 1] == "medium"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert cmd[cmd.index("-f") + 1] == "mp4"
    assert cmd[cmd.index("-maxrate") + 1] == "4M"
    assert "-b:v" not in cmd                   # x264 ignores it under -crf
    for flag in ("-qmin", "-qmax", "-deadline", "-cpu-used", "-auto-alt-ref"):
        assert flag not in cmd


@pytest.mark.parametrize("name", HARDWARE)
def test_N1_hardware_rows_use_their_encoder_and_quality_flags(no_ffmpeg, name):
    cmd, _ = cmd_for(name)
    c = enc.CODECS[name]
    assert cmd[cmd.index("-c:v") + 1] == c.encoder == name
    for flag in c.quality_flags:
        assert cmd[cmd.index(flag) + 1] == "23"
    assert "-crf" not in cmd
    assert cmd[cmd.index("-f") + 1] == "mp4"
    assert cmd[cmd.index("-maxrate") + 1] == "4M"
    assert "-cpu-used" not in cmd and "-deadline" not in cmd


def test_N1_crf_and_preset_override_the_row_defaults(no_ffmpeg):
    cmd, _ = cmd_for("h264", crf=18, preset="slow")
    assert cmd[cmd.index("-crf") + 1] == "18"
    assert cmd[cmd.index("-preset") + 1] == "slow"
    cmd, _ = cmd_for("vp8", crf=20)
    assert cmd[cmd.index("-crf") + 1] == "20"


def test_N1_cmd_encode_passes_codec_and_the_rows_crf_into_settings(db, pool):
    # Batch 11: the tier fills the number in rather than leaving it None, and
    # `good` is the codec row's own, so the command is unchanged.
    song(db, db.path.parent / "s")
    cli.cmd_encode(encode_args(codec="h264"), db)
    s = pool.calls[0]["settings"]
    assert s.codec == "h264" and enc.effective_crf(s) == 23


def test_N1_preview_applies_the_rows_override(db, pool):
    song(db, db.path.parent / "s")
    cli.cmd_encode(encode_args(codec="h264", preview=True), db)
    s = pool.calls[0]["settings"]
    assert (s.height, s.bitrate_cap) == (480, "800k")
    assert s.preset == enc.CODECS["h264"].preview["preset"]
    assert pool.calls[0]["kw"].get("keep_source") is True


# ------------------------------------------------------------- N2 fps -------

def test_N2_fps_flag_is_parsed_and_forces_the_rate(db, monkeypatch, no_ffmpeg):
    a = parsed(monkeypatch, db, ["encode", "--fps", "60", "--max-fps", "24"])
    assert a.fps == 60.0 and a.max_fps == 24.0
    assert parsed(monkeypatch, db, ["encode"]).fps is None

    s = enc.EncodeSettings(fps=60.0, max_fps=24.0)
    cmd = enc.build_command(Path("in.mp4"), Path("out.webm"), s, source_fps=25.0)
    vf = cmd[cmd.index("-vf") + 1]
    assert "fps=60.000000" in vf              # --fps wins over --max-fps

    s = enc.EncodeSettings(max_fps=30.0)
    cmd = enc.build_command(Path("in.mp4"), Path("out.webm"), s, source_fps=25.0)
    assert "fps=25.000000" in cmd[cmd.index("-vf") + 1]


def test_N2_cmd_encode_passes_fps_into_settings(db, pool):
    song(db, db.path.parent / "s")
    cli.cmd_encode(encode_args(fps=24.0), db)
    assert pool.calls[0]["settings"].fps == 24.0


# ------------------------------------------------------------- N3 hardware --

def fake_subprocess(monkeypatch, listed: bool, encodes: bool):
    calls = []

    def run(cmd, **kw):
        calls.append(list(cmd))
        if "-encoders" in cmd:
            out = "V..... libx264\n" + ("V..... h264_nvenc\n" if listed else "")
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        return SimpleNamespace(returncode=0 if encodes else 1, stdout="",
                               stderr="" if encodes else "no NVENC device")

    monkeypatch.setattr(subprocess, "run", run)
    return calls


def test_N3_check_encoder_needs_the_listing_and_a_one_frame_encode(monkeypatch):
    calls = fake_subprocess(monkeypatch, listed=False, encodes=True)
    ok, why = enc.check_encoder("h264_nvenc")
    assert ok is False and "ffmpeg" in why
    assert len(calls) == 1                     # no encode attempt if unlisted

    calls = fake_subprocess(monkeypatch, listed=True, encodes=False)
    ok, why = enc.check_encoder("h264_nvenc")
    assert ok is False and why
    probe = calls[1]
    assert probe[probe.index("-c:v") + 1] == "h264_nvenc"
    assert probe[probe.index("-frames:v") + 1] == "1"
    assert "lavfi" in probe and "-f" in probe and probe[-1] == os.devnull

    fake_subprocess(monkeypatch, listed=True, encodes=True)
    assert enc.check_encoder("h264_nvenc")[0] is True


def test_N3_resolve_codec_falls_back_to_software_and_says_so(monkeypatch):
    monkeypatch.setattr(enc, "check_encoder",
                        lambda name: (False, "no NVENC device"))
    said = []
    s = enc.resolve_codec(enc.EncodeSettings(codec="h264_nvenc"),
                          say=said.append)
    assert s.codec == "h264"
    assert len(said) == 1
    assert "h264_nvenc" in said[0] and "h264" in said[0]
    assert "no NVENC device" in said[0]

    monkeypatch.setattr(enc, "check_encoder", lambda name: (True, "ok"))
    said.clear()
    s = enc.resolve_codec(enc.EncodeSettings(codec="h264_nvenc"),
                          say=said.append)
    assert s.codec == "h264_nvenc" and said == []


def test_N3_software_rows_are_never_checked(monkeypatch):
    monkeypatch.setattr(enc, "check_encoder",
                        lambda name: pytest.fail("checked a software row"))
    for name in ("vp8", "h264"):
        assert enc.resolve_codec(enc.EncodeSettings(codec=name)).codec == name


def test_N3_cmd_encode_resolves_once_per_run_not_per_song(db, pool, monkeypatch,
                                                          capsys):
    seen = []

    def check(name):
        seen.append(name)
        return False, "driver said no"

    monkeypatch.setattr(enc, "check_encoder", check)
    lib = db.path.parent
    for n in ("a", "b", "c"):
        song(db, lib / n)
    cli.cmd_encode(encode_args(codec="h264_nvenc"), db)
    assert seen == ["h264_nvenc"]
    assert pool.calls[0]["settings"].codec == "h264"
    assert "driver said no" in capsys.readouterr().out


def test_N3_doctor_lists_hardware_rows_without_calling_them_missing_tools(
        db, monkeypatch, capsys):
    monkeypatch.setattr(enc, "have", lambda tool: True)
    monkeypatch.setattr(enc, "check_ffmpeg_vp8", lambda: True)
    monkeypatch.setattr(enc, "check_encoder",
                        lambda name: (name == "h264_nvenc", "no device"))
    assert cli.main(["doctor"]) == 0
    out = capsys.readouterr().out
    lines = {n: next(ln for ln in out.splitlines() if n in ln)
             for n in HARDWARE}
    assert "[ok]" in lines["h264_nvenc"]
    assert "[ok]" not in lines["h264_amf"] and "[ok]" not in lines["h264_qsv"]
    assert "untested" in lines["h264_amf"] and "untested" in lines["h264_qsv"]
    assert "untested" not in lines["h264_nvenc"]
    assert "Install the missing tools" not in out


# ------------------------------------------------------------- N4 filename --

def test_N4_output_helpers_follow_the_codec(tmp_path):
    vp8, h264 = enc.EncodeSettings(), enc.EncodeSettings(codec="h264")
    assert enc.output_name(vp8) == "video.webm"
    assert enc.output_name(h264) == "video.mp4"
    assert enc.output_path(tmp_path, h264) == tmp_path / "video.mp4"
    assert enc.find_output(tmp_path) is None
    (tmp_path / "video.mp4").write_bytes(b"x")
    assert enc.find_output(tmp_path) == tmp_path / "video.mp4"


def _string_constants(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                and id(node) not in docstrings):
            yield node.lineno, node.value


def test_N4_no_literal_survives_outside_encode_py():
    hits = []
    for mod in sorted(PKG.glob("*.py")):
        if mod.name == "encode.py":
            continue
        for lineno, value in _string_constants(mod):
            if "video.webm" in value or "video.mp4" in value:
                hits.append(f"{mod.name}:{lineno}: {value!r}")
    assert hits == []


def test_N4_encode_one_writes_the_codecs_file_via_a_part(tmp_path, monkeypatch,
                                                         no_ffmpeg):
    run = FakeRun()
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(enc, "source_frame_rate", lambda p: 30.0)
    d = tmp_path / "s"
    d.mkdir()
    src = d / "video.src.mp4"
    src.write_bytes(b"x")
    ok, err = enc.encode_one(src, d, enc.EncodeSettings(codec="h264"),
                             keep_source=True)
    assert (ok, err) == (True, "")
    assert (d / "video.mp4").exists()
    assert run.cmds[0][-1] == str(d / "video.mp4.part")
    assert not (d / "video.mp4.part").exists()
    assert not (d / "video.webm").exists()


def test_N4_skip_existing_sees_either_codecs_file(db, pool):
    lib = db.path.parent
    has_mp4 = song(db, lib / "has_mp4")
    (has_mp4 / "video.mp4").write_bytes(b"x")
    has_webm = song(db, lib / "has_webm")
    (has_webm / "video.webm").write_bytes(b"x")
    bare = song(db, lib / "bare")
    cli.cmd_encode(encode_args(skip_existing=True), db)
    assert [job[1] for job in pool.calls[0]["jobs"]] == [bare]


def test_N4_videos_counts_either_codecs_file(db, capsys):
    lib = db.path.parent
    s = song(db, lib / "s", encode_status="ok")
    (s / "video.mp4").write_bytes(b"x" * 1_048_576)
    cli.cmd_videos(SimpleNamespace(quiet=True, out=None, mark=False,
                                   lengths=False), db)
    out = capsys.readouterr().out
    assert "1 encoded by this pipeline" in out
    assert "0 marked encoded but the file is gone" in out


def test_N4_review_finds_either_codecs_file(db):
    lib = db.path.parent
    s = song(db, lib / "s", source_path=str(lib / "gone.mp4"))
    assert rv.source_video(row(db, s)) is None
    (s / "video.mp4").write_bytes(b"x")
    assert rv.source_video(row(db, s)) == s / "video.mp4"

    rv.drop_song(db, s)
    assert not (s / "video.mp4").exists()


# ------------------------------------------------------------- N5 one file --

def test_N5_a_successful_encode_removes_the_other_codecs_output(
        tmp_path, monkeypatch, no_ffmpeg):
    monkeypatch.setattr(subprocess, "run", FakeRun())
    monkeypatch.setattr(enc, "source_frame_rate", lambda p: 30.0)
    d = tmp_path / "s"
    d.mkdir()
    src = d / "video.src.mp4"
    src.write_bytes(b"x")
    (d / "video.mp4").write_bytes(b"old")
    ok, _ = enc.encode_one(src, d, enc.EncodeSettings(codec="vp8"),
                           keep_source=True)
    assert ok
    assert (d / "video.webm").exists()
    assert not (d / "video.mp4").exists()


# ------------------------------------------------------------- N6 size lock -

def test_N6_size_lock_flag_and_its_conflict_with_crf(db, monkeypatch):
    a = parsed(monkeypatch, db, ["encode", "--size-lock", "2500k"])
    assert a.size_lock == "2500k" and a.crf is None
    assert parsed(monkeypatch, db, ["encode"]).size_lock is None
    with pytest.raises(SystemExit):
        cli.main(["--db", str(db.path), "encode", "--size-lock", "2500k",
                  "--crf", "20"])


def test_N6_passes_and_pass_commands(no_ffmpeg):
    assert enc.passes(enc.EncodeSettings()) == 1
    assert enc.passes(enc.EncodeSettings(size_lock="2500k")) == 2
    assert enc.passes(enc.EncodeSettings(codec="h264", size_lock="2500k")) == 2
    assert enc.passes(enc.EncodeSettings(codec="h264_nvenc",
                                         size_lock="2500k")) == 1

    s = enc.EncodeSettings(codec="h264", size_lock="2500k")
    one = enc.build_command(Path("in.mp4"), Path("out.mp4"), s, 30.0, pass_no=1)
    two = enc.build_command(Path("in.mp4"), Path("out.mp4"), s, 30.0, pass_no=2)
    for cmd in (one, two):
        assert cmd[cmd.index("-b:v") + 1] == "2500k"
        assert "-crf" not in cmd and "-maxrate" not in cmd
        assert "-passlogfile" in cmd
    assert one[one.index("-pass") + 1] == "1"
    assert one[one.index("-f") + 1] == "null" and one[-1] == os.devnull
    assert two[two.index("-pass") + 1] == "2"
    assert two[-1] == "out.mp4"
    assert (one[one.index("-passlogfile") + 1]
            == two[two.index("-passlogfile") + 1])

    hw = enc.EncodeSettings(codec="h264_nvenc", size_lock="2500k")
    cmd = enc.build_command(Path("in.mp4"), Path("out.mp4"), hw, 30.0)
    assert cmd[cmd.index("-b:v") + 1] == "2500k" and "-pass" not in cmd


def test_N6_encode_one_runs_two_passes_and_cleans_the_log(tmp_path, monkeypatch,
                                                          no_ffmpeg):
    run = FakeRun()

    def run_and_log(cmd, **kw):
        if "-passlogfile" in cmd:
            Path(cmd[cmd.index("-passlogfile") + 1] + "-0.log").write_bytes(b"l")
        return run(cmd, **kw)

    monkeypatch.setattr(subprocess, "run", run_and_log)
    monkeypatch.setattr(enc, "source_frame_rate", lambda p: 30.0)
    d = tmp_path / "s"
    d.mkdir()
    src = d / "video.src.mp4"
    src.write_bytes(b"x")
    ok, err = enc.encode_one(src, d, enc.EncodeSettings(codec="h264",
                                                        size_lock="2500k"),
                             keep_source=True)
    assert (ok, err) == (True, "")
    assert [c[c.index("-pass") + 1] for c in run.cmds] == ["1", "2"]
    assert sorted(p.name for p in d.iterdir()) == ["video.mp4", "video.src.mp4"]


def test_N6_cmd_encode_passes_size_lock_into_settings(db, pool):
    song(db, db.path.parent / "s")
    cli.cmd_encode(encode_args(size_lock="2500k"), db)
    assert pool.calls[0]["settings"].size_lock == "2500k"


# ------------------------------------------------------------- N7 estimate --

def test_N7_estimate_has_encodes_flags(db, monkeypatch):
    a = parsed(monkeypatch, db, ["--limit", "5", "estimate", "--codec", "h264",
                                 "--reviewed", "--skip-existing",
                                 "--include-static", "--height", "720"],
               cmd="cmd_estimate")
    assert (a.codec, a.reviewed, a.skip_existing, a.skip_static, a.height,
            a.limit) == ("h264", True, True, False, 720, 5)
    assert a.bitrate_cap is None and a.size_lock is None
    assert cli.encode_settings(a).bitrate_cap == "4M"


def test_N7_parse_bitrate_speaks_ffmpeg():
    """ffmpeg reads SI prefixes: `4m` is milli, i.e. `-b:v 0` in disguise."""
    assert enc.parse_bitrate("4M") == 4_000_000
    assert enc.parse_bitrate("800k") == 800_000
    assert enc.parse_bitrate("2500K") == 2_500_000
    assert enc.parse_bitrate("1G") == 1_000_000_000
    assert enc.parse_bitrate("123456") == 123_456
    for bad in ("fast", "4m", "4g", "", "4 M"):
        with pytest.raises(ValueError):
            enc.parse_bitrate(bad)


def test_N7_encode_and_estimate_refuse_a_bitrate_ffmpeg_would_misread(
        db, monkeypatch):
    for cmd in ("encode", "estimate"):
        for flag in ("--bitrate-cap", "--size-lock"):
            with pytest.raises(SystemExit):
                cli.main(["--db", str(db.path), cmd, flag, "4m"])
    a = parsed(monkeypatch, db, ["encode", "--bitrate-cap", "6M"])
    assert a.bitrate_cap == "6M"


def test_N7_rate_key_carries_everything_the_rate_depends_on():
    """`estimate --crf 40` after `--crf 18` must measure again, not reuse."""
    key = enc.rate_key
    base = enc.EncodeSettings()
    assert key(enc.EncodeSettings()) == key(base)
    assert key(enc.EncodeSettings(crf=31)) == key(base)   # vp8's own default
    for other in (enc.EncodeSettings(crf=18), enc.EncodeSettings(crf=40),
                  enc.EncodeSettings(bitrate_cap="2M"),
                  enc.EncodeSettings(height=720),
                  enc.EncodeSettings(fps=24.0),
                  enc.EncodeSettings(codec="h264_nvenc")):
        assert key(other) != key(base), other
    assert key(enc.EncodeSettings(crf=18)) != key(enc.EncodeSettings(crf=40))
    # Under a size lock the rate is the lock; the table is not consulted.
    assert (key(enc.EncodeSettings(size_lock="2M"))
            == key(enc.EncodeSettings(size_lock="3M")) == key(base))


def test_N7_a_changed_quality_measures_again(db, monkeypatch, capsys):
    lib = db.path.parent
    for n in ("a", "b", "c"):
        song(db, lib / n)
    rates = iter([2_000_000.0, 800_000.0])
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: next(rates))
    cli.cmd_estimate(encode_args(crf=18), db)
    assert "~ 0.15 GB" in capsys.readouterr().out
    cli.cmd_estimate(encode_args(crf=40), db)
    assert "~ 0.06 GB" in capsys.readouterr().out
    assert len(enc.RATE_TABLE) == 2


def test_N7_estimate_prints_gb_and_max_and_fills_the_table(db, monkeypatch,
                                                           capsys):
    lib = db.path.parent
    for n in ("a", "b", "c"):
        song(db, lib / n)                      # 3 x 200 s = 600 s
    sampled = []

    def measure(rows, settings, workers=None):
        sampled.append(list(rows))
        return 2_000_000.0                     # bits per second

    monkeypatch.setattr(enc, "measure_rate", measure)
    cli.cmd_estimate(encode_args(), db)
    out = capsys.readouterr().out
    # 2 Mbit/s x 600 s = 0.15 GB; 4 Mbit/s x 600 s = 0.30 GB
    assert "~ 0.15 GB (max 0.30 GB)" in out
    assert len(sampled) == 1 and len(sampled[0]) == 3
    assert enc.RATE_TABLE[enc.rate_key(enc.EncodeSettings())] == 2_000_000.0

    # A known rate is reused: the GUI will call this repeatedly.
    cli.cmd_estimate(encode_args(), db)
    assert len(sampled) == 1
    assert "~ 0.15 GB (max 0.30 GB)" in capsys.readouterr().out


def test_N7_estimate_under_size_lock_is_exact_and_samples_nothing(
        db, monkeypatch, capsys):
    lib = db.path.parent
    for n in ("a", "b", "c"):
        song(db, lib / n)
    monkeypatch.setattr(enc, "measure_rate",
                        lambda *a, **k: pytest.fail("sampled under size lock"))
    cli.cmd_estimate(encode_args(size_lock="2M"), db)
    assert "~ 0.15 GB (max 0.15 GB)" in capsys.readouterr().out
    assert enc.RATE_TABLE == {}


def test_N7_estimate_writes_nothing_and_touches_no_song_folder(db, monkeypatch,
                                                               capsys):
    lib = db.path.parent
    s = song(db, lib / "s")
    before = tuple(row(db, s))
    ini = (s / "song.ini").read_text(encoding="utf-8")
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: 1_000_000.0)
    cli.cmd_estimate(encode_args(), db)
    assert tuple(row(db, s)) == before
    assert (s / "song.ini").read_text(encoding="utf-8") == ini
    assert sorted(p.name for p in s.iterdir()) == ["song.ini", "video.mp4.src"]


def test_N7_estimate_samples_after_the_filters(db, monkeypatch):
    lib = db.path.parent
    stills = {song(db, lib / f"still{i}", motion=0.01) for i in range(2)}
    footage = {song(db, lib / f"footage{i}") for i in range(5)}
    sampled = []
    monkeypatch.setattr(enc, "measure_rate",
                        lambda rows, *a, **k: sampled.extend(rows) or 1e6)
    cli.cmd_estimate(encode_args(), db)
    dirs = {Path(r["song_dir"]) for r in sampled}
    assert len(dirs) == 5
    assert dirs <= footage and not dirs & stills


def test_N7_estimate_names_songs_missing_a_length(db, monkeypatch, capsys):
    lib = db.path.parent
    song(db, lib / "a")
    song(db, lib / "b", video_seconds=None)
    monkeypatch.setattr(enc, "measure_rate", lambda *a, **k: 2_000_000.0)
    cli.cmd_estimate(encode_args(), db)
    out = capsys.readouterr().out
    assert "1 " in out and "videos --lengths" in out
    # Only the measured 200 s count.
    assert "~ 0.05 GB (max 0.10 GB)" in out


def test_N7_measure_rate_encodes_into_a_temp_folder_with_sources_kept(
        db, monkeypatch):
    lib = db.path.parent
    rows = [row(db, song(db, lib / n)) for n in ("a", "b")]   # 2 x 200 s
    calls = []

    def fake_many(jobs, settings, workers=None, on_done=None, **kw):
        calls.append(dict(jobs=list(jobs), kw=kw))
        out = {}
        for job in jobs:
            d = job[1]
            enc.output_path(d, settings).write_bytes(b"x" * 25_000)
            out[d] = (True, "")
        return out

    monkeypatch.setattr(enc, "encode_many", fake_many)
    bps = enc.measure_rate(rows, enc.EncodeSettings(), workers=1)
    # 8 x 50,000 bytes over the 2 x 20 s slices encoded (Batch 10b) = 10,000
    # bits per second.
    assert bps == pytest.approx(10_000.0)
    call = calls[0]
    assert call["kw"].get("keep_source") is True
    for job in call["jobs"]:
        d = job[1]
        assert lib not in d.parents and d != lib
        assert not d.exists()                  # the temp folder is gone
    for r in rows:
        assert sorted(p.name for p in Path(r["song_dir"]).iterdir()) == [
            "song.ini", "video.mp4.src"]


# ------------------------------------------------------------- P1 limit -----

def test_P1_limit_applies_after_the_filters(db, pool):
    lib = db.path.parent
    for n in ("a_still", "b_still", "c_still"):
        song(db, lib / n, motion=0.01)
    real = [song(db, lib / n) for n in ("d_real", "e_real")]
    cli.cmd_encode(encode_args(limit=2), db)
    assert [job[1] for job in pool.calls[0]["jobs"]] == real


def test_P1_limit_applies_after_reviewed_and_skip_existing(db, pool):
    lib = db.path.parent
    song(db, lib / "a_unreviewed")
    b = song(db, lib / "b_has_video", review="keep")
    (b / "video.webm").write_bytes(b"x")
    c = song(db, lib / "c_keep", review="keep")
    cli.cmd_encode(encode_args(limit=1, reviewed=True, skip_existing=True), db)
    assert [job[1] for job in pool.calls[0]["jobs"]] == [c]


def test_P1_encode_rows_is_shared_by_encode_and_estimate(db, monkeypatch):
    lib = db.path.parent
    song(db, lib / "still", motion=0.01)
    keep = song(db, lib / "keep")
    rows = cli.encode_rows(encode_args(), db)
    assert [Path(r["song_dir"]) for r in rows] == [keep]
