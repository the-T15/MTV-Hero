"""
Video encoding, one codec table wide.

Linux and Steam Deck YARG only reliably play VP8 in a WebM container, so VP8
is the default. The table below makes the codec a choice anyway, because
libvpx is slow enough that a 1,400-song library is measured in days: the
H.264 rows - libx264 and the three hardware encoders - are what make that run
finish. `--codec h264` writes an mp4 that both YARG and Clone Hero load on
Windows, run on 2026-09-17; what is still unrun is Linux and the Steam Deck,
which is the only reason the default has not moved.

Three changes from the previous approach address the "too slow" problem:

1. CORRECT CONSTANT-QUALITY FLAGS. libvpx-vp8 is not VP9. The `-b:v 0` idiom
   is VP9-only; for VP8, constrained quality needs `-crf N -b:v <cap>`, where
   the bitrate acts as a ceiling. With `-b:v 0` the encoder has no headroom to
   work with, which wastes time and hurts quality simultaneously.

2. THROUGHPUT, NOT LATENCY. libvpx-vp8 scales badly past a couple of threads.
   Running one encode on eight threads is far slower than eight encodes on one
   thread each. The pool below saturates the machine with whole jobs.

3. PRESERVE SOURCE FRAME RATE. Forcing everything to 29.97 introduces judder on
   24 and 25 fps sources for no benefit. We enforce *constant* frame rate at
   whatever the source natively runs at.

Audio is dropped entirely (`-an`): YARG plays the chart stems, so a soundtrack
in the background video is dead weight in both file size and encode time.

IMPORTANT (YARG issue #1331): YARG can select the wrong file when a folder
contains more than one video. After a successful encode the source file MUST be
removed or moved outside the song folder, or you may get a black background
with nothing logged.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from . import audio as au


@dataclass(frozen=True)
class Codec:
    """One row of the codec table: everything that differs between encoders."""

    encoder: str                        # ffmpeg -c:v name
    quality_flags: tuple[str, ...]      # flag(s) that take the quality number
    crf: int                            # default quality number for this row
    container: str                      # ffmpeg -f
    extension: str                      # output file extension
    preset: str | None = None           # value for -preset; None = omit it
    cap_flag: str = "-maxrate"          # how the bitrate ceiling is spelled
    speed_flag: str | None = None       # takes EncodeSettings.cpu_used
    hardware: bool = False
    fallback: str | None = None         # software row a failed one falls to
    preview: dict = field(default_factory=dict)   # replace() fields, --preview
    extra: tuple[str, ...] = ()         # fixed flags this encoder always wants
    cq_extra: tuple[str, ...] = ()      # ... and only in constant-quality mode


# libvpx is the only row YARG is known to play everywhere, so it stays the
# default. The H.264 rows all write mp4, where -pix_fmt yuv420p and
# +faststart are what make the file play on something other than this machine.
CODECS: dict[str, Codec] = {
    "vp8": Codec(
        encoder="libvpx",
        quality_flags=("-crf",),
        crf=31,                         # VP8 CQ: 4-63, lower is better
        container="webm",
        extension=".webm",
        # NOTES: libvpx-VP8 needs `-crf N -b:v <cap>`. `-b:v 0` is a VP9
        # idiom and on VP8 silently means 256 kbit/s.
        cap_flag="-b:v",
        speed_flag="-cpu-used",
        preview={"cpu_used": 5, "bitrate_cap": "800k"},
        extra=("-qmin", "4", "-qmax", "56", "-deadline", "good",
               "-auto-alt-ref", "0"),
    ),
    "h264": Codec(
        encoder="libx264",
        quality_flags=("-crf",),
        crf=23,
        container="mp4",
        extension=".mp4",
        preset="medium",
        preview={"bitrate_cap": "800k", "preset": "ultrafast"},
        extra=("-movflags", "+faststart"),
    ),
    "h264_nvenc": Codec(
        encoder="h264_nvenc",
        quality_flags=("-cq",),
        crf=23,
        container="mp4",
        extension=".mp4",
        hardware=True,
        fallback="h264",
        preview={"bitrate_cap": "800k"},
        extra=("-movflags", "+faststart"),
        cq_extra=("-rc", "vbr"),
    ),
    "h264_amf": Codec(
        encoder="h264_amf",
        quality_flags=("-qp_i", "-qp_p", "-qp_b"),
        crf=23,
        container="mp4",
        extension=".mp4",
        hardware=True,
        fallback="h264",
        preview={"bitrate_cap": "800k"},
        extra=("-movflags", "+faststart"),
        cq_extra=("-rc", "cqp"),
    ),
    "h264_qsv": Codec(
        encoder="h264_qsv",
        quality_flags=("-global_quality",),
        crf=23,
        container="mp4",
        extension=".mp4",
        hardware=True,
        fallback="h264",
        preview={"bitrate_cap": "800k"},
        extra=("-movflags", "+faststart"),
    ),
}

# Only the NVIDIA row has been run against real hardware by this project. The
# other two are written from the ffmpeg documentation, and `doctor` says so
# rather than letting a green line imply they were tried.
UNTESTED = ("h264_amf", "h264_qsv")

# Every filename this pipeline can write. A folder must hold at most one of
# them (YARG issue #1331), so every "is there a video here" question asks
# about all of them and a finished encode deletes the rest.
OUTPUT_NAMES = tuple(dict.fromkeys(f"video{c.extension}"
                                   for c in CODECS.values()))


@dataclass
class EncodeSettings:
    height: int = 1080          # 720 roughly halves encode time
    crf: int | None = None      # None = the codec row's own default
    bitrate_cap: str = "4M"     # ceiling for CQ mode - NOT 0 for VP8
    cpu_used: int = 3           # 0-5 with `-deadline good`; higher = faster
    threads_per_job: int = 2
    drop_audio: bool = True
    max_fps: float = 30.0       # cap only; source rate preserved below this
    codec: str = "vp8"          # a key of CODECS
    preset: str | None = None   # None = the codec row's own default
    fps: float | None = None    # force a rate; beats max_fps
    size_lock: str | None = None    # two-pass target bitrate, e.g. "2500k"
    # (start, length) in seconds: encode this slice of the source instead of
    # all of it. Only `measure_rate` sets it - a sample is a slice, and
    # nothing that lands in a song folder ever is.
    clip: tuple[float, float] | None = None


def codec_of(settings: EncodeSettings) -> Codec:
    return CODECS[settings.codec]


def output_name(settings: EncodeSettings) -> str:
    return f"video{codec_of(settings).extension}"


def output_path(song_dir: Path, settings: EncodeSettings) -> Path:
    return Path(song_dir) / output_name(settings)


def find_output(song_dir: Path) -> Path | None:
    """Whichever codec's output this folder holds, if any."""
    for name in OUTPUT_NAMES:
        candidate = Path(song_dir) / name
        if candidate.exists():
            return candidate
    return None


# What ffmpeg's own number parser does with a trailing letter. Only these
# four are SI prefixes to it, and the case matters in one place that costs
# you the whole encode - see parse_bitrate.
BITRATE_SUFFIXES = {"k": 1_000, "K": 1_000, "M": 1_000_000, "G": 1_000_000_000}

# A number, optionally with a dot, optionally with one of those four letters,
# and nothing else at all - no sign, no space, no second suffix. Anchored at
# both ends because the whole point is to be no more permissive than ffmpeg.
_BITRATE = re.compile(r"\A(\d+(?:\.\d+)?)([kKMG]?)\Z")

BITRATE_SPELLING = "a number on its own or with k, K, M or G - 800k, 4M"


def parse_bitrate(text: str) -> int:
    """
    `4M` -> 4000000, reading the string exactly as ffmpeg reads it.

    ffmpeg's expression parser takes SI prefixes, and in SI a lowercase `m`
    is MILLI. `-b:v 4m` is therefore four thousandths of a bit per second,
    which truncates to zero - and `-b:v 0` on libvpx is the documented trap
    in NOTES that silently encodes the library at a fraction of the intended
    bitrate. Measured: `-b:v 4m` and `-b:v 0` produce byte-identical output.
    A lowercase `g` is not a prefix to ffmpeg at all and is simply refused.

    So the grammar here is exactly ffmpeg's and not one character wider. The
    settings carry the string the user typed all the way to the command line,
    so a parser more generous than ffmpeg is a parser that approves a command
    line ffmpeg will read differently - which is the whole failure. That
    includes whitespace: `float()` would accept `"4 "`, ffmpeg will not.
    """
    s = str(text)
    found = _BITRATE.match(s)
    if found:
        number, suffix = found.groups()
        return int(float(number) * BITRATE_SUFFIXES.get(suffix, 1))

    # A number with a suffix ffmpeg reads differently is worth its own
    # sentence: the user wrote something meaningful and got it slightly wrong.
    tail = s[-1:]
    if tail in ("m", "g") and _BITRATE.match(s[:-1]):
        reads = ("milli - a thousandth of a bit per second, which is `-b:v 0` "
                 "in disguise" if tail == "m" else
                 "no prefix at all, and refuses the command")
        raise ValueError(
            f"{text!r}: ffmpeg reads a lowercase {tail!r} as {reads}. "
            f"Write {s[:-1]}{tail.upper()}; the spelling is "
            f"{BITRATE_SPELLING}."
        )
    raise ValueError(f"not a bitrate: {text!r}. Write {BITRATE_SPELLING}.")


def _bufsize(cap: str) -> str:
    try:
        return str(parse_bitrate(cap) * 2)
    except ValueError:
        return cap


def passes(settings: EncodeSettings) -> int:
    """
    How many ffmpeg runs one encode takes.

    Hitting an exact size means telling the encoder in advance where the bits
    went, which is what a first pass is for. Hardware encoders have no
    two-pass mode worth the wall clock: they take the target and rate-control
    to it in one.
    """
    if settings.size_lock and not codec_of(settings).hardware:
        return 2
    return 1


def output_rate(settings: EncodeSettings, source_fps: float | None) -> float:
    """`--fps` forces a rate; otherwise the source's own, under the cap."""
    if settings.fps:
        fps = float(settings.fps)
    else:
        fps = min(source_fps or settings.max_fps, settings.max_fps)
    return fps if fps > 0 else 30.0


def effective_crf(settings: EncodeSettings) -> int:
    """The quality number this encode will really use, default resolved."""
    row = codec_of(settings)
    return settings.crf if settings.crf is not None else row.crf


# What `rate_key` strips out of the command before calling it a key. The
# pairs are a flag and the value after it; the bare ones stand alone. They
# name the file being read, the file being written, ffmpeg's own invariant
# preamble, and how the machine is being driven rather than what is being
# asked of it. `-vsync` is here beside `-fps_mode` so that the key of a
# recipe does not depend on which ffmpeg measured it.
#
# `-threads` is the one that is not free. libvpx partitions the frame by
# thread count, so it moves the size: measured on one 20 s 720p-into-360p
# clip, 1,633,637 bytes at `-threads 1` against 1,887,749 at `-threads 8`,
# 16% apart. It is out anyway - `--threads` is the flag you turn to fit the
# machine, not to change the picture - but that is a decision, not a free
# one, and it is parked in WORKLOG rather than hidden here.
KEY_DROP_PAIRS = ("-i", "-threads", "-ss", "-t", "-passlogfile",
                  "-v", "-fps_mode", "-vsync")
KEY_DROP_FLAGS = ("ffmpeg", "-y", "-nostdin")


def rate_key(settings: EncodeSettings) -> tuple[str, ...]:
    """
    What a measured bits-per-second figure is actually a figure for.

    It is a figure for one ffmpeg command line, so the command line is the
    key. `build_command` is called on these settings and the parts that
    cannot change the bits are struck out: the input, the output, `-threads`,
    the clip, the pass log and the constant preamble. What is left is every
    flag that decides how big a second of video comes out, in the order
    ffmpeg will receive them.

    This is the point of doing it this way. A hand-written tuple has to be
    remembered, and it was not: the key carried six settings while `--crf`,
    `--bitrate-cap`, `--cpu-used` and the codec row's own `preset` all
    changed the answer, and `cpu_used` alone measured 44% apart under one
    key. Deriving it from the
    command means a flag added to `build_command` is in the key the same day,
    and a flag that changes nothing on the command line changes nothing here.

    Two settings are deliberately taken out first. `size_lock` is ignored
    because under a lock nothing consults the table at all - the lock IS the
    rate. `clip` is ignored because a measurement encodes 20 seconds out of
    the middle of a song precisely so that it stands for the whole of it; a
    key that carried the slice would make every sample its own answer.

    The frame rate is resolved against `max_fps` as the source rate, so
    `--fps`, `--max-fps` and a source slower than either all reach the key
    through the one `fps=` in the filter chain that ffmpeg is actually given.
    """
    probe = replace(settings, clip=None, size_lock=None)
    dst = Path("out")
    cmd = build_command(Path("in"), dst, probe, probe.max_fps)

    key: list[str] = []
    drop_value = False
    for token in cmd:
        if drop_value:
            drop_value = False
        elif token in KEY_DROP_PAIRS:
            drop_value = True
        elif token in KEY_DROP_FLAGS or token == str(dst):
            continue
        else:
            key.append(token)
    return tuple(key)


# Measured bits per second per rate_key, filled by `measure_rate`. A module
# global so the GUI can ask `estimate` for a number over and over without
# paying for a sample encode each time.
RATE_TABLE: dict[tuple, float] = {}

# Bits per second measured on this project's own library on 2026-09-17, for
# the two recipes it has actually run end to end. A first estimate comes from
# here, so "how much disk does this need" is answered the moment it is asked;
# `estimate --measure` replaces the figure with one from this machine and
# these videos. Keyed on the three settings that move the number most - the
# encoder, the height and how hard it is being asked to try - rather than on
# the whole rate_key, because a typical figure is a published constant and
# every dimension added to it is one more row nobody has measured.
TYPICAL_RATES: dict[tuple, float] = {
    ("vp8", 1080, 31): 2.9e6,
    ("h264_nvenc", 1080, 23): 3.4e6,
}


def typical_rate(settings: EncodeSettings) -> float | None:
    """The published figure for these settings, or None if there is none."""
    return TYPICAL_RATES.get(
        (settings.codec, settings.height, effective_crf(settings))
    )


def build_command(
    src: Path,
    dst: Path,
    settings: EncodeSettings,
    source_fps: float | None = None,
    pass_no: int | None = None,
) -> list[str]:
    """
    The ffmpeg command for one encode, per the codec table.

    `pass_no` is 1 or 2 under `--size-lock` on a software row. Pass 1 decodes
    the whole video to write the statistics file and throws the pixels away,
    so it has no container and no output: it ends `-f null` at the null
    device. Both passes share one `-passlogfile`, which is the only thing
    that makes pass 2 a second pass rather than a repeat of the first.

    `settings.clip` adds `-ss` and `-t` and changes nothing else, so a
    sample encode is the real command over a shorter stretch of video.
    """
    row = codec_of(settings)
    h = settings.height
    w = int(h * 16 / 9)
    if w % 2:
        w += 1

    # Downscale if larger, pad if smaller. Never stretch; SAR stays 1.
    vf = (
        f"scale={w}:{h}:flags=lanczos:force_original_aspect_ratio=decrease,"
        f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"setsar=1"
    )
    vf += f",fps={output_rate(settings, source_fps):.6f}"

    # A clip is two flags either side of the input and nothing else. -ss
    # BEFORE -i seeks the input, which is what makes a 20-second sample cost
    # 20 seconds rather than a whole decode with 20 seconds kept; -t after it
    # bounds the output.
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin"]
    if settings.clip:
        cmd += ["-ss", f"{settings.clip[0]:.3f}"]
    cmd += ["-i", str(src)]
    if settings.clip:
        cmd += ["-t", f"{settings.clip[1]:.3f}"]
    cmd += ["-c:v", row.encoder]

    if settings.size_lock:
        # Target bitrate: the size is the input and the quality is whatever
        # that buys. No ceiling, because the target IS the ceiling.
        cmd += ["-b:v", settings.size_lock]
    else:
        quality = str(settings.crf if settings.crf is not None else row.crf)
        for flag in row.quality_flags:
            cmd += [flag, quality]
        cmd += list(row.cq_extra)
        cmd += [row.cap_flag, settings.bitrate_cap]
        if row.cap_flag == "-maxrate":
            cmd += ["-bufsize", _bufsize(settings.bitrate_cap)]

    preset = settings.preset or row.preset
    if preset:
        cmd += ["-preset", preset]
    if row.speed_flag:
        cmd += [row.speed_flag, str(settings.cpu_used)]
    cmd += list(row.extra)
    cmd += [
        "-threads", str(settings.threads_per_job),
        "-pix_fmt", "yuv420p",
        "-vf", vf,
    ]
    # -vsync was deprecated in ffmpeg 5.x in favour of -fps_mode and warns
    # loudly on modern builds. Fall back only for genuinely old ffmpeg.
    cmd += ["-fps_mode", "cfr"] if _supports_fps_mode() else ["-vsync", "cfr"]
    if pass_no:
        cmd += ["-pass", str(pass_no), "-passlogfile", pass_log(dst)]
    cmd += ["-an"] if settings.drop_audio else ["-c:a", "libvorbis", "-q:a", "4"]
    if pass_no == 1:
        cmd += ["-f", "null", os.devnull]
    else:
        cmd += ["-f", row.container, str(dst)]
    return cmd


def pass_log(dst: Path) -> str:
    """The -passlogfile prefix for an output. ffmpeg appends `-N.log`."""
    return f"{dst}.2pass"


@functools.lru_cache(maxsize=1)
def _supports_fps_mode() -> bool:
    """ffmpeg 5.0 and later expose -fps_mode; earlier builds need -vsync."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=60
        ).stdout
        version = out.split("ffmpeg version", 1)[1].strip().split()[0]
        major = int("".join(c for c in version.split(".")[0] if c.isdigit()))
        return major >= 5
    except Exception:
        return True


def source_frame_rate(src: Path) -> float | None:
    info = au.probe(src)
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        rate = stream.get("r_frame_rate") or ""
        if "/" in rate:
            num, den = rate.split("/")
            try:
                if float(den) > 0:
                    return float(num) / float(den)
            except ValueError:
                pass
    return None


def _replace_with_retry(tmp: Path, dst: Path, attempts: int = 6) -> str:
    """
    Replace dst with tmp, tolerating transient Windows file locks.

    Windows refuses to replace a file that another process has open without
    FILE_SHARE_DELETE. In practice that means YARG playing the background, a
    media player, or Explorer's thumbnail handler touching the folder. Most of
    these release within seconds, so a short backoff clears them; a persistent
    lock gets a message naming the likely cause instead of a raw WinError.
    """
    delay = 0.5
    for attempt in range(attempts):
        try:
            os.replace(tmp, dst)
            return ""
        except PermissionError:
            if attempt == attempts - 1:
                tmp.unlink(missing_ok=True)
                return (
                    f"{dst.name} is locked by another process - close YARG, any "
                    f"video player, and Explorer windows showing this folder"
                )
            time.sleep(delay)
            delay *= 2
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            return f"could not write {dst.name}: {exc}"
    return ""


def _remove_with_retry(path: Path, attempts: int = 6) -> str:
    """Delete a file, tolerating the same transient locks as a replace."""
    delay = 0.5
    for attempt in range(attempts):
        try:
            path.unlink(missing_ok=True)
            return ""
        except PermissionError:
            if attempt == attempts - 1:
                return (
                    f"encoded, but {path.name} is still in the folder and "
                    f"locked by another process - YARG loads the wrong file "
                    f"when a folder holds two videos, so close YARG and any "
                    f"video player and run this song again"
                )
            time.sleep(delay)
            delay *= 2
        except OSError as exc:
            return f"encoded, but could not remove {path.name}: {exc}"
    return ""


def encode_one(
    src: Path,
    song_dir: Path,
    settings: EncodeSettings,
    keep_source: bool = False,
) -> tuple[bool, str]:
    """
    Encode one video into the codec's output file in `song_dir`.

    Previews are not a separate mode here. A preview is just this function
    called with a low `height` and `keep_source=True`: full length, correct
    timing, real filename, so it can actually be played in YARG. Truncating to
    N seconds would break any song whose video_start_time is positive - YARG
    seeks into the file, so a 30-second clip of a video seeked to 22s leaves
    almost nothing to watch.
    """
    song_dir = Path(song_dir)
    dst = output_path(song_dir, settings)
    tmp = Path(f"{dst}.part")
    fps = source_frame_rate(src)
    total = passes(settings)
    proc = None

    try:
        for n in range(1, total + 1):
            cmd = build_command(src, tmp, settings, fps,
                                pass_no=n if total > 1 else None)
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=7200,
                )
            except subprocess.TimeoutExpired:
                tmp.unlink(missing_ok=True)
                return False, "encode timed out"
            if proc.returncode != 0:
                tmp.unlink(missing_ok=True)
                return False, (proc.stderr or "ffmpeg failed").strip()[:300]

        if not tmp.exists() or tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            note = (proc.stderr if proc else "") or "ffmpeg failed"
            return False, note.strip()[:300]
    finally:
        # The statistics file is ffmpeg's scratch space, not an output. Left
        # behind it is one more file in a folder whose whole rule is that it
        # holds exactly one video.
        prefix = Path(pass_log(tmp)).name
        for log in song_dir.glob(f"{prefix}*"):
            log.unlink(missing_ok=True)

    err = _replace_with_retry(tmp, dst)
    if err:
        return False, err

    # YARG issue #1331: it picks a file out of the folder and can pick the
    # wrong one. Switching codec leaves the previous codec's output sitting
    # next to the new one, which is exactly that failure - and unlike a stray
    # source file it is a real, playable video, so it looks like the encode
    # simply had no effect.
    #
    # `missing_ok` covers a file that is not there; it does not cover one that
    # is there and locked, which on Windows is the common case - YARG playing
    # the old background is the very thing that leaves it open. Raising here
    # would report a song whose new video is already on disk as a failed
    # encode, so retry the way replacing the output does.
    for name in OUTPUT_NAMES:
        other = song_dir / name
        if other != dst:
            err = _remove_with_retry(other)
            if err:
                return False, err

    # Critical: a stray source file in the folder can shadow the output.
    if not keep_source and src.parent == song_dir:
        src.unlink(missing_ok=True)

    return True, ""


STATIC_THRESHOLD = 0.029


def motion_score(src: Path, samples: int = 14) -> float:
    """
    Mean absolute difference between frames sampled across the video.

    Titles and channel names cannot tell you whether a background actually
    moves. An 'Official Audio' upload, a '- Topic' track and an anonymous
    reupload of album art all look different in metadata and identical on
    screen. This measures the thing we actually care about.

    Returns roughly 0-255. Album art sits near zero; real footage is well
    above STATIC_THRESHOLD. Costs about a second per video.
    """
    duration = au.duration_of(src)
    if duration <= 0:
        return -1.0

    # Sample across the middle 90%, avoiding fade-ins and end cards.
    start, span = duration * 0.05, duration * 0.90
    step = span / max(1, samples)
    frames: list[np.ndarray] = []

    for i in range(samples):
        ts = start + i * step
        try:
            raw = subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-nostdin",
                    "-ss", f"{ts:.2f}", "-i", str(src),
                    "-frames:v", "1",
                    "-vf", "scale=64:36,format=gray",
                    "-f", "rawvideo", "-",
                ],
                capture_output=True, timeout=120,
            ).stdout
        except subprocess.TimeoutExpired:
            continue
        if len(raw) == 64 * 36:
            frames.append(
                np.frombuffer(raw, dtype=np.uint8).astype(np.int16)
            )

    if len(frames) < 3:
        return -1.0

    stack = np.stack(frames)
    # Absolute frame difference scales with contrast, so dim footage scores low
    # purely for being dark - a large category of music videos. Dividing by the
    # frames' own spatial contrast makes the measure brightness-independent.
    # Genuinely static art stays at zero either way, since identical frames
    # differ by nothing regardless of how the result is scaled.
    contrast = float(np.median(stack.std(axis=1)))
    diffs = [
        float(np.abs(frames[i] - frames[i - 1]).mean())
        for i in range(1, len(frames))
    ]
    return float(np.median(diffs)) / max(contrast, 1.0)


def is_static(score: float) -> bool:
    """A negative score means measurement failed - do not call that static."""
    return 0.0 <= score < STATIC_THRESHOLD


def default_workers() -> int:
    return max(1, (os.cpu_count() or 2) // 2)


# How much of each song a measurement encodes. Long enough to cover a cut or
# two and average over them, short enough that the answer arrives while you
# are still asking the question.
SAMPLE_SECONDS = 20.0


def encode_many(
    jobs: list[tuple],
    settings: EncodeSettings,
    workers: int | None = None,
    on_done=None,
    keep_source: bool = False,
) -> dict[Path, tuple[bool, str]]:
    """
    Run many encodes concurrently.

    A job is (src, song_dir), or (src, song_dir, settings) when that one job
    needs its own. The third element replaces the call's settings entirely
    for that job and is how a measurement takes a different slice out of
    every song while still filling one pool: measuring serially would make
    the sample as slow as the encode it is meant to predict.

    `keep_source` is forwarded to every job, which is what lets the preview
    pass run here rather than in a serial loop of its own: a preview is the
    same encode at a lower resolution that must not delete the source.
    """
    workers = workers or default_workers()
    results: dict[Path, tuple[bool, str]] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(encode_one, job[0], job[1],
                        job[2] if len(job) > 2 else settings,
                        keep_source): job[1]
            for job in jobs
        }
        for fut in as_completed(futures):
            song_dir = futures[fut]
            try:
                results[song_dir] = fut.result()
            except Exception as exc:  # noqa: BLE001
                results[song_dir] = (False, str(exc)[:300])
            if on_done:
                on_done(song_dir, results[song_dir])

    return results


def measure_rate(rows, settings: EncodeSettings,
                 workers: int | None = None) -> float:
    """
    Bits per second this codec actually produces, from a sample encode.

    The sample is a slice, not the song: `SAMPLE_SECONDS` taken from the
    middle of each, where the footage is representative and the titles and
    end cards are not. That is the whole difference between a measurement
    you wait for and one you ask for - the figure wanted is per second, so
    encoding whole songs to find it means paying the run to predict the run.
    Every slice goes into ONE pool call carrying its own `clip`, and the
    bytes are divided by the seconds actually encoded.

    The sample is encoded into a temporary folder, one subfolder per song, and
    the sources are kept. Nothing may be written into a song folder: this runs
    to answer a question before the run starts, and an estimate that left
    half-quality files in the library would be worse than no estimate.
    """
    rows = list(rows)
    # mkdtemp rather than TemporaryDirectory: its cleanup raises, and a
    # scanner still holding a file it has just seen written would turn a
    # finished measurement into a traceback after the work was done.
    # `ignore_cleanup_errors` would say this, but it is Python 3.10 and this
    # package declares 3.9.
    tmp = tempfile.mkdtemp(prefix="yargvid-estimate-")
    try:
        root = Path(tmp)
        jobs, seconds = [], {}
        for i, r in enumerate(rows):
            src = r["source_path"]
            length = float(r["video_seconds"] or 0.0)
            # A song of unknown length cannot contribute: its bytes would be
            # divided by seconds nobody measured, which inflates the rate.
            if not src or length <= 0:
                continue
            take = min(SAMPLE_SECONDS, length)
            # The numbered parent keeps two identically-named songs apart and
            # the leaf is the song's own name, so a crash leaves a folder
            # somebody can recognize.
            out = root / f"{i:03d}" / Path(r["song_dir"]).name
            out.mkdir(parents=True, exist_ok=True)
            jobs.append((Path(src), out,
                         replace(settings, clip=((length - take) / 2, take))))
            seconds[out] = take

        if not jobs:
            return 0.0

        results = encode_many(jobs, settings, workers, keep_source=True)

        total_bytes = total_seconds = 0.0
        for out, (ok, _err) in results.items():
            written = output_path(out, settings)
            if not ok or not written.exists():
                continue
            total_bytes += written.stat().st_size
            total_seconds += seconds.get(out, 0.0)

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if total_seconds <= 0:
        return 0.0
    return 8.0 * total_bytes / total_seconds


def _lists_encoder(listing: str, name: str) -> bool:
    """
    Is `name` the encoder column of a line of `ffmpeg -encoders`?

    A substring test is not the same question: `h264_amf` contains `h264`,
    and every build lists the decoder capabilities line `V..... = Video`.
    """
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[1] == name:
            return True
    return False


def check_encoder(name: str) -> tuple[bool, str]:
    """
    Is this encoder both compiled in and actually usable?

    Two questions, and the second is the one that matters. `ffmpeg -encoders`
    lists what the build supports, which on a machine with no NVIDIA card
    still lists h264_nvenc; the driver only says no when something asks it to
    encode a frame. So ask it to encode a frame.
    """
    try:
        listing = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-encoders"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        ).stdout or ""
    except Exception as exc:  # noqa: BLE001
        return False, f"could not run ffmpeg: {exc}"

    if not _lists_encoder(listing, name):
        return False, f"this ffmpeg build has no {name} encoder"

    try:
        proc = subprocess.run(
            [
                "ffmpeg", "-v", "error", "-nostdin",
                # 320x240, not something tiny: NVENC refuses anything under
                # 145 pixels wide with "Frame Dimension less than the minimum
                # supported value", which would fail this check on a machine
                # whose encoder is perfectly good.
                "-f", "lavfi", "-i", "color=c=black:s=320x240:d=0.1",
                "-frames:v", "1",
                "-c:v", name,
                "-f", "null", os.devnull,
            ],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120,
        )
    except Exception as exc:  # noqa: BLE001
        return False, f"{name} could not be run: {exc}"

    if proc.returncode != 0:
        return False, _first_error(proc.stderr)
    return True, "ok"


def _first_error(stderr: str) -> str:
    """
    The line that says why, not the last line ffmpeg printed.

    One failed encoder produces a cascade - the filter graph, then the output,
    each reporting that the thing before it stopped - and the last line is
    always the least informative of them ("Nothing was written into output
    file"). The cause is the first line.
    """
    for line in (stderr or "").splitlines():
        line = line.strip()
        if line:
            return line[:200]
    return "the encoder refused a test frame"


def resolve_codec(settings: EncodeSettings, say=print) -> EncodeSettings:
    """
    Settle on a codec once, before the run starts.

    A hardware encoder that is not there must not be discovered a thousand
    times, and it must not be discovered silently: a run that quietly took the
    software path is a run whose timings mean something other than what they
    appear to.
    """
    row = codec_of(settings)
    if not row.hardware:
        return settings
    ok, why = check_encoder(row.encoder)
    if ok:
        return settings
    say(f"{settings.codec} is not usable here ({why}) - "
        f"falling back to {row.fallback}")
    return replace(settings, codec=row.fallback)


def check_ffmpeg_vp8() -> bool:
    try:
        out = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-encoders"],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60,
        ).stdout
        return "libvpx" in out
    except Exception:
        return False


def have(tool: str) -> bool:
    return shutil.which(tool) is not None
