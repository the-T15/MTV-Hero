"""
VP8/WebM encoding.

Linux and Steam Deck YARG only reliably play VP8 in a WebM container, so the
transcode is unavoidable for a cross-platform library. Three changes from the
previous approach address the "too slow" problem:

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
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio as au


@dataclass
class EncodeSettings:
    height: int = 1080          # 720 roughly halves encode time
    crf: int = 31               # VP8 CQ: 4-63, lower is better quality
    bitrate_cap: str = "4M"     # ceiling for CQ mode - NOT 0 for VP8
    cpu_used: int = 3           # 0-5 with `-deadline good`; higher = faster
    threads_per_job: int = 2
    drop_audio: bool = True
    max_fps: float = 30.0       # cap only; source rate preserved below this


def build_command(
    src: Path, dst: Path, settings: EncodeSettings, source_fps: float | None = None
) -> list[str]:
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

    fps = source_fps or settings.max_fps
    fps = min(fps, settings.max_fps)
    if fps <= 0:
        fps = 30.0
    vf += f",fps={fps:.6f}"

    cmd = [
        "ffmpeg", "-y", "-v", "error", "-nostdin",
        "-i", str(src),
        "-c:v", "libvpx",
        "-crf", str(settings.crf),
        "-b:v", settings.bitrate_cap,
        "-qmin", "4", "-qmax", "56",
        "-deadline", "good",
        "-cpu-used", str(settings.cpu_used),
        "-auto-alt-ref", "0",
        "-threads", str(settings.threads_per_job),
        "-pix_fmt", "yuv420p",
        "-vf", vf,
    ]
    # -vsync was deprecated in ffmpeg 5.x in favour of -fps_mode and warns
    # loudly on modern builds. Fall back only for genuinely old ffmpeg.
    cmd += ["-fps_mode", "cfr"] if _supports_fps_mode() else ["-vsync", "cfr"]
    cmd += ["-an"] if settings.drop_audio else ["-c:a", "libvorbis", "-q:a", "4"]
    cmd += ["-f", "webm", str(dst)]
    return cmd


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


def encode_one(
    src: Path,
    song_dir: Path,
    settings: EncodeSettings,
    keep_source: bool = False,
) -> tuple[bool, str]:
    """
    Encode one video into `song_dir/video.webm`.

    Previews are not a separate mode here. A preview is just this function
    called with a low `height` and `keep_source=True`: full length, correct
    timing, real filename, so it can actually be played in YARG. Truncating to
    N seconds would break any song whose video_start_time is positive - YARG
    seeks into the file, so a 30-second clip of a video seeked to 22s leaves
    almost nothing to watch.
    """
    dst = song_dir / "video.webm"
    tmp = song_dir / "video.webm.part"

    cmd = build_command(src, tmp, settings, source_frame_rate(src))

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=7200,
        )
    except subprocess.TimeoutExpired:
        tmp.unlink(missing_ok=True)
        return False, "encode timed out"

    if proc.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        tmp.unlink(missing_ok=True)
        return False, (proc.stderr or "ffmpeg failed").strip()[:300]

    err = _replace_with_retry(tmp, dst)
    if err:
        return False, err

    # Critical: a stray source file in the folder can shadow the webm.
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


def encode_many(
    jobs: list[tuple[Path, Path]],
    settings: EncodeSettings,
    workers: int | None = None,
    on_done=None,
    keep_source: bool = False,
) -> dict[Path, tuple[bool, str]]:
    """
    Run many encodes concurrently. `jobs` is a list of (src, song_dir).

    `keep_source` is forwarded to every job, which is what lets the preview
    pass run here rather than in a serial loop of its own: a preview is the
    same encode at a lower resolution that must not delete the source.
    """
    workers = workers or default_workers()
    results: dict[Path, tuple[bool, str]] = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(encode_one, src, d, settings, keep_source): d
            for src, d in jobs
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
