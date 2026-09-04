"""Audio decode and stem mixing via ffmpeg."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from .fingerprint import SR

AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".flac", ".m4a"}
VIDEO_EXTS = {".webm", ".mp4", ".mkv", ".mov", ".avi"}

# Stems that are not part of the song as the player hears it.
EXCLUDE_STEMS = ("preview", "crowd", "ambient")


def find_stems(song_dir: Path) -> list[Path]:
    """
    All audio stems in a song folder, excluding preview/crowd/ambient.

    Returns empty for a folder that has been moved or deleted rather than
    raising: song_dir is a database key, and the library on disk changes
    underneath it. Callers already treat "no stems" as a handled condition.
    """
    try:
        entries = sorted(song_dir.iterdir())
    except OSError:
        return []
    return [
        p
        for p in entries
        if p.is_file()
        and p.suffix.lower() in AUDIO_EXTS
        and not any(tag in p.stem.lower() for tag in EXCLUDE_STEMS)
    ]


def find_video(song_dir: Path) -> Path | None:
    """Existing background video, if any. `video.*` wins over other names."""
    vids = [
        p
        for p in sorted(song_dir.iterdir())
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    ]
    if not vids:
        return None
    for v in vids:
        if v.stem.lower() == "video":
            return v
    return vids[0]


def probe(path: Path) -> dict:
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path),
            ],
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=120, check=True,
        ).stdout
        return json.loads(out)
    except Exception:
        return {}


def duration_of(path: Path) -> float:
    info = probe(path)
    try:
        return float(info["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return 0.0


def decode_mono(path: Path, sr: int = SR, max_seconds: float | None = None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if max_seconds:
        cmd += ["-t", str(max_seconds)]
    cmd += [
        "-i", str(path),
        "-vn", "-map", "a:0?",
        "-ac", "1", "-ar", str(sr),
        "-f", "f32le", "-",
    ]
    try:
        raw = subprocess.run(
            cmd, capture_output=True, timeout=1800, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return np.zeros(0, dtype=np.float32)
    x = np.frombuffer(raw, dtype=np.float32).copy()
    return np.nan_to_num(x, copy=False)


def leading_silence(
    samples: np.ndarray, sr: int = SR, threshold_db: float = -45.0
) -> float:
    """
    Seconds of near-silence before audio actually begins.

    Used to explain an offset rather than guess at it. A large negative
    video_start_time means the chart reaches a musical moment later than the
    video does, which is either chart lead-in or a video edit that trims the
    song's intro. Measuring the lead-in on both sides tells you which, without
    having to listen to anything.
    """
    if samples.size == 0:
        return 0.0
    peak = float(np.abs(samples).max())
    if peak <= 0:
        return float(samples.size) / sr

    win = max(1, sr // 100)                      # 10 ms windows
    n = samples.size // win
    if n == 0:
        return 0.0
    frames = samples[: n * win].reshape(n, win)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1))
    thresh = peak * (10.0 ** (threshold_db / 20.0))

    loud = np.nonzero(rms > thresh)[0]
    return 0.0 if loud.size == 0 else float(loud[0] * win) / sr


def mix_stems(stems: list[Path], sr: int = SR) -> np.ndarray:
    """
    Sum all stems into one mono signal.

    ffmpeg's amix normalises by input count, which quietly buries drums when a
    chart has many stems. Summing manually and peak-normalising once at the end
    preserves the transient structure the fingerprint depends on.
    """
    tracks = [decode_mono(p, sr) for p in stems]
    tracks = [t for t in tracks if t.size]
    if not tracks:
        return np.zeros(0, dtype=np.float32)

    n = max(t.size for t in tracks)
    acc = np.zeros(n, dtype=np.float32)
    for t in tracks:
        acc[: t.size] += t

    peak = float(np.abs(acc).max())
    if peak > 0:
        acc /= peak
    return acc
