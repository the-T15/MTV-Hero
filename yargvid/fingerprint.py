"""
Landmark (Shazam-style) audio fingerprinting.

This is the load-bearing component of the pipeline. It is used twice:

  1. MATCHING  - decide whether a candidate YouTube video actually contains
                 the same recording as the chart audio. Wrong videos produce a
                 flat offset histogram and are rejected outright.
  2. COARSE SYNC - the peak of that same histogram *is* the time offset.

Why not onset-envelope cross-correlation (the previous approach)?
Cross-correlation always returns an answer. Its "confidence" is a normalised
peak height, which stays high for two unrelated rock songs at similar tempo.
Landmark matching, by contrast, has a natural null hypothesis: unrelated audio
produces a uniform spread of time deltas. The signal-to-noise of the histogram
peak is therefore a real statistic, not a heuristic, and it degrades gracefully
on live versions, covers and remasters instead of silently returning garbage.

No external fingerprint dependency (no chromaprint/fpcalc binary) - numpy and
scipy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


# --- analysis parameters -----------------------------------------------------
# Resolution of the coarse offset is HOP / SR seconds. 256 / 22050 = 11.6 ms,
# which is comfortably finer than the +/-100 ms search window handed to the
# GCC-PHAT refinement stage in sync.py.
SR = 22050
NFFT = 1024
HOP = 256

# Peak picking. A peak must be the maximum in a (freq x time) neighbourhood and
# must stand above the local noise floor.
PEAK_FREQ_NEIGH = 11
PEAK_TIME_NEIGH = 15
PEAK_MIN_DB_OVER_FLOOR = 12.0

# Target zone for pairing. Each anchor peak is paired with peaks that follow it
# within this time span and frequency band.
FAN_OUT = 8
DT_MIN_FRAMES = 2
DT_MAX_FRAMES = 96
DF_MAX_BINS = 96

# Hash quantisation. Coarser frequency quantisation buys robustness to pitch
# drift and codec damage at the cost of more spurious matches.
FREQ_QUANT = 2
DT_QUANT = 1


@dataclass
class MatchResult:
    """Outcome of matching two fingerprints."""

    offset_seconds: float
    """t_b - t_a for the same musical moment. Positive means the moment occurs
    LATER in b than in a."""

    score: float
    """Histogram peak height in units of the background standard deviation.
    This is the accept/reject statistic. See ACCEPT_SCORE below."""

    peak_count: int
    """Raw number of hash pairs agreeing on the winning offset."""

    coverage: float
    """Fraction of a's duration spanned by the agreeing pairs. Guards against a
    strong match on a single repeated loop or a sampled intro."""

    total_pairs: int
    """Hash pairs matched at any offset. Very low values mean one of the inputs
    was silent, corrupt, or far too short."""


# A candidate is accepted as "the same recording" above this score.
#
# Calibrated against a 175-song real-library sample, not synthetic data - the
# synthetic degradation used in testing is far gentler than reality and put
# true matches at 700+, which is useless for setting this.
#
# At 60 the gate rejected official uploads on artists' own channels; the
# recovered cases were nearly all REMASTERS (Ramble On, Today 2011, I Wanna
# Rock 2024). Charts use original masters while official channels host
# remasters, so the same performance scores lower through different
# compression and EQ. That, not recording age or bandwidth, is what the
# strict gate was filtering out.
#
# 45 sits above observed wrong-song scores (7-15 for genuine mismatches) and
# below the remaster band (49-63). Override per run with `match --gate N`.
ACCEPT_SCORE = 45.0
ACCEPT_COVERAGE = 0.25
MIN_TOTAL_PAIRS = 200


def spectrogram_db(samples: np.ndarray) -> np.ndarray:
    if samples.size < NFFT:
        return np.zeros((NFFT // 2 + 1, 0), dtype=np.float32)

    window = np.hanning(NFFT).astype(np.float32)
    n_frames = 1 + (samples.size - NFFT) // HOP
    # Strided view avoids materialising a copy of every frame.
    frames = np.lib.stride_tricks.as_strided(
        samples,
        shape=(n_frames, NFFT),
        strides=(samples.strides[0] * HOP, samples.strides[0]),
        writeable=False,
    )
    spec = np.fft.rfft(frames * window, axis=1)
    mag = np.abs(spec).T.astype(np.float32)
    return 20.0 * np.log10(mag + 1e-10)


def find_peaks(spec_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if spec_db.shape[1] == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    local_max = ndimage.maximum_filter(
        spec_db, size=(PEAK_FREQ_NEIGH, PEAK_TIME_NEIGH), mode="nearest"
    )
    # Local noise floor: a heavily smoothed version of the same surface.
    floor = ndimage.uniform_filter(spec_db, size=(31, 61), mode="nearest")

    mask = (spec_db == local_max) & (spec_db > floor + PEAK_MIN_DB_OVER_FLOOR)
    freqs, frames = np.nonzero(mask)

    order = np.argsort(frames, kind="stable")
    return freqs[order].astype(np.int32), frames[order].astype(np.int32)


def make_hashes(samples: np.ndarray) -> dict[np.int64, list[int]]:
    """Fingerprint audio into {hash -> [anchor frame, ...]}."""
    freqs, frames = find_peaks(spectrogram_db(samples))
    table: dict[np.int64, list[int]] = {}
    n = freqs.size
    if n == 0:
        return table

    for i in range(n):
        f1, t1 = int(freqs[i]), int(frames[i])
        paired = 0
        j = i + 1
        while j < n and paired < FAN_OUT:
            dt = int(frames[j]) - t1
            if dt < DT_MIN_FRAMES:
                j += 1
                continue
            if dt > DT_MAX_FRAMES:
                break
            f2 = int(freqs[j])
            if abs(f2 - f1) <= DF_MAX_BINS:
                key = np.int64(
                    ((f1 // FREQ_QUANT) << 24)
                    | ((f2 // FREQ_QUANT) << 12)
                    | (dt // DT_QUANT)
                )
                table.setdefault(key, []).append(t1)
                paired += 1
            j += 1
    return table


def match_candidates(
    hashes_a: dict[np.int64, list[int]],
    hashes_b: dict[np.int64, list[int]],
    duration_a: float,
    top_k: int = 4,
    min_separation: int = 12,
) -> list[MatchResult]:
    """
    The strongest few alignments, best first.

    Taking only the single largest peak is unsafe on repetitive music: a false
    alignment at some multiple of a repeated riff can out-vote the true one,
    and it looks completely confident because it IS a real peak - just the
    wrong one. Offering several lets the caller check which one actually holds
    up across the track, which is a question the histogram cannot answer.

    Peaks must be `min_separation` bins apart so the same peak split across
    neighbouring bins is not returned twice.
    """
    deltas: list[int] = []
    anchors: list[int] = []

    if len(hashes_a) <= len(hashes_b):
        for key, times_a in hashes_a.items():
            times_b = hashes_b.get(key)
            if not times_b:
                continue
            for ta in times_a:
                for tb in times_b:
                    deltas.append(tb - ta)
                    anchors.append(ta)
    else:
        for key, times_b in hashes_b.items():
            times_a = hashes_a.get(key)
            if not times_a:
                continue
            for tb in times_b:
                for ta in times_a:
                    deltas.append(tb - ta)
                    anchors.append(ta)

    total = len(deltas)
    if total < MIN_TOTAL_PAIRS:
        return [MatchResult(0.0, 0.0, 0, 0.0, total)]

    d = np.asarray(deltas, dtype=np.int64)
    a = np.asarray(anchors, dtype=np.int64)
    lo = int(d.min())
    counts = np.bincount(d - lo).astype(np.float64)

    # Sum neighbouring bins before ranking: a true alignment often straddles
    # two bins and would otherwise lose to a narrower false one.
    smoothed = np.convolve(counts, np.ones(3), mode="same")

    results: list[MatchResult] = []
    working = smoothed.copy()
    for _ in range(top_k):
        best_bin = int(np.argmax(working))
        if working[best_bin] <= 0:
            break
        # Smoothing is for RANKING only. Reading peak height and the offset at
        # a smoothed argmax can land a bin off the true maximum, which both
        # understates the score and shifts the offset by a frame - so snap
        # back to the actual peak in the immediate neighbourhood.
        lo_b, hi_b = max(0, best_bin - 1), min(counts.size, best_bin + 2)
        best_bin = lo_b + int(np.argmax(counts[lo_b:hi_b]))
        peak = int(counts[best_bin])
        best_delta = best_bin + lo

        bg = counts.copy()
        bg[max(0, best_bin - 2): best_bin + 3] = np.nan
        bg_std = float(np.nanstd(bg))
        bg_mean = float(np.nanmean(bg))
        score = 0.0 if bg_std <= 1e-9 else (peak - bg_mean) / bg_std

        agree = a[np.abs(d - best_delta) <= 1]
        if agree.size and duration_a > 0:
            span = (agree.max() - agree.min()) * HOP / SR
            coverage = min(1.0, span / duration_a)
        else:
            coverage = 0.0

        results.append(MatchResult(
            offset_seconds=best_delta * HOP / SR,
            score=score, peak_count=peak, coverage=coverage, total_pairs=total,
        ))
        working[max(0, best_bin - min_separation):
                best_bin + min_separation + 1] = 0

    return results or [MatchResult(0.0, 0.0, 0, 0.0, total)]


def match(
    hashes_a: dict[np.int64, list[int]],
    hashes_b: dict[np.int64, list[int]],
    duration_a: float,
) -> MatchResult:
    """
    Best single alignment.

    Returns an offset in the sense `t_b - t_a`: positive means a given musical
    moment happens LATER in b than in a. With a = chart audio and b = video
    audio, a positive offset means the video must be seeked forward, which is
    exactly YARG's positive `video_start_time`.
    """
    return match_candidates(hashes_a, hashes_b, duration_a, top_k=1)[0]


def is_same_recording(result: MatchResult) -> bool:
    """Accept/reject gate used by the matching stage."""
    return (
        result.total_pairs >= MIN_TOTAL_PAIRS
        and result.score >= ACCEPT_SCORE
        and result.coverage >= ACCEPT_COVERAGE
    )
