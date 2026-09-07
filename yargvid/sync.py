"""
Offset estimation with a hard accept/reject gate.

Three stages:

  COARSE   Landmark fingerprint over the whole track (fingerprint.py). Robust,
           ~12 ms resolution, and self-validating.
  REFINE   GCC-PHAT on the raw waveform inside a +/-150 ms window around the
           coarse estimate, computed independently on several excerpts spread
           across the track. Sub-millisecond.
  VERIFY   The excerpts must agree. This is the part the previous pipeline
           lacked, and it is what turns "an offset" into "an offset we trust".

The verification step does double duty: scattered excerpt offsets mean either
linear drift (a fittable trend) or a wrong/edited video (no trend). Both are
detected here rather than surviving into the encode.

SIGN CONVENTION (YARG wiki): positive `video_start_time` seeks the video
forward to where the song audio starts; negative delays the video. Our offset
is t_video - t_chart, so `video_start_time = round(offset_ms)` directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fingerprint as fp

REFINE_SR = 44100
EXCERPT_SECONDS = 20.0
N_EXCERPTS = 7
N_CANDIDATES = 4   # alternative alignments to test before giving up
SEARCH_MS = 150.0

# Verification thresholds.
AGREE_MS = 35.0          # excerpt spread below this = clean, no drift
                         # (well inside the ~45 ms A/V perception limit)
DRIFT_R2 = 0.90          # linear fit quality required to call it drift
MAX_DRIFT_PPM = 60000    # 6% - beyond this the video is simply wrong
MIN_GOOD_EXCERPTS = 4

# A weaker alignment may only displace the strongest one if it is supported by
# at least this fraction of its hashes. Measured over the 281-song export: 270
# songs chose candidate 1, 11 fell through to a weaker one, and 8 of those went
# to a candidate with under half the support (five under a quarter, three at
# 1%). Every large wrong move the recheck made was of that shape - candidate 1
# locked on all seven windows but they disagreed by 41-96 ms, so a short rigid
# section carrying a tenth of the hashes won on spread alone.
SUPPORT_RATIO = 0.5


@dataclass
class SyncResult:
    status: str                       # ok | unverified | drift | rejected
    offset_ms: float = 0.0
    spread_ms: float = 0.0
    drift_ppm: float = 0.0            # parts per million; +ve = video runs slow
    r2: float = 0.0
    fp_score: float = 0.0
    sharpness: float = 0.0      # correlation strength at the chosen offset
    dominance: float = 0.0      # top peak's hashes / runner-up's
    windows: int = 0            # how many probe windows actually correlated
    windows_total: int = 0
    coverage: float = 0.0
    excerpts: list[tuple[float, float]] = field(default_factory=list)
    reason: str = ""

    @property
    def video_start_time(self) -> int:
        return int(round(self.offset_ms))


def gcc_phat(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[float, float]:
    """
    Generalised cross-correlation with phase transform.

    Whitening the cross-spectrum makes the correlation peak a sharp spike
    rather than a broad hump, which is what gives sub-sample precision. Returns
    (lag_in_samples, sharpness) where lag is `b` relative to `a`.
    """
    n = 1 << int(np.ceil(np.log2(a.size + b.size)))
    A = np.fft.rfft(a, n)
    B = np.fft.rfft(b, n)
    R = A * np.conj(B)
    R /= np.abs(R) + 1e-12
    cc = np.fft.irfft(R, n)
    cc = np.concatenate([cc[-max_lag:], cc[: max_lag + 1]])

    idx = int(np.argmax(np.abs(cc)))
    peak = float(np.abs(cc[idx]))

    # Parabolic interpolation for sub-sample resolution.
    frac = 0.0
    if 0 < idx < cc.size - 1:
        y0, y1, y2 = np.abs(cc[idx - 1]), np.abs(cc[idx]), np.abs(cc[idx + 1])
        denom = y0 - 2 * y1 + y2
        if abs(denom) > 1e-12:
            frac = 0.5 * (y0 - y2) / denom

    lag = (idx + frac) - max_lag
    sharpness = 0.0 if np.median(np.abs(cc)) <= 0 else peak / float(np.median(np.abs(cc)))
    return -lag, sharpness


def _window_range(chart_n: int, video_n: int, shift: int) -> tuple[int, int]:
    """
    Chart sample range where a window has video underneath it.

    A window at chart index s reads video at s + shift, so it is valid only
    while both indices are in bounds. That gives [max(0, -shift), min(chart_n,
    video_n - shift)].

    The lower bound is the part that used to be missing. Windows were laid out
    from 0 regardless of sign, so for a negative shift the early ones mapped to
    a negative video index and were dropped by the bounds check inside the
    loop - and because the upper bound ignored the shift too, the last |shift|
    seconds of alignable material never got a window at all. On Boney M. -
    Rasputin (chart 357.5s, video 283.4s, offset -65.3s) verification covered
    87.8-283.4s of a valid 65.3-348.7s, so 88 seconds including the whole tail
    went unchecked. Most of this library sits at a negative offset.

    For a positive shift this returns exactly what the old expression computed,
    so positive-offset songs are unaffected.
    """
    lo = max(0, -shift)
    hi = min(chart_n, video_n - shift)
    return lo, hi


# Identity was already decided at match time, on this exact video. Re-running
# the full gate here is a second coin flip on a different audio stream: a
# marginal song can pass match at 60 and fail sync at 55.8 on the same video.
# Sync keeps only a floor to catch a genuinely wrong or corrupt file.
# Bracketed by observation, not guesswork. Unrelated audio scores 14-17 in
# testing, while videos that had already passed the match gate scored as low
# as 46.9 at sync time on the merged stream. 40 sits between the two with
# margin on both sides.
IDENTITY_FLOOR = 40.0


BLOCK_WINDOWS = 20
BLOCK_SECONDS = 8.0


def analyse_blocks(
    chart_hi: np.ndarray,
    video_hi: np.ndarray,
    offsets_ms: list[float],
) -> list[dict]:
    """
    Split the song into stretches that each align at a different offset.

    A video with a skit, an extended solo or any internal cut is not one
    timeline against the chart - it is several, each correct on its own and
    separated by however long the inserted material runs. Measuring one offset
    for the whole track reports that as "excerpts disagree", which is true but
    useless.

    Rather than assuming a single offset and looking for a residual, every
    window is tested against every candidate offset and keeps whichever
    correlates most strongly. Windows preferring the same offset form a block.
    Returns blocks in song order with the fraction of the song each covers.
    """
    if chart_hi.size == 0 or video_hi.size == 0 or not offsets_ms:
        return []

    win = int(BLOCK_SECONDS * REFINE_SR)
    max_lag = int(SEARCH_MS / 1000.0 * REFINE_SR)
    total = chart_hi.size
    if total < win * 2:
        return []

    starts = np.linspace(0, total - win, BLOCK_WINDOWS).astype(int)
    picks: list[tuple[float, float, float]] = []      # frac, offset, sharpness

    for s in starts:
        a = chart_hi[s: s + win]
        if float(np.abs(a).max()) < 1e-4:
            continue
        best = (0.0, None)
        for off in offsets_ms:
            vs = s + int(round(off / 1000.0 * REFINE_SR))
            if vs < 0 or vs + win > video_hi.size:
                continue
            b = video_hi[vs: vs + win]
            if a.size != b.size:
                continue
            _, sharp = gcc_phat(a, b, max_lag)
            if sharp > best[0]:
                best = (sharp, off)
        if best[1] is not None and best[0] >= 4.0:
            picks.append((s / total, best[1], best[0]))

    if not picks:
        return []

    # Merge neighbouring windows that chose the same offset.
    blocks: list[dict] = []
    for frac, off, sharp in picks:
        if blocks and blocks[-1]["offset_ms"] == off:
            blocks[-1]["end"] = frac
            blocks[-1]["windows"] += 1
            blocks[-1]["sharp"] = max(blocks[-1]["sharp"], sharp)
        else:
            blocks.append({"start": frac, "end": frac, "offset_ms": off,
                           "windows": 1, "sharp": sharp})

    step = 1.0 / max(1, BLOCK_WINDOWS - 1)
    for b in blocks:
        b["end"] = min(1.0, b["end"] + step)
        b["covers"] = b["end"] - b["start"]
    return blocks


def probe_offset(
    chart_hi: np.ndarray, video_hi: np.ndarray, offset_ms: float
) -> dict:
    """
    Measure how strongly the audio actually correlates at one given offset.

    Diagnostic only. `_verify` asks whether windows AGREE WITH EACH OTHER,
    which a riff-only alignment can satisfy: every window finds the same
    residual and consistency confirms a wrong answer. This reports the
    correlation STRENGTH instead - how sharp the peak is - which should be
    higher where everything lines up than where only the riff does.

    Nothing is filtered out, so weak windows are visible rather than dropped.
    """
    win = int(EXCERPT_SECONDS * REFINE_SR)
    max_lag = int(SEARCH_MS / 1000.0 * REFINE_SR)
    shift = int(round(offset_ms / 1000.0 * REFINE_SR))

    lo, hi = _window_range(chart_hi.size, video_hi.size, shift)
    usable = hi - lo
    if usable < win:
        return {"windows": 0, "usable_s": max(0.0, usable / REFINE_SR)}

    starts = np.linspace(lo, max(lo, hi - win), N_EXCERPTS).astype(int)
    sharps: list[float] = []
    offs: list[float] = []
    for s in starts:
        vs = s + shift
        if vs < 0 or vs + win > video_hi.size:
            continue
        a, b = chart_hi[s: s + win], video_hi[vs: vs + win]
        if a.size != b.size or float(np.abs(a).max()) < 1e-4:
            continue
        lag, sharp = gcc_phat(a, b, max_lag)
        sharps.append(sharp)
        offs.append(offset_ms + (lag / REFINE_SR) * 1000.0)

    if not sharps:
        return {"windows": 0, "usable_s": usable / REFINE_SR}
    return {
        "windows": len(sharps),
        "usable_s": usable / REFINE_SR,
        "sharp_median": float(np.median(sharps)),
        "sharp_min": float(np.min(sharps)),
        "sharp_max": float(np.max(sharps)),
        "strong": sum(1 for s in sharps if s >= 4.0),
        "refined_ms": float(np.median(offs)),
        "spread_ms": float(np.max(offs) - np.min(offs)),
    }


def estimate(
    chart: np.ndarray,
    video: np.ndarray,
    chart_hi: np.ndarray | None = None,
    video_hi: np.ndarray | None = None,
    trust_identity: bool = False,
    static_background: bool = False,
    manual: bool = False,
) -> SyncResult:
    """
    `chart` / `video` are mono at fingerprint.SR (coarse stage).
    `chart_hi` / `video_hi` are the same audio at REFINE_SR (refine stage).
    If the high-rate versions are omitted, the coarse result is returned as-is.
    """
    if chart.size == 0 or video.size == 0:
        return SyncResult("rejected", reason="empty audio")

    dur = chart.size / fp.SR
    ha, hb = fp.make_hashes(chart), fp.make_hashes(video)

    # Several candidate alignments, best-scoring first. Only the strongest is
    # checked for identity - the rest are alternative placements of the SAME
    # recording, and the whole point is that peak strength is not what decides
    # which one is right.
    cands = fp.match_candidates(ha, hb, dur, top_k=N_CANDIDATES)
    m = cands[0]

    # How far the winning alignment outvotes the next one. A confident match
    # buries the alternatives (20x-86x on songs confirmed by eye); a flat
    # histogram means there is no real alignment to find and the chosen offset
    # is the tallest blade of grass in an empty field (1.0x-3.3x on songs
    # confirmed wrong). Reported, never used to pick - when the histogram is
    # flat there is no better candidate to switch to.
    dominance = (m.peak_count / max(cands[1].peak_count, 1)
                 if len(cands) > 1 else 999.0)

    # A hand-picked video is a decision, not a guess. The gate exists to catch
    # bad automatic matches; refusing a video the user deliberately chose just
    # discards their work. Compute the offset and warn instead.
    if manual:
        floor_ok = True
    elif trust_identity:
        floor_ok = m.score >= IDENTITY_FLOOR and m.coverage >= fp.ACCEPT_COVERAGE
    else:
        floor_ok = fp.is_same_recording(m)
    if not floor_ok:
        return SyncResult(
            "rejected",
            fp_score=m.score,
            coverage=m.coverage,
            reason=(
                f"fingerprint below gate (score {m.score:.1f} < {fp.ACCEPT_SCORE}, "
                f"coverage {m.coverage:.2f})"
            ),
        )

    weak = manual and not fp.is_same_recording(m)

    if chart_hi is not None and video_hi is not None:
        # Try each candidate and keep the first that holds up across the
        # track. A false lock onto a repeated riff fails here; the true offset
        # does not. Verification, not peak height, picks the winner.
        # Take the strongest peak, and only look further if it fails to
        # verify at all.
        #
        # Selecting on correlation strength instead was tried and was much
        # worse: across 128 songs it changed 50 offsets, and the margins it
        # decided on were noise (111 vs 111, 91 vs 90, 29 vs 28). Offsets that
        # had clustered sensibly around a few seconds of chart lead-in
        # scattered to +/-80s, with videos far too short to cover the song.
        # Hash count is right for the large majority; where it is not, the
        # answer is a manual offset, not a different automatic rule.
        #
        # `m` is the candidate the identity gate above was applied to, and
        # every candidate is an alternative placement of that same recording.
        # `_verify` fills fp_score in from whichever candidate it happened to
        # be handed, so the stored score belonged to the alignment that won
        # verification rather than the one that proved identity - 19 songs in
        # the library sit below the gate they were accepted at because of it.
        fallback: SyncResult | None = None
        support_floor = SUPPORT_RATIO * cands[0].peak_count
        for cand in cands:
            if cand is not cands[0] and cand.peak_count < support_floor:
                # Not enough of the song agrees with this placement for it to
                # overrule the strongest one, whatever its spread comes out at.
                continue
            got = _verify(cand, chart_hi, video_hi, static_background)
            got.dominance = dominance
            got.fp_score = m.score
            if got.status in ("ok", "drift"):
                if weak:
                    got.reason = (
                        f"your pick - audio only weakly matches this chart "
                        f"(score {m.score:.0f}); check the timing"
                    ) + (f" - {got.reason}" if got.reason else "")
                return got
            if fallback is None:
                fallback = got
        if fallback is not None:
            fallback.dominance = dominance
            fallback.fp_score = m.score
            # Nothing qualifying verified, and candidate 1 failed only because
            # its windows disagreed - which is exactly the case where a wrong
            # answer used to be preferred to it. It is still the best guess in
            # the file, so it goes to the review queue rather than being
            # thrown away.
            if (fallback.status == "rejected"
                    and fallback.windows >= MIN_GOOD_EXCERPTS):
                fallback.status = "unverified"
                detail = f"{fallback.reason} - " if fallback.reason else ""
                fallback.reason = (
                    f"strongest alignment kept as a guess - {detail}check it"
                )
            return fallback

    return SyncResult(
        "unverified", m.offset_seconds * 1000.0, -1.0,
        fp_score=m.score, coverage=m.coverage,
        reason="coarse offset only - no refinement audio supplied",
    )


def _verify(
    m: fp.MatchResult,
    chart_hi: np.ndarray,
    video_hi: np.ndarray,
    static_background: bool,
) -> SyncResult:
    """Refine and check one candidate offset against windows across the track."""
    coarse_ms = m.offset_seconds * 1000.0
    # --- refine on excerpts ---------------------------------------------------
    win = int(EXCERPT_SECONDS * REFINE_SR)
    max_lag = int(SEARCH_MS / 1000.0 * REFINE_SR)
    shift = int(round(m.offset_seconds * REFINE_SR))

    lo, hi = _window_range(chart_hi.size, video_hi.size, shift)
    usable = hi - lo
    if usable < win * 2:
        # Same trap as below: too little overlap to fit even two windows, so
        # nothing is verified. A short video combined with a large shift lands
        # here, and reporting 'ok' with spread 0 made a completely unchecked
        # guess look like the most confident result in the library.
        return SyncResult(
            "unverified", coarse_ms, -1.0,
            fp_score=m.score, coverage=m.coverage,
            reason=(f"only {usable / REFINE_SR:.0f}s of overlap - too little "
                    f"to check this offset anywhere"),
        )

    starts = np.linspace(lo, hi - win, N_EXCERPTS).astype(int)
    points: list[tuple[float, float]] = []
    sharps: list[float] = []

    for s in starts:
        vs = s + shift
        if vs < 0 or vs + win > video_hi.size:
            continue
        a = chart_hi[s : s + win]
        b = video_hi[vs : vs + win]
        if a.size != b.size or float(np.abs(a).max()) < 1e-4:
            continue
        lag, sharp = gcc_phat(a, b, max_lag)
        if sharp < 4.0:          # no distinct peak in this excerpt
            continue
        sharps.append(sharp)
        # Total offset = coarse shift + residual found inside the window.
        total_ms = coarse_ms + (lag / REFINE_SR) * 1000.0
        points.append((s / REFINE_SR, total_ms))

    if len(points) < MIN_GOOD_EXCERPTS:
        # Reporting spread 0.0 here was actively misleading: zero reads as
        # seven windows in perfect agreement when it actually means none of
        # them could confirm anything. The coarse offset may be a lock onto a
        # repeated section, and nothing here would catch it. Say so instead.
        return SyncResult(
            "unverified", coarse_ms, -1.0,
            fp_score=m.score, coverage=m.coverage, excerpts=points,
            reason=(f"only {len(points)} of {N_EXCERPTS} windows could confirm "
                    f"this offset - it is a guess, check it"),
        )

    strength = float(np.median(sharps)) if sharps else 0.0
    n_used, n_total = len(points), len(starts)
    # Every SyncResult below passes by keyword. These three branches used
    # positional arguments, and when `dominance` was added to the dataclass
    # between `sharpness` and `windows` they silently shifted by one: the
    # window count landed in dominance, coverage landed in windows_total, the
    # excerpt list landed in coverage, and excerpts was left empty. Nothing
    # raised, because every field from that point on is a number or a list.
    times = np.array([p[0] for p in points])
    offs = np.array([p[1] for p in points])
    spread = float(offs.max() - offs.min())

    # Case 1: excerpts agree. Clean sync, no drift.
    if spread <= AGREE_MS:
        return SyncResult(
            "ok", float(np.median(offs)), spread,
            fp_score=m.score, sharpness=strength, coverage=m.coverage,
            excerpts=points, windows=n_used, windows_total=n_total,
        )

    # Case 2: they disagree - is it a straight line?
    slope, intercept = np.polyfit(times, offs, 1)
    pred = slope * times + intercept
    ss_res = float(((offs - pred) ** 2).sum())
    ss_tot = float(((offs - offs.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    drift_ppm = slope * 1000.0   # ms drift per second -> ppm

    if r2 >= DRIFT_R2 and abs(drift_ppm) <= MAX_DRIFT_PPM:
        return SyncResult(
            "drift", float(intercept), spread, drift_ppm, r2,
            fp_score=m.score, sharpness=strength,
            windows=n_used, windows_total=n_total,
            coverage=m.coverage, excerpts=points,
            reason=f"linear drift {drift_ppm:.0f} ppm (R2={r2:.3f})",
        )

    # A still image looks the same whether it is 10 ms or 130 ms out, so
    # excerpt disagreement is not a defect for a static background - there is
    # nothing on screen for the audio to be out of step with. Rejecting these
    # applies a moving-picture test to album art.
    if static_background:
        return SyncResult(
            "ok", float(np.median(offs)), spread, drift_ppm, r2,
            fp_score=m.score, sharpness=strength,
            windows=n_used, windows_total=n_total,
            coverage=m.coverage, excerpts=points,
            reason=f"spread {spread:.0f} ms accepted - static background",
        )

    # Case 3: scattered. Edited video, different cut, or a bad match.
    return SyncResult(
        "rejected", float(np.median(offs)), spread, drift_ppm, r2,
        fp_score=m.score, sharpness=strength,
        windows=n_used, windows_total=n_total,
        coverage=m.coverage, excerpts=points,
        reason=f"excerpts disagree by {spread:.0f} ms with no linear trend (R2={r2:.2f})",
    )