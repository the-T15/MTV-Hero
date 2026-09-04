"""Synthetic validation of the fingerprint core."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yargvid import fingerprint as fp  # noqa: E402

rng = np.random.default_rng(1234)
SR = fp.SR


def fake_song(seconds: float, seed: int) -> np.ndarray:
    """A tonal + percussive signal with enough spectral structure to fingerprint."""
    r = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    x = np.zeros(n, dtype=np.float32)

    # Sustained harmonic content with chord changes every 2 s.
    for bar in range(int(seconds // 2)):
        root = r.uniform(110, 260)
        s, e = int(bar * 2 * SR), int((bar * 2 + 2) * SR)
        seg = t[s:e]
        for h in (1, 2, 3, 5, 7):
            x[s:e] += (0.35 / h) * np.sin(2 * np.pi * root * h * seg + r.uniform(0, 6))

    # Percussive transients twice a second: broadband, decaying.
    for k in range(int(seconds * 2)):
        s = int(k * 0.5 * SR)
        L = min(2500, n - s)
        if L <= 0:
            break
        env = np.exp(-np.linspace(0, 9, L))
        x[s : s + L] += (r.standard_normal(L) * env * 0.6).astype(np.float32)

    return (x / (np.abs(x).max() + 1e-9)).astype(np.float32)


def degrade(x: np.ndarray) -> np.ndarray:
    """Approximate what a YouTube re-encode does: level change, EQ tilt, noise."""
    y = x * 0.55
    y = np.convolve(y, np.array([0.25, 0.5, 0.25], dtype=np.float32), mode="same")
    y += rng.standard_normal(y.size).astype(np.float32) * 0.004
    return y.astype(np.float32)


print("=" * 66)
print("TEST 1  offset recovery + sign convention")
print("=" * 66)

chart = fake_song(45.0, seed=7)
h_chart = fp.make_hashes(chart)
dur = chart.size / SR

# Positive true offset = the video file has extra material at the front, so a
# given musical moment occurs LATER in the video than in the chart audio.
errors = []
for true_offset in (0.0, 0.35, 1.75, 4.2, 12.0):
    lead = np.zeros(int(true_offset * SR), dtype=np.float32)
    video = degrade(np.concatenate([lead, chart]))
    res = fp.match(h_chart, fp.make_hashes(video), dur)
    err_ms = (res.offset_seconds - true_offset) * 1000
    errors.append(abs(err_ms))
    ok = "PASS" if abs(err_ms) <= 15 and fp.is_same_recording(res) else "FAIL"
    print(
        f"  true {true_offset:6.2f}s -> got {res.offset_seconds:6.2f}s "
        f"(err {err_ms:+7.1f} ms)  score {res.score:6.1f}  "
        f"cov {res.coverage:.2f}  {ok}"
    )

# Negative offset: chart audio has a lead-in the video lacks.
trim = int(2.5 * SR)
video = degrade(chart[trim:])
res = fp.match(h_chart, fp.make_hashes(video), dur)
err_ms = (res.offset_seconds - (-2.5)) * 1000
ok = "PASS" if abs(err_ms) <= 15 else "FAIL"
print(
    f"  true  -2.50s -> got {res.offset_seconds:6.2f}s "
    f"(err {err_ms:+7.1f} ms)  score {res.score:6.1f}  {ok}"
)
print(f"\n  worst absolute error: {max(errors):.1f} ms")

print()
print("=" * 66)
print("TEST 2  rejection of unrelated audio (the wrong-video case)")
print("=" * 66)

for seed in (99, 100, 101, 102):
    other = degrade(fake_song(45.0, seed=seed))
    res = fp.match(h_chart, fp.make_hashes(other), dur)
    verdict = "REJECTED" if not fp.is_same_recording(res) else "*** ACCEPTED ***"
    print(
        f"  unrelated song seed {seed}:  score {res.score:6.1f}  "
        f"cov {res.coverage:.2f}  pairs {res.total_pairs:6d}   {verdict}"
    )

print()
print("=" * 66)
print("TEST 3  separation margin (positive vs negative score distribution)")
print("=" * 66)

pos = []
for s in (7, 8, 9):
    c = fake_song(40.0, seed=s)
    v = degrade(np.concatenate([np.zeros(int(1.2 * SR), dtype=np.float32), c]))
    pos.append(fp.match(fp.make_hashes(c), fp.make_hashes(v), 40.0).score)

neg = []
base = fake_song(40.0, seed=7)
hb = fp.make_hashes(base)
for s in (201, 202, 203):
    neg.append(fp.match(hb, fp.make_hashes(degrade(fake_song(40.0, seed=s))), 40.0).score)

print(f"  same recording  scores: {[round(v, 1) for v in pos]}")
print(f"  different songs scores: {[round(v, 1) for v in neg]}")
print(f"  min positive {min(pos):.1f}   max negative {max(neg):.1f}   "
      f"threshold {fp.ACCEPT_SCORE}")
print(f"  separation: {min(pos) / max(max(neg), 0.1):.1f}x")
