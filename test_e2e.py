"""End-to-end test: build a fake song folder + video, run sync and encode."""

import subprocess
import tempfile
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from yargvid import audio as au, encode as enc, sync as sy  # noqa: E402
from yargvid.cli import read_song_ini, write_video_start_time  # noqa: E402
from test_fingerprint import fake_song  # noqa: E402

ROOT = Path(tempfile.gettempdir()) / "yargtest" / "Testband - Test Song"
ROOT.mkdir(parents=True, exist_ok=True)
SR_OUT = 44100
TRUE_OFFSET = 3.4   # video has 3.4 s of extra footage before the song starts

# --- build chart stems (guitar + drums, as a real chart would have) -----------
base = fake_song(40.0, seed=21)
up = np.interp(
    np.linspace(0, base.size - 1, int(40.0 * SR_OUT)), np.arange(base.size), base
).astype(np.float32)

for name, gain in (("guitar.ogg", 0.6), ("drums.ogg", 0.55), ("preview.ogg", 0.9)):
    stem = (up * gain).astype(np.float32)
    if name == "preview.ogg":
        stem = stem[: SR_OUT * 5]     # must be excluded by find_stems
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "f32le", "-ar", str(SR_OUT),
         "-ac", "1", "-i", "-", "-c:a", "libvorbis", str(ROOT / name)],
        input=stem.tobytes(), check=True,
    )

(ROOT / "song.ini").write_text(
    "[song]\nname = Test Song\nartist = Testband\ndelay = 0\n", encoding="utf-8"
)

# --- build a video whose audio leads by TRUE_OFFSET ---------------------------
lead = np.zeros(int(TRUE_OFFSET * SR_OUT), dtype=np.float32)
vid_audio = np.concatenate([lead, up * 0.5]).astype(np.float32)
vsrc = ROOT / "source.mp4"
subprocess.run(
    ["ffmpeg", "-y", "-v", "error",
     "-f", "lavfi", "-i", f"testsrc2=size=1280x720:rate=25:duration={vid_audio.size/SR_OUT:.2f}",
     "-f", "f32le", "-ar", str(SR_OUT), "-ac", "1", "-i", "-",
     "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
     "-c:a", "aac", "-shortest", str(vsrc)],
    input=vid_audio.tobytes(), check=True,
)

print("=" * 64)
print("STAGE: stem discovery")
print("=" * 64)
stems = au.find_stems(ROOT)
print(f"  stems used: {[s.name for s in stems]}")
assert all("preview" not in s.name for s in stems), "preview.ogg was not excluded"
print(f"  metadata:   {read_song_ini(ROOT)}")

print("\n" + "=" * 64)
print("STAGE: sync")
print("=" * 64)
from yargvid import fingerprint as fp  # noqa: E402

chart = au.mix_stems(stems, fp.SR)
video = au.decode_mono(vsrc, fp.SR)
res = sy.estimate(chart, video, au.mix_stems(stems, sy.REFINE_SR),
                  au.decode_mono(vsrc, sy.REFINE_SR))

err = res.offset_ms - TRUE_OFFSET * 1000
print(f"  status            {res.status}")
print(f"  fingerprint score {res.fp_score:.0f}")
print(f"  video_start_time  {res.video_start_time} ms   (true {TRUE_OFFSET*1000:.0f})")
print(f"  error             {err:+.1f} ms")
print(f"  excerpt spread    {res.spread_ms:.1f} ms across {len(res.excerpts)} windows")
assert res.status == "ok" and abs(err) < 20, "sync failed"

print("\n" + "=" * 64)
print("STAGE: encode (720p source -> padded 1080p VP8)")
print("=" * 64)
ok, err_msg = enc.encode_one(vsrc, ROOT, enc.EncodeSettings(cpu_used=5), keep_source=True)
out = ROOT / "video.webm"
print(f"  success: {ok} {err_msg}")
info = au.probe(out)
vs = [s for s in info.get("streams", []) if s["codec_type"] == "video"][0]
has_audio = any(s["codec_type"] == "audio" for s in info.get("streams", []))
print(f"  codec {vs['codec_name']}  {vs['width']}x{vs['height']}  "
      f"pix_fmt {vs['pix_fmt']}  fps {vs['r_frame_rate']}")
print(f"  size {out.stat().st_size/1024:.0f} KB   audio stream present: {has_audio}")
assert vs["codec_name"] == "vp8" and (vs["width"], vs["height"]) == (1920, 1080)
assert not has_audio, "audio should be stripped"

print("\n" + "=" * 64)
print("STAGE: song.ini write")
print("=" * 64)
write_video_start_time(ROOT, res.video_start_time)
print((ROOT / "song.ini").read_text().strip())
assert (ROOT / "song.ini.bak").exists()

# Idempotency: writing twice must not duplicate the key.
write_video_start_time(ROOT, 999)
text = (ROOT / "song.ini").read_text()
assert text.count("video_start_time") == 1, "key duplicated on rewrite"
print("\n  rewrite is idempotent (key updated in place, not duplicated)")

print("\nALL STAGES PASSED")
