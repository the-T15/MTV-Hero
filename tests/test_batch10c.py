"""
Batch 10c - the rate key is the command.

10b made a measured rate durable, and the review found the key still missed
`cpu_used`: `-cpu-used 0/3/5` at one key measured 44% apart. `preset` was out
for the same reason. Adding columns one flag at a time is how the next flag
gets missed, so the key stops being a hand-picked tuple:

    R1  `rate_key(settings)` is derived from `build_command` itself: the
        tokens that decide the bits, with the input, output, `-threads`,
        clip, pass-log and the constant ffmpeg preamble stripped, and with
        `size_lock` ignored (a locked rate is the lock). Any setting that
        reaches the command therefore reaches the key, now and for every
        flag added later. `cpu_used` and `preset` change it; `clip`,
        `threads_per_job` and the source path do not.
    R2  The database stores the key as one text column (`rates.key`), so a
        wider key needs no schema change. `get_rate` / `set_rate` keep
        their tuple interface. The six-column `rates` table 10b created is
        replaced (it was never released).

Run from the repository root:  pytest -q
"""

from __future__ import annotations

from pathlib import Path

import pytest

from yargvid import encode as enc
from yargvid.db import Database


@pytest.fixture(autouse=True)
def no_ffmpeg(monkeypatch):
    monkeypatch.setattr(enc, "_supports_fps_mode", lambda: True)


def key(**over):
    return enc.rate_key(enc.EncodeSettings(**over))


# ------------------------------------------------------------- R1 key -------

def test_R1_every_bit_deciding_setting_changes_the_key():
    base = key()
    for name, other in [
        ("cpu_used", key(cpu_used=5)),
        ("preset", key(codec="h264", preset="ultrafast")),
        ("crf", key(crf=18)),
        ("bitrate_cap", key(bitrate_cap="2M")),
        ("height", key(height=720)),
        ("fps", key(fps=24.0)),
        ("max_fps", key(max_fps=24.0)),
        ("codec", key(codec="h264")),
    ]:
        against = key(codec="h264") if name == "preset" else base
        assert other != against, name
    assert key(codec="h264", preset="medium") == key(codec="h264")  # default


def test_R1_nothing_else_changes_the_key():
    base = key()
    assert key(clip=(90.0, 20.0)) == base
    assert key(threads_per_job=8) == base
    assert key(size_lock="2M") == key(size_lock="3M") == base
    assert key(crf=31) == base


def test_R1_key_is_the_command_minus_the_file_specific_parts():
    k = key()
    assert isinstance(k, tuple) and all(isinstance(x, str) for x in k)
    s = enc.EncodeSettings()
    cmd = enc.build_command(Path("C:/lib/a/video.src.mp4"),
                            Path("C:/lib/a/video.webm.part"), s, 30.0)
    for token in ("-c:v", "libvpx", "-crf", "31", "-b:v", "4M",
                  "-cpu-used", "3"):
        assert token in k
    for absent in ("ffmpeg", "-i", "-threads", "-y", "-nostdin", "-fps_mode",
                   "-vsync", "-ss", "-t", "-passlogfile"):
        assert absent not in k
    assert not any("video.src" in x or "video.webm" in x for x in k)
    assert set(k) <= set(cmd)                  # nothing invented
    # The preview override reaches the key through the command.
    assert key(cpu_used=5, bitrate_cap="800k", height=480) != k


def test_R1_the_key_is_stable_across_processes():
    """It is persisted, so it must not depend on anything runtime."""
    assert key() == key()
    assert key(codec="h264_nvenc") == key(codec="h264_nvenc")


# ------------------------------------------------------------- R2 storage ---

def test_R2_rates_are_stored_by_the_whole_key(tmp_path):
    d = Database(tmp_path / "r.sqlite")
    slow, fast = key(cpu_used=0), key(cpu_used=5)
    d.set_rate(slow, 600_000.0)
    d.set_rate(fast, 850_000.0)
    assert d.get_rate(slow) == 600_000.0
    assert d.get_rate(fast) == 850_000.0
    assert d.get_rate(key()) is None
    cols = [r["name"] for r in d.conn.execute("PRAGMA table_info(rates)")]
    assert "key" in cols and "cpu_used" not in cols and "crf" not in cols
    d.close()
    d = Database(tmp_path / "r.sqlite")
    assert d.get_rate(fast) == 850_000.0
    d.close()


def test_R2_a_10b_shaped_table_is_replaced_not_crashed_into(tmp_path):
    import sqlite3
    p = tmp_path / "old.sqlite"
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE rates (codec TEXT, height INTEGER, fps REAL, "
              "encoder TEXT, crf INTEGER, bitrate_cap TEXT, bps REAL, "
              "measured_at TEXT, PRIMARY KEY (codec, height, fps, encoder, "
              "crf, bitrate_cap))")
    c.execute("INSERT INTO rates VALUES ('vp8',1080,30.0,'libvpx',31,'4M',"
              "2.9e6,'x')")
    c.commit()
    c.close()
    d = Database(p)
    assert d.get_rate(key()) is None           # old rows are not trusted
    d.set_rate(key(), 2_000_000.0)
    assert d.get_rate(key()) == 2_000_000.0
    d.close()
