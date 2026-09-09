"""
Review app, what you see (Batch 5a).

From the review session after Batch 4. Clips sampled at 25/55/85% landed on
choruses - where repetition lives and where a wrong lock looks right - and
never showed the start, where the lead-in proves the offset, or the end,
where drift and a short video show. The full-song build re-encoded the video
and took minutes; it only needs to shift the audio against a copied stream
and takes seconds. Two facts the pipeline already knew were invisible in the
window: a title that says "audio" and a video that runs out before the song.

    pytest -q tests/test_batch5a.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from yargvid import audio as au                # noqa: E402
from yargvid import cli                        # noqa: E402
from yargvid import encode as enc              # noqa: E402
from yargvid import fingerprint as fp          # noqa: E402
from yargvid import review as rv               # noqa: E402
from yargvid import sync as sy                 # noqa: E402
from yargvid.db import Database                # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class R(dict):
    def __getitem__(self, k):
        return dict.get(self, k)


def _probe(path: Path) -> list[dict]:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                          "-show_streams", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)["streams"]


# ------------------------------------------------- window placement ---------

def test_windows_are_fifteen_seconds():
    assert rv.SEGMENT_SECONDS == 15


def test_windows_cover_start_middle_and_end():
    # chart 200 s, video long enough, offset -1 s: footage starts at 1 s.
    row = R(offset_ms=-1000.0, chart_seconds=200.0)
    starts = rv.segment_starts(row, video_seconds=400.0)
    assert len(starts) == 3
    first, mid, last = starts
    assert 1.0 <= first <= 5.0                       # the lead-in, not a chorus
    assert last + rv.SEGMENT_SECONDS >= 200.0 - 3.0  # the very end of the song
    assert abs(mid - (first + last) / 2) < 2.0       # middle, not 55% of the song


def test_last_window_ends_where_the_footage_does():
    # Closing Time: chart 278.3, video 232.5, offset -3355 ms -> footage reaches 235.9.
    row = R(offset_ms=-3355.0, chart_seconds=278.3)
    starts = rv.segment_starts(row, video_seconds=232.5)
    reach = rv.video_reach(232.5, -3355.0)
    assert len(starts) == 3
    assert starts[-1] + rv.SEGMENT_SECONDS <= reach + 0.01
    assert starts[-1] + rv.SEGMENT_SECONDS >= reach - 3.0
    for s in starts:
        assert s + row["offset_ms"] / 1000.0 >= 0


def test_positive_offset_first_window_starts_at_song_start():
    row = R(offset_ms=20000.0, chart_seconds=200.0)
    starts = rv.segment_starts(row, video_seconds=400.0)
    assert 0.0 <= starts[0] <= 5.0


def test_short_song_gets_fewer_windows():
    row = R(offset_ms=0.0, chart_seconds=18.8)        # blink-182 - Built This Pool
    starts = rv.segment_starts(row, video_seconds=60.0)
    assert 1 <= len(starts) <= 2
    for a, b in zip(starts, starts[1:]):
        assert b - a >= rv.SEGMENT_SECONDS


# ------------------------------------------------- fast full build ----------

@pytest.fixture
def song(tmp_path):
    if not HAVE_FFMPEG:
        pytest.skip("ffmpeg not on PATH")
    d = tmp_path / "song"
    d.mkdir()
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "testsrc=size=160x90:rate=25:duration=40",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=40",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    str(d / "video.src.mkv")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=330:duration=40",
                    "-c:a", "libvorbis", str(d / "guitar.ogg")], check=True)
    subprocess.run(["ffmpeg", "-v", "error", "-y",
                    "-f", "lavfi", "-i", "sine=frequency=220:duration=40",
                    "-c:a", "libvorbis", str(d / "drums.ogg")], check=True)
    return d


def _mean_volume_db(path: Path, t0: float, t1: float) -> float:
    import re
    proc = subprocess.run(["ffmpeg", "-v", "info", "-nostdin", "-ss", str(t0),
                           "-t", str(t1 - t0), "-i", str(path), "-vn",
                           "-af", "volumedetect", "-f", "null", "-"],
                          capture_output=True, text=True)
    m = re.search(r"mean_volume: (-?[\d.]+) dB", proc.stderr)
    return float(m.group(1)) if m else 0.0


def test_full_build_copies_the_video_stream(song, tmp_path):
    # Positive offset: the video seeks ahead of the song, so the song's audio
    # must start late - real silence in front of it, not a timestamp trick
    # that one container honours and another drops.
    row = R(song_dir=str(song), offset_ms=5000.0, chart_seconds=40.0,
            source_path=str(song / "video.src.mkv"))
    out = rv.build_full(row, tmp_path / "work")
    assert out is not None
    streams = {s["codec_type"]: s for s in _probe(out)}
    assert streams["video"]["codec_name"] == "h264"      # copied, not re-encoded
    assert abs(float(streams["audio"]["start_time"])) < 0.1
    assert _mean_volume_db(out, 0.0, 4.0) < -60          # silent lead
    assert _mean_volume_db(out, 6.0, 10.0) > -40         # then the song


def test_full_build_trims_the_song_for_a_negative_offset(song, tmp_path):
    # Negative offset: the video starts |offset| into the song, and there is
    # no footage before that, so the full build drops those seconds of song
    # and both streams start together at zero.
    row = R(song_dir=str(song), offset_ms=-5000.0, chart_seconds=40.0,
            source_path=str(song / "video.src.mkv"))
    out = rv.build_full(row, tmp_path / "work")
    streams = {s["codec_type"]: s for s in _probe(out)}
    assert streams["video"]["codec_name"] == "h264"
    assert abs(float(streams["video"]["start_time"])) < 0.1
    assert abs(float(streams["audio"]["start_time"])) < 0.1
    fmt = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", str(out)], capture_output=True, text=True).stdout
    assert float(fmt) == pytest.approx(35.0, abs=1.0)
    assert _mean_volume_db(out, 0.0, 4.0) > -40           # song audible from the start


# ------------------------------------------------- tags ---------------------

BASE = dict(review=None, fp_score=500.0, spread_ms=1.0, offset_ms=0.0,
            sync_status="ok", motion=0.5, artist="A", title="t", chart_seconds=200.0)


def test_audio_titles_are_tagged():
    r = R(match_note="REVIEW: Song (Official Audio) [Band]", **BASE)
    risk = rv.assess(r)
    assert "audio" in risk.tags
    assert "clean" not in risk.tags
    assert "audio" in rv.TAG_LABELS


def test_short_videos_are_tagged():
    r = R(match_note="T [U]", video_seconds=120.0, **BASE)   # ends 80 s early
    risk = rv.assess(r)
    assert "short" in risk.tags
    assert "1:20" in " ".join(risk.reasons)
    assert "short" in rv.TAG_LABELS


def test_video_that_covers_the_song_is_not_short():
    r = R(match_note="T [U]", video_seconds=198.0, **BASE)   # within 5 s
    assert "short" not in rv.assess(r).tags


def test_unknown_video_length_is_not_short():
    r = R(match_note="T [U]", video_seconds=None, **BASE)
    assert "short" not in rv.assess(r).tags


# ------------------------------------------------- video length column ------

@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


def test_video_seconds_column_exists(db, tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    db.add_song(d, "A", "t", 100.0)
    db.update(d, video_seconds=123.4)
    row = db.conn.execute("SELECT video_seconds FROM songs").fetchone()
    assert row["video_seconds"] == 123.4


def test_sync_stores_the_video_length(db, tmp_path, monkeypatch):
    src = tmp_path / "video.src.mkv"
    src.write_bytes(b"x")
    monkeypatch.setattr(au, "find_stems", lambda d: [Path("x.ogg")])
    monkeypatch.setattr(au, "mix_stems", lambda s, sr=fp.SR: np.ones(sr, np.float32))
    monkeypatch.setattr(au, "decode_mono",
                        lambda p, sr=fp.SR, max_seconds=None: np.ones(sr, np.float32))
    monkeypatch.setattr(au, "duration_of", lambda p: 250.5)
    monkeypatch.setattr(enc, "motion_score", lambda p: 0.5)
    monkeypatch.setattr(sy, "estimate",
                        lambda *a, **k: sy.SyncResult("ok", offset_ms=-1000.0,
                                                      spread_ms=3.0, fp_score=500.0))
    d = tmp_path / "s"
    d.mkdir()
    db.add_song(d, "A", "t", 100.0)
    db.update(d, match_status="ok", download_status="ok", source_path=str(src))
    cli.cmd_sync(SimpleNamespace(recheck=False, min_offset=0.0,
                                 skip_reviewed=False, limit=None), db)
    assert db.conn.execute("SELECT video_seconds FROM songs").fetchone()[0] == 250.5


def test_videos_lengths_backfills_synced_rows(db, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(au, "duration_of", lambda p: 222.0)
    have = tmp_path / "have"
    have.mkdir()
    (have / "video.src.mkv").write_bytes(b"x")
    db.add_song(have, "A", "have", 100.0)
    db.update(have, sync_status="ok", source_path=str(have / "video.src.mkv"))
    done = tmp_path / "done"
    done.mkdir()
    db.add_song(done, "A", "done", 100.0)
    db.update(done, sync_status="ok", source_path=str(done / "gone.mkv"),
              video_seconds=99.0)                       # already measured
    cli.cmd_videos(SimpleNamespace(quiet=True, out=None, mark=False, lengths=True), db)
    rows = {r["title"]: r["video_seconds"] for r in
            db.conn.execute("SELECT title, video_seconds FROM songs")}
    assert rows == {"have": 222.0, "done": 99.0}
    assert "--lengths" in _help(["videos", "-h"])


def _help(argv) -> str:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        cli.main(argv)
    return buf.getvalue()


def test_queue_carries_video_length_and_reach(db, tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "video.src.mkv").write_bytes(b"x")
    (d / "guitar.ogg").write_bytes(b"")
    db.add_song(d, "A", "t", 200.0)
    db.update(d, match_status="ok", match_note="T [U]", download_status="ok",
              source_path=str(d / "video.src.mkv"), sync_status="ok",
              offset_ms=-10000.0, spread_ms=1.0, fp_score=500.0, motion=0.5,
              video_seconds=150.0)
    q = rv.queue(db)[0]
    assert q["video_seconds"] == 150.0
    assert q["reach_seconds"] == pytest.approx(160.0)


# ------------------------------------------------- Qt: what the window says --

PySide6 = pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_list_and_header_name_the_video_and_channel(qapp, tmp_path):
    from PySide6.QtTest import QTest
    from yargvid.app import Window
    dbp = tmp_path / "app.sqlite"
    db = Database(dbp)
    d = tmp_path / "Closing Time"
    d.mkdir()
    (d / "video.src.mkv").write_bytes(b"x")
    (d / "guitar.ogg").write_bytes(b"")
    db.add_song(d, "Semisonic", "Closing Time", 278.0)
    db.update(d, match_status="ok",
              match_note="Semisonic - Closing Time (Official Video) [SemisonicVEVO]",
              download_status="ok", source_path=str(d / "video.src.mkv"),
              sync_status="ok", offset_ms=0.0, spread_ms=1.0, fp_score=500.0,
              motion=0.5)
    db.close()
    w = Window(dbp, tmp_path / "work")
    w.show()
    qapp.processEvents()
    try:
        item = w.list.item(0).text().splitlines()
        assert item[0].startswith("Semisonic") and "Closing Time" in item[0]
        assert item[1] == "Semisonic - Closing Time (Official Video) · SemisonicVEVO"
        w.list.setCurrentRow(0)
        qapp.processEvents()
        assert w.head.text() in ("Semisonic - Closing Time", "Semisonic – Closing Time")
        assert w.by.text() == "Semisonic - Closing Time (Official Video) · SemisonicVEVO"
    finally:
        w.close()
        QTest.qWait(50)
