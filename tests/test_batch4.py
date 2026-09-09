"""
Review clip integrity (Batch 4).

Found by using the app. Semisonic - Closing Time: chart 278 s, video 232.5 s,
offset -3.4 s. The third proof segment was placed at 236 s of song time,
which is 233 s of video time - past the end of the video. ffmpeg produced a
1.4-second scrap and reported success, the `-c copy` concat of that scrap
wrote a container claiming 72 s of video against 24 s of audio, the sidecar
said "three segments" because it recorded what was requested rather than
what was made, and the third button seeked into a broken index and restarted
at 0.

Rules pinned here:
- segments are placed only where the video exists
- every rendered piece is checked before it is joined; scraps are dropped
- the sidecar describes the file, and also records how far the video reaches
- a failed build clears the player instead of leaving the last song loaded
- still-image songs get a clip like any other
- filter chips are single-select

    pytest -q tests/test_batch4.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from yargvid import audio as au                # noqa: E402
from yargvid import review as rv               # noqa: E402
from yargvid.db import Database                # noqa: E402

HAVE_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


class R(dict):
    def __getitem__(self, k):
        return dict.get(self, k)


# ------------------------------------------------- segment placement --------

def test_segments_stay_inside_the_video():
    # Closing Time: chart 278.3 s, video 232.5 s, offset -3355 ms.
    row = R(offset_ms=-3355.0, chart_seconds=278.3)
    starts = rv.segment_starts(row, video_seconds=232.5)
    assert starts, "a short video still gets at least one segment"
    for s in starts:
        video_time = s + row["offset_ms"] / 1000.0
        assert video_time >= 0
        assert video_time + rv.SEGMENT_SECONDS <= 232.5 + 0.01, s


def test_segments_unchanged_when_the_video_covers_the_song():
    row = R(offset_ms=-1000.0, chart_seconds=200.0)
    assert rv.segment_starts(row, video_seconds=400.0) == rv.segment_starts(row)


def test_video_reach_reports_where_footage_ends():
    # video_seconds - offset_s: a delayed video ends later in song time.
    assert rv.video_reach(video_seconds=232.5, offset_ms=-3355.0) == pytest.approx(235.855)
    assert rv.video_reach(video_seconds=100.0, offset_ms=20000.0) == pytest.approx(80.0)


# ------------------------------------------------- the built file -----------

@pytest.fixture
def short_song(tmp_path):
    """A 60-second song whose video is only 40 seconds long."""
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
                    "-f", "lavfi", "-i", "sine=frequency=330:duration=60",
                    "-c:a", "libvorbis", str(d / "guitar.ogg")], check=True)
    return d


def _probe_durations(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-print_format", "json",
                          "-show_streams", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return {s["codec_type"]: float(s["duration"]) for s in json.loads(out)["streams"]}


def test_short_video_builds_a_consistent_clip(short_song, tmp_path):
    row = R(song_dir=str(short_song), offset_ms=0.0, chart_seconds=60.0,
            source_path=str(short_song / "video.src.mkv"))
    work = tmp_path / "work"
    out = rv.build_clip(row, work)
    assert out is not None
    info = rv.clip_info(out)
    starts, lengths, files = info["segments"], info["lengths"], info["files"]
    assert starts, "sidecar lists the segments that exist"
    # One file per window, each consistent with itself and with the length the
    # sidecar claims for it. The windows are separate files now, so there is
    # no joined duration to check and no container for a scrap to break.
    for s, ln, name in zip(starts, lengths, files):
        d = _probe_durations(work / name)
        assert abs(d["video"] - d["audio"]) < 0.5, (name, d)
        assert abs(d["video"] - ln) < 1.0, (name, d, ln)
        assert s + ln <= 40.0 + 0.01, "no segment reaches past the video"


def test_sidecar_records_video_reach(short_song, tmp_path):
    row = R(song_dir=str(short_song), offset_ms=-5000.0, chart_seconds=60.0,
            source_path=str(short_song / "video.src.mkv"))
    out = rv.build_clip(row, tmp_path / "work")
    info = rv.clip_info(out)
    assert info["video_seconds"] == pytest.approx(40.0, abs=0.5)
    assert info["reach_seconds"] == pytest.approx(45.0, abs=0.5)
    assert rv.clip_segments(out) == info["segments"]


def test_render_failure_says_why(tmp_path):
    ok, why = rv._render(tmp_path / "missing.mkv", [tmp_path / "missing.ogg"],
                         0.0, 0.0, 2.0, 90, tmp_path / "out.mp4")
    assert ok is False
    assert why.strip(), "ffmpeg's stderr is returned, not discarded"


# ------------------------------------------------- Qt ------------------------

PySide6 = pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _song(db, d, **cols):
    d.mkdir(parents=True, exist_ok=True)
    (d / "video.src.mkv").write_bytes(b"x")
    (d / "guitar.ogg").write_bytes(b"")
    db.add_song(d, "Artist", d.name, 100.0)
    base = dict(match_status="ok", match_note="T [U]", download_status="ok",
                source_path=str(d / "video.src.mkv"), sync_status="ok",
                offset_ms=0.0, spread_ms=1.0, fp_score=500.0, motion=0.5)
    base.update(cols)
    db.update(d, **base)


@pytest.fixture
def window(qapp, tmp_path):
    from PySide6.QtTest import QTest
    from yargvid.app import Window
    dbp = tmp_path / "app.sqlite"
    d = Database(dbp)
    _song(d, tmp_path / "plain")
    _song(d, tmp_path / "weakone", fp_score=50.0)
    _song(d, tmp_path / "still", motion=0.001)
    d.close()
    w = Window(dbp, tmp_path / "work")
    w.show()
    qapp.processEvents()
    yield w
    w.close()
    QTest.qWait(50)


def test_failed_build_clears_the_player(window, qapp):
    from PySide6.QtCore import QUrl
    w = window
    w._set_mode("third")
    w.list.setCurrentRow(0)
    qapp.processEvents()
    w.player.setSource(QUrl.fromLocalFile("C:/nowhere/previous.mp4"))
    w._clip_ready(w.current, "", "could not build")
    qapp.processEvents()
    assert w.player.source().isEmpty()
    assert not w.seg_btns
    assert "could not build" in w.status.text()


def test_still_image_songs_get_a_clip(window, qapp):
    w = window
    w._set_mode("still")
    qapp.processEvents()
    w.list.setCurrentRow(0)
    qapp.processEvents()
    assert "nothing to watch" not in w.status.text().lower()
    assert "building" in w.status.text().lower()


def test_chips_are_single_select(window, qapp):
    w = window
    w._toggle_tag("weak", True)
    w._toggle_tag("channel", True)
    assert w.active == {"channel"}
    w._toggle_tag("channel", False)
    assert w.active == set()


def test_clip_label_says_where_the_video_ends(window, qapp, tmp_path):
    w = window
    w._set_mode("third")
    w.list.setCurrentRow(0)
    qapp.processEvents()
    clip = tmp_path / "work" / "clip_00000000_00000000.mp4"
    clip.parent.mkdir(exist_ok=True)
    clip.write_bytes(b"")
    rv._sidecar(clip).write_text(json.dumps(
        {"segments": [20.0, 40.0], "video_seconds": 60.0, "reach_seconds": 65.0}))
    w._clip_ready(w.current, str(clip), "")
    qapp.processEvents()
    assert "1:05" in w.window_lbl.text()          # video reaches 1:05 of the song
    assert "1:40" in w.window_lbl.text()          # song is 100 s long
