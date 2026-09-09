"""
Review app, clips and playback (Batch 6a).

From the 5b review session. The three windows are three files: a button
loads one, nothing is concatenated, so there is no container for a scrap to
break. The end window is 30 s, because 15 was not long enough to see drift
or a short video run out. Selecting a song clears the player at once, the
build waits 300 ms so a held arrow key does not queue one per song, and a
result from an earlier build is dropped. Click the video to pause, click the
timeline to seek, skip ten seconds either way; the full build seeks on
release and snaps to a keyframe, since a stream-copied VP9 stalls between
them. Open source opens the file, not a moment in it. The window label says
when the video runs past the song as well as when it ends early.

    pytest -q tests/test_batch6a.py
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


def _no_overlap(windows):
    for (a, la), (b, _) in zip(windows, windows[1:]):
        assert b >= a + la - 0.01, windows


# ------------------------------------------------- window placement ---------

def test_intro_is_fifteen_seconds_and_the_end_is_thirty():
    assert rv.SEGMENT_SECONDS == 15
    assert rv.END_SECONDS == 30


def test_windows_are_intro_middle_and_a_long_end():
    # chart 200 s, video long enough, offset -1 s: footage starts at 1 s.
    row = R(offset_ms=-1000.0, chart_seconds=200.0)
    windows = rv.segment_windows(row, video_seconds=400.0)
    assert len(windows) == 3
    (s0, l0), (s1, l1), (s2, l2) = windows
    assert 1.0 <= s0 <= 5.0 and l0 == rv.SEGMENT_SECONDS
    assert l1 == rv.SEGMENT_SECONDS
    assert l2 == rv.END_SECONDS
    assert s2 + l2 >= 200.0 - 3.0                    # the very end of the song
    assert s2 + l2 <= 200.0 + 0.01
    assert abs(s1 - (s0 + s2) / 2) < 2.0             # halfway between the other two
    _no_overlap(windows)


def test_short_footage_still_shows_its_end():
    # 60 s song, 40 s video, offset 0: the intro and the run-out both matter,
    # and 15 + 30 do not fit in 40. Two windows, neither past the video, the
    # second ending where the footage does.
    row = R(offset_ms=0.0, chart_seconds=60.0)
    windows = rv.segment_windows(row, video_seconds=40.0)
    assert len(windows) >= 2
    _no_overlap(windows)
    for s, ln in windows:
        assert s >= 0 and s + ln <= 40.0 + 0.01, windows
    s, ln = windows[-1]
    assert s + ln >= 40.0 - 3.0


def test_windows_stay_inside_a_short_video():
    # Closing Time: chart 278.3, video 232.5, offset -3355 ms -> reach 235.9.
    row = R(offset_ms=-3355.0, chart_seconds=278.3)
    windows = rv.segment_windows(row, video_seconds=232.5)
    reach = rv.video_reach(232.5, -3355.0)
    assert len(windows) == 3
    _no_overlap(windows)
    for s, ln in windows:
        video_time = s + row["offset_ms"] / 1000.0
        assert video_time >= 0
        assert video_time + ln <= 232.5 + 0.01, (s, ln)
    s, ln = windows[-1]
    assert ln == rv.END_SECONDS
    assert s + ln >= reach - 3.0


def test_very_short_song_gets_one_window():
    row = R(offset_ms=0.0, chart_seconds=18.8)        # blink-182 - Built This Pool
    windows = rv.segment_windows(row, video_seconds=60.0)
    assert len(windows) == 1
    s, ln = windows[0]
    assert s >= 0 and s + ln <= 18.8 + 0.01


def test_segment_starts_are_the_window_starts():
    row = R(offset_ms=-1000.0, chart_seconds=200.0)
    assert rv.segment_starts(row, 400.0) == [s for s, _ in rv.segment_windows(row, 400.0)]
    assert rv.segment_starts(row) == [s for s, _ in rv.segment_windows(row)]


# ------------------------------------------------- keyframes ----------------

def test_snap_to_keyframe_takes_the_latest_one_before():
    kf = [0.0, 5.0, 10.0]
    assert rv.snap_to_keyframe(7200, kf) == 5000
    assert rv.snap_to_keyframe(10000, kf) == 10000
    assert rv.snap_to_keyframe(100, kf) == 0
    assert rv.snap_to_keyframe(7200, []) == 7200       # nothing known: seek as asked


# ------------------------------------------------- dead code ----------------

def test_open_source_dead_code_is_gone(tmp_path):
    assert not hasattr(rv, "clip_times")
    assert not hasattr(rv, "CLIP_SECONDS")
    db = Database(tmp_path / "t.sqlite")
    try:
        d = tmp_path / "s"
        d.mkdir()
        (d / "video.src.mkv").write_bytes(b"x")
        db.add_song(d, "A", "t", 200.0)
        db.update(d, match_status="ok", match_note="T [U]", download_status="ok",
                  source_path=str(d / "video.src.mkv"), sync_status="ok",
                  offset_ms=0.0, spread_ms=1.0, fp_score=500.0, motion=0.5)
        q = rv.queue(db)[0]
    finally:
        db.close()
    assert "clip_video_s" not in q and "clip_chart_s" not in q


# ------------------------------------------------- the built files ---------

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


def test_clip_is_separate_files_not_a_concat(short_song, tmp_path):
    work = tmp_path / "work"
    row = R(song_dir=str(short_song), offset_ms=0.0, chart_seconds=60.0,
            source_path=str(short_song / "video.src.mkv"))
    out = rv.build_clip(row, work)
    assert out is not None
    assert out.name.endswith(".seg0.mp4")
    info = rv.clip_info(out)
    starts, lengths, files = info["segments"], info["lengths"], info["files"]
    assert len(starts) == len(lengths) == len(files) >= 2
    assert files[0] == out.name
    for s, ln, name in zip(starts, lengths, files):
        f = work / name
        assert f.exists(), name
        assert abs(au.duration_of(f) - ln) < 1.0, (name, ln)
        assert s + ln <= 40.0 + 0.01, "no window reaches past the video"
    assert starts[-1] + lengths[-1] >= 40.0 - 3.0, "the end window shows the run-out"
    assert not (work / rv.clip_name(row)).exists(), "nothing was concatenated"
    assert not list(work.glob("*.list.txt"))
    assert rv.clip_segments(out) == starts


def test_clip_cache_hit_needs_every_file(short_song, tmp_path, monkeypatch):
    work = tmp_path / "work"
    row = R(song_dir=str(short_song), offset_ms=0.0, chart_seconds=60.0,
            source_path=str(short_song / "video.src.mkv"))
    out = rv.build_clip(row, work)
    files = rv.clip_info(out)["files"]
    monkeypatch.setattr(rv, "_render", lambda *a, **k: (False, "not rendering"))
    assert rv.build_clip(row, work) == out             # a hit: nothing rendered
    (work / files[-1]).unlink()
    assert rv.build_clip(row, work) is None            # a file is missing: not a hit


@pytest.fixture
def song(tmp_path):
    """A 40-second song with a 40-second video."""
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
    return d


def test_full_build_records_its_keyframes(song, tmp_path):
    row = R(song_dir=str(song), offset_ms=0.0, chart_seconds=40.0,
            source_path=str(song / "video.src.mkv"))
    out = rv.build_full(row, tmp_path / "work")
    assert out is not None
    kf = rv.clip_info(out)["keyframes"]
    assert len(kf) >= 2                                 # libx264 keyint 250 at 25 fps
    assert kf[0] < 0.1                                  # Matroska may start at ~23 ms
    assert kf == sorted(kf)
    assert kf[-1] <= 40.0


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
                offset_ms=0.0, spread_ms=1.0, fp_score=500.0, motion=0.5,
                video_seconds=100.0)
    base.update(cols)
    db.update(d, **base)


class FakePool:
    """Records the jobs the window would have started; runs none of them."""

    def __init__(self):
        self.started = []

    def start(self, job):
        self.started.append(job)


@pytest.fixture
def window(qapp, tmp_path):
    from PySide6.QtTest import QTest
    from yargvid.app import Window
    dbp = tmp_path / "app.sqlite"
    d = Database(dbp)
    _song(d, tmp_path / "one", video_id="dQw4w9WgXcQ")
    _song(d, tmp_path / "two")
    d.close()
    w = Window(dbp, tmp_path / "work")
    w.pool = FakePool()                        # no ffmpeg runs behind these tests
    w.show()
    qapp.processEvents()
    w._set_mode("third")                       # channel "U" is not "Artist"
    qapp.processEvents()
    yield w
    w.close()
    QTest.qWait(50)


KEY = "clip_00000000_00000000"


def _fake_clip(work: Path, starts, lengths, chart=100.0, reach=100.0) -> Path:
    """Segment files and their sidecar, the way build_clip leaves them."""
    work.mkdir(exist_ok=True)
    files = [f"{KEY}.seg{i}.mp4" for i in range(len(starts))]
    for name in files:
        (work / name).write_bytes(b"")
    (work / f"{KEY}.segments.json").write_text(json.dumps({
        "segments": starts, "lengths": lengths, "files": files,
        "video_seconds": reach, "reach_seconds": reach}))
    return work / files[0]


def _fake_full(work: Path, keyframes) -> Path:
    work.mkdir(exist_ok=True)
    full = work / "full_00000000_00000000.mkv"
    full.write_bytes(b"")
    (work / "full_00000000_00000000.segments.json").write_text(json.dumps({
        "keyframes": keyframes, "video_seconds": 100.0}))
    return full


def test_selecting_a_song_clears_the_player_at_once(window, qapp, tmp_path):
    w = window
    w.list.setCurrentRow(0)
    qapp.processEvents()
    seg0 = _fake_clip(tmp_path / "work", [1.0, 42.0, 69.0], [15, 15, 30])
    w._clip_ready(w.current, str(seg0), "")
    qapp.processEvents()
    assert not w.player.source().isEmpty()
    assert w.seg_btns and w.window_lbl.text()
    w.list.setCurrentRow(1)                    # no waiting: this is immediate
    assert w.player.source().isEmpty()
    assert w.seg_btns == []
    assert w.window_lbl.text() == ""
    assert "building" in w.status.text().lower()


def test_builds_are_debounced(window, qapp):
    from PySide6.QtTest import QTest
    from yargvid import app as appmod
    assert appmod.BUILD_DELAY_MS == 300
    w = window
    w.pool.started.clear()
    w.list.setCurrentRow(0)
    w.list.setCurrentRow(1)
    qapp.processEvents()
    assert w.pool.started == []                # nothing yet: the timer is running
    QTest.qWait(450)
    assert len(w.pool.started) == 1            # one build, for the song you stopped on
    assert w.pool.started[0].song_dir == w.current


def test_stale_build_results_are_dropped(window, qapp, tmp_path):
    from PySide6.QtTest import QTest
    w = window
    w.list.setCurrentRow(0)
    QTest.qWait(450)                           # let the build be issued a number
    assert len(w.pool.started) == 1
    seg0 = _fake_clip(tmp_path / "work", [1.0, 42.0, 69.0], [15, 15, 30])
    before = w.status.text()
    w._clip_ready(w.current, str(seg0), "", token=-1)   # a token never issued
    qapp.processEvents()
    assert w.player.source().isEmpty()
    assert w.seg_btns == []
    assert w.status.text() == before
    w._clip_ready(w.current, str(seg0), "")            # no token: trusted
    qapp.processEvents()
    assert not w.player.source().isEmpty()


def test_segment_buttons_load_separate_files(window, qapp, tmp_path):
    w = window
    w.list.setCurrentRow(0)
    qapp.processEvents()
    seg0 = _fake_clip(tmp_path / "work", [1.0, 42.0, 69.0], [15, 15, 30])
    w._clip_ready(w.current, str(seg0), "")
    qapp.processEvents()
    assert len(w.seg_btns) == 3
    assert w.player.source().toLocalFile().endswith(f"{KEY}.seg0.mp4")
    assert w.seg_btns[0].isChecked()
    label = w.window_lbl.text()
    for t in ("0:01", "0:42", "1:09", "30"):
        assert t in label, label
    w.seg_btns[2].click()
    qapp.processEvents()
    assert w.player.source().toLocalFile().endswith(f"{KEY}.seg2.mp4")
    assert w.seg_btns[2].isChecked() and not w.seg_btns[0].isChecked()


def test_label_says_when_the_video_runs_past_the_song(window, qapp, tmp_path):
    w = window
    w.list.setCurrentRow(0)
    qapp.processEvents()
    seg0 = _fake_clip(tmp_path / "work", [1.0, 42.0, 69.0], [15, 15, 30],
                      chart=100.0, reach=120.0)
    w._clip_ready(w.current, str(seg0), "")
    qapp.processEvents()
    assert "runs 20 s past the song" in w.window_lbl.text()


def test_skip_buttons_move_ten_seconds(window, qapp):
    w = window
    w.list.setCurrentRow(0)
    qapp.processEvents()
    assert "10" in w.back_btn.text() and "10" in w.fwd_btn.text()
    got = []
    w.player.position = lambda: 30000
    w.player.duration = lambda: 60000
    w.player.setPosition = lambda ms: got.append(ms)
    w.fwd_btn.click()
    w.back_btn.click()
    assert got == [40000, 20000]
    got.clear()
    w.player.position = lambda: 5000
    w.back_btn.click()
    w.player.position = lambda: 55000
    w.fwd_btn.click()
    assert got == [0, 60000]                   # clamped to the clip


def test_clicking_the_video_toggles_playback(window, qapp):
    from PySide6.QtCore import QEvent, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtMultimedia import QMediaPlayer
    from PySide6.QtWidgets import QApplication
    w = window
    w.list.setCurrentRow(0)
    qapp.processEvents()
    calls = []
    w.player.play = lambda: calls.append("play")
    w.player.pause = lambda: calls.append("pause")
    w.player.playbackState = lambda: QMediaPlayer.PlaybackState.StoppedState

    def press():
        local = QPointF(w.video.width() / 2, w.video.height() / 2)
        ev = QMouseEvent(QEvent.Type.MouseButtonPress, local,
                         QPointF(w.video.mapToGlobal(local.toPoint())),
                         Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                         Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(w.video, ev)

    press()
    assert calls == ["play"]
    w.player.playbackState = lambda: QMediaPlayer.PlaybackState.PlayingState
    press()
    assert calls == ["play", "pause"]


def test_clicking_the_timeline_seeks_a_segment(window, qapp, tmp_path):
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest
    w = window
    w.list.setCurrentRow(0)
    qapp.processEvents()
    seg0 = _fake_clip(tmp_path / "work", [1.0, 42.0, 69.0], [15, 15, 30])
    w._clip_ready(w.current, str(seg0), "")
    qapp.processEvents()
    got = []
    w.player.setPosition = lambda ms: got.append(ms)
    w.scrub.setRange(0, 10000)
    QTest.mouseClick(w.scrub, Qt.MouseButton.LeftButton, pos=QPoint(
        int(w.scrub.width() * 0.75), w.scrub.height() // 2))
    qapp.processEvents()
    assert got, "a click on the timeline seeks"
    assert 6500 <= got[-1] <= 8500, got        # to where you clicked, not a page step


def test_full_build_seeks_on_release_snapped_to_a_keyframe(window, qapp, tmp_path):
    w = window
    w.list.setCurrentRow(0)
    qapp.processEvents()
    full = _fake_full(tmp_path / "work", [0.0, 5.0, 10.0])
    w._clip_ready(w.current, str(full), "")
    qapp.processEvents()
    assert "full" in w.window_lbl.text().lower()
    got = []
    w.player.setPosition = lambda ms: got.append(ms)
    w.scrub.setRange(0, 40000)
    w.scrub.sliderMoved.emit(7200)
    assert got == [], "dragging the full build does not seek"
    w.scrub.setValue(7200)
    w.scrub.sliderReleased.emit()
    assert got == [5000], "release seeks, to the keyframe before"


def test_open_source_has_no_timestamp(window, qapp, monkeypatch):
    from yargvid import app as appmod
    w = window
    for i, s in enumerate(w.songs):
        if s["video_id"]:
            w.list.setCurrentRow(i)
            break
    qapp.processEvents()
    opened = []
    monkeypatch.setattr(appmod.QDesktopServices, "openUrl",
                        lambda url: opened.append(url.toString()))
    w._open_source()
    assert opened == ["https://www.youtube.com/watch?v=dQw4w9WgXcQ"]
