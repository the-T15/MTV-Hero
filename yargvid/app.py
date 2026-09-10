"""
Native desktop review window.

Same job as the browser version and the same engine underneath - review.assess
ranks by doubt, review.build_clip mixes the aligned clip - but in a real
window with no server, no port and no tab.

Clips are still built by ffmpeg with the offset baked in rather than played as
video-plus-separate-audio. That was never a browser limitation: aligning two
media streams at playback time means trusting two clocks to agree, which is
the exact thing under review. One muxed file has one clock.

Clip building runs on a worker thread. It takes a couple of seconds, and a
frozen window during it would make the app feel broken.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Qt's FFmpeg backend dumps a full stream listing for every clip it opens,
# which buries the app's own output. Set before Qt is imported.
os.environ.setdefault("QT_LOGGING_RULES", "qt.multimedia.ffmpeg=false")

from PySide6.QtCore import (QEvent, QObject, QPoint, QRect, QRunnable, Qt,
                            QThreadPool, QTimer, QUrl, Signal)
from PySide6.QtGui import (QDesktopServices, QFont, QKeySequence, QShortcut,
                           QWindow)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QApplication, QGridLayout, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton,
                               QSizePolicy, QSlider, QSplitter, QStyle,
                               QStyleOptionSlider, QVBoxLayout, QWidget)
from collections import Counter

from . import review as rv
from .db import Database

# Held down, an arrow key walks the list a row at a time and each row used to
# start an ffmpeg run. The window then stalled behind a queue of builds for
# songs nobody was looking at any more. Selecting is instant; building waits
# to see whether you have stopped.
BUILD_DELAY_MS = 300

# How far the skip buttons move.
SKIP_MS = 10000

# The steps the offset buttons move by, left to right. Four sizes cover a
# frame, a beat and a section, which is everything the typed box was ever
# used for.
OFFSET_STEPS = (-100, -10, -5, -1, 1, 5, 10, 100)

BASE, PANEL, LINE = "#12161c", "#171d25", "#2a323d"
INK, DIM = "#d9dfe7", "#79848f"
SIGNAL, CLEAR = "#e8a33d", "#6fbf8f"

STYLE = f"""
QWidget {{ background:{BASE}; color:{INK};
           font-family:"Segoe UI"; font-size:14px; }}
QSplitter::handle {{ background:{LINE}; width:1px; }}
QListWidget {{ background:{PANEL}; border:none; outline:none; }}
QListWidget::item {{ border-bottom:1px solid {LINE};
                     border-left:3px solid transparent; padding:10px 14px; }}
QListWidget::item:selected {{ background:#1f2731; border-left-color:{SIGNAL};
                              color:{INK}; }}
QListWidget::item:hover {{ background:#1c232c; }}
QPushButton {{ background:#1d242d; border:1px solid {LINE}; padding:8px 18px; }}
QPushButton:hover {{ border-color:{DIM}; }}
QPushButton:focus {{ border-color:{SIGNAL}; }}
QPushButton#keep {{ border-color:{CLEAR}; color:{CLEAR}; }}
QPushButton#tab {{ background:transparent; border:1px solid {LINE};
                   padding:5px 14px; color:{DIM}; }}
QPushButton#tab:checked {{ border-color:{SIGNAL}; color:{INK};
                           background:#1f2731; }}
QPushButton#chip {{ background:transparent; border:1px solid {LINE};
                    padding:3px 9px; font-size:12px; color:{DIM}; }}
QPushButton#chip:hover {{ color:{INK}; }}
QPushButton#chip:checked {{ border-color:{SIGNAL}; color:{SIGNAL};
                            background:#221b10; }}
QLineEdit {{ background:#0e1218; border:1px solid {LINE}; padding:8px 10px; }}
QLineEdit:focus {{ border-color:{SIGNAL}; }}
QLabel#head {{ font-size:22px; font-weight:600; }}
QLabel#by, QLabel#hint {{ color:{DIM}; }}
QLabel#flags {{ color:{SIGNAL}; }}
QLabel#status {{ color:{DIM}; padding:14px; }}
"""


class ClipSignals(QObject):
    done = Signal(str, str, str, int)     # song_dir, path, error, token


class ClipJob(QRunnable):
    """Build one proof clip off the UI thread."""

    def __init__(self, db_path: Path, work: Path, song_dir: str,
                 full: bool = False, token: int = 0,
                 offset_ms: float | None = None):
        super().__init__()
        self.db_path, self.work, self.song_dir = db_path, work, song_dir
        self.full = full
        # A draft offset to build at, in place of the stored one. Nothing is
        # written: the clip's name carries the offset already, so a draft is
        # just another cache entry beside the saved one.
        self.offset_ms = offset_ms
        # The serial of the build this job is. Checking the song alone was
        # not enough: nudging the offset re-selects the same song, so the
        # build at the old offset came back matching `current` and landed on
        # top of the one that replaced it.
        self.token = token
        self.signals = ClipSignals()

    def _emit(self, path: str, error: str) -> None:
        try:
            self.signals.done.emit(self.song_dir, path, error, self.token)
        except RuntimeError:
            pass          # the window moved on and dropped this job

    def run(self):
        db = Database(self.db_path)
        try:
            row = db.conn.execute(
                "SELECT * FROM songs WHERE song_dir = ?", (self.song_dir,)
            ).fetchone()
            if row is None:
                return self._emit("", "Song not found.")
            if self.offset_ms is not None:
                row = dict(row)
                row["offset_ms"] = float(self.offset_ms)
            song_dir = Path(row["song_dir"])
            # Order matters: a folder that no longer exists explains every
            # other symptom, so check it before blaming the encode step for
            # removing the source.
            if not song_dir.is_dir():
                return self._emit("", "This song folder no longer exists. It was "
                    "probably moved or renamed after indexing.\nRe-run index "
                    "to pick it up at its new path.")
            from . import audio as au
            if not au.find_stems(song_dir):
                return self._emit("", "No chart audio found in this folder.")
            if rv.source_video(row) is None:
                return self._emit("", "The source video is gone - encode removed it.\n"
                                      "Re-run download for this song to review it.")
            clip = (rv.build_full(row, self.work) if self.full
                    else rv.build_clip(row, self.work))
            if clip is None:
                return self._emit("", "ffmpeg could not build a clip.")
            self._emit(str(clip), "")
        except Exception as exc:                                   # noqa: BLE001
            self._emit("", str(exc)[:200])
        finally:
            db.close()


class Timeline(QSlider):
    """
    A timeline that goes where you click it.

    A plain QSlider treats a click on the groove as a page step: clicking
    three quarters of the way along a four-minute song moved it ten seconds,
    which reads as the control being broken rather than as a page. The press
    jumps to the position under the pointer and holds the slider down, so a
    press-and-drag continues from there and the release is a release of a
    drag that started where you meant it to.
    """

    def _value_at(self, x: int) -> int:
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()
        groove = style.subControlRect(QStyle.ComplexControl.CC_Slider, opt,
                                      QStyle.SubControl.SC_SliderGroove, self)
        handle = style.subControlRect(QStyle.ComplexControl.CC_Slider, opt,
                                      QStyle.SubControl.SC_SliderHandle, self)
        # The handle's own width is not part of the travel, and the pointer
        # grabs its middle: both ends have to come off before the fraction
        # means anything.
        span = max(1, groove.width() - handle.width())
        return QStyle.sliderValueFromPosition(
            self.minimum(), self.maximum(),
            x - handle.width() // 2 - groove.x(), span)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.MouseButton.LeftButton:
            self.setSliderDown(True)
            self.setSliderPosition(self._value_at(int(ev.position().x())))
            ev.accept()
            return
        super().mousePressEvent(ev)

    def mouseMoveEvent(self, ev):
        if self.isSliderDown():
            self.setSliderPosition(self._value_at(int(ev.position().x())))
            ev.accept()
            return
        super().mouseMoveEvent(ev)

    def mouseReleaseEvent(self, ev):
        # QSlider's own release handler ignores an event it did not see the
        # press for, which would leave the slider down for ever and never
        # emit sliderReleased.
        if self.isSliderDown():
            self.setSliderDown(False)
            ev.accept()
            return
        super().mouseReleaseEvent(ev)


class Window(QWidget):
    def __init__(self, db_path: Path, work: Path):
        super().__init__()
        self.db_path, self.work = db_path, work
        self.pool = QThreadPool.globalInstance()
        self.all_songs: list[dict] = []
        self.songs: list[dict] = []
        self.active: set[str] = set()
        self.mode = "clean"
        # None is queue order: clicking the active sort clears it rather than
        # leaving you with no way back to the order the queue arrived in.
        self.sort: str | None = "artist"
        # An offset being tried out. It is not this song's offset until Save
        # says so, so it lives here and not in the row.
        self.draft_offset: float | None = None
        self.chips: dict[str, QPushButton] = {}
        self.current: str | None = None
        # What is on screen, as opposed to what was asked for.
        self._token = 0                   # serial of the latest build
        self._pending_full = False
        self._files: list[str] = []       # the segment files, in window order
        self._full = False                # a whole-song build is loaded
        self._keyframes: list[float] = []

        self.setWindowTitle("Backgrounds to check")
        self.resize(1180, 760)
        self.setStyleSheet(STYLE)

        # --- left: the worklist ------------------------------------------
        self.count = QLabel()
        self.count.setObjectName("by")
        self.count.setContentsMargins(14, 12, 14, 8)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select)

        # Nothing found one song by name. With 1,500 of them the only way
        # back to the one you were looking at a minute ago was to scroll for
        # it, and the video's own title is as likely to be what you remember
        # as the song's.
        self.search = QLineEdit()
        self.search.setPlaceholderText("Find a song, video or channel")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(lambda _: self._on_search())
        search_holder = QWidget()
        sh = QHBoxLayout(search_holder)
        sh.setContentsMargins(14, 0, 14, 8)
        sh.setSpacing(0)
        sh.addWidget(self.search)

        # One song in exactly one tile, assigned by rv.bucket. The tile is the
        # filter. The old tabs overlapped, so a still image on a stranger's
        # channel was counted twice and listed once, in whichever tab you
        # happened to have open. Two columns because eight of these do not fit
        # across the pane, and the pane width is right as it is.
        self.tiles: dict[str, QPushButton] = {}
        grid = QGridLayout()
        grid.setContentsMargins(14, 0, 14, 8)
        grid.setSpacing(6)
        for i, tile in enumerate(rv.TILES):
            b = QPushButton(rv.TILE_LABELS[tile])
            b.setObjectName("tab")
            b.setCheckable(True)
            b.setChecked(tile == self.mode)
            b.clicked.connect(lambda _, m=tile: self._set_mode(m))
            self.tiles[tile] = b
            grid.addWidget(b, i // 2, i % 2)
        tile_holder = QWidget()
        tile_holder.setLayout(grid)

        # One choice with three settings. Three separate buttons read as three
        # unrelated controls, and two of them ranked on the match score, which
        # says nothing about where in the list a song should be.
        self.sorts: dict[str, QPushButton] = {}
        sort_bar = QHBoxLayout()
        sort_bar.setContentsMargins(14, 0, 14, 8)
        sort_bar.setSpacing(6)
        sort_lbl = QLabel("Sort")
        sort_lbl.setObjectName("hint")
        sort_bar.addWidget(sort_lbl)
        for key, label in (("artist", "Artist A-Z"),
                           ("title", "Title A-Z"),
                           ("doubt", "Doubt ↓"),
                           ("doubt_asc", "Doubt ↑")):
            b = QPushButton(label)
            b.setObjectName("chip")
            b.setCheckable(True)
            b.setChecked(key == self.sort)
            b.clicked.connect(lambda _, k=key: self._set_sort(k))
            self.sorts[key] = b
            sort_bar.addWidget(b)
        sort_bar.addStretch(1)
        sort_holder = QWidget()
        sort_holder.setLayout(sort_bar)

        # Chips filter within Unsure and nowhere else. Under any other tile
        # the tile itself is the filter, and a second row of filters under it
        # is two controls doing one job.
        self.chip_bar = QHBoxLayout()
        self.chip_bar.setContentsMargins(14, 0, 14, 10)
        self.chip_bar.setSpacing(6)
        self.chip_holder = QWidget()
        self.chip_holder.setLayout(self.chip_bar)
        self.chip_holder.setVisible(self.mode == "unsure")

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lv.addWidget(self.count)
        lv.addWidget(search_holder)
        lv.addWidget(tile_holder)
        lv.addWidget(sort_holder)
        lv.addWidget(self.chip_holder)
        lv.addWidget(self.list, 1)

        # --- right: player and detail -------------------------------------
        self.head = QLabel("Pick a song on the left")
        self.head.setObjectName("head")
        self.by = QLabel("Sorted by artist, A to Z.")
        self.by.setObjectName("by")
        self.by.setWordWrap(True)
        # The song title, the video title and the figures are the strings you
        # paste into a search when you go looking for the video by hand. A
        # label you cannot select is a string you have to retype.
        for lbl in (self.head, self.by):
            lbl.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse)

        self.video = QVideoWidget()
        self.video.setMinimumHeight(460)
        self.video.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)
        self.video.setStyleSheet("background:#000;")
        self.player = QMediaPlayer()
        self.audio = QAudioOutput()
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video)
        self.player.mediaStatusChanged.connect(self._loop)
        self.player.playbackStateChanged.connect(self._sync_play_button)

        self.play_btn = QPushButton("Pause")
        self.play_btn.setObjectName("chip")
        self.play_btn.clicked.connect(self._toggle)
        self.restart_btn = QPushButton("Restart")
        self.restart_btn.setObjectName("chip")
        self.restart_btn.clicked.connect(self._restart)
        self.back_btn = QPushButton("← 10 s")
        self.back_btn.setObjectName("chip")
        self.back_btn.clicked.connect(lambda: self._skip(-SKIP_MS))
        self.fwd_btn = QPushButton("10 s →")
        self.fwd_btn.setObjectName("chip")
        self.fwd_btn.clicked.connect(lambda: self._skip(SKIP_MS))
        # Without the timestamps there is no way to check the clip against the
        # source, which leaves the offset unverifiable outside the game.
        self.window_lbl = QLabel("")
        self.window_lbl.setObjectName("hint")
        self.scrub = Timeline(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, 0)
        self.scrub.sliderMoved.connect(self._on_scrub_moved)
        self.scrub.sliderReleased.connect(self._on_scrub_released)
        self.player.positionChanged.connect(self._on_position)
        self.player.durationChanged.connect(
            lambda ms: self.scrub.setRange(0, ms))
        self.time_lbl = QLabel("0:00")
        self.time_lbl.setObjectName("hint")

        self.full_btn = QPushButton("Load full song")
        self.full_btn.setObjectName("chip")
        self.full_btn.clicked.connect(self._load_full)
        self.full_btn.setToolTip(
            "Builds the whole song with the offset applied so you can scrub "
            "through it. Slower to prepare; cached afterwards.")

        self.open_btn = QPushButton("Open source video here")
        self.open_btn.setObjectName("chip")
        self.open_btn.clicked.connect(self._open_source)
        self.open_btn.setToolTip(
            "Opens the video on YouTube so you can compare it against the "
            "clip.")
        transport = QHBoxLayout()
        transport.setSpacing(6)
        transport.addWidget(self.play_btn)
        transport.addWidget(self.back_btn)
        transport.addWidget(self.fwd_btn)
        transport.addWidget(self.restart_btn)
        transport.addWidget(self.full_btn)
        transport.addWidget(self.open_btn)
        transport.addWidget(self.window_lbl)
        transport.addStretch(1)

        # One button per segment. Joined into a single file the segments are
        # indistinguishable, so there is no way to tell which part of the song
        # is on screen or to jump back to one.
        self.seg_row = QHBoxLayout()
        self.seg_row.setSpacing(6)
        self.seg_btns: list[QPushButton] = []

        # Nudging the offset. A step is a draft: it rebuilds the clip and
        # writes nothing, so trying a value costs nothing and leaving the song
        # is the same as not having answered. Save locks the song the way
        # `offset` locks it; Undo goes back to the stored value. The clip is
        # rebuilt rather than adjusted because the alignment is half of the
        # clip's cache key already, so a draft is a new file and there is
        # nothing stale to invalidate.
        self.off_row = QHBoxLayout()
        self.off_row.setSpacing(6)
        nudge_lbl = QLabel("Offset")
        nudge_lbl.setObjectName("hint")
        self.off_row.addWidget(nudge_lbl)
        for delta in OFFSET_STEPS:
            b = QPushButton(f"{delta:+d}")
            b.setObjectName("chip")
            b.setToolTip(f"Try the offset {abs(delta)} ms "
                         + ("later" if delta > 0 else "earlier")
                         + ". Nothing is written until you press Save.")
            b.clicked.connect(lambda _, d=delta: self._nudge(d))
            self.off_row.addWidget(b)
        self.save_offset_btn = QPushButton("Save")
        self.save_offset_btn.setObjectName("chip")
        self.save_offset_btn.setToolTip(
            "Lock this song to the offset on screen.")
        self.save_offset_btn.clicked.connect(self._save_offset)
        self.save_offset_btn.setEnabled(False)
        self.off_row.addWidget(self.save_offset_btn)
        self.undo_offset_btn = QPushButton("Undo")
        self.undo_offset_btn.setObjectName("chip")
        self.undo_offset_btn.setToolTip(
            "Go back to the offset this song is stored with.")
        self.undo_offset_btn.clicked.connect(self._undo_offset)
        self.undo_offset_btn.setEnabled(False)
        self.off_row.addWidget(self.undo_offset_btn)
        self.off_row.addStretch(1)

        scrub_row = QHBoxLayout()
        scrub_row.setSpacing(8)
        scrub_row.addWidget(self.scrub, 1)
        scrub_row.addWidget(self.time_lbl)

        self.status = QLabel("")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)

        mono = QFont("Cascadia Mono")
        mono.setStyleHint(QFont.StyleHint.Monospace)
        mono.setPointSize(12)
        self.facts = QLabel("")
        self.facts.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse)
        self.facts.setFont(mono)
        self.facts.setTextFormat(Qt.TextFormat.RichText)
        self.facts.setToolTip(
            "offset - where the video starts relative to the song. Negative "
            "holds the video back; positive skips into it.\n"
            "spread - how much that timing varies across the track. Small is "
            "good; large means no single offset fits.\n"
            "match - how strongly the video's audio matches your chart's. Low "
            "means a different recording, not bad timing.\n"
            "motion - how much the picture moves. Near zero is album art.")

        self.explain = QLabel("")
        self.explain.setObjectName("hint")
        self.explain.setWordWrap(True)

        self.flags = QLabel("")
        self.flags.setObjectName("flags")
        self.flags.setWordWrap(True)

        self.later_btn = QPushButton("Save for later")
        self.later_btn.clicked.connect(lambda: self._act("later"))

        self.drop_btn = QPushButton("No video for this song")
        self.drop_btn.clicked.connect(lambda: self._act("drop"))

        self.keep_btn = QPushButton("Looks right")
        self.keep_btn.setObjectName("keep")
        self.keep_btn.clicked.connect(lambda: self._act("keep"))

        # Approval used to be a one-way door: the only way back out of it was
        # the command line.
        self.unapprove_btn = QPushButton("Un-approve")
        self.unapprove_btn.clicked.connect(lambda: self._act("unapprove"))

        # Only ever on Nothing unusual, where the point of the tile is that
        # nothing in it asked for a song-by-song decision.
        self.approve_all_btn = QPushButton("Approve all of these")
        self.approve_all_btn.setObjectName("keep")
        self.approve_all_btn.clicked.connect(self._approve_all)
        self.approve_all_btn.setVisible(self.mode == "clean")
        self.url = QLineEdit()
        self.url.setPlaceholderText("Paste a better YouTube link")
        self.rep_btn = QPushButton("Use this instead")
        self.rep_btn.clicked.connect(lambda: self._act("replace"))

        acts = QHBoxLayout()
        acts.addWidget(self.keep_btn)
        acts.addWidget(self.unapprove_btn)
        acts.addWidget(self.approve_all_btn)
        acts.addWidget(self.url, 1)
        acts.addWidget(self.rep_btn)
        acts.addWidget(self.later_btn)
        acts.addWidget(self.drop_btn)

        hint = QLabel("Space plays or pauses · Enter keeps and moves on · "
                      "Up and Down change song")
        hint.setObjectName("hint")

        right = QWidget()
        rv_ = QVBoxLayout(right)
        rv_.setContentsMargins(26, 22, 26, 20)
        rv_.setSpacing(10)
        rv_.addWidget(self.head)
        rv_.addWidget(self.by)
        rv_.addWidget(self.video, 1)          # takes the spare vertical space
        rv_.addLayout(self.seg_row)
        rv_.addLayout(self.off_row)
        rv_.addLayout(scrub_row)
        rv_.addLayout(transport)
        rv_.addWidget(self.status)
        rv_.addWidget(self.facts)
        rv_.addWidget(self.explain)
        rv_.addWidget(self.flags)
        rv_.addLayout(acts)
        rv_.addWidget(hint)

        split = QSplitter()
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([360, 820])
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(split)

        # One timer, restarted by every selection: the build starts for
        # whatever song you are on when it finally fires.
        self.build_timer = QTimer(self)
        self.build_timer.setSingleShot(True)
        self.build_timer.timeout.connect(self._start_build)

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, self._on_return)
        # Application-level, not on the widget: QVideoWidget draws into a
        # QWindow of its own inside a container, and a press on that window
        # never reaches the widget's mousePressEvent.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)
        self.refresh()

    # ------------------------------------------------------------------ data
    def _rebuild_chips(self) -> None:
        pool = self._pool(self.mode)
        # Only the doubt tags, because chips only ever appear under Unsure and
        # Unsure is exactly the songs a doubt tag put there.
        counts = Counter(t for s in pool for t in s["tags"]
                         if t in rv.DOUBT_TAGS)
        wanted = [t for t in rv.TAG_ORDER if counts[t]]
        if set(wanted) != set(self.chips):
            while self.chip_bar.count():
                w = self.chip_bar.takeAt(0).widget()
                if w:
                    w.deleteLater()
            self.chips.clear()
            for t in wanted:
                b = QPushButton("")
                b.setObjectName("chip")
                b.setCheckable(True)
                b.toggled.connect(lambda on, tag=t: self._toggle_tag(tag, on))
                self.chips[t] = b
                self.chip_bar.addWidget(b)
            self.chip_bar.addStretch(1)
        for t, b in self.chips.items():
            b.setText(f"{rv.TAG_LABELS[t]} {counts[t]}")
            # `active` is the state and the chips only show it. Setting them
            # unblocked calls back into _toggle_tag, which refreshes, which
            # rebuilds the chips.
            b.blockSignals(True)
            b.setChecked(t in self.active)
            b.blockSignals(False)

    def _pool(self, mode: str) -> list[dict]:
        """
        The songs one tile holds.

        One definition, used for the list and for the number on the tile, and
        `rv.bucket` guarantees the eight of them are a partition: every song
        is in one, no song is in two. The old tabs were neither - the count
        said "to watch" of songs the list put under 'Has a video', and a still
        image on a stranger's channel was in two counts at once.
        """
        return [s for s in self.all_songs if rv.bucket(s) == mode]

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        for tile, btn in self.tiles.items():
            btn.setChecked(tile == mode)
        self.active.clear()
        self._drop_draft()
        self.player.stop()
        self.refresh()
        if self.list.count():
            self.list.setCurrentRow(0)

    def _on_search(self) -> None:
        self.refresh()
        if self.list.count():
            self.list.setCurrentRow(0)

    def _set_sort(self, key: str) -> None:
        """
        One choice with four settings, and a way out of all of them.

        Clicking the sort that is already on clears it, which puts the list
        back in the order `rv.queue` returns - the order of the queue itself,
        which no button could otherwise ask for.
        """
        self.sort = None if key == self.sort else key
        for k, b in self.sorts.items():
            b.setChecked(k == self.sort)
        self.refresh()
        if self.list.count():
            self.list.setCurrentRow(0)

    def _toggle_tag(self, tag: str, on: bool) -> None:
        """
        One filter at a time.

        Two chips at once meant "weak AND third-party", a narrower list than
        either chip promised - clicking a second chip emptied the pane, which
        reads as the filter being broken rather than as an intersection.
        """
        self.active = {tag} if on else set()
        self.refresh()
        if self.list.count():
            self.list.setCurrentRow(0)

    def refresh(self, keep_row: int | None = None) -> None:
        db = Database(self.db_path)
        try:
            self.all_songs = rv.queue(db)
        finally:
            db.close()
        pool = self._pool(self.mode)
        if self.active:
            pool = [s for s in pool if self.active & set(s["tags"])]
        # The song you are looking for by name, whichever of its four names
        # you happen to remember.
        needle = self.search.text().strip().lower()
        if needle:
            pool = [s for s in pool
                    if needle in " ".join((s["artist"], s["title"],
                                           s["video"], s["channel"])).lower()]
        self.songs = pool
        # Approved and Save for later are lists of things you did, and the
        # thing you did last is the one you come back to. Recency wins over
        # the sort control there, whatever it is set to. Sorted twice because
        # Python's sort is stable: names break ties between two decisions
        # made in the same second.
        if self.mode in ("approved", "later"):
            self.songs.sort(key=lambda s: (s["artist"].lower(),
                                           s["title"].lower()))
            self.songs.sort(key=lambda s: s["updated_at"] or "", reverse=True)
        elif self.sort == "artist":
            self.songs.sort(key=lambda s: (s["artist"].lower(),
                                           s["title"].lower()))
        elif self.sort == "title":
            self.songs.sort(key=lambda s: (s["title"].lower(),
                                           s["artist"].lower()))
        elif self.sort == "doubt":
            self.songs.sort(key=lambda s: (-s["risk"], s["artist"].lower(),
                                           s["title"].lower()))
        elif self.sort == "doubt_asc":
            self.songs.sort(key=lambda s: (s["risk"], s["artist"].lower(),
                                           s["title"].lower()))
        # self.sort is None: the order rv.queue returned, untouched.

        # The number on a tile is the size of the tile, not of the filtered
        # list: it is there to say where the work is, and a search that hides
        # eight of nine songs has not finished any of them.
        held = Counter(rv.bucket(s) for s in self.all_songs)
        for tile, btn in self.tiles.items():
            btn.setText(f"{rv.TILE_LABELS[tile]}  {held[tile]}")
        self.chip_holder.setVisible(self.mode == "unsure")
        self.approve_all_btn.setVisible(self.mode == "clean")
        self._rebuild_chips()

        self.list.blockSignals(True)
        self.list.clear()
        for s in self.songs:
            # The video's own title, then who uploaded it. The channel
            # alone answered "is this official?" and nothing else - not which
            # of six uploads of the song this is, which is the question you
            # are actually looking at the row to answer.
            item = QListWidgetItem(
                f"{s['artist']} — {s['title']}\n"
                + (f"{s['video']} · {s['channel']}" if s["channel"]
                   else s["video"]) + "\n"
                + ("checked" if s["review"]
                   else " · ".join(rv.TAG_LABELS[t] for t in s["tags"])))
            if s["review"]:
                item.setForeground(Qt.GlobalColor.darkGray)
            self.list.addItem(item)
        self.list.blockSignals(False)

        shown = len(self.songs)
        held = len(self._pool(self.mode))
        label = rv.TILE_LABELS[self.mode]
        if shown == held:
            self.count.setText(f"{label} — {held} of "
                               f"{len(self.all_songs)} songs")
        else:
            self.count.setText(f"{label} — {shown} of {held} shown, "
                               f"{len(self.all_songs)} songs in all")
        if keep_row is not None and 0 <= keep_row < self.list.count():
            self.list.setCurrentRow(keep_row)

    # --------------------------------------------------------------- display
    def _select(self, row: int) -> None:
        if not (0 <= row < len(self.songs)):
            return
        s = self.songs[row]
        self.current = s["song_dir"]
        # Leaving a song is the same as not having answered: a draft belongs
        # to the song it was typed against and does not follow you.
        self._drop_draft()
        self.player.stop()

        # The song above, the video below. They were run together on one
        # line with the artist in the middle, so the two titles read as one
        # string and it took a second every time to see where the song
        # stopped and the upload started.
        self.head.setText(f"{s['artist']} - {s['title']}")
        self.by.setText(s["video"] + (f" · {s['channel']}"
                                      if s["channel"] else ""))

        self._show_facts(s, s["offset_ms"] or 0)
        self.flags.setText("\n".join("— " + r for r in s["reasons"]))

        # The pre-existing-video notice is one of the reasons printed above
        # now. As a status line it was overwritten by the clip-building text a
        # moment later, so the one place it appeared was the place that could
        # not hold it.
        score = s["fp_score"] or 0
        lines = []
        if s["static"]:
            lines.append("Album art for the whole song, so the offset does not "
                         "matter - nothing on screen moves.")
        else:
            lines.append(
                "Offset is when the video starts relative to the song. "
                "Spread is how much that timing wanders across the track - "
                "under about 35 ms is solid.")
        if score and score < 45:
            lines.append(
                "Match is low (" + str(int(score)) + "), which means the video "
                "carries a DIFFERENT RECORDING than your chart - a re-record, "
                "a live take or a cover. That is about the audio, not the "
                "timing: the video can still line up perfectly by eye. Watch "
                "the clip and trust what you see over this number.")
            if score < 20:
                lines.append(
                    "At this score the spread figure is not evidence either: "
                    "two versions of a song share a beat grid, so the windows "
                    "can agree perfectly on an offset that is still wrong.")
        elif score:
            lines.append(
                "Match " + str(int(score)) + " means the video carries the "
                "same recording as your chart, so the offset is reliable.")
        self.explain.setText(" ".join(lines))

        # A still gets a clip like everything else. Skipping the build
        # saved a couple of seconds and cost the only way to tell an
        # album-art upload from a video the motion figure got wrong - and
        # even a real still has to be looked at to know it is the right
        # artwork for this song.
        self.keep_btn.setText("Keep the still" if s["static"]
                              else "Looks right")
        self.full_btn.setEnabled(True)
        self._queue_build(full=False)

    def _song(self) -> dict | None:
        """The row on screen, or None if the selection is gone."""
        return next((x for x in self.songs if x["song_dir"] == self.current),
                    None)

    def _drop_draft(self) -> None:
        """Forget an unsaved offset and put Save and Undo away with it."""
        self.draft_offset = None
        self.save_offset_btn.setEnabled(False)
        self.undo_offset_btn.setEnabled(False)

    def _show_facts(self, s: dict, off: float) -> None:
        """
        The figures line, rendered at whatever offset is on screen.

        `off` rather than the stored value, because a draft has to show the
        number you would be saving. It is marked as a draft: an offset that
        reads exactly like a saved one is a way to believe you pressed Save.
        """
        moves = ("video waits" if off < 0 else
                 "video skips ahead" if off > 0 else "aligned")
        draft = (f" <span style='color:{DIM}'>draft</span>"
                 if self.draft_offset is not None else "")
        self.facts.setText(
            f"<span style='color:{DIM}'>offset</span> {off:+,.0f} ms "
            f"<span style='color:{DIM}'>{moves}</span>{draft} &nbsp;&nbsp;"
            f"<span style='color:{DIM}'>spread</span> "
            f"{s['spread_text']} &nbsp;&nbsp;"
            f"<span style='color:{DIM}'>match</span> "
            f"{(s['fp_score'] or 0):.0f} &nbsp;&nbsp;"
            f"<span style='color:{DIM}'>motion</span> "
            + ("still image" if s["static"] else f"{s['motion'] or 0:.2f}"))

    def _clear_player(self) -> None:
        """
        Leave nothing of the last song on screen.

        Whatever is loaded describes the song it was built for. Held while the
        next one builds, it reads as this song's clip - the frame, the segment
        buttons and the window text all belong to a different alignment.
        """
        self.player.stop()
        self.player.setSource(QUrl())
        self._clear_segments()
        self.window_lbl.setText("")
        self._files = []
        self._full = False
        self._keyframes = []

    def _queue_build(self, full: bool) -> None:
        """Clear the player now; start the build once the selection settles."""
        self._token += 1
        self._pending_full = full
        self._clear_player()
        self.status.setText("Building the full song - this takes longer…"
                            if full else "Building the clips…")
        self.build_timer.start(BUILD_DELAY_MS)

    def _start_build(self) -> None:
        if self.current is None:
            return
        job = ClipJob(self.db_path, self.work, self.current,
                      full=self._pending_full, token=self._token,
                      offset_ms=self.draft_offset)
        job.signals.done.connect(self._clip_ready)
        self._job = job                 # keep alive until it finishes
        self.pool.start(job)

    def _clip_ready(self, song_dir: str, path: str, error: str,
                    token: int | None = None) -> None:
        # No token means the caller is not a build - the tests drive this
        # directly - so it is trusted. A token that is not the current one is
        # a build for a song, or for an offset, that has since been left
        # behind.
        if token is not None and token != self._token:
            return
        if song_dir != self.current:
            return                                  # user already moved on
        if error:
            # Clear the player before reporting. The failed song used to be
            # described by whatever was still loaded from the last one -
            # frame, segment buttons, window text and all - which reads as a
            # clip that built fine and happens to be wrong.
            self._clear_player()
            self.status.setText(error)
            return
        self.status.setText("")
        self.full_btn.setEnabled(True)
        # Describe what the files actually hold, not what was requested.
        built = Path(path)
        self._full = built.name.startswith("full_")
        info = rv.clip_info(built)
        starts = [] if self._full else (info.get("segments") or [])
        lengths = (info.get("lengths")
                   or [float(rv.SEGMENT_SECONDS)] * len(starts))
        # A sidecar from before the windows were separate files describes one
        # file, which is what it was.
        names = info.get("files") or [built.name]
        self._files = [str(built.with_name(n)) for n in names]
        self._keyframes = info.get("keyframes") or []
        self._show_segments(starts)
        if starts:
            label = " · ".join(f"{ln:.0f}s from {self._mmss(s)}"
                                    for s, ln in zip(starts, lengths))
        elif self._full:
            label = "full song"
        else:
            label = ""
        # Say it outright when the footage and the song do not end together.
        # Fewer windows than usual is the symptom of a short video; this is
        # the cause, and without it a short clip reads as a build that went
        # wrong. The other direction is the same measurement saying that
        # nothing is missing.
        song = next((x for x in self.songs if x["song_dir"] == self.current),
                    None)
        reach = info.get("reach_seconds")
        chart = (song or {}).get("chart_seconds") or 0.0
        if reach is not None and chart and reach < chart - 1.0:
            label += (" · video ends at " + self._mmss(reach)
                      + " of " + self._mmss(chart))
        elif reach is not None and chart and reach > chart + 1.0:
            label += f" · video runs {reach - chart:.0f} s past the song"
        self.window_lbl.setText(label)
        self._play_segment(0)

    def _loop(self, status) -> None:
        # Segments loop: a fifteen-second window gets watched several times
        # over. The full build does not - it has an end, and restarting a
        # four-minute file from the top is not what reaching it means.
        if status == QMediaPlayer.MediaStatus.EndOfMedia and not self._full:
            self.player.setPosition(0)
            self.player.play()

    @staticmethod
    def _mmss(seconds: float) -> str:
        return rv.fmt_mmss(seconds)

    def _open_source(self) -> None:
        s = next((x for x in self.songs if x["song_dir"] == self.current), None)
        if not s or not s.get("video_id"):
            return
        # No timestamp: this opens the video, not a moment in it. The moment
        # it used to name was the first window's, and every button under it
        # now goes somewhere else.
        QDesktopServices.openUrl(QUrl(
            "https://www.youtube.com/watch?v=" + s["video_id"]))

    def _clear_segments(self) -> None:
        while self.seg_row.count():
            w = self.seg_row.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.seg_btns = []

    def _show_segments(self, starts: list[float]) -> None:
        self._clear_segments()
        if len(self._files) < 2:
            return
        for i, start in enumerate(starts[:len(self._files)]):
            b = QPushButton(f"{i + 1}.  {self._mmss(start)}")
            b.setObjectName("chip")
            b.setCheckable(True)
            b.clicked.connect(lambda _, n=i: self._play_segment(n))
            self.seg_btns.append(b)
            self.seg_row.addWidget(b)
        self.seg_row.addStretch(1)

    def _play_segment(self, n: int) -> None:
        """
        Load one window.

        A button is a file now, not a position in a joined one. The join is
        what a short segment used to break, and the button that seeked into an
        index the concat never wrote restarted the clip at zero instead.
        """
        if not (0 <= n < len(self._files)):
            return
        self.player.setSource(QUrl.fromLocalFile(self._files[n]))
        self.player.play()
        for i, b in enumerate(self.seg_btns):
            b.blockSignals(True)
            b.setChecked(i == n)
            b.blockSignals(False)

    def _on_position(self, ms: int) -> None:
        if not self.scrub.isSliderDown():
            self.scrub.setValue(ms)
        total = self.player.duration()
        self.time_lbl.setText(
            self._mmss(ms / 1000) + " / " + self._mmss(total / 1000))

    def _on_scrub_moved(self, ms: int) -> None:
        """
        Follow the drag on a segment; wait for the release on the full build.

        A segment is fifteen seconds of re-encoded h264 and seeks anywhere.
        The full build's video stream was copied, so its keyframes are as much
        as ten seconds apart and every seek in a drag decodes forward to the
        next one - the window locks up for the length of the drag.
        """
        if not self._full:
            self.player.setPosition(ms)

    def _on_scrub_released(self) -> None:
        ms = self.scrub.value()
        if self._full:
            ms = rv.snap_to_keyframe(ms, self._keyframes)
        self.player.setPosition(ms)

    def _skip(self, delta_ms: int) -> None:
        """Ten seconds either way, inside the clip."""
        total = self.player.duration()
        target = self.player.position() + delta_ms
        self.player.setPosition(int(max(0, min(target, total))))

    def _load_full(self) -> None:
        if self.current is None:
            return
        self.full_btn.setEnabled(False)
        self._queue_build(full=True)

    def _restart(self) -> None:
        self.player.setPosition(0)
        self.player.play()

    # ---------------------------------------------------------------- events
    def eventFilter(self, obj, event) -> bool:
        """
        A left click on the picture plays or pauses.

        QVideoWidget renders into a QWindow of its own inside a container
        widget, so the press lands on that window and never reaches any
        widget's mousePressEvent. The filter goes on the application because
        that window is where the event has to be caught.
        """
        try:
            if (event.type() == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton
                    and self._is_video_press(obj, event)):
                self._toggle()
                return True
        except RuntimeError:
            pass          # something went away underneath us
        return super().eventFilter(obj, event)

    def _is_video_press(self, obj, event) -> bool:
        if obj is self.video:
            return True
        handle = self.window().windowHandle()
        if handle is None or not isinstance(obj, QWindow) \
                or obj.parent() is not handle:
            return False
        top_left = self.video.mapToGlobal(QPoint(0, 0))
        return QRect(top_left, self.video.size()).contains(
            event.globalPosition().toPoint())

    def closeEvent(self, event) -> None:
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().closeEvent(event)

    def _sync_play_button(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_btn.setText("Pause" if playing else "Play")

    def _on_return(self) -> None:
        """
        Enter goes where the focus is.

        The window-level shortcut runs before the key reaches the URL box, so
        `returnPressed` never fired there: pressing Enter after pasting a link
        approved the video you were in the middle of replacing.
        """
        if self.url.hasFocus() and self.url.text().strip():
            self._act("replace")
        else:
            self._act("keep")

    def _toggle(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    # --------------------------------------------------------------- actions
    def _nudge(self, delta_ms: int) -> None:
        """
        Move the draft offset by a step and rebuild the clip at it.

        Nothing is written. A step used to lock the song the moment it was
        pressed, which made trying a value and finding out it was wrong a
        thing you had to undo through the database.
        """
        s = self._song()
        if s is None:
            return
        base = (self.draft_offset if self.draft_offset is not None
                else (s["offset_ms"] or 0.0))
        self.draft_offset = float(base + delta_ms)
        self.save_offset_btn.setEnabled(True)
        self.undo_offset_btn.setEnabled(True)
        self._show_facts(s, self.draft_offset)
        self._queue_build(full=False)

    def _save_offset(self) -> None:
        """Lock the song to the draft."""
        if self.draft_offset is None:
            return
        value = self.draft_offset
        # Cleared before the write, so the rebuild the write triggers is the
        # one for the saved value rather than for a draft that no longer
        # differs from it.
        self._drop_draft()
        self._store_offset(value)

    def _undo_offset(self) -> None:
        """Throw the draft away and go back to what is stored."""
        s = self._song()
        if s is None or self.draft_offset is None:
            return
        self._drop_draft()
        self._show_facts(s, s["offset_ms"] or 0.0)
        self._queue_build(full=False)

    def _store_offset(self, offset_ms: float) -> None:
        """
        Write a hand-set offset the way `yargvid offset` writes one.

        The same lock: MANUAL in the note, so `sync --recheck` computes over
        it no more than it does over a hand-picked video, spread zero because
        this offset was not measured across windows, and the review cleared
        because the clip you approved was built at the old one.
        """
        if self.current is None:
            return
        target = self.current
        db = Database(self.db_path)
        try:
            db.update(Path(target), offset_ms=float(offset_ms), spread_ms=0.0,
                      sync_status="ok",
                      sync_note="MANUAL: offset set by hand", review=None)
        finally:
            db.close()
        self.player.stop()
        self.refresh()
        # Reselect the song, not the row. The row number usually does not
        # move, and setCurrentRow to the row already current emits nothing -
        # the clip would then still be the one built at the old offset, which
        # is the one thing this must not leave on screen.
        idx = next((i for i, s in enumerate(self.songs)
                    if s["song_dir"] == target), None)
        if idx is None:
            return
        self.list.blockSignals(True)
        self.list.setCurrentRow(idx)
        self.list.blockSignals(False)
        self._select(idx)

    def _approve_all(self) -> None:
        """
        Approve a whole tile at once.

        Nothing unusual only. That tile exists to hold the songs nothing was
        measured against; every other tile exists because something was, and
        a batch decision there would be a decision not to look.
        """
        if self.mode != "clean":
            return
        pool = self._pool("clean")
        if not pool:
            return
        if QMessageBox.question(
            self, "Approve all of these?",
            f"Mark all {len(pool)} songs in "
            f"“{rv.TILE_LABELS['clean']}” as approved? "
            "Nothing was flagged on any of them; they move to Approved.",
        ) != QMessageBox.StandardButton.Yes:
            return
        db = Database(self.db_path)
        try:
            for s in pool:
                db.update(Path(s["song_dir"]), review="keep")
        finally:
            db.close()
        self.player.stop()
        self.refresh()
        if self.list.count():
            self.list.setCurrentRow(0)

    def _act(self, action: str) -> None:
        if self.current is None:
            return
        row = self.list.currentRow()
        db = Database(self.db_path)
        try:
            if action == "keep":
                db.update(Path(self.current), review="keep")
            elif action == "unapprove":
                db.update(Path(self.current), review=None)
            elif action == "later":
                db.update(Path(self.current), review="later")
            elif action == "drop":
                if QMessageBox.question(
                    self, "Remove this video?",
                    "This song will get no background video.\n\n"
                    "Any downloaded or encoded video in its folder is deleted, "
                    "and matching will not pick it up again.",
                ) != QMessageBox.StandardButton.Yes:
                    return
                rv.drop_song(db, Path(self.current))
            else:
                url = self.url.text().strip()
                if not url:
                    return
                from types import SimpleNamespace

                from .cli import cmd_set
                outcome = cmd_set(SimpleNamespace(pattern=self.current, url=url,
                                                  cookies=None), db)
                self.url.clear()
                if outcome == "same":
                    # Nothing was changed, so nothing downstream needs redoing.
                    # Say so and stay put rather than advancing as if replaced.
                    QMessageBox.information(
                        self, "Already this video",
                        "That link is the video this song already uses.\n\n"
                        "Nothing was changed. Use \u201cLooks right\u201d if "
                        "you are happy with it.")
                    return
                if outcome != "set":
                    # Say which thing failed. Told only that something had,
                    # this box blamed the link for a song it could not find.
                    QMessageBox.warning(
                        self, "Could not use that link", {
                            "not-found": "This song is no longer in the "
                                         "database. Re-run index.",
                            "ambiguous": "More than one song matches this "
                                         "folder, so nothing was changed.",
                        }.get(outcome,
                              "No YouTube video ID could be read from it.\n"
                              "Paste a normal watch or youtu.be link."))
                    return
                QMessageBox.information(
                    self, "Replaced",
                    "Requeued. Run download, sync and encode again for this song.")
        finally:
            db.close()

        self.player.stop()
        self.refresh()

        # The acted-on song either sinks to the bottom (kept) or leaves the
        # queue until it is re-synced (replaced). Either way everything below
        # it shifts up one, so the next song to look at now sits at `row`,
        # not `row + 1` - advancing past it skipped one every time.
        total = len(self.songs)
        nxt = next((i for i in range(row, total)
                    if not self.songs[i]["review"]), None)
        if nxt is None:                      # nothing left below; wrap upward
            nxt = next((i for i in range(0, min(row, total))
                        if not self.songs[i]["review"]), None)
        if nxt is not None:
            self.list.setCurrentRow(nxt)
        elif total:
            self.list.setCurrentRow(min(row, total - 1))


def run(db_path: Path, work: Path) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    db = Database(db_path)
    try:
        n = len(rv.queue(db))
    finally:
        db.close()
    if n == 0:
        QMessageBox.information(
            None, "Nothing to check",
            "No songs have been synced yet.\nRun match, download and sync first.")
        return 0
    win = Window(db_path, work)
    win.show()
    return app.exec()
