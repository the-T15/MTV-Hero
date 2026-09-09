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

from PySide6.QtCore import (QObject, QRunnable, Qt, QThreadPool, QUrl, Signal,
                            Slot)
from PySide6.QtGui import (QDesktopServices, QFont, QKeySequence,
                           QShortcut)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (QApplication, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMessageBox,
                               QPushButton, QSizePolicy, QSlider, QSplitter,
                               QVBoxLayout, QWidget)
from collections import Counter

from . import review as rv
from .db import Database

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
    done = Signal(str, str, str)          # song_dir, path, error


class ClipJob(QRunnable):
    """Build one proof clip off the UI thread."""

    def __init__(self, db_path: Path, work: Path, song_dir: str,
                 full: bool = False):
        super().__init__()
        self.db_path, self.work, self.song_dir = db_path, work, song_dir
        self.full = full
        self.signals = ClipSignals()

    def _emit(self, path: str, error: str) -> None:
        try:
            self.signals.done.emit(self.song_dir, path, error)
        except RuntimeError:
            pass          # the window moved on and dropped this job

    @Slot()
    def run(self):
        db = Database(self.db_path)
        try:
            row = db.conn.execute(
                "SELECT * FROM songs WHERE song_dir = ?", (self.song_dir,)
            ).fetchone()
            if row is None:
                return self._emit("", "Song not found.")
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


class Window(QWidget):
    def __init__(self, db_path: Path, work: Path):
        super().__init__()
        self.db_path, self.work = db_path, work
        self.pool = QThreadPool.globalInstance()
        self.all_songs: list[dict] = []
        self.songs: list[dict] = []
        self.active: set[str] = set()
        self.mode = "watch"
        self.sort = "doubt"
        self.chips: dict[str, QPushButton] = {}
        self.current: str | None = None

        self.setWindowTitle("Backgrounds to check")
        self.resize(1180, 760)
        self.setStyleSheet(STYLE)

        # --- left: the worklist ------------------------------------------
        self.count = QLabel()
        self.count.setObjectName("by")
        self.count.setContentsMargins(14, 12, 14, 8)
        self.list = QListWidget()
        self.list.currentRowChanged.connect(self._select)

        # Filter chips. Selecting none shows everything; selecting several
        # shows anything carrying at least one of them, so "third-party" plus
        # "weak match" answers "what might simply be the wrong video".
        # Two lists, not one. A still image needs a keep-or-replace decision,
        # not twenty seconds of watching, and mixing them in buries the songs
        # that do need an eye on them.
        self.tab_watch = QPushButton("To watch")
        self.tab_still = QPushButton("Still images")
        self.tab_later = QPushButton("Saved for later")
        self.tab_existing = QPushButton("Has a video")
        for btn, mode in ((self.tab_watch, "watch"), (self.tab_still, "still"),
                          (self.tab_existing, "existing"),
                          (self.tab_later, "later")):
            btn.setObjectName("tab")
            btn.setCheckable(True)
            btn.clicked.connect(lambda _, m=mode: self._set_mode(m))
        self.tab_watch.setChecked(True)
        tabs = QHBoxLayout()
        tabs.setContentsMargins(14, 0, 14, 8)
        tabs.setSpacing(6)
        tabs.addWidget(self.tab_watch)
        tabs.addWidget(self.tab_still)
        tabs.addWidget(self.tab_existing)
        tabs.addWidget(self.tab_later)
        tabs.addStretch(1)

        # Sort order. Weakest-match-first is the useful one for working
        # through hand-picked videos, where a low score means a different
        # recording and the clip has to be judged by eye.
        self.sorts: dict[str, QPushButton] = {}
        for key, label in (("doubt", "Most doubtful"),
                           ("match_asc", "Match \u2191"),
                           ("match_desc", "Match \u2193")):
            b = QPushButton(label)
            b.setObjectName("chip")
            b.setCheckable(True)
            b.setChecked(key == "doubt")
            b.clicked.connect(lambda _, k=key: self._set_sort(k))
            self.sorts[key] = b
            tabs.addWidget(b)
        tab_holder = QWidget()
        tab_holder.setLayout(tabs)

        self.chip_bar = QHBoxLayout()
        self.chip_bar.setContentsMargins(14, 0, 14, 10)
        self.chip_bar.setSpacing(6)
        chip_holder = QWidget()
        chip_holder.setLayout(self.chip_bar)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(0)
        lv.addWidget(self.count)
        lv.addWidget(tab_holder)
        lv.addWidget(chip_holder)
        lv.addWidget(self.list, 1)

        # --- right: player and detail -------------------------------------
        self.head = QLabel("Pick a song on the left")
        self.head.setObjectName("head")
        self.by = QLabel("The most doubtful ones are at the top.")
        self.by.setObjectName("by")
        self.by.setWordWrap(True)

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
        # Without the timestamps there is no way to check the clip against the
        # source, which leaves the offset unverifiable outside the game.
        self.window_lbl = QLabel("")
        self.window_lbl.setObjectName("hint")
        self.scrub = QSlider(Qt.Orientation.Horizontal)
        self.scrub.setRange(0, 0)
        self.scrub.sliderMoved.connect(self.player.setPosition)
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
            "Opens the video on YouTube at the same moment the clip shows, so "
            "you can compare directly.")
        transport = QHBoxLayout()
        transport.setSpacing(6)
        transport.addWidget(self.play_btn)
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
        self.url = QLineEdit()
        self.url.setPlaceholderText("Paste a better YouTube link")
        self.rep_btn = QPushButton("Use this instead")
        self.rep_btn.clicked.connect(lambda: self._act("replace"))

        acts = QHBoxLayout()
        acts.addWidget(self.keep_btn)
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

        QShortcut(QKeySequence(Qt.Key.Key_Space), self, self._toggle)
        QShortcut(QKeySequence(Qt.Key.Key_Return), self, self._on_return)
        self.refresh()

    # ------------------------------------------------------------------ data
    def _rebuild_chips(self) -> None:
        pool = self._pool(self.mode)
        skip = {"still"} if self.mode in ("watch", "still") else set()
        counts = Counter(t for s in pool for t in s["tags"] if t not in skip)
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
        The songs a tab holds.

        One definition, used for both the list and the number on the tab. Two
        of them disagreed: the count said "to watch" of songs the list put in
        'Has a video', so the tab promised work that was not there.
        """
        if mode == "later":
            return [s for s in self.all_songs if s["review"] == "later"]
        if mode == "existing":
            # Songs that already had a video before this pipeline encoded
            # anything. Encoding replaces it, so these are worth comparing
            # rather than overwriting unseen.
            return [s for s in self.all_songs
                    if s.get("existing_video") and s["review"] != "later"]
        want_still = mode == "still"
        return [s for s in self.all_songs
                if bool(s["static"]) == want_still
                and s["review"] != "later"
                and not s.get("existing_video")]

    def _todo(self, mode: str) -> int:
        """
        How many songs in a tab are still waiting on a decision.

        The list keeps showing songs you have kept - at the bottom, so the
        order is stable and you can go back to one. The number on the tab is
        not for that: it answers "how much is left", and counting work already
        done meant every tab stayed the same size however long you worked.

        'later' is not a decision, it is a deferral, so those still count.
        """
        return sum(1 for s in self._pool(mode) if s["review"] != "keep")

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        self.tab_watch.setChecked(mode == "watch")
        self.tab_still.setChecked(mode == "still")
        self.tab_later.setChecked(mode == "later")
        self.tab_existing.setChecked(mode == "existing")
        self.active.clear()
        self.player.stop()
        self.refresh()
        if self.list.count():
            self.list.setCurrentRow(0)

    def _set_sort(self, key: str) -> None:
        self.sort = key
        for k, b in self.sorts.items():
            b.setChecked(k == key)
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
        self.songs = ([s for s in pool if self.active & set(s["tags"])]
                      if self.active else pool)
        # Reviewed songs stay at the bottom whatever the order.
        if self.sort == "match_asc":
            self.songs.sort(key=lambda s: (s["review"] is not None,
                                           s["fp_score"] or 0, s["artist"]))
        elif self.sort == "match_desc":
            self.songs.sort(key=lambda s: (s["review"] is not None,
                                           -(s["fp_score"] or 0), s["artist"]))
        self.tab_watch.setText(f"To watch  {self._todo('watch')}")
        self.tab_still.setText(f"Still images  {self._todo('still')}")
        self.tab_later.setText(f"Saved for later  {self._todo('later')}")
        self.tab_existing.setText(f"Has a video  {self._todo('existing')}")
        self._rebuild_chips()

        self.list.blockSignals(True)
        self.list.clear()
        for s in self.songs:
            item = QListWidgetItem(
                f"{s['artist']} — {s['title']}\n"
                f"{s['channel'] or s['video']}\n"
                + ("checked" if s["review"]
                   else " · ".join(rv.TAG_LABELS[t] for t in s["tags"])))
            if s["review"]:
                item.setForeground(Qt.GlobalColor.darkGray)
            self.list.addItem(item)
        self.list.blockSignals(False)

        todo = sum(1 for s in self.songs if not s["review"])
        if self.active:
            self.count.setText(
                f"{todo} left of {len(self.songs)} shown "
                f"({len(self.all_songs)} total)")
        else:
            self.count.setText(f"{todo} left of {len(self.songs)}, "
                               f"most doubtful first")
        if keep_row is not None and 0 <= keep_row < self.list.count():
            self.list.setCurrentRow(keep_row)

    # --------------------------------------------------------------- display
    def _select(self, row: int) -> None:
        if not (0 <= row < len(self.songs)):
            return
        s = self.songs[row]
        self.current = s["song_dir"]
        self.player.stop()

        self.head.setText(s["title"])
        self.by.setText(f"{s['artist']} · {s['video']}"
                        + (f" · {s['channel']}" if s["channel"] else ""))

        off = s["offset_ms"] or 0
        moves = ("video waits" if off < 0 else
                 "video skips ahead" if off > 0 else "aligned")
        self.facts.setText(
            f"<span style='color:{DIM}'>offset</span> {off:+,.0f} ms "
            f"<span style='color:{DIM}'>{moves}</span> &nbsp;&nbsp;"
            f"<span style='color:{DIM}'>spread</span> "
            f"{s['spread_text']} &nbsp;&nbsp;"
            f"<span style='color:{DIM}'>match</span> "
            f"{(s['fp_score'] or 0):.0f} &nbsp;&nbsp;"
            f"<span style='color:{DIM}'>motion</span> "
            + ("still image" if s["static"] else f"{s['motion'] or 0:.2f}"))
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
        self.status.setText("Building a clip from the middle of the song…")
        self.full_btn.setEnabled(True)
        self._clear_segments()
        self.window_lbl.setText("")
        job = ClipJob(self.db_path, self.work, s["song_dir"])
        job.signals.done.connect(self._clip_ready)
        self._job = job                 # keep alive until it finishes
        self.pool.start(job)

    @Slot(str, str, str)
    def _clip_ready(self, song_dir: str, path: str, error: str) -> None:
        if song_dir != self.current:
            return                                  # user already moved on
        if error:
            # Clear the player before reporting. The failed song used to be
            # described by whatever was still loaded from the last one -
            # frame, segment buttons, window text and all - which reads as a
            # clip that built fine and happens to be wrong.
            self.player.stop()
            self.player.setSource(QUrl())
            self._clear_segments()
            self.window_lbl.setText("")
            self.status.setText(error)
            return
        self.status.setText("")
        self.full_btn.setEnabled(True)
        # Describe what the file actually holds, not what was requested.
        full = Path(path).name.startswith("full_")
        info = rv.clip_info(Path(path))
        starts = [] if full else (info.get("segments") or [])
        self._show_segments(starts)
        if starts:
            label = (f"{len(starts)} x {rv.SEGMENT_SECONDS}s from song "
                     + ", ".join(self._mmss(x) for x in starts))
        elif full:
            label = "full song"
        else:
            label = ""
        # Say it outright when the footage runs out before the song does.
        # Fewer windows than usual is the symptom; this is the cause, and
        # without it a short clip reads as a build that went wrong.
        song = next((x for x in self.songs if x["song_dir"] == self.current),
                    None)
        reach = info.get("reach_seconds")
        chart = (song or {}).get("chart_seconds") or 0.0
        if reach is not None and chart and reach < chart - 1.0:
            label += (" · video ends at " + self._mmss(reach)
                      + " of " + self._mmss(chart))
        self.window_lbl.setText(label)
        self.player.setSource(QUrl.fromLocalFile(path))
        self.player.play()

    def _loop(self, status) -> None:
        if status == QMediaPlayer.MediaStatus.EndOfMedia and \
                not self.player.source().toLocalFile().split("/")[-1] \
                        .startswith("full_"):
            self.player.setPosition(0)
            self.player.play()

    @staticmethod
    def _mmss(seconds: float) -> str:
        s = int(round(seconds))
        return f"{s // 60}:{s % 60:02d}"

    def _open_source(self) -> None:
        s = next((x for x in self.songs if x["song_dir"] == self.current), None)
        if not s or not s.get("video_id"):
            return
        at = int(s.get("clip_video_s") or 0)
        QDesktopServices.openUrl(QUrl(
            "https://www.youtube.com/watch?v=" + s["video_id"] + "&t=" + str(at) + "s"))

    def _clear_segments(self) -> None:
        while self.seg_row.count():
            w = self.seg_row.takeAt(0).widget()
            if w:
                w.deleteLater()
        self.seg_btns = []

    def _show_segments(self, starts: list[float]) -> None:
        self._clear_segments()
        if len(starts) < 2:
            return
        for i, start in enumerate(starts):
            b = QPushButton(f"{i + 1}.  {self._mmss(start)}")
            b.setObjectName("chip")
            b.setCheckable(True)
            b.clicked.connect(
                lambda _, n=i: self.player.setPosition(
                    int(n * rv.SEGMENT_SECONDS * 1000)))
            self.seg_btns.append(b)
            self.seg_row.addWidget(b)
        self.seg_row.addStretch(1)

    def _on_position(self, ms: int) -> None:
        if not self.scrub.isSliderDown():
            self.scrub.setValue(ms)
        total = self.player.duration()
        self.time_lbl.setText(
            self._mmss(ms / 1000) + " / " + self._mmss(total / 1000))
        if self.seg_btns:
            idx = min(int(ms / (rv.SEGMENT_SECONDS * 1000)),
                      len(self.seg_btns) - 1)
            for i, b in enumerate(self.seg_btns):
                b.setChecked(i == idx)

    def _load_full(self) -> None:
        if self.current is None:
            return
        self.player.stop()
        self.status.setText("Building the full song - this takes longer…")
        self.full_btn.setEnabled(False)
        job = ClipJob(self.db_path, self.work, self.current, full=True)
        job.signals.done.connect(self._clip_ready)
        self._job = job
        self.pool.start(job)

    def _restart(self) -> None:
        self.player.setPosition(0)
        self.player.play()

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
    def _act(self, action: str) -> None:
        if self.current is None:
            return
        row = self.list.currentRow()
        db = Database(self.db_path)
        try:
            if action == "keep":
                db.update(Path(self.current), review="keep")
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
