"""
Review app, offset draft and sorts (Batch 6b).

The offset is a draft until you save it: the step buttons rebuild the clip
and write nothing, Save writes what `yargvid offset` writes, Undo goes back
to the stored value, and the typed box is gone. Artist A-Z is the default
sort, clicking the active sort clears it back to queue order, doubt is two
sorts, and Approved / Save for later are always most-recent-first. Header
and facts text can be selected. Low motion is a tag that files a song under
Unsure, not a verdict.

    pytest -q tests/test_batch6b.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from yargvid import encode as enc               # noqa: E402
from yargvid import review as rv                # noqa: E402
from yargvid.db import Database                 # noqa: E402

MANUAL_NOTE = "MANUAL: offset set by hand"
STEPS = (-100, -10, -5, -1, 1, 5, 10, 100)


def S(**kw) -> dict:
    base = dict(review=None, tags=["clean"], existing_video=None, static=False)
    base.update(kw)
    return base


def R(**kw) -> dict:
    """A bare songs row for `assess`."""
    base = dict(song_dir="/lib/x", artist="A", title="T",
                match_note="T video [A]", review=None, fp_score=500.0,
                spread_ms=1.0, sync_status="ok", offset_ms=-1000.0,
                chart_seconds=200.0, video_seconds=400.0, motion=0.5,
                existing_video=None)
    base.update(kw)
    return base


# ------------------------------------------------- low motion, the tag ------

def test_low_motion_band_is_named_and_bounded():
    assert rv.LOW_MOTION_MAX == 0.30
    assert rv.LOW_MOTION_MAX > enc.STATIC_THRESHOLD
    assert rv.TAG_LABELS["low_motion"]
    assert "low_motion" in rv.DOUBT_TAGS


def test_motion_in_the_band_is_low_motion_not_still():
    tags = rv.assess(R(motion=0.15)).tags
    assert "low_motion" in tags
    assert "still" not in tags
    # The band starts where still stops: exactly the threshold is low motion.
    tags = rv.assess(R(motion=enc.STATIC_THRESHOLD)).tags
    assert "low_motion" in tags
    assert "still" not in tags


def test_motion_outside_the_band_is_not_low_motion():
    assert "low_motion" not in rv.assess(R(motion=rv.LOW_MOTION_MAX)).tags
    assert "low_motion" not in rv.assess(R(motion=0.5)).tags
    still = rv.assess(R(motion=0.01)).tags
    assert "still" in still and "low_motion" not in still
    # A failed measurement is neither.
    failed = rv.assess(R(motion=-1.0)).tags
    assert "still" not in failed and "low_motion" not in failed


def test_low_motion_files_under_unsure():
    assert rv.bucket(S(tags=["low_motion"])) == "unsure"
    assert rv.bucket(S(tags=["low_motion", "channel"])) == "third"
    assert rv.bucket(S(static=True, tags=["still"])) == "still"


# ------------------------------------------------- queue carries updated_at -

def test_queue_rows_carry_updated_at(tmp_path):
    db = Database(tmp_path / "q.sqlite")
    d = tmp_path / "s"
    d.mkdir()
    (d / "guitar.ogg").write_bytes(b"")
    db.add_song(d, "A", "T", 200.0)
    db.update(d, sync_status="ok", offset_ms=0.0, spread_ms=1.0,
              match_note="T [A]")
    db.conn.execute("UPDATE songs SET updated_at = '2026-09-01 10:00:00'")
    db.conn.commit()
    rows = rv.queue(db)
    db.close()
    assert rows[0]["updated_at"] == "2026-09-01 10:00:00"


# ------------------------------------------------- Qt ------------------------

PySide6 = pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _song(db, d, artist, title, updated_at=None, **cols):
    d.mkdir(parents=True, exist_ok=True)
    (d / "video.src.mkv").write_bytes(b"x")
    (d / "guitar.ogg").write_bytes(b"")
    db.add_song(d, artist, title, 200.0)
    base = dict(match_status="ok", match_note=f"{title} video [{artist}]",
                download_status="ok", source_path=str(d / "video.src.mkv"),
                sync_status="ok", offset_ms=-1000.0, spread_ms=1.0,
                fp_score=500.0, motion=0.5, video_seconds=400.0)
    base.update(cols)
    db.update(d, **base)
    if updated_at:
        # `update` stamps now; the recency tests need distinct, known times.
        db.conn.execute("UPDATE songs SET updated_at = ? WHERE song_dir = ?",
                        (updated_at, str(d)))
        db.conn.commit()


@pytest.fixture
def window(qapp, tmp_path):
    from PySide6.QtTest import QTest
    from yargvid.app import Window
    dbp = tmp_path / "app.sqlite"
    db = Database(dbp)
    _song(db, tmp_path / "clean1", "Beck", "Loser")
    _song(db, tmp_path / "clean2", "ABBA", "Dancing Queen")
    # Unsure, at four distinct risk levels: weak 4, shift 2, unsteady 1,
    # low motion below all of them.
    _song(db, tmp_path / "weak1", "Muse", "Hysteria", fp_score=50.0)
    _song(db, tmp_path / "shift1", "Air", "Sexy Boy", offset_ms=-20000.0)
    _song(db, tmp_path / "unsteady1", "Zwan", "Honestly", spread_ms=15.0)
    _song(db, tmp_path / "lowmo1", "Cake", "Short Skirt", motion=0.15)
    # Approved and saved, with recency that disagrees with A-Z.
    _song(db, tmp_path / "kept1", "Weezer", "Buddy Holly", review="keep",
          updated_at="2026-09-01 10:00:00")
    _song(db, tmp_path / "kept2", "Blur", "Coffee and TV", review="keep",
          updated_at="2026-09-03 10:00:00")
    _song(db, tmp_path / "kept3", "Pulp", "Common People", review="keep",
          updated_at="2026-09-02 10:00:00")
    _song(db, tmp_path / "later1", "Ash", "Girl From Mars", review="later",
          updated_at="2026-09-01 10:00:00")
    _song(db, tmp_path / "later2", "Suede", "Trash", review="later",
          updated_at="2026-09-02 10:00:00")
    db.close()
    w = Window(dbp, tmp_path / "work")
    w.show()
    qapp.processEvents()
    yield w, dbp
    w.close()
    QTest.qWait(50)


def _titles(w) -> list[str]:
    return [w.list.item(i).text().splitlines()[0] for i in range(w.list.count())]


def _row(dbp, needle):
    db = Database(dbp)
    try:
        return dict(db.conn.execute("SELECT * FROM songs WHERE song_dir LIKE ?",
                                    (f"%{needle}%",)).fetchone())
    finally:
        db.close()


def _pick(w, qapp, mode, row=0):
    w._set_mode(mode)
    w.list.setCurrentRow(row)
    qapp.processEvents()
    return w.current


# ------------------------------------------------- the offset draft ---------

def test_offset_row_is_eight_steps_save_and_undo(window):
    from yargvid import app as appmod
    w, _ = window
    assert appmod.OFFSET_STEPS == STEPS
    assert not hasattr(w, "offset_box")
    assert not hasattr(w, "_apply_offset")
    assert w.save_offset_btn.text() == "Save"
    assert w.undo_offset_btn.text() == "Undo"
    assert w.draft_offset is None
    assert not w.save_offset_btn.isEnabled()
    assert not w.undo_offset_btn.isEnabled()


def test_nudge_is_a_draft_and_writes_nothing(window, qapp):
    w, dbp = window
    target = _pick(w, qapp, "clean")
    before = _row(dbp, Path(target).name)
    w._nudge(100)
    qapp.processEvents()
    assert w.draft_offset == pytest.approx(-900.0)
    assert w.save_offset_btn.isEnabled()
    assert w.undo_offset_btn.isEnabled()
    assert "-900" in w.facts.text()                    # what you would save
    assert "building" in w.status.text().lower()      # clip rebuilt at draft
    after = _row(dbp, Path(target).name)
    assert after["offset_ms"] == before["offset_ms"]
    assert after["sync_note"] == before["sync_note"]
    assert after["review"] == before["review"]
    assert after["updated_at"] == before["updated_at"]


def test_nudges_accumulate_on_the_draft(window, qapp):
    w, _ = window
    _pick(w, qapp, "clean")
    w._nudge(10)
    w._nudge(5)
    w._nudge(-1)
    qapp.processEvents()
    assert w.draft_offset == pytest.approx(-986.0)


def test_save_writes_what_cmd_offset_writes_and_locks(window, qapp):
    w, dbp = window
    target = _pick(w, qapp, "clean")
    w._nudge(-100)
    w._nudge(-100)
    qapp.processEvents()
    w._save_offset()
    qapp.processEvents()
    r = _row(dbp, Path(target).name)
    assert r["offset_ms"] == -1200.0
    assert r["spread_ms"] == 0.0
    assert r["sync_status"] == "ok"
    assert r["sync_note"] == MANUAL_NOTE
    assert r["review"] is None
    assert w.current == target                        # still on the song
    assert w.draft_offset is None
    assert not w.save_offset_btn.isEnabled()
    assert not w.undo_offset_btn.isEnabled()
    assert "-1,200" in w.facts.text()


def test_save_clears_an_approval(window, qapp):
    w, dbp = window
    target = _pick(w, qapp, "approved")
    w._nudge(1)
    qapp.processEvents()
    w._save_offset()
    qapp.processEvents()
    r = _row(dbp, Path(target).name)
    assert r["review"] is None
    assert r["sync_note"] == MANUAL_NOTE
    assert r["offset_ms"] == -999.0


def test_save_without_a_draft_writes_nothing(window, qapp):
    w, dbp = window
    target = _pick(w, qapp, "approved")
    before = _row(dbp, Path(target).name)
    w._save_offset()
    qapp.processEvents()
    assert _row(dbp, Path(target).name) == before


def test_undo_reverts_to_the_stored_value(window, qapp):
    w, dbp = window
    target = _pick(w, qapp, "clean")
    before = _row(dbp, Path(target).name)
    w._nudge(100)
    w._nudge(10)
    qapp.processEvents()
    w.status.setText("")
    w._undo_offset()
    qapp.processEvents()
    assert w.draft_offset is None
    assert not w.save_offset_btn.isEnabled()
    assert not w.undo_offset_btn.isEnabled()
    assert "-1,000" in w.facts.text()
    assert "building" in w.status.text().lower()      # clip rebuilt at stored
    assert _row(dbp, Path(target).name) == before


def test_leaving_the_song_drops_the_draft(window, qapp):
    w, dbp = window
    target = _pick(w, qapp, "clean")
    w._nudge(100)
    qapp.processEvents()
    w.list.setCurrentRow(1)
    qapp.processEvents()
    assert w.current != target
    assert w.draft_offset is None
    assert not w.save_offset_btn.isEnabled()
    assert _row(dbp, Path(target).name)["offset_ms"] == -1000.0
    w.list.setCurrentRow(0)
    qapp.processEvents()
    assert w.current == target
    assert w.draft_offset is None
    assert "-1,000" in w.facts.text()


def test_build_is_started_at_the_draft_offset(window, qapp, monkeypatch):
    from yargvid import app as appmod
    w, _ = window
    seen = []

    class Spy(appmod.ClipJob):
        def __init__(self, *a, **k):
            seen.append(k.get("offset_ms"))
            super().__init__(*a, **k)

        def run(self):
            pass

    monkeypatch.setattr(appmod, "ClipJob", Spy)
    _pick(w, qapp, "clean")
    w._start_build()
    assert seen[-1] is None                           # stored value: no override
    w._nudge(5)
    w._start_build()
    assert seen[-1] == pytest.approx(-995.0)
    w._undo_offset()
    w._start_build()
    assert seen[-1] is None


def test_clip_job_builds_the_row_at_the_override(window, monkeypatch):
    from yargvid import app as appmod
    w, dbp = window
    built = []

    def fake_build(row, work):
        built.append(row["offset_ms"])
        return Path(work) / "clip.mp4"

    monkeypatch.setattr(appmod.rv, "build_clip", fake_build)
    song = _row(dbp, "clean1")["song_dir"]
    appmod.ClipJob(dbp, w.work, song).run()
    appmod.ClipJob(dbp, w.work, song, offset_ms=-950.0).run()
    assert built == [-1000.0, -950.0]
    assert _row(dbp, "clean1")["offset_ms"] == -1000.0


# ------------------------------------------------- sorts --------------------

def test_artist_az_is_the_default(window):
    w, _ = window
    assert w.sort == "artist"
    assert w.sorts["artist"].isChecked()
    assert [k for k, b in w.sorts.items() if b.isChecked()] == ["artist"]


def test_doubt_is_two_sorts(window, qapp):
    w, _ = window
    assert {"artist", "title", "doubt", "doubt_asc"} <= set(w.sorts)
    w._set_mode("unsure")
    w._set_sort("doubt")
    qapp.processEvents()
    assert _titles(w) == ["Muse — Hysteria", "Air — Sexy Boy",
                          "Zwan — Honestly", "Cake — Short Skirt"]
    w._set_sort("doubt_asc")
    qapp.processEvents()
    assert _titles(w) == ["Cake — Short Skirt", "Zwan — Honestly",
                          "Air — Sexy Boy", "Muse — Hysteria"]
    assert w.sorts["doubt_asc"].isChecked()
    assert not w.sorts["doubt"].isChecked()


def test_clicking_the_active_sort_clears_it(window, qapp):
    w, _ = window
    w._set_mode("unsure")
    qapp.processEvents()
    assert w.sort == "artist"                         # the default is active
    assert _titles(w)[0] == "Air — Sexy Boy"
    w._set_sort("artist")                             # clicking it clears it
    qapp.processEvents()
    assert w.sort is None
    assert not any(b.isChecked() for b in w.sorts.values())
    # Queue order: what rv.queue returns, most doubtful first.
    assert _titles(w) == ["Muse — Hysteria", "Air — Sexy Boy",
                          "Zwan — Honestly", "Cake — Short Skirt"]
    w._set_sort("title")
    qapp.processEvents()
    assert w.sort == "title"
    assert _titles(w) == ["Zwan — Honestly", "Muse — Hysteria",
                          "Air — Sexy Boy", "Cake — Short Skirt"]


def test_approved_and_later_are_most_recent_first(window, qapp):
    w, _ = window
    for mode, want in (("approved", ["Blur — Coffee and TV",
                                     "Pulp — Common People",
                                     "Weezer — Buddy Holly"]),
                       ("later", ["Suede — Trash", "Ash — Girl From Mars"])):
        w._set_mode(mode)
        for key in ("artist", "title", "doubt", "doubt_asc"):
            w._set_sort(key)
            qapp.processEvents()
            assert _titles(w) == want, (mode, key)
        w._set_sort(w.sort)                           # cleared
        qapp.processEvents()
        assert w.sort is None
        assert _titles(w) == want, (mode, "cleared")


def test_approving_moves_the_song_to_the_top_of_approved(window, qapp):
    w, _ = window
    # Cake sorts after Blur, so only recency can put it first.
    cake = next(s for s in w.all_songs if s["artist"] == "Cake")
    w._set_mode(rv.bucket(cake))
    w._set_sort("artist")
    qapp.processEvents()
    w.list.setCurrentRow(_titles(w).index("Cake — Short Skirt"))
    qapp.processEvents()
    w._act("keep")
    qapp.processEvents()
    w._set_mode("approved")
    qapp.processEvents()
    assert _titles(w)[0] == "Cake — Short Skirt"


# ------------------------------------------------- selectable text ----------

def test_header_and_facts_are_selectable(window, qapp):
    from PySide6.QtCore import Qt
    w, _ = window
    _pick(w, qapp, "clean")
    flag = Qt.TextInteractionFlag.TextSelectableByMouse
    for lbl in (w.head, w.by, w.facts):
        assert lbl.textInteractionFlags() & flag, lbl.objectName()


# ------------------------------------------------- low motion in the app ----

def test_low_motion_song_shows_under_unsure_with_a_chip(window, qapp):
    w, _ = window
    w._set_mode("unsure")
    qapp.processEvents()
    assert "Cake — Short Skirt" in _titles(w)
    assert "low_motion" in w.chips
    w._set_mode("still")
    qapp.processEvents()
    assert w.list.count() == 0


# ------------------------------------------------- follow-up: Save that ------
# ------------------------------------------------- leaves the tile ----------

@pytest.mark.parametrize("mode", ["approved", "later"])
def test_save_follows_the_song_to_its_new_tile(window, qapp, mode):
    """
    Save clears the review, so a song saved from Approved or Save for later
    leaves that tile. The app used to stop there: nothing selected, no clip,
    the old song's text still on screen.
    """
    w, dbp = window
    target = _pick(w, qapp, mode)
    was = w.list.count()
    w._nudge(1)
    qapp.processEvents()
    w.status.setText("")
    w._save_offset()
    qapp.processEvents()
    song = next(s for s in w.all_songs if s["song_dir"] == target)
    assert song["review"] is None
    assert w.mode == rv.bucket(song) == "clean"
    assert w.tiles["clean"].isChecked()
    assert not w.tiles[mode].isChecked()
    assert w.current == target
    assert w.songs[w.list.currentRow()]["song_dir"] == target
    assert w.draft_offset is None
    assert "-999" in w.facts.text()
    assert "building" in w.status.text().lower()
    w._set_mode(mode)
    qapp.processEvents()
    assert w.list.count() == was - 1
