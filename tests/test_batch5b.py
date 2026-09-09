"""
Review app, how you work (Batch 5b).

Agreed after the Batch 4/5a review sessions. One song lives in exactly one
tile, assigned by priority; tag chips only appear under Unsure; sorting and
search find a song instead of scrolling for it; the offset can be nudged in
place and the result is locked like `offset` locks it; approval is a
two-way door; and Nothing unusual can be approved as a batch.

    pytest -q tests/test_batch5b.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from yargvid import review as rv               # noqa: E402
from yargvid.db import Database                # noqa: E402

TILES = ["clean", "unsure", "third", "still", "override", "existing",
         "approved", "later"]


def S(**kw) -> dict:
    base = dict(review=None, tags=["clean"], existing_video=None, static=False)
    base.update(kw)
    return base


# ------------------------------------------------- the partition ------------

def test_tile_order_is_agreed():
    assert list(rv.TILES) == TILES
    assert rv.TILE_LABELS["clean"] == "Nothing unusual"
    assert rv.TILE_LABELS["existing"] == "Video already in folder"
    assert rv.TILE_LABELS["override"] == "User override"


def test_every_song_lands_in_exactly_one_tile():
    songs = [
        S(review="keep"), S(review="later"), S(tags=["replaced"]),
        S(existing_video="foreign"), S(static=True, tags=["still"]),
        S(tags=["channel"]), S(tags=["weak"]), S(tags=["clean"]),
    ]
    for s in songs:
        assert rv.bucket(s) in TILES


def test_decisions_outrank_measurements():
    assert rv.bucket(S(review="keep", tags=["channel", "weak"])) == "approved"
    assert rv.bucket(S(review="later", tags=["channel"])) == "later"
    assert rv.bucket(S(tags=["replaced", "weak"])) == "override"
    assert rv.bucket(S(existing_video="foreign", tags=["channel"])) == "existing"


def test_still_beats_third_party_beats_unsure():
    assert rv.bucket(S(static=True, tags=["still", "channel"])) == "still"
    assert rv.bucket(S(tags=["channel", "unverified"])) == "third"
    for tag in ("weak", "unverified", "unsteady", "shift", "drift", "audio",
                "short", "flat"):
        assert rv.bucket(S(tags=[tag])) == "unsure", tag
    assert rv.bucket(S(tags=["clean"])) == "clean"


def test_user_override_is_the_new_name():
    assert rv.TAG_LABELS["replaced"] == "user override"


# ------------------------------------------------- Qt ------------------------

PySide6 = pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _song(db, d, artist, title, **cols):
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


@pytest.fixture
def window(qapp, tmp_path):
    from PySide6.QtTest import QTest
    from yargvid.app import Window
    dbp = tmp_path / "app.sqlite"
    db = Database(dbp)
    _song(db, tmp_path / "clean1", "Beck", "Loser")
    _song(db, tmp_path / "clean2", "ABBA", "Dancing Queen")
    _song(db, tmp_path / "unsure1", "Muse", "Hysteria", fp_score=50.0)
    _song(db, tmp_path / "third1", "Toto", "Africa",
          match_note="Africa (Official Video) [SomeStranger]")
    _song(db, tmp_path / "still1", "Cher", "Believe", motion=0.001)
    _song(db, tmp_path / "over1", "Queen", "Somebody to Love",
          match_note="MANUAL: Somebody to Love [Queen]")
    _song(db, tmp_path / "exist1", "Oasis", "Wonderwall", existing_video="foreign")
    _song(db, tmp_path / "kept1", "Weezer", "Buddy Holly", review="keep")
    _song(db, tmp_path / "later1", "Blur", "Song 2", review="later")
    db.close()
    w = Window(dbp, tmp_path / "work")
    w.show()
    qapp.processEvents()
    yield w, dbp
    w.close()
    QTest.qWait(50)


def _titles(w) -> list[str]:
    return [w.list.item(i).text().splitlines()[0] for i in range(w.list.count())]


def _review(dbp, needle) -> str | None:
    db = Database(dbp)
    try:
        return db.conn.execute("SELECT review FROM songs WHERE song_dir LIKE ?",
                               (f"%{needle}%",)).fetchone()[0]
    finally:
        db.close()


def _row(dbp, needle):
    db = Database(dbp)
    try:
        return dict(db.conn.execute("SELECT * FROM songs WHERE song_dir LIKE ?",
                                    (f"%{needle}%",)).fetchone())
    finally:
        db.close()


def test_tiles_exist_in_order_and_partition_the_queue(window, qapp):
    w, _ = window
    assert list(w.tiles) == TILES
    counts = {}
    for tile in TILES:
        w._set_mode(tile)
        qapp.processEvents()
        counts[tile] = w.list.count()
        assert int(w.tiles[tile].text().split()[-1]) == counts[tile]
    assert sum(counts.values()) == 9
    assert counts == {"clean": 2, "unsure": 1, "third": 1, "still": 1,
                      "override": 1, "existing": 1, "approved": 1, "later": 1}


def test_chips_only_under_unsure(window, qapp):
    w, _ = window
    w._set_mode("clean")
    qapp.processEvents()
    assert not w.chip_holder.isVisibleTo(w)
    w._set_mode("unsure")
    qapp.processEvents()
    assert w.chip_holder.isVisibleTo(w)


def test_sorts(window, qapp):
    w, _ = window
    w._set_mode("clean")
    w._set_sort("artist")
    qapp.processEvents()
    assert _titles(w) == ["ABBA — Dancing Queen", "Beck — Loser"]
    w._set_sort("title")
    qapp.processEvents()
    assert _titles(w) == ["ABBA — Dancing Queen", "Beck — Loser"]  # D before L
    assert "doubt" in w.sorts


def test_search_finds_a_song_by_any_of_its_names(window, qapp):
    w, _ = window
    w._set_mode("clean")
    w.search.setText("loser")
    qapp.processEvents()
    assert _titles(w) == ["Beck — Loser"]
    w.search.setText("abba")
    qapp.processEvents()
    assert _titles(w) == ["ABBA — Dancing Queen"]
    w.search.setText("")
    qapp.processEvents()
    assert len(_titles(w)) == 2


def test_nudging_the_offset_locks_it(window, qapp):
    w, dbp = window
    w._set_mode("clean")
    w.list.setCurrentRow(0)
    qapp.processEvents()
    before = _row(dbp, "clean")["offset_ms"]
    target = w.current
    w._nudge(100)
    qapp.processEvents()
    r = _row(dbp, Path(target).name)
    assert r["offset_ms"] == pytest.approx(before + 100)
    assert (r["sync_note"] or "").startswith("MANUAL")
    assert r["review"] is None
    assert "building" in w.status.text().lower()      # clip rebuilt in place


def test_typing_an_offset_sets_it(window, qapp):
    w, dbp = window
    w._set_mode("clean")
    w.list.setCurrentRow(0)
    qapp.processEvents()
    target = w.current
    w.offset_box.setText("-2500")
    w._apply_offset()
    qapp.processEvents()
    r = _row(dbp, Path(target).name)
    assert r["offset_ms"] == -2500.0
    assert (r["sync_note"] or "").startswith("MANUAL")


def test_unapprove_is_possible(window, qapp):
    w, dbp = window
    w._set_mode("approved")
    w.list.setCurrentRow(0)
    qapp.processEvents()
    assert _review(dbp, "kept1") == "keep"
    w._act("unapprove")
    qapp.processEvents()
    assert _review(dbp, "kept1") is None


def test_batch_approve_nothing_unusual_only(window, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from yargvid import app as appmod
    w, dbp = window
    asked = {}
    monkeypatch.setattr(appmod.QMessageBox, "question",
                        lambda *a, **k: (asked.setdefault("text", a[2]),
                                         QMessageBox.StandardButton.Yes)[1])
    w._set_mode("clean")
    qapp.processEvents()
    w._approve_all()
    qapp.processEvents()
    assert "2" in asked["text"]                      # the count was shown
    assert _review(dbp, "clean1") == "keep"
    assert _review(dbp, "clean2") == "keep"
    assert _review(dbp, "unsure1") is None           # other tiles untouched
    assert _review(dbp, "third1") is None
    w._set_mode("clean")
    qapp.processEvents()
    assert w.list.count() == 0                       # they moved to Approved


def test_batch_approve_is_refused_outside_nothing_unusual(window, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from yargvid import app as appmod
    w, dbp = window
    monkeypatch.setattr(appmod.QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    w._set_mode("third")
    qapp.processEvents()
    w._approve_all()
    qapp.processEvents()
    assert _review(dbp, "third1") is None
