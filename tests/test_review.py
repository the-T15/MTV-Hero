"""
Review-app tests (Batch 2). Pure-Python parts run everywhere; the Qt parts
run offscreen and are skipped if PySide6 is not installed.

    pytest -q tests/test_review.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from yargvid import cli                      # noqa: E402
from yargvid import match as mt              # noqa: E402
from yargvid import review as rv             # noqa: E402
from yargvid.db import Database              # noqa: E402


# ----------------------------------------------------------------- helpers ---

class R(dict):
    """sqlite3.Row stand-in: missing keys read as None."""

    def __getitem__(self, k):
        return dict.get(self, k)


@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


def song(db, path, **cols):
    Path(path).mkdir(parents=True, exist_ok=True)
    db.add_song(Path(path), "Artist", Path(path).name, 100.0)
    if cols:
        db.update(Path(path), **cols)
    return Path(path)


def row(db, path):
    return db.conn.execute("SELECT * FROM songs WHERE song_dir = ?",
                           (str(path),)).fetchone()


def synced(db, tmp_path, name, **cols):
    d = tmp_path / name
    src = d / "video.src.mkv"
    d.mkdir(parents=True, exist_ok=True)
    src.write_bytes(b"x" * 100)
    (d / "guitar.ogg").write_bytes(b"")
    base = dict(match_status="ok", match_note="T [U]", download_status="ok",
                source_path=str(src), sync_status="ok", offset_ms=0.0,
                spread_ms=1.0, fp_score=500.0, motion=0.5)
    base.update(cols)
    return song(db, d, **base)


# -------------------------------------------------- A9: clip cache naming ----

def _clip_row(tmp_path, offset=-1000.0):
    src = tmp_path / "s" / "video.src.mkv"
    src.parent.mkdir(exist_ok=True)
    if not src.exists():
        src.write_bytes(b"x" * 100)
    return R(song_dir=str(tmp_path / "s"), offset_ms=offset,
             source_path=str(src), chart_seconds=200.0)


def test_A9_clip_name_is_stable_across_processes(tmp_path):
    r = _clip_row(tmp_path)
    here = rv.clip_name(r)
    code = (
        "import json, sys\n"
        "from yargvid import review as rv\n"
        "class R(dict):\n"
        "    def __getitem__(self, k): return dict.get(self, k)\n"
        "print(rv.clip_name(R(json.loads(sys.argv[1]))))\n"
    )
    there = subprocess.run([sys.executable, "-c", code, json.dumps(dict(r))],
                           capture_output=True, text=True, check=True).stdout.strip()
    assert here == there


def test_A9_clip_name_changes_with_offset_and_source(tmp_path):
    a = rv.clip_name(_clip_row(tmp_path, -1000.0))
    b = rv.clip_name(_clip_row(tmp_path, -1500.0))
    assert a != b
    src = tmp_path / "s" / "video.src.mkv"
    src.write_bytes(b"y" * 250)               # replaced video, different size
    c = rv.clip_name(_clip_row(tmp_path, -1000.0))
    assert c != a


def test_A9_stale_clips_for_the_same_song_are_removed(tmp_path):
    r = _clip_row(tmp_path)
    work = tmp_path / "work"
    work.mkdir()
    keep = work / rv.clip_name(r)
    keep.write_bytes(b"new")
    prefix = keep.name.split("_")[1]           # the song part of the name
    stale = work / f"clip_{prefix}_deadbeef.mp4"
    stale.write_bytes(b"old")
    stale_side = work / f"clip_{prefix}_deadbeef.segments.json"
    stale_side.write_text("[]")
    other = work / "clip_00000000_00000000.mp4"   # a different song
    other.write_bytes(b"keep me")
    rv.purge_stale(work, keep)
    assert keep.exists() and other.exists()
    assert not stale.exists() and not stale_side.exists()


# ------------------------------------------------------- D4: spread text ----

def test_D4_unverified_spread_reads_as_unverified():
    assert rv.fmt_ms(-1.0) == "unverified"
    assert rv.fmt_ms(None) == "unverified"
    assert rv.fmt_ms(3.4) == "3 ms"


def test_D4_queue_carries_spread_text(db, tmp_path):
    synced(db, tmp_path, "unv", sync_status="unverified", spread_ms=-1.0)
    q = rv.queue(db)
    assert q[0]["spread_text"] == "unverified"


# -------------------------------------------------------- D5: drop_song -----

def test_D5_drop_clears_every_measurement(db, tmp_path):
    d = synced(db, tmp_path, "drop", fp_score=80.0, dominance=3.0,
               drift_ppm=5.0, sync_note="n", windows=7, encode_note="e",
               review="keep", video_id="aaaaaaaaaaa")
    (d / "video.webm").write_bytes(b"enc")
    rv.drop_song(db, d)
    r = row(db, d)
    assert r["match_status"] == "skipped"
    assert r["match_note"].startswith("MANUAL")
    for col in ("video_id", "source_path", "offset_ms", "spread_ms", "motion",
                "fp_score", "dominance", "drift_ppm", "sync_note", "windows",
                "encode_note", "review"):
        assert r[col] is None, col
    assert not (d / "video.webm").exists()
    assert not (d / "video.src.mkv").exists()


# ---------------------------------------------- A11: cmd_set outcomes -------

@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setattr(mt, "fetch_metadata",
                        lambda vid, cookies=None: {"title": "T", "uploader": "U"})


def _set(db, pattern, url):
    return cli.cmd_set(SimpleNamespace(pattern=pattern, url=url, cookies=None), db)


def test_A11_cmd_set_reports_why_it_failed(db, tmp_path, no_network):
    song(db, tmp_path / "Foo", match_status="ok", video_id="aaaaaaaaaaa")
    song(db, tmp_path / "Foo (Live)", match_status="ok", video_id="aaaaaaaaaaa")
    assert _set(db, "nothing-like-this", "bbbbbbbbbbb") == "not-found"
    assert _set(db, "Foo", "bbbbbbbbbbb") == "ambiguous"
    assert _set(db, str(tmp_path / "Foo"), "not a link") == "bad-url"
    assert _set(db, str(tmp_path / "Foo"), "aaaaaaaaaaa") == "same"
    assert _set(db, str(tmp_path / "Foo"), "bbbbbbbbbbb") == "set"


# ------------------------------------------------- D6: browser server -------

@pytest.fixture
def server(tmp_path):
    dbp = tmp_path / "srv.sqlite"
    d = Database(dbp)
    synced(d, tmp_path, "one")
    d.close()
    srv = rv.make_server(dbp, tmp_path / "work", port=0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv, dbp, str(tmp_path / "one")
    srv.shutdown()


def _post(srv, path, body: bytes):
    port = srv.server_address[1]
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body,
                                 method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code


def test_D6_post_to_unknown_path_is_404(server):
    srv, _, song_dir = server
    assert _post(srv, "/nope", json.dumps({"song": song_dir, "action": "keep"}).encode()) == 404


def test_D6_malformed_json_is_400_not_a_traceback(server):
    srv, _, _ = server
    assert _post(srv, "/api/act", b"{not json") == 400


def test_D6_unknown_action_is_400(server):
    srv, _, song_dir = server
    assert _post(srv, "/api/act", json.dumps({"song": song_dir, "action": "explode"}).encode()) == 400


def test_D6_keep_still_works(server):
    srv, dbp, song_dir = server
    assert _post(srv, "/api/act", json.dumps({"song": song_dir, "action": "keep"}).encode()) == 200
    d = Database(dbp)
    try:
        assert row(d, song_dir)["review"] == "keep"
    finally:
        d.close()


# ---------------------------------------------------------- Qt (offscreen) --

PySide6 = pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp, tmp_path):
    from PySide6.QtTest import QTest
    from yargvid.app import Window
    dbp = tmp_path / "app.sqlite"
    d = Database(dbp)
    synced(d, tmp_path, "plain")
    synced(d, tmp_path, "still", motion=0.001)
    synced(d, tmp_path, "foreign", existing_video="foreign")
    synced(d, tmp_path, "unv", sync_status="unverified", spread_ms=-1.0)
    d.close()
    w = Window(dbp, tmp_path / "work")
    w.show()
    qapp.processEvents()
    yield w, dbp
    w.close()
    QTest.qWait(50)


def test_D2_pre_existing_video_notice_is_visible(window, qapp):
    w, _ = window
    w._set_mode("existing")
    qapp.processEvents()
    w.list.setCurrentRow(0)
    qapp.processEvents()
    shown = w.flags.text() + w.facts.text() + w.status.text()
    assert "before this project" in shown


def test_D4_facts_show_unverified(window, qapp):
    w, _ = window
    w._set_mode("unsure")
    qapp.processEvents()
    for i, s in enumerate(w.songs):
        if s["title"] == "unv":
            w.list.setCurrentRow(i)
            break
    qapp.processEvents()
    assert "unverified" in w.facts.text()
    assert "-1" not in w.facts.text()


def test_A10_enter_in_url_field_never_keeps(window, qapp, monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    w, _ = window
    w.list.setCurrentRow(0)
    qapp.processEvents()
    calls = []
    monkeypatch.setattr(type(w), "_act", lambda self, a: calls.append(a))

    w.url.setFocus()
    w.url.setText("https://youtu.be/dQw4w9WgXcQ")
    qapp.processEvents()
    QTest.keyClick(w.url, Qt.Key.Key_Return)
    qapp.processEvents()
    assert calls == ["replace"]

    calls.clear()
    w.list.setFocus()
    qapp.processEvents()
    QTest.keyClick(w.list, Qt.Key.Key_Return)
    qapp.processEvents()
    assert calls == ["keep"]


def test_D5_app_drop_uses_drop_song(window, qapp, monkeypatch):
    from PySide6.QtWidgets import QMessageBox
    from yargvid import app as appmod
    w, dbp = window
    monkeypatch.setattr(appmod.QMessageBox, "question",
                        lambda *a, **k: QMessageBox.StandardButton.Yes)
    w._set_mode("third")
    w.list.setCurrentRow(0)
    qapp.processEvents()
    target = w.current
    w._act("drop")
    qapp.processEvents()
    d = Database(dbp)
    try:
        r = row(d, target)
        assert r["match_status"] == "skipped"
        assert r["fp_score"] is None and r["windows"] is None
    finally:
        d.close()
