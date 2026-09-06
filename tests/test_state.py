"""
State-rule tests. Deterministic bookkeeping only - nothing here decodes
audio or touches the network.

NOTES.md records that synthetic tests never caught an audio bug. These are
not audio tests. "Does reset clear review" is a state-machine question, and
the A1 coverage clamp survived in two other places precisely because nothing
re-checked it. Every test names the review item it pins.

Run from the repository root:  pytest -q
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yargvid import audio as au
from yargvid import cli
from yargvid import encode as enc
from yargvid import fingerprint as fp
from yargvid import match as mt
from yargvid import review as rv
from yargvid import sync as sy
from yargvid.db import Database


# ----------------------------------------------------------------- helpers ---

@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


def song(db, path, **cols):
    db.add_song(Path(path), "Artist", Path(path).name, 100.0)
    if cols:
        db.update(Path(path), **cols)
    return Path(path)


def row(db, path):
    return db.conn.execute(
        "SELECT * FROM songs WHERE song_dir = ?", (str(path),)
    ).fetchone()


class R(dict):
    """A sqlite3.Row stand-in: missing keys read as None."""

    def __getitem__(self, k):
        return dict.get(self, k)


# ------------------------------------------------------------- db.reset ------

def test_A5_retry_match_leaves_dropped_songs_alone(db):
    s = song(db, "/lib/dropped", match_status="skipped",
             match_note="MANUAL: no video wanted")
    db.reset("match", only_failed=True)
    assert row(db, s)["match_status"] == "skipped"


def test_A6_retry_sync_leaves_unverified_alone(db):
    # unverified is an accepted outcome: PREREQ['encode'], review.queue and
    # export all treat it that way. reset must agree.
    s = song(db, "/lib/unv", match_status="ok", download_status="ok",
             sync_status="unverified", offset_ms=-1000.0)
    db.reset("sync", only_failed=True)
    assert row(db, s)["sync_status"] == "unverified"
    assert row(db, s)["offset_ms"] == -1000.0


def test_A6_manual_queue_does_not_list_unverified(db):
    song(db, "/lib/unv", match_status="ok", download_status="ok",
         sync_status="unverified")
    assert db.manual_queue() == []


def test_A4_reset_sync_clears_review(db):
    s = song(db, "/lib/kept", match_status="ok", download_status="ok",
             sync_status="ok", offset_ms=5.0, review="keep")
    db.reset("sync", only_failed=False)
    assert row(db, s)["review"] is None


def test_A4_reset_match_clears_review(db):
    s = song(db, "/lib/kept", match_status="ok", download_status="ok",
             sync_status="ok", offset_ms=5.0, review="keep")
    db.reset("match", only_failed=False)
    assert row(db, s)["review"] is None


# ------------------------------------------------------------- cmd_set -------

@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setattr(mt, "fetch_metadata",
                        lambda vid, cookies=None: {"title": "T", "uploader": "U"})


def test_A3_set_clears_review_and_measurements(db, no_network):
    s = song(db, "/lib/a", match_status="ok", video_id="aaaaaaaaaaa",
             sync_status="ok", review="keep", motion=0.5, fp_score=80.0,
             dominance=3.0, windows=7)
    out = cli.cmd_set(SimpleNamespace(pattern=str(s), url="bbbbbbbbbbb",
                                      cookies=None), db)
    assert out == "set"
    r = row(db, s)
    assert r["review"] is None
    assert r["motion"] is None and r["fp_score"] is None
    assert r["dominance"] is None and r["windows"] is None


def test_A11_set_prefers_an_exact_folder_match(db, no_network):
    # The review app passes the full song_dir. A folder whose path is a
    # prefix of another's must still resolve to exactly that folder.
    short = song(db, "/lib/Foo", match_status="ok", video_id="aaaaaaaaaaa")
    song(db, "/lib/Foo (Live)", match_status="ok", video_id="aaaaaaaaaaa")
    out = cli.cmd_set(SimpleNamespace(pattern=str(short), url="bbbbbbbbbbb",
                                      cookies=None), db)
    assert out == "set"
    assert row(db, short)["video_id"] == "bbbbbbbbbbb"


# ------------------------------------------------------------- doctor --------

def test_A7_doctor_creates_no_database(tmp_path, monkeypatch):
    monkeypatch.setattr(enc, "have", lambda tool: True)
    monkeypatch.setattr(enc, "check_ffmpeg_vp8", lambda: True)
    target = tmp_path / "nothing.sqlite"
    cli.main(["--db", str(target), "doctor"])
    assert not target.exists()


# ------------------------------------------------------------- review --------

def test_A12_manual_pick_saved_for_later_is_not_checked():
    r = R(match_note="MANUAL: T [U]", review="later", fp_score=80.0,
          spread_ms=5.0, offset_ms=100.0, sync_status="ok", motion=0.5,
          artist="A", title="t")
    assert "clean" not in rv.assess(r).tags


def test_manual_pick_kept_is_checked():
    r = R(match_note="MANUAL: T [U]", review="keep", fp_score=80.0,
          spread_ms=5.0, offset_ms=100.0, sync_status="ok", motion=0.5,
          artist="A", title="t")
    assert rv.assess(r).tags == ["clean"]


# ------------------------------------------------------------- recheck -------

@pytest.fixture
def stubbed_audio(monkeypatch, tmp_path):
    """cmd_sync without ffmpeg: audio decodes to ones, motion is 0.5."""
    src = tmp_path / "video.src.mkv"
    src.write_bytes(b"x")
    monkeypatch.setattr(au, "find_stems", lambda d: [Path("x.ogg")])
    monkeypatch.setattr(au, "mix_stems",
                        lambda stems, sr=fp.SR: np.ones(sr, np.float32))
    monkeypatch.setattr(au, "decode_mono",
                        lambda p, sr=fp.SR, max_seconds=None: np.ones(sr, np.float32))
    monkeypatch.setattr(au, "duration_of", lambda p: 100.0)
    monkeypatch.setattr(enc, "motion_score", lambda p: 0.5)
    return src


def _recheck(db, monkeypatch, result):
    monkeypatch.setattr(sy, "estimate", lambda *a, **k: result)
    cli.cmd_sync(SimpleNamespace(recheck=True, min_offset=0.0,
                                 skip_reviewed=False, limit=None), db)


def _synced(db, src, **extra):
    return song(db, "/lib/s", match_status="ok", download_status="ok",
                source_path=str(src), sync_status="ok", offset_ms=-1000.0,
                spread_ms=3.0, motion=0.5, review="keep", encode_status="ok",
                ini_status="ok", **extra)


def test_A8_status_flip_to_rejected_clears_approval(db, stubbed_audio, monkeypatch):
    s = _synced(db, stubbed_audio)
    _recheck(db, monkeypatch, sy.SyncResult("rejected", offset_ms=-1020.0,
                                            spread_ms=400.0, fp_score=80.0))
    r = row(db, s)
    assert r["sync_status"] == "rejected"
    assert r["review"] is None
    assert r["encode_status"] == "pending" and r["ini_status"] == "pending"


def test_A8_moved_offset_clears_approval(db, stubbed_audio, monkeypatch):
    s = _synced(db, stubbed_audio)
    _recheck(db, monkeypatch, sy.SyncResult("ok", offset_ms=-1500.0,
                                            spread_ms=3.0, fp_score=80.0))
    r = row(db, s)
    assert r["offset_ms"] == -1500.0
    assert r["review"] is None and r["encode_status"] == "pending"


def test_A8_unchanged_keeps_approval_but_refreshes_measurements(
        db, stubbed_audio, monkeypatch):
    s = _synced(db, stubbed_audio)
    _recheck(db, monkeypatch, sy.SyncResult("ok", offset_ms=-1010.0,
                                            spread_ms=4.0, fp_score=80.0,
                                            windows=6, windows_total=7))
    r = row(db, s)
    assert r["review"] == "keep" and r["encode_status"] == "ok"
    assert r["windows"] == 6            # measurements are refreshed
    assert r["spread_ms"] == 4.0


def test_A8_flip_between_encodable_states_keeps_approval(
        db, stubbed_audio, monkeypatch):
    s = _synced(db, stubbed_audio)
    _recheck(db, monkeypatch, sy.SyncResult("unverified", offset_ms=-1010.0,
                                            spread_ms=-1.0, fp_score=80.0))
    r = row(db, s)
    assert r["sync_status"] == "unverified"
    assert r["review"] == "keep"


# ------------------------------------------------------------- fp_score ------

def test_A2_fp_score_is_the_identity_score(monkeypatch):
    # Candidate 1 fails verification, candidate 2 holds. The stored fp_score
    # must be candidate 1's (the score that passed the gate), not 2's.
    c1 = fp.MatchResult(0.0, 600.0, 9000, 0.9, 20000)
    c2 = fp.MatchResult(3.0, 36.0, 800, 0.9, 20000)
    monkeypatch.setattr(fp, "match_candidates", lambda *a, **k: [c1, c2])

    def fake_verify(m, chart_hi, video_hi, static):
        status = "rejected" if m is c1 else "ok"
        return sy.SyncResult(status, m.offset_seconds * 1000, 3.0,
                             fp_score=m.score, coverage=m.coverage)
    monkeypatch.setattr(sy, "_verify", fake_verify)

    res = sy.estimate(np.ones(fp.SR, np.float32), np.ones(fp.SR, np.float32),
                      np.ones(fp.SR, np.float32), np.ones(fp.SR, np.float32),
                      trust_identity=True)
    assert res.status == "ok" and res.offset_ms == 3000.0
    assert res.fp_score == 600.0


# ------------------------------------------------------------- coverage ------

def test_A1_coverage_counts_a_delayed_video(db):
    # chart 200 s, video 150 s, video delayed by 60 s: it ends at 210 s of
    # song time and covers the chart. The old clamp said it ended at 150.
    assert cli.covers_song(video_s=150.0, offset_ms=-60000.0, chart_s=200.0)
    assert not cli.covers_song(video_s=150.0, offset_ms=60000.0, chart_s=200.0)


def test_A1_export_refuses_to_overwrite(db, tmp_path, monkeypatch):
    out = tmp_path / "baseline.csv"
    out.write_text("keep me")
    song(db, "/lib/s", source_path=str(tmp_path / "gone.mkv"),
         sync_status="ok")
    cli.cmd_export(SimpleNamespace(out=str(out), limit=None, force=False), db)
    assert out.read_text() == "keep me"


# ------------------------------------------------------------- song.ini ------

def test_song_ini_keeps_crlf_and_bom(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "song.ini").write_bytes(b"\xef\xbb\xbf[song]\r\nname = x\r\n")
    cli.write_video_start_time(d, -123, backup=False)
    got = (d / "song.ini").read_bytes()
    assert got == b"\xef\xbb\xbf[song]\r\nvideo_start_time = -123\r\nname = x\r\n"


def test_song_ini_keeps_lf(tmp_path):
    d = tmp_path / "s"
    d.mkdir()
    (d / "song.ini").write_bytes(b"[song]\nname = x\n")
    cli.write_video_start_time(d, 5, backup=False)
    assert (d / "song.ini").read_bytes() == b"[song]\nvideo_start_time = 5\nname = x\n"


# ------------------------------------------------------------- subprocess ----

def test_B4_run_survives_a_hung_process():
    proc = mt._run([sys.executable, "-c", "import time; time.sleep(5)"],
                   timeout=1)
    assert proc.returncode != 0
    assert "timed out" in proc.stderr.lower()


def test_B4_run_survives_a_missing_binary():
    proc = mt._run(["definitely-not-a-binary-xyz"])
    assert proc.returncode != 0
