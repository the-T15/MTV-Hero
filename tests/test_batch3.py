"""
Batch 3: dead code, stale text, small rule fixes, and README claims.

Most of this batch is words. Words are pinned the same way as code: the
sentence that was wrong must be gone, the thing the README says exists must
exist, and every command the parser knows must be documented.

    pytest -q tests/test_batch3.py
"""

from __future__ import annotations

import ast
import csv
import os
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from yargvid import audio as au                # noqa: E402
from yargvid import cli                        # noqa: E402
from yargvid import encode as enc              # noqa: E402
from yargvid import fingerprint as fp          # noqa: E402
from yargvid import match as mt                # noqa: E402
from yargvid import review as rv               # noqa: E402
from yargvid import sync as sy                 # noqa: E402
from yargvid.db import Database                # noqa: E402

PKG = Path(cli.__file__).parent
ROOT = PKG.parent
README = ROOT / "README.md"


def src(module) -> str:
    return Path(module.__file__).read_text(encoding="utf-8")


class R(dict):
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


def help_text(argv) -> str:
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        cli.main(argv)
    return buf.getvalue()


# ------------------------------------------------------------- dead code -----

def test_C1_unused_matching_symbols_are_gone():
    for name in ("version_mismatch", "MUSIC_VIDEO_TITLE", "FLAG_BELOW"):
        assert not hasattr(mt, name), name
    assert "version" not in rv.TAG_LABELS


def test_C1_pick_best_docstring_does_not_claim_a_margin_check():
    doc = mt.pick_best.__doc__ or ""
    assert "beat the runner-up" not in doc and "we require the margin" not in doc


def test_C2_unused_audio_helpers_are_gone():
    assert not hasattr(au, "find_video")
    assert not hasattr(au, "VIDEO_EXTS")


# ------------------------------------------------------------- stale text ----

def test_C3_gate_help_states_the_real_default():
    out = help_text(["match", "-h"])
    assert "60" not in out.split("--gate")[1].split("--")[0]
    assert str(int(fp.ACCEPT_SCORE)) in out


def test_C3_review_help_says_window_not_browser():
    out = help_text(["-h"])
    assert "open the review app in a browser" not in out


def test_C3_status_flag_wording_matches_what_is_flagged():
    assert "lyric/audio/gameplay" not in src(cli)


def test_C4_fingerprint_comment_matches_search_window():
    assert "+/-100 ms" not in src(fp)
    assert f"+/-{int(sy.SEARCH_MS)} ms" in src(fp)


def test_C5_syncresult_calls_pass_fields_by_keyword():
    tree = ast.parse(src(sy))
    bad = [n.lineno for n in ast.walk(tree)
           if isinstance(n, ast.Call)
           and getattr(n.func, "id", None) == "SyncResult"
           and len(n.args) > 1]
    assert bad == [], f"positional SyncResult fields at lines {bad}"


def test_C7_offsets_footer_records_the_verdict():
    s = src(cli)
    assert "that is the signal to" not in s
    assert "cmd_offsets" in s


def test_C8_redundant_imports_are_gone():
    from yargvid import app
    assert "_QUrl" not in src(app)
    assert "Path as _P" not in src(cli)


def test_C9_every_module_ends_with_a_newline():
    for p in PKG.glob("*.py"):
        assert p.read_bytes().endswith(b"\n"), p.name


# ------------------------------------------------------------- small rules ---

def test_B5_rejection_reason_names_the_gate_that_failed(monkeypatch):
    ones = np.ones(fp.SR, np.float32)
    monkeypatch.setattr(fp, "match_candidates",
                        lambda *a, **k: [fp.MatchResult(1.0, 38.0, 999, 0.9, 5000)])
    res = sy.estimate(ones, ones, None, None, trust_identity=True)
    assert res.status == "rejected"
    assert str(int(sy.IDENTITY_FLOOR)) in res.reason
    assert "45" not in res.reason

    monkeypatch.setattr(fp, "match_candidates",
                        lambda *a, **k: [fp.MatchResult(1.0, 50.0, 999, 0.1, 5000)])
    res = sy.estimate(ones, ones, None, None, trust_identity=True)
    assert res.status == "rejected"
    assert "coverage" in res.reason.lower()
    assert "50.0 <" not in res.reason


def test_doctor_checks_for_a_javascript_runtime(monkeypatch, capsys):
    monkeypatch.setattr(enc, "check_ffmpeg_vp8", lambda: True)
    monkeypatch.setattr(enc, "have", lambda tool: tool not in ("deno", "node", "bun"))
    cli.cmd_doctor(SimpleNamespace(), None)
    out = capsys.readouterr().out
    assert "MISSING" in out and "javascript" in out.lower()

    monkeypatch.setattr(enc, "have", lambda tool: True)
    cli.cmd_doctor(SimpleNamespace(), None)
    assert "MISSING" not in capsys.readouterr().out


def test_existing_video_reason_only_for_foreign():
    base = dict(match_note="T [U]", review=None, fp_score=500.0, spread_ms=1.0,
                offset_ms=0.0, sync_status="ok", motion=0.5, artist="A", title="t")
    assert "existing" not in rv.assess(R(existing_video="preview", **base)).tags
    assert "existing" in rv.assess(R(existing_video="foreign", **base)).tags


def test_export_carries_manual_flags(db, tmp_path):
    song(db, tmp_path / "s", sync_status="ok",
         source_path=str(tmp_path / "gone.mkv"),
         match_note="MANUAL: T [U]", sync_note="MANUAL: set by hand")
    out = tmp_path / "x.csv"
    cli.cmd_export(SimpleNamespace(out=str(out), limit=None, force=False), db)
    with out.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    assert "manual_video" in header and "manual_offset" in header


def test_links_command_prints_the_chosen_url(db, tmp_path, capsys):
    song(db, tmp_path / "Foo", video_id="dQw4w9WgXcQ", match_note="T [U]")
    cli.cmd_links(SimpleNamespace(pattern="Foo"), db)
    assert "https://youtu.be/dQw4w9WgXcQ" in capsys.readouterr().out
    assert "links" in help_text(["-h"])


def test_recheck_log_says_your_pick_for_every_hand_picked_video(
        db, tmp_path, monkeypatch, capsys):
    src_file = tmp_path / "video.src.mkv"
    src_file.write_bytes(b"x")
    monkeypatch.setattr(au, "find_stems", lambda d: [Path("x.ogg")])
    monkeypatch.setattr(au, "mix_stems", lambda s, sr=fp.SR: np.ones(sr, np.float32))
    monkeypatch.setattr(au, "decode_mono",
                        lambda p, sr=fp.SR, max_seconds=None: np.ones(sr, np.float32))
    monkeypatch.setattr(au, "duration_of", lambda p: 100.0)
    monkeypatch.setattr(enc, "motion_score", lambda p: 0.5)
    monkeypatch.setattr(sy, "estimate",
                        lambda *a, **k: sy.SyncResult("ok", offset_ms=-1000.0,
                                                      spread_ms=3.0, fp_score=500.0))
    song(db, tmp_path / "s", match_status="ok", match_note="MANUAL: T [U]",
         download_status="ok", source_path=str(src_file))
    cli.cmd_sync(SimpleNamespace(recheck=False, min_offset=0.0,
                                 skip_reviewed=False, limit=None), db)
    assert "your pick" in capsys.readouterr().out.lower()


# ------------------------------------------------------------- README --------

def _subcommands() -> list[str]:
    out = help_text(["-h"])
    m = re.search(r"\{([a-z,\-]+)\}", out)
    assert m, "could not read the subcommand list from -h"
    return m.group(1).split(",")


def test_readme_documents_every_command():
    text = README.read_text(encoding="utf-8")
    missing = [c for c in _subcommands()
               if f"`{c}`" not in text and f"yargvid {c}" not in text]
    assert missing == [], missing


def test_readme_documents_the_flags_that_exist():
    text = README.read_text(encoding="utf-8")
    for flag in ("--recheck", "--redo", "--gate", "--preview", "--skip-static",
                 "--skip-existing", "--mark", "--force", "retry"):
        assert flag in text, flag


def test_readme_no_longer_claims_browser_parity():
    text = README.read_text(encoding="utf-8")
    assert "same interface" not in text
    assert "Three lists" not in text


def test_readme_tests_section_names_files_that_exist():
    text = README.read_text(encoding="utf-8")
    assert "pytest" in text
    for name in re.findall(r"python (test_\w+\.py)", text):
        assert (ROOT / name).exists(), name


def test_readme_says_limit_goes_before_the_subcommand():
    text = README.read_text(encoding="utf-8")
    assert "Every command takes" not in text        # the sentence that misled
    assert "yargvid --limit" in text                # the flag shown in position

