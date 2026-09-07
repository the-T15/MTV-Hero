"""
Sync fallthrough rule (Batch 2b).

Measured on the 281-song export before this was written: 270 songs chose
the strongest hash peak, 11 fell through to a weaker candidate, 8 of those
to one with under half the hashes (five under a quarter, three at 1%). The
recheck's four large wrong moves were all of this kind: candidate 1 locked
on all seven windows but they disagreed by 41-96 ms, so a short rigid
section with a tenth of the support won on spread alone.

Rule: a fallback is only tried if it has at least SUPPORT_RATIO of
candidate 1's hashes. If nothing qualifying verifies and candidate 1 failed
only on spread, candidate 1 comes back as `unverified` - into the review
queue, not accepted. Gate failures and too-few-windows stay `rejected`.
"""

from __future__ import annotations

import numpy as np
import pytest

from yargvid import fingerprint as fp
from yargvid import sync as sy


ONES = np.ones(fp.SR, np.float32)


def cand(offset_s, hashes, score=600.0):
    return fp.MatchResult(offset_s, score, hashes, 0.9, 20000)


def fake_verify(outcomes):
    """Map id(candidate) -> SyncResult, recording what was tried."""
    tried = []

    def _verify(m, chart_hi, video_hi, static):
        tried.append(m)
        status, spread, windows = outcomes[id(m)]
        return sy.SyncResult(status, m.offset_seconds * 1000, spread,
                             fp_score=m.score, coverage=m.coverage,
                             windows=windows, windows_total=sy.N_EXCERPTS,
                             reason="excerpts disagree" if status == "rejected" else "")
    return _verify, tried


def run(monkeypatch, cands, outcomes):
    monkeypatch.setattr(fp, "match_candidates", lambda *a, **k: cands)
    verify, tried = fake_verify(outcomes)
    monkeypatch.setattr(sy, "_verify", verify)
    res = sy.estimate(ONES, ONES, ONES, ONES, trust_identity=True)
    return res, tried


def test_ratio_constant_exists():
    assert 0.0 < sy.SUPPORT_RATIO <= 1.0


def test_strong_first_candidate_still_wins(monkeypatch):
    c1, c2 = cand(-5.515, 1299), cand(17.206, 135)
    res, tried = run(monkeypatch, [c1, c2],
                     {id(c1): ("ok", 3.0, 7), id(c2): ("ok", 0.0, 7)})
    assert res.status == "ok" and res.offset_ms == -5515.0
    assert tried == [c1]


def test_weak_fallback_is_not_even_tried(monkeypatch):
    # Dancing Queen shape: c1 29914 hashes, spread 48; c3 107 hashes, spread 15.
    c1, c2, c3 = cand(-2.067, 29914), cand(-6.803, 3000), cand(-4.423, 107)
    res, tried = run(monkeypatch, [c1, c2, c3],
                     {id(c1): ("rejected", 48.0, 7),
                      id(c2): ("rejected", 117.0, 7),
                      id(c3): ("ok", 15.0, 7)})
    assert c3 not in tried
    assert res.offset_ms == -2067.0
    assert res.status == "unverified"
    assert res.fp_score == c1.score


def test_qualifying_fallback_still_displaces(monkeypatch):
    # Same shape, but the fallback carries 60% of c1's hashes: it may win.
    c1, c2 = cand(-4.458, 5917), cand(-11.413, 3600)
    res, tried = run(monkeypatch, [c1, c2],
                     {id(c1): ("rejected", 41.0, 7), id(c2): ("ok", 0.0, 7)})
    assert tried == [c1, c2]
    assert res.status == "ok" and res.offset_ms == -11413.0


def test_spread_failure_becomes_unverified_not_rejected(monkeypatch):
    c1, c2 = cand(-3.808, 10000), cand(49.505, 2200)
    res, _ = run(monkeypatch, [c1, c2],
                 {id(c1): ("rejected", 200.0, 7), id(c2): ("ok", 0.0, 7)})
    assert res.status == "unverified"
    assert res.offset_ms == -3808.0
    assert "guess" in res.reason.lower() or "check" in res.reason.lower()
    assert res.windows == 7


def test_too_few_windows_stays_as_it_was(monkeypatch):
    # Candidate 1 could not be verified at all (2 of 7 windows). That is not a
    # spread disagreement, so the rule does not promote it.
    c1, c2 = cand(-1.0, 1000), cand(9.0, 50)
    res, tried = run(monkeypatch, [c1, c2],
                     {id(c1): ("unverified", -1.0, 2), id(c2): ("ok", 0.0, 7)})
    assert tried == [c1]
    assert res.status == "unverified" and res.offset_ms == -1000.0
    assert res.windows == 2


def test_gate_failure_is_still_rejected(monkeypatch):
    c1 = cand(-1.0, 1000, score=30.0)
    monkeypatch.setattr(fp, "match_candidates", lambda *a, **k: [c1])
    res = sy.estimate(ONES, ONES, ONES, ONES, trust_identity=True)
    assert res.status == "rejected"
