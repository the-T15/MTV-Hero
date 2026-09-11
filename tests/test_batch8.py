"""
The matching policy (Batch 8).

The first bench run scored the shipped rule at 169 of 170 approvals and
11 of 11 reachable overrides. Its one winnable miss is a title-words win: a
stranger's "Official Music Video" outranked the label's "Official Video"
because the channel bonus is added to the title score, and a big enough
title number swamps it. The policy that replaces it ranks lexicographically
instead, so the channel is decided before the title is ever read:

    (channel class, title score, view count, fingerprint score)

- Channel class first. `official` treats the artist's own channel, their
  label's and VEVO as one class; `artist_first` ranks them artist > VEVO >
  label, the order the existing channel bonus already gives them. Under
  both, a third-party channel ranks below every official one and an
  auto-generated Topic channel below every third-party one. Both are
  registered in `bench.POLICIES`; the bench decides which `pick_best`
  applies.
- Title score within a class. `title_score` already orders music video >
  visualizer > lyric video > audio upload and carries every penalty term,
  so it is reused as-is rather than re-encoded as buckets.
- View count within that, as the tie-break. A known count outranks an
  unknown one. Views never lift a stranger's upload over an official one:
  the class is decided first.
- A third-party upload is chosen only when no official candidate passes
  the fingerprint gate; views then rank the strangers among themselves.
- "fan MV available": when the pick is an official channel's non-music
  video (visualizer, lyric, audio) and a third-party candidate titled as
  the music video also passed the gate, the pick is right by policy and
  still worth a look. `pick_best` keeps probing such candidates instead of
  stopping early, so the flag can fire at match time; the bench reports the
  same thing in a `fan_mv` column. The record is a `match_note` prefix,
  `REVIEW: fan MV available: <id> - <title> [<channel>]`, which reaches
  `manual_queue` and `flagged_matches` through the plumbing that exists;
  the review app tags it `fan_mv`, not `audio`.
- Nothing stored before Batch 7 has a view count, so the bench reports how
  many ranked candidates carry one. The tie-break's coverage is visible
  rather than assumed.

The gate is not a policy question: every policy ranks only what passes
`ACCEPT_SCORE` and `ACCEPT_COVERAGE`, exactly as `rank_current` does.

    pytest -q tests/test_batch8.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from yargvid import audio as au
from yargvid import bench
from yargvid import cli
from yargvid import fingerprint as fp
from yargvid import match as mt
from yargvid import review as rv
from yargvid.db import Database

GATE = fp.ACCEPT_SCORE + 100.0    # comfortably heard
COV = fp.ACCEPT_COVERAGE + 0.5
NEW = ("official", "artist_first")
CHART = "Artist Song"


# ----------------------------------------------------------------- helpers ---

@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


def song(db, path, artist="Artist", title="Song", **cols):
    db.add_song(Path(path), artist, title, 200.0)
    if cols:
        db.update(Path(path), **cols)
    return Path(path)


def row(db, path):
    return db.conn.execute(
        "SELECT * FROM songs WHERE song_dir = ?", (str(path),)
    ).fetchone()


def cand(video_id, title, uploader="Artist", score=GATE, coverage=COV,
         views=None):
    return mt.Candidate(video_id=video_id, title=title, uploader=uploader,
                        duration=200.0, score=score, coverage=coverage,
                        view_count=views)


def ids(ranked):
    return [c.video_id for c in ranked]


# ------------------------------------------------------------ the policies ---

def test_both_policies_are_registered():
    assert set(NEW) <= set(bench.POLICIES)
    assert bench.POLICIES["current"] is bench.rank_current


@pytest.mark.parametrize("policy", NEW)
def test_channel_class_is_decided_before_the_title(policy):
    # The band's own visualizer against a stranger's "Official Music Video".
    # The shipped rule adds 7 for the channel and 10.5 for the title words,
    # so the stranger wins; under a channel-first policy it cannot.
    cands = [cand("sssssssssss", "Song (Official Music Video)",
                  uploader="Stranger"),
             cand("aaaaaaaaaaa", "Song (Visualizer)")]
    assert ids(bench.rank_current(cands, CHART))[0] == "sssssssssss"
    assert ids(bench.POLICIES[policy](cands, CHART))[0] == "aaaaaaaaaaa"


def test_one_official_class_keeps_the_label_pick():
    # The Alexisonfire shape: the label posted the video, the band's own
    # channel posted a visualizer, a fan reuploaded the video. One official
    # class lets the title decide between label and band; three classes
    # put the band's channel above the label whatever it posted.
    cands = [cand("lllllllllll", "Song (Official Video)",
                  uploader="Dine Alone Records"),
             cand("aaaaaaaaaaa", "Song (Visualizer)"),
             cand("sssssssssss", "Song (Official Music Video)",
                  uploader="Stranger")]
    assert ids(bench.POLICIES["official"](cands, CHART))[0] == "lllllllllll"
    assert ids(bench.POLICIES["artist_first"](cands, CHART))[0] == "aaaaaaaaaaa"


def test_artist_first_orders_artist_vevo_label_stranger():
    # Identical titles, so only the channel separates them. Scores run the
    # other way so that `official`, which merges the three, shows the merge:
    # it falls through to the fingerprint score and orders them backwards.
    cands = [cand("aaaaaaaaaaa", "Song (Official Video)", score=GATE),
             cand("vvvvvvvvvvv", "Song (Official Video)",
                  uploader="ArtistVEVO", score=GATE + 10),
             cand("lllllllllll", "Song (Official Video)",
                  uploader="Dine Alone Records", score=GATE + 20),
             cand("sssssssssss", "Song (Official Video)",
                  uploader="Stranger", score=GATE + 30)]
    assert ids(bench.POLICIES["artist_first"](cands, CHART)) == [
        "aaaaaaaaaaa", "vvvvvvvvvvv", "lllllllllll", "sssssssssss"]
    assert ids(bench.POLICIES["official"](cands, CHART)) == [
        "lllllllllll", "vvvvvvvvvvv", "aaaaaaaaaaa", "sssssssssss"]


@pytest.mark.parametrize("policy", NEW)
def test_title_order_within_a_class(policy):
    cands = [cand("uuuuuuuuuuu", "Song (Official Audio)"),
             cand("yyyyyyyyyyy", "Song (Lyric Video)"),
             cand("iiiiiiiiiii", "Song (Visualizer)"),
             cand("mmmmmmmmmmm", "Song (Official Video)")]
    assert ids(bench.POLICIES[policy](cands, CHART)) == [
        "mmmmmmmmmmm", "iiiiiiiiiii", "yyyyyyyyyyy", "uuuuuuuuuuu"]


@pytest.mark.parametrize("policy", NEW)
def test_views_break_ties_before_the_fingerprint_score(policy):
    cands = [cand("aaaaaaaaaa1", "Song (Official Video)", score=GATE + 50,
                  views=10),
             cand("aaaaaaaaaa2", "Song (Official Video)", score=GATE,
                  views=1000)]
    assert ids(bench.POLICIES[policy](cands, CHART))[0] == "aaaaaaaaaa2"


@pytest.mark.parametrize("policy", NEW)
def test_a_known_view_count_outranks_an_unknown_one(policy):
    # Zero is a real count; None is a row stored before the column existed.
    cands = [cand("aaaaaaaaaa1", "Song (Official Video)", views=None),
             cand("aaaaaaaaaa2", "Song (Official Video)", views=0)]
    assert ids(bench.POLICIES[policy](cands, CHART))[0] == "aaaaaaaaaa2"


@pytest.mark.parametrize("policy", NEW)
def test_views_never_lift_a_stranger_over_an_official_channel(policy):
    cands = [cand("sssssssssss", "Song (Official Video)", uploader="Stranger",
                  views=10_000_000),
             cand("aaaaaaaaaaa", "Song (Official Video)", views=None)]
    assert ids(bench.POLICIES[policy](cands, CHART))[0] == "aaaaaaaaaaa"


@pytest.mark.parametrize("policy", NEW)
def test_third_party_only_when_no_official_passes_the_gate(policy):
    # The band's upload was found but never heard (score 0). Two strangers
    # passed with the same title; views rank them.
    cands = [cand("aaaaaaaaaaa", "Song (Official Video)", score=0.0,
                  coverage=0.0),
             cand("sssssssssss", "Song (Official Video)", uploader="Stranger",
                  views=10),
             cand("ttttttttttt", "Song (Official Video)", uploader="Someone",
                  views=1000)]
    ranked = bench.POLICIES[policy](cands, CHART)
    assert ids(ranked) == ["ttttttttttt", "sssssssssss"]


@pytest.mark.parametrize("policy", NEW)
def test_topic_channel_ranks_below_a_stranger(policy):
    # An auto-generated channel is album art for the whole song; a
    # stranger's lyric video at least moves.
    cands = [cand("ttttttttttt", "Song", uploader="Artist - Topic"),
             cand("sssssssssss", "Song (Lyric Video)", uploader="Stranger")]
    assert ids(bench.POLICIES[policy](cands, CHART)) == [
        "sssssssssss", "ttttttttttt"]


# --------------------------------------------------------------- the bench ---

@pytest.fixture
def policy_db(db):
    """
    Two songs with stored candidates.

    a  the band's visualizer against a stranger's music video: the shipped
       rule picks the stranger, both new policies pick the band and flag
       the stranger as the fan MV.
    b  three gate passers, two of them with a view count, plus one that
       was never heard and carries a count that must not be counted.
    """
    a = song(db, "/lib/a", match_status="ok", video_id="sssssssssss",
             match_note="Song (Official Music Video) [Stranger]")
    db.save_candidates(a, [
        cand("aaaaaaaaaaa", "Song (Visualizer)"),
        cand("sssssssssss", "Song (Official Music Video)",
             uploader="Stranger"),
    ])
    b = song(db, "/lib/b", match_status="ok", video_id="bbbbbbbbbbb",
             match_note="Song (Official Video) [Artist]")
    db.save_candidates(b, [
        cand("bbbbbbbbbbb", "Song (Official Video)", views=500),
        cand("bbbbbbbbbb2", "Song (Lyric Video)", views=20),
        cand("bbbbbbbbbb3", "Song (Visualizer)"),
        cand("bbbbbbbbbb4", "Song (Official Audio)", score=0.0,
             coverage=0.0, views=9),
    ])
    return db


def _by_dir(rows):
    return {r["song_dir"]: r for r in rows}


@pytest.mark.parametrize("policy", NEW)
def test_bench_runs_the_new_policies(policy_db, policy):
    got = _by_dir(bench.run(policy_db, policy))
    assert got["/lib/a"]["policy_pick"] == "aaaaaaaaaaa"
    assert got["/lib/b"]["policy_pick"] == "bbbbbbbbbbb"


def test_bench_reports_the_fan_mv(policy_db):
    for policy in NEW:
        got = _by_dir(bench.run(policy_db, policy))
        assert got["/lib/a"]["fan_mv"] == "sssssssssss"
        assert got["/lib/b"]["fan_mv"] == ""
    # The shipped rule picks the stranger outright, so there is no official
    # non-MV to flag.
    got = _by_dir(bench.run(policy_db, "current"))
    assert got["/lib/a"]["fan_mv"] == ""


def test_bench_reports_view_count_coverage(policy_db):
    rows = bench.run(policy_db, "official")
    got = _by_dir(rows)
    assert (got["/lib/a"]["ranked"], got["/lib/a"]["views_known"]) == (2, 0)
    assert (got["/lib/b"]["ranked"], got["/lib/b"]["views_known"]) == (3, 2)
    assert bench.summary(rows)["views"] == dict(ranked=5, known=2)


def test_bench_prints_view_count_coverage(policy_db, tmp_path, monkeypatch,
                                          capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["--db", str(policy_db.path), "bench"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if "view" in ln.lower()]
    assert any("2 of 5" in ln for ln in lines), lines


def test_bench_policy_flag_takes_the_new_names(policy_db, tmp_path,
                                               monkeypatch):
    monkeypatch.chdir(tmp_path)
    for policy in NEW:
        assert cli.main(["--db", str(policy_db.path), "bench",
                         "--policy", policy]) == 0
        out = tmp_path / f"yargvid_bench_{policy}.csv"
        assert out.exists()
        header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
        assert "fan_mv" in header


# ---------------------------------------------------------------- pick_best ---

@pytest.fixture
def picker(monkeypatch, tmp_path):
    """
    `pick_best` without yt-dlp or ffmpeg.

    `state.found` is what the search returns; `state.heard` maps a video id
    to the (score, coverage) its probe measures, and an id not listed fails
    the gate. `state.probed` records the probe order. The decoded "audio"
    is the video id as bytes, which is how the stubbed matcher knows which
    probe it is scoring.
    """
    state = SimpleNamespace(found=[], heard={}, probed=[])
    monkeypatch.setattr(mt, "search_candidates",
                        lambda *a, **k: [replace(c) for c in state.found])

    def probe(video_id, work, cookies=None, sleep=0.0):
        state.probed.append(video_id)
        return Path(work) / f"{video_id}.m4a", ""
    monkeypatch.setattr(mt, "probe_audio", probe)
    monkeypatch.setattr(
        au, "decode_mono",
        lambda path, sr=fp.SR, max_seconds=None:
            np.frombuffer(path.stem.encode("ascii"), np.uint8))
    monkeypatch.setattr(fp, "make_hashes", lambda samples: samples)

    def matched(chart_hashes, hashes, chart_seconds):
        score, cov = state.heard.get(bytes(hashes).decode("ascii"),
                                     (0.0, 0.0))
        return SimpleNamespace(score=score, offset_seconds=0.0, coverage=cov)
    monkeypatch.setattr(fp, "match", matched)

    def run():
        chart = np.ones(fp.SR * 200, np.float32)
        return mt.pick_best(chart, "Artist", "Song", tmp_path)
    state.run = run
    return state


def found(video_id, title, uploader="Artist"):
    return mt.Candidate(video_id=video_id, title=title, uploader=uploader,
                        duration=200.0)


def test_pick_best_probes_official_channels_first(picker):
    picker.found = [found("sssssssssss", "Song (Official Music Video)",
                          uploader="Stranger"),
                    found("aaaaaaaaaaa", "Song (Visualizer)")]
    picker.heard = {"aaaaaaaaaaa": (GATE, COV), "sssssssssss": (GATE, COV)}
    picker.run()
    assert picker.probed[0] == "aaaaaaaaaaa"


def test_pick_best_takes_the_band_visualizer_over_a_fan_music_video(picker):
    picker.found = [found("sssssssssss", "Song (Official Music Video)",
                          uploader="Stranger"),
                    found("aaaaaaaaaaa", "Song (Visualizer)")]
    picker.heard = {"aaaaaaaaaaa": (GATE, COV), "sssssssssss": (GATE, COV)}
    winner, _, _ = picker.run()
    assert winner.video_id == "aaaaaaaaaaa"


def test_pick_best_flags_a_fan_music_video_that_passed_the_gate(picker):
    picker.found = [found("sssssssssss", "Song (Official Music Video)",
                          uploader="Stranger"),
                    found("aaaaaaaaaaa", "Song (Visualizer)")]
    # Heard and passed: the band's visualizer is still the pick, and the
    # stranger is probed even though nothing it could score would beat it.
    picker.heard = {"aaaaaaaaaaa": (GATE, COV), "sssssssssss": (GATE, COV)}
    winner, _, reason = picker.run()
    assert winner.video_id == "aaaaaaaaaaa"
    assert reason.startswith("ok (fan MV available")
    assert "sssssssssss" in reason
    assert "sssssssssss" in picker.probed
    # Heard and failed: not the same recording, nothing to look at.
    picker.probed.clear()
    picker.heard = {"aaaaaaaaaaa": (GATE, COV)}
    winner, _, reason = picker.run()
    assert winner.video_id == "aaaaaaaaaaa"
    assert reason == "ok"


def test_pick_best_does_not_flag_or_probe_past_a_plain_official_title(picker):
    # "Artist - Song" on the band's channel says nothing against the
    # picture, so it is not a non-MV: the stranger cannot win and is not
    # worth a download.
    picker.found = [found("sssssssssss", "Song (Official Music Video)",
                          uploader="Stranger"),
                    found("aaaaaaaaaaa", "Artist - Song")]
    picker.heard = {"aaaaaaaaaaa": (GATE, COV), "sssssssssss": (GATE, COV)}
    winner, _, reason = picker.run()
    assert winner.video_id == "aaaaaaaaaaa"
    assert reason == "ok"
    assert "sssssssssss" not in picker.probed


# ---------------------------------------------------------- the match note ---

def test_match_writes_the_fan_mv_prefix(db, tmp_path, monkeypatch):
    monkeypatch.setattr(au, "find_stems", lambda d: [Path("x.ogg")])
    monkeypatch.setattr(au, "mix_stems",
                        lambda stems, sr=fp.SR: np.ones(sr, np.float32))
    winner = cand("aaaaaaaaaaa", "Song (Visualizer)")
    monkeypatch.setattr(
        mt, "pick_best",
        lambda *a, **k: (winner, [winner],
                         "ok (fan MV available: sssssssssss - review)"))
    s = song(db, "/lib/a")
    args = SimpleNamespace(songs=None, redo=False, limit=None, sample=False,
                           work=str(tmp_path), cookies=None, sleep=0.0,
                           gate=None)
    cli.cmd_match(args, db)
    note = row(db, s)["match_note"]
    assert note.startswith("REVIEW: fan MV available")
    assert "sssssssssss" in note
    assert note.endswith("Song (Visualizer) [Artist]")
    # The prefix is the plumbing: it reaches the queue as any REVIEW: does.
    assert [r["song_dir"] for r in db.manual_queue()] == [str(s)]
    assert [r["song_dir"] for r in db.flagged_matches()] == [str(s)]


class R(dict):
    def __getitem__(self, k):
        return dict.get(self, k)


BASE = dict(review=None, fp_score=500.0, spread_ms=1.0, offset_ms=0.0,
            sync_status="ok", motion=0.5, artist="Artist", title="Song",
            chart_seconds=200.0)


def test_review_tags_the_fan_mv_note_as_its_own_thing():
    r = R(match_note="REVIEW: fan MV available: sssssssssss - "
                     "Song (Visualizer) [Artist]", **BASE)
    risk = rv.assess(r)
    assert "fan_mv" in risk.tags
    assert "audio" not in risk.tags          # it is a video, just not the MV
    assert "channel" not in risk.tags        # the pick is the band's own
    assert "fan_mv" in rv.TAG_LABELS
    assert "fan_mv" in rv.DOUBT_TAGS
    assert rv.bucket(dict(review=None, tags=risk.tags)) == "unsure"
