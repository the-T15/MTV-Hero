"""
Scoring a matching policy against what a person said.

Every argument about the ranking rules so far has been settled by an example:
one song where the official upload lost to a stranger's, one where a lyric
video won. That is not a measurement, and a rule tuned on three songs is as
likely to cost twenty as to save them.

This module supplies the thing a rule has to beat. The `candidates` table
already holds every video the search returned for every song, with the
fingerprint score and coverage that were measured at the time, so a policy can
be re-run over all of them offline - no downloads, no writes, seconds rather
than days.

Ground truth is what a person said, and the two ways of saying it are kept
apart. An approval (`review = 'keep'`) means "the pipeline's pick is right".
An override (`match_note` starting `MANUAL:`) means "not that one, this one",
which is the more informative of the two and the harder test. A hand pick that
was afterwards approved is counted as an override only: the classes partition
the labelled set, so no policy can be scored twice for one song.

A labelled song is only WINNABLE if its known video is among the stored
candidates and passed the fingerprint gate. One that was never returned by the
search (`absent`) or was never heard (`gated`) cannot be picked by any ranking
rule whatsoever, so it is reported separately rather than counted as a loss -
those are search and download problems, and a policy change cannot fix them.
"""

from __future__ import annotations

from pathlib import Path

from . import fingerprint as fp
from . import match as mt


def rank_current(candidates: list[mt.Candidate],
                 chart_text: str) -> list[mt.Candidate]:
    """
    The rule `pick_best` applies today, as a ranking.

    Gate first, title preference second, score breaking ties - deliberately
    the same order of operations, not an approximation of it. This is the
    baseline every other policy is measured against, so any difference between
    it and the shipped matcher would show up as a change in the policy being
    tested.

    A candidate that never passed the gate is not ranked at all. That includes
    everything with a score of 0.0, which in the stored table means "found but
    never downloaded" - it was never heard, so it can never be the pick.
    """
    passers = [
        c for c in candidates
        if c.score >= fp.ACCEPT_SCORE and c.coverage >= fp.ACCEPT_COVERAGE
    ]
    passers.sort(
        key=lambda c: (mt.title_preference(c.title, c.uploader, chart_text),
                       c.score),
        reverse=True,
    )
    return passers


# `current` is the baseline and never changes. The rest come from
# `match.RANKERS`, so the bench measures the code that ships rather than a
# copy of it, and the policy `pick_best` did not take stays measurable.
POLICIES = {"current": rank_current, **mt.RANKERS}


def label_of(song) -> str:
    """
    Which class of ground truth this song carries, if any.

    Order matters: override is tested before approval, so a hand pick that was
    later confirmed by eye lands in exactly one class. A drop - MANUAL with no
    video - says nothing about which video is right, so it is unlabelled.
    """
    if not song["video_id"]:
        return ""
    if (song["match_note"] or "").startswith("MANUAL:"):
        return "override"
    if song["review"] == "keep":
        return "approval"
    return ""


def _candidates_for(db, song_dir: str) -> list[mt.Candidate]:
    rows = db.conn.execute(
        "SELECT video_id, title, uploader, duration, score, coverage, "
        "       view_count "
        "FROM candidates WHERE song_dir = ? ORDER BY video_id",
        (song_dir,),
    ).fetchall()
    return [
        mt.Candidate(
            video_id=r["video_id"], title=r["title"] or "",
            uploader=r["uploader"] or "", duration=r["duration"] or 0.0,
            score=r["score"] or 0.0, coverage=r["coverage"] or 0.0,
            view_count=r["view_count"],
        )
        for r in rows
    ]


def _known_status(known: str, candidates: list[mt.Candidate]) -> str:
    """reachable, gated or absent - see the module docstring."""
    for c in candidates:
        if c.video_id == known:
            passes = (c.score >= fp.ACCEPT_SCORE
                      and c.coverage >= fp.ACCEPT_COVERAGE)
            return "reachable" if passes else "gated"
    return "absent"


def run(db, policy: str = "current") -> list[dict]:
    """
    One row per song that has stored candidates. Reads only.

    Songs without candidates are left out entirely rather than reported as
    zero: nothing was ever searched for them, so there is no ranking decision
    to score.
    """
    rank = POLICIES[policy]
    songs = db.conn.execute(
        "SELECT * FROM songs WHERE song_dir IN "
        "(SELECT DISTINCT song_dir FROM candidates) ORDER BY song_dir"
    ).fetchall()

    out: list[dict] = []
    for s in songs:
        cands = _candidates_for(db, s["song_dir"])
        chart_text = f"{s['artist'] or ''} {s['title'] or ''}".strip()
        ranked = rank(cands, chart_text)
        pick = ranked[0].video_id if ranked else ""
        alt = mt.fan_mv(ranked, chart_text)

        label = label_of(s)
        known = s["video_id"] if label else ""
        out.append({
            # Forward slashes, whatever the database holds. This column is
            # meant to be cut out of the CSV and pasted into a `--songs` file,
            # which reads either separator, and a backslashed path in a CSV is
            # one escaping accident away from naming a different folder.
            "song_dir": Path(s["song_dir"]).as_posix(),
            "artist": s["artist"] or "",
            "title": s["title"] or "",
            "label": label,
            "known_video": known,
            "known_status": _known_status(known, cands) if label else "",
            "policy_pick": pick,
            "current_video": s["video_id"] or "",
            "win": (pick == known) if label else None,
            "fan_mv": alt.video_id if alt else "",
            # How much of the ranked field the view tie-break can actually
            # see. Nothing stored before the column existed carries a count,
            # so a coverage of zero means the tie-break was never exercised -
            # which is a fact about the data, not a result.
            "ranked": len(ranked),
            "views_known": sum(1 for c in ranked if c.view_count is not None),
        })
    return out


def summary(rows: list[dict]) -> dict[str, dict[str, int]]:
    """
    Per-class tallies. The labelled classes are never merged.

    A policy that wins every approval and loses every override has learnt to
    agree with the pipeline, which is the failure this bench exists to catch,
    and one combined number would hide it.
    """
    out: dict[str, dict[str, int]] = {}
    for cls in ("approval", "override"):
        mine = [r for r in rows if r["label"] == cls]
        out[cls] = dict(
            labelled=len(mine),
            reachable=sum(1 for r in mine if r["known_status"] == "reachable"),
            gated=sum(1 for r in mine if r["known_status"] == "gated"),
            absent=sum(1 for r in mine if r["known_status"] == "absent"),
            win=sum(1 for r in mine if r["win"]),
        )
    plain = [r for r in rows if not r["label"]]
    # An unlabelled song has no right answer, so it cannot be won or lost.
    # What it can say is how much a policy would CHANGE: every one of these
    # is a video the pipeline chose and nobody has looked at, and a policy
    # that moves a thousand of them is a different proposition from one that
    # moves twelve.
    out["unlabelled"] = dict(
        songs=len(plain),
        differs=sum(1 for r in plain
                    if r["current_video"]
                    and r["policy_pick"] != r["current_video"]),
    )
    out["views"] = dict(
        ranked=sum(r["ranked"] for r in rows),
        known=sum(r["views_known"] for r in rows),
    )
    return out
