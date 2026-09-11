"""
Finding the right video.

The previous pipeline picked a video by searching on artist + title and
trusting the top result. That fails constantly: live cuts, lyric videos,
covers, 8-bit remixes, reactions, and wrong artists all rank well, and chart
metadata in customs is frequently wrong to begin with.

This stage inverts the problem. Metadata only ever produces *candidates*; the
audio decides. For each song we pull several candidates, download AUDIO ONLY
for each (a few hundred KB, seconds per candidate), and fingerprint them
against the chart audio. Only the winner gets its video downloaded.

The cost asymmetry is the whole point: probing five candidates by audio is far
cheaper than downloading one wrong 1080p video, and it is the only method that
actually verifies identity rather than guessing at it.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio as au
from . import fingerprint as fp

N_CANDIDATES = 6              # results requested per search query
CANDIDATE_POOL = 12           # stop searching once this many unique survive
DURATION_TOLERANCE = 0.30      # candidate must be within +/-30% of chart length
MAX_CANDIDATE_SECONDS = 900    # skip hour-long uploads and full-album rips

# Title preferences. These decide which candidate WINS among those that pass
# the fingerprint gate, and the order they are tried in. They never let a
# candidate through the gate - the audio alone decides identity.
#
# The fingerprint cannot help here: a lyric video and the official video carry
# identical audio and both score in the thousands. Only the title separates
# them, and for a background-video project the distinction is the whole point.
PREFER_TERMS = (
    ("official music video", 6.0),
    ("official video", 5.0),
    ("music video", 3.0),
    ("official", 1.5),
)
# Terms meaning "not a music video". Note these must outweigh the generic
# "official" bonus: 'Official Audio' nets negative, which is correct - it is
# usually a static image and useless as a background.
AUDIO_RE = re.compile(
    r"\b(?:hd|hq|full|official|high[ -]?quality|album|studio)?\s*audio\b"
)
AUDIO_PENALTY = -9.0

# Auto-generated YouTube Music channels. Always static album art, never
# footage, and they carry the artist name so they defeat name matching.
TOPIC_PENALTY = -8.0

# Channels that carry the artist name but are not the artist: fan archives,
# tribute and reupload channels. These must not earn the official bonus.
NOT_OFFICIAL_MARKERS = (
    "archive", "fan", "tribute", "unofficial", "bootleg", "- topic",
    "lyrics", "reupload", "vault",
)

PENALISE_TERMS = (
    ("full album", -9.0), ("lyric", -3.5), ("visualizer", -3.0),
    ("visualiser", -3.0), ("live", -2.0), ("cover", -8.0),
    ("karaoke", -8.0), ("remix", -5.0), ("reaction", -9.0),
    ("instrumental", -6.0), ("8 bit", -9.0), ("acoustic", -4.0),
    ("tutorial", -9.0), ("guitar cover", -9.0), ("slowed", -6.0),
    # Documentaries about a video are not the video. "The Making Of The
    # Official Video" otherwise scores +5 for containing "official video".
    ("making of", -12.0), ("the making", -12.0), ("behind the scenes", -10.0),
    ("making the", -10.0), ("bts video", -10.0), ("bts footage", -10.0),
    ("nightcore", -9.0), ("playthrough", -8.0),
    # Rhythm-game and gameplay captures. These matter far more here than for a
    # general music library: every song in a YARG library is by definition a
    # song someone charted, so gameplay footage of that exact track is a
    # high-probability search result rather than a rare accident.
    ("rocksmith", -12.0), ("cdlc", -12.0), ("clone hero", -12.0),
    ("guitar hero", -12.0), ("rock band", -10.0), ("yarg", -12.0),
    ("beat saber", -12.0), ("osu!", -12.0), ("fretboard", -10.0),
    ("chart preview", -12.0), ("custom song", -10.0), ("gameplay", -12.0),
    ("expert+", -12.0), ("100% fc", -12.0), ("full combo", -12.0),
    ("drum cover", -10.0), ("bass cover", -10.0), ("midi", -9.0),
    # A montage cut to the song is not the song's video, and "Behind The
    # Curtain" is the making-of series under another name.
    ("behind the curtain", -10.0), ("montage", -6.0),
    # Rock Band disc abbreviations. These appear in the titles of chart
    # previews and gameplay captures that "rock band" alone does not catch.
    ("rb2", -10.0), ("rb3", -10.0), ("rb4", -10.0), ("rbn", -10.0),
)


def _channel_matches_artist(uploader: str, chart_text: str) -> bool:
    """True when the upload appears to come from the artist's own channel."""
    low = uploader.lower()
    if any(m in low for m in NOT_OFFICIAL_MARKERS):
        return False
    up = "".join(ch for ch in low if ch.isalnum())
    if not up:
        return False
    # chart_text is 'Artist Title'; the artist is the leading portion, so test
    # progressively longer prefixes rather than assuming a split point.
    words = [w for w in chart_text.lower().split() if w]
    for n in range(len(words), 0, -1):
        cand = "".join(ch for ch in "".join(words[:n]) if ch.isalnum())
        if len(cand) >= 4 and (cand in up or up in cand):
            return True
    return False


LABEL_MARKERS = (
    # "record" as a stem covers Records / Recording / Recordings / Record Co,
    # which the plural-only forms missed - "Domino Recording Co." read as a
    # stranger's channel.
    "record", "music group", "entertainment", "rec.", "label",
    "distribution", "musicgroup", "records",
)


def _looks_like_label(uploader: str) -> bool:
    """
    True for a record-label channel.

    Labels host a large share of official videos - Hopeless, Fueled By Ramen,
    Roadrunner, Rise, Epitaph - so matching only the artist's own name misses
    them entirely.
    """
    up = uploader.lower()
    return any(m in up for m in LABEL_MARKERS)


def title_preference(title: str, uploader: str = "", chart_text: str = "") -> float:
    """
    Heuristic ordering score. Never used to accept or reject a candidate.

    `chart_text` is the chart's own artist and title. Any penalty term that
    appears there is suppressed, because it describes the song rather than
    disqualifying the video. A chart called 'ringtone (remix)' should not
    penalise a video called 'ringtone (Remix)' - that IS the right recording.
    The same protects genuinely live, acoustic or instrumental charts, and
    bands whose name collides with a penalty word (the band Live).

    The uploader carries more signal than any title keyword. A band's own
    channel posting a visualizer is the official release; a stranger's channel
    posting something titled 'Official Video' frequently is not. The channel
    bonus is large enough to outweigh the visualizer and lyric penalties,
    because when a band's only release is a visualizer, that IS the video.
    """
    return title_score(title, chart_text) + channel_bonus(uploader, chart_text)


def is_not_a_video(title: str, uploader: str = "", chart_text: str = "") -> bool:
    """
    True when the background will be a STATIC IMAGE rather than moving footage.

    This is deliberately narrow. Live sets, montages, lyric videos and
    visualizers are all motion and make perfectly good backgrounds, so
    flagging them is noise - and a visualizer on the artist's own channel is
    frequently the official release, with nothing better in existence. Only
    two things reliably mean album art on screen for the whole song: an audio
    upload, and an auto-generated '- Topic' channel.

    Ranking still prefers a real music video over all of these; the flag only
    answers "will I be staring at a still image".
    """
    if "- topic" in uploader.lower():
        return True
    t = title.lower()
    return bool(AUDIO_RE.search(t) and not AUDIO_RE.search(chart_text.lower()))


def title_score(title: str, chart_text: str = "") -> float:
    """
    How much the TITLE suggests a real music video. Drives the review flag.

    Kept separate from the channel bonus on purpose. An official channel
    posting a static audio track is still not a video, and letting the channel
    bonus cancel the audio penalty silently removed those review flags.
    """
    t = title.lower()
    chart = chart_text.lower()

    # A guest credit the chart does not have means a different cut of the
    # song. "Levitating Featuring DaBaby" against a chart of plain
    # "Levitating" is a different arrangement, and no offset reconciles them.
    feat = ("feat.", "feat ", "featuring", "ft.", "ft ", "with ")
    if any(k in t for k in feat) and not any(k in chart for k in feat):
        return sum(w for term, w in PREFER_TERMS if term in t) - 8.0

    score = sum(w for term, w in PREFER_TERMS if term in t)
    score += sum(
        w for term, w in PENALISE_TERMS
        if term in t and term not in chart
    )
    if AUDIO_RE.search(t) and not AUDIO_RE.search(chart):
        score += AUDIO_PENALTY
    return score


def channel_bonus(uploader: str, chart_text: str = "") -> float:
    """How much the CHANNEL suggests an official release. Drives ranking."""
    low = uploader.lower()
    if "- topic" in low:
        return TOPIC_PENALTY
    if "vevo" in low:
        return 4.0
    if _channel_matches_artist(uploader, chart_text):
        return 7.0
    if _looks_like_label(uploader):
        return 3.0
    return 0.0

# --------------------------------------------------------------- the policy ---
#
# The shipped rule added the channel bonus to the title score, so a big enough
# title number swamped the channel: a stranger's "Official Music Video" (15.5)
# beat the label's "Official Video" on its own channel (6.5 + 3). The policy
# below ranks lexicographically instead, so the channel is decided before the
# title is ever read.

FAN_MV_MARK = "fan MV available"

# The positive `channel_bonus` values, ascending: a label, VEVO, the artist's
# own channel. Read as thresholds rather than matched as values, so a new
# official channel kind lands in the right tier by its bonus alone.
OFFICIAL_TIERS = (3.0, 4.0, 7.0)


def channel_class(uploader: str, chart_text: str = "",
                  tiered: bool = False) -> int:
    """
    Which class of channel this is. Higher is better; ranked before the title.

    Derived from `channel_bonus` rather than re-testing the uploader, so the
    two can never disagree about what counts as official. Flat (the default):
    an auto-generated Topic channel is 0, a stranger 1, anything official 2.
    `tiered` splits that top class the way the bonus already orders it -
    label 2, VEVO 3, the artist's own channel 4.
    """
    bonus = channel_bonus(uploader, chart_text)
    if bonus < 0:
        return 0
    if bonus == 0:
        return 1
    if not tiered:
        return 2
    return 2 + sum(1 for tier in OFFICIAL_TIERS if bonus > tier)


def policy_key(c: "Candidate", chart_text: str = "",
               tiered: bool = False) -> tuple[int, float, int]:
    """
    The ranking key: channel class, then title, then views.

    The fingerprint score is deliberately NOT in here. It is identity, not
    preference - everything being ranked has already cleared the gate - and
    rankers append it themselves as the final tie-break. An unknown view
    count sorts as -1, below a genuine zero: a row stored before the column
    existed is not evidence of an unwatched video.
    """
    return (
        channel_class(c.uploader, chart_text, tiered),
        title_score(c.title, chart_text),
        c.view_count if c.view_count is not None else -1,
    )


def _rank(candidates: list["Candidate"], chart_text: str,
          tiered: bool) -> list["Candidate"]:
    passers = [
        c for c in candidates
        if c.score >= fp.ACCEPT_SCORE and c.coverage >= fp.ACCEPT_COVERAGE
    ]
    passers.sort(key=lambda c: (policy_key(c, chart_text, tiered), c.score),
                 reverse=True)
    return passers


def rank_official(candidates: list["Candidate"],
                  chart_text: str) -> list["Candidate"]:
    """One official class: artist, label and VEVO are decided by their titles."""
    return _rank(candidates, chart_text, False)


def rank_artist_first(candidates: list["Candidate"],
                      chart_text: str) -> list["Candidate"]:
    """Three official classes: artist > VEVO > label, whatever they posted."""
    return _rank(candidates, chart_text, True)


RANKERS = {"official": rank_official, "artist_first": rank_artist_first}

# Which of them `pick_best` applies. The bench decides this; both stay
# registered so the loser remains measurable.
POLICY = "official"


def fan_mv(ranked: list["Candidate"], chart_text: str = "") -> "Candidate | None":
    """
    The third-party music video that lost to an official non-music video.

    The pick is right by policy - a band's own visualizer is the official
    release - and a stranger's upload of the actual video is still worth a
    look, so the song is flagged rather than re-ranked. A plain title scores
    zero and says nothing either way: it is never the flag and never flagged.
    """
    if not ranked:
        return None
    top = ranked[0]
    if channel_bonus(top.uploader, chart_text) <= 0:
        return None
    if title_score(top.title, chart_text) >= 0:
        return None
    for c in ranked[1:]:
        if (channel_bonus(c.uploader, chart_text) == 0
                and title_score(c.title, chart_text) > 0):
            return c
    return None


QUERY_TEMPLATES = [
    "{artist} {title} official music video",
    "{artist} {title} official video",
    "{artist} {title}",
]


@dataclass
class Candidate:
    video_id: str
    title: str
    uploader: str
    duration: float
    score: float = 0.0
    offset_s: float = 0.0
    coverage: float = 0.0
    # Comes back with every search result at no extra cost. Nothing ranks on
    # it yet; it is stored so a policy can be measured against it later.
    view_count: int | None = None


def parse_video_id(url_or_id: str) -> str | None:
    """
    Accept a YouTube URL in any common shape, or a bare 11-character ID.

    Users paste whatever the browser gave them: watch links, share links,
    embeds, links with playlist or timestamp parameters. All resolve to the
    same ID.
    """
    s = url_or_id.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", s):
        return s
    patterns = (
        r"[?&]v=([A-Za-z0-9_-]{11})",
        r"youtu\.be/([A-Za-z0-9_-]{11})",
        r"/embed/([A-Za-z0-9_-]{11})",
        r"/shorts/([A-Za-z0-9_-]{11})",
        r"/live/([A-Za-z0-9_-]{11})",
    )
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return m.group(1)
    return None


def fetch_metadata(video_id: str, cookies: str | None = None) -> dict:
    cmd = [
        "yt-dlp", f"https://www.youtube.com/watch?v={video_id}",
        "--dump-json", "--skip-download", "--no-warnings", "--no-playlist",
    ]
    cmd += cookie_args(cookies)
    proc = _run(cmd, timeout=180)
    if proc.returncode != 0:
        return {}
    try:
        return json.loads(proc.stdout.strip().splitlines()[0])
    except (json.JSONDecodeError, IndexError):
        return {}


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """
    Run a command and report failure the way callers already expect it.

    A hung yt-dlp or a missing binary raised TimeoutExpired / FileNotFoundError
    straight out of here and killed the whole batch on one bad song, losing
    every result after it. Every caller already branches on returncode, so a
    failure to finish is returned as one rather than thrown.
    """
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            cmd, 124, "", f"{cmd[0]} timed out after {timeout}s")
    except OSError as e:
        # FileNotFoundError when the tool is not installed, and anything else
        # the OS refuses at launch (permissions, a broken PATH entry).
        return subprocess.CompletedProcess(
            cmd, 127, "", f"could not run {cmd[0]}: {e}")


def _sleep_args(sleep: float) -> list[str]:
    """Pace requests to YouTube. Cheap insurance across thousands of songs."""
    return ["--sleep-requests", str(sleep)] if sleep and sleep > 0 else []


def cookie_args(cookies: str | None) -> list[str]:
    """
    Build yt-dlp cookie flags from a single user-facing option.

    Accepts either a browser name ('firefox') or a path to a Netscape-format
    cookies.txt file. The distinction matters on Windows: since Chrome 127 all
    Chromium-derived browsers (Chrome, Edge, Brave, Opera, Vivaldi) encrypt
    cookies with a key bound to the browser process, so yt-dlp cannot read
    them at all, closed or not. Firefox stores cookies in plain SQLite and
    works directly. For a Chromium browser, export a cookies.txt with a
    browser extension and pass that path here instead.
    """
    if not cookies:
        return []
    path = Path(cookies).expanduser()
    if path.is_file():
        return ["--cookies", str(path)]
    return ["--cookies-from-browser", cookies]


def _views(raw) -> int | None:
    """
    The search result's view count, or None.

    A flat-playlist entry omits the field for some uploads and occasionally
    carries a string. Neither is an error worth failing a search over, so both
    read as "not known" rather than zero - zero is a real view count.
    """
    if raw is None:
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def search_candidates(
    artist: str, title: str, chart_seconds: float,
    cookies: str | None = None, sleep: float = 0.0,
) -> list[Candidate]:
    seen: dict[str, Candidate] = {}

    for template in QUERY_TEMPLATES:
        query = template.format(artist=artist, title=title)
        cmd = [
            "yt-dlp",
            f"ytsearch{N_CANDIDATES}:{query}",
            "--flat-playlist", "--dump-json",
            "--no-warnings", "--ignore-errors",
        ]
        cmd += cookie_args(cookies)
        cmd += _sleep_args(sleep)

        proc = _run(cmd, timeout=180)
        for line in proc.stdout.splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            vid = d.get("id")
            dur = float(d.get("duration") or 0)
            if not vid or vid in seen or dur <= 0 or dur > MAX_CANDIDATE_SECONDS:
                continue
            # Duration prefilter - free, and removes most obvious mismatches.
            if chart_seconds > 0:
                ratio = abs(dur - chart_seconds) / chart_seconds
                if ratio > DURATION_TOLERANCE:
                    continue
            seen[vid] = Candidate(
                video_id=vid,
                title=d.get("title") or "",
                uploader=d.get("uploader") or d.get("channel") or "",
                duration=dur,
                view_count=_views(d.get("view_count")),
            )
        if len(seen) >= CANDIDATE_POOL:
            break

    return list(seen.values())[:CANDIDATE_POOL]


def probe_audio(
    video_id: str, work_dir: Path, cookies: str | None = None, sleep: float = 0.0,
) -> tuple[Path | None, str]:
    """
    Download the smallest available audio stream for a candidate.

    Returns (path, error). The error string matters: a failed download and a
    failed fingerprint are completely different problems, and collapsing them
    into one message makes the pipeline undiagnosable.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    out = work_dir / f"{video_id}.probe.%(ext)s"
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f", "worstaudio/bestaudio",
        "-o", str(out),
        "--no-warnings", "--no-playlist", "--no-part",
    ]
    cmd += cookie_args(cookies)
    cmd += _sleep_args(sleep)

    proc = _run(cmd, timeout=600)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().replace("\n", " ")
        return None, err[:200] or f"yt-dlp exit {proc.returncode}"

    hits = list(work_dir.glob(f"{video_id}.probe.*"))
    if not hits:
        return None, "yt-dlp reported success but wrote no file"
    return hits[0], ""


def pick_best(
    chart_audio: np.ndarray,
    artist: str,
    title: str,
    work_dir: Path,
    cookies: str | None = None,
    sleep: float = 0.0,
) -> tuple[Candidate | None, list[Candidate], str]:
    """
    Returns (winner, all_scored_candidates, reason).

    A winner has to clear the fingerprint gate - score and coverage - and
    nothing else. There is no comparison against the runner-up: two candidates
    scoring alike are usually the same recording uploaded twice, and either
    will do. Among everything that clears the gate the winner is the one with
    the highest title preference, with the score breaking ties.
    """
    chart_seconds = chart_audio.size / fp.SR
    if chart_seconds <= 0:
        return None, [], "no chart audio"

    candidates = search_candidates(artist, title, chart_seconds, cookies, sleep)
    if not candidates:
        return None, [], "no candidates passed the duration filter"

    chart_hashes = fp.make_hashes(chart_audio)
    probed = 0
    last_error = ""

    # Try the most promising titles first so the usual case costs one download
    # instead of ten. Ordering is a heuristic; acceptance still requires the
    # fingerprint gate below.
    chart_text = f"{artist} {title}"
    tiered = POLICY == "artist_first"
    candidates.sort(
        key=lambda c: policy_key(c, chart_text, tiered),
        reverse=True,
    )

    prefs = [policy_key(c, chart_text, tiered) for c in candidates]
    best: Candidate | None = None
    # The empty tuple is below every key, so the first gate passer always
    # takes it. A zero would have been a real preference, and a candidate
    # ranking below zero could never have become the best.
    best_pref: tuple = ()

    for i, cand in enumerate(candidates):
        path, err = probe_audio(cand.video_id, work_dir, cookies, sleep)
        if path is None:
            last_error = err
        else:
            probed += 1
            try:
                samples = au.decode_mono(path, fp.SR)
                if samples.size == 0:
                    last_error = "downloaded file had no decodable audio"
                else:
                    res = fp.match(
                        chart_hashes, fp.make_hashes(samples), chart_seconds
                    )
                    cand.score = res.score
                    cand.offset_s = res.offset_seconds
                    cand.coverage = res.coverage
            finally:
                path.unlink(missing_ok=True)

            if (
                cand.score >= fp.ACCEPT_SCORE
                and cand.coverage >= fp.ACCEPT_COVERAGE
                and (best is None or (prefs[i], cand.score) > (best_pref, best.score))
            ):
                best, best_pref = cand, prefs[i]

        # Stop once something has passed the gate and nothing left outranks it.
        # This must compare against the best candidate found SO FAR, not just
        # look ahead: Knife Party's official video passed at position 0 but a
        # tie prevented stopping there, and a later, much weaker candidate then
        # found nothing above it remaining and would have won by default.
        # One exception: when the best is an official channel's non-music
        # video, a third-party candidate titled as the music video is worth
        # the download even though it cannot win. It is what `fan_mv` flags,
        # and the flag can only fire on a candidate that was actually heard.
        if best is not None and all(p < best_pref for p in prefs[i + 1:]):
            if (title_score(best.title, chart_text) < 0
                    and any(channel_bonus(c.uploader, chart_text) == 0
                            and title_score(c.title, chart_text) > 0
                            for c in candidates[i + 1:])):
                continue
            break

    # No candidate was ever heard. This is a download problem, not a matching
    # problem, and lowering the fingerprint threshold would not help at all.
    if probed == 0:
        return None, candidates, (
            f"DOWNLOAD FAILED for all {len(candidates)} candidates - "
            f"no audio was compared. yt-dlp said: {last_error or 'unknown'}"
        )

    # `best` was tracked inside the loop by (preference, score) among everything
    # that cleared the gate. Deliberately not recomputed here: the loop can stop
    # early, so a second pass over `candidates` would rank unprobed entries with
    # their default score of 0.0 and could disagree with the loop's own choice.
    if best is None:
        top = max(candidates, key=lambda c: c.score)
        return None, candidates, (
            f"no audio match among {probed} downloaded candidates "
            f"(best score {top.score:.1f}, coverage {top.coverage:.2f})"
        )

    candidates.sort(key=lambda c: c.score, reverse=True)

    # The pick is right by policy and there is still a music video to look at.
    # This outranks the static-image flag: both say "come and look", and the
    # fan MV names the video to look at.
    alt = fan_mv(RANKERS[POLICY](candidates, chart_text), chart_text)
    if alt is not None:
        return best, candidates, f"ok ({FAN_MV_MARK}: {alt.video_id} - review)"

    # It passed the gate, but the background will be a still image rather than
    # footage. Better than nothing, so keep it - and flag it for review.
    if is_not_a_video(best.title, best.uploader, chart_text):
        return best, candidates, "ok (static image - review)"

    return best, candidates, "ok"


def download_video(
    video_id: str, dest: Path, max_height: int = 1080,
    cookies: str | None = None, sleep: float = 0.0,
) -> tuple[Path | None, str]:
    """
    Fetch the winning video at up to `max_height`, WITH its audio track.

    Returns (path, note). The note is empty on success and carries the failure
    reason otherwise: a pulled video, a format problem and a network blip all
    exit non-zero, and collapsing them into None left the download stage
    recording 'failed' with no way to tell which it had been.

    The audio is not kept in the final webm - encode.py strips it with `-an`
    because YARG plays the chart stems. But the sync stage has to hear the
    video to align it, so the source must carry audio.

    `bestvideo` alone is a video-only DASH stream on YouTube. Using it here
    produces a silent file, and every song then fails sync with "empty audio".
    The `+bestaudio` is load-bearing.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    # Remove any previous source. Otherwise a leftover from an earlier match
    # can be picked up by the glob below when the new download uses a
    # different container extension.
    for stale in dest.parent.glob(f"{dest.stem}.src.*"):
        stale.unlink(missing_ok=True)
    out = dest.parent / f"{dest.stem}.src.%(ext)s"
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={video_id}",
        "-f", (
            f"bestvideo[height<={max_height}]+bestaudio/"
            f"best[height<={max_height}]/best"
        ),
        "--merge-output-format", "mkv",
        "-o", str(out),
        "--no-warnings", "--no-playlist", "--no-part",
    ]
    cmd += cookie_args(cookies)
    cmd += _sleep_args(sleep)

    proc = _run(cmd, timeout=3600)
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().replace("\n", " ")
        return None, err[:200] or f"yt-dlp exit {proc.returncode}"

    hits = list(dest.parent.glob(f"{dest.stem}.src.*"))
    if not hits:
        return None, "yt-dlp reported success but wrote no file"

    # Fail loudly here rather than letting a silent file reach sync, where the
    # only symptom is a useless "empty audio" rejection one stage later.
    src = hits[0]
    info = au.probe(src)
    streams = info.get("streams", [])
    has_video = any(s.get("codec_type") == "video" for s in streams)
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not (has_video and has_audio):
        src.unlink(missing_ok=True)
        return None, "no audio or video stream in the downloaded file"
    return src, ""
