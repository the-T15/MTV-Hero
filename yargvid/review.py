"""
Local review app.

Checking 1500 songs by hand is not viable, so the point of this is not a
player - it is the ordering. Everything the pipeline knows about its own
uncertainty (fingerprint score near the gate, a third-party channel, excerpt
spread above zero, an implausible offset, drift, a static background) is
combined into one risk number, and the worklist is sorted by it. The songs
where something actually went wrong cluster at the top; a song matched on the
artist's own channel scoring 4000 with 0 ms spread does not need a human.

Playback avoids the two-clock problem entirely. Rather than syncing a <video>
element against separate <audio> elements in the browser - which is only
accurate to a video frame, and is exactly the thing under review - ffmpeg
bakes the offset into a short proof clip: the video seeked to its aligned
position, muxed with the mixed chart stems. If the clip looks in time, the
offset is right, because the clip was built the same way YARG will play it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from . import audio as au
from . import encode as enc
from . import fingerprint as fp
from .db import Database
from .match import FAN_MV_MARK

# 15 s, not 12: twelve was long enough to see a beat land and too short to
# see one drift away from the beat after it.
SEGMENT_SECONDS = 15
# The last window is twice as long. The end of the song is where drift has
# had the whole track to accumulate and where a short video runs out, and
# fifteen seconds there was not long enough to watch either happen.
END_SECONDS = 30
# An end window shorter than this is not worth building: it shows less than
# a phrase, and the intro window already covers that much of the footage.
MIN_END_SECONDS = 5
CLIP_HEIGHT = 360
FULL_HEIGHT = 360


# ------------------------------------------------------------------ risk ----

# Short keys so the interface can group and filter, paired with the sentence
# a person actually reads. A bare list of prose was unfilterable and only the
# first line ever showed, which said nothing about the shape of the queue.
TAG_LABELS = {
    "replaced":  "user override",
    "weak":      "weak match",
    "channel":   "third-party",
    "audio":     "audio only",
    "fan_mv":    "fan MV available",
    "flat":      "no clear alignment",
    "unverified": "unverified",
    "unsteady":  "unsteady",
    "shift":     "big shift",
    "drift":     "speed drift",
    "short":     "video ends early",
    "still":     "still image",
    "low_motion": "low motion",
    "existing":  "has a video",
    "clean":     "nothing unusual",
}
TAG_ORDER = list(TAG_LABELS)

# The tiles the worklist is divided into, left to right, and the one line
# each of them is allowed to say for itself.
TILES = ("clean", "unsure", "third", "still", "override", "existing",
         "approved", "later")
TILE_LABELS = {
    "clean":    "Nothing unusual",
    "unsure":   "Unsure",
    "third":    "Third party",
    "still":    "Still image",
    "override": "User override",
    "existing": "Video already in folder",
    "approved": "Approved",
    "later":    "Save for later",
}

# The tags that mean "this might be the wrong video or the wrong time", as
# opposed to the ones that describe what the video is. Together they are the
# Unsure tile, and inside it they are the filter chips.
DOUBT_TAGS = frozenset({"weak", "unverified", "unsteady", "shift", "drift",
                        "audio", "short", "flat", "low_motion", "fan_mv"})

# The top of the band that is moving but barely: a slideshow, a lyric card
# with a drifting background, a single locked-off camera. A guess until the
# motion values of the approved set say where real footage actually sits -
# take the bound from those before trusting this number.
LOW_MOTION_MAX = 0.30


def bucket(song) -> str:
    """
    The one tile a song belongs to.

    First match wins, so no song is in two places and no count double-reports
    it. The old tabs were not a partition: a still image on a third-party
    channel was in two counts and one list, and which list depended on the tab
    you happened to be looking at.

    Decisions outrank measurements. Approved, saved and overridden are things
    you did, and they are the reason you would go looking for the song again;
    everything below them is something the pipeline measured.
    """
    review = song.get("review")
    if review == "keep":
        return "approved"
    if review == "later":
        return "later"
    tags = set(song.get("tags") or ())
    if "replaced" in tags:
        return "override"
    if song.get("existing_video"):
        return "existing"
    if song.get("static"):
        return "still"
    if "channel" in tags:
        return "third"
    if tags & DOUBT_TAGS:
        return "unsure"
    return "clean"


def fmt_mmss(seconds: float) -> str:
    """A number of seconds as minutes and seconds."""
    whole = int(round(max(0.0, seconds)))
    return f"{whole // 60}:{whole % 60:02d}"


def fmt_ms(value) -> str:
    """
    A millisecond figure as a person should read it.

    A spread of -1 is the sentinel for "no window agreed", not a measurement:
    both interfaces printed it as "-1 ms", which reads as an agreement tighter
    than perfect rather than as no agreement at all.
    """
    if value is None or value < 0:
        return "unverified"
    return f"{value:.0f} ms"


@dataclass
class Risk:
    score: float
    flags: list[tuple[str, str]]          # (tag, sentence)

    @property
    def tags(self) -> list[str]:
        return [t for t, _ in self.flags]

    @property
    def reasons(self) -> list[str]:
        return [r for _, r in self.flags]


def assess(row) -> Risk:
    """Combine what the pipeline knows about its own uncertainty."""
    pts = 0.0
    why: list[tuple[str, str]] = []

    note = row["match_note"] or ""
    if note.startswith("MANUAL:"):
        # Only once it has been confirmed does a manual pick become the most
        # trusted thing here. Before that it is a brand new video with a brand
        # new offset that nobody has watched - the least verified song there
        # is, and burying it was backwards.
        #
        # 'later' is the opposite of a confirmation: it is the mark for a song
        # put off precisely because it had not been decided. Any truthy review
        # counted as checked, so those sank to the bottom of the queue.
        if row["review"] == "keep":
            return Risk(-10.0, [("clean", "you chose this one, checked")])
        return Risk(9.0, [("replaced",
                           "you replaced this - check the new video and timing")])

    fps = row["fp_score"]
    if fps is not None and fps > 0:
        margin = fps - fp.ACCEPT_SCORE
        if margin < 15:
            pts += 4
            why.append(("weak",
                        f"fingerprint {fps:.0f}, near the {fp.ACCEPT_SCORE:.0f} cutoff"))
        elif margin < 60:
            pts += 1.5
            why.append(("weak", f"fingerprint {fps:.0f}"))

    # Channel: the artist's own or a label reads as official; anything else is
    # a stranger's upload and worth a look.
    chan = note.rsplit("[", 1)[-1].rstrip("]") if "[" in note else ""
    if chan:
        from .match import _channel_matches_artist, _looks_like_label
        artist = row["artist"] or ""
        known = (
            _channel_matches_artist(chan, f"{artist} {row['title'] or ''}")
            or _looks_like_label(chan)
            or "vevo" in chan.lower()
        )
        if not known:
            pts += 3
            why.append(("channel", f"third-party channel: {chan}"))

    # `match` writes the REVIEW: prefix when the winning title says lyric
    # video, audio upload or gameplay capture - the uploader telling you
    # there is nothing to watch, before motion is ever measured. It reached
    # the row and then stopped: nothing in either interface read it. One
    # point, because the timing can still be perfect; it is the picture that
    # is not worth having.
    if note.startswith("REVIEW:"):
        pts += 1
        if FAN_MV_MARK in note:
            # Not the same complaint. The pick is an official channel's
            # visualizer or lyric video - a real picture, correctly chosen -
            # and somewhere below it a stranger uploaded the music video.
            why.append(("fan_mv",
                        "a third-party music video is available - compare"))
        else:
            why.append(("audio", "titled as audio, not a video"))

    # Peak dominance is recorded but deliberately NOT ranked on. Measured
    # across 172 songs, 21 of the 71 confirmed correct by eye fell below 5x -
    # including two the user called perfect - so a low figure does not mean a
    # wrong offset. Kept in the export for future analysis, kept out of the
    # queue where it would have buried a third of the good songs.

    spread = row["spread_ms"]
    if row["sync_status"] == "unverified" or (spread is not None and spread < 0):
        pts += 5
        why.append(("unverified", "the offset could not be confirmed anywhere "
                                  "in the track - it may be a wrong lock"))
    elif spread is not None and spread > 10:
        pts += 2 if spread > 25 else 1
        why.append(("unsteady", f"{spread:.0f} ms disagreement across the track"))

    off = row["offset_ms"]
    if off is not None and abs(off) > 15000:
        pts += 2
        why.append(("shift",
                    f"{abs(off) / 1000:.0f}s shift - check it starts in the right place"))

    if row["sync_status"] == "drift":
        pts += 4
        why.append(("drift", "video runs at a different speed"))

    # The song outlasting the footage is a fact about the file, not a doubt
    # about the match, and nothing in the window said it - you found out by
    # watching the background stop. Five seconds of slack: an outro fading
    # under a black frame is not worth flagging.
    chart = row["chart_seconds"] or 0.0
    vid = row["video_seconds"]
    if vid and chart > 0:
        missing = chart - video_reach(vid, row["offset_ms"] or 0.0)
        if missing > 5.0:
            pts += 2
            why.append(("short",
                        f"video ends {fmt_mmss(missing)} before the song"))

    motion = row["motion"]
    if motion is not None and enc.is_static(motion):
        pts += 0.5
        why.append(("still", "still image, not footage"))
    elif motion is not None and enc.STATIC_THRESHOLD <= motion < LOW_MOTION_MAX:
        # A tag, not a verdict. `is_static` already excludes the negative
        # sentinel, so this band is only ever a measurement that came back.
        pts += 0.5
        why.append(("low_motion",
                    f"little motion ({motion:.2f}) - may be a slideshow or a "
                    "lyric video"))

    # Not doubt about the match - a note about what is at stake. It rides in
    # the reasons because the status line it used to own is overwritten by the
    # clip-building text a moment later, so nobody ever read it. It carries no
    # points: an existing video says nothing about whether this one is right.
    #
    # 'foreign' only. The status line this replaced tested for that value; any
    # truthy existing_video also covers videos this pipeline put there itself,
    # and telling you to compare against our own preview is nonsense.
    if row["existing_video"] == "foreign":
        why.append(("existing",
                    "this folder already held a video from before this "
                    "project - encoding replaces it, so compare first"))

    if not why:
        why.append(("clean", "nothing unusual"))
    return Risk(pts, why)


def queue(db: Database) -> list[dict]:
    rows = db.conn.execute(
        "SELECT * FROM songs WHERE sync_status IN ('ok', 'drift', 'unverified') "
        "ORDER BY song_dir"
    ).fetchall()
    out = []
    for r in rows:
        risk = assess(r)
        note = r["match_note"] or ""
        spread = None if r["sync_status"] == "unverified" else r["spread_ms"]
        out.append({
            "song_dir": r["song_dir"],
            "artist": r["artist"] or "",
            "title": r["title"] or "",
            "video": (note.split(" [")[0]
                      .removeprefix("REVIEW: ").removeprefix("MANUAL: ")),
            "channel": note.rsplit("[", 1)[-1].rstrip("]") if "[" in note else "",
            "offset_ms": r["offset_ms"],
            "spread_ms": r["spread_ms"],
            "spread_text": fmt_ms(spread),
            "fp_score": r["fp_score"],
            "motion": r["motion"],
            "dominance": r["dominance"],
            "existing_video": r["existing_video"],
            "static": r["motion"] is not None and enc.is_static(r["motion"]),
            "review": r["review"],
            "updated_at": r["updated_at"],
            "chart_seconds": r["chart_seconds"],
            "video_seconds": r["video_seconds"],
            "reach_seconds": (video_reach(r["video_seconds"],
                                          r["offset_ms"] or 0.0)
                              if r["video_seconds"] else None),
            "segment_starts": segment_starts(r, r["video_seconds"]),
            "risk": round(risk.score, 1),
            "reasons": risk.reasons,
            "tags": risk.tags,
            "video_id": r["video_id"],
        })
    # 'later' is a decision to defer, not an approval, so those stay live in
    # their own list rather than sinking with the finished ones.
    out.sort(key=lambda d: (d["review"] == "keep", -d["risk"], d["artist"]))
    return out


def drop_song(db: Database, song: Path) -> None:
    """
    Record that this song wants no background video, and delete what was got.

    Every measurement in the row describes the video being deleted. Left
    behind - as `fp_score`, `dominance`, `drift_ppm`, `sync_note`, `windows`
    and `encode_note` all were - they read afterwards as current figures for a
    video that is not on disk and is never coming back.
    """
    song = Path(song)
    row = db.conn.execute(
        "SELECT source_path FROM songs WHERE song_dir = ?", (str(song),)
    ).fetchone()
    if row and row["source_path"]:
        Path(row["source_path"]).unlink(missing_ok=True)
    # Whichever codec wrote it, and whatever the source arrived as.
    for name in enc.OUTPUT_NAMES:
        (song / name).unlink(missing_ok=True)
    for leftover in song.glob("video.src.*"):
        leftover.unlink(missing_ok=True)
    # 'skipped' is not 'ok', so no later stage will queue it again.
    db.update(song, match_status="skipped",
              match_note="MANUAL: no video wanted",
              video_id=None, source_path=None,
              download_status="pending", sync_status="pending",
              offset_ms=None, spread_ms=None, motion=None,
              fp_score=None, dominance=None, drift_ppm=None,
              sync_note=None, windows=None,
              encode_status="pending", encode_note=None,
              ini_status="pending", review=None)


# ------------------------------------------------------------------ clip ----

def source_video(row) -> Path | None:
    """The downloaded source if still present, else the encoded output."""
    src = row["source_path"]
    if src and Path(src).exists():
        return Path(src)
    return enc.find_output(Path(row["song_dir"]))


def _sha8(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def _state(row) -> str:
    """
    What the clip's contents depend on: the offset and the video file.

    Anything that changes either of these makes every clip already on disk a
    picture of a different alignment, so it has to change the name.
    """
    video = source_video(row)
    size = video.stat().st_size if video is not None else 0
    return _sha8(f"{video}|{size}|{round(row['offset_ms'] or 0.0, 3)}")


def clip_name(row) -> str:
    """
    Name a proof clip after the song and the alignment it shows.

    sha1, not hash(): hash() of a string is salted per process, so the same
    song named a different file every launch and nothing was ever a cache hit.
    The two halves are separate on purpose - the song half finds this song's
    other clips, the state half says whether one of them is still current.
    """
    return f"clip_{_sha8(row['song_dir'])}_{_state(row)}.mp4"


def full_name(row) -> str:
    """
    The whole-song build, named on the same two halves as `clip_name`.

    Matroska, because the video stream is copied rather than re-encoded and
    whatever YouTube sent - VP9, AV1, h264 - has to go in the container as
    it is.
    """
    return f"full_{_sha8(row['song_dir'])}_{_state(row)}.mkv"


def purge_stale(work: Path, keep: Path) -> None:
    """
    Delete this song's files from earlier alignments, keeping `keep` and any
    other file describing the same one.

    A clip is only true of the offset and source video it was built from, so
    once either moves every older file for that song is a wrong answer taking
    up room - 371 of them, 498 MB, before the name carried the state.
    """
    parts = keep.name.split("_")
    if len(parts) < 3:
        return
    song, state = parts[1], parts[2].split(".")[0]
    for f in work.glob(f"*_{song}_*"):
        bits = f.name.split(".")[0].split("_")
        if not f.is_file() or len(bits) < 3 or bits[2] == state:
            continue
        try:
            f.unlink()
        except OSError:
            pass          # a player still holds it open; it goes next time


def video_reach(video_seconds: float, offset_ms: float) -> float:
    """
    The song time at which the video runs out.

    The offset is subtracted, not added: a video held back (negative offset)
    is still playing after its own duration has elapsed, so it reaches
    further into the song than its length suggests.
    """
    return video_seconds - (offset_ms or 0.0) / 1000.0


def segment_windows(row, video_seconds: float | None = None
                    ) -> list[tuple[float, float]]:
    """
    Where in the song each window begins and how long it runs.

    Sampling at 25/55/85% of the song put all three windows in the body of
    the track, which is where choruses are - the most repetitive part, where
    a wrong lock looks most convincing. It never showed the lead-in, which is
    where the offset is easiest to judge, and never showed the last bar,
    which is where drift has had the whole song to accumulate and where a
    video that runs out shows itself.

    So: one window a second after the footage starts, one ending a second
    before it ends, one halfway between. The end window is `END_SECONDS`
    rather than `SEGMENT_SECONDS` - fifteen seconds of the run-out was not
    long enough to see drift arrive or the picture stop.

    Both ends come from the same two numbers, so no window can be placed
    outside the footage by construction. A window is kept only if it is a
    full window clear of the one before it: a short song gets two windows or
    one rather than three overlapping views of the same ten seconds. When the
    intro and the end collide, the end window shrinks into what is left
    rather than moving - the run-out is the part of it worth watching.
    """
    offset = (row["offset_ms"] or 0) / 1000.0
    chart = row["chart_seconds"] or 0.0
    begins = max(0.0, -offset)          # song time the footage starts at
    ends = chart if chart > 0 else begins + SEGMENT_SECONDS + 2.0
    if video_seconds:
        # Where the footage stops, if that is before the song does.
        ends = min(ends, video_reach(video_seconds, offset * 1000.0))

    first = begins + 1.0
    stop = ends - 1.0                   # the last window finishes here
    if stop - SEGMENT_SECONDS < first:
        # Too little footage to put a window at each end of it. Take the one
        # that fits, as late as it can sit without running past the end.
        start = max(begins, min(first, ends - SEGMENT_SECONDS))
        return [(start, float(SEGMENT_SECONDS))]

    out: list[tuple[float, float]] = [(first, float(SEGMENT_SECONDS))]
    floor = first + SEGMENT_SECONDS
    end_start, end_len = stop - END_SECONDS, float(END_SECONDS)
    if end_start < floor:
        end_start, end_len = floor, stop - floor
        if end_len < MIN_END_SECONDS:
            return out

    middle = (first + end_start) / 2.0
    if middle >= floor and end_start - middle >= SEGMENT_SECONDS:
        out.append((middle, float(SEGMENT_SECONDS)))
    out.append((end_start, end_len))
    return out


def segment_starts(row, video_seconds: float | None = None) -> list[float]:
    """Song times the windows begin at, without their lengths."""
    return [s for s, _ in segment_windows(row, video_seconds)]


def build_clip(row, work: Path) -> Path | None:
    """
    Mux each window of the aligned video against the mixed chart stems.

    Sync is correct by construction: the offset is applied by seeking, so
    there are no two clocks to drift apart. What plays is what YARG will show.

    Three windows are three files. Joining them was work spent defending a
    join nothing needed - a segment that came back short broke the container
    it was concatenated into, and the buttons then seeked into an index that
    was not there. Separate files cannot do that to each other, so a bad
    piece costs its own window and nothing else. The first one is returned;
    the sidecar names them all.
    """
    video = source_video(row)
    stems = au.find_stems(Path(row["song_dir"]))
    if video is None or not stems:
        return None

    offset = (row["offset_ms"] or 0) / 1000.0
    video_seconds = au.duration_of(video)
    work.mkdir(parents=True, exist_ok=True)
    out = work / clip_name(row)
    # Before the cache check, not after: a hit is only a hit because the name
    # already says this clip matches the current offset and video, and the
    # same reasoning condemns everything else this song left behind.
    purge_stale(work, out)
    # A hit is the sidecar plus every file it names. The sidecar alone is not
    # enough: one segment deleted by hand, or left half-written by a crash,
    # would otherwise be reported as a built clip with a file missing under it.
    cached = clip_info(out).get("files") or []
    if cached and all((work / name).is_file() for name in cached):
        return work / cached[0]
    key = out.stem

    # Renders that share nothing, on a machine with more than one core: run
    # them together. The wait before a song appears is the whole cost of
    # reviewing 1500 of them, and it was three waits in a row.
    jobs = []
    with ThreadPoolExecutor(max_workers=3) as pool:
        for i, (chart_start, length) in enumerate(
                segment_windows(row, video_seconds)):
            part = work / f"{key}.seg{i}.mp4"
            jobs.append((chart_start, length, part, pool.submit(
                _render, video, stems, chart_start,
                max(0.0, chart_start + offset),
                length, CLIP_HEIGHT, part)))

    made: list[tuple[float, float, Path]] = []
    for chart_start, length, part, job in jobs:
        ok, why = job.result()
        # Measure the piece rather than trusting the exit code. Running off
        # the end of the video leaves ffmpeg nothing to encode and it still
        # exits 0, and a 1.4-second scrap named as a fifteen-second window is
        # a file that lies about itself.
        if ok and au.duration_of(part) < length - 0.5:
            ok, why = False, f"{part.name}: too short to be a segment"
        if not ok:
            print(f"    segment at {chart_start:.1f}s dropped - "
                  + " ".join(why.split())[:160])
            part.unlink(missing_ok=True)
            continue
        made.append((chart_start, length, part))
    if not made:
        return None

    # The sidecar describes the files, not the request. A segment can fail -
    # the video may not reach that far - and claiming three when two were
    # made is how a clip ends up not matching its own description. The reach
    # goes in too, so the window can say where the footage stops.
    _sidecar(out).write_text(json.dumps({
        "segments": [round(s, 2) for s, _, _ in made],
        "lengths": [round(ln, 2) for _, ln, _ in made],
        "files": [part.name for _, _, part in made],
        "video_seconds": round(video_seconds, 3),
        "reach_seconds": round(video_reach(video_seconds,
                                           row["offset_ms"] or 0.0), 3),
    }), encoding="utf-8")
    return made[0][2]


def _sidecar(path: Path) -> Path:
    """
    The one sidecar describing a build, from any file belonging to it.

    Keyed on the name before the first dot, because a build is now several
    files - `clip_<song>_<state>.seg0.mp4`, `.seg1.mp4` - that share a stem
    and must share one description of themselves.
    """
    return path.with_name(path.name.split(".")[0] + ".segments.json")


def clip_info(clip: Path) -> dict:
    """
    What a built clip holds: its segments, and how far its video reaches.

    Sidecars written before the reach was recorded are bare lists; they are
    still true about the segments, so they are read as such rather than
    thrown away and rebuilt.
    """
    try:
        data = json.loads(_sidecar(clip).read_text(encoding="utf-8"))
    except Exception:
        return {"segments": []}
    if isinstance(data, list):
        return {"segments": data}
    return data if isinstance(data, dict) else {"segments": []}


def clip_segments(clip: Path) -> list[float]:
    """Song times of the segments actually present in a built clip."""
    return clip_info(clip).get("segments") or []


def _render(video: Path, stems: list[Path], chart_start: float,
            video_start: float, seconds: float, height: int,
            out: Path) -> tuple[bool, str]:
    """
    One aligned segment: video seeked to its matching point, chart audio.

    Returns ffmpeg's own account of what happened along with the verdict. It
    used to be thrown away, so a build that produced nothing could only say
    "ffmpeg could not build a clip" - true, and no help at all.
    """
    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin",
           "-ss", f"{video_start:.3f}", "-t", f"{seconds}", "-i", str(video)]
    for st in stems:
        cmd += ["-ss", f"{chart_start:.3f}", "-t", f"{seconds}", "-i", str(st)]

    n = len(stems)
    if n == 1:
        amap = "1:a"
    else:
        mix = "".join(f"[{i}:a]" for i in range(1, n + 1))
        cmd += ["-filter_complex", f"{mix}amix=inputs={n}:normalize=0[a]"]
        amap = "[a]"

    cmd += ["-map", "0:v:0", "-map", amap,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-vf", f"scale=-2:{height},setsar=1,fps=30", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "44100",
            "-movflags", "+faststart", "-shortest", str(out)]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=300,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return False, f"ffmpeg timed out after 300s on {out.name}"
    except OSError as exc:
        return False, f"ffmpeg could not be run: {exc}"
    err = (done.stderr or "").strip()
    if done.returncode == 0 and out.exists():
        return True, err
    return False, err or f"ffmpeg exited {done.returncode} with nothing to say"


def build_full(row, work: Path) -> Path | None:
    """
    The whole song with the offset baked in, for scrubbing through.

    The video stream is copied, not re-encoded: nothing about it changes, and
    re-encoding a four-minute video to move its audio took minutes per song
    on a step that exists to answer one question quickly.

    So the offset is applied entirely on the audio side, and in the direction
    the offset means. Video time is chart time plus the offset, so:

    - positive offset - the video runs ahead of the song. Play the video from
      its own start and delay the song into place with `adelay`, which pads
      with real silence. Not `-itsoffset`: a negative start timestamp is a
      note in the container asking the player to be late, and players
      disagree about whether to honour it. Silence is not a request.
    - negative offset - the song starts before there is any footage. There is
      nothing to show for those seconds, so seek each stem past them and let
      both streams start together.

    Either way the length is the chart plus the offset: the extra lead-in for
    a positive one, the dropped intro for a negative one.
    """
    video = source_video(row)
    stems = au.find_stems(Path(row["song_dir"]))
    if video is None or not stems:
        return None

    offset = (row["offset_ms"] or 0) / 1000.0
    duration = row["chart_seconds"] or au.duration_of(stems[0])
    work.mkdir(parents=True, exist_ok=True)
    out = work / full_name(row)
    purge_stale(work, out)
    if out.exists():
        return out

    cmd = ["ffmpeg", "-y", "-v", "error", "-nostdin", "-i", str(video)]
    for st in stems:
        if offset < 0:
            cmd += ["-ss", f"{-offset:.3f}"]
        cmd += ["-i", str(st)]

    n = len(stems)
    chain = []
    if n == 1:
        tap = "[1:a]"
    else:
        mix = "".join(f"[{i}:a]" for i in range(1, n + 1))
        chain.append(f"{mix}amix=inputs={n}:normalize=0[m]")
        tap = "[m]"
    if offset > 0:
        chain.append(f"{tap}adelay={round(offset * 1000)}:all=1[a]")
        amap = "[a]"
    else:
        amap = tap if chain else "1:a"
    if chain:
        cmd += ["-filter_complex", ";".join(chain)]

    cmd += ["-map", "0:v:0", "-map", amap,
            "-t", f"{max(0.0, duration + offset):.2f}",
            "-c:v", "copy",
            "-c:a", "aac", "-b:a", "160k", "-ac", "2", str(out)]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=1200,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return None
    if done.returncode != 0 or not out.exists():
        print("    full build failed - "
              + " ".join((done.stderr or "").split())[:160])
        return None
    # Where this file can actually be seeked to. The video stream was copied,
    # so its keyframes are whatever YouTube's encoder chose - as much as ten
    # seconds apart - and a seek between two of them leaves the picture
    # frozen while the decoder waits for the next one.
    _sidecar(out).write_text(json.dumps({
        "keyframes": keyframes_of(out),
        "video_seconds": round(au.duration_of(video), 3),
    }), encoding="utf-8")
    return out


def keyframes_of(path: Path) -> list[float]:
    """
    The seconds at which a file can be seeked to without decoding to get there.

    Read from the packet flags rather than by decoding: ffprobe reports the K
    flag straight out of the container index, so a four-minute video is a
    tenth of a second's work.
    """
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
           "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0",
           str(path)]
    try:
        done = subprocess.run(cmd, capture_output=True, timeout=120,
                              encoding="utf-8", errors="replace")
    except (subprocess.TimeoutExpired, OSError):
        return []
    if done.returncode != 0:
        return []
    out: list[float] = []
    for line in (done.stdout or "").splitlines():
        bits = line.strip().split(",")
        if len(bits) < 2 or "K" not in bits[-1]:
            continue
        try:
            out.append(round(float(bits[0]), 3))
        except ValueError:
            continue          # a packet with no timestamp says nothing
    return sorted(out)


def snap_to_keyframe(ms: int, keyframes: list[float]) -> int:
    """
    The latest keyframe at or before `ms`, in milliseconds.

    Unchanged when nothing is known about the file: a seek to where you asked
    is a better answer than a seek to zero.
    """
    before = [k for k in keyframes if k * 1000.0 <= ms + 1]
    return int(round(max(before) * 1000)) if before else int(ms)


# ---------------------------------------------------------------- server ----

def make_handler(db_path: Path, work: Path):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):        # keep the console usable
            pass

        def _send(self, code, body, ctype="application/json"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body).encode()
            elif isinstance(body, str):
                body = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_file(self, path: Path, ctype: str):
            """
            Serve a file with HTTP range support.

            Browsers request video with `Range: bytes=...` and expect a 206.
            Answering 200 with the whole body makes Chrome unreliable and
            stops Firefox playing at all, and neither can seek.
            """
            size = path.stat().st_size
            rng = self.headers.get("Range", "")
            start, end = 0, size - 1
            partial = False
            if rng.startswith("bytes="):
                spec = rng.split("=", 1)[1].split(",")[0]
                a, _, b = spec.partition("-")
                try:
                    if a:
                        start = int(a)
                        end = int(b) if b else size - 1
                    elif b:                       # suffix range: last N bytes
                        start = max(0, size - int(b))
                    partial = True
                except ValueError:
                    partial = False
            start = max(0, min(start, size - 1))
            end = max(start, min(end, size - 1))
            length = end - start + 1

            self.send_response(206 if partial else 200)
            self.send_header("Content-Type", ctype)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(length))
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            try:
                with path.open("rb") as fh:
                    fh.seek(start)
                    remaining = length
                    while remaining > 0:
                        chunk = fh.read(min(262144, remaining))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        remaining -= len(chunk)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass          # the browser moved on; not an error

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == "/":
                return self._send(200, PAGE, "text/html; charset=utf-8")

            # Clips are served as plain static files so range requests work.
            if u.path.startswith("/clip/"):
                f = work / Path(u.path).name
                if not f.exists() or f.suffix != ".mp4":
                    return self._send(404, {"error": "no such clip"})
                return self._serve_file(f, "video/mp4")

            db = Database(db_path)
            try:
                if u.path == "/api/queue":
                    return self._send(200, queue(db))
                if u.path == "/api/prepare":
                    # Building the clip is separated from serving it: the
                    # browser gets a URL only once the file exists, instead of
                    # waiting on a video request that ffmpeg has not filled yet.
                    song = parse_qs(u.query).get("song", [""])[0]
                    row = db.conn.execute(
                        "SELECT * FROM songs WHERE song_dir = ?", (song,)
                    ).fetchone()
                    if row is None:
                        return self._send(404, {"error": "unknown song"})
                    if source_video(row) is None:
                        return self._send(200, {"error":
                            "The source video is gone - encode deleted it. "
                            "Re-run download for this song to review it."})
                    if not au.find_stems(Path(row["song_dir"])):
                        return self._send(200, {"error": "No chart audio found."})
                    clip = build_clip(row, work)
                    if clip is None:
                        return self._send(200, {"error":
                            "ffmpeg could not build a clip from this video."})
                    return self._send(200, {"url": f"/clip/{clip.name}"})
            finally:
                db.close()
            self._send(404, {"error": "not found"})

        def do_POST(self):
            """
            Act on one song.

            The body is read whatever happens: answering before the request
            has been drained leaves the client reading a reset connection
            instead of the status it was told.
            """
            try:
                length = int(self.headers.get("Content-Length", 0))
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b"{}"

            # Every path used to be treated as /api/act, so a typo or a stray
            # request from another tab acted on a song, and a body that was
            # not JSON came back as a traceback and a dead connection.
            if urlparse(self.path).path != "/api/act":
                return self._send(404, {"error": "not found"})
            try:
                payload = json.loads(raw or b"{}")
            except (ValueError, UnicodeDecodeError):
                return self._send(400, {"error": "body is not JSON"})
            if not isinstance(payload, dict):
                return self._send(400, {"error": "body is not an object"})

            db = Database(db_path)
            try:
                song = Path(payload.get("song", ""))
                action = payload.get("action")
                if action in ("keep", "clear"):
                    db.update(song, review=None if action == "clear" else "keep")
                elif action == "replace":
                    from .cli import cmd_set
                    from types import SimpleNamespace
                    cmd_set(SimpleNamespace(
                        pattern=str(song), url=payload.get("url", ""),
                        cookies=None), db)
                else:
                    return self._send(400, {"error": "unknown action"})
                return self._send(200, {"ok": True})
            finally:
                db.close()

    return Handler


def make_server(db_path: Path, work: Path,
                port: int = 8770) -> ThreadingHTTPServer:
    """The review server, built but not started. Port 0 picks a free one."""
    return ThreadingHTTPServer(("127.0.0.1", port), make_handler(db_path, work))


def serve(db_path: Path, work: Path, port: int = 8770) -> None:
    server = make_server(db_path, work, port)
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Review app running at {url}")
    print("Press Ctrl+C to stop.")
    threading.Timer(0.7, lambda: _open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


def _open(url: str) -> None:
    try:
        import webbrowser
        webbrowser.open(url)
    except Exception:
        pass


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Backgrounds to check</title>
<style>
  :root{
    --base:#12161c; --panel:#171d25; --line:#2a323d;
    --ink:#d9dfe7; --dim:#79848f; --signal:#e8a33d; --clear:#6fbf8f; --alert:#d9686a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--base);color:var(--ink);
    font:15px/1.5 "Segoe UI",system-ui,sans-serif;
    display:grid;grid-template-columns:minmax(300px,380px) 1fr;height:100vh}
  @media(max-width:820px){body{grid-template-columns:1fr;height:auto}}

  aside{border-right:1px solid var(--line);overflow-y:auto;background:var(--panel)}
  .top{padding:18px 20px 14px;border-bottom:1px solid var(--line);
    position:sticky;top:0;background:var(--panel);z-index:2}
  .top h1{margin:0 0 4px;font-size:17px;font-weight:600;letter-spacing:-.01em}
  .top p{margin:0;color:var(--dim);font-size:13px}

  .row{display:block;width:100%;text-align:left;background:none;border:0;
    border-bottom:1px solid var(--line);padding:12px 20px 12px 16px;cursor:pointer;
    color:inherit;font:inherit;border-left:3px solid transparent}
  .row:hover{background:#1c232c}
  .row:focus-visible{outline:2px solid var(--signal);outline-offset:-2px}
  .row[aria-current=true]{background:#1f2731;border-left-color:var(--signal)}
  .row.done{opacity:.42;border-left-color:var(--clear)}
  .row .name{font-weight:600;font-size:14px}
  .row .sub{color:var(--dim);font-size:12.5px;margin-top:2px}
  .row .why{color:var(--signal);font-size:12.5px;margin-top:5px}
  .row.done .why{color:var(--clear)}

  main{overflow-y:auto;padding:28px 32px 60px;max-width:900px}
  h2{margin:0 0 2px;font-size:24px;font-weight:600;letter-spacing:-.02em}
  .by{color:var(--dim);margin:0 0 22px}

  video{width:100%;max-width:640px;background:#000;border:1px solid var(--line);
    display:block}
  .loading{color:var(--dim);padding:60px 0}

  .facts{display:flex;gap:34px;flex-wrap:wrap;margin:22px 0 8px;
    padding:16px 0;border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
  .fact .k{color:var(--dim);font-size:12.5px}
  .fact .v{font-family:ui-monospace,"Cascadia Mono",Consolas,monospace;
    font-size:19px;font-variant-numeric:tabular-nums;margin-top:3px}
  .fact .v small{font-size:12px;color:var(--dim);margin-left:3px}

  .flags{margin:16px 0 26px;padding:0;list-style:none;color:var(--signal);font-size:14px}
  .flags li{margin:4px 0}
  .flags li::before{content:"— ";color:var(--dim)}

  .acts{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
  button.act{font:inherit;padding:9px 18px;border-radius:2px;cursor:pointer;
    border:1px solid var(--line);background:#1d242d;color:var(--ink)}
  button.act:hover{border-color:var(--dim)}
  button.keep{border-color:var(--clear);color:var(--clear)}
  input[type=url]{flex:1;min-width:230px;font:inherit;padding:9px 12px;border-radius:2px;
    background:#0e1218;border:1px solid var(--line);color:var(--ink)}
  input[type=url]:focus{outline:2px solid var(--signal);outline-offset:-1px}
  .hint{color:var(--dim);font-size:13px;margin:10px 0 0}
  .empty{color:var(--dim);padding:80px 0;max-width:44ch}
</style></head><body>
<aside>
  <div class="top">
    <h1>Backgrounds to check</h1>
    <p id="count">Loading…</p>
  </div>
  <div id="list"></div>
</aside>
<main id="detail"><p class="empty">Pick a song on the left. The most doubtful
ones are at the top.</p></main>

<script>
let songs = [], current = null;

const ms = v => v === null || v === undefined ? "—" : Math.round(v).toLocaleString();

async function load(){
  songs = await (await fetch("/api/queue")).json();
  const todo = songs.filter(s => !s.review).length;
  document.getElementById("count").textContent =
    `${todo} left of ${songs.length}, most doubtful first`;
  const list = document.getElementById("list");
  list.innerHTML = "";
  songs.forEach(s => {
    const b = document.createElement("button");
    b.className = "row" + (s.review ? " done" : "");
    b.setAttribute("aria-current", current === s.song_dir);
    b.innerHTML =
      `<div class="name"></div><div class="sub"></div><div class="why"></div>`;
    b.querySelector(".name").textContent = `${s.artist} — ${s.title}`;
    b.querySelector(".sub").textContent = s.channel || s.video;
    b.querySelector(".why").textContent = s.review ? "checked" : s.reasons[0];
    b.onclick = () => show(s.song_dir);
    list.appendChild(b);
  });
}

function show(dir){
  current = dir;
  const s = songs.find(x => x.song_dir === dir);
  const d = document.getElementById("detail");
  d.innerHTML = `
    <h2></h2><p class="by"></p>
    <p class="loading">Building a clip from the middle of the song…</p>
    <div class="facts">
      <div class="fact"><div class="k">Video offset</div>
        <div class="v" id="f-off"></div></div>
      <div class="fact"><div class="k">Agreement across track</div>
        <div class="v" id="f-spr"></div></div>
      <div class="fact"><div class="k">Audio match</div>
        <div class="v" id="f-fp"></div></div>
      <div class="fact"><div class="k">Motion</div>
        <div class="v" id="f-mot"></div></div>
    </div>
    <ul class="flags"></ul>
    <div class="acts">
      <button class="act keep" id="keep">Looks right</button>
      <input type="url" id="url" placeholder="Paste a better YouTube link">
      <button class="act" id="rep">Use this instead</button>
    </div>
    <p class="hint">Replacing requeues the song: run download, sync and encode
      again afterwards.</p>`;
  d.querySelector("h2").textContent = s.title;
  d.querySelector(".by").textContent =
    `${s.artist} · ${s.video}${s.channel ? " · " + s.channel : ""}`;

  const off = s.offset_ms || 0;
  document.getElementById("f-off").innerHTML =
    `${ms(off)}<small>ms ${off < 0 ? "video waits"
      : off > 0 ? "video skips ahead" : ""}</small>`;
  document.getElementById("f-spr").textContent = s.spread_text;
  document.getElementById("f-fp").textContent =
    s.fp_score ? Math.round(s.fp_score) : "—";
  document.getElementById("f-mot").textContent =
    s.static ? "still image" : (s.motion ?? 0).toFixed(2);

  const ul = d.querySelector(".flags");
  s.reasons.forEach(r => { const li = document.createElement("li");
    li.textContent = r; ul.appendChild(li); });

  // Ask for the clip first, then point the player at a real file. Setting
  // src directly made the browser wait on a request ffmpeg had not filled.
  fetch("/api/prepare?song=" + encodeURIComponent(dir)).then(r => r.json())
    .then(res => {
      if (current !== dir) return;              // user already moved on
      const note = d.querySelector(".loading");
      if (res.error){ if (note) note.textContent = res.error; return; }
      const v = document.createElement("video");
      v.controls = true; v.preload = "auto"; v.src = res.url;
      v.onloadeddata = () => note?.remove();
      v.onerror = () => { if (note) note.textContent =
        "The clip was built but would not play. Try another song."; };
      d.querySelector(".by").after(v);
      // Browsers block autoplay with sound until the page has been clicked;
      // ignore the rejection rather than logging an error.
      v.play().catch(() => {});
    })
    .catch(() => { const note = d.querySelector(".loading");
      if (note) note.textContent = "Could not reach the review server."; });

  document.getElementById("keep").onclick = () => act(dir, "keep");
  document.getElementById("rep").onclick = () => {
    const u = document.getElementById("url").value.trim();
    if (u) act(dir, "replace", u);
  };
  load();
}

async function act(song, action, url){
  await fetch("/api/act", {method:"POST",
    headers:{"Content-Type":"application/json"},
    body: JSON.stringify({song, action, url})});
  const i = songs.findIndex(s => s.song_dir === song);
  const next = songs.slice(i + 1).find(s => !s.review);
  await load();
  if (next) show(next.song_dir);
}

load();
</script></body></html>
"""
