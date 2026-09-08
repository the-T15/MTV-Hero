"""Stage runner. Every command is resumable and safe to re-run."""

from __future__ import annotations

import argparse
import configparser
import os
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from . import audio as au
from . import encode as enc
from . import fingerprint as fp
from . import match as mt
from . import review as rv
from . import sync as sy
from .db import ENCODABLE, STAGES, Database


# ---------------------------------------------------------------- coverage ---

def covers_song(video_s: float, offset_ms: float, chart_s: float) -> bool:
    """
    Does the video still reach the end of the song once it is shifted?

    A positive `video_start_time` seeks the video forward, so its end arrives
    that much earlier in song time. A negative one DELAYS the video, so its end
    lands |offset| later and coverage goes UP. The old expression clamped the
    shift with max(0, off), which threw the negative half away and reported
    most of a negative-offset library as too short. Five seconds of slack
    absorbs a fade-out.

    One function because that clamp survived in two further copies of the same
    test after the first was fixed.
    """
    return (video_s - offset_ms / 1000.0) >= chart_s - 5


# ---------------------------------------------------------------- song.ini ---

def _split_folder_name(name: str) -> tuple[str, str]:
    """
    Parse 'Artist - Title' from a folder name.

    Pack folders often carry a track number prefix ('01 - Band - Song',
    '1 Band - Song'). Left in place it poisons the search query, which is what
    produced entries like ' - 1 Silverstein - Massachusetts'.
    """
    cleaned = re.sub(r"^\s*\d{1,3}\s*[-._)]*\s+", "", name).strip()
    parts = cleaned.split(" - ", 1)
    if len(parts) == 2 and parts[0].strip():
        return parts[0].strip(), parts[1].strip()
    return "", cleaned


def read_song_ini(song_dir: Path) -> tuple[str, str]:
    ini = song_dir / "song.ini"
    if not ini.exists():
        return _split_folder_name(song_dir.name)

    cfg = configparser.ConfigParser(strict=False, interpolation=None)
    cfg.optionxform = str
    try:
        cfg.read(ini, encoding="utf-8-sig")
    except Exception:
        return _split_folder_name(song_dir.name)

    artist = title = ""
    for section in cfg.sections():
        if section.lower() == "song":
            s = cfg[section]
            artist = (s.get("artist") or s.get("Artist") or "").strip()
            title = (s.get("name") or s.get("Name") or s.get("title") or "").strip()
            break

    # Strip any leading track number first - otherwise '1 Silverstein - X'
    # does not match the artist prefix below and the artist stays duplicated
    # in the search query.
    title = re.sub(r"^\s*\d{1,3}\s*[-._)]*\s+", "", title).strip()

    # A missing artist makes the search query useless, so recover it from the
    # folder name rather than searching on a title alone.
    if not artist:
        fa, ft = _split_folder_name(song_dir.name)
        artist = fa
        if not title:
            title = ft

    # Remove a redundant 'Artist - ' prefix left inside the title field.
    if artist and title.lower().startswith(artist.lower()):
        stripped = title[len(artist):].lstrip(" -\u2013").strip()
        if stripped:
            title = stripped

    if not title:
        _, title = _split_folder_name(song_dir.name)
    return artist, title


def write_video_start_time(song_dir: Path, value: int, backup: bool = True) -> None:
    """
    Set `video_start_time` in [song], preserving everything else verbatim.

    Deliberately line-based rather than configparser round-tripping: song.ini
    files in the wild contain duplicate keys, stray comments and odd encodings
    that a full rewrite silently mangles.
    """
    ini = song_dir / "song.ini"
    if backup and ini.exists() and not (song_dir / "song.ini.bak").exists():
        shutil.copy2(ini, song_dir / "song.ini.bak")

    # Bytes, not text. read_text(encoding="utf-8-sig") normalises every line
    # ending to \n before this function can look at them, so the CRLF check
    # below could never fire and the "preserved verbatim" promise was a no-op.
    # It also swallowed the BOM, which was then not written back.
    raw = ini.read_bytes() if ini.exists() else b"[song]"
    bom = b"\xef\xbb\xbf" if raw.startswith(b"\xef\xbb\xbf") else b""
    text = raw[len(bom):].decode("utf-8", errors="replace")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.splitlines()

    new_line = f"video_start_time = {value}"
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("video_start_time"):
            lines[i] = new_line
            break
    else:
        insert_at = len(lines)
        for i, line in enumerate(lines):
            if line.strip().lower() == "[song]":
                insert_at = i + 1
                break
        lines.insert(insert_at, new_line)

    # Write-then-rename rather than writing in place. A crash or a full disk
    # partway through a direct write leaves a truncated song.ini, which the game
    # reads as a song with no metadata. os.replace is atomic within a
    # directory, and the temp file is in the song folder, so it never crosses
    # a filesystem boundary.
    tmp = ini.parent / (ini.name + ".tmp")
    tmp.write_bytes(bom + (newline.join(lines) + newline).encode("utf-8"))
    os.replace(tmp, ini)


# ------------------------------------------------------------------ stages ---

def _count_stfs_packages(root: Path) -> int:
    """
    Count single-file Rock Band CON packages.

    These are Xbox 360 STFS containers with no file extension - Windows shows
    them as a generic "File" - so they can only be found by their magic header
    (CON /LIVE/PIRS). A suffix-based scan reports zero and hides them.
    """
    n = 0
    for p in root.rglob("*"):
        try:
            if not p.is_file() or p.suffix.lower() not in ("", ".con"):
                continue
            if p.stat().st_size < 1_000_000:
                continue
            with p.open("rb") as fh:
                if fh.read(4) in (b"CON ", b"LIVE", b"PIRS"):
                    n += 1
        except OSError:
            continue
    return n


def cmd_index(args, db: Database) -> None:
    root = Path(args.root)
    if not root.is_dir():
        print(f"Not a directory: {root}")
        return

    found = 0
    no_stems = 0
    for ini in root.rglob("song.ini"):
        d = ini.parent
        stems = au.find_stems(d)
        if not stems:
            no_stems += 1
            continue
        artist, title = read_song_ini(d)
        seconds = max((au.duration_of(s) for s in stems), default=0.0)
        db.add_song(d, artist, title, seconds)
        found += 1
        if found % 100 == 0:
            print(f"  indexed {found}...", flush=True)

    # Packed formats cannot receive a per-song video: YARG loads the background
    # from loose files beside the chart, and there is nowhere to put one inside
    # a .sng archive or a CON. Extract them first if you want videos on these.
    sng = sum(1 for _ in root.rglob("*.sng"))
    con = sum(1 for _ in root.rglob("songs.dta"))
    con += _count_stfs_packages(root)

    print(f"\nIndexed {found} songs into {db.path}")
    if no_stems:
        print(f"  {no_stems} folders had song.ini but no usable audio stems")
    if sng:
        print(f"  {sng} .sng archives skipped - packed, cannot hold a video")
    if con:
        print(f"  {con} CON/ex-CON packs skipped - extract to folders first")
    if sng or con:
        print("\n  These are a YARG limitation, not a pipeline one: background")
        print("  videos are loaded from loose files in the song folder.")


def cmd_match(args, db: Database) -> None:
    # Overriding the module global is deliberate: fingerprint.is_same_recording
    # reads ACCEPT_SCORE at call time, so this reaches every gate check for the
    # rest of the run without threading a parameter through five call sites.
    if getattr(args, "gate", None):
        print(f"fingerprint gate: {args.gate} (default {fp.ACCEPT_SCORE})")
        fp.ACCEPT_SCORE = args.gate

    work = Path(args.work)
    if getattr(args, "redo", False):
        rows = db.retry_queue("match", args.limit)
        print("re-attempting previously failed matches only")
    else:
        rows = db.pending("match", args.limit, getattr(args, "sample", False))
    print(f"{len(rows)} songs to match")

    for n, row in enumerate(rows, 1):
        d = Path(row["song_dir"])
        print(f"[{n}/{len(rows)}] {row['artist']} - {row['title']}", flush=True)

        chart = au.mix_stems(au.find_stems(d))
        if chart.size == 0:
            db.update(d, match_status="failed", match_note="no decodable stems")
            continue

        winner, candidates, reason = mt.pick_best(
            chart, row["artist"], row["title"], work, args.cookies, args.sleep
        )
        db.save_candidates(d, candidates)

        if winner is None:
            db.update(d, match_status="failed", match_note=reason)
            print(f"    rejected: {reason}")
        else:
            # A winner can still be flagged: it passed the audio gate but its
            # title says lyric video, static image or gameplay capture. Store
            # that marker so `status` can surface it for review rather than
            # letting it disappear behind a successful-looking line.
            flagged = reason != "ok"
            # Record the uploader. No heuristic can tell an official video
            # from a fan edit by title, but the channel usually can - and it
            # costs nothing to store, unlike another flag that would add noise.
            who = winner.uploader or "unknown channel"
            note = f"{winner.title} [{who}]"
            if flagged:
                note = f"REVIEW: {note}"
            db.update(
                d, match_status="ok", video_id=winner.video_id,
                match_score=winner.score, match_note=note,
            )
            mark = "  [REVIEW - static image]" if flagged else ""
            print(f"    -> {winner.title} (score {winner.score:.0f}){mark}")
            print(f"       channel: {who}")


def cmd_download(args, db: Database) -> None:
    rows = db.pending("download", args.limit, getattr(args, "sample", False))
    print(f"{len(rows)} videos to download")
    for n, row in enumerate(rows, 1):
        d = Path(row["song_dir"])
        print(f"[{n}/{len(rows)}] {row['title']}", flush=True)
        src, note = mt.download_video(
            row["video_id"], d / "video", args.height, args.cookies, args.sleep
        )
        if src is None:
            db.update(d, download_status="failed", download_note=note)
            print(f"  Failed: {note}")
        else:
            db.update(d, download_status="ok", source_path=str(src),
                      download_note=None)


def cmd_sync(args, db: Database) -> None:
    recheck = getattr(args, "recheck", False)
    if recheck:
        # Recompute already-synced songs and write ONLY where the answer
        # changes. Anything that comes out the same is left completely alone,
        # so review marks and encode state survive - unlike `retry sync --all`,
        # which invalidates every song to fix the few that were wrong.
        floor = getattr(args, "min_offset", 0.0) or 0.0
        skip_done = getattr(args, "skip_reviewed", False)
        rows = [
            r for r in db.conn.execute(
                "SELECT * FROM songs WHERE sync_status IN "
                "('ok','drift','unverified') AND source_path IS NOT NULL "
                "ORDER BY song_dir").fetchall()
            if abs(r["offset_ms"] or 0) >= floor
            and not (r["sync_note"] or "").startswith("MANUAL")
            and not (skip_done and r["review"])
        ]
        if args.limit:
            rows = rows[: args.limit]
        print(f"Rechecking {len(rows)} songs"
              + (" (skipping ones you confirmed)" if skip_done else "")
              + (f" with an offset of {floor:.0f} ms or more" if floor else "")
              + " - only changes will be written")
    else:
        rows = db.pending("sync", args.limit, getattr(args, "sample", False))
        print(f"{len(rows)} songs to sync")
    changed = 0

    for n, row in enumerate(rows, 1):
        d = Path(row["song_dir"])
        src = Path(row["source_path"])
        print(f"[{n}/{len(rows)}] {row['artist']} - {row['title']}", flush=True)
        if not src.exists():
            db.update(d, sync_status="failed", sync_note="source video missing")
            continue

        stems = au.find_stems(d)
        chart = au.mix_stems(stems, fp.SR)
        video = au.decode_mono(src, fp.SR)
        chart_hi = au.mix_stems(stems, sy.REFINE_SR)
        video_hi = au.decode_mono(src, sy.REFINE_SR)

        # Measure motion first: whether the background moves decides whether
        # excerpt agreement is a meaningful test at all.
        #
        # On a recheck the video file has not changed, so a stored figure is
        # still valid. Recomputing costs 14 separate ffmpeg seeks per song,
        # which on Windows is a large share of the total time.
        motion = row["motion"] if recheck and row["motion"] is not None \
            else enc.motion_score(src)
        manual = (row["match_note"] or "").startswith("MANUAL")
        res = sy.estimate(
            chart, video, chart_hi, video_hi,
            trust_identity=True, static_background=enc.is_static(motion),
            manual=manual,
        )
        db.update(
            d,
            sync_status=res.status,
            offset_ms=res.offset_ms,
            spread_ms=res.spread_ms,
            drift_ppm=res.drift_ppm,
            fp_score=res.fp_score,
            sync_note=res.reason,
            motion=motion,
            dominance=res.dominance,
            windows=res.windows,
        )
        if recheck:
            # The measurements above are refreshed unconditionally: they
            # describe this run, and a stale spread or window count next to a
            # fresh status is not a reading anyone can use.
            #
            # What survives an unchanged answer is the human part - the
            # approval and the work done on the back of it. "Unchanged" has to
            # mean the answer, not just the number: a flip to 'rejected' two
            # ms away from the old offset used to print "unchanged" and leave
            # review='keep' and encode_status='ok' on a video the pipeline
            # now refuses to encode.
            old = row["offset_ms"]
            moved = old is None or abs(res.offset_ms - old) > 100
            encodable = res.status in ENCODABLE
            if not moved and encodable:
                print(f"    unchanged ({res.video_start_time} ms)")
                continue
            changed += 1
            was = "?" if old is None else f"{old:.0f}"
            note = "" if encodable else f"  [now {res.status}]"
            print(f"    CHANGED  {was} -> {res.video_start_time} ms{note}")
            if res.reason:
                print(f"      {res.reason[:76]}")
            # The stored answer no longer holds, so any earlier approval of it
            # no longer applies - send it back for review.
            db.update(d, review=None, encode_status="pending",
                      ini_status="pending")

        flag = {"ok": "", "unverified": "  [UNVERIFIED]",
                "drift": "  [DRIFT]", "rejected": "  [REJECTED]"}[res.status]
        # Spread alone is ambiguous: 3 ms across 7 windows and 3 ms across 4
        # print identically, and the second is a far weaker claim.
        agreed = (f" over {res.windows}/{res.windows_total} windows"
                  if res.windows_total else "")
        print(
            f"    video_start_time = {res.video_start_time}  "
            f"spread {res.spread_ms:.0f} ms{agreed}{flag}"
        )
        if manual:
            # Every hand-picked video, not only the ones whose fingerprint is
            # weak - those are the only ones estimate() mentions in its reason.
            # The offset above was measured against a video you chose, which is
            # what decides how the numbers next to it should be read.
            print("    your pick - video chosen by hand")
        if enc.is_static(motion):
            print(f"    [STATIC IMAGE - motion {motion:.2f}, no moving footage]")
        vid_len = au.duration_of(src)
        chart_len = chart.size / fp.SR
        covered = vid_len - res.offset_ms / 1000.0
        if chart_len > 0 and not covers_song(vid_len, res.offset_ms, chart_len):
            print(f"    [SHORT - video runs out {chart_len - covered:.0f}s "
                  f"before the song ends]")
        if res.reason and not recheck:
            print(f"    {res.reason}")

    if recheck:
        print(f"\n{changed} of {len(rows)} offsets changed.")
        if changed:
            print("Those were sent back for review; run: yargvid review")


def cmd_encode(args, db: Database) -> None:
    rows = db.pending("encode", args.limit, getattr(args, "sample", False))

    if getattr(args, "skip_existing", False):
        before = len(rows)
        rows = [r for r in rows
                if not (Path(r["song_dir"]) / "video.webm").exists()]
        kept = before - len(rows)
        if kept:
            print(f"Leaving {kept} folders alone - they already have a "
                  f"video.webm")

    # Encoding is the slow, destructive step - it deletes source videos. Being
    # able to run it over only what you have signed off keeps the unreviewed
    # ones recoverable.
    if getattr(args, "reviewed", False):
        before = len(rows)
        rows = [r for r in rows if r["review"] == "keep"]
        print(f"Encoding {len(rows)} approved songs "
              f"({before - len(rows)} not yet approved, left alone)")

    # A measured-static background is album art for the whole song. Skipping
    # those leaves YARG's own venue in place, which many people prefer to a
    # still image. They stay 'pending', so dropping the flag encodes them later
    # without redoing anything.
    if getattr(args, "skip_static", False):
        before = len(rows)
        rows = [r for r in rows if not enc.is_static(
            r["motion"] if r["motion"] is not None else -1.0)]
        skipped = before - len(rows)
        if skipped:
            print(f"Skipping {skipped} static backgrounds "
                  f"(left pending; rerun without --skip-static to include them)")
    settings = enc.EncodeSettings(
        height=args.height, crf=args.crf, cpu_used=args.cpu_used,
        threads_per_job=args.threads,
    )
    workers = args.workers or enc.default_workers()
    print(f"{len(rows)} to encode, {workers} concurrent jobs x "
          f"{settings.threads_per_job} threads")

    if args.preview:
        # A preview is the real encode at low resolution: full length, correct
        # timing, written as video.webm so YARG will actually load it. Sources
        # are kept and the DB is left untouched, so the full-quality run
        # afterwards simply overwrites these.
        preview_settings = replace(
            settings, height=args.preview_height, cpu_used=5, bitrate_cap="800k"
        )
        print(f"Previewing {len(rows)} songs at {args.preview_height}p "
              f"(full length, sources kept)")
        failed = 0
        for row in rows:
            d = Path(row["song_dir"])
            # One bad song must never abort the batch. The parallel encode
            # path already isolates failures; this loop needs the same.
            try:
                ok, err = enc.encode_one(
                    Path(row["source_path"]), d, preview_settings, keep_source=True
                )
                if ok:
                    write_video_start_time(d, int(round(row["offset_ms"] or 0)))
            except Exception as exc:  # noqa: BLE001
                ok, err = False, str(exc)[:200]
            failed += not ok
            print(f"  {'ok ' if ok else 'ERR'} {d.name} {err}")
        if failed:
            print(f"\n  {failed} of {len(rows)} failed - rerun to retry them")
        print("\nWritten as video.webm at preview quality. Check sync in YARG,")
        print("then re-run without --preview for the full-quality encode.")
        return

    jobs = [(Path(r["source_path"]), Path(r["song_dir"])) for r in rows]
    done = [0]

    def progress(song_dir, result):
        done[0] += 1
        ok, err = result
        db.update(song_dir, encode_status="ok" if ok else "failed", encode_note=err)
        print(f"  [{done[0]}/{len(jobs)}] {'ok ' if ok else 'ERR'} "
              f"{Path(song_dir).name} {err}", flush=True)

    enc.encode_many(jobs, settings, workers, on_done=progress)


def cmd_ini(args, db: Database) -> None:
    rows = db.pending("ini", args.limit, getattr(args, "sample", False))
    print(f"Writing video_start_time for {len(rows)} songs")
    for row in rows:
        d = Path(row["song_dir"])
        write_video_start_time(d, int(round(row["offset_ms"] or 0)))
        db.update(d, ini_status="ok")
    print("Done. Originals backed up as song.ini.bak")


def cmd_diagnose(args, db: Database) -> None:
    """Run the match stage on one song with full output. Writes nothing."""
    work = Path(args.work)

    if args.song_dir:
        row = db.conn.execute(
            "SELECT * FROM songs WHERE song_dir = ?", (args.song_dir,)
        ).fetchone()
        if row is None:
            print(f"Not in the database: {args.song_dir}")
            return
    else:
        rows = db.pending("match", 1) or db.conn.execute(
            "SELECT * FROM songs LIMIT 1"
        ).fetchall()
        if not rows:
            print("No songs in the database. Run `index` first.")
            return
        row = rows[0]

    d = Path(row["song_dir"])
    print(f"Song folder : {d}")
    print(f"Parsed as   : artist={row['artist']!r}  title={row['title']!r}")

    stems = au.find_stems(d)
    print(f"Stems       : {[s.name for s in stems]}")
    chart = au.mix_stems(stems)
    print(f"Chart audio : {chart.size / fp.SR:.1f} s decoded")
    if chart.size == 0:
        print("\nNo decodable audio - matching cannot work for this song.")
        return

    print("\n--- searching YouTube ---")
    cands = mt.search_candidates(
        row["artist"], row["title"], chart.size / fp.SR, args.cookies
    )
    if not cands:
        print("No candidates survived the duration filter.")
        return
    for c in cands:
        print(f"  {c.video_id}  {c.duration:6.0f}s  {c.title[:60]}")

    print(f"\n--- downloading audio for {cands[0].video_id} (verbose) ---")
    cmd = [
        "yt-dlp",
        f"https://www.youtube.com/watch?v={cands[0].video_id}",
        "-f", "worstaudio/bestaudio",
        "-o", str(work / "diag.%(ext)s"),
        "--no-playlist", "--no-part",
    ]
    cmd += mt.cookie_args(args.cookies)
    work.mkdir(parents=True, exist_ok=True)
    proc = mt._run(cmd, timeout=600)

    print(f"exit code: {proc.returncode}")
    if proc.stdout.strip():
        print("stdout:")
        for line in proc.stdout.strip().splitlines()[-12:]:
            print(f"  {line}")
    if proc.stderr.strip():
        print("stderr:")
        for line in proc.stderr.strip().splitlines()[-12:]:
            print(f"  {line}")

    got = list(work.glob("diag.*"))
    if not got:
        print("\nNo file written. This is a yt-dlp/network problem, NOT a")
        print("fingerprint problem - the audio was never compared.")
        return

    print(f"\nDownloaded {got[0].name} ({got[0].stat().st_size / 1024:.0f} KB)")
    samples = au.decode_mono(got[0], fp.SR)
    print(f"Decoded {samples.size / fp.SR:.1f} s")
    if samples.size:
        res = fp.match(
            fp.make_hashes(chart), fp.make_hashes(samples), chart.size / fp.SR
        )
        print(f"\nscore {res.score:.1f} (gate {fp.ACCEPT_SCORE})   "
              f"coverage {res.coverage:.2f}   offset {res.offset_seconds:.2f}s")
        print("VERDICT:", "MATCH" if fp.is_same_recording(res) else "no match")
    for g in got:
        g.unlink(missing_ok=True)


def cmd_inspect(args, db: Database) -> None:
    """
    Explain one song's offset using files already on disk. No network.

    Answers the question a bare offset number cannot: is a large negative
    video_start_time caused by lead-in silence in the chart, or by a video
    edit that trims the song's intro?
    """
    rows = db.conn.execute(
        "SELECT * FROM songs WHERE song_dir LIKE ? ORDER BY song_dir",
        (f"%{args.pattern}%",),
    ).fetchall()
    if not rows:
        print(f"No song matching {args.pattern!r}")
        return

    for row in rows[: args.limit or 5]:
        d = Path(row["song_dir"])
        print(f"\n{row['artist']} - {row['title']}")
        print(f"  matched      {row['match_note'] or '(not matched)'}")

        offset = row["offset_ms"]
        if offset is None:
            print("  not synced yet")
            continue
        print(f"  offset       video_start_time = {int(round(offset))} ms "
              f"(spread {row['spread_ms'] or 0:.0f} ms)")

        chart = au.mix_stems(au.find_stems(d))
        chart_lead = au.leading_silence(chart)
        print(f"  chart audio  {chart.size / fp.SR:6.1f} s, "
              f"lead-in silence {chart_lead:.2f} s")

        src = Path(row["source_path"]) if row["source_path"] else None
        if src and src.exists():
            vid = au.decode_mono(src)
            vid_lead = au.leading_silence(vid)
            print(f"  video audio  {vid.size / fp.SR:6.1f} s, "
                  f"lead-in silence {vid_lead:.2f} s")

            # Lead-in difference accounts for this much of the offset; the
            # remainder must come from an edit inside the song.
            explained = (vid_lead - chart_lead) * 1000.0
            residual = offset - explained
            print(f"\n  lead-in difference explains {explained:+.0f} ms")
            print(f"  unexplained residual        {residual:+.0f} ms")
            if abs(residual) < 500:
                print("  -> offset is entirely lead-in. Nothing was trimmed.")
            elif residual < -500:
                print("  -> the video starts LATER in the song than the chart")
                print("     does: the video edit skips part of the intro.")
            else:
                print("  -> the chart starts later in the song than the video:")
                print("     the chart audio is trimmed, or this is a different cut.")
        else:
            print("  source video not on disk (already encoded or not downloaded)")


def _one_song(db: Database, pattern: str):
    """
    Resolve a folder-path substring to exactly one song row.

    An exact song_dir wins outright before the substring search runs. The
    review app passes the full path, and a folder whose path is a prefix of
    another's ('...\\Foo' inside '...\\Foo (Live)') matched both through the
    LIKE - so `set` reported "be more specific" about a song it had been
    handed by name.
    """
    exact = db.conn.execute(
        "SELECT * FROM songs WHERE song_dir = ?", (pattern,)
    ).fetchone()
    if exact is not None:
        return exact

    rows = db.conn.execute(
        "SELECT * FROM songs WHERE song_dir LIKE ? ORDER BY song_dir",
        (f"%{pattern}%",),
    ).fetchall()
    if not rows:
        print(f"No song matching {pattern!r}")
        return None
    if len(rows) > 1:
        print(f"{len(rows)} songs match {pattern!r} - be more specific:")
        for r in rows[:10]:
            print(f"  {r['artist']} - {r['title']}")
        return None
    return rows[0]


def cmd_check(args, db: Database) -> None:
    """
    Test a specific video against a chart. Downloads audio only, writes nothing.

    Reports the full sync verdict, not just an identity score, so a live
    performance can be assessed before committing: a genuinely live recording
    is a different take and drifts continuously against the studio chart, which
    no single video_start_time can correct. This shows whether that is
    happening.
    """
    row = _one_song(db, args.pattern)
    if row is None:
        return
    vid = mt.parse_video_id(args.url)
    if not vid:
        print(f"Could not read a YouTube video ID from {args.url!r}")
        return

    d = Path(row["song_dir"])
    print(f"Song  : {row['artist']} - {row['title']}")
    meta = mt.fetch_metadata(vid, args.cookies)
    if meta:
        print(f"Video : {meta.get('title', '?')}")
        print(f"        {meta.get('uploader', '?')}  "
              f"{meta.get('duration', 0):.0f}s")

    stems = au.find_stems(d)
    chart = au.mix_stems(stems, fp.SR)
    if chart.size == 0:
        print("No decodable chart audio.")
        return

    work = Path(args.work)
    print("\nDownloading audio to test...")
    path, err = mt.probe_audio(vid, work, args.cookies, args.sleep)
    if path is None:
        print(f"Download failed: {err}")
        return

    try:
        video = au.decode_mono(path, fp.SR)
        video_hi = au.decode_mono(path, sy.REFINE_SR)
        chart_hi = au.mix_stems(stems, sy.REFINE_SR)
        res = sy.estimate(chart, video, chart_hi, video_hi, trust_identity=True)
    finally:
        path.unlink(missing_ok=True)

    # Show the competing alignments, so a false lock onto a repeated section
    # is visible rather than inferred.
    cands = fp.match_candidates(fp.make_hashes(chart), fp.make_hashes(video),
                                chart.size / fp.SR, top_k=4)
    print("\n  candidate alignments the fingerprint found:")
    for i, c in enumerate(cands, 1):
        mark = "  <- chosen" if abs(c.offset_seconds * 1000 - res.offset_ms) < 50 else ""
        print(f"    {i}. {c.offset_seconds * 1000:+9.0f} ms   "
              f"peak {c.peak_count:6d}   score {c.score:6.1f}{mark}")

    print(f"\n  fingerprint score  {res.fp_score:8.1f}   (gate {fp.ACCEPT_SCORE})")
    print(f"  coverage           {res.coverage:8.2f}")
    print(f"  video_start_time   {res.video_start_time:8d} ms")
    print(f"  window spread      {res.spread_ms:8.1f} ms "
          f"across {len(res.excerpts)} windows")
    if res.drift_ppm:
        print(f"  drift              {res.drift_ppm:8.0f} ppm (R2={res.r2:.2f})")

    print()
    if res.status == "ok":
        print("  USABLE - stable offset, no drift.")
        print(f"  Apply with:  yargvid set \"{args.pattern}\" {args.url}")
    elif res.status == "drift":
        print("  USABLE WITH DRIFT - the offset changes steadily across the")
        print("  track. YARG applies one fixed offset, so it will be right at")
        print("  the start and progressively wrong later.")
    else:
        print(f"  NOT USABLE - {res.reason}")
        print("  If this is a genuine live performance, that is expected: a")
        print("  different take drifts unpredictably and cannot be aligned by")
        print("  a single offset.")


def cmd_set(args, db: Database) -> str:
    """
    Force a specific video for a song and requeue the later stages.

    Returns 'set', 'same', 'not-found', 'ambiguous' or 'bad-url'. The review
    app can only report what went wrong if it is told which thing went wrong:
    a single 'error' had it blaming the link for a pattern that matched two
    songs, and telling the user to paste a proper URL when they just had.
    """
    row = _one_song(db, args.pattern)
    if row is None:
        n = db.conn.execute(
            "SELECT COUNT(*) FROM songs WHERE song_dir LIKE ?",
            (f"%{args.pattern}%",),
        ).fetchone()[0]
        return "ambiguous" if n > 1 else "not-found"
    vid = mt.parse_video_id(args.url)
    if not vid:
        print(f"Could not read a YouTube video ID from {args.url!r}")
        return "bad-url"

    # Requeuing deletes the downloaded source and clears the measured offset.
    # Doing that to arrive back at the same video is pure loss, so check first.
    if row["video_id"] == vid:
        print(f"That is already the video for {row['artist']} - {row['title']}.")
        print("Nothing changed.")
        return "same"

    d = Path(row["song_dir"])
    meta = mt.fetch_metadata(vid, args.cookies)
    title = meta.get("title", vid)
    who = meta.get("uploader", "unknown channel")

    # Drop any previously downloaded source so the new one is fetched clean.
    old = row["source_path"]
    if old and Path(old).exists():
        Path(old).unlink(missing_ok=True)

    # Everything measured belongs to the video being replaced. `review` is an
    # approval of footage that is about to be deleted, and motion, fp_score,
    # dominance and windows all describe it - left behind they read as current
    # measurements of a video nobody has downloaded yet.
    db.update(
        d,
        match_status="ok", video_id=vid, match_score=None,
        match_note=f"MANUAL: {title} [{who}]",
        download_status="pending", source_path=None,
        sync_status="pending", offset_ms=None, spread_ms=None,
        drift_ppm=None, sync_note=None, fp_score=None,
        motion=None, dominance=None, windows=None,
        review=None,
        encode_status="pending", encode_note=None,
        ini_status="pending",
    )
    print(f"Set {row['artist']} - {row['title']}")
    print(f"  -> {title}")
    print(f"     {who}")
    print("\nRequeued. Run: download, sync, encode, ini")
    return "set"


def cmd_offset(args, db: Database) -> None:
    """
    Set a song's offset by hand and lock it.

    Automatic selection is right most of the time and wrong occasionally, and
    no rule tried so far separates the two reliably. For the handful that need
    it, measuring the offset yourself is exact - and marking it MANUAL keeps
    `sync --recheck` from computing it away again.
    """
    row = _one_song(db, args.pattern)
    if row is None:
        return
    d = Path(row["song_dir"])
    old = row["offset_ms"]
    db.update(d, offset_ms=float(args.ms), spread_ms=0.0,
              sync_status="ok", sync_note="MANUAL: offset set by hand",
              review=None, encode_status="pending", ini_status="pending")
    print(f"{row['artist']} - {row['title']}")
    print(f"  offset {old if old is None else f'{old:.0f}'} -> {args.ms} ms (locked)")
    print("\nRun: encode, ini")


def cmd_candidates(args, db: Database) -> None:
    """
    Show every candidate that was considered for a song, and why one won.

    Answers 'the official video exists, why did it pick that instead?' without
    guesswork: a candidate absent from this list was never downloaded (search
    did not return it, or the duration filter culled it), while one present
    with a score of 0.0 was found but failed to download. Anything with a real
    score was heard and lost on ranking.
    """
    songs = db.conn.execute(
        "SELECT song_dir, artist, title, video_id, match_note FROM songs "
        "WHERE song_dir LIKE ? ORDER BY song_dir",
        (f"%{args.pattern}%",),
    ).fetchall()
    if not songs:
        print(f"No song matching {args.pattern!r}")
        return

    for song in songs[:3]:
        print(f"\n{song['artist']} - {song['title']}")
        chart_text = f"{song['artist']} {song['title']}"
        rows = db.conn.execute(
            "SELECT * FROM candidates WHERE song_dir = ?", (song["song_dir"],)
        ).fetchall()
        if not rows:
            print("  no candidates recorded (song not matched yet)")
            continue

        scored = sorted(
            rows,
            key=lambda r: (
                mt.title_preference(r["title"], r["uploader"] or "", chart_text),
                r["score"] or 0.0,
            ),
            reverse=True,
        )
        print(f"  {'pref':>6} {'score':>8} {'cov':>5} {'dur':>6}  title / channel")
        for r in scored:
            pref = mt.title_preference(r["title"], r["uploader"] or "", chart_text)
            mark = " <-- CHOSEN" if r["video_id"] == song["video_id"] else ""
            note = ""
            if (r["score"] or 0) == 0.0:
                note = "  (never downloaded or failed)"
            print(f"  {pref:>+6.1f} {r['score'] or 0:>8.0f} "
                  f"{r['coverage'] or 0:>5.2f} {r['duration'] or 0:>5.0f}s  "
                  f"{r['title'][:52]}{mark}{note}")
            print(f"  {'':>6} {'':>8} {'':>5} {'':>6}  channel: "
                  f"{r['uploader'] or 'unknown'}")


def cmd_review(args, db: Database) -> None:
    """Open the review app - a desktop window by default."""
    n = len(rv.queue(db))
    if n == 0:
        print("Nothing synced yet - run match, download and sync first.")
        return
    print(f"{n} songs ready to check.")

    if getattr(args, "browser", False):
        return rv.serve(Path(args.db), Path(args.work), args.port)
    try:
        from . import app as desktop
    except ImportError:
        print("PySide6 is not installed. Either:\n"
              "  pip install PySide6      (desktop window)\n"
              "  yargvid review --browser (no extra install)")
        return
    sys.exit(desktop.run(Path(args.db), Path(args.work)))


def cmd_offsets(args, db: Database) -> None:
    """
    Compare correlation strength at each candidate offset. Read-only.

    Answers one question: does the audio correlate more strongly at the true
    offset than at the one the pipeline chose? If it does, strength is the
    signal that should pick the winner. If both look alike, it is not, and a
    different approach is needed.
    """
    row = _one_song(db, args.pattern)
    if row is None:
        return
    d = Path(row["song_dir"])
    src = Path(row["source_path"]) if row["source_path"] else None
    if src is None or not src.exists():
        print("No source video on disk - run download for this song first.")
        return

    stems = au.find_stems(d)
    chart = au.mix_stems(stems, fp.SR)
    video = au.decode_mono(src, fp.SR)
    chart_hi = au.mix_stems(stems, sy.REFINE_SR)
    video_hi = au.decode_mono(src, sy.REFINE_SR)
    if chart.size == 0 or video.size == 0:
        print("Could not decode chart or video audio.")
        return

    print(f"{row['artist']} - {row['title']}")
    print(f"  chart {chart.size / fp.SR:.1f}s   video {video.size / fp.SR:.1f}s")

    # An offset predicted from silence alone, independent of any correlation.
    # It only holds when both recordings begin at the same musical point - a
    # video that trims the intro breaks it - but where it does hold it is a
    # completely separate line of evidence.
    chart_lead = au.leading_silence(chart)
    video_lead = au.leading_silence(video)
    predicted = (video_lead - chart_lead) * 1000.0
    print(f"  lead-in: chart {chart_lead:.2f}s, video {video_lead:.2f}s "
          f"-> predicts {predicted:+.0f} ms")
    vid_len = video.size / fp.SR
    chart_len = chart.size / fp.SR

    cands = fp.match_candidates(fp.make_hashes(chart), fp.make_hashes(video),
                                chart.size / fp.SR, top_k=args.top)
    tests = [(f"candidate {i}", c.offset_seconds * 1000.0, c.peak_count)
             for i, c in enumerate(cands, 1)]
    if args.true is not None:
        tests.append(("KNOWN CORRECT", float(args.true), 0))
    if row["offset_ms"] is not None:
        chosen = float(row["offset_ms"])
        if not any(abs(chosen - o) < 50 for _, o, _ in tests):
            tests.append(("stored", chosen, 0))

    print()
    print("  " + "offset".rjust(10) + "hashes".rjust(9) + "win".rjust(5)
          + "strong".rjust(8) + "sharpness".rjust(11) + "spread".rjust(9)
          + "vs lead-in".rjust(12) + "  covers  label")
    print("  " + "-" * 70)
    for label, off, peak in tests:
        r = sy.probe_offset(chart_hi, video_hi, off)
        if not r.get("windows"):
            print(f"  {off:>9.0f}ms {peak:>8} {0:>4}    "
                  f"(only {r.get('usable_s', 0):.0f}s of overlap)  {label}")
            continue
        delta = off - predicted
        covers = "yes" if covers_song(vid_len, off, chart_len) else "NO"
        print(f"  {off:>9.0f}ms {peak:>8} {r['windows']:>4} "
              f"{r['strong']:>3}/{r['windows']:<3} "
              f"{r['sharp_median']:>10.1f} {r['spread_ms']:>7.0f}ms "
              f"{delta:>+10.0f}ms  {covers:>6}  {label}")

    print()
    print("  sharpness = how cleanly the waveforms line up at that offset.")
    print("  Diagnostic only. Choosing the offset by sharpness was tried and")
    print("  reverted: over 128 songs it moved 50 offsets on margins that were")
    print("  noise (111 vs 111, 91 vs 90), scattering them by +/-80s. See NOTES.")


def cmd_links(args, db: Database) -> None:
    """
    The chosen video URL for every song matching a folder substring.

    Read-only, and the one thing the database holds that cannot be opened from
    anywhere else: `candidates` shows what was considered and `inspect` shows
    where the offset came from, but neither hands you a link you can paste.
    """
    rows = db.conn.execute(
        "SELECT * FROM songs WHERE song_dir LIKE ? AND video_id IS NOT NULL "
        "ORDER BY artist, title",
        (f"%{args.pattern}%",),
    ).fetchall()
    if not rows:
        print(f"No song with a chosen video matches {args.pattern!r}")
        return
    for r in rows:
        print(f"https://youtu.be/{r['video_id']}  "
              f"{r['artist']} - {r['title']}")


def cmd_reviewed(args, db: Database) -> None:
    """List songs you have confirmed by eye, with their measurements."""
    rows = db.conn.execute(
        "SELECT * FROM songs WHERE review = 'keep' ORDER BY artist, title"
    ).fetchall()
    if not rows:
        print("No songs marked as checked yet.")
        return

    print(f"{len(rows)} confirmed by eye\n")
    print("  " + "offset".rjust(10) + "spread".rjust(9) + "match".rjust(9)
          + "motion".rjust(9) + "  song")
    print("  " + "-" * 76)
    for r in rows:
        off = r["offset_ms"]
        spread = r["spread_ms"]
        motion = r["motion"]
        print("  "
              + (f"{off:>9.0f}ms" if off is not None else "        —")
              + (f"{spread:>8.0f}ms" if spread is not None and spread >= 0
                 else "       —")
              + (f"{r['fp_score']:>9.0f}" if r["fp_score"] else "        —")
              + (f"{motion:>9.2f}" if motion is not None else "        —")
              + f"  {r['artist']} - {r['title']}")
    if args.verbose:
        print()
        for r in rows:
            print(f"  {r['artist']} - {r['title']}")
            print(f"      {r['match_note'] or ''}")


def cmd_export(args, db: Database) -> None:
    """
    Recompute every measurement for every song and write one CSV row each.

    The point is to stop reasoning from a handful of hand-picked examples.
    Everything the pipeline can measure goes in the same table so a pattern
    has to hold across the whole set, not just the songs that happened to get
    looked at.
    """
    import csv

    # The baseline export is the only record of what the pipeline measured
    # before a change, and the default --out is the name it was written under.
    # Re-running export to look at something destroys the thing every result
    # is compared against, after twenty minutes of decoding.
    out = Path(args.out)
    if out.exists() and not getattr(args, "force", False):
        print(f"{out} already exists.")
        print("Pass a different --out, or --force to overwrite it.")
        return

    rows = db.conn.execute(
        "SELECT * FROM songs WHERE source_path IS NOT NULL "
        "AND sync_status IN ('ok','drift','unverified','rejected') "
        "ORDER BY artist, title"
    ).fetchall()
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        print("Nothing to export - no songs have a downloaded video.")
        return

    print(f"Measuring {len(rows)} songs -> {out}")
    print("This decodes and fingerprints each one; expect a few seconds each.")

    # manual_video / manual_offset: a hand-picked video and a hand-set offset
    # are decisions, not measurements, and any rule measured over this file has
    # to be able to leave them out. Without the columns they were invisible -
    # the two approvals the fall-through rule touched had to be found by hand.
    fields = ["artist", "title", "reviewed", "sync_status", "chart_s",
              "video_s", "chart_lead_s", "video_lead_s", "chosen_offset_ms",
              "spread_ms", "fp_score", "motion", "covers_song", "dominance",
              "manual_video", "manual_offset",
              "video_id", "channel", "blocks", "largest_block_pct",
              "largest_block_offset_ms"]
    for i in range(1, 5):
        fields += [f"c{i}_offset_ms", f"c{i}_hashes", f"c{i}_sharp",
                   f"c{i}_windows", f"c{i}_strong", f"c{i}_spread_ms"]

    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for n, r in enumerate(rows, 1):
            d = Path(r["song_dir"])
            src = Path(r["source_path"])
            print(f"[{n}/{len(rows)}] {r['artist']} - {r['title']}", flush=True)
            if not src.exists():
                continue
            stems = au.find_stems(d)
            chart = au.mix_stems(stems, fp.SR)
            video = au.decode_mono(src, fp.SR)
            if chart.size == 0 or video.size == 0:
                continue
            chart_hi = au.mix_stems(stems, sy.REFINE_SR)
            video_hi = au.decode_mono(src, sy.REFINE_SR)

            chart_s, video_s = chart.size / fp.SR, video.size / fp.SR
            off = r["offset_ms"] or 0
            note = r["match_note"] or ""
            rec = {
                "artist": r["artist"], "title": r["title"],
                "reviewed": r["review"] or "", "sync_status": r["sync_status"],
                "chart_s": round(chart_s, 1), "video_s": round(video_s, 1),
                "chart_lead_s": round(au.leading_silence(chart), 2),
                "video_lead_s": round(au.leading_silence(video), 2),
                "chosen_offset_ms": round(off),
                "spread_ms": r["spread_ms"], "fp_score": r["fp_score"],
                "motion": r["motion"],
                "covers_song": int(covers_song(video_s, off, chart_s)),
                "manual_video": int(note.startswith("MANUAL")),
                "manual_offset": int(
                    (r["sync_note"] or "").startswith("MANUAL")),
                "video_id": r["video_id"] or "",
                "channel": note.rsplit("[", 1)[-1].rstrip("]") if "[" in note else "",
            }
            cands = fp.match_candidates(fp.make_hashes(chart),
                                        fp.make_hashes(video), chart_s, top_k=4)

            # Structural, not statistical: a video with a skit or an extended
            # solo genuinely aligns at different offsets in different stretches
            # of the song. That is measurable without anyone labelling it, and
            # it is the failure mode every confidence metric has missed.
            blocks = sy.analyse_blocks(
                chart_hi, video_hi, [c.offset_seconds * 1000.0 for c in cands])
            rec["blocks"] = len(blocks)
            if blocks:
                big = max(blocks, key=lambda b: b["covers"])
                rec["largest_block_pct"] = round(big["covers"] * 100)
                rec["largest_block_offset_ms"] = round(big["offset_ms"])
            else:
                rec["largest_block_pct"] = ""
                rec["largest_block_offset_ms"] = ""
            rec["dominance"] = round(
                cands[0].peak_count / max(cands[1].peak_count, 1), 2
            ) if len(cands) > 1 else ""
            for i, c in enumerate(cands[:4], 1):
                pr = sy.probe_offset(chart_hi, video_hi, c.offset_seconds * 1000.0)
                rec[f"c{i}_offset_ms"] = round(c.offset_seconds * 1000)
                rec[f"c{i}_hashes"] = c.peak_count
                rec[f"c{i}_sharp"] = round(pr.get("sharp_median", 0), 1)
                rec[f"c{i}_windows"] = pr.get("windows", 0)
                rec[f"c{i}_strong"] = pr.get("strong", 0)
                rec[f"c{i}_spread_ms"] = round(pr.get("spread_ms", -1), 1)
            w.writerow(rec)
            fh.flush()

    print(f"\nWritten to {out.resolve()}")


def cmd_blocks(args, db: Database) -> None:
    """Show which stretches of a song align, and at what offset. Read-only."""
    row = _one_song(db, args.pattern)
    if row is None:
        return
    src = Path(row["source_path"]) if row["source_path"] else None
    if src is None or not src.exists():
        print("No source video on disk - run download for this song first.")
        return

    d = Path(row["song_dir"])
    stems = au.find_stems(d)
    chart = au.mix_stems(stems, fp.SR)
    video = au.decode_mono(src, fp.SR)
    if chart.size == 0 or video.size == 0:
        print("Could not decode chart or video audio.")
        return
    chart_hi = au.mix_stems(stems, sy.REFINE_SR)
    video_hi = au.decode_mono(src, sy.REFINE_SR)
    chart_len = chart.size / fp.SR

    cands = fp.match_candidates(fp.make_hashes(chart), fp.make_hashes(video),
                                chart_len, top_k=args.top)
    blocks = sy.analyse_blocks(chart_hi, video_hi,
                               [c.offset_seconds * 1000.0 for c in cands])

    print(f"{row['artist']} - {row['title']}")
    print(f"  chart {chart_len:.1f}s   video {video.size / fp.SR:.1f}s   "
          f"stored offset {row['offset_ms'] or 0:.0f} ms")
    if not blocks:
        print()
        print("  Could not align any part of this song to the video.")
        return

    def stamp(frac):
        secs = int(frac * chart_len)
        return f"{secs // 60}:{secs % 60:02d}"

    print()
    print(f"  {'song range':<24}{'offset':>11}{'covers':>9}   windows")
    print("  " + "-" * 56)
    for b in blocks:
        rng = (f"{b['start'] * 100:3.0f}%-{b['end'] * 100:3.0f}%  "
               f"({stamp(b['start'])}-{stamp(b['end'])})")
        print(f"  {rng:<24}{b['offset_ms']:>+9.0f}ms{b['covers'] * 100:>8.0f}%"
              f"{b['windows']:>10}")

    biggest = max(blocks, key=lambda b: b["covers"])
    print()
    if len(blocks) == 1:
        print("  One continuous alignment - no internal cuts detected.")
        return
    print(f"  {len(blocks)} blocks: the video has internal cuts - a skit, an")
    print("  extended solo, or a different edit. YARG applies a single offset,")
    print("  so only one block can be in time.")
    print()
    print(f"  Largest block covers {biggest['covers'] * 100:.0f}% of the song "
          f"at {biggest['offset_ms']:+.0f} ms:")
    print(f"      yargvid offset \"{args.pattern}\" {biggest['offset_ms']:.0f}")


def cmd_videos(args, db: Database) -> None:
    """
    Which song folders already hold a video, and where each one came from.

    Files on disk and the database can disagree in two different ways, and the
    difference matters. A preview writes a real video.webm without marking the
    song encoded. A file left by an earlier project is invisible to this
    pipeline but very much visible to YARG, which has been playing it with
    whatever offset produced it. Encoding overwrites both.
    """
    rows = db.conn.execute("SELECT * FROM songs ORDER BY artist, title").fetchall()
    encoded, previews, foreign, sources, missing = [], [], [], [], []

    for r in rows:
        d = Path(r["song_dir"])
        webm = d / "video.webm"
        if webm.exists():
            entry = (r, webm.stat().st_size / 1_048_576)
            if r["encode_status"] == "ok":
                encoded.append(entry)
            elif r["sync_status"] in ("ok", "drift", "unverified"):
                previews.append(entry)      # this pipeline made it
            else:
                foreign.append(entry)       # predates this project
        elif r["encode_status"] == "ok":
            missing.append(r)
        try:
            if any(d.glob("video.src.*")):
                sources.append(r)
        except OSError:
            pass

    groups = (("encoded", encoded), ("preview", previews),
              ("pre-existing", foreign))
    total = sum(s for _, g in groups for _, s in g)
    print(f"{sum(len(g) for _, g in groups)} song folders contain video.webm "
          f"({total / 1024:.2f} GB)")
    print(f"  {len(encoded):>5} encoded by this pipeline")
    print(f"  {len(previews):>5} previews - synced here, not encoded yet")
    print(f"  {len(foreign):>5} NOT from this pipeline - predate this project")
    print(f"  {len(missing):>5} marked encoded but the file is gone")
    print(f"  {len(sources):>5} still hold a downloaded source file")

    if getattr(args, "mark", False):
        # Record the classification now. Once sync finishes for these songs
        # they become indistinguishable from this pipeline's own work, and the
        # distinction is the whole point - encode overwrites either.
        n = 0
        for state, group in (("preview", previews), ("foreign", foreign)):
            for r, _ in group:
                db.update(Path(r["song_dir"]), existing_video=state)
                n += 1
        print(f"\nMarked {n} songs as already having a video.")
        print("They appear under 'Has a video' in the review application.")
        return

    if args.out:
        out = Path(args.out)
        with out.open("w", encoding="utf-8") as fh:
            fh.write("state,size_mb,artist,title,folder\n")
            for state, group in groups:
                for r, size in group:
                    fh.write(f'{state},{size:.1f},"{r["artist"]}",'
                             f'"{r["title"]}","{r["song_dir"]}"\n')
            for r in missing:
                fh.write(f'file_missing,,"{r["artist"]}",'
                         f'"{r["title"]}","{r["song_dir"]}"\n')
        print(f"\nFull list written to {out.resolve()}")
        return

    if args.quiet:
        return

    def dump(title, group):
        if not group:
            return
        print(f"\n{title} - {len(group)}, "
              f"{sum(s for _, s in group) / 1024:.2f} GB:")
        for r, size in sorted(group, key=lambda x: -x[1]):
            print(f"  {size:7.1f} MB  {r['artist']} - {r['title']}")

    dump("Previews from this pipeline", previews)
    dump("Pre-existing, made before this project", foreign)

    if missing:
        print(f"\nMarked encoded but no file present - {len(missing)}:")
        for r in missing:
            print(f"  {r['artist']} - {r['title']}")

    if previews or foreign:
        print("\nEncoding overwrites all of the above; --skip-existing "
              "leaves them alone.")
    if foreign:
        print(f"\nThe {len(foreign)} pre-existing files were made before this "
              f"project. YARG is")
        print("playing them now, with offsets nothing here has checked.")


def cmd_status(args, db: Database) -> None:
    counts = db.counts()
    cols = ("pending", "ok", "unverified", "drift", "rejected", "failed")
    print(f"{'stage':<10}" + "".join(f"{s:>12}" for s in cols))
    print("-" * (10 + 12 * len(cols)))
    for stage in STAGES:
        c = counts[stage]
        print(f"{stage:<10}" + "".join(f"{c.get(s, 0):>12}" for s in cols))
        # Anything not in cols would otherwise vanish from the report.
        extra = {k: v for k, v in c.items() if k not in cols}
        if extra:
            print(f"{'':<10} other: {extra}")

    flagged = db.flagged_matches()
    if flagged:
        print(f"\n{len(flagged)} matched but flagged as a still image "
              f"(audio-only upload or a '- Topic' channel). First 15:")
        for row in flagged[:15]:
            note = (row["match_note"] or "").removeprefix("REVIEW: ")
            m = row["motion"]
            measured = f"  [motion {m:.2f}]" if m is not None and m >= 0 else ""
            print(f"  {row['artist']} - {row['title']}: {note[:52]}{measured}")

    queue = db.manual_queue()
    if queue:
        print(f"\n{len(queue)} songs need manual review. First 15:")
        for row in queue[:15]:
            note = (row["match_note"] or row["download_note"]
                    or row["sync_note"] or row["encode_note"] or "")
            print(f"  {row['artist']} - {row['title']}: {note[:70]}")


def cmd_retry(args, db: Database) -> None:
    n = db.reset(args.stage, only_failed=not args.all)
    print(f"Reset {n} rows in stage '{args.stage}' to pending")


JS_RUNTIMES = ("deno", "node", "bun")


def cmd_doctor(args, db: Database) -> None:
    # Without a JavaScript runtime yt-dlp cannot solve YouTube's signature
    # challenges, and the response comes back with every audio and video
    # format stripped out - which reads as a download bug rather than as a
    # missing tool. Any one of the three will do.
    js = [t for t in JS_RUNTIMES if enc.have(t)]
    checks = [
        ("ffmpeg", enc.have("ffmpeg")),
        ("ffprobe", enc.have("ffprobe")),
        ("yt-dlp", enc.have("yt-dlp")),
        ("libvpx (VP8) in ffmpeg", enc.check_ffmpeg_vp8()),
        ("a JavaScript runtime for yt-dlp "
         f"({'/'.join(JS_RUNTIMES)}{': ' + js[0] if js else ''})", bool(js)),
    ]
    for name, ok in checks:
        print(f"  [{'ok' if ok else 'MISSING'}] {name}")
    if not all(ok for _, ok in checks):
        print("\nInstall the missing tools before running the pipeline.")


# -------------------------------------------------------------------- main ---

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="yargvid")
    p.add_argument("--db", default="yargvid.sqlite")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--sample", action="store_true",
                   help="draw a random sample rather than the first N")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("doctor", help="check required tools");  s.set_defaults(fn=cmd_doctor)  # noqa: E501
    s = sub.add_parser("index",  help="scan the song library")
    s.add_argument("root"); s.set_defaults(fn=cmd_index)

    s = sub.add_parser("match", help="find + verify videos by audio")
    s.add_argument("--work", default="./.yargvid_work")
    s.add_argument("--cookies", default=None,
                   help="browser name (firefox) or path to a cookies.txt file")
    s.add_argument("--sleep", type=float, default=1.0,
                   help="seconds between yt-dlp requests (default 1.0)")
    s.add_argument("--gate", type=float, default=None,
                   help=f"fingerprint accept score "
                        f"(default {fp.ACCEPT_SCORE:.0f})")
    s.add_argument("--redo", action="store_true",
                   help="only re-attempt songs that previously failed")
    s.set_defaults(fn=cmd_match)

    s = sub.add_parser("download", help="fetch the winning videos")
    s.add_argument("--height", type=int, default=1080)
    s.add_argument("--cookies", default=None)
    s.add_argument("--sleep", type=float, default=1.0)
    s.set_defaults(fn=cmd_download)

    s = sub.add_parser("sync", help="estimate and verify offsets")
    s.add_argument("--recheck", action="store_true",
                   help="recompute already-synced songs, write only changes")
    s.add_argument("--min-offset", type=float, default=0.0,
                   help="with --recheck, skip songs whose offset is smaller "
                        "than this many ms")
    s.add_argument("--skip-reviewed", action="store_true",
                   help="with --recheck, leave songs you have already "
                        "confirmed alone")
    s.set_defaults(fn=cmd_sync)

    s = sub.add_parser("encode", help="transcode to VP8 webm")
    s.add_argument("--height", type=int, default=1080)
    s.add_argument("--crf", type=int, default=31)
    s.add_argument("--cpu-used", type=int, default=3)
    s.add_argument("--threads", type=int, default=2)
    s.add_argument("--workers", type=int, default=None)
    s.add_argument("--preview", action="store_true",
                   help="low-res full-length encode to check sync in YARG")
    s.add_argument("--preview-height", type=int, default=480)
    s.add_argument("--skip-static", action="store_true",
                   help="leave album-art backgrounds unencoded")
    s.add_argument("--skip-existing", action="store_true",
                   help="leave folders that already contain a video.webm")
    s.add_argument("--reviewed", action="store_true",
                   help="only encode songs marked \u2018Looks right\u2019")
    s.set_defaults(fn=cmd_encode)

    s = sub.add_parser("ini", help="write video_start_time"); s.set_defaults(fn=cmd_ini)
    s = sub.add_parser("diagnose", help="debug one song's match, verbosely")
    s.add_argument("song_dir", nargs="?", default=None)
    s.add_argument("--work", default="./.yargvid_work")
    s.add_argument("--cookies", default=None); s.set_defaults(fn=cmd_diagnose)

    s = sub.add_parser("check", help="test a specific video URL against a chart")
    s.add_argument("pattern", help="substring of the song folder path")
    s.add_argument("url", help="YouTube URL or video ID")
    s.add_argument("--work", default="./.yargvid_work")
    s.add_argument("--cookies", default=None)
    s.add_argument("--sleep", type=float, default=1.0)
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("set", help="force a specific video for a song")
    s.add_argument("pattern", help="substring of the song folder path")
    s.add_argument("url", help="YouTube URL or video ID")
    s.add_argument("--cookies", default=None)
    s.set_defaults(fn=cmd_set)

    s = sub.add_parser("offset", help="set a song's offset by hand and lock it")
    s.add_argument("pattern")
    s.add_argument("ms", type=int, help="video_start_time in milliseconds")
    s.set_defaults(fn=cmd_offset)

    s = sub.add_parser("candidates", help="show all candidates for a song")
    s.add_argument("pattern", help="substring of the song folder path")
    s.set_defaults(fn=cmd_candidates)

    s = sub.add_parser("blocks", help="find internal cuts in a video")
    s.add_argument("pattern")
    s.add_argument("--top", type=int, default=4)
    s.set_defaults(fn=cmd_blocks)

    s = sub.add_parser("offsets", help="compare correlation strength per offset")
    s.add_argument("pattern")
    s.add_argument("--true", type=float, default=None,
                   help="known-correct offset in ms, e.g. -12000")
    s.add_argument("--top", type=int, default=4)
    s.set_defaults(fn=cmd_offsets)

    s = sub.add_parser("inspect", help="explain a song's offset (offline)")
    s.add_argument("pattern", help="substring of the song folder path")
    s.set_defaults(fn=cmd_inspect)

    s = sub.add_parser("review", help="open the review app in a window")
    s.add_argument("--browser", action="store_true",
                   help="serve in a browser instead of a window")
    s.add_argument("--port", type=int, default=8770)
    s.add_argument("--work", default="./.yargvid_work")
    s.set_defaults(fn=cmd_review)

    s = sub.add_parser("export", help="measure every song into a CSV")
    s.add_argument("--out", default="yargvid_analysis.csv")
    s.add_argument("--force", action="store_true",
                   help="overwrite the output file if it already exists")
    s.set_defaults(fn=cmd_export)

    s = sub.add_parser("videos", help="which folders already hold a video")
    s.add_argument("--quiet", action="store_true", help="counts only")
    s.add_argument("--out", default=None,
                   help="write the full list to a CSV instead of printing")
    s.add_argument("--mark", action="store_true",
                   help="record which songs already have a video, for review")
    s.set_defaults(fn=cmd_videos)

    s = sub.add_parser("links", help="print the chosen video URL per song")
    s.add_argument("pattern", help="substring of the song folder path")
    s.set_defaults(fn=cmd_links)

    s = sub.add_parser("reviewed", help="songs you confirmed by eye")
    s.add_argument("--verbose", action="store_true",
                   help="also show which video each one uses")
    s.set_defaults(fn=cmd_reviewed)

    s = sub.add_parser("status", help="progress and manual queue"); s.set_defaults(fn=cmd_status)  # noqa: E501

    s = sub.add_parser("retry", help="reset a stage to pending")
    s.add_argument("stage", choices=STAGES)
    s.add_argument("--all", action="store_true"); s.set_defaults(fn=cmd_retry)

    args = p.parse_args(argv)

    # `doctor` checks the tools installed on this machine and has nothing to
    # ask a database. Opening one CREATES it, so the command whose whole job
    # is to reassure you left an empty yargvid.sqlite behind in whatever
    # folder you ran it from - and an empty database looks exactly like a
    # library with every song lost.
    if args.cmd == "doctor":
        args.fn(args, None)
        return 0

    # Only `index` should ever create a database. Every other command opening a
    # missing file would quietly make an empty one, which looks identical to
    # having lost all your work - and the default --db is a RELATIVE path, so
    # running from a different folder is enough to trigger it.
    db_file = Path(args.db)
    if not db_file.exists() and args.cmd != "index":
        print(f"No database at {db_file.resolve()}")
        print()
        print("The database lives wherever you first ran `index`.")
        print("Either cd to that folder, or pass the path:")
        print(f"    yargvid --db C:\\path\\to\\yargvid.sqlite {args.cmd}")
        return 1

    db = Database(db_file)
    try:
        args.fn(args, db)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
