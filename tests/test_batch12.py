"""
Batch 12 - source sizes, and a full-quality encode that keeps its source.

Batch 11 made `--height` a ceiling and gave the encoder four quality tiers,
which together mean an upper tier only buys anything when the SOURCE has
more detail than 1080p. Nothing in the database records what was downloaded
or what was on offer, so `--quality super` is currently a way to spend three
times the disk re-encoding 1080p footage. This batch makes the source size a
stored fact, backfills it, and lets `download` go and get the bigger file
where one exists.

    S1  Two columns on `songs`: `source_height`, what was downloaded, and
        `source_max_height`, the largest height YouTube offers. Both arrive
        through the existing ALTER TABLE path, so a database made before
        them gains them on open.
    S2  `match.heights_from_info(info)` reads both out of one yt-dlp info
        dict: the top-level `height`, or the largest of `requested_formats`
        when a merge left none, against the largest height in `formats`.
        Audio-only entries have no height and are ignored. What was
        downloaded is proof that it is on offer, so the offered figure is
        never below it. Nothing known at all is (None, None).
    S3  `download_video` asks for that dict (`--print-json`) and returns it
        alongside the path, so a download costs no extra request. Its new
        `keep_existing` argument holds an existing source until the new file
        is on disk: a failed re-fetch must not leave a song with no video.
        Off by default, which is today's behaviour unchanged.
    S4  `tag-sources` backfills both columns for rows that lack them, one
        `fetch_metadata` call per song, honouring `--reviewed` and pausing
        `--sleep` between requests. The file on disk is what we actually
        have, so `encode.source_height(path)` wins over the info dict for
        `source_height`; the dict is what gets written when there is no
        file. Nothing else about the row is touched.
    S5  `download --quality best|super` raises the ceiling to 2160, a typed
        `--height` overrides it, and `download --upgrade` re-fetches only
        the songs that would actually gain: `source_max_height` above
        `source_height`, bounded by that ceiling. An upgrade is the same
        video, so `video_id`, `offset_ms`, `sync_status` and the approval
        all survive it and only `encode_status` goes back to pending. A
        failed one changes nothing but the note.
    S6  `cli.source_size_line(rows)` is the one sentence both `estimate` and
        `videos` print: how many of those songs have a source above 1080p,
        and how to fill the gap when some have no recorded size.
    S7  A passenger, unrelated to source sizes: `encode --keep-source` is a
        full-quality encode that keeps the source and leaves the row
        pending, so two tiers can be encoded from one download and compared
        by eye. Today only `--preview` keeps a source, and a preview is
        480p under the codec's own 800k ceiling, so it cannot show a tier
        difference at all.

Nothing here reaches YouTube or ffmpeg: `match._run` is faked, `audio.probe`
is faked, and `encode_many` is a fake pool. Every path is a `Path`.

Run from the repository root:  pytest -q
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from yargvid import audio as au
from yargvid import cli
from yargvid import encode as enc
from yargvid import match as mt
from yargvid.db import Database


# ----------------------------------------------------------------- helpers ---

@pytest.fixture
def db(tmp_path):
    d = Database(tmp_path / "t.sqlite")
    yield d
    d.close()


def song(db, path, **cols):
    """A downloaded, synced, unencoded song with a file on disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "song.ini").write_text("[song]\nname = X\n", encoding="utf-8")
    src = path / "video.src.mkv"
    src.write_bytes(b"old")
    db.add_song(path, "Artist", path.name, 100.0)
    base = dict(match_status="ok", video_id="aaaaaaaaaaa",
                download_status="ok", source_path=str(src),
                sync_status="ok", offset_ms=1234.0, motion=0.5,
                video_seconds=200.0)
    base.update(cols)
    db.update(path, **base)
    return path


def row(db, path):
    return db.conn.execute(
        "SELECT * FROM songs WHERE song_dir = ?", (str(path),)).fetchone()


def rows_of(db, *paths):
    return [row(db, p) for p in paths]


def info_dict(height=1080, offered=(1080, 2160), **extra):
    """One yt-dlp info dict: what was taken, and what was on the shelf."""
    formats = [{"format_id": "140", "acodec": "mp4a", "height": None}]
    formats += [{"format_id": str(h), "height": h} for h in offered]
    out = {"id": "aaaaaaaaaaa", "formats": formats}
    if height is not None:
        out["height"] = height
    out.update(extra)
    return out


def fake_yt_dlp(monkeypatch, info=None, ok=True, content=b"new"):
    """`match._run` that writes whatever `-o` names and prints the JSON."""
    calls = []

    def run(cmd, timeout=600):
        calls.append(list(cmd))
        if not ok:
            return subprocess.CompletedProcess(cmd, 1, "", "yt-dlp said no")
        out = Path(cmd[cmd.index("-o") + 1].replace("%(ext)s", "mkv"))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(content)
        text = "" if info is None else json.dumps(info)
        return subprocess.CompletedProcess(cmd, 0, text + "\n", "")

    monkeypatch.setattr(mt, "_run", run)
    return calls


def fake_probe(monkeypatch, height=720, streams=None):
    """`audio.probe` reporting one video stream of `height`, plus audio."""
    if streams is None:
        streams = [{"codec_type": "video", "height": height},
                   {"codec_type": "audio"}]
    monkeypatch.setattr(au, "probe", lambda p: {"streams": list(streams)})


def fake_metadata(monkeypatch, by_id=None, default=None):
    """`match.fetch_metadata`, recording every video id it was asked for."""
    asked = []

    def fetch(video_id, cookies=None):
        asked.append(video_id)
        if by_id is not None:
            return by_id.get(video_id, {})
        return {} if default is None else default

    monkeypatch.setattr(mt, "fetch_metadata", fetch)
    return asked


def download_args(**over):
    a = dict(limit=None, sample=False, height=None, quality=None,
             cookies=None, sleep=0.0, upgrade=False)
    a.update(over)
    return SimpleNamespace(**a)


def tag_args(**over):
    a = dict(limit=None, sample=False, reviewed=False, cookies=None,
             sleep=0.0)
    a.update(over)
    return SimpleNamespace(**a)


def encode_args(**over):
    a = dict(limit=None, sample=False, skip_existing=False, reviewed=False,
             skip_static=True, height=1080, crf=None, cpu_used=3, threads=2,
             workers=None, preview=False, preview_height=480,
             bitrate_cap=None, max_fps=None, codec="vp8", fps=None,
             size_lock=None, quality=None, keep_source=False)
    a.update(over)
    return SimpleNamespace(**a)


def estimate_args(**over):
    a = encode_args(measure=False, sample_size=enc.SAMPLE_SONGS)
    return SimpleNamespace(**{**vars(a), **over})


def parsed(monkeypatch, db, argv, cmd):
    """Run `main` far enough to parse; hand back the namespace."""
    seen = {}
    monkeypatch.setattr(cli, cmd, lambda args, db_: seen.update(vars(args)))
    assert cli.main(["--db", str(db.path), *argv]) == 0
    return SimpleNamespace(**seen)


class FakePool:
    """Stands in for `encode_many`: records the call, reports each job."""

    def __init__(self):
        self.failing = set()
        self.calls = []

    def __call__(self, jobs, settings, workers=None, on_done=None, **kw):
        self.calls.append(dict(jobs=list(jobs), settings=settings,
                               workers=workers, kw=kw))
        results = {}
        for _src, d in jobs:
            results[d] = ((False, "boom") if d.name in self.failing
                          else (True, ""))
            if on_done:
                on_done(d, results[d])
        return results


@pytest.fixture
def pool(monkeypatch):
    fake = FakePool()
    monkeypatch.setattr(enc, "encode_many", fake)
    monkeypatch.setattr(enc, "resolve_codec", lambda s, say=print: s)
    return fake


# ------------------------------------------------------------ S1 columns -----

def test_S1_songs_table_has_both_source_size_columns(db):
    cols = {r[1] for r in db.conn.execute("PRAGMA table_info(songs)")}
    assert {"source_height", "source_max_height"} <= cols


def test_S1_existing_database_gains_the_columns_on_open(tmp_path):
    # A database made before they existed must be migrated the way
    # candidates.view_count is, not left a column short and unreadable.
    path = tmp_path / "old.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE songs (song_dir TEXT PRIMARY KEY, match_status TEXT, "
        "download_status TEXT, sync_status TEXT, encode_status TEXT, "
        "ini_status TEXT, review TEXT, source_path TEXT, updated_at TEXT);"
    )
    conn.execute("INSERT INTO songs (song_dir) VALUES ('/lib/a')")
    conn.commit()
    conn.close()
    d = Database(path)
    try:
        cols = {r[1] for r in d.conn.execute("PRAGMA table_info(songs)")}
        kept = d.conn.execute("SELECT COUNT(*) FROM songs").fetchone()[0]
    finally:
        d.close()
    assert {"source_height", "source_max_height"} <= cols
    assert kept == 1


def test_S1_the_columns_start_empty_and_hold_numbers(db):
    s = song(db, db.path.parent / "a")
    assert row(db, s)["source_height"] is None
    assert row(db, s)["source_max_height"] is None
    db.update(s, source_height=720, source_max_height=2160)
    assert row(db, s)["source_height"] == 720
    assert row(db, s)["source_max_height"] == 2160


# -------------------------------------------------------------- S2 reading ---

def test_S2_reads_the_downloaded_height_and_the_largest_offered():
    assert mt.heights_from_info(info_dict(1080, (720, 1080, 2160))) == (
        1080, 2160)


def test_S2_falls_back_to_requested_formats_when_a_merge_left_no_height():
    # A bestvideo+bestaudio merge reports the two formats it joined rather
    # than one height for the result. The video half is the picture size.
    info = {"requested_formats": [{"height": 1440}, {"height": None}],
            "formats": [{"height": 720}, {"height": 1440}]}
    assert mt.heights_from_info(info) == (1440, 1440)


def test_S2_audio_only_formats_are_not_heights():
    info = {"height": 720,
            "formats": [{"height": None}, {"height": 0}, {"height": 720}]}
    assert mt.heights_from_info(info) == (720, 720)


def test_S2_nothing_known_is_two_nones():
    assert mt.heights_from_info({}) == (None, None)
    assert mt.heights_from_info({"formats": []}) == (None, None)


def test_S2_what_was_downloaded_is_never_above_what_is_offered():
    # The file is proof the size exists, whatever the format list says.
    got, offered = mt.heights_from_info(
        {"height": 1080, "formats": [{"height": 720}]})
    assert (got, offered) == (1080, 1080)
    got, offered = mt.heights_from_info({"height": 1080})
    assert (got, offered) == (1080, 1080)


def test_S2_a_height_that_is_not_a_number_is_no_height():
    assert mt.heights_from_info({"height": "tall"}) == (None, None)


# ------------------------------------------------------------ S3 download ----

def test_S3_the_download_asks_for_the_info_json(tmp_path, monkeypatch):
    fake_probe(monkeypatch)
    calls = fake_yt_dlp(monkeypatch, info=info_dict())
    mt.download_video("aaaaaaaaaaa", tmp_path / "video")
    assert "--print-json" in calls[0]


def test_S3_download_video_returns_the_info_dict(tmp_path, monkeypatch):
    fake_probe(monkeypatch)
    fake_yt_dlp(monkeypatch, info=info_dict(1080, (1080, 2160)))
    src, note, info = mt.download_video("aaaaaaaaaaa", tmp_path / "video")
    assert src is not None and note == ""
    assert mt.heights_from_info(info) == (1080, 2160)


def test_S3_unreadable_json_is_an_empty_dict_not_a_failed_download(
        tmp_path, monkeypatch):
    # The video is on disk. A size we could not read is a gap in a report,
    # not a reason to throw the download away.
    fake_probe(monkeypatch)
    monkeypatch.setattr(mt, "_run", lambda cmd, timeout=600: (
        (tmp_path / "video.src.mkv").write_bytes(b"x"),
        subprocess.CompletedProcess(cmd, 0, "not json at all", ""))[1])
    src, note, info = mt.download_video("aaaaaaaaaaa", tmp_path / "video")
    assert src is not None and note == "" and info == {}


def test_S3_by_default_a_stale_source_is_cleared_before_the_fetch(
        tmp_path, monkeypatch):
    # Unchanged behaviour: a leftover from an earlier match must not be
    # picked up by the glob when the new download uses another container.
    (tmp_path / "video.src.webm").write_bytes(b"old")
    fake_probe(monkeypatch)
    fake_yt_dlp(monkeypatch, ok=False)
    src, note, info = mt.download_video("aaaaaaaaaaa", tmp_path / "video")
    assert src is None and note
    assert not list(tmp_path.glob("video.src.*"))


def test_S3_keep_existing_leaves_the_old_video_when_the_fetch_fails(
        tmp_path, monkeypatch):
    old = tmp_path / "video.src.mkv"
    old.write_bytes(b"old")
    fake_probe(monkeypatch)
    fake_yt_dlp(monkeypatch, ok=False)
    src, note, info = mt.download_video(
        "aaaaaaaaaaa", tmp_path / "video", keep_existing=True)
    assert src is None and note
    assert old.exists() and old.read_bytes() == b"old"


def test_S3_keep_existing_replaces_the_old_video_once_the_new_one_lands(
        tmp_path, monkeypatch):
    old = tmp_path / "video.src.mkv"
    old.write_bytes(b"old")
    fake_probe(monkeypatch)
    fake_yt_dlp(monkeypatch, info=info_dict(), content=b"new")
    src, note, info = mt.download_video(
        "aaaaaaaaaaa", tmp_path / "video", keep_existing=True)
    left = sorted(p.name for p in tmp_path.glob("video.*"))
    assert note == ""
    assert src is not None and src.read_bytes() == b"new"
    # One source file, under the name every other stage looks for.
    assert left == ["video.src.mkv"]


def test_S3_a_silent_file_still_fails_with_the_old_video_intact(
        tmp_path, monkeypatch):
    # The no-audio rejection deletes what it downloaded. With an older video
    # held back, that must not take the older video with it.
    old = tmp_path / "video.src.mkv"
    old.write_bytes(b"old")
    fake_probe(monkeypatch, streams=[{"codec_type": "video", "height": 720}])
    fake_yt_dlp(monkeypatch, info=info_dict())
    src, note, info = mt.download_video(
        "aaaaaaaaaaa", tmp_path / "video", keep_existing=True)
    assert src is None and "audio" in note
    assert old.exists() and old.read_bytes() == b"old"


def test_S3_the_height_ceiling_still_reaches_the_format_selector(
        tmp_path, monkeypatch):
    fake_probe(monkeypatch)
    calls = fake_yt_dlp(monkeypatch, info=info_dict())
    mt.download_video("aaaaaaaaaaa", tmp_path / "video", max_height=2160)
    assert any("height<=2160" in part for part in calls[0])


# ---------------------------------------------------------- S4 tag-sources ---

def test_S4_tag_sources_is_a_subcommand_with_its_own_defaults(db,
                                                              monkeypatch):
    a = parsed(monkeypatch, db, ["tag-sources"], "cmd_tag_sources")
    assert a.reviewed is False and a.cookies is None
    a = parsed(monkeypatch, db,
               ["tag-sources", "--reviewed", "--sleep", "2"],
               "cmd_tag_sources")
    assert a.reviewed is True and a.sleep == 2.0


def test_S4_fills_both_columns_from_the_info_dict(db, tmp_path, monkeypatch):
    s = song(db, tmp_path / "a", source_path=None)
    fake_metadata(monkeypatch, default=info_dict(720, (720, 1080)))
    cli.cmd_tag_sources(tag_args(), db)
    r = row(db, s)
    assert (r["source_height"], r["source_max_height"]) == (720, 1080)


def test_S4_the_file_on_disk_wins_over_the_info_dict(db, tmp_path,
                                                    monkeypatch):
    # The dict says what YouTube would hand over today. The file says what
    # we actually have, and only the file can be re-encoded.
    s = song(db, tmp_path / "a")
    fake_metadata(monkeypatch, default=info_dict(1080, (1080, 2160)))
    fake_probe(monkeypatch, height=720)
    cli.cmd_tag_sources(tag_args(), db)
    r = row(db, s)
    assert r["source_height"] == 720
    assert r["source_max_height"] == 2160


def test_S4_source_height_reads_the_video_stream(tmp_path, monkeypatch):
    f = tmp_path / "video.src.mkv"
    f.write_bytes(b"x")
    fake_probe(monkeypatch, streams=[{"codec_type": "audio", "height": 99},
                                     {"codec_type": "video", "height": 1440}])
    assert enc.source_height(f) == 1440
    fake_probe(monkeypatch, streams=[{"codec_type": "audio"}])
    assert enc.source_height(f) is None
    fake_probe(monkeypatch, streams=[{"codec_type": "video", "height": 0}])
    assert enc.source_height(f) is None


def test_S4_a_missing_file_falls_back_to_the_dict(db, tmp_path, monkeypatch):
    s = song(db, tmp_path / "a", source_path=str(tmp_path / "gone.mkv"))
    fake_metadata(monkeypatch, default=info_dict(1080, (1080, 2160)))
    cli.cmd_tag_sources(tag_args(), db)
    assert row(db, s)["source_height"] == 1080


def test_S4_rows_that_already_know_both_are_not_asked_about(db, tmp_path,
                                                            monkeypatch):
    song(db, tmp_path / "done", source_height=720, source_max_height=1080)
    s = song(db, tmp_path / "todo", source_path=None)
    asked = fake_metadata(monkeypatch, default=info_dict())
    cli.cmd_tag_sources(tag_args(), db)
    assert len(asked) == 1
    assert row(db, s)["source_max_height"] == 2160


def test_S4_a_song_with_no_video_is_not_asked_about(db, tmp_path,
                                                    monkeypatch):
    song(db, tmp_path / "a", video_id=None, match_status="failed",
         download_status="pending", source_path=None)
    asked = fake_metadata(monkeypatch, default=info_dict())
    cli.cmd_tag_sources(tag_args(), db)
    assert asked == []


def test_S4_reviewed_limits_it_to_approved_songs(db, tmp_path, monkeypatch):
    song(db, tmp_path / "a", review="keep", video_id="aaaaaaaaaaa",
         source_path=None)
    song(db, tmp_path / "b", review=None, video_id="bbbbbbbbbbb",
         source_path=None)
    asked = fake_metadata(monkeypatch, default=info_dict())
    cli.cmd_tag_sources(tag_args(reviewed=True), db)
    assert asked == ["aaaaaaaaaaa"]


def test_S4_sleep_paces_the_requests_but_not_the_first(db, tmp_path,
                                                       monkeypatch):
    for name in ("a", "b", "c"):
        song(db, tmp_path / name, source_path=None)
    fake_metadata(monkeypatch, default=info_dict())
    naps = []
    monkeypatch.setattr(cli.time, "sleep", naps.append)
    cli.cmd_tag_sources(tag_args(sleep=1.5), db)
    assert naps == [1.5, 1.5]


def test_S4_a_failed_lookup_leaves_the_columns_empty(db, tmp_path,
                                                     monkeypatch):
    s = song(db, tmp_path / "a", source_path=None)
    fake_metadata(monkeypatch, default={})
    cli.cmd_tag_sources(tag_args(), db)
    r = row(db, s)
    assert r["source_height"] is None and r["source_max_height"] is None


def test_S4_nothing_but_the_two_columns_is_written(db, tmp_path, monkeypatch):
    s = song(db, tmp_path / "a", review="keep")
    before = dict(row(db, s))
    fake_metadata(monkeypatch, default=info_dict(1080, (1080, 2160)))
    fake_probe(monkeypatch, height=1080)
    cli.cmd_tag_sources(tag_args(), db)
    after = dict(row(db, s))
    changed = {k for k in before if before[k] != after[k]}
    assert changed <= {"source_height", "source_max_height", "updated_at"}
    assert after["review"] == "keep" and after["encode_status"] == "pending"


# ------------------------------------------------------------- S5 quality ----

@pytest.mark.parametrize("quality,height", [
    (None, 1080), ("good", 1080), ("better", 1080),
    ("best", 2160), ("super", 2160),
])
def test_S5_the_upper_tiers_raise_the_download_ceiling(quality, height):
    assert cli.download_height(download_args(quality=quality)) == height


def test_S5_a_typed_height_overrides_the_tier():
    assert cli.download_height(
        download_args(quality="super", height=720)) == 720


def test_S5_download_takes_quality_and_upgrade_and_no_height_by_default(
        db, monkeypatch):
    a = parsed(monkeypatch, db, ["download"], "cmd_download")
    assert a.height is None and a.quality is None and a.upgrade is False
    a = parsed(monkeypatch, db,
               ["download", "--quality", "super", "--upgrade"],
               "cmd_download")
    assert a.quality == "super" and a.upgrade is True
    with pytest.raises(SystemExit):
        cli.main(["--db", str(db.path), "download", "--quality", "ultra"])


def test_S5_cmd_download_passes_the_resolved_ceiling(db, tmp_path,
                                                     monkeypatch):
    song(db, tmp_path / "a", download_status="pending", source_path=None)
    seen = {}

    def fake(video_id, dest, max_height=1080, cookies=None, sleep=0.0,
             keep_existing=False):
        seen["height"] = max_height
        return Path(dest).parent / "video.src.mkv", "", info_dict()

    monkeypatch.setattr(mt, "download_video", fake)
    cli.cmd_download(download_args(quality="best"), db)
    assert seen["height"] == 2160


def test_S5_a_download_records_both_sizes(db, tmp_path, monkeypatch):
    s = song(db, tmp_path / "a", download_status="pending", source_path=None)
    monkeypatch.setattr(mt, "download_video", lambda *a, **k: (
        tmp_path / "a" / "video.src.mkv", "", info_dict(1080, (720, 2160))))
    cli.cmd_download(download_args(), db)
    r = row(db, s)
    assert (r["source_height"], r["source_max_height"]) == (1080, 2160)
    assert r["download_status"] == "ok"


def test_S5_upgrade_picks_only_the_songs_that_would_gain(db, tmp_path):
    small = song(db, tmp_path / "small", source_height=720,
                 source_max_height=2160)
    equal = song(db, tmp_path / "equal", source_height=1080,
                 source_max_height=1080)
    # Bigger exists, but not within a 1080 ceiling - nothing to gain today.
    capped = song(db, tmp_path / "capped", source_height=1080,
                  source_max_height=2160)
    unknown = song(db, tmp_path / "unknown")
    got = [Path(r["song_dir"]) for r in cli.upgrade_rows(db, 1080)]
    assert got == [small]
    assert equal not in got and capped not in got and unknown not in got
    # Raise the ceiling and the capped one is worth fetching after all.
    got = [Path(r["song_dir"]) for r in cli.upgrade_rows(db, 2160)]
    assert sorted(got) == sorted([small, capped])


def test_S5_an_upgrade_keeps_the_timing_and_the_approval(db, tmp_path,
                                                         monkeypatch):
    s = song(db, tmp_path / "a", source_height=720, source_max_height=2160,
             review="keep", encode_status="ok", ini_status="ok")
    seen = {}

    def fake(video_id, dest, max_height=1080, cookies=None, sleep=0.0,
             keep_existing=False):
        seen["keep_existing"] = keep_existing
        seen["video_id"] = video_id
        return Path(dest).parent / "video.src.mkv", "", info_dict(
            2160, (720, 2160))

    monkeypatch.setattr(mt, "download_video", fake)
    cli.cmd_download(download_args(upgrade=True, quality="super"), db)
    r = row(db, s)
    assert seen["keep_existing"] is True
    assert seen["video_id"] == "aaaaaaaaaaa"
    assert r["video_id"] == "aaaaaaaaaaa"
    assert r["offset_ms"] == 1234.0
    assert r["sync_status"] == "ok"
    assert r["review"] == "keep"
    assert r["source_height"] == 2160
    assert r["encode_status"] == "pending"


def test_S5_a_failed_upgrade_leaves_the_song_as_it_was(db, tmp_path,
                                                       monkeypatch):
    s = song(db, tmp_path / "a", source_height=720, source_max_height=2160,
             review="keep")
    before = dict(row(db, s))
    monkeypatch.setattr(mt, "download_video",
                        lambda *a, **k: (None, "yt-dlp said no", {}))
    cli.cmd_download(download_args(upgrade=True), db)
    r = row(db, s)
    assert r["download_status"] == "ok"
    assert r["source_path"] == before["source_path"]
    assert r["source_height"] == 720
    assert r["review"] == "keep"
    assert r["download_note"] == "yt-dlp said no"


def test_S5_upgrade_does_not_queue_the_ordinary_pending_downloads(
        db, tmp_path, monkeypatch):
    song(db, tmp_path / "new", download_status="pending", source_path=None)
    song(db, tmp_path / "up", source_height=720, source_max_height=2160)
    asked = []

    def fake(video_id, dest, max_height=1080, cookies=None, sleep=0.0,
             keep_existing=False):
        asked.append(Path(dest).parent.name)
        return Path(dest).parent / "video.src.mkv", "", info_dict()

    monkeypatch.setattr(mt, "download_video", fake)
    cli.cmd_download(download_args(upgrade=True), db)
    assert asked == ["up"]


# ----------------------------------------------------------- S6 reporting ----

def test_S6_the_line_counts_the_sources_above_1080(db, tmp_path):
    a = song(db, tmp_path / "a", source_height=2160)
    b = song(db, tmp_path / "b", source_height=1080)
    c = song(db, tmp_path / "c", source_height=720)
    line = cli.source_size_line(rows_of(db, a, b, c))
    assert "1 of 3" in line and "1080" in line
    assert "tag-sources" not in line


def test_S6_it_says_how_to_close_a_gap_it_cannot_report(db, tmp_path):
    a = song(db, tmp_path / "a", source_height=2160)
    b = song(db, tmp_path / "b")
    line = cli.source_size_line(rows_of(db, a, b))
    assert "1 of 2" in line
    assert "tag-sources" in line


def test_S6_no_rows_is_no_line(db):
    assert cli.source_size_line([]) is None


def test_S6_estimate_prints_it_for_the_songs_in_the_run(db, tmp_path,
                                                        capsys):
    a = song(db, tmp_path / "a", source_height=2160)
    song(db, tmp_path / "b", source_height=720)
    cli.cmd_estimate(estimate_args(), db)
    out = capsys.readouterr().out
    assert cli.source_size_line(rows_of(db, a, tmp_path / "b")) in out


def test_S6_videos_prints_it_for_the_approved_songs(db, tmp_path, capsys,
                                                    monkeypatch):
    monkeypatch.setattr(enc, "find_output", lambda d: None)
    a = song(db, tmp_path / "a", source_height=2160, review="keep")
    song(db, tmp_path / "b", source_height=720)      # not approved
    cli.cmd_videos(SimpleNamespace(lengths=False, mark=False, out=None,
                                   quiet=True), db)
    out = capsys.readouterr().out
    assert cli.source_size_line(rows_of(db, a)) in out


# -------------------------------------------------------- S7 keep-source -----

def test_S7_encode_takes_keep_source_and_it_is_off_by_default(db,
                                                              monkeypatch):
    a = parsed(monkeypatch, db, ["encode"], "cmd_encode")
    assert a.keep_source is False
    a = parsed(monkeypatch, db, ["encode", "--keep-source"], "cmd_encode")
    assert a.keep_source is True


def test_S7_a_kept_source_run_is_full_quality(db, tmp_path, pool):
    song(db, tmp_path / "a")
    cli.cmd_encode(encode_args(keep_source=True, quality="best"), db)
    call = pool.calls[0]
    assert call["kw"]["keep_source"] is True
    # Not the preview row: full height, and the tier's own ceiling.
    assert call["settings"].height == 1080
    assert (call["settings"].crf, call["settings"].bitrate_cap) == (18, "8M")


def test_S7_a_kept_source_leaves_the_song_to_encode(db, tmp_path, pool):
    # The point of keeping it is that another tier can be encoded from it,
    # and that the real run still has the song on its list.
    s = song(db, tmp_path / "a")
    cli.cmd_encode(encode_args(keep_source=True), db)
    r = row(db, s)
    assert r["encode_status"] == "pending"
    assert r["ini_status"] == "ok"      # the timing was still written
    assert "video_start_time = 1234" in (
        (tmp_path / "a" / "song.ini").read_text(encoding="utf-8"))


def test_S7_a_failed_kept_source_run_is_not_a_failed_encode(db, tmp_path,
                                                            pool):
    s = song(db, tmp_path / "a")
    pool.failing.add("a")
    cli.cmd_encode(encode_args(keep_source=True), db)
    assert row(db, s)["encode_status"] == "pending"
    assert row(db, s)["encode_note"] is None


def test_S7_the_ordinary_run_still_deletes_and_completes(db, tmp_path, pool):
    s = song(db, tmp_path / "a")
    cli.cmd_encode(encode_args(), db)
    assert pool.calls[0]["kw"]["keep_source"] is False
    assert row(db, s)["encode_status"] == "ok"


def test_S7_a_preview_still_keeps_its_source(db, tmp_path, pool):
    s = song(db, tmp_path / "a")
    cli.cmd_encode(encode_args(preview=True), db)
    assert pool.calls[0]["kw"]["keep_source"] is True
    assert pool.calls[0]["settings"].height == 480
    assert row(db, s)["encode_status"] == "pending"


def test_S5_a_new_pick_drops_the_old_video_s_sizes(db, tmp_path):
    # `set` and a re-match send every later stage back to pending. The
    # sizes describe the file that was downloaded, and the new pick is a
    # different video - kept, they would be read as its.
    s = song(db, tmp_path / "a", source_height=720, source_max_height=2160)
    cli._requeue_after_match(db, s, row(db, s))
    r = row(db, s)
    assert r["source_height"] is None and r["source_max_height"] is None


def test_S5_resetting_the_download_drops_them_too(db, tmp_path):
    s = song(db, tmp_path / "a", source_height=720, source_max_height=2160,
             download_status="failed")
    db.reset("download")
    r = row(db, s)
    assert r["source_height"] is None and r["source_max_height"] is None
    assert r["source_path"] is None
