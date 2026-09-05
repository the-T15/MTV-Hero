"""
Resumable state.

A notebook holding results in memory is the wrong container for a job that runs
for days over thousands of folders and can fail at four independent stages. One
crash, one bad song, one closed laptop lid and the work is gone.

Every song is a row. Every stage records its own status, so any stage can be
re-run in isolation without redoing the others: re-match the rejects, re-encode
at a different CRF, rewrite the ini files after a sign-convention fix. Nothing
is ever recomputed by accident, and nothing is lost.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS songs (
    song_dir        TEXT PRIMARY KEY,
    artist          TEXT,
    title           TEXT,
    chart_seconds   REAL,

    match_status    TEXT DEFAULT 'pending',
    video_id        TEXT,
    match_score     REAL,
    match_note      TEXT,

    download_status TEXT DEFAULT 'pending',
    source_path     TEXT,

    sync_status     TEXT DEFAULT 'pending',
    offset_ms       REAL,
    spread_ms       REAL,
    drift_ppm       REAL,
    fp_score        REAL,
    sync_note       TEXT,

    encode_status   TEXT DEFAULT 'pending',
    encode_note     TEXT,

    ini_status      TEXT DEFAULT 'pending',
    updated_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_match  ON songs(match_status);
CREATE INDEX IF NOT EXISTS idx_sync   ON songs(sync_status);
CREATE INDEX IF NOT EXISTS idx_encode ON songs(encode_status);

CREATE TABLE IF NOT EXISTS candidates (
    song_dir   TEXT,
    video_id   TEXT,
    title      TEXT,
    uploader   TEXT,
    duration   REAL,
    score      REAL,
    coverage   REAL,
    PRIMARY KEY (song_dir, video_id)
);
"""

STAGES = ("match", "download", "sync", "encode", "ini")
# What must already be true for a song to be eligible for a stage at all.
#
# Shared by pending() and counts() deliberately. When counts() had its own
# unfiltered GROUP BY, `status` reported every song that failed an earlier
# stage as 'pending' for all the later ones
PREREQ = {
    "match": "1=1",
    "download": "match_status = 'ok'",
    "sync": "download_status = 'ok'",
    "encode": "sync_status IN ('ok', 'drift', 'unverified')",
    "ini": "encode_status = 'ok'",
}

class Database:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.conn = sqlite3.connect(self.path, timeout=60)
        self.conn.row_factory = sqlite3.Row
        # WAL lets the encode workers write progress while readers run.
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        # Columns added after a database may already exist in the wild.
        existing = {r[1] for r in self.conn.execute("PRAGMA table_info(songs)")}
        for col, decl in (("motion", "REAL"), ("review", "TEXT"),
                          ("dominance", "REAL"),
                          ("existing_video", "TEXT"),
                          ("download_note", "TEXT"),
                          ("windows", "INTEGER")):
            if col not in existing:
                self.conn.execute(f"ALTER TABLE songs ADD COLUMN {col} {decl}")
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def add_song(self, song_dir: Path, artist: str, title: str, seconds: float) -> None:
        """
        Insert a song, or refresh its metadata if already present.

        This must be an upsert, not INSERT OR IGNORE. Artist and title are
        parsed at index time and feed the search query, so any improvement to
        the parser is worthless if re-indexing cannot update existing rows.
        Stage statuses are deliberately untouched - re-indexing refreshes what
        a song IS, never how far it has progressed.
        """
        self.conn.execute(
            "INSERT INTO songs (song_dir, artist, title, chart_seconds) "
            "VALUES (?, ?, ?, ?) "
            "ON CONFLICT(song_dir) DO UPDATE SET "
            "  artist = excluded.artist, "
            "  title = excluded.title, "
            "  chart_seconds = excluded.chart_seconds",
            (str(song_dir), artist, title, seconds),
        )
        self.conn.commit()

    def update(self, song_dir: Path, **fields) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE songs SET {cols}, updated_at = CURRENT_TIMESTAMP "
            f"WHERE song_dir = ?",
            (*fields.values(), str(song_dir)),
        )
        self.conn.commit()

    def save_candidates(self, song_dir: Path, candidates) -> None:
        self.conn.executemany(
            "INSERT OR REPLACE INTO candidates "
            "(song_dir, video_id, title, uploader, duration, score, coverage) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (str(song_dir), c.video_id, c.title, c.uploader,
                 c.duration, c.score, c.coverage)
                for c in candidates
            ],
        )
        self.conn.commit()

    def pending(self, stage: str, limit: int | None = None,
                shuffle: bool = False) -> list[sqlite3.Row]:
        """
        Rows awaiting `stage`, with the previous stage already complete.

        `shuffle` draws a random sample instead of the alphabetically first N.
        Path order is not random order: song folders cluster by source, so a
        sequential batch measures one corner of a library and its rates do not
        generalise. For a diagnostic run, sample randomly.
        """
        prereq = PREREQ[stage]
        order = "RANDOM()" if shuffle else "song_dir"
        sql = (
            f"SELECT * FROM songs WHERE {stage}_status = 'pending' AND {prereq} "
            f"ORDER BY {order}"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.conn.execute(sql).fetchall()

    def retry_queue(self, stage: str, limit: int | None = None) -> list[sqlite3.Row]:
        """
        Rows pending for `stage` that have been ATTEMPTED before.

        After `retry match`, previously-failed songs sit at 'pending' alongside
        every song never matched at all, and the stage runner cannot tell them
        apart - so re-testing 30 failures would queue the whole library. Reset
        deliberately preserves the note column, which is the only thing marking
        a row as having been tried, so it is what distinguishes them here.
        """
        note = f"{stage}_note"
        # Both 'failed' (straight after a failed run) and 'pending' (after a
        # `retry`) mean "tried before". Requiring only 'pending' made --redo
        # silently return nothing unless retry had been run first.
        sql = (
            f"SELECT * FROM songs WHERE {stage}_status IN ('pending', 'failed') "
            f"AND {note} IS NOT NULL AND {note} != '' ORDER BY song_dir"
        )
        if limit:
            sql += f" LIMIT {int(limit)}"
        return self.conn.execute(sql).fetchall()

    def reset(self, stage: str, only_failed: bool = True) -> int:
        """
        Reset a stage to pending, and invalidate every stage after it.

        Without the cascade, `retry match` re-picks a video but leaves
        download_status at 'ok' pointing to the previously downloaded file.
        The download stage then has nothing to do, and sync, encode and ini
        all keep operating on the OLD video while the match output claims a
        new one. Every result looks plausible and none of it is current.
        """
        where = f"{stage}_status != 'pending'"
        if only_failed:
            where = f"{stage}_status NOT IN ('pending', 'ok', 'drift')"

        downstream = STAGES[STAGES.index(stage) + 1:]
        sets = [f"{stage}_status = 'pending'"]
        sets += [f"{s}_status = 'pending'" for s in downstream]

        # Clear anything derived from the stages being invalidated, so nothing
        # stale can be mistaken for a current result.
        derived = {
            "download": ["source_path"],
            "sync": ["offset_ms", "spread_ms", "drift_ppm", "sync_note",
                     "motion", "dominance"],
            "encode": ["encode_note"],
        }
        for s in (stage, *downstream):
            sets += [f"{col} = NULL" for col in derived.get(s, [])]

        cur = self.conn.execute(
            f"UPDATE songs SET {', '.join(sets)} WHERE {where}"
        )
        self.conn.commit()
        return cur.rowcount

    def counts(self) -> dict[str, dict[str, int]]:
        """
        Per-stage status tallies for `status`.

        'pending' is filtered by the stage's prerequisite so it means "still to
        do" rather than "has not happened", which for a song blocked upstream
        is never going to change. Every other status is counted unconditionally:
        a row that failed or completed is a real outcome regardless of what
        happened before it, and hiding those would understate the failures.
        """
        out: dict[str, dict[str, int]] = {}
        for stage in STAGES:
            rows = self.conn.execute(
                f"SELECT {stage}_status AS s, COUNT(*) AS n FROM songs "
                f"WHERE {stage}_status != 'pending' OR ({PREREQ[stage]}) "
                f"GROUP BY {stage}_status"
            ).fetchall()
            out[stage] = {r["s"]: r["n"] for r in rows}
        return out

    def manual_queue(self) -> list[sqlite3.Row]:
        """Everything a human needs to look at, with the reason attached."""
        return self.conn.execute(
            "SELECT song_dir, artist, title, match_status, match_note, "
            "       download_status, download_note, "
            "       sync_status, sync_note, encode_note "
            "FROM songs "
            "WHERE match_status NOT IN ('pending', 'ok') "
            "   OR match_note LIKE 'REVIEW:%' "
            "   OR download_status = 'failed' "
            "   OR sync_status NOT IN ('pending', 'ok', 'drift') "
            "   OR encode_status NOT IN ('pending', 'ok') "
            "ORDER BY song_dir"
        ).fetchall()

    def flagged_matches(self) -> list[sqlite3.Row]:
        """Songs whose background is measurably static, plus title-based flags."""
        return self.conn.execute(
            "SELECT song_dir, artist, title, match_note, motion FROM songs "
            "WHERE match_note LIKE 'REVIEW:%' "
            "   OR (motion IS NOT NULL AND motion >= 0 AND motion < 0.029) "
            "ORDER BY song_dir"
        ).fetchall()