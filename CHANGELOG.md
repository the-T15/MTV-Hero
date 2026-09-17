# Changelog

All notable changes to yargvid (MTV Hero). Versions follow SemVer on the
`0.x` line. `1.0.0` is the first packaged build someone else can run.

## [Unreleased]

### Added
- `encode --codec {vp8,h264,h264_nvenc,h264_amf,h264_qsv}`: a codec table in
  `encode.py` carries the encoder, its quality flags, the container and the
  preview override for each row. The default stays `vp8`, the only codec
  confirmed to play in YARG on every platform.
- `encode --fps N` forces a frame rate, distinct from `--max-fps`, which only
  caps one. Given both, `--fps` wins.
- `encode --size-lock B`: target bitrate, so the size is predictable and the
  quality varies per song. Two passes on `vp8` and `h264`, which land within
  about 1% of the target; one pass on the hardware codecs, which overshoot.
  Refused together with `--crf`, which asks for the opposite.
- `yargvid estimate`: what an encode run would cost in disk space, before it
  starts. Takes every `encode` flag, so it describes the same run. Prints
  `~ N GB (max M GB)` - the maximum from the bitrate ceiling, the estimate
  from a three-song sample encoded into a temporary folder. Writes nothing.
- `doctor` reports each hardware encoder as `[ok]` or `[absent]`, having asked
  it to encode a frame rather than trusting `ffmpeg -encoders`. A missing one
  is not a missing tool; `h264_amf` and `h264_qsv` are labelled untested.

### Changed
- `encode --crf` defaults to the codec row's own number (31 for vp8, 23 for
  the H.264 rows) instead of a hardcoded 31.
- One helper names the output file. `encode`, `videos`, `--skip-existing` and
  the review app all ask `encode.find_output` which video a folder holds,
  rather than each testing for `video.webm`.
- A hardware encoder is resolved once per run, not once per song, and a
  fallback to software prints a line rather than happening quietly.

### Fixed
- `--bitrate-cap` and `--size-lock` are validated at parse time on both
  `encode` and `estimate`, so a bad value is an argparse error naming the
  flag. A bitrate is a number on its own or with `k`, `K`, `M` or `G`, and
  nothing else - no whitespace, and a zero is refused outright. ffmpeg reads
  SI prefixes, so `-b:v 4m` is four thousandths of a bit per second: it
  truncates to zero and produces byte-identical output to `-b:v 0`, the VP8
  trap NOTES has recorded since the beginning. `encode --bitrate-cap 4m`
  silently encoded the library at a sixth of the intended bitrate, and the
  sources are deleted as it goes.
- `estimate` reuses a measured bitrate only for an encode that would actually
  produce it. `encode.rate_key` now carries the effective quality number and
  the bitrate cap alongside the codec, height, frame rate and encoder, so
  `estimate --crf 40` after `estimate --crf 18` measures again instead of
  repeating the first answer. Measured 81% apart at those two numbers.
- A successful encode now deletes the other codec's output from the folder.
  YARG can select the wrong file when a folder holds two videos (YARG #1331),
  and unlike a leftover source file the other codec's output is a real,
  playable video - so switching codec looked like it had no effect.
- `encode --limit N` applied its limit in SQL, before `--skip-existing`,
  `--reviewed` and the static-background filter ran in Python, so a run could
  encode far fewer than N songs and still report a full batch. The limit is
  now applied last, to the songs that are actually going to be encoded.

## [0.5.0] — 2026-09-17

### Added
- `encode --bitrate-cap` (default `4M`) and `encode --max-fps` (default 30):
  the two settings `cmd_encode` never passed are now flags.
- `encode --include-static`: skipping album-art backgrounds is the default,
  and this turns it off.

### Changed
- Default bitrate cap `2M` -> `4M`, so at 1080p `-crf 31` governs and the cap
  is a ceiling rather than the operating point.
- `encode` writes `video_start_time` itself and sets `ini_status='ok'` in the
  same update as `encode_status='ok'`; `ini` is now the repair command for
  rows that encoded but whose `song.ini` is still pending.
- `--preview` runs through the worker pool, so `--workers` applies to it. A
  preview writes the ini but leaves `encode_status` pending.

### Fixed
- A successful encode stored `""` in `encode_note` where the schema holds
  NULL; every other stage writes NULL.
- A `song.ini` that cannot be written no longer aborts the encode run. The
  callback that writes it runs inside the worker pool's block, so a raise
  there left every queued song encoded, its source deleted and its row
  unwritten. The row now stays `encode_status='ok'` with `ini_status`
  pending, which is what `ini` repairs.

## [0.4.0] — 2026-09-11

### Added
- `yargvid bench`: re-ranks stored candidates offline per policy and scores
  each policy against approvals and hand-picked overrides; one CSV per
  policy, `--force` to overwrite.
- `match --songs <file>` and `sync --songs <file>`: run a listed set of
  song folders; a changed pick requeues download, sync, encode and ini.
- `candidates.view_count` stored from search; used as a tie-break in
  ranking.
- `REVIEW: fan MV available` match note and a `fan MV available` review
  tag when an official non-MV upload is chosen over a third-party music
  video.

### Changed
- Ranking is lexicographic: channel class (Topic < third party < official),
  then title class (MV > visualizer > lyric > audio), then view count, then
  fingerprint score. A title score can no longer outrank an official channel.
  Policy `official` ships; `artist_first` stays registered for the bench.
- Early stop keeps probing while the best candidate is an official non-MV
  and an unprobed third-party MV remains.
- Penalty terms gain "behind the curtain", "montage", `rb2`/`rb3`/`rb4`/`rbn`.

## [0.3.0] — 2026-09-10

### Added
- Review app playback: separate segment files (15 s intro, 15 s middle,
  30 s end), per-segment buttons, click-to-seek, click-to-pause, ±10 s
  skip, keyframe-snapped seeking on the full-song build, and a label when
  the video runs past the song.
- Offset as a draft: ±1/±5/±10/±100 ms step buttons rebuild the clip
  without writing; Save locks the offset the way `offset` does, Undo
  reverts.
- Four sorts (artist default, title, doubt ↓, doubt ↑); Approved and Save
  for later are most-recent-first.
- `low motion` tag (`LOW_MOTION_MAX = 0.30`) routes near-still footage to
  Unsure.

### Removed
- The typed offset box, concatenated clips and the `&t=` timestamp on
  Open source.

### Fixed
- Save follows the song out of the tile it just left.

## [0.2.0] — 2026-09-09

### Added
- Review windows at start/middle/end of the footage, a stream-copied
  full-song build with the offset applied on the audio side, `audio` and
  `short` tags, and a stored `video_seconds` per synced song.
- Eight exclusive review tiles with priority, chips under Unsure only,
  search, offset nudge, un-approve, and batch approve for Nothing Unusual.

### Fixed
- Review clips: segments stay inside the video, every rendered piece is
  measured before use, sidecars record the file rather than the request,
  a failed build clears the player, still-image songs get a clip, filter
  chips are single-select.

## [0.1.0] — 2026-09-07

### Added
- Test suite (`tests/`) pinning the state rules; `ruff` clean.
- `links` command, `manual_video` / `manual_offset` export columns,
  `doctor` check for a JavaScript runtime.

### Changed
- Sync verification fallthrough rule: a fallback candidate must carry half
  of candidate 1's hashes (`SUPPORT_RATIO = 0.5`) or candidate 1 returns as
  `unverified`.
- `export` refuses to overwrite its output without `--force`.

### Fixed
- `sync --recheck` no longer rewrites unchanged rows or keeps a `keep`
  review on a row that flipped to rejected; `fp_score` is stored from the
  candidate that passed the identity gate; `set`, `reset` and `drop` clear
  every derived column; `doctor` no longer creates an empty database;
  exact folder lookup before `LIKE`; `song.ini` written as bytes so line
  endings and BOM survive; `match` no longer aborts the batch on a
  subprocess timeout.
- Review app: stable clip cache keys, Enter routed by focus, correct
  status text and tab counts, browser server validates its path and JSON.
- Dead code and stale text removed across the package; README documents
  every subcommand and flag.

[Unreleased]: https://github.com/the-T15/MTV-Hero/compare/v0.5.0...HEAD
[0.5.0]: https://github.com/the-T15/MTV-Hero/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/the-T15/MTV-Hero/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/the-T15/MTV-Hero/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/the-T15/MTV-Hero/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/the-T15/MTV-Hero/releases/tag/v0.1.0
