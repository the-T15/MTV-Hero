# Changelog

All notable changes to yargvid (MTV Hero). Versions follow SemVer on the
`0.x` line. `1.0.0` is the first packaged build someone else can run.

## [Unreleased]

### Changed
- **The size estimates for the vp8 quality tiers are measured, not
  reasoned.** `better`, `best` and `super` were seeded as `good`'s measured
  figure scaled by the ratio of their bitrate ceilings, which is arithmetic
  and was printed with the same `[typical]` label as a measurement. All four
  are now the figures the library itself encodes at - 2.40, 3.22, 3.84 and
  4.81 Mbit/s, measured on 2026-09-18 over the same fixed 50 songs at each
  tier. The old numbers were high on every row and nearly half as much again
  on `super` (8.7 against 4.81): bits bought under a ceiling are not bits the
  encoder finds a use for. `good` moves too, from 2.9 to 2.40 - its old
  figure came from three songs drawn at random, and leaving it beside four
  fifty-song figures would have put the one unlike number in the table where
  nobody could spot it. The h264_nvenc figure was not re-measured and is
  unchanged.
- **`estimate --measure` measures the same songs every time.** It drew three
  songs at random per call, so two tiers measured on one library were measured
  on different footage: busy footage costs more bits than calm footage, and on
  the real library the draw put `better` above `best` in one round and moved
  every tier by 25-35% between rounds. The sample is now a fixed set -
  `encode.sample_songs` orders the rows by the SHA-1 of the song folder's name
  and takes the first `encode.SAMPLE_SONGS` (50) - so a tier's figure differs
  from another tier's by the setting and nothing else, a re-run repeats
  itself, and approving one more song swaps at most one member of the sample.
- Fifty songs rather than three. The sample has been 20 s a song since 0.6.0,
  so widening it costs one short encode each, and three videos is thin for a
  figure the whole library's estimate is multiplied by.

### Added
- `estimate --sample-size N`, how many songs a measurement encodes a slice of.
  `estimate`'s alone; `encode` does not take it.
- A measurement now reports itself: the measured rate in Mbit/s, how long the
  sample took, and how long the whole run would take at that speed - the
  sample's wall time scaled by seconds of video in the run over seconds of
  video in the sample. Printed only when something was actually measured, so a
  typical or remembered figure says nothing about time it cannot know.

## [0.7.0] — 2026-09-18

### Added
- `encode --quality {good,better,best,super}`, and the same flag on
  `estimate`. A tier is one word for two numbers - the quality number and the
  bitrate ceiling - because moving one without the other buys nothing: vp8
  goes 31/`4M`, 24/`6M`, 18/`8M`, 12/`12M`, and the H.264 family 23/`4M`,
  20/`6M`, 17/`8M`, 14/`12M`, the three hardware rows included. `good` is
  today's default unchanged. `--crf` and `--bitrate-cap` each override their
  own half of a tier and leave the other alone. The numbers live in exactly
  one table, `encode.QUALITY_TIERS`.
- `encode.TYPICAL_RATES` carries the vp8 tiers, so `estimate --quality best`
  answers without measuring. The H.264 tiers above `good` have no published
  figure, and `estimate` now says which settings nothing is published for and
  points at `--measure` instead of going quiet.

### Changed
- **`--height` is a ceiling, not a target: a video smaller than it is no
  longer enlarged.** A 720p download used to be blown up to 1080p, which adds
  no detail the file does not have, costs bits and roughly doubles the encode
  time - and YARG scales whatever it is handed to the screen anyway. The
  scale box is now the ceiling met against the source and the pad box is the
  16:9 box at the output height, so at `--height 1080` a 1280x720 source
  stays 1280x720, a 640x480 one is pillarboxed to 854x480 rather than
  enlarged, and 3840x2160 still comes down to 1920x1080. `--preview` follows
  the same rule.
- **The 30 fps cap is gone.** Both the YARG and the Clone Hero wikis say to
  keep the source frame rate, and 30 was a cost default rather than a
  compatibility rule; frames dropped at this stage cannot be got back.
  `--max-fps` and `EncodeSettings.max_fps` both default to no cap, so a 25
  fps source encodes at 25 and a 60 fps source at 60. `--max-fps 30` still
  caps, and 30 is still the fallback for a source whose rate cannot be read.
- `--bitrate-cap` and `--max-fps` no longer carry a value of their own as a
  default: the ceiling comes from `--quality`, and an explicit `4M` has to be
  tellable from the absence of the flag.
- `--size-lock` refuses a typed `--quality` the way it already refused
  `--crf`. A lock sets the bitrate and lets the quality fall where it may; a
  tier sets both, and under a lock `build_command` reads neither.
- `encode.rate_key` builds its probe command at `KEY_SOURCE_FPS` rather than
  at `max_fps`. With no cap as the default, `--max-fps 30` and no cap at all
  would otherwise both resolve to `fps=30.000000` and share one row of the
  `rates` table. Every row already in a database is orphaned by the new
  filter chain regardless; nothing migrates them, they simply never match.

### Fixed
- The source frame rate the encoder reads is bounded and read from the right
  field. `source_frame_rate` asks `avg_frame_rate` first and falls back to
  `r_frame_rate`, and believes neither above `MAX_SOURCE_FPS` (120): on the
  mkv yt-dlp writes, `r_frame_rate` can be the container's `1000/1` time base
  rather than a rate, and since the 30 fps cap went away that number would
  have been the encode rate. An unreadable rate falls back to 30, or to the
  cap when one is set. The bound is on what a file claims; `--fps` and
  `--max-fps` are unaffected.

## [0.6.1] — 2026-09-17

### Added
- `estimate --measure`: encode a short sample at these settings and use the
  rate it measures. Without it, `estimate` answers from what is already
  known - this process, then this database, then the typical figures in
  `encode.TYPICAL_RATES` - and only measures when all three are silent. The
  printed line ends `[measured]` or `[typical]`, because those are two
  different claims and the number cannot tell you which one you are reading.

### Changed
- An `estimate` measurement encodes 20 seconds out of the middle of each
  sampled song instead of three songs end to end, and submits every slice in
  one pool call. The figure wanted is per second, so encoding whole songs to
  find it meant paying for the run in order to predict it.
- A measured rate is remembered in the database, in a `rates` table keyed
  exactly as `encode.rate_key`, so a setting is measured once per library
  rather than once per process.
- `encode.rate_key` is derived from `build_command` instead of being a
  hand-written tuple: the command line with the input, the output,
  `-threads`, the clip, the pass log and ffmpeg's constant preamble struck
  out. Every setting that reaches the command now reaches the key, so
  `--cpu-used` and the codec row's preset select their own measurement —
  `-cpu-used 0` against `5` measured 44% apart under one key before this —
  and a flag added later cannot be forgotten. The `rates` table holds the key as one text
  column for the same reason; a table in the previous six-column shape is
  dropped on open, since its rows were not keyed on everything they measured.
- `encode.EncodeSettings.clip` encodes a slice of the source: `-ss` before
  the input, `-t` after it, nothing else changed. A job given to
  `encode_many` may carry its own settings as `(src, song_dir, settings)`.
- `--codec h264` is confirmed: an mp4 from it plays in both YARG and Clone
  Hero on Windows. VP8 stays the default, now for the Linux and Steam Deck
  reason alone rather than because H.264 was unproven.

## [0.6.0] — 2026-09-17

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

[Unreleased]: https://github.com/the-T15/MTV-Hero/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/the-T15/MTV-Hero/compare/v0.6.1...v0.7.0
[0.6.1]: https://github.com/the-T15/MTV-Hero/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/the-T15/MTV-Hero/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/the-T15/MTV-Hero/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/the-T15/MTV-Hero/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/the-T15/MTV-Hero/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/the-T15/MTV-Hero/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/the-T15/MTV-Hero/releases/tag/v0.1.0
