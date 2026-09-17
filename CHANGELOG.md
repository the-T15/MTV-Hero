# Changelog

All notable changes to yargvid (MTV Hero). Versions follow SemVer on the
`0.x` line. `1.0.0` is the first packaged build someone else can run.

## [Unreleased]

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

[Unreleased]: https://github.com/the-T15/MTV-Hero/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/the-T15/MTV-Hero/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/the-T15/MTV-Hero/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/the-T15/MTV-Hero/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/the-T15/MTV-Hero/releases/tag/v0.1.0
