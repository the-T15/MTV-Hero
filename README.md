# MTV Hero

Finds, syncs and encodes a background music video for every song in a
YARG/Clone Hero library.

---

## What it does

For each song in your library:

1. **Match** — searches for candidate videos, downloads each candidate's audio
   only, and fingerprints it against your chart. The audio decides which video
   is correct; titles only decide what order to try them in.
2. **Download** — fetches the winner, video and audio together.
3. **Sync** — measures the offset between the video and your chart, then checks
   that offset holds across the whole song rather than at one point.
4. **Review** — a desktop application that plays a proof clip with the offset
   already applied, ordered so the doubtful songs come first.
5. **Encode** — transcodes to VP8/WebM and writes `video_start_time` into
   `song.ini`.

Every stage is resumable. Progress lives in SQLite, so a run can be stopped and
restarted without losing work.

---

## Requirements

Installed for you by pip:

- yt-dlp, numpy, scipy

Installed by you:

- Python 3.9 or newer
- **ffmpeg**, built with libvpx (`winget install Gyan.FFmpeg`)
- **a JavaScript runtime**, for YouTube's challenge solving
  (`winget install DenoLand.Deno`)

```
pip install -e .              # the tool, plus yt-dlp, numpy and scipy
pip install -e ".[desktop]"   # the same, plus PySide6 for the review app
yargvid doctor                # checks ffmpeg, ffprobe, libvpx and yt-dlp
```

`doctor` checks those four individually, including whether your ffmpeg actually
has libvpx — the "essentials" builds often do not. It does not check your
Python version or your JavaScript runtime.

If `doctor` reports yt-dlp missing straight after a successful install, pip's
scripts directory is probably not on your PATH.

---

## yt-dlp and YouTube

Video comes from YouTube via yt-dlp, and YouTube's terms of service prohibit
downloading with third-party tools.

yt-dlp is not bundled here. It is invoked as a separate program, the same as
ffmpeg, and `--sleep` leaves a second between requests by default.

`--cookies` hands your browser's YouTube session to yt-dlp. It gets past
age-gated and challenge-protected videos, and it also attaches the downloads to
your account rather than to an anonymous request.

**When it stops working, update yt-dlp first.** YouTube changes how it serves
video and yt-dlp adapts; the lag between the two is a common cause of
failure here, and it has nothing to do with this code.

    pip install -U yt-dlp

---

## Usage

```
yargvid index "D:\path\to\YARG\Songs"

yargvid match
yargvid download
yargvid sync
yargvid review              # check the results
yargvid encode --reviewed
yargvid ini
```

Every command takes `--limit N`, and `--sample` draws a random selection rather
than the alphabetically first — song folders cluster by source, so a sequential
batch measures one corner of a library.

Start with a sample. `yargvid --limit 100 --sample match` tells you your match
rate, your static-background rate and your failure modes before committing
hours to the whole library.

### Diagnostics

| command | question it answers |
|---|---|
| `status` | how far along is each stage |
| `candidates <song>` | what else was considered, and why did this win |
| `offsets <song>` | what alignments exist, and how strong is each |
| `blocks <song>` | does this video have internal cuts, and where |
| `inspect <song>` | where does this offset come from |
| `videos` | which folders already contain a video |
| `export` | every measurement for every song, as CSV |

### Fixing individual songs

```
yargvid check <song> <url>     # test a video before committing to it
yargvid set <song> <url>       # use this video instead
yargvid offset <song> <ms>     # set the offset by hand and lock it
```

---

## The review application

```
yargvid review
```

Three lists: songs worth watching, songs whose background is a still image, and
songs saved for later. Each song plays a 36-second clip — three 12-second
segments from across the track — with the offset already applied by ffmpeg, so
what you see is what YARG will show.

The ordering matters more than the player. Reviewing 1,500 songs one by one is
not realistic, so everything the pipeline knows about its own uncertainty is
combined into a single rank: a weak fingerprint match, an unfamiliar channel,
an offset that could not be verified, a background that never moves. The songs
most likely to be wrong appear first, and the rest can be spot-checked.

Buttons: keep, replace with a link, defer, or drop the video entirely.

`--browser` serves the same interface over HTTP for machines without Qt.

---

## Notes on the less obvious decisions

**Audio decides, titles rank.** A lyric video and the official video carry
identical audio and fingerprint identically. Only the title separates them, so
titles order the candidates while the audio decides which are admissible at
all. Terms describing the chart itself are exempt — a remix chart is not
penalised for matching a remix.

**Sync is verified, not assumed.** An offset is measured at seven points across
the song. Agreement means it holds; disagreement means the video has internal
cuts, or is a different edit. Where verification cannot run, the result says
`unverified` rather than reporting a confident zero.

**VP8 is not VP9.** Constant quality for libvpx-VP8 needs `-crf` with `-b:v` as
a ceiling. The `-b:v 0` idiom is VP9-only; VP8 silently falls back to
256 kbit/s and ignores the requested quality.

**Motion is measured, not inferred.** Whether a background is album art or real
footage is decided by comparing frames, normalised for contrast so dark videos
are not mistaken for still images. No title reliably predicts it.

**Some videos cannot be synced at all.** A video with a skit, an extended solo
or a different edit aligns at several offsets in different stretches of the
song. YARG applies one, so only one stretch can be in time. `blocks` shows
which, and what fraction of the song each covers.

**Confidence in a single offset is unsolved.** Several statistical approaches
were tried and measured against real results; each failed. A video that aligns
continuously but in the wrong place looks identical to a correct one by every
metric available. That is why the review step exists.

---

## Limitations

- Only unpacked folder charts. `.sng` archives and CON packages have nowhere to
  put a video file; extract them first.
- One video file per song folder — YARG may load the wrong one otherwise.
- Songs with no music video get a static background, or none.
- Videos shorter than the chart leave the end of the song without footage.

---

## Tests

```
python test_fingerprint.py    offset accuracy, sign convention, rejection margin
python test_e2e.py            the full pipeline against a synthetic song folder
```

Both need ffmpeg on PATH.

---

## License

MIT
