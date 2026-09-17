"""
The twelve songs the channel-first policy would move, pinned to what a
person decided.

When the bench compared the shipped rule against the one now in `pick_best`,
twelve unlabelled songs came out with a different video. Each was looked at
by hand on 2026-09-16 and sorted into one of four groups:

- two textbook moves: a stranger's "Official Music Video" upload loses to
  the label's own (Party in the U.S.A., Wherever You Will Go);
- four moves that only exist because penalty terms were added after the
  pick was stored - a making-of, a Rock Band capture, gameplay, a montage
  (Red Wine Supernova, Halo Theme, Makeshift Vehicles, Mountain Man);
- three where the title beat a much stronger fingerprint on the same
  official channel (Wonderwall, Worth A Dollar, MONTERO);
- three that are not moves at all: the same upload stored twice, and the
  bench picked the other copy (Dear Agony, Mony Mony, Battery).

Rulings: Red Wine Supernova is the Magician's Cut; Wonderwall is the
Official Video (the Remastered upload is album art); MONTERO is the Official
Video, which the search never returned; Mountain Man is the VEVO upload,
which the search returned but never probed. The last two are hand picks -
a test here shows what the policy does once the right row is in front of
it, and where it cannot get there on its own.

Fixtures are the heard rows from `yargvid candidates`, scores and coverage
as stored. `candidates` cuts titles at 52 characters; where the cut-off
tail carried a scoring term the fixture spells out which term, so the row
scores the way the stored one does.

    pytest -q tests/test_policy_moves.py
"""

from __future__ import annotations

import pytest

from yargvid import match as mt

# A title `candidates` cut off before a term that is known to be in it.
# The visible score gives the term away; the placeholder reproduces it.
GUEST_TAIL = " ft. (tail cut off by `candidates`; stored score says feat)"
ROCK_BAND_TAIL = " (rock band tail cut off by `candidates`; stored score -10)"


def cand(video_id, title, uploader, duration, score, coverage):
    return mt.Candidate(video_id=video_id, title=title, uploader=uploader,
                        duration=duration, score=score, coverage=coverage)


def pick(cands, chart_text):
    ranked = mt.rank_official(cands, chart_text)
    assert ranked, "nothing passed the gate"
    return ranked[0].video_id


# ---------------------------------------------------- textbook moves ---

def test_party_in_the_usa_official_beats_strangers_official_mv():
    chart = "Miley Cyrus Party in the U.S.A."
    cands = [
        cand("OWjMlBy8n-I", "Miley Cyrus - Party In The U.S.A - Official Music Video",
             "Bárbara Marques", 201, 466, 0.92),
        cand("enjoy-4k", "Miley Cyrus - Party In The U.S.A. [Remastered In 4K] (Official Music Video)",
             "Enjoy it🤍", 202, 449, 0.92),
        cand("M11SvDtPBhA", "Miley Cyrus - Party In The U.S.A. (Official Video)",
             "HollywoodRecordsVEVO", 208, 396, 0.92),
    ]
    assert pick(cands, chart) == "M11SvDtPBhA"


def test_wherever_you_will_go_vevo_beats_strangers_official_mv():
    chart = "The Calling Wherever You Will Go"
    cands = [
        cand("v8Ut_8P55vw", "The Calling - Wherever You Will Go Official Music Video",
             "Bad Boy Edd", 205, 170, 0.96),
        cand("iAP9AF6DCu4", "The Calling - Wherever You Will Go (Official Video)",
             "TheCallingVEVO", 207, 160, 0.92),
    ]
    assert pick(cands, chart) == "iAP9AF6DCu4"


# ------------------------------------ penalty terms added after the pick ---

def test_red_wine_supernova_magicians_cut_beats_making_of():
    chart = "Chappell Roan Red Wine Supernova"
    cands = [
        cand("VS6ixn2berk", "Chappell Roan - Red Wine Supernova (Magician's Cut)",
             "Chappell Roan", 211, 459, 0.97),
        cand("plain", "Red Wine Supernova", "Chappell Roan", 193, 404, 0.97),
        cand("oZ0w4bFD1Ec", "Chappell Roan - Red Wine Supernova (Behind The Curtain)",
             "Chappell Roan", 193, 761, 0.97),
    ]
    # The making-of has the strongest fingerprint and still loses: its title
    # is what marks it as not the video.
    assert pick(cands, chart) == "VS6ixn2berk"
    assert mt.title_score(cands[2].title, chart) == -10.0


def test_halo_theme_soundtrack_upload_beats_rock_band_capture():
    chart = "Martin O'Donnell & Michael Salvatori Halo Theme MJOLNIR Mix"
    cands = [
        cand("Qz7EhV4t4Es", "halo 2 soundtrack (Mjolnir mix) HD",
             "Games Soundtracks", 252, 423, 0.96),
        cand("ms", "Martin O'Donnell & Michael Salvatori - Halo Theme (Mjolnir Mix)",
             "ms", 252, 420, 0.96),
        cand("attila", "Martin O'Donnell, Michael Salvatori - Halo (Mjolnir Mix)",
             "Nagy B. Attila", 252, 369, 0.96),
        cand("udholm", "Halo 2 Volume 1 OST 01 Halo Theme Mjolnir Mix",
             "Oliver Udholm", 252, 230, 0.97),
        cand("UE8rjSnkV2E",
             "Halo Theme MJOLNIR Mix - Martin O'Donnell & Michael Salvatori"
             + ROCK_BAND_TAIL,
             "DJX", 270, 1928, 0.95),
    ]
    # Nobody official uploads this one. Every row is class 1, so the title
    # decides, and a gameplay capture scores -10 whatever its fingerprint.
    assert all(mt.channel_class(c.uploader, chart) == 1 for c in cands)
    assert pick(cands, chart) == "Qz7EhV4t4Es"


def test_makeshift_vehicles_label_visualizer_beats_rb4_gameplay():
    chart = "Body Thief Makeshift Vehicles"
    cands = [
        cand("BJ99mMzUVuA", "Body Thief - Makeshift Vehicles (Visualizer)",
             "riserecords", 228, 330, 0.97),
        cand("vIUw0C6i-gQ",
             "RB4 DLC: Makeshift Vehicles by Body Thief - Expert Full Band",
             "Team Opus", 250, 876, 0.98),
    ]
    assert mt.channel_class("riserecords", chart) == 2
    assert pick(cands, chart) == "BJ99mMzUVuA"


def test_mountain_man_montage_loses_but_policy_cannot_prefer_vevo():
    chart = "Crash Kings Mountain Man"
    heard = [
        cand("Qg6P09NiDwA", "Mountain Man", "The Crash Kings", 224, 96, 0.91),
        cand("txc4l5r3veA", 'Crash Kings "Mountain Man" montage',
             "The Crash Kings", 198, 2493, 0.97),
    ]
    # Among what was heard, the montage penalty moves the pick off it.
    assert pick(heard, chart) == "Qg6P09NiDwA"

    # The ruling is the VEVO upload, which was found but never probed. Even
    # once it is heard, the policy cannot prefer it: VEVO and the band's own
    # channel are one official class, both titles are plain, so the key ties
    # and the fingerprint decides. That is why this song is a `set`, not a
    # re-match.
    vevo = cand("2OvqpNP7dTI", "Crash Kings - Mountain Man",
                "CrashKingsVEVO", 198, 300, 0.95)
    assert (mt.policy_key(vevo, chart)
            == mt.policy_key(heard[0], chart))


# ---------------------- title outranks a stronger fingerprint, same channel ---

def test_wonderwall_official_video_beats_remastered_album_art():
    chart = "Oasis Wonderwall"
    cands = [
        cand("bx1Bh8ZvH84", "Oasis - Wonderwall (Official Video)", "Oasis", 278, 56, 0.94),
        cand("ov2", "Oasis - Wonderwall (Official Video)", "Oasis", 280, 55, 0.96),
        cand("jp", "【日本語訳】オアシス – ワンダーウォール / Oasis – Wonderwall (Official Video)",
             "Oasis", 278, 52, 0.95),
        cand("FVdjZYfDuLE", "Wonderwall (Remastered)", "Oasis", 259, 456, 0.97),
    ]
    # Ruled correct: the Remastered upload is album art. The fingerprint gap
    # (456 against 56) is real and the title still decides, on purpose.
    assert pick(cands, chart) == "bx1Bh8ZvH84"


def test_worth_a_dollar_full_length_upload_beats_short_guest_cut():
    chart = ("Queens of the Stone Age You Think I Ain't Worth A Dollar, "
             "But I Feel Like A Millionaire")
    cands = [
        cand("UC5XbyV4OFE",
             "You Think I Ain't Worth A Dollar, But I Feel Like A Millionaire",
             "Queens Of The Stone Age", 194, 93, 0.94),
        cand("ESxRPlhLZNA",
             "You Think I Ain't Worth A Dollar, But I Feel Like A Millionaire"
             + GUEST_TAIL,
             "Queens Of The Stone Age", 157, 252, 0.94),
    ]
    # The stored pick runs 157 s against a 190 s song; the guest-credit
    # penalty is what moves it, and the move is right.
    assert mt.title_score(cands[1].title, chart) == -8.0
    assert pick(cands, chart) == "UC5XbyV4OFE"


def test_montero_official_video_wins_once_it_is_a_candidate():
    chart = "Lil Nas X MONTERO (Call Me by Your Name)"
    heard = [
        cand("NIlz3g3DgCU",
             "Lil Nas X - MONTERO (Call Me By Your Name) [SATAN'S EXTENDED VERSION]",
             "Lil Nas X", 171, 233, 0.94),
        cand("YspVHSxhncI",
             "Lil Nas X - MONTERO (Call Me By Your Name) (But Lil Nas X is"
             + GUEST_TAIL,
             "Lil Nas X", 139, 525, 0.92),
    ]
    # Among what the search returned, the extended version wins. Neither is
    # the ruling: the official video was never a candidate, which is a
    # search miss, not a ranking one.
    assert pick(heard, chart) == "NIlz3g3DgCU"

    official = cand("6swmTBVI83k",
                    "Lil Nas X - MONTERO (Call Me By Your Name) (Official Video)",
                    "Lil Nas X", 190, 400, 0.9)
    assert pick(heard + [official], chart) == "6swmTBVI83k"


# ------------------------------------------------- not moves: duplicates ---

@pytest.mark.parametrize("chart, uploader, title, current, other, score", [
    ("Breaking Benjamin Dear Agony", "Breaking Benjamin", "Dear Agony",
     "daSX_3hadlA", "6ve9KYdslFo", 1174),
    ("Billy Idol Mony Mony", "Billy Idol", "Mony Mony",
     "VPwMsu63PUU", "QkvE9yV5to4", 61),
    ("Metallica Battery", "Metallica", "Battery (Remastered)",
     "a_ITtP033RI", "RvW4OQFA_UY", 66),
])
def test_duplicate_rows_tie_on_every_key(chart, uploader, title, current,
                                         other, score):
    a = cand(current, title, uploader, 259, score, 0.96)
    b = cand(other, title, uploader, 259, score, 0.96)
    # Identical key and identical fingerprint: the policy has no opinion
    # between them, so whichever it returns is not a move. The bench counted
    # these three as `differs`; that is the finding, not a wrong pick.
    assert mt.policy_key(a, chart) == mt.policy_key(b, chart)
    assert a.score == b.score
    assert pick([a, b], chart) in {current, other}
