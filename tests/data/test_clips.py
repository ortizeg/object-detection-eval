"""Clip/game identity parsing for the Roboflow frame-per-file naming scheme.

These names are the only evidence in the repo that the images are consecutive
video frames rather than independent photos, so the parse is load-bearing twice
over: the clip-clustered bootstrap resamples by it, and ``docs/dataset.md``
counts clips and games by it. The tests pin (a) that the grouping is what the
clustered bootstrap has always used, (b) that ``game_id`` is strictly coarser
than ``clip_key``, and (c) that an unparsed name isolates itself rather than
silently joining a group it does not belong to.
"""

from __future__ import annotations

from object_detection_eval.data.clips import clip_key, game_id

# Three real test-split filenames, one per clip in the 94-image test set.
_TEST_G1 = "boston-celtics-new-york-knicks-game-1-q1-07_41-07_34-0000_png.rf.2c5dcc6a.jpg"
_TEST_G4 = "boston-celtics-new-york-knicks-game-4-q1-05_06-05_01-0120_png.rf.deadbeef.jpg"
_MAGIC_G4 = "boston-celtics-orlando-magic-game-4-q1-11_44-11_36-0300_png.rf.cafebabe.jpg"


def test_clip_key_is_game_quarter_pipe_span() -> None:
    """The historical key format, unchanged by the move out of the script.

    ``bootstrap_clustered_7models.json`` records cluster identities in this
    exact shape, so a format change here would silently invalidate a committed
    results file.
    """
    assert clip_key(_TEST_G1) == "boston-celtics-new-york-knicks-game-1-q1|07_41-07_34"


def test_frames_of_one_clip_share_a_key() -> None:
    """Different frame indices of the same span are one cluster, not many."""
    first = "boston-celtics-new-york-knicks-game-1-q1-07_41-07_34-0000_png.rf.aaa.jpg"
    later = "boston-celtics-new-york-knicks-game-1-q1-07_41-07_34-0870_png.rf.bbb.jpg"
    assert clip_key(first) == clip_key(later)


def test_different_spans_in_one_quarter_are_different_clips() -> None:
    """A second segment from the same quarter is a separate clip.

    This is the distinction that makes train's 15 clips 15 and not 5: the
    quarter alone would collapse them.
    """
    span_a = "boston-celtics-new-york-knicks-game-4-q1-00_05-00_01-0000_png.rf.aaa.jpg"
    span_b = "boston-celtics-new-york-knicks-game-4-q1-05_27-05_21-0000_png.rf.bbb.jpg"
    assert clip_key(span_a) != clip_key(span_b)
    # ...but they ARE the same game, which is the point of game_id.
    assert game_id(span_a) == game_id(span_b)


def test_game_id_drops_quarter_and_span() -> None:
    assert game_id(_TEST_G1) == "boston-celtics-new-york-knicks-game-1"
    assert game_id(_TEST_G4) == "boston-celtics-new-york-knicks-game-4"
    assert game_id(_MAGIC_G4) == "boston-celtics-orlando-magic-game-4"


def test_game_id_separates_teams_at_the_same_game_number() -> None:
    """`-game-4-` appears in two different matchups; they must not merge."""
    assert game_id(_TEST_G4) != game_id(_MAGIC_G4)


def test_quarters_of_one_game_share_a_game_id_but_not_a_clip_key() -> None:
    q1 = "boston-celtics-new-york-knicks-game-1-q1-03_16-03_11-0000_png.rf.aaa.jpg"
    q2 = "boston-celtics-new-york-knicks-game-1-q2-08_43-08_38-0000_png.rf.bbb.jpg"
    assert game_id(q1) == game_id(q2)
    assert clip_key(q1) != clip_key(q2)


def test_unparsed_filename_becomes_its_own_singleton() -> None:
    """An unrecognised name isolates itself in BOTH groupings.

    Merging it into a real cluster would understate correlation; the singleton
    fallback fails toward more clusters, never fewer.
    """
    odd = "not-a-roboflow-name.jpg"
    assert clip_key(odd) == "__singleton__not-a-roboflow-name.jpg"
    assert game_id(odd) == "__singleton__not-a-roboflow-name.jpg"
    assert clip_key(odd) != clip_key("also-not-a-roboflow-name.jpg")
