"""Clip and game identity for the Roboflow frame-per-file naming convention.

The basketball images are **consecutive video frames**, not independent photos:
each file is one frame sampled from a short contiguous clip, and the filename
carries the clip it came from::

    <teams>-game-<N>-q<M>-<mm_ss>-<mm_ss>-<frame>_png.rf.<hash>.jpg
    \\_______________ game _______________/\\____ span ____/\\_ frame _/

Two things depend on recovering that grouping, and they must agree:

- ``scripts/run_clustered_bootstrap.py`` clusters the test frames by clip so the
  bootstrap resamples clips instead of frames (the honest CI for a 3-clip test
  set — see :mod:`object_detection_eval.metrics.bootstrap`).
- ``scripts/write_dataset_stats.py`` counts clips and games per split for
  ``docs/dataset.md``.

If those two disagreed about what a clip is, the page would document a grouping
the statistics do not use. So the parse lives here once, in importable
stdlib-only code, rather than being re-derived at each call site.
"""

from __future__ import annotations

import re

from loguru import logger

#: ``<teams>-game-N-qM-<start>-<end>-<frame>`` (the filename with the
#: ``_png.rf.<hash>.jpg`` suffix already stripped). The frame index is dropped;
#: everything before it identifies the contiguous source segment.
_CLIP_RE = re.compile(r"^(?P<game>.*?-game-\d+)-(?P<quarter>q\d+)-(?P<span>[\d_]+-[\d_]+)-\d+$")

#: Prefix marking a filename that did not parse. Such a name becomes its own
#: singleton group rather than being folded into a real one — an unparsed name
#: must never widen a cluster it does not belong to.
_SINGLETON = "__singleton__"


def _match(filename: str) -> re.Match[str] | None:
    """Strip the Roboflow ``_png.rf.<hash>`` suffix and parse what remains."""
    base = filename.split("_png.rf.")[0]
    match = _CLIP_RE.match(base)
    if match is None:
        logger.warning(f"unparsed filename, treating as its own cluster: {filename}")
    return match


def clip_key(filename: str) -> str:
    """Map an image filename to its source-clip identifier.

    Returns:
        ``"<teams>-game-N-qM|<start>-<end>"`` — game-quarter and time span
        joined by a pipe. A filename that does not match the known Roboflow
        naming pattern becomes ``"__singleton__<filename>"``, its own cluster.
    """
    match = _match(filename)
    if match is None:
        return f"{_SINGLETON}{filename}"
    return f"{match.group('game')}-{match.group('quarter')}|{match.group('span')}"


def game_id(filename: str) -> str:
    """Map an image filename to its source *game*, ignoring quarter and clip.

    Coarser than :func:`clip_key` on purpose. The splits are clip-disjoint but
    still drawn from the same handful of games, so quantifying that second,
    weaker correlation needs a grouping one level above the clip.

    Returns:
        ``"<teams>-game-N"``, or ``"__singleton__<filename>"`` if unparsed.
    """
    match = _match(filename)
    if match is None:
        return f"{_SINGLETON}{filename}"
    return match.group("game")
