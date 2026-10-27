"""Seeded, clip-stratified selection of the validation frames the cost study reruns.

The cost study measures USD and seconds per frame on a small, committed, VALIDATION-only
list of frames, so the test split is never touched by a paid call. The frames are
consecutive video frames cut from three clips, so a plain random sample could land all its
frames in one clip; the selection is therefore stratified by clip with a largest-remainder
allocation, and ordered round-robin across clips so that any prefix (the 3-frame
projection run takes the first 3) already spans every clip.
"""

from __future__ import annotations

import random
from collections import defaultdict
from collections.abc import Callable, Iterable

from object_detection_eval.data.clips import clip_key


def largest_remainder(sizes: dict[str, int], n: int) -> dict[str, int]:
    """Split ``n`` across groups in proportion to ``sizes`` (Hamilton / largest remainder).

    Ties on the remainder go to the group that sorts first by name, so the result does not
    depend on dict order.

    Raises:
        ValueError: ``n`` exceeds the number of available frames or a group is empty.
    """
    total = sum(sizes.values())
    if n > total:
        msg = f"cannot take {n} frames out of only {total} available"
        raise ValueError(msg)
    if any(size <= 0 for size in sizes.values()):
        msg = "every clip must have at least one frame"
        raise ValueError(msg)
    quotas = {clip: n * size / total for clip, size in sizes.items()}
    alloc = {clip: int(q) for clip, q in quotas.items()}
    leftover = n - sum(alloc.values())
    by_remainder = sorted(quotas, key=lambda clip: (-(quotas[clip] - alloc[clip]), clip))
    for clip in by_remainder[:leftover]:
        alloc[clip] += 1
    return alloc


def select_stratified(
    filenames: Iterable[str],
    n: int,
    seed: int,
    clip_fn: Callable[[str], str] = clip_key,
) -> list[str]:
    """Pick ``n`` frames, stratified by clip, deterministic in ``seed``.

    Within a clip the frames are drawn with ``random.Random(seed)`` from the
    filename-sorted list, clips taken in name order, so the result does not depend on the
    order of ``filenames``. The result is ordered round-robin across clips (clip-name
    order), each clip's own picks in filename order.
    """
    by_clip: dict[str, list[str]] = defaultdict(list)
    for name in sorted(set(filenames)):
        by_clip[clip_fn(name)].append(name)
    clips = sorted(by_clip)
    alloc = largest_remainder({clip: len(by_clip[clip]) for clip in clips}, n)

    rng = random.Random(seed)  # noqa: S311 - a reproducible shuffle, not a secret
    picked: dict[str, list[str]] = {
        clip: sorted(rng.sample(by_clip[clip], alloc[clip])) for clip in clips
    }

    ordered: list[str] = []
    for rank in range(max(alloc.values(), default=0)):
        for clip in clips:
            if rank < len(picked[clip]):
                ordered.append(picked[clip][rank])
    return ordered
