"""Marker-comment injection guarantees (REPORT-01, T-07-05).

`inject_table` must replace exactly the interior of one marker pair, raise on a
zero-or-multiple match, leave surrounding prose byte-for-byte intact, and be
idempotent.
"""

from __future__ import annotations

import pytest

from object_detection_eval.report import inject_table

_DOC = "intro prose\n<!-- TABLE:foo START -->\nold table\n<!-- TABLE:foo END -->\noutro prose\n"


def test_inject_replaces_interior() -> None:
    out = inject_table(_DOC, "foo", "NEW")
    assert "NEW" in out
    assert "old table" not in out


def test_inject_surrounding_prose_byte_for_byte() -> None:
    out = inject_table(_DOC, "foo", "NEW")
    assert out == (
        "intro prose\n<!-- TABLE:foo START -->\nNEW\n<!-- TABLE:foo END -->\noutro prose\n"
    )


def test_inject_is_idempotent() -> None:
    once = inject_table(_DOC, "foo", "NEW")
    twice = inject_table(once, "foo", "NEW")
    assert once == twice


def test_inject_zero_matches_raises() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        inject_table(_DOC, "missing", "NEW")


def test_inject_multiple_matches_raises() -> None:
    dup = _DOC + _DOC
    with pytest.raises(ValueError, match="exactly one"):
        inject_table(dup, "foo", "NEW")
