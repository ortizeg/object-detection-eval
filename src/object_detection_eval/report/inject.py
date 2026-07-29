"""Marker-comment table injection (REPORT-01, T-07-05).

A generated table lives between two HTML comments the generator recognizes by
exact string match. Everything outside the markers is untouched, byte-for-byte,
on every write. :func:`inject_table` replaces the interior of *exactly one*
marker pair and raises :class:`ValueError` on a zero-or-multiple match, so a
mis-authored document can never silently no-op or corrupt an unintended span of
prose.
"""

from __future__ import annotations

import re


def inject_table(doc: str, name: str, table_markdown: str) -> str:
    """Replace the interior of the ``<!-- TABLE:{name} ... -->`` marker pair.

    Args:
        doc: The full markdown document.
        name: The marker name (the ``foo`` in ``<!-- TABLE:foo START -->``).
        table_markdown: The rendered table to place between the markers.

    Returns:
        A new document with the single named marker pair's interior replaced
        by ``table_markdown``. All text outside the marker pair is preserved
        character-for-character; calling twice with the same table is
        idempotent.

    Raises:
        ValueError: If the document does not contain exactly one marker pair
            for ``name`` (zero matches or two-or-more matches).
    """
    pattern = re.compile(
        rf"<!-- TABLE:{re.escape(name)} START -->\n.*?<!-- TABLE:{re.escape(name)} END -->",
        re.DOTALL,
    )
    replacement = f"<!-- TABLE:{name} START -->\n{table_markdown}\n<!-- TABLE:{name} END -->"
    new_doc, count = pattern.subn(replacement, doc)
    if count != 1:
        msg = f"expected exactly one marker pair for {name!r}, found {count}"
        raise ValueError(msg)
    return new_doc
