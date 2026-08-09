"""Keyset paging for the console's lists (CLD-104).

`OFFSET n` is wrong for every list in this console, and not merely slow.
All of them are ordered by recency and all of them are read while ingest
is writing: each new event shifts the window, so the row that was at
offset 50 when the operator asked for page 1 is at offset 51 by the time
they ask for page 2. They see one row twice and never see another. That
is tolerable at one click per minute and indefensible under continuous
scroll, which asks far more often.

A keyset cursor asks a different question — "the rows after *this row*"
rather than "the rows after the 50th row" — so rows arriving above the
cursor cannot move anything below it.

Two details that are easy to get wrong and expensive to debug:

* **The cursor must carry every column the ORDER BY does.** Events are
  ordered `first_seen DESC, id DESC` because sampled cameras produce
  many events sharing a timestamp; a cursor on `first_seen` alone would
  drop every row that ties with the last one delivered. The comparison is
  therefore a row value — `(first_seen, id) < (:ts, :id)` — which SQLite
  (≥3.15) and Postgres both implement natively.
* **A cursor is not a filter.** It describes a position in one client's
  scroll, so it must never reach a shareable URL: a link someone sends a
  colleague has to open at the top of that filtered set, not halfway down
  wherever the sender happened to be. Filter state lives in the query
  string; the cursor rides on the load-more request only.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import tuple_

#: Separator inside the encoded cursor. Not present in an ISO timestamp
#: or a decimal id, so no component needs escaping.
_SEP = "\x1f"


class CursorError(ValueError):
    """A cursor the server will not read, with a reason worth logging."""


def encode_cursor(*values: Any) -> str:
    """Pack the last delivered row's sort key into one opaque token.

    Opaque on purpose: base64 of the joined parts rather than the parts
    themselves. It is not a security boundary — `decode_cursor` validates
    everything it produces — it is a contract boundary. A legible
    `?after_id=412` invites a caller to construct one, and then the sort
    key can never change without breaking them.
    """
    raw = _SEP.join(
        v.isoformat() if isinstance(v, datetime) else str(v) for v in values
    )
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def decode_cursor(token: str, types: tuple[type, ...]) -> tuple[Any, ...]:
    """Unpack a cursor, or raise `CursorError`.

    Every failure mode here is a caller sending something the server did
    not mint — a truncated copy-paste, a stale token from before a sort
    key changed, or someone experimenting. None of them may 500, and none
    may fall through to an unfiltered query: a cursor that silently
    becomes "no cursor" restarts the list from the top mid-scroll, which
    looks like duplicated rows rather than like an error.
    """
    padding = "=" * (-len(token) % 4)
    try:
        raw = base64.urlsafe_b64decode(token + padding).decode()
    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise CursorError(f"undecodable cursor: {exc}") from None
    parts = raw.split(_SEP)
    if len(parts) != len(types):
        raise CursorError(
            f"cursor has {len(parts)} parts, this list sorts on {len(types)}"
        )
    out = []
    for part, kind in zip(parts, types):
        try:
            out.append(datetime.fromisoformat(part) if kind is datetime else kind(part))
        except (TypeError, ValueError) as exc:
            raise CursorError(f"bad cursor component {part!r}: {exc}") from None
    return tuple(out)


def after(columns: tuple, values: tuple, descending: bool = True):
    """A clause selecting the rows strictly after `values` in sort order.

    One row-value comparison rather than the unrolled
    `a < x OR (a = x AND b < y)` chain — the chain is where a tie-break
    column gets forgotten, and it is what stops the index being used.
    """
    row = tuple_(*columns)
    return row < tuple_(*values) if descending else row > tuple_(*values)


@dataclass
class Slice:
    """One page of rows plus the cursor that continues it.

    `next_cursor` is None exactly when the list is exhausted, which is
    what lets a screen distinguish "nothing more to load" from "the last
    fetch failed" — two states that look identical if the only signal is
    an empty response.
    """

    rows: list
    next_cursor: str | None

    @property
    def exhausted(self) -> bool:
        return self.next_cursor is None


def take(rows: list, size: int, key) -> Slice:
    """Trim an over-fetched result to `size` and derive its cursor.

    Callers fetch `size + 1` rows: whether a further row exists is not
    answerable from a full page of results, and asking the database for a
    COUNT over the same filters to find out is a second query that can
    disagree with the first while ingest writes between them.
    """
    if len(rows) <= size:
        return Slice(rows, None)
    kept = rows[:size]
    return Slice(kept, encode_cursor(*key(kept[-1])))
