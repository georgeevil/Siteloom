"""Reading operator input, with the refusal attached to the value.

The console's JSON endpoints are hand-written rather than declared as
pydantic request models, and for a reason: they patch a live config
object field by field, so "the fields present in this body" *is* the
request, and a request model would have to invent values for the rest.
What that shape gives up is pydantic's other half — the part that says
no. These helpers are that half, in one place instead of scattered
across handlers, because a threshold is a threshold wherever it is read.

Three rules run through all of them:

* **Nothing is coerced into validity.** `bool("false")` is True and
  `float(True)` is 1.0; both are how a typo silently becomes a setting.
  A value of the wrong type is refused, never converted. A number
  written as text is still a number, so `"0.4"` is read — its meaning
  does not change on the way in.
* **Every refusal names the field, the rule, and what arrived.** A 400 an
  operator can act on beats a traceback they can only forward. Compare
  `/identities/{id}/split`, which names the ids it will not take rather
  than dropping them.
* **A value that is out of range is malformed.** A cosine similarity of
  42 parses as a float and means nothing; accepting it writes a
  threshold no match can ever clear, into a YAML file, silently.

Parsing is kept separate from applying for the same reason: a body whose
ninth field is wrong must not leave the first eight applied. Handlers
read every value through these helpers first, and mutate afterwards.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping

from fastapi import HTTPException, Request


#: Long values are truncated in messages: the point is to let the caller
#: recognise what it sent, not to echo a payload back at it.
_SHOWN = 60


def _show(value: object) -> str:
    text = repr(value)
    return text if len(text) <= _SHOWN else text[: _SHOWN - 3] + "..."


def _refuse(field: str, rule: str, value: object) -> HTTPException:
    return HTTPException(400, f"{field} must be {rule}, got {_show(value)}")


async def json_object(request: Request) -> dict[str, Any]:
    """The request's JSON body as an object, or a 400.

    An unparseable body and a JSON array are the same mistake from the
    handler's point of view — neither can be read field by field — and
    both used to arrive as a 500.
    """
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(
            400, "request body must be JSON (Content-Type: application/json)"
        ) from None
    return as_object(body, "request body")


def as_object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _refuse(field, "a JSON object", value)
    return value


def as_list(value: object, field: str) -> list[Any]:
    # str is iterable, which is exactly the trap: iterating "person"
    # yields seven one-character class names.
    if not isinstance(value, list):
        raise _refuse(field, "a JSON array", value)
    return value


def as_bool(value: object, field: str) -> bool:
    """A JSON boolean. `bool("false")` is True — never that."""
    if not isinstance(value, bool):
        raise _refuse(field, "true or false", value)
    return value


def as_number(
    value: object,
    field: str,
    *,
    rule: str,
    integer: bool = False,
    low: float | None = None,
    high: float | None = None,
) -> Any:
    """One number, of the stated kind and within the stated range.

    Booleans are rejected before anything else: `float(True)` is 1.0, so
    a checkbox posted into a numeric field would arrive as a plausible
    value rather than as the mistake it is. NaN and the infinities are
    rejected for the same reason — they survive every comparison a
    threshold is later used in, and pass none of them.
    """
    if isinstance(value, bool) or isinstance(value, (list, dict)) or value is None:
        raise _refuse(field, rule, value)
    if integer and isinstance(value, int):
        # Taken exactly: a large integer must not lose its last digits to
        # a float round trip on the way to a field that counts things.
        number: float = value
    else:
        try:
            real = float(value)
        except (TypeError, ValueError, OverflowError):
            raise _refuse(field, rule, value) from None
        # Before anything else does arithmetic on it: `int(inf)` raises,
        # and a NaN passes no bound check below by failing all of them.
        if math.isnan(real) or math.isinf(real):
            raise _refuse(field, rule, value)
        number = real
        if integer:
            number = int(real)
            if number != real:
                raise _refuse(field, rule, value)
    if low is not None and number < low:
        raise _refuse(field, rule, value)
    if high is not None and number > high:
        raise _refuse(field, rule, value)
    return number


def as_int(
    value: object, field: str, *, low: int | None = None, high: int | None = None
) -> int:
    bounds = ""
    if low is not None and high is not None:
        bounds = f" between {low} and {high}"
    elif low is not None:
        bounds = f" of at least {low}"
    elif high is not None:
        bounds = f" of at most {high}"
    return int(
        as_number(
            value,
            field,
            rule=f"a whole number{bounds}",
            integer=True,
            low=low,
            high=high,
        )
    )


def as_float(
    value: object, field: str, *, low: float | None = None, high: float | None = None
) -> float:
    bounds = ""
    if low is not None and high is not None:
        bounds = f" between {low} and {high}"
    elif low is not None:
        bounds = f" of at least {low}"
    elif high is not None:
        bounds = f" of at most {high}"
    return float(as_number(value, field, rule=f"a number{bounds}", low=low, high=high))


def as_similarity(value: object, field: str) -> float:
    """A cosine similarity — the scale every threshold here is on.

    Each identifier's cutoff sits on its own algorithm's distribution
    (face ≈0.36, generic ≈0.8+), but all of them are cosine similarities,
    so 0..1 is the one bound that holds for every one of them.
    """
    return float(
        as_number(value, field, rule="a cosine similarity in 0..1", low=0.0, high=1.0)
    )


def as_confidence(value: object, field: str) -> float:
    return float(
        as_number(value, field, rule="a confidence in 0..1", low=0.0, high=1.0)
    )


def as_name(value: object, field: str) -> str:
    """A non-empty label: a class name, an identifier key, a tag."""
    if not isinstance(value, str) or not value.strip():
        raise _refuse(field, "a non-empty name", value)
    return value.strip()


def as_names(value: object, field: str) -> list[str]:
    """A list of non-empty names, de-duplicated, order preserved."""
    return list(
        dict.fromkeys(
            as_name(item, f"every entry of {field}") for item in as_list(value, field)
        )
    )


def as_row_id(value: object, field: str) -> int:
    """A database row id, as an integer or as digits.

    Browsers post numbers as text often enough that refusing `"12"` would
    be pedantry; refusing `"12abc"` is not — see `split_identity`, which
    names bad tokens rather than splitting off a subset and reporting
    success.
    """
    if isinstance(value, bool) or value is None:
        raise _refuse(field, "an integer id", value)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # int() is the check as well as the conversion — hand-rolling one
        # from `isdigit()` accepts "²" and "--5", which int() then throws
        # on, which is the 500 this function exists to prevent.
        try:
            return int(value.strip())
        except ValueError:
            raise _refuse(field, "an integer id", value) from None
    raise _refuse(field, "an integer id", value)


def one_of(value: object, field: str, allowed: Iterable[str]) -> str:
    choices = list(allowed)
    if not isinstance(value, str) or value not in choices:
        raise _refuse(field, "one of " + ", ".join(choices), value)
    return value


def only_keys(body: Mapping[str, Any], allowed: Iterable[str], what: str) -> None:
    """Refuse a field this endpoint does not write.

    A key nobody reads is not harmless: the caller believes it set
    something, and the endpoint returns 200 having set nothing. Naming
    what *is* editable turns that into one round trip.
    """
    choices = list(allowed)
    unknown = [key for key in body if key not in choices]
    if unknown:
        raise HTTPException(
            400,
            f"unknown {what}: {', '.join(sorted(unknown))} — "
            f"this endpoint accepts {', '.join(choices)}",
        )
