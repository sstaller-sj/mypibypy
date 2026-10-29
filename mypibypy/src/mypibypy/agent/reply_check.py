"""Checking the chat model's reply against what it was shown, before it is shown.

The facts are worked out by the software (`mypibypy.diagnosis`); the model's
job is to pick the ones that answer the question and say them. What it got
wrong, when it was asked to reason over numbers, were three things -- and each
is checkable mechanically, so each is checked here instead of being forbidden
in the prompt one more time:

* **A number it was not given.** "Placed 4 of 6" when the run placed 3. Every
  number in the reply must appear in what the model read (the prompt, the
  CURRENT DESIGN block, the conversation), allowing for rounding and for a
  fraction said as a percentage.
* **A comparison the wrong way round.** "70.4 is higher than 73.8". A
  comparison between two numbers is checked against the numbers.
* **"Rejected" when nothing was.** With the GC/Tm window not enforced, no
  window is rejected for it; a reply that says one was is refused (a negated
  "nothing was rejected" is fine).

A refused reply is asked for once more, with the problems stated; if that one
fails too, the user gets the software's facts as they are (`checked_note`).
"""
from __future__ import annotations

import re
from typing import Callable

#: A number as written: thousands commas, decimals, an exponent. A sign is not
#: part of it -- "-1" as a sentinel and "1" are the same claim here.
NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])"
                       r"|(?<![\w.])\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")

#: "higher than", "lower than", ... and which way each one points. Not "more
#: than" / "fewer than": "a 31 nt gap, and a pair needs more than 51" is not a
#: claim that 31 exceeds 51, and those are how such sentences go.
_COMPARATIVES = {"higher": 1, "greater": 1, "larger": 1,
                 "lower": -1, "smaller": -1}
_COMPARE_RE = re.compile(r"\b(" + "|".join(_COMPARATIVES) + r")\s+than\b",
                         re.IGNORECASE)
#: A number naming a pair ("pair 3", "#2") is an index, not a value to compare.
_INDEX_BEFORE = re.compile(r"(?:\bpairs?|#|\bp)\s*$", re.IGNORECASE)
#: How far either side of "higher than" the compared numbers may sit.
_REACH = 48

#: Not "excluded": the screen excludes the probe's own gene, which is true.
_REJECT_RE = re.compile(r"\b(?:reject(?:ed|s|ing)?|filtered\s+out)\b", re.IGNORECASE)
_NEGATION_RE = re.compile(r"\b(?:not|no|nothing|none|never|neither|nor)\b|n't",
                          re.IGNORECASE)


def _value(token: str) -> float:
    return float(token.replace(",", ""))


def _decimals(token: str) -> int:
    body = token.replace(",", "").lower().split("e")[0]
    return len(body.split(".")[1]) if "." in body else 0


def numbers_in(text: str) -> list[str]:
    """Every number in `text`, as written."""
    return NUMBER_RE.findall(text)


def _grounded(token: str, allowed: list[float]) -> bool:
    """Whether `token` is one of `allowed`, as written or rounded, or as a
    percentage of a fraction (GC 0.48 said as 48%)."""
    x, places = _value(token), _decimals(token)
    for a in allowed:
        if round(a, places) == x or round(a * 100, places) == x:
            return True
    return False


def _value_numbers(text: str):
    """(start, end, value) for numbers that are values, not pair indexes."""
    for m in NUMBER_RE.finditer(text):
        if not _INDEX_BEFORE.search(text[max(0, m.start() - 6):m.start()]):
            yield m.start(), m.end(), _value(m.group())


_SENTENCE_END = re.compile(r"[.;!?](?:\s|$)")


def _comparison_problems(note: str) -> list[str]:
    problems = []
    for m in _COMPARE_RE.finditer(note):
        # Both numbers from the comparison's own sentence.
        ends = [e.end() for e in _SENTENCE_END.finditer(note, 0, m.start())]
        lo = max([m.start() - _REACH, *ends])
        stop = _SENTENCE_END.search(note, m.end())
        hi = min(m.end() + _REACH, stop.start() if stop else len(note))
        before = [v for s, _, v in _value_numbers(note) if lo <= s < m.start()]
        after = [v for s, _, v in _value_numbers(note) if m.end() <= s < hi]
        if not before or not after:
            continue
        x, y, sign = before[-1], after[0], _COMPARATIVES[m.group(1).lower()]
        if (x - y) * sign <= 0:
            problems.append(f"it says {x:g} is {m.group(1).lower()} than {y:g}, "
                            f"which is the wrong way round")
    return problems


def check_reply(note: str, grounds: str, *, enforce_window: bool) -> list[str]:
    """What is wrong with `note`, in words; empty if nothing is.

    `grounds` is everything the model read for this turn -- the system prompt
    and the request -- since a number from any of it is one it was given.
    """
    allowed = [_value(t) for t in numbers_in(grounds)]
    problems = [f"{t} does not appear in CURRENT DESIGN or the conversation"
                for t in dict.fromkeys(numbers_in(note)) if not _grounded(t, allowed)]
    problems += _comparison_problems(note)
    if not enforce_window:
        for m in _REJECT_RE.finditer(note):
            if not _NEGATION_RE.search(note[max(0, m.start() - 30):m.start()]):
                problems.append(f'it says "{m.group()}", but the GC/Tm window is '
                                f"not enforced, so nothing was rejected for it")
                break
    return problems


def retry_request(request: str, problems: list[str]) -> str:
    """The request again, with why the last reply was not shown."""
    return (request + "\n\nYOUR PREVIOUS REPLY WAS NOT SHOWN, because "
            + "; ".join(problems) + ". Answer again using only the facts and "
            "numbers in CURRENT DESIGN, quoted as they are.")


def checked_note(note: str, grounds: str, *, enforce_window: bool,
                 ask_again: Callable[[list[str]], str | None],
                 fallback: str) -> tuple[str, str]:
    """(the note to show, how it was settled).

    How: "model" (passed first time), "retried" (the second reply passed), or
    "fallback" (`fallback` is shown instead). `ask_again(problems)` asks the
    model once more and returns its new note, or None if it could not.
    """
    problems = check_reply(note, grounds, enforce_window=enforce_window)
    if not problems:
        return note, "model"
    again = ask_again(problems)
    if again is not None and not check_reply(again, grounds,
                                             enforce_window=enforce_window):
        return again, "retried"
    return fallback, "fallback"
