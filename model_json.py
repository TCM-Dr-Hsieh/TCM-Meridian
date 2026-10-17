"""Parsing of the JSON the agents' models answer with."""
from __future__ import annotations

import json
from typing import Any


def loads(text: str) -> Any:
    """`json.loads` for a model's reply. Raw tab and newline characters inside string values are accepted (`strict=False`).

    The records are tab-indented and the models copy their lines verbatim into string values (an operation's `content`, a long
    `thinking`); a strict parse throws a whole, otherwise valid reply away and the call is wasted, again and again, because the
    retry copies the tabs once more. Malformed JSON still raises `json.JSONDecodeError`. Callers validate the content afterwards.
    """
    return json.loads(text, strict=False)
