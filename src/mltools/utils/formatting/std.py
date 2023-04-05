import datetime
from typing import Any, Iterable

import pandas

from .colors import yel


def fmt(obj: datetime.datetime | float | pandas.Timestamp) -> str:
    if isinstance(obj, float):
        return format(obj, ".4f")
    if isinstance(obj, datetime.datetime):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, pandas.Timestamp):
        return obj.strftime("%Y-%m-%d")
    raise TypeError(f"unsupported type: {type(obj)}")


def compact_repr(target: list, length: int = 5) -> str:
    if not target or length < 1:
        raise Exception("bad inputs")
    left_bracket = str(target)[0]
    if len(target) <= length or length <= 0:
        return str(target)
    return f"{left_bracket}{target[0]}, ... " + repr(target[-length:]).lstrip(
        left_bracket
    )


def data_print(labeled_data: dict[str, Any]) -> Iterable[str]:
    longest_label_len = max(map(len, labeled_data.keys()))
    total_len = longest_label_len + len(yel(""))
    for label, data in labeled_data.items():
        if isinstance(data, float):
            data = round(data, 5)
        yield f"{yel(str(label))}:".ljust(total_len + 2) + str(data)
