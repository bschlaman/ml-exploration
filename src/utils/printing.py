# this file is for utilities that are not important to the topic at hand,
# and are merely convenience functions
from __future__ import annotations

from typing import Any, Iterable

from colorama import Fore, Style


def _apply_color(s: str, ansi_color: str) -> str:
    return f"{ansi_color}{s}{Fore.RESET}"


def bld(s: str) -> str:
    return f"{Style.BRIGHT}{s}{Style.NORMAL}"


def red(s: str) -> str:
    return _apply_color(s, Fore.RED)


def blu(s: str) -> str:
    return _apply_color(s, Fore.BLUE)


def mag(s: str) -> str:
    return _apply_color(s, Fore.MAGENTA)


def yel(s: str) -> str:
    return _apply_color(s, Fore.YELLOW)


def grn(s: str) -> str:
    return _apply_color(s, Fore.GREEN)


def succ(s: str) -> str:
    _succ_icon = "\u2705"
    return f"{_succ_icon} {s}"


def fail(s: str) -> str:
    _fail_icon = "\u274C"
    return f"{_fail_icon} {s}"


def succ_inline(s: str) -> str:
    _succ_icon = "\u2705"
    return f"{grn(s)} {_succ_icon}"


def fail_inline(s: str) -> str:
    _fail_icon = "\u274C"
    return f"{red(s)} {_fail_icon}"


def compact_repr(target: list, length: int = 5) -> str:
    if not target or length < 1:
        raise Exception("bad inputs")
    left_bracket = str(target)[0]
    if len(target) <= length or length <= 0:
        return str(target)
    return f"{left_bracket}{target[0]}, ... " + repr(target[-length:]).lstrip(
        left_bracket
    )


def data_print(labeled_data: dict[str, Any], ansi_color: Fore) -> Iterable[str]:
    longest_label_len = max(map(len, labeled_data.keys()))
    total_len = longest_label_len + len(ansi_color + Fore.RESET)
    for label, data in labeled_data.items():
        if isinstance(data, float):
            data = round(data, 5)
        yield f"{_apply_color(str(label), ansi_color)}:".ljust(total_len + 2) + str(
            data
        )
