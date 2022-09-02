# this file is for utilities that are not important to the topic at hand,
# and are merely convenience functions
from typing import Iterable, Any
from colorama import Fore


def compact_repr(target: list, length: int = 5) -> str:
    # TODO: (2022.08.29) unit test this
    left_bracket = str(target)[0]
    if len(target) <= length or length <= 0:
        return str(target)
    return f"{left_bracket}{target[0]}, ... " + repr(target[-length:]).lstrip(
        left_bracket
    )


def data_print(labeled_data: dict[str, Any], ansi_color: str) -> Iterable[str]:
    def _apply_color(s: str) -> str:
        return f"{ansi_color}{s}{Fore.RESET}"

    longest_label_len = max(map(len, labeled_data.keys()))
    total_len = longest_label_len + len(ansi_color + Fore.RESET)
    for label, data in labeled_data.items():
        if isinstance(data, float):
            data = round(data, 4)
        yield f"{_apply_color(str(label))}:".ljust(total_len + 2) + str(data)
