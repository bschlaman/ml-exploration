from colorama import Fore, Style


def _apply_color(s: str, ansi_color: str) -> str:
    return f"{ansi_color}{s}{Fore.RESET}"


def bld(s: str) -> str:
    return f"{Style.BRIGHT}{s}{Style.NORMAL}"


def dim(s: str) -> str:
    return f"{Style.DIM}{s}{Style.NORMAL}"


def em(s: str) -> str:
    return f"\033[3m{s}\033[0m"


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
    _fail_icon = "\u274c"
    return f"{_fail_icon} {s}"


def succ_inline(s: str) -> str:
    _succ_icon = "\u2705"
    return f"{grn(s)} {_succ_icon}"


def fail_inline(s: str) -> str:
    _fail_icon = "\u274c"
    return f"{red(s)} {_fail_icon}"
