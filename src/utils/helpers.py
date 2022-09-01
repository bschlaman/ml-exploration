# this file is for utilities that are not important to the topic at hand,
# and are merely convenience functions


def compact_repr(target: list, length: int = 5) -> str:
    # TODO: (2022.08.29) unit test this
    left_bracket = str(target)[0]
    if len(target) <= length or length <= 0:
        return str(target)
    return f"{left_bracket}{target[0]}, ... " + repr(target[-length:]).lstrip(
        left_bracket
    )
