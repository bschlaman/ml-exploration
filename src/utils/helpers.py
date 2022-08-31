# this file is for utilities that are not important to the topic at hand,
# and are merely convenience functions


def compact_repr(target: list, length: int = 5) -> str:
    # TODO: (2022.08.29) unit test this
    if len(target) <= length or length <= 0:
        return repr(target)
    return f"[{target[0]}, ... " + repr(target[-length:]).lstrip("[")
