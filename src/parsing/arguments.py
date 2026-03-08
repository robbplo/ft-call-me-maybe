import re


def get_floats(s: str) -> list[float]:
    # re.findall(r"[-+]?\d*\.\d+|\d+", s)
    matches = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", s)
    print(matches)
    return [float(x) for x in matches]
