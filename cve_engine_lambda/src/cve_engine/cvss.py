from enum import Enum
from dataclasses import dataclass


class CVSSMetricType(Enum):
    Exploit = "Exploit"
    Impact = "Impact"
    Scope = "Scope"


@dataclass
class CVSSMetricMeta:
    type: CVSSMetricType
    abbrev: str
    name: str
    # ordered from higher to lower CVSS Score weighting
    categories: list[str]


CVSS_BASE_METRICS = {
    "AV": CVSSMetricMeta(
        CVSSMetricType.Exploit,
        "AV",
        "Attack Vector",
        ["Network", "Adjacent", "Local", "Physical"],
    ),
    "AC": CVSSMetricMeta(
        CVSSMetricType.Exploit, "AC", "Attack Complexity", ["Low", "High"]
    ),
    "PR": CVSSMetricMeta(
        CVSSMetricType.Exploit, "PR", "Privileges Required", ["None", "Low", "High"]
    ),
    "UI": CVSSMetricMeta(
        CVSSMetricType.Exploit, "UI", "User Interaction", ["None", "Required"]
    ),
    "S": CVSSMetricMeta(CVSSMetricType.Scope, "S", "Scope", ["Changed", "Unchanged"]),
    "C": CVSSMetricMeta(
        CVSSMetricType.Impact, "C", "Confidentiality", ["High", "Low", "None"]
    ),
    "I": CVSSMetricMeta(
        CVSSMetricType.Impact, "I", "Integrity", ["High", "Low", "None"]
    ),
    "A": CVSSMetricMeta(
        CVSSMetricType.Impact, "A", "Availability", ["High", "Low", "None"]
    ),
}
