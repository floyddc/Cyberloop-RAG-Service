import re
from typing import List


_PORT_SCAN = re.compile(
    r"\bport[\s-]+(?:scan|scans|scanning|scanner|scanners)\b"
    r"|\b(?:scan|scanning)\b.{0,30}\bports?\b",
    re.IGNORECASE,
)
_BEACON_SIGNAL = re.compile(r"\b(?:beacon(?:ing)?|periodic(?:ally)?|jitter)\b", re.IGNORECASE)
_ENCRYPTED_PROTOCOL = re.compile(r"\b(?:tls|ssl|https|encrypted)\b", re.IGNORECASE)
_NETWORK_SIGNAL = re.compile(
    r"\b(?:outbound|connection|connections|traffic|network|c2)\b",
    re.IGNORECASE,
)


def attack_query_expansions(query: str) -> List[str]:
    expansions = []
    if _PORT_SCAN.search(query):
        expansions.append(
            "T1046 Network Service Discovery port scanning network service scanning open ports"
        )

    if (
        _BEACON_SIGNAL.search(query)
        and _ENCRYPTED_PROTOCOL.search(query)
        and _NETWORK_SIGNAL.search(query)
    ):
        expansions.extend(
            [
                "T1071.001 Web Protocols HTTP HTTPS web traffic command and control",
                "T1573 Encrypted Channel TLS SSL encrypted command and control",
            ]
        )

    return expansions
