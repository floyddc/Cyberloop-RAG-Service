import math
import threading
import time
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import config


@lru_cache(maxsize=65536)
def _stem(token: str) -> str:
    if len(token) < 4 or any(char.isdigit() for char in token):
        return token
    stripped = False
    if token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    if token.endswith("ing") and len(token) >= 6:
        token, stripped = token[:-3], True
    elif token.endswith("ed") and len(token) >= 5:
        token, stripped = token[:-2], True
    if stripped and len(token) >= 4 and token[-1] == token[-2] and token[-1] not in config._VOWELS:
        token = token[:-1]
    if token.endswith("e") and len(token) >= 4:
        token = token[:-1]
    return token


def tokenize(text: str) -> List[str]:
    tokens: List[str] = []

    def keep_technique(match) -> str:
        tokens.append(f"t{match.group(1)}" + (f".{match.group(2)}" if match.group(2) else ""))
        return " "

    remaining = config._TECHNIQUE_ID.sub(keep_technique, text)
    for token in config._TOKEN.findall(remaining.casefold()):
        if len(token) > 1 and token not in config._LEXICAL_STOP_WORDS:
            tokens.append(_stem(token))
    return tokens


class LexicalIndex:


    def __init__(
        self,
        keys: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[Dict[str, Any]],
    ):
        self.keys = list(keys)
        self.texts = list(texts)
        self.metadatas = list(metadatas)
        self._position_by_key = {key: position for position, key in enumerate(self.keys)}
        self._postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._lengths: List[int] = []
        for position, (text, metadata) in enumerate(zip(self.texts, self.metadatas)):
            title = str((metadata or {}).get("title") or "")
            terms = tokenize(f"{title} {text}")
            self._lengths.append(len(terms))
            for term, frequency in Counter(terms).items():
                self._postings[term].append((position, frequency))
        self._average_length = max(sum(self._lengths) / max(len(self._lengths), 1), 1.0)

    def __len__(self) -> int:
        return len(self.keys)

    def position_of(self, key: str) -> Optional[int]:
        return self._position_by_key.get(key)

    def search(self, query: str, limit: Optional[int] = None) -> List[Tuple[int, float]]:
        total = len(self.keys)
        scores: Dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_frequency = math.log(1 + (total - document_frequency + 0.5) / (document_frequency + 0.5))
            for position, frequency in postings:
                length_norm = 1 - config._B + config._B * self._lengths[position] / self._average_length
                scores[position] += (
                    inverse_frequency * frequency * (config._K1 + 1) / (frequency + config._K1 * length_norm)
                )
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return ranked[:limit] if limit else ranked  # position, BM25 score pairs


class LexicalIndexCache:

    def __init__(self, max_entries: int = 32):
        self._entries: Dict[str, Tuple[Any, float, LexicalIndex]] = {}
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def get(
        self,
        name: str,
        version: Any,
        ttl_seconds: float,
        build: Callable[[], LexicalIndex],
    ) -> LexicalIndex:
        with self._lock:
            entry = self._entries.get(name)
            now = time.monotonic()
            if entry and entry[0] == version and now - entry[1] < ttl_seconds:
                return entry[2]
            index = build()
            self._entries[name] = (version, now, index)
            while len(self._entries) > self._max_entries:
                oldest = min(self._entries, key=lambda key: self._entries[key][1])
                del self._entries[oldest]
            return index
