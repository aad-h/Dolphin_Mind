"""In-memory LRU + TTL cache of answered questions for the HTTP service."""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict
from typing import Callable, Hashable, Optional, Tuple

CacheKey = Tuple[Hashable, ...]


def normalize_question(question: str) -> str:
    """Case- and whitespace-insensitive form, so trivially different spellings share an entry."""
    return " ".join(question.lower().split())


def response_key(doc_id: str, question: str, max_steps: int) -> CacheKey:
    """Cache key: same document (content-addressed id), same question, same step budget."""
    return (doc_id, normalize_question(question), int(max_steps))


class ResponseCache:
    """Thread-safe LRU cache of JSON-ready agent responses with an optional TTL.

    Answers are only as deterministic as the backend, so the cache trades freshness for
    cost: a hit costs no tokens and no latency. Entries older than ``ttl`` seconds are
    treated as missing; beyond ``max_entries`` the least recently used one is dropped.
    Values are deep-copied in and out, so callers can mutate what they get back.
    """

    def __init__(self, max_entries: int = 256, ttl: Optional[float] = None,
                 clock: Callable[[], float] = time.monotonic):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be > 0 seconds")
        self.max_entries, self.ttl, self._clock = max_entries, ttl, clock
        self._data: "OrderedDict[CacheKey, Tuple[float, dict]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def __len__(self) -> int:
        return len(self._data)

    def get(self, key: CacheKey) -> Optional[dict]:
        with self._lock:
            entry = self._data.get(key)
            if entry is not None and self.ttl is not None and self._clock() - entry[0] > self.ttl:
                del self._data[key]
                entry = None
            if entry is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return copy.deepcopy(entry[1])

    def put(self, key: CacheKey, value: dict) -> None:
        with self._lock:
            self._data[key] = (self._clock(), copy.deepcopy(value))
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._data), "max_entries": self.max_entries, "ttl": self.ttl,
                    "hits": self.hits, "misses": self.misses}
