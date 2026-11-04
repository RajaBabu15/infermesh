"""Mooncake FAST'25 trace loader and prompt synthesis.

The traces (github.com/kvcache-ai/Mooncake, FAST25-release/traces, Apache-2.0 repo)
carry no text: each row is {timestamp (ms), input_length, output_length, hash_ids},
where every hash id names a 512-token block and equal ids mean a reusable prefix
block. The router under test matches characters, so each request is rendered as
a string:

  * every FULL 512-token block becomes 2048 deterministic characters (512 tokens x 4
    chars, the gateway's own chars//4 token estimate), identical for equal hash ids;
  * the remainder (a partial last block, which engines do not cache, or a whole
    prompt shorter than one block) becomes a request-unique tail, so an uncached
    tail can never produce a false cache hit;
  * the total is exactly 4 * input_length characters, so _estimate_tokens() on the
    prompt returns input_length.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

BLOCK_TOKENS = 512
CHARS_PER_TOKEN = 4
BLOCK_CHARS = BLOCK_TOKENS * CHARS_PER_TOKEN


@dataclass(frozen=True)
class TraceRequest:
    index: int
    ts_s: float                      # arrival, seconds from trace start
    input_len: int                   # tokens
    output_len: int                  # tokens
    hash_ids: tuple[int, ...]

    @property
    def shared_block_ids(self) -> tuple[int, ...]:
        """Ids of the full 512-token blocks (the only ones an engine caches)."""
        return self.hash_ids[: self.input_len // BLOCK_TOKENS]


def load_trace(path: str | Path, limit: int | None = None) -> list[TraceRequest]:
    requests: list[TraceRequest] = []
    with open(path) as f:
        for index, line in enumerate(f):
            if limit is not None and index >= limit:
                break
            row = json.loads(line)
            requests.append(TraceRequest(
                index=index,
                ts_s=row["timestamp"] / 1000.0,
                input_len=int(row["input_length"]),
                output_len=int(row["output_length"]),
                hash_ids=tuple(row["hash_ids"]),
            ))
    return requests


def _filler(seed: str, length: int) -> str:
    digest = hashlib.blake2b(seed.encode(), digest_size=32).hexdigest()   # 64 hex chars
    return (digest * (length // len(digest) + 1))[:length]


@lru_cache(maxsize=None)
def block_text(hash_id: int) -> str:
    """2048 deterministic characters for one 512-token block. The text starts with
    hash digits, so two different blocks share a leading character only by chance."""
    return _filler(f"block-{hash_id}", BLOCK_CHARS)


def synthesize_prompt(req: TraceRequest) -> str:
    shared = req.shared_block_ids
    tail_len = req.input_len * CHARS_PER_TOKEN - len(shared) * BLOCK_CHARS
    tail = _filler(f"tail-{req.index}", tail_len) if tail_len > 0 else ""
    prompt = "".join(block_text(h) for h in shared) + tail
    assert len(prompt) == req.input_len * CHARS_PER_TOKEN
    return prompt
