#!/usr/bin/env python3
import json
import os
import regex
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterator
from tqdm import tqdm


# GPT-2 style pre-tokenization pattern
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> list[str]:
    """
    Split text on special tokens to avoid merging across pretoken boundaries.
    """
    if not special_tokens:
        return [text]

    # Create regex pattern with escaped special tokens
    pattern = "|".join(re.escape(token) for token in special_tokens)

    # Split but keep the separators
    parts = re.split(f"({pattern})", text)

    # Filter out empty strings
    return [part for part in parts if part]


def _stream_text_parts(input_path: Path, special_tokens: list[str]) -> Iterator[str]:
    tokens = special_tokens or []
    prefixes = {token[:i] for token in tokens for i in range(1, len(token))}
    buffer = ""

    with input_path.open("r", encoding="utf-8") as handle:
        for chunk in handle:
            buffer += chunk
            parts = _split_on_special_tokens(buffer, tokens)
            if parts and parts[-1] not in tokens and parts[-1] in prefixes:
                buffer = parts.pop()
            else:
                buffer = ""

            for part in parts:
                yield part

    if buffer:
        yield from _split_on_special_tokens(buffer, tokens)


def _batched_parts(
    input_path: Path, special_tokens: list[str], batch_size: int
) -> Iterator[list[str]]:
    batch: list[str] = []
    for part in _stream_text_parts(input_path, special_tokens):
        batch.append(part)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def _count_pretokens_chunk(payload: tuple[list[str], tuple[str, ...]]) -> Counter[tuple[bytes, ...]]:
    parts, special_tokens = payload
    tokens = set(special_tokens)
    counter: Counter[tuple[bytes, ...]] = Counter()

    for part in parts:
        if part in tokens:
            continue
        for pretoken in regex.findall(PAT, part):
            byte_tuple = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
            counter[byte_tuple] += 1

    return counter


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    num_workers: int | None = None,
    batch_size: int = 2048,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given corpus.

    Args:
        input_path: Path to the text file containing training data
        vocab_size: Total desired vocabulary size (including special tokens and bytes)
        special_tokens: list of special tokens to add to vocabulary
        num_workers: number of parallel workers used during pre-tokenization (None => cpu count)
        batch_size: number of text segments processed per worker task

    Returns:
        - vocab: dict mapping token IDs to their byte representations
        - merges: list of merge operations (pairs of bytes that were merged)
    """
    input_path = Path(input_path)
    special_tokens = special_tokens or []

    # Step 1: Initialize vocabulary with special tokens and all bytes
    vocab: dict[int, bytes] = {}
    next_id = 0

    # Add special tokens first
    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    # Add all 256 possible byte values
    for i in range(256):
        vocab[next_id] = bytes([i])
        next_id += 1

    # Step 2: Read and pre-tokenize the corpus
    print(f"Streaming corpus from {input_path}...")

    # Pre-tokenize and count word frequencies
    worker_count = 1 if num_workers == 1 else (num_workers or os.cpu_count() or 1)
    word_in_bytes_counter: dict[tuple[bytes, ...], int] = Counter()

    if worker_count > 1:
        payloads = (
            (batch, tuple(special_tokens))
            for batch in _batched_parts(input_path, special_tokens, batch_size)
        )
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            for chunk_counter in tqdm(
                executor.map(_count_pretokens_chunk, payloads, chunksize=1),
                desc="Pretokenizing",
            ):
                word_in_bytes_counter.update(chunk_counter)
    else:
        for part in tqdm(_stream_text_parts(input_path, special_tokens), desc="Pretokenizing"):
            if part in special_tokens:
                continue
            for pretoken in regex.findall(PAT, part):
                byte_tuple = tuple(bytes([b]) for b in pretoken.encode("utf-8"))
                word_in_bytes_counter[byte_tuple] += 1

    print(f"Found {len(word_in_bytes_counter)} unique pre-tokens")

    # Step 3: Perform BPE merges
    num_merges = vocab_size - len(vocab)
    merges: list[tuple[bytes, bytes]] = []

    print(f"Performing {num_merges} BPE merges...")
    # At first the vocab looks like:
    # {0: b"<|endoftext|>", 1: b"\x00", 2: b"\x01", ..., 257: b"\xff"}
    # At the end it contains mappings from new merged ids to the concatenated bytes.
    for merge_idx in tqdm(range(num_merges)):
        # Count all adjacent byte pairs
        pair_counts = _count_pairs(word_in_bytes_counter)

        if not pair_counts:
            print(f"No more pairs to merge after {merge_idx} merges")
            break

        # Find most common pair (lexicographic tiebreaking)
        p1, p2 = max(pair_counts.items(), key=lambda x: (x[1], x[0]))[0]

        # Record this merge (pairs should be bytes objects)
        merge_byte_pair: tuple[bytes, bytes] = (p1, p2)
        merges.append(merge_byte_pair)

        # Concatenate two bytes as the merged token
        new_token = p1 + p2
        vocab[next_id] = new_token
        next_id += 1

        # Update word frequencies by merging the pair
        word_in_bytes_counter = _merge_text(word_in_bytes_counter, merge_byte_pair, new_token)

    print(f"BPE training completed. Final vocab size: {len(vocab)}")

    return vocab, merges


def _count_pairs(word_in_bytes_counter: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """
    Count byte pairs in all words without exceeding the word boundary.
    """
    pair_counts: dict[tuple[bytes, bytes], int] = Counter()

    for word, count in word_in_bytes_counter.items():
        # Count adjacent pairs in this word
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pair_counts[pair] += count

    return pair_counts


def _merge_text(
    word_in_bytes_counter: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
    new_token: bytes,
) -> dict[tuple[bytes, ...], int]:
    """
    Given a word in bytes counter of a text, the target byte pair and the new token,
    merge the byte pair and return the updated counter.
    """
    new_counter: dict[tuple[bytes, ...], int] = Counter()

    for word_in_bytes, count in word_in_bytes_counter.items():
        new_bytes = _merge(word_in_bytes, pair, new_token)
        new_counter[new_bytes] = count

    return new_counter


def _merge(ids: list[int], pair: tuple[int, int], idx: int) -> tuple[int, ...]:
    """
    In the list of ids, replace all consecutive occurrences of the pair with the new id.
    For example, if ids is [1, 2, 3, 2, 1] and pair is (2, 3), then the new ids will be
    [1, 4, 2, 1].
    Args:
        - ids: list[int] The ids to merge.
        - pair: tuple[int, int] The pair of ids to merge.
        - idx: int The index of the pair to merge.
    Returns:
        - new_ids: list[int] The merged ids.
    """
    new_ids: list[int] = []
    i = 0
    while i < len(ids):
        # Check if we can merge at this position
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return tuple(new_ids)


def _bytes_to_text(blob: bytes) -> str:
    """Round-trip safe conversion from bytes to text for JSON serialization."""
    return blob.decode("latin-1")


def _serialize_vocab(vocab: dict[int, bytes]) -> dict[str, str]:
    """Convert vocab byte values to JSON-safe strings keyed by token id."""
    return {str(idx): _bytes_to_text(token_bytes) for idx, token_bytes in vocab.items()}


def _serialize_merges(merges: list[tuple[bytes, bytes]]) -> list[list[str]]:
    """Convert merge byte pairs into JSON-safe list of string pairs."""
    return [[_bytes_to_text(left), _bytes_to_text(right)] for left, right in merges]


# Example usage and testing
if __name__ == "__main__":
    # Train BPE
    corpus_path = Path("data/owt_train.txt")
    
    vocab, merges = train_bpe(
        input_path=corpus_path,
        vocab_size=32000,  # 1 special + 256 bytes + 43 merges
        special_tokens=["<|endoftext|>"],
    )

    Path("cs336_basics/vocab.json").write_text(
        json.dumps(_serialize_vocab(vocab), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    Path("cs336_basics/merges.json").write_text(
        json.dumps(_serialize_merges(merges), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
