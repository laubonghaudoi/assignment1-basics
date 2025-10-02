#!/usr/bin/env python3
import regex
import re
from collections import Counter
from pathlib import Path
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


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train a byte-level BPE tokenizer on the given corpus.

    Args:
        input_path: Path to the text file containing training data
        vocab_size: Total desired vocabulary size (including special tokens and bytes)
        special_tokens: list of special tokens to add to vocabulary

    Returns:
        - vocab: dict mapping token IDs to their byte representations
        - merges: list of merge operations (pairs of bytes that were merged)
    """
    input_path = Path(input_path)

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
    print(f"Reading corpus from {input_path}...")
    text: str = Path(input_path).read_text(encoding="utf-8")
    # Split on special tokens to prevent cross-boundary merging
    text_parts: list[str] = _split_on_special_tokens(text, special_tokens)

    # Pre-tokenize and count word frequencies
    word_in_bytes_counter: dict[tuple[bytes, ...], int] = Counter()
    for part in text_parts:
        if part in special_tokens:
            # Because we split the text on special tokens, we need to skip them here
            continue
        else:
            # Apply regex pre-tokenization
            pretokens = regex.findall(PAT, part)
            for pretoken in pretokens:
                # Convert to tuple of bytes objects (not individual byte values)
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


# Example usage and testing
if __name__ == "__main__":
    # Train BPE
    vocab, merges = train_bpe(
        input_path=Path("data/TinyStoriesV2-GPT4-train.txt"),
        vocab_size=10000,  # 1 special + 256 bytes + 43 merges
        special_tokens=["<|endoftext|>"],
    )

    print(f"\nVocabulary size: {len(vocab)}")
    print(f"Number of merges: {len(merges)}")
    print(merges)