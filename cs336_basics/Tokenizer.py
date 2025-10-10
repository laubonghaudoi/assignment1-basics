from pathlib import Path
from typing import Iterable, Iterator
from collections import Counter
import json
import regex
from cs336_basics.train_bpe import _split_on_special_tokens

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def _text_to_bytes(text: str) -> bytes:
    return text.encode("latin-1")


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        """Construct a tokenizer from serialized vocabulary, merges, and optional special tokens."""
        self.vocab = vocab
        self.byte_to_id = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.merge_ranks = {pair: i for i, pair in enumerate(merges)}
        self.special_tokens = special_tokens or ["<|endoftext|>"]
        self.special_tokens = sorted(self.special_tokens, key=len, reverse=True)

    def _pre_tokenize(self, text: str) -> list[str]:
        """Apply GPT-2 style regex pre-tokenization."""
        pretokens = regex.findall(PAT, text)
        return pretokens

    def _apply_bpe(self, token_bytes: bytes) -> list[bytes]:
        pieces = [bytes([b]) for b in token_bytes]
        if len(pieces) <= 1:
            return pieces

        while True:
            best_pair = None
            best_rank = None
            best_pos = -1

            for i in range(len(pieces) - 1):
                pair = (pieces[i], pieces[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_pair = pair
                    best_rank = rank
                    best_pos = i

            if best_pair is None:
                break

            merged = best_pair[0] + best_pair[1]
            pieces = pieces[:best_pos] + [merged] + pieces[best_pos + 2 :]

        return pieces

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ):
        """Load tokenizer artifacts from JSON dumped by the BPE trainer."""
        vocab_data = json.loads(Path(vocab_filepath).read_text(encoding="utf-8"))
        merges_data = json.loads(Path(merges_filepath).read_text(encoding="utf-8"))

        vocab = {int(idx): _text_to_bytes(token) for idx, token in vocab_data.items()}
        merges = [(_text_to_bytes(left), _text_to_bytes(right)) for left, right in merges_data]

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def _get_pair_counts(self, token_ids: list[int]) -> dict[tuple[int, int], int]:
        """
        Get the counts of all adjacent pairs in the token IDs.
        """
        pair_counts = Counter()
        for pair in zip(token_ids, token_ids[1:]):
            pair_counts[pair] += 1
        return pair_counts

    def _merge(self, bytes_list: list[bytes], b1: bytes, b2: bytes) -> list[int]:
        """
        A byte pair can be merged, return the new list of ids of a word.
        """
        merged_byte: bytes = b1 + b2

        new_bytes: list[bytes] = []
        i = 0
        while i < len(bytes_list):
            if bytes_list[i] == b1 and bytes_list[i + 1] == b2:
                new_bytes.append(merged_byte)
                i += 2
            else:
                new_bytes.append(bytes_list[i])
                i += 1

        return new_bytes

    def encode(self, text: str) -> list[int]:
        """
        Encode a string into a list of token IDs.
        """
        segments = _split_on_special_tokens(text, list(self.special_tokens)) if self.special_tokens else [text]
        token_ids: list[int] = []
        for segment in segments:
            if segment in self.special_tokens:
                token_ids.append(self.byte_to_id[segment.encode("utf-8")])
                continue

            for pretoken in regex.findall(PAT, segment):
                pieces = self._apply_bpe(pretoken.encode("utf-8"))
                token_ids.extend(self.byte_to_id[piece] for piece in pieces)

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields
        token IDs. This is required for memory-eﬀicient tokenization of large files that we cannot directly
        load into memory.
        """
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        tokens = b"".join(self.vocab[token_id] for token_id in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text


if __name__ == "__main__":
    tokenizer = Tokenizer.from_files(
        "cs336_basics/vocab.json",
        "cs336_basics/merges.json",
        special_tokens=["<|endoftext|>"],
    )
    print(tokenizer.encode("H"))
    print(tokenizer.decode(tokenizer.encode("H")))
