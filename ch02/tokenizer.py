import re
from collections.abc import Iterable


_TOKEN_PATTERN = re.compile(r"""([,.:;?_!"()']|--|\s)""")
_PUNCTUATION_SPACING_PATTERN = re.compile(r"""\s+([,.?!"()'])""")


def build_vocab(text: str, include_special_tokens: bool = False) -> dict[str, int]:
    """Build a sorted token-to-ID vocabulary from text."""
    tokens = [token.strip() for token in _TOKEN_PATTERN.split(text) if token.strip()]
    all_tokens = sorted(set(tokens))
    if include_special_tokens:
        all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    return {token: token_id for token_id, token in enumerate(all_tokens)}


class SimpleTokenizerV1:
    """Whitespace-and-punctuation tokenizer that requires every token in its vocabulary."""

    def __init__(self, vocab: dict[str, int]):
        self.str_to_int = vocab
        self.int_to_str = {token_id: token for token, token_id in vocab.items()}

    def encode(self, text: str) -> list[int]:
        tokens = [token.strip() for token in _TOKEN_PATTERN.split(text) if token.strip()]
        return [self.str_to_int[token] for token in tokens]

    def decode(self, ids: Iterable[int]) -> str:
        text = " ".join(self.int_to_str[token_id] for token_id in ids)
        return _PUNCTUATION_SPACING_PATTERN.sub(r"\1", text)


class SimpleTokenizerV2(SimpleTokenizerV1):
    """Simple tokenizer that maps out-of-vocabulary tokens to <|unk|>."""

    def __init__(self, vocab: dict[str, int]):
        super().__init__(vocab)
        if "<|unk|>" not in self.str_to_int:
            raise ValueError("vocab must contain an '<|unk|>' token")

    def encode(self, text: str) -> list[int]:
        tokens = [token.strip() for token in _TOKEN_PATTERN.split(text) if token.strip()]
        unknown_id = self.str_to_int["<|unk|>"]
        return [self.str_to_int.get(token, unknown_id) for token in tokens]
