from loguru import logger
from typing import List
from sentencepiece import SentencePieceProcessor

from config import MODEL_NAMES, MODEL_FILES

class Tokenizer:
    def __init__(self, fpath_model) -> None:
        logger.debug(f"loading SentencePieceProcessor model from {fpath_model}")
        self.model = SentencePieceProcessor(model_file=fpath_model)

        self.V = self.model.vocab_size()
        self.BOS = self.model.bos_id()
        self.EOS = self.model.eos_id()
        self.PAD = self.model.pad_id()

        logger.debug(f"V={self.V} BOS={self.BOS}, EOS={self.EOS} PAD={self.PAD}")

    def encode(self, s: str) -> List[int]:
        assert type(s) is str, type(s)
        return self.model.encode(s)

    def decode(self, t: List[int]) -> str:
        return self.model.decode(t)


def load_tokenizer(model_name: str) -> Tokenizer:
    """
    >>> tkr = load_tokenizer("tinystories260K")
    """
    assert model_name in MODEL_NAMES

    if model_name.startswith("tinystories"):
        if model_name == "tinystories260K":
            return Tokenizer(fpath_model=MODEL_FILES["tokenizer512"])
        else:
            return Tokenizer(fpath_model=MODEL_FILES["tokenizer32K"])
    elif model_name.startswith("tokenizer"):
        return Tokenizer(fpath_model=MODEL_FILES[model_name])
    else:
        raise NotImplementedError(f"{model_name=} is not supported")