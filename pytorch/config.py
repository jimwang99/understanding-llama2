import os.path
from dataclasses import dataclass

@dataclass
class HyperParam:  # default hyper-parameters for llama2 7B model
    """Hyper-parameters of the model. The default value is for llama2-7B
    The namings are different from OSS llama/model.py,
    because I want to be consistent with my diagram
    """

    D: int = 4096  # dimension of embedding, dim
    Nl: int = 32  # num of transformer layers
    Nh: int = 32  # num of heads, n_heads
    Nkv: int = -1  # num of KV heads, n_kv_heads
    V: int = 32000  # size of the vocabulary, vocab_size
    Dh: int = (
        -1
    )  # hidden dimension of FFN layer, default is calculated in __post_init__
    E: float = (
        1e-06  # a small number that is used in RMS normalization to avoid divide by 0
    )

    Lm: int = 2048  # max sequence length

    # following params are used to calculate Dh
    _multiple_of: int = 256  # make hidden dimension the multiple of large power of 2
    _ffn_dim_multiplier: float = (
        1.0  # custom multiplier for hidden dimension of FFN layer
    )

    # dropout: float = 0.0

    def __post_init__(self):
        assert self.D % self.Nh == 0

        if self.Nkv == -1:
            self.Nkv = self.Nh
        assert self.Nh % self.Nkv == 0, f"{self.Nh=} {self.Nkv=}"

        if self.Dh == -1:
            self.Dh = 4 * self.D
            self.Dh = int(2 * self.Dh / 3)
            self.Dh = self._multiple_of * (
                (self.Dh + self._multiple_of - 1) // self._multiple_of
            )  # round up to N * self._multiple_of


MODEL_HPARAMS = {}
MODEL_HPARAMS["llama2-7B"] = HyperParam()
MODEL_HPARAMS["tinystories260K"] = HyperParam(
    D=64, Nl=5, Nh=8, Nkv=4, V=512, Dh=-1, E=1e-05, Lm=512, _multiple_of=4
)
MODEL_HPARAMS["tinystories15M"] = HyperParam(
    D=288, Nl=6, Nh=6, Nkv=6, V=32000, Dh=-1, E=1e-05, Lm=256, _multiple_of=32
)
MODEL_HPARAMS["tinystories42M"] = HyperParam(
    D=512, Nl=8, Nh=8, Nkv=8, V=32000, Dh=-1, E=1e-05, Lm=1024, _multiple_of=32
)
MODEL_HPARAMS["tinystories110M"] = HyperParam(
    D=768, Nl=12, Nh=12, Nkv=12, V=32000, Dh=-1, E=1e-05, Lm=1024, _multiple_of=32
)

MODEL_NAMES = MODEL_HPARAMS.keys()

MODEL_FILES = {
    "tinystories260K": "../third-party/tinyllamas/stories260K.pt",
    "tinystories15M": "../third-party/tinyllamas/stories15M.pt",
    "tinystories42M": "../third-party/tinyllamas/stories42M.pt",
    "tinystories110M": "../third-party/tinyllamas/stories110M.pt",
    "tokenizer32K": "../third-party/tinyllamas/tokenizer.pt",
    "tokenizer512": "../third-party/tinyllamas/tok512.pt",
}

for fpath in MODEL_FILES.values():
    assert os.path.isfile(fpath), f"Cannot find file {fpath}"