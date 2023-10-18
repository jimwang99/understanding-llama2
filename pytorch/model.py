# minimalist version llama2

# References:
# https://github.com/facebookresearch/llama/blob/main/llama/model.py
# https://github.com/karpathy/llama2.c/blob/master/model.py

import math
import torch

from typing import Optional, Sequence, Tuple
from loguru import logger

from config import HyperParam, MODEL_HPARAMS, MODEL_NAMES, MODEL_FILES

logger.add("/tmp/understanding-llama2/pytorch-model.log")


class RMSNorm(torch.nn.Module):
    def __init__(self, D: int, E: float):
        super().__init__()
        self.D = D
        self.E = E
        self.weight = torch.nn.Parameter(torch.ones(self.D))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.E)

    def forward(self, x):
        output = self._norm(x)
        return output * self.weight


class Model(torch.nn.Module):
    """Vanilla model of llama architecture
    >>> m = Model(MODEL_HPARAMS["tinystories260K"])
    """

    def __init__(self, hparam: HyperParam) -> None:
        super().__init__()

        self.hparam = hparam

        D = hparam.D
        Nl = hparam.Nl
        Nh = hparam.Nh
        Nkv = hparam.Nkv
        V = hparam.V
        Dh = hparam.Dh
        E = hparam.E
        Lm = hparam.Lm

        # -------------------------------------------------------------------
        # define model

        # embedding layer
        self.embedding = torch.nn.Embedding(V, D)

        # transformer
        self.layers = torch.nn.ModuleList()
        for _ in range(Nl):
            layer = torch.nn.ModuleDict()
            self.layers.append(layer)

            layer["attention_norm"] = RMSNorm(D, E)

            # Multi-head attention MHA
            ## Projection of inputs in MHA
            layer["projection_q"] = torch.nn.Linear(D, D, bias=False)
            layer["projection_k"] = torch.nn.Linear(D, D * Nkv // Nh, bias=False)
            layer["projection_v"] = torch.nn.Linear(D, D * Nkv // Nh, bias=False)

            ## Projection of output in MHA
            layer["projection_a"] = torch.nn.Linear(D, D, bias=False)

            # Feed-forward network FFN
            layer["ffn_norm"] = RMSNorm(D, E)

            ## Projections inside FFN
            layer["projection_gate"] = torch.nn.Linear(D, Dh, bias=False)
            layer["projection_up"] = torch.nn.Linear(D, Dh, bias=False)
            layer["projection_down"] = torch.nn.Linear(Dh, D, bias=False)

        # output normalization
        self.output_norm = RMSNorm(D, E)

        # model head
        self.output_linear = torch.nn.Linear(D, V, bias=False)

        # -------------------------------------------------------------------
        # prepare cos|sin(mθ)
        cos, sin = self._precompute_freqs_cis(D, Nh, Lm)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self._trace_and_check(cos, "cosm", (Lm, D // Nh // 2))
        self._trace_and_check(sin, "sinm", (Lm, D // Nh // 2))

        self._layer_idx = -1

    def _trace_and_check(
        self,
        tensor: torch.Tensor,
        name: str,
        shape: Optional[Sequence[int]] = None,
        layer_idx: Optional[int] = None,
    ):
        """TACT (trace and check tensor)"""
        prefix = f"layer-{layer_idx} " if layer_idx else ""
        logger.trace(f"{prefix}{name}.shape={tensor.shape}")
        if shape:
            assert (
                tensor.shape == shape
            ), f"{name}.shape={tensor.shape} != {shape} while {self.hparam=}"

    def _precompute_freqs_cis(
        self, D: int, Nh: int, Lm: int, theta_base: float = 10000.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        theta = 1.0 / (
            theta_base
            ** (
                torch.arange(0, D // Nh, 2)[: (D // Nh // 2)].to(torch.float32)
                / (D // Nh)
            )
        )
        abs_pos = torch.arange(Lm)
        freqs = torch.outer(abs_pos, theta)
        freqs_cos = torch.cos(freqs)
        freqs_sin = torch.sin(freqs)
        return (freqs_cos, freqs_sin)

    def forward(self, ti: torch.Tensor) -> torch.Tensor:
        D = self.hparam.D
        Nl = self.hparam.Nl
        Nh = self.hparam.Nh
        Nkv = self.hparam.Nkv
        V = self.hparam.V
        Dh = self.hparam.Dh
        E = self.hparam.E
        Lm = self.hparam.Lm

        assert ti.ndim == 2
        B, L = ti.shape
        self._trace_and_check(ti, "ti")
        assert L <= Lm

        # -------------------------------------------------------------------
        # prepare

        # prepare cis
        cos = self.cos[:L].view((1, L, 1, D // Nh // 2))  # type: ignore
        sin = self.sin[:L].view((1, L, 1, D // Nh // 2))  # type: ignore
        self._trace_and_check(cos, "cos")
        self._trace_and_check(sin, "sin")

        # prepare mask matrix
        mf = torch.full((1, 1, L, L), float("-inf"), dtype=torch.float32)
        m = torch.triu(mf, diagonal=1)
        self._trace_and_check(m, "m", (1, 1, L, L))

        # -------------------------------------------------------------------
        # model inference

        # Embeddings
        xi = self.embedding(ti)
        self._trace_and_check(xi, "xi", (B, L, D))

        # transformer
        for layer_idx, layer in enumerate(self.layers):
            logger.trace(f"> transformer layer {layer_idx}")
            self._layer_idx = layer_idx

            # Attention norm
            xi1 = layer["attention_norm"](xi)
            self._trace_and_check(xi1, "xi1", (B, L, D))

            # ===============================================================
            # MHA

            # Projection
            q = layer["projection_q"](xi1)
            k = layer["projection_k"](xi1)
            v = layer["projection_v"](xi1)
            self._trace_and_check(q, "q", (B, L, D))
            self._trace_and_check(k, "k", (B, L, D * Nkv // Nh))
            self._trace_and_check(v, "v", (B, L, D * Nkv // Nh))

            q1 = q.view(B, L, Nh, D // Nh)
            k1 = k.view(B, L, Nkv, D // Nh)
            v1 = v.view(B, L, Nkv, D // Nh)

            # ---------------------------------------------------------------
            # RoPE

            # To complex
            q1 = q1.reshape(B, L, Nh, -1, 2)
            k1 = k1.reshape(B, L, Nkv, -1, 2)
            qr, qi = q1.unbind(-1)
            kr, ki = k1.unbind(-1)
            self._trace_and_check(qr, "qr", (B, L, Nh, D // Nh // 2))
            self._trace_and_check(qi, "qi", (B, L, Nh, D // Nh // 2))
            self._trace_and_check(kr, "kr", (B, L, Nkv, D // Nh // 2))
            self._trace_and_check(ki, "ki", (B, L, Nkv, D // Nh // 2))

            # Rotate
            qr1 = qr * cos - qi * sin
            qi1 = qr * sin + qi * cos
            kr1 = kr * cos - ki * sin
            ki1 = kr * sin + ki * cos
            self._trace_and_check(qr1, "qr1", (B, L, Nh, D // Nh // 2))
            self._trace_and_check(qi1, "qi1", (B, L, Nh, D // Nh // 2))
            self._trace_and_check(kr1, "kr1", (B, L, Nkv, D // Nh // 2))
            self._trace_and_check(ki1, "ki1", (B, L, Nkv, D // Nh // 2))

            # Merge
            q2 = torch.stack([qr1, qi1], dim=-1).flatten(3)
            k2 = torch.stack([kr1, ki1], dim=-1).flatten(3)
            self._trace_and_check(q2, "q2", (B, L, Nh, D // Nh))
            self._trace_and_check(k2, "k2", (B, L, Nkv, D // Nh))

            # ---------------------------------------------------------------
            # GQA

            kx = k2.repeat_interleave(repeats=(Nh // Nkv), dim=2)
            vx = v1.repeat_interleave(repeats=(Nh // Nkv), dim=2)
            self._trace_and_check(kx, "kx", (B, L, Nh, D // Nh))
            self._trace_and_check(kx, "vx", (B, L, Nh, D // Nh))

            qt = q2.transpose(1, 2)
            kt = kx.transpose(1, 2)
            vt = vx.transpose(1, 2)
            self._trace_and_check(qt, "qt", (B, Nh, L, D // Nh))
            self._trace_and_check(kt, "kt", (B, Nh, L, D // Nh))
            self._trace_and_check(vt, "vt", (B, Nh, L, D // Nh))

            # scaled dot product attention
            kt1 = kt.transpose(2, 3)
            self._trace_and_check(kt1, "kt1", (B, Nh, D // Nh, L))

            a = torch.matmul(qt, kt1) / math.sqrt(D // Nh)
            self._trace_and_check(a, "a", (B, Nh, L, L))

            am = a + m[:, :, :L, :L]
            self._trace_and_check(am, "am", (B, Nh, L, L))

            as_ = torch.nn.functional.softmax(am, dim=-1)
            self._trace_and_check(as_, "as", (B, Nh, L, L))

            sa = torch.matmul(as_, vt)
            self._trace_and_check(sa, "sa", (B, Nh, L, D // Nh))

            # concate
            sac = sa.transpose(1, 2).contiguous()
            sac1 = sac.view(B, L, D)
            self._trace_and_check(sac1, "sac1", (B, L, D))

            # self-attention projection
            sap = layer["projection_a"](sac1)
            self._trace_and_check(sap, "sap", (B, L, D))

            # ---------------------------------------------------------------

            ha = sap + xi  # residual
            self._trace_and_check(ha, "ha", (B, L, D))

            hi = layer["ffn_norm"](ha)
            self._trace_and_check(hi, "hi", (B, L, D))

            # ---------------------------------------------------------------
            # FFN

            hg = layer["projection_gate"](hi)
            self._trace_and_check(hg, "hg", (B, L, Dh))

            hu = layer["projection_up"](hi)
            self._trace_and_check(hu, "hu", (B, L, Dh))

            hs = torch.nn.functional.silu(hg)
            self._trace_and_check(hs, "hs", (B, L, Dh))

            hm = hs * hu
            self._trace_and_check(hm, "hm", (B, L, Dh))

            hd = layer["projection_down"](hm)
            self._trace_and_check(hd, "hd", (B, L, D))

            # ---------------------------------------------------------------

            hf = hd + ha  # residual
            self._trace_and_check(hf, "hf", (B, L, D))

            xi = hf  # for the next layer

        xo = self.output_norm(hf)
        self._trace_and_check(xo, "xo", (B, L, D))

        lo = self.output_linear(xo[:, [-1], :])  # only run the last (future) token
        self._trace_and_check(lo, "lo", (B, 1, V))

        return lo


def load_model(model_name: str) -> Model:
    """
    >>> m = load_model("tinystories260K")
    """
    logger.debug(f"{model_name=}")
    assert model_name in MODEL_NAMES, f"{model_name=} is not supported"

    from export import export_tinyllamas

    state_dict = export_tinyllamas(MODEL_FILES[model_name])

    model = Model(hparam=MODEL_HPARAMS[model_name])
    model.load_state_dict(state_dict)
    model.eval()

    return model
