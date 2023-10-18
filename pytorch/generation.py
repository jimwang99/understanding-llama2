from typing import Callable, Iterator, List

import torch
from loguru import logger

from tokenizer import Tokenizer
from model import Model


class Generator:
    """
    >>> from tokenizer import load_tokenizer
    >>> from model import load_model
    >>> tkr = load_tokenizer("tinystories260K")
    >>> mdl = load_model("tinystories260K")
    >>> gnr = Generator(tkr, mdl)
    """
    def __init__(
        self,
        tokenizer: Tokenizer,
        model: Model,
        prompt: str = "",
        max_new_len: int = 1024,
        temperature: float = 0.6,
        sampling_method: str = "top_k",
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model

        assert type(prompt) == str, type(prompt)
        assert max_new_len > 0, max_new_len
        assert temperature >= 0.0, temperature
        assert sampling_method in ["greedy", "top_k", "top_p"]
        sampling_method = "greedy" if temperature == 0.0 else sampling_method

        self.temperature = temperature
        self.sampling: Callable[[torch.Tensor, float], int] = {
            "greedy": Generator._sampling_greedy,
            "top_k": Generator._sampling_top_k,
            "top_p": Generator._sampling_top_p,
        }[sampling_method]

        # tokenize input
        self.tokens = [tokenizer.BOS] + tokenizer.encode(prompt)
        L = len(self.tokens)

        self.Lt = min(self.model.hparam.Lm, L + max_new_len)
        assert (
            self.Lt > L
        ), f"either prompt={L} is too long, or max_new_len={max_new_len} is too small"

    @staticmethod
    def _sampling_greedy(logits: torch.Tensor, temperature: float) -> int:
        return torch.argmax(logits[:, -1], dim=-1)

    @staticmethod
    def _sampling_top_k(
        logits: torch.Tensor, temperature: float, top_k: int = 300
    ) -> int:
        assert len(logits.shape) == 1, logits.shape
        V = logits.shape[0]

        top_k = min(top_k, V)

        logits = logits / temperature  # scale by temperature
        values, _ = torch.topk(logits, top_k)  # find top-k values
        logits[logits < values[-1]] = float(
            "-inf"
        )  # set unselected logit's value to -Infinite
        probs = torch.nn.functional.softmax(logits, dim=-1)  # to (normalized) probabilities
        new_token = (
            torch.multinomial(probs, num_samples=1).view(-1).to(torch.int32).item()
        )  # one sample from the multinomial probability distribution
        assert type(new_token) == int, f"{new_token=} {type(new_token)=}"

        return new_token

    @staticmethod
    def _sampling_top_p(
        logits: torch.Tensor, temperature: float, top_p: float = 0.9
    ) -> int:
        assert len(logits.shape) == 1, logits.shape

        assert temperature > 0, temperature
        probs = torch.nn.functional.softmax(logits / temperature, dim=-1)  # to (normalized) probabilities

        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1) # cumulative sum

        # filter the elements whose cumulative sum is larger than top_p
        mask = (probs_sum - probs_sort) > top_p
        probs_sort[mask] = 0.

        # scale so that sum is 1.0
        probs_sort.div_(probs_sort.sum(dim=-1))

        new_token_idx = torch.multinomial(probs_sort, num_samples=1)
        new_token = torch.gather(probs_idx, -1, new_token_idx)

        return new_token.view(-1).to(torch.int32).item()

    def _next(self) -> Iterator[List[str]]:
        L = len(self.tokens)
        V = self.model.hparam.V

        while L < self.Lt:
            logits = self.model(
                torch.tensor(self.tokens, dtype=torch.int32).view(1, -1)
            )  # inference with model
            assert logits.shape == (1, 1, V), f"{logits.shape=} {L=} {V=}"
            new_token = self.sampling(logits.view(-1), self.temperature)  # sampling
            if new_token in [self.tokenizer.EOS, self.tokenizer.BOS]:
                return
            self.tokens.append(new_token)
            L = len(self.tokens)

            yield self.tokens

    def next(self) -> str:
        for _ in self._next():
            yield self.tokenizer.decode(self.tokens)

    def all(self) -> str:
        for _ in self._next():
            pass
        return self.tokenizer.decode(self.tokens)
