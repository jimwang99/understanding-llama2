# convert trained weight from llama2 and tinyllamas to our DIY model

import torch

from loguru import logger
from collections import OrderedDict


def export_tinyllamas(fpath: str) -> OrderedDict:
    """
    >>> dt = export_tinyllamas("../third-party/tinyllamas/stories260K.pt")
    """
    assert fpath.endswith(".pt"), fpath
    logger.debug(f"{fpath=}")

    mi = torch.load(fpath)
    state_dict = mi["model"]
    new_state_dict = OrderedDict()
    for name, weight in state_dict.items():
        new_name = "^" + name[10:] if name.startswith("_orig_mod.") else "^" + name

        mapping = {
            "^tok_embeddings.weight": "embedding.weight",
            ".attention.wq": ".projection_q",
            ".attention.wk": ".projection_k",
            ".attention.wv": ".projection_v",
            ".attention.wo": ".projection_a",
            ".feed_forward.w1": ".projection_gate",
            ".feed_forward.w2": ".projection_down",
            ".feed_forward.w3": ".projection_up",
            "^norm.": "output_norm.",
            "^output.": "output_linear.",
        }

        for key, value in mapping.items():
            new_name = new_name.replace(key, value)

        new_name = new_name[1:] if new_name.startswith("^") else new_name

        new_state_dict[new_name] = weight

        logger.trace(f"    {name} -> {new_name}")

    return new_state_dict
