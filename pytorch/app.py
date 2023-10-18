import sys

import fire
from loguru import logger

from tokenizer import load_tokenizer
from model import load_model
from generation import Generator


def tinystories(model_size: str = "110M") -> None:
    model_name = "tinystories" + model_size
    logger.debug(f"{model_size=}")

    tkr = load_tokenizer(model_name)
    mdl = load_model(model_name)
    gnr = Generator(tkr, mdl)

    print(f"------ {model_name} ------")
    cur_pos = 0
    for new_string in gnr.next():
        print(new_string[cur_pos:], end="")
        sys.stdout.flush()
        cur_pos = len(new_string)
    print()
    print("------- the end ------")


if __name__ == "__main__":
    fire.Fire()
