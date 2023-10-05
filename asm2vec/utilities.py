import logging
import torch

from asm2vec.datatype import Instruction

logging.basicConfig(level=logging.INFO, format='%(message)s')


# TODO - Why do we have both logging and print?
# TODO - Doc strings

def accuracy(y, probs):
    return torch.mean(torch.tensor([torch.sum(probs[i][yi]) for i, yi in enumerate(y)]))


def callback(context) -> None:
    """Prettifies the display of accuracy, if chosen
    """
    progress = f'{context["epoch"]} | time = {context["time"]:.2f},\
                  loss = {context["loss"]:.4f}'

    if context["accuracy"]:
        progress += f', accuracy = {context["accuracy"]:.4f}'
    logging.info(f"{progress}")


def show_probs(x, y, probs, tokens, limit=None, pretty=False):
    if pretty:
        tl, tr, bl, br = '┌', '┐', '└', '┘'
        lm, rm, tm, bm = '├', '┤', '┬', '┴'
        h, v = '─', '│'
        arrow = ' ➔'
    else:
        tl, tr, bl, br = '+', '+', '+', '+'
        lm, rm, tm, bm = '+', '+', '+', '+'
        h, v = '-', '|'
        arrow = '->'
    top = probs.topk(5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if limit and i >= limit:
            break
        xi, yi = xi.tolist(), yi.tolist()
        print(tl + h * 42 + tr)
        print(f'{v}    {str(Instruction(tokens[xi[1]], tokens[xi[2:4]])):37} {v}')
        print(f'{v} {arrow} {str(Instruction(tokens[yi[0]], tokens[yi[1:3]])):37} {v}')
        print(f'{v}    {str(Instruction(tokens[xi[4]], tokens[xi[5:7]])):37} {v}')
        print(lm + h * 8 + tm + h * 33 + rm)
        for value, index in zip(top.values[i], top.indices[i]):
            if index in yi:
                colorbegin, colorclear = '\033[92m', '\033[0m'
            else:
                colorbegin, colorclear = '', ''
            print(f'{v} {colorbegin}{value * 100:05.2f}%{colorclear} {v} {colorbegin}'
                  f'{tokens[index.item()].name:31}{colorclear} {v}')
        print(bl + h * 8 + bm + h * 33 + br)
