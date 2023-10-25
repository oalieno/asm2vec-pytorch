import os
from pathlib import Path
from torch.utils.data import Dataset

from asm2vec.datatype import Tokens, Function


class AsmDataset(Dataset):
    # TODO - doc string - explain what this class does - how does it extend `Dataset`?
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def load_data(paths, limit=None):
    # TODO - doc string
    if type(paths) is not list:
        paths = [paths]

    filenames = []
    for path in paths:
        if os.path.isdir(path):
            filenames += [Path(path) / filename for filename in sorted(os.listdir(path))
                          if os.path.isfile(Path(path) / filename)]
        else:
            filenames += [Path(path)]

    functions, tokens = [], Tokens()
    for i, filename in enumerate(filenames):
        if limit and i >= limit:
            break
        with open(filename) as f:
            fn = Function.load(f.read())
            functions.append(fn)
            tokens.add(fn.tokens())

    return functions, tokens
