import os
import time
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from asm2vec.model import ASM2VEC
from asm2vec.datatype import Tokens, Function, Instruction

logging.basicConfig(level=logging.INFO, format='%(message)s')


class AsmDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def load_data(paths, limit=None):
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


def preprocess(functions, tokens):
    x, y = [], []
    for i, fn in enumerate(functions):
        for seq in fn.random_walk():
            for j in range(1, len(seq) - 1):
                x.append([i] + [tokens[token].index for token in seq[j - 1].tokens() + seq[j + 1].tokens()])
                y.append([tokens[token].index for token in seq[j].tokens()])
    return torch.tensor(x), torch.tensor(y)


def train(
        functions,
        tokens,
        model=None,
        embedding_size=100,
        batch_size=1024,
        epochs=10,
        neg_sample_num=25,
        calc_acc=False,
        device='cpu',
        mode='train',
        callback=None,
        learning_rate=0.02
):
    if mode == 'train':
        if model is None:
            model = ASM2VEC(tokens.size(), function_size=len(functions), embedding_size=embedding_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif mode == 'test':
        if model is None:
            raise ValueError("test mode required pretrained model")
        optimizer = torch.optim.Adam(model.embeddings_f.parameters(), lr=learning_rate)
    else:
        raise ValueError("Unknown mode")

    loader = DataLoader(AsmDataset(*preprocess(functions, tokens)), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        start = time.time()
        loss_sum, loss_count, accs = 0.0, 0, []

        model.train()
        for i, (inp, pos) in enumerate(loader):
            neg = tokens.sample(inp.shape[0], neg_sample_num)
            loss = model(inp.to(device), pos.to(device), neg.to(device))
            loss_sum, loss_count = loss_sum + loss, loss_count + 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0 and calc_acc:
                probs = model.predict(inp.to(device), pos.to(device))
                accs.append(accuracy(pos, probs))

        if callback:
            callback({
                'model': model,
                'tokens': tokens,
                'epoch': epoch,
                'time': time.time() - start,
                'loss': loss_sum / loss_count,
                'accuracy': torch.tensor(accs).mean() if calc_acc else None
            })

    return model


def save_model(path, model, tokens):
    torch.save({
        'model_params': (
            model.embeddings.num_embeddings,
            model.embeddings_f.num_embeddings,
            model.embeddings.embedding_dim
        ),
        'model': model.state_dict(),
        'tokens': tokens.state_dict(),
    }, path)


def load_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    tokens = Tokens()
    tokens.load_state_dict(checkpoint['tokens'])
    model = ASM2VEC(*checkpoint['model_params'])
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    return model, tokens


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


def train_asm2vec_model(
        train_set: str,
        new_model: str,
        model_path: str | None,
        epochs: int,
        limit: int = None,
        calc_acc: bool = False,
        embedding_size: int = 100,
        batch_size: int = 1024,
        neg_sample: int = 25,
        learning_rate: float = 0.02,
        device: str = 'cpu'
) -> ASM2VEC:

    """Trains an asm2vec model
    :param train_set: path to the training dataset
    :param new_model: path to the model to be trained
    :param model_path: path to already trained model
    :param limit: number of the assembly functions that the model will be trained on;
    if not defined, all the assembly functions in train_set_path
    :param epochs: number of epochs
    :param calc_acc: displays the accuracy per training epoch; setting it to True will slow down the training
    :param embedding_size: size of the vector representation for a token; an assembly function
    will be represented with a vector twice that size
    :param batch_size: the size of batches for training
    :param neg_sample: negative sampling amount
    :param device: 'auto' | 'cuda' | 'cpu'
    :param learning_rate: learning rate
    """

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if model_path:
        model, tokens = load_model(model_path, device=device)
        functions, tokens_new = load_data(train_set, limit=limit)
        tokens.update(tokens_new)
        model.update(len(functions), tokens.size())
    else:
        model = None
        functions, tokens = load_data(Path(train_set), limit=limit)

    model = train(
        functions,
        tokens,
        model=model,
        embedding_size=embedding_size,
        batch_size=batch_size,
        epochs=epochs,
        neg_sample_num=neg_sample,
        calc_acc=calc_acc,
        device=device,
        callback=callback,
        learning_rate=learning_rate
    )
    save_model(new_model, model, tokens)

    return model
