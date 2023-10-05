import time
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from asm2vec.data import AsmDataset, load_data
from asm2vec.datatype import Function, Tokens
from asm2vec.model import ASM2VEC, load_model, save_model
from asm2vec.utilities import accuracy, callback


def preprocess(functions, tokens):
    x, y = [], []
    for i, fn in enumerate(functions):
        for seq in fn.random_walk():
            for j in range(1, len(seq) - 1):
                x.append([i] + [tokens[token].index for token in seq[j - 1].tokens() + seq[j + 1].tokens()])
                y.append([tokens[token].index for token in seq[j].tokens()])
    return torch.tensor(x), torch.tensor(y)


def train(
        functions: list[Function], tokens: Tokens, model: ASM2VEC | None = None, embedding_size: int = 100,
        batch_size: int = 1024, epochs: int = 10, neg_sample_num: int = 25, calc_acc: bool = False, device: str = 'cpu',
        mode: str = 'train', verbose: bool = False, learning_rate: float = 0.02
):
    # TODO: doc string
    # TODO: test mode in train... this is confusing!
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

        if verbose:
            callback({
                'model': model,
                'tokens': tokens,
                'epoch': epoch,
                'time': time.time() - start,
                'loss': loss_sum / loss_count,
                'accuracy': torch.tensor(accs).mean() if calc_acc else None
            })

    return model


def train_asm2vec_model(
        train_set: str, new_model: str, model_path: str | None, epochs: int, limit: int | None = None,
        calc_acc: bool = False, embedding_size: int = 100, batch_size: int = 1024, neg_sample: int = 25,
        learning_rate: float = 0.02, device: str = 'cpu'
) -> ASM2VEC:
    # TODO - this is just a wrapper - can we do this smarter?
    """Trains an ASM2VEC model
    :param train_set: path to the training dataset
    :param new_model: path to the model to be trained
    :param model_path: path to already trained model
    :param limit: number of the assembly functions that the model will be trained on; if not defined, all the assembly
        functions in train_set_path
    :param epochs: number of epochs
    :param calc_acc: displays the accuracy per training epoch; setting it to True will slow down the training
    :param embedding_size: size of the vector representation for a token; an assembly function will be represented
        with a vector twice that size
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
        verbose=True,
        learning_rate=learning_rate
    )
    save_model(new_model, model, tokens)

    return model
