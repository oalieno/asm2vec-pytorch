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
    """This function trains a model on the given assembly functions and tokens
    :param functions: list of assembly functions
    :param tokens: tokens (operations, operands) of the assembly function
    :param model: type of the model; ; (Optional, default ASM2VEC)
    :param embedding_size: size of the tensor representation of an assembly function; (Optional, default value = 100)
    :param batch_size: size of the batch for each epoch of training; (Optional, default value = 1024)
    :param epochs: number of epochs for training the model; (Optional, default value = 10)
    :param neg_sample_num: size of the negative sample; (Optional, default value = 25)
    :param calc_acc: if set to True, the accuracy per training epoch is displayed; (Optional, default False)
    :param device: the device used for processing; (Optional, default 'cpu')
    :param mode: 'train' (to train a new model) | 'update' (to add to an already trained  model's dictionary);
    (Optional, default 'train')
    :param verbose: if True performs training in verbose mode; (Optional, default False)
    :param learning_rate: learning rate
    """
    if mode == 'train':
        if model is None:
            model = ASM2VEC(tokens.size(), function_size=len(functions), embedding_size=embedding_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif mode == 'update':
        if model is None:
            raise ValueError("Update mode requires a pretrained model")
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
    :return an ASM2VEC model
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
