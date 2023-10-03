import torch
import asm2vec
import logging
from pathlib import Path
from asm2vec import utils
from asm2vec.model import ASM2VEC

logging.basicConfig(level=logging.INFO, format='%(message)s')


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
        limit: int,
        epochs: int,
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
        model, tokens = utils.load_model(model_path, device=device)
        functions, tokens_new = utils.load_data(train_set, limit=limit)
        tokens.update(tokens_new)
        model.update(len(functions), tokens.size())
    else:
        model = None
        functions, tokens = utils.load_data(Path(train_set), limit=limit)

    model = utils.train(
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
    utils.save_model(new_model, model, tokens)

    return model
