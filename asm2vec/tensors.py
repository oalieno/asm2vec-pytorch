import os
import torch
import logging
from pathlib import Path

from asm2vec.train import train, load_model, load_data

logging.basicConfig(level=logging.INFO, format='%(message)s')


def calc_tensors(
        asm_path: str, tensor_path: str, model_path: str, epochs: int, device: str = 'cpu', learning_rate: float = 0.02
) -> list:
    """
    Calculates vector representation of a binary as the mean per column of the vector representations of its assembly
    functions.
    :param asm_path: Path to folder with assembly function in a sub-folder per binary
    :param tensor_path: Path to folder to store the tensors
    :param model_path: Path to the trained model
    :param epochs: Number of epochs
    :param device: 'auto' | 'cuda' | 'cpu'
    :param learning_rate: Learning rate
    :return: List of tensors
    """
    tensors_list = []
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if os.path.isfile(model_path):
        model, tokens = load_model(model_path, device=device)
    else:
        print("No valid model")
        return []

    dir0 = Path(tensor_path)
    if not (os.path.exists(dir0)):
        os.mkdir(dir0)

    if os.path.isdir(asm_path):
        obj = os.scandir(asm_path)
        for entry in obj:
            if entry.is_dir() and os.listdir(entry) and entry.name:
                tensor_file = os.path.join(dir0, entry.name)
                if not (os.path.exists(tensor_file)):
                    functions, tokens_new = load_data([entry])
                    file_count = sum(len(files) for _, _, files in os.walk(entry))
                    tokens.update(tokens_new)
                    logging.info(f"Binary {entry.name}: {file_count} assembly functions")
                    model.update(file_count, tokens.size())
                    model = model.to(device)

                    model = train(
                        functions,
                        tokens,
                        model=model,
                        epochs=epochs,
                        device=device,
                        mode='test',
                        learning_rate=learning_rate
                    )

                    tensor = model.to('cpu').embeddings_f(torch.tensor([list(range(0, file_count))]))
                    tens = torch.squeeze(tensor)
                    if file_count == 1:
                        torch.save(tensor, tensor_file)
                    else:
                        torch.save(tens.mean(0), tensor_file)
                    tensors_list.append(entry.name)

    else:
        logging.info("No valid directory")

    return tensors_list
