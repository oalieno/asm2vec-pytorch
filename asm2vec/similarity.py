import torch

from asm2vec.data import load_data
from asm2vec.model import load_model
from asm2vec.train import train


def cosine_similarity(v1, v2) -> float:
    return (v1 @ v2 / (v1.norm() * v2.norm())).item()


def compare_two(
        data_path_1: str, data_path_2: str, model_path: str, epochs: int = 10, device: str = "cpu",
        learning_rate: float = 0.02
) -> float:
    """This function produces the cosine similarity of a pair of assembly functions
    :param data_path_1: the path to the assembly function no. 1
    :param data_path_2: the path to the assembly function no. 2
    :param model_path: the path to the trained asm2vec model
    :param epochs: the number of epochs for calculating the tensor representations; (Optional, default = 10)
    :param device: 'auto' | 'cuda' | 'cpu' (Optional, default 'cpu')
    :param learning_rate: learning rate; (Optional; default = 0.02)
    :return the cosine similarity value
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model, tokens = load_model(model_path, device=device)
    functions, tokens_new = load_data([data_path_1, data_path_2])
    tokens.update(tokens_new)
    model.update(2, tokens.size())
    model = model.to(device)

    model = train(
        functions,
        tokens,
        model=model,
        epochs=epochs,
        device=device,
        mode="update",
        learning_rate=learning_rate
    )

    v1, v2 = model.to("cpu").embeddings_f(torch.tensor([0, 1]))
    similarity = cosine_similarity(v1, v2)
    print(f"Cosine similarity : {similarity:.6f}")

    return similarity
