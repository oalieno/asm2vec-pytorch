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
    # TODO - doc string
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model, tokens
    model, tokens = load_model(model_path, device=device)
    functions, tokens_new = load_data([data_path_1, data_path_2])
    tokens.update(tokens_new)
    model.update(2, tokens.size())
    model = model.to(device)
    
    # train function embedding
    model = train(
        functions,
        tokens,
        model=model,
        epochs=epochs,
        device=device,
        mode="test",
        learning_rate=learning_rate
    )

    # compare 2 function vectors
    v1, v2 = model.to("cpu").embeddings_f(torch.tensor([0, 1]))
    similarity = cosine_similarity(v1, v2)
    print(f"cosine similarity : {similarity:.6f}")
    return similarity
