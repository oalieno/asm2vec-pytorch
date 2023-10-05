import torch

from asm2vec.data import load_data
from asm2vec.model import load_model
from asm2vec.train import train, preprocess
from asm2vec.utilities import show_probs


def test_model(
        data_path: str, model_path: str, epochs: int = 10, neg_sample_num: int = 25, limit: int | None = None,
        device: str = "cpu", learning_rate: float = 0.02, pretty: bool = False
) -> None:
    # TODO - doc string
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model, tokens
    model, tokens = load_model(model_path, device=device)
    functions, tokens_new = load_data(data_path)
    tokens.update(tokens_new)
    model.update(1, tokens.size())
    model = model.to(device)

    # train function embedding
    model = train(
        functions,
        tokens,
        model=model,
        epochs=epochs,
        neg_sample_num=neg_sample_num,
        device=device,
        mode="test",
        learning_rate=learning_rate
    )

    # show predicted probability results
    x, y = preprocess(functions, tokens)
    probs = model.predict(x.to(device), y.to(device))
    show_probs(x, y, probs, tokens, limit=limit, pretty=pretty)
