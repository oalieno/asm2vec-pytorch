import torch
import torch.nn as nn
import click
import asm2vec

@click.command()
@click.option('-i', '--input', 'ipath', help='target function', required=True)
@click.option('-m', '--model', 'mpath', help='model path', required=True)
@click.option('-e', '--epochs', default=10, help='training epochs', show_default=True)
@click.option('-c', '--device', default='auto', help='hardware device to be used: cpu / cuda / auto', show_default=True)
def cli(ipath, mpath, epochs, device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load tokens, model
    model, tokens = asm2vec.utils.load_model(mpath)
    # reset model function embedding
    model.embeddings_f = nn.Embedding(1, 2 * model.embeddings.embedding_dim)

    # train function embedding
    functions, _ = asm2vec.utils.load_data(ipath)
    model = asm2vec.utils.train(functions, tokens, model=model, epochs=epochs, device=device, mode='test')

    # show predict probabilities
    x, y = asm2vec.utils.preprocess(functions, tokens)
    probs = model.predict(x, y)
    asm2vec.utils.show_probs(x, y, probs, tokens)

if __name__ == '__main__':
    cli()
