import torch
import click
import asm2vec

@click.command()
@click.option('-i', '--input', 'ipath', help='training data folder', required=True)
@click.option('-o', '--output', 'opath', default='./model.pt', help='output model path', show_default=True)
@click.option('-d', '--ebedding-dimension', 'embedding_size', default=100, help='embedding dimension', show_default=True)
@click.option('-b', '--batch-size', 'batch_size', default=1024, help='batch size', show_default=True)
@click.option('-e', '--epochs', default=10, help='training epochs', show_default=True)
@click.option('-n', '--neg-sample-num', 'neg_sample_num', default=25, help='negative sampling amount', show_default=True)
@click.option('-c', '--device', default='auto', help='hardware device to be used: cpu / cuda / auto', show_default=True)
def cli(ipath, opath, embedding_size, batch_size, epochs, neg_sample_num, device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    functions, tokens = asm2vec.utils.load_data(ipath)
    model = asm2vec.utils.train(
        functions,
        tokens,
        embedding_size=embedding_size,
        batch_size=batch_size,
        epochs=epochs,
        neg_sample_num=neg_sample_num,
        device=device
    )

    asm2vec.utils.save_model(opath, model, tokens)

if __name__ == '__main__':
    cli()
