import torch
import click
import asm2vec
import json

@click.command()
@click.option('-m', '--model', 'mpath', help='model path', required=True)
@click.option('-c', '--device', default='auto', help='hardware device to be used: cpu / cuda / auto', show_default=True)
@click.option('-o', '--output', 'out', help='output file path that contains vectors', required=True)
def cli(mpath, device, out):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load model, tokens
    model, token, fn_mapper = asm2vec.utils.load_model(mpath, device=device)
    with open(out, 'w+') as f:
        json.dump({'embeddings': model.to('cpu').embeddings_f.weight.data.numpy().tolist(), 'function_mapper': fn_mapper.state_dict()}, f)

if __name__ == '__main__':
    cli()
