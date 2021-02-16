import torch
import click
import asm2vec

@click.command()
@click.option('-i', '--input', 'ipath', help='training data folder', required=True)
@click.option('-o', '--output', 'opath', default='model.pt', help='output model path', show_default=True)
@click.option('-m', '--model', 'mpath', help='load previous trained model path', type=str)
@click.option('-l', '--limit', help='limit the number of functions to be loaded', show_default=True, type=int)
@click.option('-d', '--ebedding-dimension', 'embedding_size', default=100, help='embedding dimension', show_default=True)
@click.option('-b', '--batch-size', 'batch_size', default=1024, help='batch size', show_default=True)
@click.option('-e', '--epochs', default=10, help='training epochs', show_default=True)
@click.option('-n', '--neg-sample-num', 'neg_sample_num', default=25, help='negative sampling amount', show_default=True)
@click.option('-a', '--calculate-accuracy', 'calc_acc', help='whether calculate accuracy ( will be significantly slower )', is_flag=True)
@click.option('-c', '--device', default='auto', help='hardware device to be used: cpu / cuda / auto', show_default=True)
def cli(ipath, opath, mpath, limit, embedding_size, batch_size, epochs, neg_sample_num, calc_acc, device):
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if mpath:
        model, tokens = asm2vec.utils.load_model(mpath, device=device)
        functions, tokens_new = asm2vec.utils.load_data(ipath, limit=limit)
        tokens.update(tokens_new)
        model.update(len(functions), tokens.size())
    else:
        model = None
        functions, tokens = asm2vec.utils.load_data(ipath, limit=limit)

    def callback(context):
        progress = f'{context["epoch"]} | time = {context["time"]:.2f}, loss = {context["loss"]:.4f}'
        if context["accuracy"]:
            progress += f', accuracy = {context["accuracy"]:.4f}'
        print(progress)
        asm2vec.utils.save_model(opath, context["model"], context["tokens"])

    model = asm2vec.utils.train(
        functions,
        tokens,
        model=model,
        embedding_size=embedding_size,
        batch_size=batch_size,
        epochs=epochs,
        neg_sample_num=neg_sample_num,
        calc_acc=calc_acc,
        device=device,
        callback=callback
    )

if __name__ == '__main__':
    cli()
