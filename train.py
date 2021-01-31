import os
import click
import time
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from asm2vec import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class AsmDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def preprocess(ipath, limit=None):
    tokens, functions = Tokens(), []
    for i, filename in enumerate(os.listdir(ipath)):
        with open(Path(ipath) / filename) as f:
            fn = Function.load(f.read())
            functions.append(fn)
            tokens.add(fn.tokens())
        if limit and i >= limit:
            break

    x, y = [], []
    for i, fn in enumerate(functions):
        for seq in fn.random_walk():
            for j in range(1, len(seq) - 1):
                x.append([i] + [tokens[token].index for token in seq[j-1].tokens() + seq[j+1].tokens()])
                y.append([tokens[token].index for token in seq[j].tokens()])
    x, y = torch.tensor(x), torch.tensor(y)

    return x, y, tokens, functions

def train(ipath, epochs, batch_size=1024):
    x, y, tokens, functions = preprocess(ipath)
    loader = DataLoader(AsmDataset(x, y), batch_size=batch_size, shuffle=True)
    
    model = ASM2VEC(tokens, vocab_size=tokens.size(), function_size=len(functions), embedding_size=100).to(device)
    optimizer = torch.optim.Adam(model.parameters())

    for epoch in range(epochs):
        start = time.time()
        loss_sum, loss_count = 0.0, 0

        model.train()
        for i, (inp, pos) in enumerate(loader):
            loss = model(inp.to(device), pos.to(device))
            loss_sum, loss_count = loss_sum + loss, loss_count + 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'{epoch} | time = {time.time() - start:.2f}, loss = {loss_sum / loss_count:.4f}')

    return model

@click.command()
@click.option('-i', '--input', 'ipath', help='training data folder', required=True)
@click.option('-o', '--output', 'opath', default='./model.pt', help='output model path')
@click.option('-e', '--epoch', default=10, help='training epoch')
def cli(ipath, opath, epoch):
    model = train(ipath, epoch)
    torch.save(model.state_dict(), opath)

if __name__ == '__main__':
    cli()
