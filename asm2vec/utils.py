import os
import time
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from .datatype import Tokens, Function, Instruction
from .model import ASM2VEC

def load_data(path, limit=None):
    if os.path.isdir(path):
        filenames = [Path(path) / filename for filename in os.listdir(path) if os.path.isfile(Path(path) / filename)]
    else:
        filenames = [Path(path)]
    
    functions, tokens = [], Tokens()
    for i, filename in enumerate(filenames):
        with open(filename) as f:
            fn = Function.load(f.read())
            functions.append(fn)
            tokens.add(fn.tokens())
        if limit and i >= limit:
            break
    
    return functions, tokens

def preprocess(functions, tokens):
    x, y = [], []
    for i, fn in enumerate(functions):
        for seq in fn.random_walk():
            for j in range(1, len(seq) - 1):
                x.append([i] + [tokens[token].index for token in seq[j-1].tokens() + seq[j+1].tokens()])
                y.append([tokens[token].index for token in seq[j].tokens()])
    return torch.tensor(x), torch.tensor(y)

class AsmDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index]

def train(functions, tokens, model=None, embedding_size=100, batch_size=1024, epochs=10, neg_sample_num=25, device='cpu', mode='train'):
    if mode == 'train':
        if model is None:
            model = ASM2VEC(tokens.size(), function_size=len(functions), embedding_size=embedding_size).to(device)
        optimizer = torch.optim.Adam(model.parameters())
    elif mode == 'test':
        if model is None:
            raise ValueError("test mode required pretrained model")
        optimizer = torch.optim.Adam(model.embeddings_f.parameters())
    else:
        raise ValueError("Unknown mode")

    loader = DataLoader(AsmDataset(*preprocess(functions, tokens)), batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        start = time.time()
        loss_sum, loss_count = 0.0, 0

        model.train()
        for i, (inp, pos) in enumerate(loader):
            neg = tokens.sample(inp.shape[0], neg_sample_num)
            loss = model(inp.to(device), pos.to(device), neg.to(device))
            loss_sum, loss_count = loss_sum + loss, loss_count + 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'{epoch} | time = {time.time() - start:.2f}, loss = {loss_sum / loss_count:.4f}')

    return model

def save_model(path, model, tokens):
    torch.save({
        'model_params': (
            model.embeddings.num_embeddings,
            model.embeddings_f.num_embeddings,
            model.embeddings.embedding_dim
        ),
        'model': model.state_dict(),
        'tokens': tokens.state_dict(),
    }, path)

def load_model(path):
    checkpoint = torch.load(path)
    tokens = Tokens()
    tokens.load_state_dict(checkpoint['tokens'])
    model = ASM2VEC(*checkpoint['model_params'])
    model.load_state_dict(checkpoint['model'])
    return model, tokens

def show_probs(x, y, probs, tokens):
    top = probs.topk(5)
    for i, (xi, yi) in enumerate(zip(x, y)):
        xi, yi = xi.tolist(), yi.tolist()
        print('┌' + '─' * 42 + '┐')
        print(f'│ {str(Instruction(tokens[xi[1]], tokens[xi[2:4]])):40} │')
        print(f'│ {str(Instruction(tokens[yi[0]], tokens[yi[1:3]])):40} │')
        print(f'│ {str(Instruction(tokens[xi[4]], tokens[xi[5:7]])):40} │')
        print('├' + '─' * 8 + '┬' + '─' * 33 + '┤')
        for value, index in zip(top.values[i], top.indices[i]):
            if index in yi:
                colorbegin, colorclear = '\033[96m', '\033[0m'
            else:
                colorbegin, colorclear = '', ''
            print(f'│ {colorbegin}{value*100:05.2f}%{colorclear} │ {colorbegin}{tokens[index.item()].name:31}{colorclear} │')
        print('└' + '─' * 8 + '┴' + '─' * 33 + '┘')
