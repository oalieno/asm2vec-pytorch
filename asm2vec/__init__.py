import torch
import random
from .model import ASM2VEC

class Token:
    def __init__(self, name, index):
        self.name = name
        self.index = index
        self.count = 1

class Tokens:
    def __init__(self):
        self.tokens = {}
        self.tokens_list = []
        self.total = 0
        self._weights = None
    def __getitem__(self, key):
        return self.tokens[key]
    def size(self):
        return len(self.tokens)
    def add(self, names):
        self._weights = None
        if type(names) is not list:
            names = [names]
        for name in names:
            if name not in self.tokens:
                self.tokens[name] = Token(name, len(self.tokens))
                self.tokens_list.append(self.tokens[name])
            else:
                self.tokens[name].count += 1
                self.total += 1
    def weights(self):
        # if no cache, calculate
        if self._weights is None:
            self._weights = torch.zeros(len(self.tokens))
            for token in self.tokens.values():
                self._weights[token.index] = (token.count / self.total) ** 0.75
        return self._weights
    def sample(self, batch_size, num=5):
        return torch.multinomial(self.weights(), num * batch_size, replacement=True).view(batch_size, num)

class Function:
    def __init__(self, insts, blocks):
        self.insts = insts
        self.blocks = blocks
        self.meta = {}
    @classmethod
    def load(cls, text):
        '''
        gcc -S format compatiable
        '''
        label, labels, insts, blocks, meta = None, {}, [], [], {}
        for line in text.strip('\n').split('\n'):
            if line[0] in [' ', '\t']:
                line = line.strip()
                # meta data
                if line[0] == '.':
                    key, _, value = line[1:].strip().partition(' ')
                    meta[key] = value
                # instruction
                else:
                    inst = Instruction.load(line)
                    insts.append(inst)
                    if len(blocks) == 0 or blocks[-1].end():
                        blocks.append(BasicBlock())
                        # link prev and next block
                        if len(blocks) > 1:
                            blocks[-2].successors.add(blocks[-1])
                    if label:
                        labels[label], label = blocks[-1], None
                    blocks[-1].add(inst)
            # label
            else:
                label = line.partition(':')[0]
        # link label
        for block in blocks:
            inst = block.insts[-1]
            if inst.is_jmp() and labels.get(inst.args[0]):
                block.successors.add(labels[inst.args[0]])
        # replace label with CONST
        for inst in insts:
            for i, arg in enumerate(inst.args):
                if labels.get(arg):
                    inst.args[i] = 'CONST'
        return cls(insts, blocks)
    def tokens(self):
        return [token for inst in self.insts for token in inst.tokens()]
    def random_walk(self, num=3):
        return [self._random_walk() for _ in range(num)]
    def _random_walk(self):
        current, visited, seq = self.blocks[0], [], []
        while current not in visited:
            visited.append(current)
            seq += current.insts
            # no following block / hit return
            if len(current.successors) == 0 or current.insts[-1].op == 'ret':
                break
            current = random.choice(list(current.successors))
        return seq

class BasicBlock:
    def __init__(self):
        self.insts = []
        self.successors = set()
    def add(self, inst):
        self.insts.append(inst)
    def end(self):
        inst = self.insts[-1]
        return inst.is_jmp() or inst.op == 'ret'

class Instruction:
    def __init__(self, op, args):
        self.op = op
        self.args = args
    def __str__(self):
        return f'{self.op} {", ".join([arg for arg in self.args if arg])}'
    @classmethod
    def load(cls, text):
        text = text.strip().strip('bnd').strip() # get rid of BND prefix
        op, _, args = text.strip().partition(' ')
        if args:
            args = [arg.strip() for arg in args.split(',')]
        else:
            args = []
        args = (args + ['', ''])[:2]
        return cls(op, args)
    def tokens(self):
        return [self.op] + self.args
    def is_jmp(self):
        if 'jmp' in self.op or self.op[0] == 'j':
            return True
        else:
            return False
    def is_call(self):
        return self.op == 'call'

__all__ = ['Token', 'Tokens', 'Function', 'BasicBlock', 'Instruction', 'ASM2VEC']
