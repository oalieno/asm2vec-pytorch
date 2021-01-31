import torch
import torch.nn as nn

bce, sigmoid, softmax = nn.BCELoss(), nn.Sigmoid(), nn.Softmax(dim=1)

class ASM2VEC(nn.Module):
    def __init__(self, tokens, vocab_size, function_size, embedding_size, neg_sample_num=25):
        super(ASM2VEC, self).__init__()
        self.tokens = tokens
        self.neg_sample_num = neg_sample_num
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.embeddings_f = nn.Embedding(function_size, 2 * embedding_size)
        self.embeddings_r = nn.Embedding(vocab_size, 2 * embedding_size)
        
    def v(self, inp):
        e  = self.embeddings(inp[:,1:])
        v_f = self.embeddings_f(inp[:,0])
        v_prev = torch.cat([e[:,0], (e[:,1] + e[:,2]) / 2], dim=1)
        v_next = torch.cat([e[:,3], (e[:,4] + e[:,5]) / 2], dim=1)
        v = ((v_f + v_prev + v_next) / 3).unsqueeze(2)
        return v

    def forward(self, inp, pos):
        device, batchsize = inp.device, inp.shape[0]
        v = self.v(inp)
        # negative sampling loss
        neg = self.tokens.sample(batchsize, self.neg_sample_num).to(device)
        pred = torch.bmm(self.embeddings_r(torch.cat([pos, neg], dim=1)), v).squeeze()
        label = torch.cat([torch.ones(batchsize, 3), torch.zeros(batchsize, self.neg_sample_num)], dim=1).to(device)
        return bce(sigmoid(pred), label)
