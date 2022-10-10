import numpy as np
import torch
import torch.nn as nn


class SelfAttention(nn.Module):

    def __init__(self, nhid, attn_units, attn_hops, drop_low_weight=False, dropout=0.1):
        """
        :param nhid: lstm hidden size
        :param attn_units: project hidden size * 2 to this
        :param attn_hops: num of attention head
        :param drop_low_weight: set attn weights below the mean weight to zero
        :param dropout:
        """
        super(SelfAttention, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.ws1 = nn.Linear(nhid * 2, attn_units, bias=False)
        self.ws2 = nn.Linear(attn_units, attn_hops, bias=False)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.drop_low = drop_low_weight
        self.softmax = nn.Softmax(dim=-1)
        #        self.init_weights()
        self.attention_hops = attn_hops

    def forward(self, inp, lengths=None, mask=None):
        size = inp.size()  # [bsz, len, nhid*2]
        compressed_embeddings = inp.contiguous().view(-1, size[2])  # [bsz*len, nhid*2]

        hbar = self.relu(self.ws1(self.drop(compressed_embeddings)))  # [bsz*len, attn_units]
        alphas = self.ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        if lengths is None:
            # batch size, 1, 1
            lengths = torch.tensor([inp.size[1]]).repeat(inp.size[0]).unsqueeze(1).unsqueeze(2).cuda()
            mean_weight = 1 / lengths
        else:
            mean_weight = (1/lengths).unsqueeze(1).unsqueeze(2)
        if mask is not None:
            alphas = alphas.masked_fill(~mask.unsqueeze(1).bool(), -np.inf)
        alphas = self.softmax(alphas)  # 归一化
        if self.drop_low:
            mask_low_weight = alphas < mean_weight
            alphas = alphas.masked_fill(mask_low_weight.bool(), -np.inf)
            alphas = self.softmax(alphas)  # 再归一化
        return torch.bmm(alphas, inp), alphas

