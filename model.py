#!/usr/bin/env python
# coding: utf-8

# In[1]:


from typing import Optional, List

from uawa_att import *


class XfModel(nn.Module):
    def __init__(self, branches: Optional[List[List]] = None, hidden_size=128, alpha=0.1):
        """

        :param branches: showing classification targets, e.g.[[0,1], [1,3]]means two sub classification targeting on
                        ([0,1], [2,3]) and ([1,3], [0,2])
        :param hidden_size: LSTM hidden size
        """
        super(XfModel, self).__init__()
        # 输入[40,1,128,301]
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1))
        # 输出[40,16,65,151]
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1))
        # 输出[40,32,33,76]
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1))
        # 输出[40,48,17,39]
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, padding=1))
        # 输出[40,48,9,20]
        self.dropout = nn.Dropout(0.5)
        # LSTM(特征尺度，隐藏层），输入是[batch,时序，特征]，输出是[batch,len，hidden_size*2]
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.branches = branches
        self.LSTMs = nn.ModuleDict()
        self.FCs = nn.ModuleDict()
        self.attentions = nn.ModuleDict()
        self.LSTMs['emotion'] = nn.GRU(256, self.hidden_size, batch_first=True, bidirectional=True)
        self.branch_num = len(self.branches) if self.branches is not None else 0
        self.FCs['emotion'] = nn.Linear(self.hidden_size * 2 * (1 + self.branch_num), 4)
        self.attentions['emotion'] = SelfAttention(nhid=hidden_size, attn_units=64, attn_hops=1, drop_low_weight=False)
        if self.branch_num != 0:
            for i in range(self.branch_num):
                lstm_layer = nn.GRU(256, self.hidden_size, batch_first=True, bidirectional=True)
                fc_layer = nn.Linear(self.hidden_size * 2, 2)
                attention_layer = SelfAttention(nhid=hidden_size, attn_units=64, attn_hops=1, drop_low_weight=True)
                self.LSTMs['%s' % i] = lstm_layer
                self.FCs['%s' % i] = fc_layer
                self.attentions['%s' % i] = attention_layer

    def forward(self, x, lengths, is_train=False, labels=None):
        lengths = lengths.clone().detach()
        x = x.view(x.shape[0], 1, x.shape[1], x.shape[2])
        if is_train:
            loss = 0
            if self.branch_num != 0:
                sub_labels = []
                for i in range(self.branch_num):
                    sub_label = torch.zeros(labels.shape).cuda()
                    for cls in self.branches[i]:
                        sub_label = torch.logical_or(sub_label, labels == cls)
                    sub_label = sub_label.long()
                    sub_labels.append(sub_label)

        out = self.conv1(x)
        lengths = lengths // 2 + 1
        out = self.conv2(out)
        lengths = lengths // 2 + 1
        out = self.conv3(out)
        lengths = lengths // 2 + 1
        out = self.conv4(out)

        out = self.dropout(out)

        lengths = lengths // 2 + 1
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, out.size()[3]))
        # 把通道和特征维度相乘，LSTM的输入是（batch，时序，特征），之后交换特征和时序位置
        out = out.view(out.size()[0], out.size()[1] * out.size()[2], -1)
        out = out.transpose(1, 2).contiguous()
        out = torch.nn.utils.rnn.pack_padded_sequence(out, lengths.cpu(), batch_first=True)  # ready for lstm
        features = []
        if self.branch_num != 0:
            for i in range(self.branch_num):
                branch_lstm_out, branch_fc_out = self.branch_forward(out, lengths, key='%s' % i)
                features.append(branch_lstm_out)
                if is_train:
                    loss += self.alpha * nn.CrossEntropyLoss()(branch_fc_out, sub_labels[i])
        _, fc_out = self.branch_forward(out, lengths, 'emotion', features=features)
        if is_train:
            loss += nn.CrossEntropyLoss()(fc_out, labels)
            return fc_out, loss
        else:
            return fc_out, 0

    def branch_forward(self, lstm_input, lengths, key, features=None):
        """

        :param lstm_input:
        :param lengths:
        :param key:used to find branch layer in nn.ModuleDict
        :param features: used only in main branch, when key=='emotion'
        :return: tuple of (lstm output, fc output)
        """
        out, (hn, cn) = self.LSTMs[key](lstm_input)

        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (torch.arange(lengths[0]).cuda()[None, :] < lengths[:, None]).float()
        res, alpha = self.attentions[key](out, lengths, mask)
        res = res.squeeze(1)
        if key == 'emotion':
            features.append(res)
            res = torch.cat(features, dim=1)
        out = self.FCs[key](res)
        return res, out
