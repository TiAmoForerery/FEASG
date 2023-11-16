import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, SpGraphAttentionLayer
from layers import Model

class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads,config):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.transformer = Model(config)
        self.dropout = dropout

        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, nhid * nheads, dropout=dropout, alpha=alpha, concat=False)
        #加入一个全连接层
        #self.liner = nn.LazyLinear(1)
        self.fc1 = nn.Linear(77*64, nhid * nheads)
        self.fc2 = nn.Linear(nhid * nheads,nclass)


    def forward(self, x, adj):

        x = self.transformer(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.out_att(x, adj))  #激活函数  是函数的激活函数
        #x = F.log_softmax(x, dim=1)  #按照行做归一化
        #全连接层
        x = x.view([1,-1])
        #x = x.mean(axis = 0)
        x = self.fc1(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        #x = F.softmax(x, dim=1)
        return x


class SpGAT(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, alpha, nheads):
        """Sparse version of GAT."""
        super(SpGAT, self).__init__()
        self.dropout = dropout

        self.attentions = [SpGraphAttentionLayer(nfeat, 
                                                 nhid, 
                                                 dropout=dropout, 
                                                 alpha=alpha, 
                                                 concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = SpGraphAttentionLayer(nhid * nheads, 
                                             nclass, 
                                             dropout=dropout, 
                                             alpha=alpha, 
                                             concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.out_att(x, adj))
        x = F.log_softmax(x, dim=1)

        return x

