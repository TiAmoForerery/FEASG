from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import scipy.sparse as sp
import utils
from utils import load_data, accuracy
from models import GAT, SpGAT
from layers import Config


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=10000, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.00001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.01, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
#训练集
filenames_train_1 = os.listdir('./data/camel1.6/train/isbug') #153个文件
filenames_train_0 = os.listdir('./data/camel1.6/train/nobug') #290个文件
#测试集
# filenames_train_1 = os.listdir('./data/test/isbug') #150个文件
# filenames_train_0 = os.listdir('./data/test/nobug') #150个文件
#adj, features, labels, idx_train, idx_val, idx_test = load_data()
features_1, adj_1 = utils.list_features_adj(filenames_train_1, 1, 'train')
features_0, adj_0 = utils.list_features_adj(filenames_train_0, 0, 'train')
features = features_1 + features_0

#features = torch.tensor(features)

adjs = adj_1 + adj_0
labels = []
#训练用
for i in range(442):
    if i < 152:
        labels.append([1])
    else:
        labels.append([0])
#生成随机序列 1254的随机序列
import random
s = [i for i in range(0, 442)]
r = random.shuffle(s)
shf_features = []
shf_adjs = []
shf_labels = []
for j in s:
    shf_features.append(features[j])
    shf_adjs.append(adjs[j])
    shf_labels.append(labels[j])



# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=features.shape[1], 
                nhid=args.hidden, 
                #nclass=int(labels.max()) + 1,
                nclass=7,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha,
                config = Config()  )
else:
    model = GAT(nfeat=300,
                nhid=args.hidden, 
                #nclass=int(labels.max()) + 1,
                nclass=2,
                dropout=args.dropout, 
                nheads=args.nb_heads, 
                alpha=args.alpha,
                config = Config())
optimizer = optim.Adam(model.parameters(), 
                       lr=args.lr, 
                       weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()


def train(feature, adj, label):
    model.train()
    optimizer.zero_grad()
    output = model(feature, adj)
    loss_train = F.nll_loss(output, label)
    loss_train.backward()
    optimizer.step()
    predict = torch.max(output, 1)[1].numpy()

    return predict



def compute_test(features,adj,labels):
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output, labels)
    predict = torch.max(output, 1)[1].numpy()

    return predict



#训练用

for k in range(80):
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    P = 1
    N = 1
    for n in range(442):

        feature =torch.tensor(shf_features[n])
        feature = feature.unsqueeze(0)
        #adj = utils.normalize_adj(shf_adjs[n])
        adj = torch.from_numpy(shf_adjs[n])
        #norma_adj = utils.normalize_adj(adj)
        label = torch.tensor(shf_labels[n])
        predict  = train(feature, adj, label)

        if numpy.array_equal(label.cpu().numpy(), numpy.array([1])):
            P = P + 1

            # if predict.data == labels.data:
            #print("1predict=",predict, "label=",label.numpy())
            if numpy.array_equal(predict,numpy.array([1])):
                TP = TP + 1
            else:
                FN = FN + 1
        else:
            N = N + 1
            #print("2predict=", predict, "label=", label.numpy())
            if numpy.array_equal(predict,numpy.array([1])):
                FP = FP + 1
            else:
                TN = TN + 1
    print('第',k,'次训练','准确率=',(TP + TN) / (P + N),'召回率=', TP / P, 'F1-分数=',2 * TP / (P + TP + FP))



filenames_train_1 = os.listdir('./data/camel1.6/test/isbug',)
filenames_train_0 = os.listdir('./data/camel1.6/test/nobug',)
features_1, adj_1 = utils.list_features_adj(filenames_train_1, 1, 'test') #35
features_0, adj_0 = utils.list_features_adj(filenames_train_0, 0, 'test')
features = features_1 + features_0
#features = torch.tensor(features)
adjs = adj_1 + adj_0
labels = []
for i in range(74):
    if i < 34:
        labels.append([1])
    else:
        labels.append([0])

import random
s = [i for i in range(0, 74)]
r = random.shuffle(s)
shf_features = []
shf_adjs = []
shf_labels = []

for j in s:
    shf_features.append(features[j])
    shf_adjs.append(adjs[j])
    shf_labels.append(labels[j])


a = 0
for n in range(74):
    feature = torch.tensor(shf_features[n])
    feature = feature.unsqueeze(0)
    adj = torch.from_numpy(shf_adjs[n])
    label = torch.tensor(shf_labels[n])


    predduct = compute_test(feature,adj,label)
    if predduct == label.numpy():
        a = a + 1
print('测试集准确率=',a/74)


