import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# #5、#6、#7，设计的原因？为什么有#5?没有微调的数据。我们在推理时主要选择分类器权重，也就是直接推理，但是在训练时需要用直接推理的结果和AU分类的结果更新unseen的P(AU|EMO)

class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True, device='cuda:0'):
        super().__init__()
        self.device = device
        self.outdimension = out_channels

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        torch.nn.init.kaiming_normal_(self.w, a=0, mode='fan_in', nonlinearity='leaky_relu')

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        outputs = torch.mm(adj, torch.mm(inputs, self.w)) + self.b
        m = nn.BatchNorm1d(outputs.shape[1], track_running_stats=False).to(outputs.device)
        outputs = m(outputs)

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs

class GraphConv_vm(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True, device='cuda:0'):
        super().__init__()
        self.device = device
        self.outdimension = out_channels

        if dropout:
            self.dropout = nn.Dropout(p=0.5)
        else:
            self.dropout = None

        self.w = nn.Parameter(torch.empty(in_channels, out_channels))
        self.b = nn.Parameter(torch.zeros(out_channels))
        self.alpha = nn.Parameter(torch.zeros(1))
        torch.nn.init.kaiming_normal_(self.w, a=0, mode='fan_in', nonlinearity='leaky_relu')

        if relu:
            self.relu = nn.LeakyReLU(negative_slope=0.2)
        else:
            self.relu = None

    def forward(self, inputs, adj):
        if self.dropout is not None:
            inputs = self.dropout(inputs)

        neighbors = (1-self.alpha)*torch.mm(adj, inputs)
        selfinfo = self.alpha*inputs
        allinfo = neighbors+selfinfo
        outputs = torch.mm(allinfo, self.w)

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs

class zeroshotExpander(nn.Module):

    def __init__(self, n, edges, in_channels, out_channels, hidden_layers, device):
        super().__init__()
        self.device = device

        # edges = np.array(edges)
        # adj = torch.from_numpy(edges).float()
        self.adj = edges#.to(self.device)

        hl = hidden_layers.split(',')
        if hl[-1] == 'd':
            dropout_last = True
            hl = hl[:-1]
        else:
            dropout_last = False
            hl = hl[:-1]

        i = 0
        layers = []
        last_c = in_channels
        for c in hl:
            if c[0] == 'd':
                dropout = True
                c = c[1:]
            else:
                dropout = False
                c = c[1:]
            c = int(c)

            i += 1
            conv = GraphConv(last_c, c, dropout=dropout)
            # conv = GraphConv_vm(last_c, c, dropout=dropout)
            self.add_module('conv{}'.format(i), conv)
            layers.append(conv)

            last_c = c

        conv = GraphConv(last_c, out_channels, relu=False, dropout=dropout_last)
        # conv = GraphConv_vm(last_c, out_channels, relu=False, dropout=dropout_last)
        self.add_module('conv-last', conv)
        layers.append(conv)

        self.layers = layers

    def forward(self, x, if_att=False):
        for conv in self.layers:
            x = conv(x, self.adj)
        if if_att is True:
            alpha_att = torch.zeros(x.shape[0], x.shape[0]).to(x.device)
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    f_i = x[i].reshape(-1, x.shape[1])
                    f_j = x[j].reshape(-1, x.shape[1])
                    alpha_att[i][j] = torch.cosine_similarity(f_i, f_j)
            alpha_att_softmax = F.softmax(alpha_att, dim=0)
            x = torch.mm(alpha_att_softmax, x)
        return F.normalize(x)
    
