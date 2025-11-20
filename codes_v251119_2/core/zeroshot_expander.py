import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConv(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
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
        m = nn.BatchNorm1d(outputs.shape[1], track_running_stats=False).cuda()
        outputs = m(outputs)

        if self.relu is not None:
            outputs = self.relu(outputs)
        return outputs

class GraphConv_vm(nn.Module):

    def __init__(self, in_channels, out_channels, dropout=False, relu=True):
        super().__init__()
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

    def __init__(self, n, edges, in_channels, out_channels, hidden_layers):
        super().__init__()

        edges = np.array(edges)
        adj = torch.from_numpy(edges).float()
        self.adj = adj.cuda()

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
            alpha_att = torch.zeros(x.shape[0], x.shape[0]).cuda()
            for i in range(x.shape[0]):
                for j in range(x.shape[0]):
                    f_i = x[i].reshape(-1, x.shape[1])
                    f_j = x[j].reshape(-1, x.shape[1])
                    alpha_att[i][j] = torch.cosine_similarity(f_i, f_j)
            alpha_att_softmax = F.softmax(alpha_att, dim=0)
            x = torch.mm(alpha_att_softmax, x)
        return F.normalize(x)
    

# method to calculate ex trans matrix
def getTransitionProb(x1, x2, num_emotions=6):
    prob_sum = np.sum(x1 * x2)
    x1_x2 = prob_sum/np.sum(x2)/num_emotions # p(x1|x2)
    x2_x1 = prob_sum/np.sum(x1)/num_emotions # p(x2|x1)
    return x1_x2, x2_x1


# get ex trans matrix
def getTransitionMatrix(ex_au, threhold):
    num_exs = ex_au.shape[0]
    num_aus = ex_au.shape[1]
    trans_ex = np.zeros((num_exs, num_exs))
    self_connection = np.identity(num_exs)
    b = trans_ex
    for i in range(num_exs - 1):
        for j in range(i + 1, num_exs):
            y1, y2 = getTransitionProb(ex_au[i], ex_au[j])
            b[i][j] = y1
            b[j][i] = y2
    for i in range(num_exs):
        trans_ex[i] = b[i] / np.sum(b[i])
    trans_ex = trans_ex + self_connection
    return trans_ex