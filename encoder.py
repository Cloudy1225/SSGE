import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class MLP(nn.Module):
    def __init__(self, in_dim, hid_dims, act_fn, use_bn=True):
        super(MLP, self).__init__()

        self.layer1 = nn.Linear(in_dim, hid_dims[0], bias=True)
        self.layer2 = nn.Linear(hid_dims[0], hid_dims[-1], bias=True)

        if use_bn:
            self.bn = nn.BatchNorm1d(hid_dims[0])
        else:
            self.register_parameter('bn', None)

        self.act_fn = act_fn

    def forward(self, _, X):
        Z = self.layer1(X)
        if self.bn is not None:
            Z = self.bn(Z)
        Z = self.act_fn(Z)
        Z = self.layer2(Z)

        return Z


class GCN(nn.Module):
    def __init__(self, in_dim, hid_dims, act_fn):
        super(GCN, self).__init__()

        self.encoder = nn.ModuleList()
        hid_dims = [in_dim] + list(hid_dims)
        if len(hid_dims) > 2:
            for i in range(len(hid_dims) - 2):
                self.encoder.append(GraphConv(hid_dims[i], hid_dims[i + 1], activation=act_fn))
        self.encoder.append(GraphConv(hid_dims[-2], hid_dims[-1], activation=None))

    def forward(self, G, X):
        Z = X
        for layer in self.encoder:
            Z = layer(G, Z)
        return Z
