import torch
import torch_geometric.nn as gnn
from torch import nn


def make_edge_conv(self, in_channels, out_channels, k, aggr):
    class HTheta(nn.Module):
        def __init__(self):
            super().__init__()
            features = 2 * in_channels
            self.linear1 = nn.Linear(2 * in_channels, features)
            self.linear2 = nn.Linear(features, out_channels)

        def forward(self, x):
            x = torch.cat([x[:, :in_channels], x[:, in_channels:] - x[:, :in_channels]], dim=1)
            x = self.linear1(x)
            x = F.gelu(x)
            x = self.linear2(x)
            return x

    return gnn.DynamicEdgeConv(HTheta(), k=k, aggr=aggr)
