from typing import Union
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import GATConv, GATv2Conv


class GAT(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout, 
                 layer_heads,  # expect a list of length num_layers
                 batchnorm=True):
        super(GAT, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm

        # First layer
        self.convs.append(
            GATConv(in_channels, hidden_channels, heads=layer_heads[0], concat=True)
        )
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[0]))

        # Hidden layers
        for i in range(1, num_layers - 1):
            self.convs.append(
                GATConv(hidden_channels * layer_heads[i - 1],
                        hidden_channels,
                        heads=layer_heads[i],
                        concat=True)
            )
            if self.batchnorm:
                # After GAT with concat=True, output dimension is hidden_channels * layer_heads[i]
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[i]))

        # Final layer: Note concat=False so the output dimension is out_channels.
        self.convs.append(
            GATConv(hidden_channels * layer_heads[num_layers - 2],
                    out_channels,
                    heads=layer_heads[num_layers - 1],
                    concat=False)
        )

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Tensor):
        # Apply all layers except the last
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Last layer (no activation/dropout; we return log softmax)
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)


class GATv2(torch.nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels, 
                 num_layers, 
                 dropout, 
                 layer_heads,  # expect a list of length num_layers
                 batchnorm=True):
        super(GATv2, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm

        # First layer
        self.convs.append(
            GATv2Conv(in_channels, hidden_channels, heads=layer_heads[0], concat=True)
        )
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[0]))

        # Hidden layers
        for i in range(1, num_layers - 1):
            self.convs.append(
                GATv2Conv(hidden_channels * layer_heads[i - 1],
                          hidden_channels,
                          heads=layer_heads[i],
                          concat=True)
            )
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels * layer_heads[i]))

        # Final layer: Note concat=False so the output dimension is out_channels.
        self.convs.append(
            GATv2Conv(hidden_channels * layer_heads[num_layers - 2],
                      out_channels,
                      heads=layer_heads[num_layers - 1],
                      concat=False)
        )

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, 
                edge_index: Tensor):
        # Apply all layers except the last
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Final layer
        x = self.convs[-1](x, edge_index)
        return x.log_softmax(dim=-1)
