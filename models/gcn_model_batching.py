import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

"""
    GCN: Graph Convolutional Networks
    Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)
    http://arxiv.org/abs/1609.02907
"""

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, out_channels, batch_norm=False, dropout=0.0, drop_input=False, residual=False, head_depth=1):
        """
        Initialize the Graph Convolutional Network (GCN) model.

        Args:
            hidden_channels (int): Number of hidden channels in GCN layers.
            num_layers (int): Number of GCN layers in the network.
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            batch_norm (bool, optional): Whether to apply batch normalization. Default is False.
            dropout (float, optional): Dropout rate for dropout layers. Default is 0.0.
            drop_input (bool, optional): Whether to apply dropout to input features. Default is False.
            residual (bool, optional): Whether to use residual connections. Default is False.
            head_depth (int, optional): Number of layers in the prediction head. Default is 1.
        """
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.drop_input = drop_input
        self.residual = residual
        self.head_depth = head_depth

        self.convs = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()

        # Adding input layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        if self.batch_norm:
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))

        # Adding hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            if self.batch_norm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))

        # Adding output layer
        self.convs.append(GCNConv(hidden_channels, hidden_channels))

        # Adding prediction head layers
        self.prediction_head = torch.nn.Sequential()
        for i in range(head_depth):
            self.prediction_head.add_module(f'linear_{i}', torch.nn.Linear(hidden_channels if i == 0 else out_channels, out_channels))
            if i < head_depth - 1:  # No activation after the last layer
                self.prediction_head.add_module(f'relu_{i}', torch.nn.ReLU())

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the GCN model.

        Args:
            x (Tensor): Input feature matrix.
            edge_index (Tensor): Edge index tensor for graph convolution.
            batch (Tensor, optional): Batch tensor for batching graphs. Default is None.

        Returns:
            Tensor: Output feature matrix or graph-level output.
        """
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            if self.residual:
                x = torch.cat([x, self.convs[i](x, edge_index)], dim=1)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Apply the last layer
        x = self.convs[-1](x, edge_index)
        if self.residual:
            x = torch.cat([x, self.convs[-1](x, edge_index)], dim=1)

        # Global mean pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)

        # Apply the prediction head
        x = self.prediction_head(x)

        return x