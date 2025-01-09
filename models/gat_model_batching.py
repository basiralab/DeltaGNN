import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

"""
    GAT: Graph Attention Networks
    Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, Yoshua Bengio
    Graph Attention Networks (ICLR 2018)
    https://arxiv.org/abs/1710.10903
"""

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, out_channels, 
                 num_heads=1, batch_norm=False, dropout=0.0, drop_input=False, residual=False, head_depth=1):
        """
        Initialize the Graph Attention Network (GAT) model with a prediction head.

        Args:
            hidden_channels (int): Number of hidden channels in GAT layers.
            num_layers (int): Number of GAT layers in the network.
            in_channels (int): Number of input features.
            out_channels (int): Number of output features.
            num_heads (int, optional): Number of attention heads in each GAT layer. Default is 1.
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
        self.num_heads = num_heads
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.drop_input = drop_input
        self.residual = residual
        self.head_depth = head_depth

        self.convs = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()

        # Adding input layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout))
        if self.batch_norm:
            self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels * num_heads))

        # Adding hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout))
            if self.batch_norm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels * num_heads))

        # Adding output layer
        self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout))

        # Adding prediction head layers
        self.prediction_head = torch.nn.Sequential()
        for i in range(head_depth):
            self.prediction_head.add_module(f'linear_{i}', torch.nn.Linear(hidden_channels if i == 0 else out_channels, out_channels))
            if i < head_depth - 1:  # No activation after the last layer
                self.prediction_head.add_module(f'relu_{i}', torch.nn.ReLU())

    def forward(self, x, edge_index, batch=None):
        """
        Forward pass of the GAT model.

        Args:
            x (Tensor): Input feature matrix.
            edge_index (Tensor): Edge index tensor for graph convolution.
            batch (Tensor, optional): Batch tensor for batching graphs. Default is None.

        Returns:
            Tensor: Output feature matrix or graph-level output.
        """
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        residual = x  # Save the initial x for residual connections

        for i in range(self.num_layers - 1):
            x = self.convs[i](x, edge_index)
            if self.batch_norm:
                x = self.batch_norm_layers[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.residual:
                x = x + residual
                residual = x  # Update residual for the next layer

        # Apply the last layer
        x = self.convs[-1](x, edge_index)
        if self.residual:
            x = x + residual  # Add residual after the last layer

        # Global mean pooling
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = torch.mean(x, dim=0, keepdim=True)

        # Apply the prediction head
        x = self.prediction_head(x)

        return x
