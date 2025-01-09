import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
import networkx as nx
from itertools import combinations
from torch_sparse import SparseTensor
from .custom_layers import GCNConvWithAggregation


class DeltaGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, batch_norm=False, dropout=0.0, 
                 drop_input=False, residual=False, density=0.95, 
                 max_communities=500, linear = False):
        """
        Initializes the DeltaGNN model.

        Args:
            hidden_channels (int): Number of hidden channels.
            num_layers (int): Number of layers.
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            batch_norm (bool): Whether to use batch normalization.
            dropout (float): Dropout rate.
            drop_input (bool): Whether to apply dropout to input.
            residual (bool): Whether to use residual connections.
            density (float): Density of the filtered graph.
            max_communities (int): Maximum number of communities.
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
        self.density = density
        self.max_communities = max_communities
        self.linear = linear
        
        self.convs_a = torch.nn.ModuleList()
        self.convs_b = torch.nn.ModuleList()
        self.batch_norm_layers_a = torch.nn.ModuleList()
        self.batch_norm_layers_b = torch.nn.ModuleList()

        # Adding input layer
        input_dim = 2 * in_channels if residual else in_channels
        self.convs_a.append(GCNConvWithAggregation(input_dim, hidden_channels))
        self.convs_b.append(GCNConvWithAggregation(input_dim, hidden_channels))
        
        if self.batch_norm:
            self.batch_norm_layers_a.append(torch.nn.BatchNorm1d(hidden_channels))
            self.batch_norm_layers_b.append(torch.nn.BatchNorm1d(hidden_channels))

        hidden_dim = 2 * hidden_channels if residual else hidden_channels
        # Adding hidden layers
        for _ in range(num_layers - 2):
            self.convs_a.append(GCNConvWithAggregation(hidden_dim, hidden_channels))
            self.convs_b.append(GCNConvWithAggregation(hidden_dim, hidden_channels))
            if self.batch_norm:
                self.batch_norm_layers_a.append(torch.nn.BatchNorm1d(hidden_channels))
                self.batch_norm_layers_b.append(torch.nn.BatchNorm1d(hidden_channels))

        self.merge_layer_b = torch.nn.Linear(num_layers * hidden_dim, hidden_channels)

        # Adding output layer
        output_dim = 4 * hidden_channels if residual else 2 * hidden_channels
        self.output_layer = torch.nn.Linear(output_dim, out_channels)

    def generate_heterophily_graph(self, edge_index, scores, device):
        """
        Generates a heterophily graph based on node scores.

        Args:
            edge_index (Tensor): Edge index tensor.
            scores (Tensor): Node scores.
            device (torch.device): Device to use.

        Returns:
            SparseTensor: Generated heterophily graph.
        """
        G = nx.Graph()
        G.add_edges_from(edge_index.t().cpu().numpy())
        components = list(nx.connected_components(G))
        components.sort(key=len, reverse=True)  # Sort by size in descending order
        components = components[:min(len(components), self.max_communities)]
        
        selected_nodes = [max(component, key=lambda node: scores[node].item()) for component in components]

        edge_pairs = list(combinations(selected_nodes, 2))
        edge_index_list = edge_pairs + [(j, i) for i, j in edge_pairs]
        edge_index_list.extend([(node, node) for node in selected_nodes])  # Add self-loops

        filtered_edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        filtered_edge_attr = torch.ones(filtered_edge_index.size(1), dtype=torch.float32)

        adj_hete = SparseTensor(row=filtered_edge_index[0], col=filtered_edge_index[1], value=filtered_edge_attr,
                                sparse_sizes=(scores.size(0), scores.size(0))).to(device)

        return adj_hete

    def filter_edges(self, edge_index, scores, device, layer):
        """
        Filters edges based on scores to control graph density.

        Args:
            edge_index (Tensor): Edge index tensor.
            scores (Tensor): Node scores.
            device (torch.device): Device to use.

        Returns:
            Tensor: Filtered edge index.
        """
        edge_scores = (scores[edge_index[0]] + scores[edge_index[1]]) / 2
        _, sorted_indices = torch.sort(edge_scores, descending=True)
        if self.linear:
            edges_to_keep = int((((self.density-1)/(self.num_layers-1))*layer + 1) * edge_scores.size(0))
        else:
            edges_to_keep = int(self.density*edge_scores.size(0))
        edges_to_keep_indices = sorted_indices[:edges_to_keep]
        filtered_edge_index = edge_index[:, edges_to_keep_indices]
        return filtered_edge_index

    def forward_single(self, x, edge_index, device, flow_flag=False, heterophily=False):
        """
        Forward pass for a single graph.

        Args:
            x (Tensor): Input features.
            edge_index (Tensor): Edge index tensor.
            device (torch.device): Device to use.
            flow_flag (bool): Whether to calculate flow scores.
            heterophily (bool): Whether to use heterophily graph.

        Returns:
            Tensor: Output features.
            Tensor: Processed edge index.
            Tensor: Scores (if flow_flag is True).
        """
        all_x = []
        if flow_flag:
            deltas = torch.zeros((x.size(0), self.num_layers), device=device)
            second_deltas = torch.zeros((x.size(0), self.num_layers - 1), device=device)
            all_scores = []

        def compute_ema(new_value, old_ema, alpha=0.6):
            if old_ema is None:
                return new_value
            return alpha * new_value + (1 - alpha) * old_ema

        def compute_exponential_moving_std(new_value, old_variance, old_mean, alpha=0.6):
            if old_variance is None:
                return 0.0
            
            # Update the mean with exponential smoothing
            new_mean = alpha * new_value + (1 - alpha) * old_mean
            
            # Update the variance with exponential smoothing
            new_variance = alpha * (new_value - new_mean) ** 2 + (1 - alpha) * old_variance
            
            # Return the square root of the variance (standard deviation)
            return new_variance ** 0.5

        ema_deltas = None
        variance_second_deltas = None

        for i in range(self.num_layers - 1):  # exclude output layer
            prev_x = x
            next_x, aggregated_features = self.convs_a[i](x, edge_index) if not heterophily else self.convs_b[i](x, edge_index)

            if self.residual:
                x = torch.cat((next_x, x), 1)
            else:
                x = next_x

            if flow_flag:
                deltas[:, i] = torch.norm(aggregated_features - prev_x, dim=1)
                if i > 0:
                    second_deltas[:, i - 1] = torch.abs(deltas[:, i] - deltas[:, i - 1])
                    ema_deltas = compute_ema(deltas[:, i], ema_deltas)
                    variance_second_deltas = compute_exponential_moving_std(second_deltas[:, i - 1], variance_second_deltas, ema_deltas)
                    scores = (variance_second_deltas + 1) / (ema_deltas + 1)
                else:
                    scores = 1 / (deltas[:, i] + 1)

                edge_index = self.filter_edges(edge_index, scores, device, i)
                all_scores.append(scores)

            if self.batch_norm:
                x = self.batch_norm_layers_a[i](x) if not heterophily else self.batch_norm_layers_b[i](x)
                
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            all_x.append(x)

        next_x, aggregated_features = self.convs_a[-1](x, edge_index) if not heterophily else self.convs_b[-1](x, edge_index)
        if self.residual:
            x = torch.cat((next_x, x), 1)
        else:
            x = next_x
        all_x.append(next_x)

        if flow_flag:
            deltas[:, self.num_layers - 1] = torch.norm(aggregated_features - prev_x, dim=1)
            second_deltas[:, self.num_layers - 2] = torch.abs(deltas[:, self.num_layers - 1] - deltas[:, self.num_layers - 2])
            ema_deltas = compute_ema(deltas[:, self.num_layers - 1], ema_deltas)
            variance_second_deltas = compute_exponential_moving_std(second_deltas[:, self.num_layers - 2], variance_second_deltas, ema_deltas)
            scores = (variance_second_deltas + 1) / (ema_deltas + 1)
            edge_index = self.filter_edges(edge_index, scores, device, self.num_layers - 1)
            all_scores.append(scores)

        if heterophily:
            x = self.merge_layer_b(torch.cat(all_x, dim=1))
            x = F.relu(x)

        if flow_flag:
            return x, edge_index, all_scores
        else:
            return x, edge_index, None

    def forward(self, x, edge_index_a, edge_index_b=None, batch=None, flow_flag=False):
        """
        Forward pass through the DeltaGNN model.

        Args:
            x (Tensor): Input features.
            edge_index_a (Tensor): Edge index tensor for the first graph (homogeneous).
            device (torch.device): Device to use (CPU or GPU).
            edge_index_b (Tensor, optional): Edge index tensor for the second graph (heterogeneous). Default is None.
            batch (Tensor, optional): Batch tensor for graph-level pooling.
            flow_flag (bool, optional): Whether to calculate flow scores. Default is False.

        Returns:
            Tensor: Output features.
            float: Difference in flow scores between the current and previous iterations.
            List[Tensor]: List containing the edge indices (homogeneous and heterogeneous).
        """

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
        scores_means = []

        x_a = x.clone()
        x_b = x.clone()

        x_a, edge_index_homo, all_scores = self.forward_single(x_a, edge_index_a, device, flow_flag=flow_flag, heterophily=False)

        if flow_flag:
            scores = torch.mean(torch.stack(all_scores, dim=0), dim=0)
            scores_means = [torch.mean(score).item() for score in all_scores]
            edge_index_b = self.generate_heterophily_graph(edge_index_a, scores, device)

        x_b, _, _ = self.forward_single(x_b, edge_index_b, device, False, heterophily=True)

        x = torch.cat((x_a, x_b), 1)

        if batch is not None:
            x = global_mean_pool(x, batch)

        x = self.output_layer(x)

        return x, scores_means, [edge_index_a, edge_index_homo, edge_index_b]
