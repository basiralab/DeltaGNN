import torch
import torch.nn.functional as F
import networkx as nx
from torch_geometric.typing import SparseTensor
from itertools import combinations

class DeltaGNN(torch.nn.Module):
    def __init__(self, hidden_channels, num_layers, in_channels, 
                 out_channels, batch_norm=False, dropout=0.0, 
                 drop_input=False, residual=False, graph_task=False,
                 density=0.95, max_communities=500, linear = False):
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
            graph_task (bool): Whether the task is graph-level.
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
        self.graph_task = graph_task
        self.density = density
        self.max_communities = max_communities
        self.linear = linear
        #self.prev_scores_mean = None
        
        self.linear_layers_a = torch.nn.ModuleList()
        self.linear_layers_b = torch.nn.ModuleList()
        self.batch_norm_layers_a = torch.nn.ModuleList()
        self.batch_norm_layers_b = torch.nn.ModuleList()

        # Adding input layer
        input_dim = 2 * in_channels if residual else in_channels
        self.linear_layers_a.append(torch.nn.Linear(input_dim, hidden_channels))
        self.linear_layers_b.append(torch.nn.Linear(input_dim, hidden_channels))
        
        if self.batch_norm:
            self.batch_norm_layers_a.append(torch.nn.BatchNorm1d(hidden_channels))
            self.batch_norm_layers_b.append(torch.nn.BatchNorm1d(hidden_channels))

        # Adding hidden layers
        for _ in range(num_layers - 2):
            hidden_dim = 2 * hidden_channels if residual else hidden_channels
            self.linear_layers_a.append(torch.nn.Linear(hidden_dim, hidden_channels))
            self.linear_layers_b.append(torch.nn.Linear(hidden_dim, hidden_channels))
            if self.batch_norm:
                self.batch_norm_layers_a.append(torch.nn.BatchNorm1d(hidden_channels))
                self.batch_norm_layers_b.append(torch.nn.BatchNorm1d(hidden_channels))

        self.merge_layer_b = torch.nn.Linear(num_layers * hidden_dim, hidden_channels)

        # Adding output layer
        output_dim = 4 * hidden_channels if residual else 2 * hidden_channels
        self.output_layer = torch.nn.Linear(output_dim, out_channels)

        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initializes the parameters of the linear layers."""
        for layer in self.linear_layers_a:
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)
        for layer in self.linear_layers_b:
            torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            if layer.bias is not None:
                layer.bias.data.fill_(0.01)

    def generate_heterophily_graph(self, adj, scores, device):
        """
        Generates a heterophily graph based on node scores.

        Args:
            adj (SparseTensor): Adjacency matrix.
            scores (Tensor): Node scores.
            device (torch.device): Device to use.

        Returns:
            SparseTensor: Generated heterophily graph.
        """
        # Initialize lists to store edge indices and values
        edge_index_list = []
        edge_attr_list = []

        # Detect connected components of adj
        row, col, _ = adj.coo()
        G = nx.Graph()
        G.add_edges_from(zip(row.cpu().numpy(), col.cpu().numpy()))
        components = list(nx.connected_components(G))
        components.sort(key=len, reverse=True)  # Sort by size in descending order
        components = components[:min(len(components), self.max_communities)]
        
        # For each component, pick the node with the highest score
        selected_nodes = [max(component, key=lambda node: scores[node].item()) for component in components]

        # Create all pairs of selected nodes to form a fully connected subgraph
        edge_pairs = list(combinations(selected_nodes, 2))

        # Add edges for the fully connected subgraph and self-loops
        edge_index_list.extend(edge_pairs + [(j, i) for i, j in edge_pairs])
        edge_index_list.extend([(node, node) for node in list(G.nodes)])  # Add self-loops
        edge_attr_list.extend([1] * (len(edge_pairs) * 2 + len(selected_nodes)))

        # Create the new edge index and values
        filtered_edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        filtered_edge_attr = torch.tensor(edge_attr_list, dtype=torch.float32)

        # Create the new sparse tensor adj_hete
        adj_hete = SparseTensor(row=filtered_edge_index[0], col=filtered_edge_index[1], value=filtered_edge_attr,
                                sparse_sizes=adj.sparse_sizes()).to(device)

        return adj_hete

    def filter_edges(self, adj, scores, device, layer):
        """
        Filters edges based on scores to control graph density.

        Args:
            adj (SparseTensor): Adjacency matrix.
            scores (Tensor): Node scores.
            device (torch.device): Device to use.

        Returns:
            SparseTensor: Filtered adjacency matrix.
        """
        # Extract COO format components
        row, col, edge_attr = adj.coo()
        edge_index = torch.stack([row, col], dim=0)

        # Calculate edge scores as the average of node scores
        edge_scores = (scores[edge_index[0]] + scores[edge_index[1]]) / 2

        # Sort edges by score
        _, sorted_indices = torch.sort(edge_scores, descending=True)

        # Remove the edges with the lowest scores
        if self.linear:
            edges_to_keep = int((((self.density-1)/(self.num_layers-1))*layer + 1) * len(edge_scores))
        else:
            edges_to_keep = int(self.density*len(edge_scores))
        edges_to_keep_indices = sorted_indices[:edges_to_keep]
        filtered_edge_index = edge_index[:, edges_to_keep_indices]
        filtered_edge_attr = edge_attr[edges_to_keep_indices]

        # Create the new sparse tensor
        adj_filtered = SparseTensor(row=filtered_edge_index[0], col=filtered_edge_index[1], value=filtered_edge_attr,
                                    sparse_sizes=adj.sparse_sizes()).to(device)

        return adj_filtered

    def get_density(self):
        """
        Returns the current density value.

        Returns:
            float: Current density value.
        """
        return self.density

    def update_density(self, new_value):
        """
        Updates the density value.

        Args:
            new_value (float): New density value.
        """
        self.density = new_value

    def forward_single(self, x, adj, device, flow_flag=False, heterophily=False):
        """
        Forward pass for a single graph.

        Args:
            x (Tensor): Input features.
            adj (SparseTensor): Adjacency matrix.
            device (torch.device): Device to use.
            flow_flag (bool): Whether to calculate flow scores.
            heterophily (bool): Whether to use heterophily graph.

        Returns:
            Tensor: Output features.
            SparseTensor: Processed adjacency matrix.
            Tensor: Scores (if flow_flag is True).
        """
        if heterophily:
            all_x = []

        # Initialize deltas and second_deltas if flow_flag is true
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
            next_x = adj @ x  # Compute next_x first

            # aggregation phase
            if self.residual:
                if i == 0 and heterophily:
                    x = torch.cat((x, x), 1)
                else:
                    x = torch.cat((next_x, x), 1)
            else:
                if i != 0:
                    x = next_x

            # Calculate deltas if flow_flag is true
            if flow_flag:
                deltas[:, i] = torch.norm(next_x - prev_x, dim=1)
                if i > 0:
                    second_deltas[:, i - 1] = torch.abs(deltas[:, i] - deltas[:, i - 1])
                    ema_deltas = compute_ema(deltas[:, i], ema_deltas)
                    variance_second_deltas = compute_exponential_moving_std(second_deltas[:, i - 1], variance_second_deltas, ema_deltas)
                    scores = (variance_second_deltas + 1) / (ema_deltas + 1)
                else:
                    scores = 1 / (deltas[:, i] + 1)

                adj = self.filter_edges(adj, scores, device, i)
                all_scores.append(scores)

            # transformation phase
            if not heterophily:
                x = self.linear_layers_a[i](x)
                if self.batch_norm:
                    x = self.batch_norm_layers_a[i](x)
            else:
                x = self.linear_layers_b[i](x)
                if self.batch_norm:
                    x = self.batch_norm_layers_b[i](x)
            x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

            if heterophily:
                all_x.append(x)

        # aggregation phase (output layer)
        prev_x = x
        next_x = adj @ x  # Compute next_x first

        if self.residual:
            x = torch.cat((next_x, x), 1)
        else:
            x = next_x

        # Calculate deltas for the last layer if flow_flag is true
        if flow_flag:
            deltas[:, self.num_layers - 1] = torch.norm(next_x - prev_x, dim=1)
            second_deltas[:, self.num_layers - 2] = torch.abs(deltas[:, self.num_layers - 1] - deltas[:, self.num_layers - 2])
            # Calculate final scores
            ema_deltas = compute_ema(deltas[:, self.num_layers - 1], ema_deltas)
            variance_second_deltas = compute_exponential_moving_std(second_deltas[:, self.num_layers - 2], variance_second_deltas, ema_deltas)
            scores = (variance_second_deltas + 1) / (ema_deltas + 1)
            adj = self.filter_edges(adj, scores, device, self.num_layers-1)
            all_scores.append(scores)

        if heterophily:
            all_x.append(x)
            x = self.merge_layer_b(torch.cat(all_x, dim=1))
            x = x.relu_()

        if self.graph_task:
            x = torch.mean(x, dim=0)

        if flow_flag:
            return x, adj, all_scores
        else:
            return x, adj, None

    def forward(self, x, adj_a, device, adj_b=None, flow_flag=False):
        """
        Forward pass through the DeltaGNN model.

        Args:
            x (Tensor): Input features.
            adj_a (SparseTensor): Adjacency matrix for the first graph (homogeneous).
            device (torch.device): Device to use (CPU or GPU).
            adj_b (SparseTensor, optional): Adjacency matrix for the second graph (heterogeneous). Default is None.
            flow_flag (bool, optional): Whether to calculate flow scores. Default is False.

        Returns:
            Tensor: Output features.
            float: Difference in flow scores between the current and previous iterations.
            List[SparseTensor]: List containing the adjacency matrices (homogeneous and heterogeneous).
        """
        # if using input dropout
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)
        #flow_diff = 0
        scores_means = []

        x_a = x.clone()
        x_b = x.clone()

        x_a, adj_homo, all_scores = self.forward_single(x_a, adj_a, device, flow_flag=flow_flag, heterophily=False)

        if flow_flag:
            del adj_b
            scores = torch.mean(torch.stack(all_scores, dim=0), dim=0)
            scores_means = [torch.mean(score).item() for score in all_scores]
            #if self.prev_scores_mean is None:
                #flow_diff = 0
            #else:
                #flow_diff = scores_means[-1] - self.prev_scores_mean
            #self.prev_scores_mean = scores_means[-1]
            adj_b = self.generate_heterophily_graph(adj_a, scores, device)

        x_b, _, _ = self.forward_single(x_b, adj_b, device, False, heterophily=True)

        x = torch.cat((x_a, x_b), 1)

        x = self.output_layer(x)

        return x, scores_means, [adj_a, adj_homo, adj_b]
