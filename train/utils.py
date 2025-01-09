import torch
import numpy as np
import networkx as nx
from torch_geometric.typing import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from .metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
import time


def plot_homophily_distributions(adjs, x, y, stages, epoch):
    """
    Plot the distribution of homophily ratios for three different stages of graphs.

    Args:
        adjs (list of torch.Tensor): A list of three adjacency matrices (graphs) for which to calculate and plot homophily ratios.
        x (torch.Tensor): Input features for the nodes.
        y (torch.Tensor): Node labels.
        stages (list of str): Names or descriptions of the stages corresponding to each graph in `adjs`.
        epoch (int): The current epoch number, used to name the saved plot file.

    Raises:
        ValueError: If `adjs` or `stages` does not contain exactly three elements.

    This function calculates the homophily ratio for each graph in `adjs`, plots the distributions of these ratios,
    and saves the plot as a PNG file. Each distribution is plotted with a different color to distinguish between stages.
    """
    if len(adjs) != 3 or len(stages) != 3:
        raise ValueError("You must provide exactly three graphs and three stages.")
    
    homophily_data = [calculate_homophily_ratios(adj, x, y) for adj in adjs]
    means = [np.mean(data) for data in homophily_data]
    colors = ['#2E8B57', '#FF6347','#8670FD']  # Different shades for better comparison
    print(means)
    plt.figure(figsize=(12, 6))

    for i in range(3):
        sns.kdeplot(homophily_data[i], color=colors[i], fill=True, common_norm=True)
        plt.axvline(means[i], color=colors[i], linestyle='--')  # Plot vertical dashed line at mean


    # Customize tick colors and grid
    plt.xlabel('')
    plt.ylabel('')
    plt.tick_params(axis='x', colors='grey')
    plt.tick_params(axis='y', colors='grey')
    plt.grid(True, color='grey', linewidth=0.5)

    # Save the plot
    plot_filename = f'results/images/distribution_homophily_{epoch}.png'
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()


def construct_graph(x, y, edge_index, train_mask = None, val_mask = None, test_mask = None):
    """
    Construct a NetworkX graph from node features, labels, and edge information.

    Args:
        x (np.ndarray or torch.Tensor): Node features with shape (num_nodes, feature_dim).
        y (np.ndarray or torch.Tensor): Node labels with shape (num_nodes,).
        edge_index (torch.Tensor or list of tuples): Edge indices either in PyTorch Geometric format (2xN tensor) or standard edge list format.
        train_mask (np.ndarray or list): Boolean mask indicating training nodes.
        val_mask (np.ndarray or list): Boolean mask indicating validation nodes.
        test_mask (np.ndarray or list): Boolean mask indicating test nodes.

    Returns:
        nx.Graph: A NetworkX graph with nodes having attributes for features, labels, and masks, and edges with default weights.
    """

    # Construct NetworkX Graph
    nodes = [i for i in range(x.shape[0])]

    G = nx.Graph()

    # Add nodes with attributes
    for i in nodes:
        if train_mask is not None:
            G.add_node(i, x=x[i], y=y[i], train=train_mask[i], 
                   val=val_mask[i], test=test_mask[i])
        else:
            G.add_node(i, x=x[i])
    
    # Handle edge_index input (PyTorch Geometric format)
    if isinstance(edge_index, torch.Tensor) and edge_index.dim() == 2 and edge_index.shape[0] == 2:
        edge_list = edge_index.t().tolist()
    else:
        # Assuming edge_index is in standard edge list format
        edge_list = edge_index

    # Add edges with a default weight of 1
    weighted_edges = [(edge[0], edge[1], 1) for edge in edge_list]
    G.add_weighted_edges_from(weighted_edges)

    return G

def split_graph(G, multilabel = True):
    """
    Split the graph into training, validation, and test sets.

    Args:
        G (nx.Graph): The input NetworkX graph with node attributes specifying train, val, and test masks.
        multilabel (bool): Flag indicating if the graph is multilabel. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - x_train (np.ndarray or torch.Tensor): Node features for training nodes.
            - y_train (np.ndarray or torch.Tensor): Node labels for training nodes.
            - edge_train (np.ndarray or torch.Tensor): Edge indices for training nodes.
            - train_mask (np.ndarray or torch.Tensor): Boolean mask for training nodes.
            - x_val (np.ndarray or torch.Tensor): Node features for validation nodes.
            - y_val (np.ndarray or torch.Tensor): Node labels for validation nodes.
            - edge_val (np.ndarray or torch.Tensor): Edge indices for validation nodes.
            - val_mask (np.ndarray or torch.Tensor): Boolean mask for validation nodes.
            - x_test (np.ndarray or torch.Tensor): Node features for test nodes.
            - y_test (np.ndarray or torch.Tensor): Node labels for test nodes.
            - edge_test (np.ndarray or torch.Tensor): Edge indices for test nodes.
            - test_mask (np.ndarray or torch.Tensor): Boolean mask for test nodes.
    """
    print("Splitting Graph...")
    print("=============== Graph Splitting ===============")
    
    # Get complete test graph
    x_test, y_test, edge_test, _, _, test_mask = convert_graph_to_tensor(G, multilabel=multilabel)
    
    print(f"Unlabeled + Test + Validation + Training graph nodes: {x_test.shape[0]}")
    print(f"Unlabeled + Test + Validation + Training graph edges: {edge_test.shape[0]}")
    print(f"Total test nodes: {test_mask.sum()}")
    
    # Get training + val graph
    # remove all test nodes
    test_nodes = []
    for node in G.nodes(data=True):
        if node[1]['test']:
            test_nodes.append(node[0])
    G.remove_nodes_from(test_nodes)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    x_val, y_val, edge_val, _, val_mask, _ = convert_graph_to_tensor(G, multilabel=multilabel)
    
    print(f"Unlabeled + Validation + Training graph nodes: {x_val.shape[0]}")
    print(f"Unlabeled + Validation + Training graph edges: {edge_val.shape[0]}")
    print(f"Total val nodes: {val_mask.sum()}")
    # Get training graph
    # remove all val nodes
    val_nodes = []
    for node in G.nodes(data=True):
        if node[1]['val']:
            val_nodes.append(node[0])
    G.remove_nodes_from(val_nodes)
    G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')
    
    x_train, y_train, edge_train, train_mask, _, _ = convert_graph_to_tensor(G, multilabel = multilabel)
    
    print(f"Unlabeled + Training graph nodes: {x_train.shape[0]}")
    print(f"Unlabeled + Training graph edges: {edge_train.shape[0]}")
    print(f"Total train nodes: {train_mask.sum()}")
    print()
    
    return (x_train, y_train, edge_train, train_mask, x_val, y_val, edge_val, 
            val_mask, x_test, y_test, edge_test, test_mask)

def convert_graph_to_tensor(G, multilabel = True):
    """
    Convert a NetworkX graph into tensors or numpy arrays for use in machine learning models.

    Args:
        G (nx.Graph): The input NetworkX graph with node attributes for features, labels, and masks.
        multilabel (bool): Flag indicating if the graph is multilabel. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - x (np.ndarray or torch.Tensor): Node features.
            - y (np.ndarray or torch.Tensor): Node labels.
            - edge_index (np.ndarray or torch.Tensor): Edge indices.
            - train_mask (np.ndarray or torch.Tensor): Boolean mask for training nodes.
            - val_mask (np.ndarray or torch.Tensor): Boolean mask for validation nodes.
            - test_mask (np.ndarray or torch.Tensor): Boolean mask for test nodes.
    """
    x = np.empty((G.number_of_nodes(),G.nodes[0]['x'].shape[0]))
    if multilabel:
        y = np.empty((G.number_of_nodes(),G.nodes[0]['y'].shape[0]),dtype = 'int')
    else:
        y = np.empty((G.number_of_nodes(),),dtype = 'int')
        
    edge_index = np.array([edge for edge in G.edges()])
    train_mask = np.empty((G.number_of_nodes(),),dtype = 'bool')
    val_mask = np.empty((G.number_of_nodes(),),dtype = 'bool')
    test_mask = np.empty((G.number_of_nodes(),),dtype = 'bool')
    
    for node in G.nodes(data=True):
        x[node[0],:] = node[1]['x']
        if multilabel:
            y[node[0],:] = node[1]['y']
        else:
            y[node[0]] = node[1]['y']

        train_mask[node[0]] = node[1]['train']
        val_mask[node[0]] = node[1]['val']
        test_mask[node[0]] = node[1]['test']
    
    return x, y, edge_index, train_mask, val_mask, test_mask

def construct_normalized_adj(edge_index, num_nodes, self_loops = True):
    """
    Construct a normalized adjacency matrix from edge indices.

    Args:
        edge_index (np.ndarray or torch.Tensor): Edge indices in the format [2, num_edges].
        num_nodes (int): Number of nodes in the graph.

    Returns:
        SparseTensor: Normalized adjacency matrix with self-loops added and GCN normalization applied.
    """
    if len(edge_index) == 0:
        # Create a self-loop only adjacency matrix
        row = torch.arange(num_nodes)
        col = torch.arange(num_nodes)
        adj = SparseTensor(row=row, col=col, sparse_sizes=(num_nodes, num_nodes))
        adj = adj.set_diag()  # adding self-loops
        adj = gcn_norm(adj, add_self_loops=False)  # normalization
        return adj

    edge_index = torch.tensor(edge_index)
    edge_index = torch.transpose(edge_index,0,1)
    edge_index_flip = torch.flip(edge_index,[0]) # re-adds flipped edges that were removed by networkx
    edge_index = torch.cat((edge_index, edge_index_flip), 1)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes,num_nodes))
    if self_loops:
        adj = adj.set_diag() # adding self loops
    adj = gcn_norm(adj, add_self_loops=False) # normalization

    return adj