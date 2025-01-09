import torch
import numpy as np
import torch.utils.data
import sklearn
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import WebKB
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import LRGBDataset
from torch_geometric.utils import erdos_renyi_graph
import torch_geometric.transforms as T
from sklearn.model_selection import train_test_split
from torch_geometric.transforms import NormalizeFeatures, Constant, OneHotDegree

def get_planetoid_dataset(name="Cora"):
    """
    Load the Planetoid dataset (Cora, CiteSeer, or PubMed) with normalized features.

    Args:
        name (str): The name of the dataset. Must be one of 'Cora', 'CiteSeer', or 'PubMed'.

    Returns:
        tuple: A tuple containing node features, labels, edge indices, 
               and masks for training, validation, and testing.
    """
    assert name in ["Cora", "CiteSeer", "PubMed"], "Dataset name must be 'Cora', 'CiteSeer', or 'PubMed'."
    dataset = Planetoid(root=f'/data/{name}', name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    x, y, edge_index = data.x, data.y, data.edge_index
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask

    return x, y, edge_index, train_mask, val_mask, test_mask

def get_webkb_dataset(name="Cornell"):
    """
    Load the WebKB dataset (Cornell, Texas, or Wisconsin) with normalized features.

    Args:
        name (str): The name of the dataset. Must be one of 'Cornell', 'Texas', or 'Wisconsin'.

    Returns:
        tuple: A tuple containing node features, labels, edge indices, 
               and masks for training, validation, and testing.
    """
    assert name in ["Cornell", "Texas", "Wisconsin"], "Dataset name must be 'Cornell', 'Texas', or 'Wisconsin'."
    dataset = WebKB(root=f'/data/{name}', name=name, transform=T.NormalizeFeatures())
    data = dataset[0]
    x, y, edge_index = data.x, data.y, data.edge_index

    train_mask = data.train_mask if data.train_mask.dim() == 1 else data.train_mask[:, 0]
    val_mask = data.val_mask if data.val_mask.dim() == 1 else data.val_mask[:, 0]
    test_mask = data.test_mask if data.test_mask.dim() == 1 else data.test_mask[:, 0]

    # Convert data to numpy arrays for print_statistics
    x_np = x.numpy()
    y_np = y.numpy()
    edge_index_np = edge_index.numpy()
    
    # Print dataset statistics
    print_statistics(x_np, y_np, edge_index_np, train_mask, val_mask, test_mask)

    return x, y, edge_index, train_mask, val_mask, test_mask

def get_tudataset(name="PROTEINS"):
    """
    Load the TUDataset (PROTEINS, ENZYMES, MUTAG, COLLAB, IMDB-BINARY, REDDIT-BINARY) with synthetic features and one-hot encoded labels.

    Args:
        name (str): The name of the dataset. Must be one of 'PROTEINS', 'ENZYMES', 'MUTAG', 'COLLAB', 'IMDB-BINARY', 'REDDIT-BINARY'.

    Returns:
        tuple: A tuple containing node features, one-hot encoded labels, edge indices, and the dataset.
    """
    assert name in ["PROTEINS", "ENZYMES", "MUTAG", "COLLAB", "IMDB-BINARY", "REDDIT-BINARY"], \
        "Dataset name must be 'PROTEINS', 'ENZYMES', 'MUTAG', 'COLLAB', 'IMDB-BINARY', or 'REDDIT-BINARY'."

    # Define the number of classes for each dataset
    num_classes_map = {
        "MUTAG": 2,
        "ENZYMES": 6,
        "PROTEINS": 2,
        "COLLAB": 3,
        "IMDB-BINARY": 2,
        "REDDIT-BINARY": 2
    }
    
    # Define a transform for datasets without node features
    if name in ["COLLAB", "IMDB-BINARY", "REDDIT-BINARY"]:
        transform = OneHotDegree(max_degree=10)  # You can adjust `max_degree` as needed
    else:
        transform = NormalizeFeatures()  # Normalization for datasets with existing features

    # Load the dataset
    dataset = TUDataset(root=f'/data/{name}', name=name, transform=transform, force_reload=True)
    
    x_list = []
    y_list = []
    edge_index_list = []
    
    if name in ["COLLAB", "IMDB-BINARY", "REDDIT-BINARY"]:
        for i in range(len(dataset)):
            data = dataset.get(i)
            if data.x is None:
                # Create synthetic features
                num_nodes = data.num_nodes
                data.x = torch.ones((num_nodes, 1))  # Create a tensor of shape [num_nodes, 1] with constant value 1
            x_list.append(data.x)
            y_list.append(data.y)
            edge_index_list.append(data.edge_index)
    else:
        for data in dataset:
            if data.x is None:
                # Create synthetic features
                num_nodes = data.num_nodes
                data.x = torch.ones((num_nodes, 1))  # Create a tensor of shape [num_nodes, 1] with constant value 1
            x_list.append(data.x)
            y_list.append(data.y)
            edge_index_list.append(data.edge_index)
    
    return x_list, y_list, edge_index_list, dataset

def get_lrgb_dataset(name="PascalVOC-SP", split='train'):
    """
    Load an LRGB dataset (e.g., PascalVOC-SP, COCO-SP, PCQM-Contact, Peptides-func, Peptides-struct) with normalized features.

    Args:
        name (str): The name of the dataset. Must be one of 'PascalVOC-SP', 'COCO-SP', 'PCQM-Contact', 'Peptides-func', or 'Peptides-struct'.
        split (str): The split of the dataset to load ('train', 'val', or 'test').

    Returns:
        tuple: A tuple containing lists of node features, labels, and edge indices for all graphs, and the dataset object.
    """
    valid_datasets = ["PascalVOC-SP", "COCO-SP", "PCQM-Contact", "Peptides-func", "Peptides-struct"]
    assert name in valid_datasets, f"Dataset name must be one of {valid_datasets}."
    
    dataset = LRGBDataset(root=f'/data/{name}', name=name, split=split, transform=T.NormalizeFeatures())
    
    x_list = []
    y_list = []
    edge_index_list = []
    
    for i in range(len(dataset)):
        data = dataset.get(i)
        x_list.append(data.x)
        y_list.append(data.y)
        edge_index_list.append(data.edge_index)
    
    return x_list, y_list, edge_index_list, dataset

def get_er_graph(n_nodes=1000, p_edge=None, num_edges=None, num_features = 16):
    """
    Generate an Erdos-Renyi (ER) random graph with optional parameters for edges and probability.

    Args:
        n_nodes (int): Number of nodes in the graph.
        p_edge (float, optional): Probability of edge creation between nodes. 
                                   If specified, `num_edges` should be None.
        num_edges (int, optional): Number of edges in the graph. 
                                    If specified, `p_edge` should be None.

    Returns:
        tuple: A tuple containing node features, edge indices, and the number of nodes and edges.
    """
    assert (p_edge is not None) or (num_edges is not None), "Either `p_edge` or `num_edges` must be specified."
    assert not (p_edge is not None and num_edges is not None), "Specify only one of `p_edge` or `num_edges`."
    
    if p_edge is not None:
        # Generate graph using probability of edge creation
        edge_index = erdos_renyi_graph(n_nodes, p_edge)
    else:
        # Generate graph with a specific number of edges
        # Using p_edge=0.5 as a placeholder to get a graph with `num_edges` edges
        edge_index = erdos_renyi_graph(n_nodes, 0.5)
        num_current_edges = edge_index.size(1)
        
        # Adjust to get exactly num_edges edges
        if num_current_edges > num_edges:
            # Truncate to num_edges if more edges are generated
            edge_index = edge_index[:, :num_edges]
        elif num_current_edges < num_edges:
            # Repeat edges to reach num_edges if fewer edges are generated
            repeat_count = (num_edges // num_current_edges) + 1
            edge_index = torch.cat([edge_index] * repeat_count, dim=1)[:, :num_edges]
    
    # Generate node features (random for demonstration purposes)
    x = torch.randn(n_nodes, num_features)  # Random features with dimension 16

    # Return features, edge indices, and graph statistics
    return x, edge_index, n_nodes, edge_index.size(1)

def get_organ_dataset(view='c', sparse=True, balanced=True, demo=False):
    """
    Load the Organ dataset (c or c view) in either sparse or dense format.

    Args:
        view (str): The view of the dataset. Must be 'c' or 's'.
        sparse (bool): Whether to load the sparse version of the dataset.
        balanced (bool): Whether to balance the dataset.
        demo (bool): If true, only return specific records for demo purposes.

    Returns:
        tuple: A tuple containing node features, labels, edge indices, 
               and masks for training, validation, and testing.
    """
    print(f"Loading Organ-{view} Dataset...")
    dataset_name = f'organ{view.lower()}{"_sparse" if sparse else "_dense"}'
        
    # Load masks
    train_mask = torch.tensor(np.load(f'data/{dataset_name}/train_mask.npy'))
    val_mask = torch.tensor(np.load(f'data/{dataset_name}/val_mask.npy'))
    test_mask = torch.tensor(np.load(f'data/{dataset_name}/test_mask.npy'))

    # Load labels
    labels = np.load(f'data/{dataset_name}/data_label.npy')

    # Load and normalize features
    features = np.load(f'data/{dataset_name}/data_feat.npy')
    features = sklearn.preprocessing.StandardScaler().fit_transform(features)

    # Load edge indices
    edge_index = np.load(f'data/{dataset_name}/edge_index.npy')
        
    # Print dataset statistics
    print_statistics(features, labels, edge_index, train_mask, val_mask, test_mask)

    if demo:
        demo_class_counts = {2: 5, 8: 9, 4: 1}
        demo_indices = []

        for label, count in demo_class_counts.items():
            class_indices = np.where(labels == label)[0]
            demo_indices.extend(class_indices[:count])
        
        demo_indices = np.array(demo_indices)

        demo_train_mask = torch.full_like(train_mask, False)
        demo_val_mask = torch.full_like(val_mask, False)
        demo_test_mask = torch.full_like(test_mask, False)

        # Assign all demo indices to the training mask for simplicity
        demo_train_mask[demo_indices] = True

        # Extract demo features, labels and edge indices
        demo_features = features[demo_indices]
        demo_labels = labels[demo_indices]

        # Since this is a demo, edge_index might not be applicable. Adjust as needed.
        demo_edge_index = edge_index

        return demo_features, demo_labels, demo_edge_index, demo_train_mask, demo_val_mask, demo_test_mask

    if not balanced:
        all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        chosen_labels = [0, 1, 2, 4]

        print("[Before unbalancing] Class distribution in the training set:")
        for label in all_labels:
            count = np.sum(labels[train_mask] == label)
            print(f"Label {label}: {count} samples")
        print("[Before unbalancing] Class distribution in the validation set:")
        for label in all_labels:
            count = np.sum(labels[val_mask] == label)
            print(f"Label {label}: {count} samples")

        print("[Before unbalancing] Class distribution in the test set:")
        for label in all_labels:
            count = np.sum(labels[test_mask] == label)
            print(f"Label {label}: {count} samples")

        chosen_indices = np.where(np.isin(labels[train_mask], chosen_labels))[0]
        train_indices, test_indices = train_test_split(chosen_indices, test_size=0.8, stratify=labels[train_mask][chosen_indices])
                
        new_train_mask = torch.full_like(train_mask, False)
        new_train_mask[train_indices] = True

        for i, label in enumerate(labels):
            if label in chosen_labels and new_train_mask[i] == False:
                train_mask[i] = False

        train_mask[train_indices] = True
        test_mask[test_indices] = True

        print("Class distribution in the training set:")
        for label in all_labels:
            count = np.sum(labels[train_mask] == label)
            print(f"Label {label}: {count} samples")

        print("Class distribution in the validation set:")
        for label in all_labels:
            count = np.sum(labels[val_mask] == label)
            print(f"Label {label}: {count} samples")

        print("Class distribution in the test set:")
        for label in all_labels:
            count = np.sum(labels[test_mask] == label)
            print(f"Label {label}: {count} samples")
    
    return features, labels, edge_index, train_mask, val_mask, test_mask

def print_statistics(features, labels, edge_index, train_mask, val_mask, test_mask):
    """
    Print statistics of the dataset.

    Args:
        features (np.ndarray): Node features.
        labels (np.ndarray): Node labels.
        edge_index (np.ndarray): Edge indices.
        train_mask (torch.Tensor): Mask for training nodes.
        val_mask (torch.Tensor): Mask for validation nodes.
        test_mask (torch.Tensor): Mask for test nodes.
    """
    print("=============== Dataset Properties ===============")
    print(f"Total Nodes: {features.shape[0]}")
    print(f"Total Edges: {edge_index.shape[0]}")
    print(f"Number of Features: {features.shape[1]}")
    if labels.ndim == 1:
        print(f"Number of Labels: {labels.max() + 1}")
        print("Task Type: Multi-class Classification")
    else:
        print(f"Number of Labels: {labels.shape[1]}")
        print("Task Type: Multi-label Classification")
    print(f"Training Nodes: {train_mask.sum().item()}")
    print(f"Validation Nodes: {val_mask.sum().item()}")
    print(f"Testing Nodes: {test_mask.sum().item()}")
    print()
