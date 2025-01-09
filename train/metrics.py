from sklearn.metrics import confusion_matrix
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import torch
import networkx as nx


def calculate_homophily_ratios(adj, x, y):
    """
    Calculate homophily ratios based on feature and label similarities.

    Args:
        adj (scipy.sparse.coo_matrix): Adjacency matrix in COO format.
        x (numpy.ndarray or torch.Tensor): Node features.
        y (numpy.ndarray or torch.Tensor): Node labels.

    Returns:
        list: List of homophily ratios for each edge in the adjacency matrix.
    """
    homophily_ratios = []
    
    # Extract COO format components
    row, col, _ = adj.coo()
    
    for r, c in zip(row.tolist(), col.tolist()):

        # Extract features and labels using row and col indices
        x1, y1 = x[r], y[r]
        x2, y2 = x[c], y[c]
        
        # Calculate similarity in features (assuming features are numpy arrays)
        feature_similarity = np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        
        # Calculate similarity in labels
        label_similarity = 1 if y1 == y2 else 0
        
        # Homophily ratio can be a combination of both similarities
        homophily_ratio = 0.5 * feature_similarity + 0.5 * label_similarity
        
        homophily_ratios.append(homophily_ratio)
    
    return homophily_ratios

def logit_to_label(out):
    """
    Convert logits to predicted labels using argmax.

    Args:
        out (torch.Tensor): Logits tensor.

    Returns:
        torch.Tensor: Predicted labels.
    """
    return out.argmax(dim=1)


def metrics(logits, y):
    """
    Calculate classification metrics.

    Args:
        logits (torch.Tensor): Model output logits.
        y (torch.Tensor): True labels.

    Returns:
        tuple: (accuracy, micro F1 score, sensitivity, specificity, average precision)
    """
    if y.dim() == 1:  # Multi-class
        y_pred = logit_to_label(logits)
        cm = confusion_matrix(y.cpu(), y_pred.cpu())
        FP = cm.sum(axis=0) - np.diag(cm)
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        
        acc = np.diag(cm).sum() / cm.sum()
        micro_f1 = acc  # Micro F1 = accuracy for multi-class
        sens = TP.sum() / (TP.sum() + FN.sum())
        spec = TN.sum() / (TN.sum() + FP.sum())
        # Calculate AP for multi-class
        y_one_hot = torch.nn.functional.one_hot(y, num_classes=logits.size(1))
        ap = average_precision_score(y_one_hot.cpu().numpy(), logits.cpu().numpy(), average='macro')
        
    else:  # Multi-label
        y_pred = logits >= 0
        y_true = y >= 0.5
        
        tp = int((y_true & y_pred).sum())
        tn = int((~y_true & ~y_pred).sum())
        fp = int((~y_true & y_pred).sum())
        fn = int((y_true & ~y_pred).sum())
        
        acc = (tp + tn) / (tp + fp + tn + fn)
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        micro_f1 = 2 * (precision * recall) / (precision + recall)
        sens = tp / (tp + fn)
        spec = tn / (tn + fp)
        
        ap = average_precision_score(y.cpu().numpy(), logits.cpu().numpy(), average='micro')
        
    return acc, micro_f1, sens, spec, ap

def regression_metrics(predictions, y):
    """
    Calculate regression metrics.

    Args:
        predictions (torch.Tensor): Model output predictions.
        y (torch.Tensor): True values.

    Returns:
        tuple: (MAE, R2, MSE, RMSE)
    """
    predictions = predictions.cpu().numpy()
    y = y.cpu().numpy()
    
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    
    return mae, r2, mse, rmse

def edge_degree_centrality(graph):
    """
    Compute the degree centrality for each edge in the graph.

    Args:
        graph (networkx.Graph): A NetworkX graph object.

    Returns:
        dict: A dictionary where keys are edges and values are the average degree of the nodes incident to the edge.
    """
    edge_degree = {}
    
    # Iterate over edges
    for edge in graph.edges():
        u, v = edge
        
        # Compute degrees of the nodes incident to the edge
        degree_u = graph.degree(u)
        degree_v = graph.degree(v)
        
        # Calculate average degree
        avg_degree = (degree_u + degree_v) / 2
        
        # Assign average degree as edge degree centrality
        edge_degree[edge] = avg_degree
    
    return edge_degree

def edge_metric_compute(metric, graph):
    """
    Compute a specified edge metric by averaging the provided node metrics for each edge.

    Args:
        metric (dict): A dictionary where keys are nodes and values are the metric values for the nodes.
        graph (networkx.Graph): A NetworkX graph object.

    Returns:
        dict: A dictionary where keys are edges and values are the average metric values of the nodes incident to the edge.
    """
    edges = {}
    
    # Iterate over edges
    for edge in graph.edges():
        u, v = edge
        
        metric_u = metric[u]
        metric_v = metric[v]
        
        avg_centrality = (metric_u + metric_v) / 2
        
        edges[edge] = avg_centrality
    
    return edges

def old_node_flow_metric(G, iterations):
    # Determine the type of the features
    sample_feature = G.nodes[next(iter(G.nodes))]['x']
    if isinstance(sample_feature, np.ndarray):
        use_torch = False
    elif isinstance(sample_feature, torch.Tensor):
        use_torch = True
    else:
        raise TypeError("Node features must be either NumPy arrays or PyTorch tensors.")
    
    def aggregate_neighborhood_features(A, features):
        if use_torch:
            return torch.matmul(A, features) / A.sum(dim=1, keepdim=True)
        else:
            return A.dot(features) / A.sum(axis=1, keepdims=True)
    
    A = nx.to_numpy_array(G)
    A += np.eye(len(G.nodes))
    
    if use_torch:
        A = torch.tensor(A, dtype=torch.float32)
        features = torch.stack([G.nodes[node]['x'] for node in G.nodes])
        deltas = torch.zeros((G.number_of_nodes(), iterations), dtype=torch.float32)
        second_deltas = torch.zeros((G.number_of_nodes(), iterations - 1), dtype=torch.float32)
    else:
        features = np.array([G.nodes[node]['x'] for node in G.nodes])
        deltas = np.zeros((G.number_of_nodes(), iterations))
        second_deltas = np.zeros((G.number_of_nodes(), iterations - 1))
    
    for step in range(iterations):
        new_features = aggregate_neighborhood_features(A, features)
        if use_torch:
            deltas[:, step] = torch.norm(new_features - features, dim=1)
        else:
            deltas[:, step] = np.linalg.norm(new_features - features, axis=1)
        features = new_features

    # Calculate second differences
    for step in range(1, iterations):
        if use_torch:
            second_deltas[:, step - 1] = torch.abs(deltas[:, step] - deltas[:, step - 1])
        else:
            second_deltas[:, step - 1] = np.abs(deltas[:, step] - deltas[:, step - 1])

    # Calculate sum for every node
    if use_torch:
        sum_deltas = deltas.sum(dim=1)
        sum_second_deltas = second_deltas.sum(dim=1)
    else:
        sum_deltas = np.sum(deltas, axis=1)
        sum_second_deltas = np.sum(second_deltas, axis=1)
    
    # Normalize the results to 0/1 values
    def normalize(data):
        data_min = data.min()
        data_max = data.max()
        return (data - data_min) / (data_max - data_min)
    
    if use_torch:
        normalized_sum_deltas = normalize(sum_deltas)
        normalized_sum_second_deltas = normalize(sum_second_deltas)
    else:
        normalized_sum_deltas = normalize(sum_deltas)
        normalized_sum_second_deltas = normalize(sum_second_deltas)

    # Normalize to normal standard deviation
    def normalize_std(data):
        return (data - data.mean()) / data.std()
    
    if use_torch:
        normalized_sum_deltas = normalize_std(normalized_sum_deltas)
        normalized_sum_second_deltas = normalize_std(normalized_sum_second_deltas)
    else:
        normalized_sum_deltas = normalize_std(normalized_sum_deltas)
        normalized_sum_second_deltas = normalize_std(normalized_sum_second_deltas)
    
    # Calculate the final score
    score = 1 - normalized_sum_deltas + normalized_sum_second_deltas
    
    return score

def node_flow_metric(G, iterations, demo=False, aggregate=True):
    """
    Compute a node flow metric for a graph by analyzing the changes in node features over several iterations.

    Args:
        G (networkx.Graph): A NetworkX graph object where nodes have 'x' attributes representing their features.
        iterations (int): The number of iterations to perform in the flow metric computation.

    Returns:
        torch.Tensor or np.ndarray: A tensor or array of scores for each node in the graph based on their feature flow.
    """
    # Determine the type of the features
    sample_feature = G.nodes[next(iter(G.nodes))]['x']
    if isinstance(sample_feature, np.ndarray):
        use_torch = False
    elif isinstance(sample_feature, torch.Tensor):
        use_torch = True
    else:
        raise TypeError("Node features must be either NumPy arrays or PyTorch tensors.")
    
    def aggregate_neighborhood_features(A, features):
        if use_torch:
            # Ensure both tensors are of type float
            A = A.float()
            features = features.float()
            return torch.matmul(A, features) / A.sum(dim=1, keepdim=True)
        else:
            return A.dot(features) / A.sum(axis=1, keepdims=True)
    
    if aggregate:
        A = nx.to_numpy_array(G)
        A += np.eye(len(G.nodes))

    # Calculate node degrees and max degree
    #node_degrees = np.array([G.degree(node) for node in G.nodes])
    #max_degree = node_degrees.max()
    #normalization_factors = max_degree + 1 - node_degrees

    if use_torch:
        if aggregate:
            A = torch.tensor(A, dtype=torch.float32)
        features = torch.stack([G.nodes[node]['x'] for node in G.nodes])
        deltas = torch.zeros((G.number_of_nodes(), iterations), dtype=torch.float32)
        second_deltas = torch.zeros((G.number_of_nodes(), iterations - 1), dtype=torch.float32)
        #normalization_factors = torch.tensor(normalization_factors, dtype=torch.float32)
    else:
        features = np.array([G.nodes[node]['x'] for node in G.nodes])
        deltas = np.zeros((G.number_of_nodes(), iterations))
        second_deltas = np.zeros((G.number_of_nodes(), iterations - 1))

    for step in range(iterations):
        if aggregate:
            new_features = aggregate_neighborhood_features(A, features)
        else:
            new_features = features
        if use_torch:
            deltas[:, step] = torch.norm(new_features - features, dim=1)# / normalization_factors
        else:
            deltas[:, step] = np.linalg.norm(new_features - features, axis=1)# / normalization_factors
        features = new_features

    # Calculate second differences
    for step in range(1, iterations):
        if use_torch:
            second_deltas[:, step - 1] = torch.abs(deltas[:, step] - deltas[:, step - 1])
        else:
            second_deltas[:, step - 1] = np.abs(deltas[:, step] - deltas[:, step - 1])

    # Calculate exponential moving average for deltas
    alpha = 0.6
    if use_torch:
        mean_deltas = torch.zeros(G.number_of_nodes(), dtype=torch.float32)
        for i in range(G.number_of_nodes()):
            mean_deltas[i] = deltas[i, 0]
            for t in range(1, iterations):
                mean_deltas[i] = alpha * deltas[i, t] + (1 - alpha) * mean_deltas[i]
    else:
        mean_deltas = np.zeros(G.number_of_nodes())
        for i in range(G.number_of_nodes()):
            mean_deltas[i] = deltas[i, 0]
            for t in range(1, iterations):
                mean_deltas[i] = alpha * deltas[i, t] + (1 - alpha) * mean_deltas[i]

    # Directly calculate the standard deviation of the second deltas
    if use_torch:
        std_second_deltas = torch.std(second_deltas, dim=1)
    else:
        std_second_deltas = np.std(second_deltas, axis=1)

    if False:
        def compute_exponential_moving_std(new_value, old_variance, old_mean, alpha=0.6):
            if old_variance is None:
                return 0.0
                
            # Update the mean with exponential smoothing
            new_mean = alpha * new_value + (1 - alpha) * old_mean
                
            # Update the variance with exponential smoothing
            new_variance = alpha * (new_value - new_mean) ** 2 + (1 - alpha) * old_variance
                
            # Return the square root of the variance (standard deviation)
            return new_variance ** 0.5, new_mean, new_variance

        if use_torch:
            # Initialize the old mean and variance (can be set to zero or the first value)
            old_mean = torch.zeros_like(second_deltas[:, 0])
            old_variance = torch.zeros_like(second_deltas[:, 0])
            
            for i in range(second_deltas.size(1)):
                new_value = second_deltas[:, i]
                old_variance, old_mean, std_second_deltas = compute_exponential_moving_std(new_value, old_variance, old_mean, alpha)
        else:
            # Initialize the old mean and variance (can be set to zero or the first value)
            old_mean = np.zeros_like(second_deltas[:, 0])
            old_variance = np.zeros_like(second_deltas[:, 0])
            std_second_deltas = np.zeros_like(second_deltas[:, 0])
                
            for i in range(second_deltas.shape[1]):
                new_value = second_deltas[:, i]
                std_second_deltas, old_mean, old_variance = compute_exponential_moving_std(new_value, old_variance, old_mean, alpha)

    # Calculate the final score
    score = (std_second_deltas + 1) / (mean_deltas + 1)

    if demo:
        return score, deltas, second_deltas
    else:
        return score

def normalize(data):
    """
    Normalize the data to have mean 0 and standard deviation 1.
    
    Args:
        data (numpy array, torch tensor, or dict): The data to normalize.
    
    Returns:
        numpy array: Normalized data.
    """
    if isinstance(data, dict):
        # Convert dictionary values to a list
        keys = list(data.keys())
        values = np.array(list(data.values()))
    elif torch.is_tensor(data):
        values = data.detach().cpu().numpy()
    else:
        values = np.array(data)
    
    mean = np.mean(values)
    std = np.std(values)
    normalized_values = (values - mean) / std

    return np.array(normalized_values)

