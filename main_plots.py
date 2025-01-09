import numpy as np
from sklearn import metrics as sk
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import networkx as nx
import numpy as np
import data.databases as datasets
import train.metrics as metrics
import train.utils as train_utils
import matplotlib.pyplot as plt
import visualization.utils as visual_utils
import matplotlib.cm as cm

def draw_graph_with_attributes(G, node_attribute1, node_attribute2, edge_attribute, filename='graph_with_attributes.png'):
    """
    Draw the graph and annotate nodes with their IDs and two specified attributes, as well as edges with a specified attribute,
    then save the plot as a .png file.
    
    Args:
        G (nx.Graph): The NetworkX graph.
        node_attribute1 (str): The first node attribute to display.
        node_attribute2 (str): The second node attribute to display.
        edge_attribute (str): The edge attribute to display.
        filename (str): The filename for the saved image. Default is 'graph_with_attributes.png'.
    """
    filename = "results/images/" + filename
    pos = nx.spring_layout(G)  # You can use other layouts as well
    
    # Get node attributes and prepare labels
    node_attributes1 = nx.get_node_attributes(G, node_attribute1)
    node_attributes2 = nx.get_node_attributes(G, node_attribute2)
    node_labels = {node: f'ID: {node}\n{node_attribute1}: {attr1}\n{node_attribute2}: {attr2}' 
                   for node, attr1 in node_attributes1.items() 
                   for attr2 in [node_attributes2[node]]}
    
    # Get edge attributes and prepare labels
    edge_attributes = nx.get_edge_attributes(G, edge_attribute)
    edge_labels = {(u, v): f'{edge_attribute}: {attr}' for (u, v), attr in edge_attributes.items()}

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')
    
    # Draw node labels with ID and attributes
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_color='red', font_size=10)
    
    # Draw edge labels with the specified attribute
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue', font_size=8)
    
    plt.title(f"Graph with Node IDs and Attributes: {node_attribute1}, {node_attribute2}, and Edge Attribute: {edge_attribute}")
    
    # Save the figure
    plt.savefig(filename, format='png')
    plt.close()
    print(f"Graph saved as {filename}")

def load_medmnist_data(view, split):
    """
    Load MedMNIST data for the given view and split.

    Args:
        view (str): Either 'c' or 's' indicating the dataset view.
        split (str): The dataset split, one of 'train', 'val', or 'test'.

    Returns:
        (np.ndarray, np.ndarray): Tuple of images and labels.
    """
    info = INFO[f'organ{view}mnist']
    DataClass = getattr(medmnist, info['python_class'])
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = DataClass(split=split, transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data in dataloader:
        images, labels = data
    images = images.numpy().squeeze()
    labels = labels.numpy().squeeze()
    return images, labels

def extract_demo_medmnist_data(view):
    images, labels = load_medmnist_data(view, 'train')
    
    # Dictionary to store images by class
    class_images = {label: [] for label in np.unique(labels)}
    
    # Collect images for each class
    for img, label in zip(images, labels):
        if label in class_images and len(class_images[label]) < 5:
            class_images[label].append(img)
    
    # Determine the number of unique labels
    num_labels = len(class_images)
    
    # Plot 5 images for each class
    fig, axs = plt.subplots(num_labels, 5, figsize=(15, num_labels * 3))
    for i, (label, imgs) in enumerate(class_images.items()):
        for j, img in enumerate(imgs):
            axs[i, j].imshow(img, cmap='gray')
            axs[i, j].axis('off')
        while len(imgs) < 5:  # If fewer than 5 images are available
            axs[i, len(imgs)].axis('off')
    
    # Save the figure
    plt.savefig('results/images/classes_images.png')
    plt.close()

def plot_node_values_over_time(values, highlight_nodes, filename='node_values_over_time.png', dpi=300):
    """
    Plot the values of nodes over time and save the plot to a file, 
    with specified nodes highlighted in color and others in gray.
    
    Args:
        values (np.ndarray): A 2D numpy array where dim 0 is the number of nodes and dim 1 is the number of steps.
        highlight_nodes (list): List of node ids to be highlighted in color.
        filename (str): The filename for the saved image. Default is 'node_values_over_time.png'.
    """
    filename = "results/images/" + filename
    num_nodes, num_steps = values.shape
    
    # Use a colormap to ensure distinguishable colors for highlighted nodes
    colormap = cm.get_cmap('tab20', len(highlight_nodes))  # 'tab20' has 20 distinct colors
    
    # Create a new figure
    plt.figure(figsize=(12, 8))
    
    # Plot each node's values over time
    for node in range(num_nodes):
        if node in highlight_nodes:
            color = colormap(highlight_nodes.index(node) / len(highlight_nodes))
        else:
            color = 'lightgray'
        plt.plot(values[node], color=color, label=f'Node {node}' if node in highlight_nodes else "", linewidth=2 if node in highlight_nodes else 1)
    
    # Add title and labels with increased font size
    plt.title('Node Values Over Time', fontsize=18)
    plt.xlabel('Time Step', fontsize=16)
    plt.ylabel('Value', fontsize=16)
    plt.xticks(np.arange(0, num_steps, 1), fontsize=14)
    plt.yticks(fontsize=14)
    
    # Add legend for highlighted nodes and "Others"
    handles, labels = plt.gca().get_legend_handles_labels()
    if highlight_nodes:
        # Create custom legend handles for highlighted nodes
        custom_handles = [plt.Line2D([0], [0], color=colormap(i / len(highlight_nodes)), linewidth=2) for i in range(len(highlight_nodes))]
        # Add a custom handle for "Others"
        custom_handles.append(plt.Line2D([0], [0], color='lightgray', linewidth=2))
        labels = [f'Node {node}' for node in highlight_nodes] + ['Others']
        # Increase the font size of the legend and move it inside the plot
        plt.legend(custom_handles, labels, loc='upper right', fontsize=14, bbox_to_anchor=(0.98, 0.98))
    
    # Save the plot to a file
    plt.savefig(filename, format='png', bbox_inches='tight', dpi=dpi)
    plt.close()
    print(f"Plot saved as {filename}")


def add_edges_in_range(G, start, end):
    """
    Add all edges connecting nodes in the range [start, end].
    
    Args:
        G (nx.Graph): The NetworkX graph.
        start (int): The starting node in the range.
        end (int): The ending node in the range.
    """
    nodes_range = range(start, end + 1)
    edges_to_add = [(i, j) for i in nodes_range for j in nodes_range if i != j]
    G.add_edges_from(edges_to_add)



view = 'c'
extract_demo_medmnist_data(view)
x, y, edge_index, train_mask, val_mask, test_mask = datasets.get_organ_dataset(view = view, demo=True)

G = nx.Graph()
nodes = [i for i in range(x.shape[0])]
    
# Add nodes with attributes
for i in nodes:
        G.add_node(i, x=x[i], y=y[i], train=train_mask[i], val=val_mask[i], test=test_mask[i])


# Add edges manually
add_edges_in_range(G,0,3)
G.add_edge(3, 4)
G.add_edge(4, 5)
add_edges_in_range(G,5,8)
G.add_edge(8, 9)
G.add_edge(9, 10)
add_edges_in_range(G,10,13)
G.add_edge(13, 14)
G.add_edge(14, 0)

nodes_topo_results, deltas, second_deltas = train_utils.node_flow_metric(G, 5, demo = True)
edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G)
for node in G.nodes():
        G.nodes[node]['topo'] = nodes_topo_results[node]
for edge, degree_centrality in edges_topo_results.items():
        G[edge[0]][edge[1]]['topo'] = degree_centrality

overall_mean = visual_utils.plot_mse_results(nodes_topo_results, filename='flow_distribution_demo.png')
plot_node_values_over_time(deltas, highlight_nodes=[4,9,14], filename="deltas_distribution.png")
plot_node_values_over_time(second_deltas, highlight_nodes=[4,9,14], filename="second_deltas_distribution.png")
draw_graph_with_attributes(G,node_attribute1='topo', node_attribute2='y', edge_attribute='topo', filename='graph_with_attributes.png')

# Filtering
G.remove_edge(3, 4)
G.remove_edge(14, 0)
G.remove_edge(4, 5)
G.remove_edge(14, 13)

#G.remove_edge(2, 3)
#G.remove_edge(10, 11)
#G.remove_edge(11, 12)
#G.remove_edge(2, 1)

nodes_topo_results, deltas, second_deltas = train_utils.node_flow_metric(G, 5, demo = True)
edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G)
for node in G.nodes():
        G.nodes[node]['topo'] = nodes_topo_results[node]
for edge, degree_centrality in edges_topo_results.items():
        G[edge[0]][edge[1]]['topo'] = degree_centrality

visual_utils.plot_mse_results(nodes_topo_results, filename='flow_distribution_demo_after.png', additional_line=overall_mean)
plot_node_values_over_time(deltas, highlight_nodes=[4,9,14], filename="deltas_distribution_after.png")
plot_node_values_over_time(second_deltas, highlight_nodes=[4,9,14], filename="second_deltas_distribution_after.png")
draw_graph_with_attributes(G,node_attribute1='topo', node_attribute2='y', edge_attribute='topo', filename='graph_with_attributes_after.png')