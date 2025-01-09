import data.databases as datasets
import torch
import numpy as np
import models.gcn_model_batching as gcn_model
import models.delta_gnn_model_batching as dual_gcn_model
import models.gin_model_batching as gin_model
import models.gat_model_batching as gat_model
import train.train_multi_class_classification_multigraph as train
import train.train_regression_multigraph as train_regression
import results.writer as writer
import argparse
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import random
import visualization.utils as visual_utils
import train.utils as train_utils
import train.metrics as metrics
from torch_geometric.data import Data, DataLoader
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout
import os
import io
from copy import deepcopy
import math


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='GCN')
parser.add_argument('--max_hop', type=int, default=1)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=2048)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--num_batch', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--multilabel', action='store_true')
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--residual', action='store_true')
parser.add_argument('--print_result', action='store_true')
parser.add_argument('--dataset_name', type=str, default='Organ-C')
parser.add_argument('--aggregations_flow', type=int, default=5)
parser.add_argument('--max_communities', type=int, default=1000)
parser.add_argument('--head_depth', type=int, default=1)
parser.add_argument('--remove_edges', type=int, default=0, help='Number of edges with lowest curvature to remove')
parser.add_argument('--make_unbalanced', action='store_true')
parser.add_argument('--dense', action='store_true')
parser.add_argument('--flow_control', action='store_true')
parser.add_argument('--flow_degradation', action='store_true')
parser.add_argument('--flow_density', type=float, default=0.95, help='Density preserved through the flow control')
parser.add_argument('--topological_measure', type=str, default='none')
parser.add_argument('--plot_distribution', action='store_true')
parser.add_argument('--distribution_comparison', action='store_true')
parser.add_argument('--linear', action='store_true')

args = parser.parse_args()
print(args)

model_type = args.model_type
max_hop = args.max_hop
layers = args.layers
hidden_channels = args.hidden_channels
dropout = args.dropout
batch_norm = args.batch_norm
lr = args.lr
num_batch = args.num_batch
num_epoch = args.num_epoch
multilabel = args.multilabel
do_evaluation = args.do_eval
residual = args.residual
print_result = args.print_result
dataset_name = args.dataset_name
remove_edges = args.remove_edges
aggregations_flow = args.aggregations_flow
max_communities = args.max_communities
head_depth = args.head_depth
topological_measure = args.topological_measure
make_unbalanced = args.make_unbalanced
dense = args.dense
flow_control = args.flow_control
flow_degradation = args.flow_degradation
flow_density = args.flow_density
plot_distribution = args.plot_distribution
distribution_comparison = args.distribution_comparison
linear = args.linear

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def suppress_print(func, *args, **kwargs):
    f = io.StringIO()
    with redirect_stdout(f):
        func(*args, **kwargs)

#%% Prepare Dataset
# get dataset

assert dataset_name in ["PascalVOC-SP", "COCO-SP", "PCQM-Contact", "Peptides-func", "Peptides-struct","PROTEINS", "ENZYMES", "MUTAG","COLLAB","IMDB-BINARY","REDDIT-BINARY"]
assert model_type == "DeltaGNN" or model_type == "GCN" or model_type == "GIN" or model_type == "GAT"

regression = dataset_name == "Peptides-struct"
use_accuracy = dataset_name in ["PROTEINS", "ENZYMES", "MUTAG","COLLAB","IMDB-BINARY","REDDIT-BINARY"]

if dataset_name in ["PascalVOC-SP", "COCO-SP", "PCQM-Contact", "Peptides-func", "Peptides-struct"]:
        (train_x_list, train_y_list, train_edge_index_list, train_dataset) = datasets.get_lrgb_dataset(dataset_name, split='train')
        (val_x_list, val_y_list, val_edge_index_list, val_dataset) = datasets.get_lrgb_dataset(dataset_name, split='val')
        (test_x_list, test_y_list, test_edge_index_list, test_dataset) = datasets.get_lrgb_dataset(dataset_name, split='test')

        print("Training set size: " + str(len(train_x_list)))
        print("Validation set size: " + str(len(val_x_list)))
        print("Test set size: " + str(len(test_x_list)))

if dataset_name in ["PROTEINS", "ENZYMES", "MUTAG","COLLAB","IMDB-BINARY","REDDIT-BINARY"]:
        x_list, y_list, edge_index_list, dataset = datasets.get_tudataset(dataset_name)

        train_ratio = 0.8
        val_ratio = 0.1
        test_ratio = 0.1

        # Number of samples in the dataset
        num_samples = len(dataset)

        # Generate indices for splitting
        indices = torch.randperm(num_samples)

        train_size = int(num_samples * train_ratio)
        val_size = int(num_samples * val_ratio)

        print("Training set size: " + str(train_size))
        print("Validation set size: " + str(val_size))
        print("Test set size: " + str(num_samples - train_size - val_size))

        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]

        # Create subsets using list comprehensions
        train_x_list = [x_list[i] for i in train_indices]
        val_x_list = [x_list[i] for i in val_indices]
        test_x_list = [x_list[i] for i in test_indices]
        train_y_list = [y_list[i] for i in train_indices]
        val_y_list = [y_list[i] for i in val_indices]
        test_y_list = [y_list[i] for i in test_indices]
        train_edge_index_list = [edge_index_list[i] for i in train_indices]
        val_edge_index_list = [edge_index_list[i] for i in val_indices]
        test_edge_index_list = [edge_index_list[i] for i in test_indices]

duo_architecture = model_type == "DeltaGNN"
data_list = []
data_cond_list = []
num_features = train_x_list[0].shape[1]

if train_y_list[0].dim() == 2:
        num_classes = train_y_list[0].shape[1]
else:
        num_classes = int(torch.cat(train_y_list).max().item() + 1)

print("Input Features:" + str(num_features))
print("Output Features:" + str(num_classes))

data_cond = None
for split in ['train','val','test']:
        if split == 'train':
                x_list = train_x_list
                y_list = train_y_list
                edge_index_list = train_edge_index_list
                train_data_list = []
                train_data_cond_list = []
        if split == 'val':
                x_list = val_x_list
                y_list = val_y_list
                edge_index_list = val_edge_index_list
                val_data_list = []
                val_data_cond_list = []
        if split == 'test':
                x_list = test_x_list
                y_list = test_y_list
                edge_index_list = test_edge_index_list
                test_data_list = []
                test_data_cond_list = []
        for x, y, edge_index in zip(x_list, y_list, edge_index_list):
                G = train_utils.construct_graph(x, y, edge_index)
                forman_curvature = FormanRicci(G)
                ricci_curvature = OllivierRicci(G)
                G_topo = G.copy()
                data_cond = None

                if topological_measure != "none" and not flow_control:
                        if topological_measure == "ricci_curvature":
                                suppress_print(ricci_curvature.compute_ricci_curvature)
                                G_topo = ricci_curvature.G
                                # Rename edge/node attribute 'RicciCurvature' to 'topo'
                                for node in G_topo.nodes():
                                        if 'ricciCurvature' in G_topo.nodes[node]:
                                                G_topo.nodes[node]['topo'] = G_topo.nodes[node].pop('ricciCurvature')
                                        else:
                                                G_topo.nodes[node]['topo'] = 0
                                for u, v in G_topo.edges():
                                        if 'ricciCurvature' in G_topo[u][v]:
                                                G_topo[u][v]['topo'] = G_topo[u][v].pop('ricciCurvature')
                                        else:
                                                G_topo[u][v]['topo'] = 0
                                nodes_topo_results = np.array([G_topo.nodes[node]['topo'] for node in G_topo.nodes() if 'topo' in G_topo.nodes[node]])
                                normalized_nodes_curv_results = metrics.normalize(nodes_topo_results)
                        if topological_measure == "forman_curvature":
                                suppress_print(forman_curvature.compute_ricci_curvature)
                                G_topo = forman_curvature.G
                                # Rename edge/node attribute 'formanCurvature' to 'topo'
                                for node in G_topo.nodes():
                                        if 'formanCurvature' in G_topo.nodes[node]:
                                                G_topo.nodes[node]['topo'] = G_topo.nodes[node].pop('formanCurvature')
                                        else:
                                                G_topo.nodes[node]['topo'] = 0
                                for u, v in G_topo.edges():
                                        if 'formanCurvature' in G_topo[u][v]:
                                                G_topo[u][v]['topo'] = G_topo[u][v].pop('formanCurvature')
                                        else:
                                                G_topo[u][v]['topo'] = 0
                                nodes_topo_results = np.array([G_topo.nodes[node]['topo'] for node in G_topo.nodes() if 'topo' in G_topo.nodes[node]])
                                normalized_nodes_curv_results = metrics.normalize(nodes_topo_results)
                        if topological_measure == "degree_centrality":
                                nodes_topo_results = nx.degree_centrality(G_topo)
                                edges_topo_results = train_utils.edge_degree_centrality(G_topo)
                                if not distribution_comparison:
                                        for node in G_topo.nodes():
                                                G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                                        for edge, degree_centrality in edges_topo_results.items():
                                                G_topo[edge[0]][edge[1]]['topo'] = degree_centrality
                                normalized_nodes_degree_results = metrics.normalize(nodes_topo_results)
                        if topological_measure == "betweenness_centrality":
                                nodes_topo_results = nx.betweenness_centrality(G_topo)
                                edges_topo_results = nx.edge_betweenness_centrality(G_topo)
                                for node in G_topo.nodes():
                                        G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                                for u,v in G_topo.edges():
                                        G_topo[u][v]['topo'] = edges_topo_results[(u,v)]
                                normalized_nodes_betw_results = metrics.normalize(nodes_topo_results)
                        if topological_measure == "closeness_centrality":
                                nodes_topo_results = nx.closeness_centrality(G_topo)
                                edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G_topo)
                                for node in G_topo.nodes():
                                        G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                                for u,v in G_topo.edges():
                                        G_topo[u][v]['topo'] = edges_topo_results[(u,v)]
                                normalized_nodes_betw_results = metrics.normalize(nodes_topo_results)
                        if topological_measure == "eigenvector_centrality":
                                nodes_topo_results = nx.eigenvector_centrality(G_topo, tol=1e-03)
                                edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G_topo)
                                if not distribution_comparison:
                                        for node in G_topo.nodes():
                                                G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                                        for edge, degree_centrality in edges_topo_results.items():
                                                G_topo[edge[0]][edge[1]]['topo'] = degree_centrality
                                normalized_nodes_eig_results = metrics.normalize(nodes_topo_results)
                        if topological_measure == "random":
                                for node in G_topo.nodes():
                                        G_topo.nodes[node]['topo'] = random.random()
                                for u,v in G_topo.edges():
                                        G_topo[u][v]['topo'] = random.random()
                        if topological_measure == "flow":
                                nodes_topo_results = train_utils.node_flow_metric(G_topo, aggregations_flow)
                                edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G_topo)
                                for node in G_topo.nodes():
                                        G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                                for edge, degree_centrality in edges_topo_results.items():
                                        G_topo[edge[0]][edge[1]]['topo'] = degree_centrality
                                normalized_nodes_flow_results = metrics.normalize(nodes_topo_results)

                        if remove_edges > 0:
                                edge_topo = [(u, v, G_topo[u][v]['topo']) for u, v in G_topo.edges()]
                                edge_topo.sort(key=lambda x: x[2])
                                num_edges_to_remove = min(remove_edges, len(edge_topo))
                                edges_to_remove_high = edge_topo[-num_edges_to_remove:]
                                edges_to_remove_low = edge_topo[:num_edges_to_remove]
                                if topological_measure in ["flow","ricci_curvature","forman_curvature"]:
                                        edges_to_remove = edges_to_remove_low
                                else:
                                        edges_to_remove = edges_to_remove_high
                                G_topo.remove_edges_from([(u, v) for u, v, _ in edges_to_remove])
                        if model_type == "DeltaGNN":
                                if G_topo.number_of_edges() > 0:
                                        edge_list = list(G_topo.edges())
                                        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                                else:
                                        edge_index = torch.empty((2, 0), dtype=torch.long)
                                connected_components = list(nx.connected_components(G_topo))
                                connected_components.sort(key=len, reverse=True)  
                                selected_components = connected_components[:min(len(connected_components), max_communities)]


                                max_topo_nodes = []
                                for component in selected_components:
                                        max_node = None
                                        max_topo = float('-inf')
                                        for node in component:    
                                                topo = G_topo.nodes[node]['topo']
                                                if topo > max_topo:
                                                        max_topo = topo
                                                        max_node = node
                                        if max_node is not None:
                                                max_topo_nodes.append(max_node)
                                G_topo.remove_edges_from(list(G_topo.edges()))
                                        
                                if len(selected_components)>1:
                                        edges_to_add = [(max_topo_nodes[i], max_topo_nodes[j]) for i in range(len(max_topo_nodes)) for j in range(i + 1, len(max_topo_nodes))]
                                        G_topo.add_edges_from(edges_to_add)
                                        edge_list_cond = list(G_topo.edges())
                                        edge_index_cond = torch.tensor(edge_list_cond, dtype=torch.long).t().contiguous()
                                else:
                                        edge_index_cond = torch.empty((2, 0), dtype=torch.long)
                        else:
                                if G_topo.number_of_edges() > 0:
                                        edge_list = list(G_topo.edges())
                                        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                                else:
                                        edge_index = torch.empty((2, 0), dtype=torch.long)

                x = torch.tensor(x, dtype=torch.float32)
                y = torch.tensor(y)
                if model_type == "DeltaGNN" and not flow_control:
                        data = Data(x=x, edge_index=edge_index, y=y)
                        data_cond = Data(x=x, edge_index=edge_index_cond, y=y)
                else:
                        data = Data(x=x, edge_index=edge_index, y=y)
                if split == 'train':
                        train_data_list.append(data)
                        if data_cond is not None:
                                train_data_cond_list.append(data_cond)
                if split == 'val':
                        val_data_list.append(data)
                        if data_cond is not None:
                                val_data_cond_list.append(data_cond)
                if split == 'test':
                        test_data_list.append(data)
                        if data_cond is not None:
                                test_data_cond_list.append(data_cond)

# Create DataLoaders
train_cond_loader = None
val_cond_loader = None
test_cond_loader = None
train_loader = DataLoader(train_data_list, batch_size=num_batch)
val_loader = DataLoader(val_data_list, batch_size=num_batch)
test_loader = DataLoader(test_data_list, batch_size=num_batch)
if train_data_cond_list:
        train_cond_loader = DataLoader(train_data_cond_list, batch_size=num_batch)
        val_cond_loader = DataLoader(val_data_cond_list, batch_size=num_batch)
        test_cond_loader = DataLoader(test_data_cond_list, batch_size=num_batch)


if model_type == "DeltaGNN":
        model = dual_gcn_model.DeltaGNN(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= num_features, out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual, 
                        density= flow_density, max_communities=max_communities, linear=linear)
if model_type == "GCN":
        model = gcn_model.GCN(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= num_features, out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual,
                        head_depth=head_depth)
if model_type == "GIN":
        model = gin_model.GIN(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= num_features, out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual,
                        head_depth=head_depth)
if model_type == "GAT":
        model = gat_model.GAT(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= num_features, out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual,
                        head_depth=head_depth)

print(model)

# Print the number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

if not regression:
        if do_evaluation:
                if model_type == "DeltaGNN":
                        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
                        max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
                        train_memory, train_time_avg) = train.train(model, device, 
                                                                        train_loader = train_loader,
                                                                        val_loader = val_loader,
                                                                        test_loader = test_loader,
                                                                        train_cond_loader = train_cond_loader,
                                                                        val_cond_loader = val_cond_loader,
                                                                        test_cond_loader = test_cond_loader,
                                                                        multilabel=multilabel, 
                                                                        lr=lr, num_epoch=num_epoch, 
                                                                        flow_flag=flow_control,
                                                                        flow_control=flow_degradation, 
                                                                        duo_architecture = duo_architecture,
                                                                        use_accuracy=use_accuracy)
                else:
                        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
                        max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
                        train_memory, train_time_avg) = train.train(model, device, 
                                                                        train_loader = train_loader,
                                                                        val_loader = val_loader,
                                                                        test_loader = test_loader,
                                                                        multilabel=multilabel, 
                                                                        lr=lr, num_epoch=num_epoch,
                                                                        duo_architecture = duo_architecture,
                                                                        use_accuracy=use_accuracy)
        else:
                if model_type == "DeltaGNN":
                        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
                        max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
                        train_memory, train_time_avg) = train.train(model, device, 
                                                                        train_loader = train_loader,
                                                                        train_cond_loader = train_cond_loader,
                                                                        multilabel=multilabel, 
                                                                        lr=lr, num_epoch=num_epoch,
                                                                        flow_flag=flow_control, 
                                                                        flow_control=flow_degradation,
                                                                        duo_architecture = duo_architecture,
                                                                        use_accuracy=use_accuracy)
                else:
                        (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
                        max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
                        train_memory, train_time_avg) = train.train(model, device, 
                                                                        train_loader = train_loader,
                                                                        multilabel=multilabel, 
                                                                        lr=lr, num_epoch=num_epoch,
                                                                        duo_architecture = duo_architecture,
                                                                        use_accuracy=use_accuracy)

        if print_result:
                writer.write_result(dataset_name, model_type, num_epoch, num_features, 
                                        max_val_acc, max_val_f1, max_val_sens, max_val_spec, 
                                        max_val_test_acc, max_val_test_f1, max_val_test_sens, 
                                        max_val_test_spec, session_memory, train_memory, 
                                        train_time_avg, filename = "result_"+dataset_name + ".csv", hidden_dimension=hidden_channels, max_communities=max_communities, remove_edges=remove_edges, topological_measure = topological_measure, make_unbalanced = make_unbalanced, dense = dense, flow_control=flow_control, linear=linear)
else:
        if do_evaluation:
                if model_type == "DeltaGNN":
                        (min_mae_val, max_r2_val, min_mse_val, min_rmse_val, min_mae_test,
            max_r2_test, min_mse_test, min_rmse_test, session_memory, 
            train_memory, train_time_avg) = train_regression.train(model, device, 
                                                                        train_loader = train_loader,
                                                                        val_loader = val_loader,
                                                                        test_loader = test_loader,
                                                                        train_cond_loader = train_cond_loader,
                                                                        val_cond_loader = val_cond_loader,
                                                                        test_cond_loader = test_cond_loader,
                                                                        multilabel=multilabel, 
                                                                        lr=lr, num_epoch=num_epoch, 
                                                                        flow_flag=flow_control,
                                                                        flow_control=flow_degradation, 
                                                                        duo_architecture = duo_architecture)
                else:
                        (min_mae_val, max_r2_val, min_mse_val, min_rmse_val, min_mae_test,
            max_r2_test, min_mse_test, min_rmse_test, session_memory, 
            train_memory, train_time_avg) = train_regression.train(model, device, 
                                                                        train_loader = train_loader,
                                                                        val_loader = val_loader,
                                                                        test_loader = test_loader,
                                                                        multilabel=multilabel, 
                                                                        lr=lr, num_epoch=num_epoch,
                                                                        duo_architecture = duo_architecture)
        else:
                if model_type == "DeltaGNN":
                        (min_mae_val, max_r2_val, min_mse_val, min_rmse_val, min_mae_test,
            max_r2_test, min_mse_test, min_rmse_test, session_memory, 
            train_memory, train_time_avg) = train_regression.train(model, device, 
                                                                        train_loader = train_loader,
                                                                        train_cond_loader = train_cond_loader,
                                                                        multilabel=multilabel, 
                                                                        lr=lr, num_epoch=num_epoch,
                                                                        flow_flag=flow_control, 
                                                                        flow_control=flow_degradation,
                                                                        duo_architecture = duo_architecture)
                else:
                        (min_mae_val, max_r2_val, min_mse_val, min_rmse_val, min_mae_test,
            max_r2_test, min_mse_test, min_rmse_test, session_memory, 
            train_memory, train_time_avg) = train_regression.train(model, device, 
                                                                        train_loader = train_loader,
                                                                        multilabel=multilabel, 
                                                                        lr=lr, num_epoch=num_epoch,
                                                                        duo_architecture = duo_architecture)

        if print_result:
                writer.write_result_regression(dataset_name, model_type, num_epoch, num_features, 
                                        min_mae_val, max_r2_val, min_mse_val, min_rmse_val, min_mae_test,
            max_r2_test, min_mse_test, min_rmse_test, session_memory, train_memory, 
                                        train_time_avg, filename = "result_regression_"+dataset_name + ".csv", hidden_dimension=hidden_channels, max_communities=max_communities, remove_edges=remove_edges, topological_measure = topological_measure, make_unbalanced = make_unbalanced, dense = dense, flow_control= flow_control, linear = linear)
# %%
