import data.databases as datasets
import torch
import numpy as np
import models.gcn_model as gcn_model
import models.delta_gnn_model as dual_gcn_model
import models.gin_model as gin_model
import models.gat_model as gat_model
import train.train_multi_class_classification as train
import results.writer as writer
import argparse
import networkx as nx
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import random
import visualization.utils as visual_utils
import train.utils as train_utils
import train.metrics as metrics
from contextlib import redirect_stdout
import os
import io
from copy import deepcopy
import time


parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='GCN')
parser.add_argument('--max_hop', type=int, default=1)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=2048)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--num_batch', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--multilabel', action='store_true')
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--residual', action='store_true')
parser.add_argument('--print_result', action='store_true')
parser.add_argument('--dataset_name', type=str, default='Organ-C')
parser.add_argument('--aggregations_flow', type=int, default=5)
parser.add_argument('--max_communities', type=int, default=1000)
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

assert dataset_name in ['Organ-C', 'Organ-S', 'Cora', 'CiteSeer', 'PubMed', "Cornell", "Texas", "Wisconsin","PascalVOC-SP", "COCO-SP", "PCQM-Contact", "Peptides-func", "Peptides-struct"]
assert model_type == "DeltaGNN" or model_type == "GCN" or model_type == "GIN" or model_type == "GAT"


if dataset_name == 'Organ-C':
        (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_organ_dataset(view='C',sparse=not dense, balanced = not make_unbalanced)
if dataset_name == 'Organ-S':
        (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_organ_dataset(view='S',sparse=not dense, balanced = not make_unbalanced)
if dataset_name == "Cora" or dataset_name == "CiteSeer" or dataset_name == "PubMed":
        (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_planetoid_dataset(dataset_name)
if dataset_name in ["Cornell", "Texas", "Wisconsin"]:
        (x, y, edge_index, train_mask, val_mask, test_mask) = datasets.get_webkb_dataset(dataset_name)
num_features = x.shape[1]
num_classes = int(max(y) + 1)

# construct networkx graph
G = train_utils.construct_graph(x, y, edge_index, train_mask, val_mask, 
                                test_mask)
forman_curvature = FormanRicci(G)
ricci_curvature = OllivierRicci(G)
G_topo = deepcopy(G)
adj_train_cond = None
adj_val_cond = None
adj_test_cond = None
edge_train = None
edge_val = None
edge_test = None

#%% split graph to train, val, and test (inductive training)
(x_train, y_train, edge_train, train_mask, x_val, y_val, edge_val, val_mask, x_test, 
 y_test, edge_test, test_mask) = train_utils.split_graph(G, multilabel = multilabel)

start_time = time.time()
if topological_measure != "none" and not flow_control:
        graphs = []
        stages = []
        output_file="degree_distribution"+model_type+"_"+topological_measure+".png"
        if model_type == "DeltaGNN":
                graphs.append(train_utils.construct_normalized_adj(edge_train, x_train.shape[0]))
                stages.append("Original graph")

        if (topological_measure == "ricci_curvature" or topological_measure == "forman_curvature" or distribution_comparison) and not flow_control:
                if topological_measure == "ricci_curvature":
                        suppress_print(ricci_curvature.compute_ricci_curvature)
                        G_topo = ricci_curvature.G
                        # Rename edge/node attribute 'RicciCurvature' to 'topo'
                        for node in G_topo.nodes():
                                if 'ricciCurvature' in G_topo.nodes[node]:
                                        G_topo.nodes[node]['topo'] = G_topo.nodes[node].pop('ricciCurvature')
                        for u, v in G_topo.edges():
                                if 'ricciCurvature' in G_topo[u][v]:
                                        G_topo[u][v]['topo'] = G_topo[u][v].pop('ricciCurvature')
                        nodes_topo_results = np.array([G_topo.nodes[node]['topo'] for node in G_topo.nodes() if 'topo' in G_topo.nodes[node]])
                        normalized_nodes_curv_results = metrics.normalize(nodes_topo_results)
                        visual_utils.plot_mse_results(normalized_nodes_curv_results, filename='ricci_curvature_distribution.png')
                if topological_measure == "forman_curvature":
                        suppress_print(forman_curvature.compute_ricci_curvature)
                        G_topo = forman_curvature.G
                        # Rename edge/node attribute 'formanCurvature' to 'topo'
                        for node in G_topo.nodes():
                                if 'formanCurvature' in G_topo.nodes[node]:
                                        G_topo.nodes[node]['topo'] = G_topo.nodes[node].pop('formanCurvature')
                        for u, v in G_topo.edges():
                                if 'formanCurvature' in G_topo[u][v]:
                                        G_topo[u][v]['topo'] = G_topo[u][v].pop('formanCurvature')
                        nodes_topo_results = np.array([G_topo.nodes[node]['topo'] for node in G_topo.nodes() if 'topo' in G_topo.nodes[node]])
                        normalized_nodes_curv_results = metrics.normalize(nodes_topo_results)
                        visual_utils.plot_mse_results(normalized_nodes_curv_results, filename='forman_curvature_distribution.png')
        if topological_measure == "degree_centrality" or distribution_comparison:
                nodes_topo_results = nx.degree_centrality(G_topo)
                edges_topo_results = train_utils.edge_degree_centrality(G_topo)
                if not distribution_comparison:
                        for node in G_topo.nodes():
                                G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                        for edge, degree_centrality in edges_topo_results.items():
                                G_topo[edge[0]][edge[1]]['topo'] = degree_centrality
                normalized_nodes_degree_results = metrics.normalize(nodes_topo_results)
                visual_utils.plot_mse_results(normalized_nodes_degree_results, filename='degree_centrality_distribution.png')
        if topological_measure == "betweenness_centrality":
                nodes_topo_results = nx.betweenness_centrality(G_topo)
                edges_topo_results = nx.edge_betweenness_centrality(G_topo)
                for node in G_topo.nodes():
                        G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                for u,v in G_topo.edges():
                        G_topo[u][v]['topo'] = edges_topo_results[(u,v)]
                normalized_nodes_betw_results = metrics.normalize(nodes_topo_results)
                visual_utils.plot_mse_results(normalized_nodes_betw_results, filename='betweenness_centrality_distribution.png')
        if topological_measure == "closeness_centrality":
                nodes_topo_results = nx.closeness_centrality(G_topo)
                edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G_topo)
                for node in G_topo.nodes():
                        G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                for u,v in G_topo.edges():
                        G_topo[u][v]['topo'] = edges_topo_results[(u,v)]
                normalized_nodes_betw_results = metrics.normalize(nodes_topo_results)
                visual_utils.plot_mse_results(normalized_nodes_betw_results, filename='closeness_centrality_distribution.png')
        if topological_measure == "eigenvector_centrality" or distribution_comparison:
                nodes_topo_results = nx.eigenvector_centrality(G_topo, tol=1e-03)
                edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G_topo)
                if not distribution_comparison:
                        for node in G_topo.nodes():
                                G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                        for edge, degree_centrality in edges_topo_results.items():
                                G_topo[edge[0]][edge[1]]['topo'] = degree_centrality
                normalized_nodes_eig_results = metrics.normalize(nodes_topo_results)
                visual_utils.plot_mse_results(normalized_nodes_eig_results, filename='eigenvector_centrality_distribution.png')
        if topological_measure == "random":
                for node in G_topo.nodes():
                        G_topo.nodes[node]['topo'] = random.random()
                for u,v in G_topo.edges():
                        G_topo[u][v]['topo'] = random.random()
        if topological_measure == "flow" or distribution_comparison:
                nodes_topo_results = train_utils.node_flow_metric(G_topo, aggregations_flow)
                edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G_topo)
                if not distribution_comparison:
                        for node in G_topo.nodes():
                                G_topo.nodes[node]['topo'] = nodes_topo_results[node]
                        for edge, degree_centrality in edges_topo_results.items():
                                G_topo[edge[0]][edge[1]]['topo'] = degree_centrality
                normalized_nodes_flow_results = metrics.normalize(nodes_topo_results)
                visual_utils.plot_mse_results(normalized_nodes_flow_results, filename='flow_distribution.png')
        print(f"Number of edges before filtering: {G_topo.number_of_edges()}")
        if distribution_comparison:
                visual_utils.compare_distributions([normalized_nodes_curv_results,
                                                normalized_nodes_degree_results,
                                                normalized_nodes_eig_results,
                                                normalized_nodes_flow_results],
                                                ['Distribution Curvature',
                                                'Distribution Degree Centrality',
                                                'Distribution Eigenvalue Centrality',
                                                'Distribution Flow'])
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
        print(f"Number of edges after filtering: {G_topo.number_of_edges()}")
        


        if model_type == "DeltaGNN":
                #visual_utils.plot_degree_distribution(G_old,G_topo,output_file)
                G_hete = G_topo.copy()

        (_, _, edge_train, _, _, _, edge_val, _, _, 
 _, edge_test, _) = train_utils.split_graph(G_topo, multilabel = multilabel)
        
        if model_type == "DeltaGNN":
                graphs.append(train_utils.construct_normalized_adj(edge_train, x_train.shape[0]))
                stages.append("Homophilic graph - filtered")

        if model_type == "DeltaGNN":
                connected_components = list(nx.connected_components(G_hete))
                print(f"Connected components: {len(connected_components)}")
                connected_components.sort(key=len, reverse=True)  
                selected_components = connected_components[:min(len(connected_components), max_communities)]
                
                print("Finding max topo nodes")

                max_topo_nodes = []
                for component in selected_components:
                        max_node = None
                        max_topo = float('-inf')
                        for node in component:    
                                topo = G_hete.nodes[node]['topo']
                                if topo > max_topo:
                                        max_topo = topo
                                        max_node = node
                        if max_node is not None:
                                max_topo_nodes.append(max_node)
                G_hete.remove_edges_from(list(G_hete.edges()))

                print("Adding new edges")
                print(len(max_topo_nodes))

                edges_to_add = [(max_topo_nodes[i], max_topo_nodes[j]) for i in range(len(max_topo_nodes)) for j in range(i + 1, len(max_topo_nodes))]

                G_hete.add_edges_from(edges_to_add)
                (_, _, edge_train_cond, _, _, _, edge_val_cond, _, _,_, edge_test_cond, _) = train_utils.split_graph(G_hete, multilabel=multilabel)

                print("Generating new adj")
                adj_train_cond = train_utils.construct_normalized_adj(edge_train_cond, x_train.shape[0],self_loops = True)
                adj_val_cond = train_utils.construct_normalized_adj(edge_val_cond, x_val.shape[0],self_loops = True)
                adj_test_cond = train_utils.construct_normalized_adj(edge_test_cond, x_test.shape[0],self_loops = True)

                graphs.append(adj_train_cond)
                stages.append("Heterophilic graph - condensed")

                train_utils.plot_homophily_distributions(graphs,torch.tensor(x_train, dtype=torch.float32),torch.tensor(y_train),stages,topological_measure)
end_time = time.time()
elapsed_time = end_time - start_time
print("Time: "+str(elapsed_time))

adj_train = train_utils.construct_normalized_adj(edge_train, x_train.shape[0])
adj_val = train_utils.construct_normalized_adj(edge_val, x_val.shape[0])
adj_test = train_utils.construct_normalized_adj(edge_test, x_test.shape[0])


#%% Convert feature and labels to torch tensor
x_train = torch.tensor(x_train, dtype=torch.float32)
y_train = torch.tensor(y_train)
x_val = torch.tensor(x_val, dtype=torch.float32)
y_val = torch.tensor(y_val)
val_mask = torch.tensor(val_mask)
x_test = torch.tensor(x_test, dtype=torch.float32)
y_test = torch.tensor(y_test)
test_mask = torch.tensor(test_mask)


#%% Preparing Model
if model_type == "DeltaGNN":
        model = dual_gcn_model.DeltaGNN(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= x_train.shape[1], out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual, 
                        density= flow_density, max_communities=max_communities, linear=linear)
if model_type == "GCN":
        model = gcn_model.GCN(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= x_train.shape[1], out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual)
if model_type == "GIN":
        model = gin_model.GIN(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= x_train.shape[1], out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual)
if model_type == "GAT":
        model = gat_model.GAT(hidden_channels=hidden_channels, num_layers= layers, 
                        in_channels= x_train.shape[1], out_channels=num_classes, 
                        batch_norm=batch_norm, dropout=dropout, residual=residual)

print(model)

# Print the number of parameters
num_params = sum(p.numel() for p in model.parameters())
print(f"Number of parameters: {num_params}")

#%% Training
if do_evaluation:
        if model_type == "DeltaGNN":
                (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
                max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
                train_memory, train_time_avg) = train.train(model, device, 
                                                                x_train=x_train, 
                                                                y_train=y_train, 
                                                                adj_train=adj_train,
                                                                adj_train_cond=adj_train_cond,
                                                                x_val=x_val, 
                                                                y_val=y_val, 
                                                                adj_val=adj_val,
                                                                adj_val_cond=adj_val_cond,
                                                                val_mask=val_mask,
                                                                x_test=x_test, 
                                                                y_test=y_test, 
                                                                adj_test=adj_test,
                                                                adj_test_cond=adj_test_cond,
                                                                test_mask=test_mask,
                                                                multilabel=multilabel, 
                                                                lr=lr, num_epoch=num_epoch, 
                                                                flow_flag=flow_control,
                                                                plot_distribution=plot_distribution,
                                                                flow_control=flow_degradation)
        else:
                (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
                max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
                train_memory, train_time_avg) = train.train(model, device, 
                                                                x_train=x_train, 
                                                                y_train=y_train, 
                                                                adj_train=adj_train,
                                                                x_val=x_val, 
                                                                y_val=y_val, 
                                                                adj_val=adj_val,
                                                                val_mask=val_mask,
                                                                x_test=x_test, 
                                                                y_test=y_test, 
                                                                adj_test=adj_test,
                                                                test_mask=test_mask,
                                                                multilabel=multilabel, 
                                                                lr=lr, num_epoch=num_epoch)
else:
        if model_type == "DeltaGNN":
                (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
                max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
                train_memory, train_time_avg) = train.train(model, device, 
                                                                x_train=x_train, 
                                                                y_train=y_train, 
                                                                adj_train=adj_train,
                                                                adj_train_cond=adj_train_cond,
                                                                x_val=None, 
                                                                y_val=None, 
                                                                adj_val=None,
                                                                val_mask=None,
                                                                x_test=None, 
                                                                y_test=None, 
                                                                adj_test=None,
                                                                test_mask=None,
                                                                multilabel=multilabel, 
                                                                lr=lr, num_epoch=num_epoch,
                                                                flow_flag=flow_control, 
                                                                plot_distribution=plot_distribution,
                                                                flow_control=flow_degradation)
        else:
                (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc, 
                max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
                train_memory, train_time_avg) = train.train(model, device, 
                                                                x_train=x_train, 
                                                                y_train=y_train, 
                                                                adj_train=adj_train,
                                                                x_val=None, 
                                                                y_val=None, 
                                                                adj_val=None,
                                                                val_mask=None,
                                                                x_test=None, 
                                                                y_test=None, 
                                                                adj_test=None,
                                                                test_mask=None,
                                                                multilabel=multilabel, 
                                                                lr=lr, num_epoch=num_epoch)
    
#%% Printing Result
if print_result:
    writer.write_result(dataset_name, model_type, num_epoch, x_train.shape[0], 
                        max_val_acc, max_val_f1, max_val_sens, max_val_spec, 
                        max_val_test_acc, max_val_test_f1, max_val_test_sens, 
                        max_val_test_spec, session_memory, train_memory, 
                        train_time_avg, filename = "result_"+dataset_name + ".csv", hidden_dimension=hidden_channels, max_communities=max_communities, remove_edges=remove_edges, topological_measure = topological_measure, make_unbalanced = make_unbalanced, dense = dense, flow_control = flow_control, linear = linear)