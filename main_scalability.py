import time
import torch
from torch_geometric.data import Data
import data.databases as datasets
import models.gcn_model_batching as gcn_model
import models.delta_gnn_model_batching as dual_gcn_model
import models.gin_model_batching as gin_model
import models.gat_model_batching as gat_model
import train.utils as train_utils
import argparse
import numpy as np
import gc
from GraphRicciCurvature.FormanRicci import FormanRicci
from GraphRicciCurvature.OllivierRicci import OllivierRicci
import networkx as nx
import random
from contextlib import redirect_stdout
import os
import io
import multiprocessing
import time

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', type=str, default='GCN')
parser.add_argument('--max_hop', type=int, default=1)
parser.add_argument('--layers', type=int, default=3)
parser.add_argument('--hidden_channels', type=int, default=200)
parser.add_argument('--dropout', type=float, default=0)
parser.add_argument('--batch_norm', action='store_true')
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--num_batch', type=int, default=1)
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--multilabel', action='store_true')
parser.add_argument('--do_eval', action='store_true')
parser.add_argument('--residual', action='store_true')
parser.add_argument('--print_result', action='store_true')
parser.add_argument('--dataset_name', type=str, default='Organ-C')
parser.add_argument('--aggregations_flow', type=int, default=3)
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

features = 1
classes = 1
# Setting the time limit to 10 minutes (600 seconds)
time_limit_seconds = 10 * 60

def suppress_print(func, *args, **kwargs):
    f = io.StringIO()
    with redirect_stdout(f):
        func(*args, **kwargs)

def with_time_limit(time_limit, func, *args, **kwargs):
    def wrapper(queue, *args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        queue.put((result, elapsed_time))

    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=wrapper, args=(queue, *args), kwargs=kwargs)
    process.start()
    process.join(time_limit)

    if process.is_alive():
        process.terminate()
        process.join()
        return "OOT"

    if not queue.empty():
        result, elapsed_time = queue.get()
        return elapsed_time
    else:
        return "OOT"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_model(model, graph_sizes, edge_densities, num_folds=10, device='cpu', extra = False, flow = False):
    results = []

    for size in graph_sizes:
        for density in edge_densities:
            print(f"\nBenchmarking with size={size}, density={density}")
            # Initialize lists to collect results across folds
            fold_times = []
            fold_memory_usages = []
            fold_params = count_parameters(model)

            for _ in range(num_folds):
                # Generate graph
                x, edge_index, n_nodes, n_edges = datasets.get_er_graph(n_nodes=size, p_edge=density, num_features=features)
                data = Data(x=x, edge_index=edge_index)
                data = data.to(device)

                torch.cuda.empty_cache()
                gc.collect()

                # Measure model run time and memory usage
                def run_model(model):
                    model.train()
                    with torch.no_grad():
                        if extra:
                            return model(data.x, data.edge_index, device, edge_index_b = data.edge_index)
                        if flow:
                            return model(data.x, data.edge_index, device, flow_flag = True)
                        else:
                            return model(data.x, data.edge_index)
                
                # Measure memory usage
                mem_usage = torch.cuda.max_memory_allocated(device)*2**(-20)

                # Measure run time
                start_time = time.time()
                with torch.no_grad():
                    run_model(model)
                run_time = time.time() - start_time
                
                fold_times.append(run_time)
                fold_memory_usages.append(mem_usage)

                # Free up memory
                del data
                del edge_index
                del x
                torch.cuda.empty_cache()
                gc.collect()

            # Calculate average and standard deviation
            avg_run_time = np.mean(fold_times)
            std_run_time = np.std(fold_times)
            avg_memory_usage = np.mean(fold_memory_usages)
            std_memory_usage = np.std(fold_memory_usages)

            results.append({
                'size': size,
                'density': density,
                'num_edges': n_edges,
                'avg_run_time': avg_run_time,
                'std_run_time': std_run_time,
                'avg_memory_usage': avg_memory_usage,
                'std_memory_usage': std_memory_usage,
                'num_parameters': fold_params,
                'cumulative_size': size*num_folds,
                'cumulative_num_edges': n_edges*num_folds,
                'cumulative_run_time': avg_run_time*num_folds,
            })

def benchmark_measures(graph_sizes, edge_densities, num_folds=10):
    results = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    torch.cuda.empty_cache()
    gc.collect()

    for measure in ['flow']:
    #for measure in ['eigenvector_centrality','flow','degree_centrality','ricci_curvature','forman_curvature','betweenness_centrality','closeness_centrality']:
        print(f"Benchmarking: {measure}")
        for size in graph_sizes:
            for density in edge_densities:
                print(f"\nBenchmarking with size={size}, density={density}")
                # Initialize lists to collect results across folds
                fold_times = []
                fold_memory_usages = []

                for _ in range(num_folds):
                    # Generate graph
                    x, edge_index, n_nodes, n_edges = datasets.get_er_graph(n_nodes=size, p_edge=density, num_features=features)
                    x.to(device)
                    edge_index.to(device)
                    G = train_utils.construct_graph(x, None, edge_index)

                    def run_measure(measure):
                        nodes_topo_results = []
                        edges_topo_results = []
                        if measure == "ricci_curvature":
                                curvature = OllivierRicci(G)
                                suppress_print(curvature.compute_ricci_curvature)
                        if measure == "forman_curvature":
                                curvature = FormanRicci(G)
                                suppress_print(curvature.compute_ricci_curvature)
                        if measure == "degree_centrality":
                                nodes_topo_results = nx.degree_centrality(G)
                                edges_topo_results = train_utils.edge_degree_centrality(G)
                        if measure == "betweenness_centrality":
                                nodes_topo_results = nx.betweenness_centrality(G)
                                edges_topo_results = nx.edge_betweenness_centrality(G)
                        if measure == "eigenvector_centrality":
                                nodes_topo_results = nx.eigenvector_centrality(G, tol=1e-03)
                                edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G)
                        if measure == "closeness_centrality":
                                nodes_topo_results = nx.closeness_centrality(G)
                                edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G)
                        if measure == "random":
                                for node in G.nodes():
                                        G.nodes[node]['topo'] = random.random()
                                for u,v in G.edges():
                                        G[u][v]['topo'] = random.random()
                        if measure == "flow":
                                nodes_topo_results = train_utils.node_flow_metric(G, aggregations_flow, aggregate=False)
                                edges_topo_results = train_utils.edge_metric_compute(nodes_topo_results, G)
                        if measure != 'curvature' and measure != 'random':
                            del nodes_topo_results
                            del edges_topo_results


                    start_time = time.time()
                    run_measure(measure)
                    end_time = time.time()
                    elapsed_time = end_time - start_time

                    #elapsed_time = with_time_limit(time_limit_seconds, run_measure, measure)

                    # Measure memory usage
                    mem_usage = torch.cuda.max_memory_allocated(device)*2**(-20)
                    
                    fold_times.append(elapsed_time)
                    fold_memory_usages.append(mem_usage)

                    # Free up memory
                    del G
                    del edge_index
                    del x
                    torch.cuda.empty_cache()
                    gc.collect()

                    if elapsed_time == 'OOT':
                         print("Breaking due to OOT")
                         break

                # Calculate average and standard deviation
                avg_run_time = 0
                std_run_time = 0
                valid_fold_times = [time for time in fold_times if time != 'OOT']
                if valid_fold_times:
                    avg_run_time = np.mean(valid_fold_times)
                    std_run_time = np.std(valid_fold_times)
                else:
                    avg_run_time = 'OOT'
                    std_run_time = 'OOT'
                avg_memory_usage = np.mean(fold_memory_usages)
                std_memory_usage = np.std(fold_memory_usages)
                print("Done: "+ str(avg_run_time))

                results.append({
                    'measure': measure,
                    'size': size,
                    'density': density,
                    'num_edges': n_edges,
                    'avg_run_time': avg_run_time,
                    'std_run_time': std_run_time,
                    'avg_memory_usage': avg_memory_usage,
                    'std_memory_usage': std_memory_usage,
                    'cumulative_size': size*num_folds,
                    'cumulative_num_edges': n_edges*num_folds,
                    'cumulative_run_time': avg_run_time*num_folds if avg_run_time != 'OOT' else 'OOT',
                })

    return results

def print_results_measures(results):
    os.makedirs('results/stats', exist_ok=True)
    file_path = 'results/stats/scalability_measures.txt'
    with open(file_path, 'w') as f:
        f.write(f"{'Measure':<20} {'Size':<10} {'Density':<10} {'Num Edges':<10} {'Avg Run Time':<15} {'Std Run Time':<15} {'Avg Memory Usage':<20} {'Std Memory Usage':<20} {'Cumulative Size':<20} {'Cumulative Num Edges':<20} {'Cumulative Run Time':<20}\n")
        f.write("="*150 + "\n")
        for result in results:
            avg_run_time = result['avg_run_time']
            std_run_time = result['std_run_time']
            cumulative_run_time = result['cumulative_run_time']
            
            # Handle 'OOT' cases
            avg_run_time_str = f"{avg_run_time:<15.5f}" if avg_run_time != 'OOT' else 'OOT'
            std_run_time_str = f"{std_run_time:<15.5f}" if std_run_time != 'OOT' else 'OOT'
            cumulative_run_time = f"{cumulative_run_time:<15.5f}" if cumulative_run_time != 'OOT' else 'OOT'
            
            f.write(f"{result['measure']:<20} {result['size']:<10} {result['density']:<10} {result['num_edges']:<10} {avg_run_time_str:<15} {std_run_time_str:<15} {result['avg_memory_usage']:<20.5f} {result['std_memory_usage']:<20.5f} {result['cumulative_size']:<20} {result['cumulative_num_edges']:<20} {cumulative_run_time:<15}\n")

def print_results_models(results, model_name):
    os.makedirs('results/stats', exist_ok=True)
    file_path = f'results/stats/scalability_model_{model_name}.txt'
    headers = [
        'Size', 'Density', 'Num Edges', 'Avg Run Time (s)', 'Std Run Time (s)', 
        'Avg Memory Usage (MB)', 'Std Memory Usage (MB)', 'Num Parameters',
        'Cumulative Size', 'Cumulative Num Edges', 'Cumulative Run Time (s)'
    ]
    with open(file_path, 'w') as f:
        f.write(f"\nResults for model: {model_name}\n")
        f.write("="*150 + "\n")
        f.write(f"{headers[0]:<10} {headers[1]:<10} {headers[2]:<12} {headers[3]:<20} {headers[4]:<20} {headers[5]:<25} {headers[6]:<25} {headers[7]:<15} {headers[8]:<15} {headers[9]:<18} {headers[10]:<25}\n")
        f.write("="*150 + "\n")
        for r in results:
            f.write(f"{r['size']:<10} {r['density']:<10} {r['num_edges']:<12} {r['avg_run_time']:<20.4f} {r['std_run_time']:<20.4f} {r['avg_memory_usage']:<25.2f} {r['std_memory_usage']:<25.2f} {r['num_parameters']:<15} {r['cumulative_size']:<15} {r['cumulative_num_edges']:<18} {r['cumulative_run_time']:<25.4f}\n")

# Parameters
graph_sizes = [100, 1000, 5000]
edge_densities = [0.01, 0.03, 0.06]
num_folds = 20


results = benchmark_measures(graph_sizes, edge_densities, num_folds)
print_results_measures(results)
if False:
    model = gcn_model.GCN(hidden_channels=hidden_channels, num_layers= layers, 
                            in_channels= features, out_channels=classes, 
                            batch_norm=batch_norm, dropout=dropout, residual=residual)
    model = model.to(device)
    print(model)

    print(f"Benchmarking model: {model.__class__.__name__}")
    results = benchmark_model(model, graph_sizes, edge_densities, num_folds, device)
    print_results_models(results, gcn_model.GCN.__name__)
    # Free up memory
    del model
    torch.cuda.empty_cache()
    gc.collect()


    model = gin_model.GIN(hidden_channels=hidden_channels, num_layers= layers, 
                            in_channels= features, out_channels=classes, 
                            batch_norm=batch_norm, dropout=dropout, residual=residual)
    model = model.to(device)
    print(model)

    print(f"Benchmarking model: {model.__class__.__name__}")
    results = benchmark_model(model, graph_sizes, edge_densities, num_folds, device)
    print_results_models(results, gin_model.GIN.__name__)
    # Free up memory
    del model
    torch.cuda.empty_cache()
    gc.collect()


    model = gat_model.GAT(hidden_channels=hidden_channels, num_layers= layers, 
                            in_channels= features, out_channels=classes, 
                            batch_norm=batch_norm, dropout=dropout, residual=residual)
    model = model.to(device)
    print(model)

    print(f"Benchmarking model: {model.__class__.__name__}")
    results = benchmark_model(model, graph_sizes, edge_densities, num_folds, device)
    print_results_models(results, gat_model.GAT.__name__)
    # Free up memory
    del model
    torch.cuda.empty_cache()
    gc.collect()


    model= dual_gcn_model.DeltaGNN(hidden_channels=hidden_channels, num_layers= layers, 
                            in_channels= features, out_channels=classes, 
                            batch_norm=batch_norm, dropout=dropout, residual=residual, 
                            density= flow_density, max_communities=max_communities)
    model = model.to(device)
    print(model)

    print(f"Benchmarking model: {model.__class__.__name__}")
    results = benchmark_model(model, graph_sizes, edge_densities, num_folds, device, extra= True)
    print_results_models(results, dual_gcn_model.DeltaGNN.__name__)
    # Free up memory
    del model
    torch.cuda.empty_cache()
    gc.collect()


    model= dual_gcn_model.DeltaGNN(hidden_channels=hidden_channels, num_layers= layers, 
                            in_channels= features, out_channels=classes, 
                            batch_norm=batch_norm, dropout=dropout, residual=residual, 
                            density= flow_density, max_communities=max_communities)
    model = model.to(device)
    print(model)

    print(f"Benchmarking model: {model.__class__.__name__} + Flow")
    results = benchmark_model(model, graph_sizes, edge_densities, num_folds, device, flow = True)
    print_results_models(results, dual_gcn_model.DeltaGNN.__name__)
    # Free up memory
    del model
    torch.cuda.empty_cache()
    gc.collect()