#!/bin/bash

# Function to run the commands n times
run_commands() {
  local n=${1:-1} # Default to 1 if no parameter is specified

  for ((i = 0; i < n; i++)); do
    echo "Running iteration $((i + 1)) of $n"

    # Running commands for dataset Cornell with different models and parameters

    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure random
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure random
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure random
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 5 --max_communities 20 --hidden_channels 2048 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 5 --max_communities 20 --hidden_channels 2048 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 5 --max_communities 20 --hidden_channels 2048 --topological_measure random
    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 5 --max_communities 20 --hidden_channels 2048 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 5 --max_communities 20 --hidden_channels 2048 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 5 --max_communities 20 --hidden_channels 2048 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 5 --max_communities 20 --hidden_channels 2048 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 5 --max_communities 20 --hidden_channels 2048 --topological_measure flow --aggregations_flow 3
    
    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.988 --linear
    python main_multilabel_classification.py --dataset_name Cornell  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.988

    # Running commands for dataset Texas with different models and parameters

    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure random
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure random
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure random
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure random
    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure flow --aggregations_flow 3
    
    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.98 --linear
    python main_multilabel_classification.py --dataset_name Texas  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.98

    # Running commands for dataset Wisconsin with different models and parameters

    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure random
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GCN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure random
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GIN --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure random
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result --model_type GAT --num_epoch 1500 --hidden_channels 2048 --remove_edges 10 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure random
    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --remove_edges 10 --max_communities 20 --hidden_channels 2048 --topological_measure flow --aggregations_flow 3
    
    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.97 --linear
    python main_multilabel_classification.py --dataset_name Wisconsin  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 1500 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.97

  done
}

# Call the function with the argument passed to the script
run_commands "$1"