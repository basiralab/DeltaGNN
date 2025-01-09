#!/bin/bash

# Function to run the commands n times
run_commands() {
  local n=${1:-1} # Default to 1 if no parameter is specified

  for ((i = 0; i < n; i++)); do
    echo "Running iteration $((i + 1)) of $n"

    # Running commands for dataset Organ-S with different models and parameters

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure random
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure random
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure random
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure random
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.90 --linear
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.97

    # Running commands for dataset Organ-S (dense) with different models and parameters

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure random --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure degree_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure eigenvector_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure flow --aggregations_flow 4 --dense

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure random --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure degree_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure eigenvector_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure flow --aggregations_flow 4 --dense

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 15000 --max_communities 500 --hidden_channels 1024 --topological_measure random --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 15000 --max_communities 500 --hidden_channels 1024 --topological_measure degree_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 15000 --max_communities 500 --hidden_channels 1024 --topological_measure eigenvector_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 15000 --max_communities 500 --hidden_channels 1024 --topological_measure flow --aggregations_flow 5 --dense

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.95 --linear --dense
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.95 --dense

    # Running commands for dataset Organ-C with different models and parameters

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure random
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 1000 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure random
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure random
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 1000 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure random
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure flow --aggregations_flow 5

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 100000 --max_communities 500 --hidden_channels 1024 --topological_measure flow --aggregations_flow 5 --plot_distribution

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.997 --linear
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.97

    # Running commands for dataset Organ-C (dense) with different models and parameters

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure random --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure degree_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure eigenvector_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure flow --aggregations_flow 4 --dense

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure random --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure degree_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure eigenvector_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 3000 --topological_measure flow --aggregations_flow 4 --dense

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 15000 --max_communities 500 --hidden_channels 1024 --topological_measure random --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 15000 --max_communities 500 --hidden_channels 1024 --topological_measure degree_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 15000 --max_communities 500 --hidden_channels 1024 --topological_measure eigenvector_centrality --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 15000 --max_communities 500 --hidden_channels 1024 --topological_measure flow --aggregations_flow 4 --dense

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.95 --linear --dense
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.95 --dense

  done
}

# Call the function with the argument passed to the script
run_commands "$1"