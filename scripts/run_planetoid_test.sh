#!/bin/bash

# Function to run the commands n times
run_commands() {
  local n=${1:-1} # Default to 1 if no parameter is specified

  for ((i = 0; i < n; i++)); do
    echo "Running iteration $((i + 1)) of $n"

    # Running commands for dataset Cora with different models and parameters

    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure random
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure random
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure random
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure flow --aggregations_flow 3

    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure random
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure flow --aggregations_flow 3
    
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --max_communities 40 --hidden_channels 2048 --flow_control --flow_density 0.97 --linear
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --max_communities 40 --hidden_channels 2048 --flow_control --flow_density 0.97

    # Running commands for dataset CiteSeer with different models and parameters

    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure random
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure random
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure random
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure ricci_curvature
    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure random
    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure betweenness_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure flow --aggregations_flow 4

    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.99 --linear
    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.99

    # Running commands for dataset PubMed with different models and parameters

    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GCN --num_epoch 150 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GCN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GCN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure random
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GCN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GCN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GCN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure flow --aggregations_flow 5

    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GIN --num_epoch 150 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GIN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GIN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure random
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GIN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GIN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GIN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure closeness_centrality
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GIN --num_epoch 150 --hidden_channels 1024 --remove_edges 400 --topological_measure flow --aggregations_flow 5

    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GAT --num_epoch 150 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GAT --num_epoch 150 --hidden_channels 256 --remove_edges 400 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GAT --num_epoch 150 --hidden_channels 256 --remove_edges 400 --topological_measure random
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GAT --num_epoch 150 --hidden_channels 256 --remove_edges 400 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GAT --num_epoch 150 --hidden_channels 256 --remove_edges 400 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GAT --num_epoch 150 --hidden_channels 256 --remove_edges 400 --topological_measure flow --aggregations_flow 5

    python main_multilabel_classification.py --dataset_name PubMed  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 150 --remove_edges 200 --max_communities 10 --hidden_channels 1024 --topological_measure forman_curvature
    python main_multilabel_classification.py --dataset_name PubMed  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 150 --remove_edges 200 --max_communities 10 --hidden_channels 1024 --topological_measure random
    python main_multilabel_classification.py --dataset_name PubMed  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 150 --remove_edges 200 --max_communities 10 --hidden_channels 1024 --topological_measure degree_centrality
    python main_multilabel_classification.py --dataset_name PubMed  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 150 --remove_edges 200 --max_communities 10 --hidden_channels 1024 --topological_measure eigenvector_centrality
    python main_multilabel_classification.py --dataset_name PubMed  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 150 --remove_edges 200 --max_communities 10 --hidden_channels 1024 --topological_measure flow --aggregations_flow 5

    python main_multilabel_classification.py --dataset_name PubMed  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 150 --max_communities 10 --hidden_channels 1024 --flow_control --flow_density 0.99 --linear
    python main_multilabel_classification.py --dataset_name PubMed  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 150 --max_communities 10 --hidden_channels 1024 --flow_control --flow_density 0.99

  done
}

# Call the function with the argument passed to the script
run_commands "$1"