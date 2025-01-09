#!/bin/bash

# Function to run the commands n times
run_commands() {
  local n=${1:-1} # Default to 1 if no parameter is specified

  for ((i = 0; i < n; i++)); do
    echo "Running iteration $((i + 1)) of $n"

    # Running commands for dataset Cora with different models and parameters

    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name Cora --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 1024 --remove_edges 40 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure curvature
    python main_multilabel_classification.py --dataset_name Cora  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.97

    # Running commands for dataset CiteSeer with different models and parameters

    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 40 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 512
    python main_multilabel_classification.py --dataset_name CiteSeer --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 512 --remove_edges 40 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 2048 --topological_measure curvature
    python main_multilabel_classification.py --dataset_name CiteSeer  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --max_communities 20 --hidden_channels 2048 --flow_control --flow_density 0.97

    # Running commands for dataset PubMed with different models and parameters

    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 2048 --remove_edges 400 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 2048 --remove_edges 400 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256
    python main_multilabel_classification.py --dataset_name PubMed --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 400 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name PubMed  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 1024 --topological_measure curvature
    python main_multilabel_classification.py --dataset_name PubMed  --batch_norm --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --remove_edges 30 --max_communities 20 --hidden_channels 1024 --flow_control --flow_density 0.97

    # Running commands for dataset Organ-S with different models and parameters

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 5000 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 5000 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 5000 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.97
    python main_multilabel_classification.py --dataset_name Organ-S --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure curvature

    # Running commands for dataset Organ-C with different models and parameters

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GAT --num_epoch 300 --hidden_channels 256 --remove_edges 5000 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 1024 --remove_edges 5000 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result --model_type GIN --num_epoch 300 --hidden_channels 1024 --remove_edges 5000 --topological_measure curvature

    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --max_communities 500 --hidden_channels 1024 --flow_control --flow_density 0.97
    python main_multilabel_classification.py --dataset_name Organ-C --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 300 --remove_edges 5000 --max_communities 500 --hidden_channels 1024 --topological_measure curvature

  done
}

# Call the function with the argument passed to the script
run_commands "$1"

python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 5000 --remove_edges 50 --max_communities 20 --hidden_channels 1024 --topological_measure curvature
python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 5000 --remove_edges 50 --max_communities 20 --hidden_channels 1024 --topological_measure curvature
python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 5000 --remove_edges 50 --max_communities 20 --hidden_channels 1024 --topological_measure curvature
python main_multilabel_classification.py --dataset_name Cornell --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 5000 --max_communities 20 --hidden_channels 1024 --flow_control --flow_density 0.97
python main_multilabel_classification.py --dataset_name Texas --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 5000 --max_communities 20 --hidden_channels 1024 --flow_control --flow_density 0.97
python main_multilabel_classification.py --dataset_name Wisconsin --batch_norm --do_eval --print_result  --model_type DeltaGNN --num_epoch 5000 --max_communities 20 --hidden_channels 1024 --flow_control --flow_density 0.97


python main_multilabel_classification_multigraph.py --dataset_name Peptides-func --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 235 --num_batch 500 --remove_edges 50 --topological_measure eigenvector_centrality
python main_multilabel_classification_multigraph.py --dataset_name Peptides-func --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 235 --num_batch 500
python main_multilabel_classification_multigraph.py --dataset_name Peptides-struct --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 235 --num_batch 500
python main_multilabel_classification_multigraph.py --dataset_name Peptides-struct --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --hidden_channels 235 --num_batch 200 --layers 4 --dropout 0.1 --batch_norm --lr 0.001 --head_depth 3 --remove_edges 50 --topological_measure random
python main_multilabel_classification_multigraph.py --dataset_name Peptides-func --do_eval --print_result --model_type DeltaGNN --num_epoch 300 --hidden_channels 235 --num_batch 200 --layers 4 --dropout 0.1 --batch_norm --lr 0.001 --head_depth 3  --flow_control --flow_density 0.97


python main_multilabel_classification_multigraph.py --dataset_name PROTEINS --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 100 --layers 5 --lr 0.001 --num_batch 2000 --num_epoch 1000
python main_multilabel_classification_multigraph.py --dataset_name ENZYMES --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 100 --layers 5 --lr 0.001 --num_batch 2000 --num_epoch 1000
python main_multilabel_classification_multigraph.py --dataset_name MUTAG --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 100 --layers 5 --lr 0.001 --num_batch 2000 --num_epoch 1000

python main_multilabel_classification_multigraph.py --dataset_name REDDIT-BINARY --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 100 --layers 5 --lr 0.001 --num_batch 500 --num_epoch 1000
python main_multilabel_classification_multigraph.py --dataset_name IMDB-BINARY --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 100 --layers 5 --lr 0.001 --num_batch 2000 --num_epoch 1000
python main_multilabel_classification_multigraph.py --dataset_name COLLAB --batch_norm --do_eval --print_result --model_type GCN --num_epoch 300 --hidden_channels 100 --layers 5 --lr 0.001 --num_batch 500 --num_epoch 1000
