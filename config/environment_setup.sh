#!/bin/bash

conda env create -f config/environment_slim.yml
conda activate benchmark_gnn_slim
pip3 install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 torchdata==0.7.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.5.1
pip install torch_scatter==2.1.2+pt22cu118 torch_sparse==0.6.18+pt22cu118 -f https://data.pyg.org/whl/torch-2.2.0+cu118.html
pip install  dgl==2.3.0+cu118 -f https://data.dgl.ai/wheels/torch-2.2/cu118/repo.html
