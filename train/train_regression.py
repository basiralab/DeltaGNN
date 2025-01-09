import torch
import time
import numpy as np

from torch_geometric.typing import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate(model, device, val_data):

    with torch.no_grad():
        model.eval()

        predictions = []
        truths = []

        for (x, y, adj,adj_curv) in val_data: 

            x = x.to(device)            
            y = y.to(device)
            adj = adj.to(device)
            if adj_curv is not None and adj_curv.dim() > 1:
                adj_curv = adj_curv.to(device)

            if adj_curv is not None and adj_curv.dim() > 1:
                out = model(x, adj, adj_curv)
            else:
                out = model(x, adj)
            predictions.append(out)
            truths.append(y)

        mse, mae, r2 = metrics(predictions,truths)
    
    return mse, mae, r2


def train(model, device, train_data, val_data, test_data, 
          lr = 0.0005, num_epoch = 100):
    
    # passing model and training data to GPU
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    
    min_val_mse = float('inf')
    min_val_mae = float('inf')
    min_val_r2 = float('inf')
    min_val_test_mse = float('inf')
    min_val_test_mae = float('inf')
    min_val_test_r2 = float('inf')

    time_arr = np.zeros((num_epoch,))
    for epoch in range(num_epoch):

        t = time.time()

        model.train()

        total_loss = 0

        for x, y, adj, adj_curv in train_data:
            optimizer.zero_grad()

            x = x.to(device)
            adj = adj.to(device)
            if adj_curv is not None and adj_curv.dim() > 1:
                adj_curv = adj_curv.to(device)
            y = y.to(device)

            if adj_curv is not None and adj_curv.dim() > 1:
                out = model(x, adj, adj_curv)
            else:
                out = model(x, adj)

            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss

        loss = total_loss / len(train_data)

        time_per_epoch = time.time() - t
        time_arr[epoch] = time_per_epoch
        
        if epoch == 0:
            train_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
        
        print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, training time: {time_per_epoch:.5f}')
        print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
        
        # evaluation
        if val_data != None:
            mse, mae, r2 = evaluate(model, device, val_data)
            
            if epoch % 100 == 0:
                print(f"Val MSE: {mse}, Val MAE: {mae}, Val R2: {r2}")
            
            if mse < min_val_mse:
                min_val_mse = mse
                min_val_mae = mae
                min_val_r2 = r2
                
                if (test_data != None):
                    mse, mae, r2 = evaluate(model, device, test_data)
                    min_val_test_mse = mse
                    min_val_test_mae = mae
                    min_val_test_r2 = r2
                    
                    print("===========================================Best Model Update:=======================================")
                    print(f"Val MSE: {min_val_mse}, Val MAE: {min_val_mae}, Val R2: {min_val_r2}")
                    print(f"Test MSE: {min_val_test_mse}, Test MAE: {min_val_test_mae}, Test R2: {min_val_test_r2}")
                    print("====================================================================================================")

    print("Best Model:")
    print(f"Val MSE: {min_val_mse}, Val MAE: {min_val_mae}, Val R2: {min_val_r2}")
    print(f"Test MSE: {min_val_test_mse}, Test MAE: {min_val_test_mae}, Test R2: {min_val_test_r2}")
    print(f"Average time per epoch: {time_arr[10:].mean()}") # don't include the first few epoch (slower due to Torch initialization)
    print(f"Training GPU Memory Usage: {train_memory} MB")
    print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
    
    # cleaning memory and stats
    session_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
    train_time_avg = time_arr[10:].mean()
    del val_data
    del train_data
    del test_data

    model = model.to('cpu')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    return (min_val_mse, min_val_mae, min_val_r2, min_val_test_mse,
            min_val_test_mae, min_val_test_r2, session_memory, 
            train_memory, train_time_avg)

def logit_to_label(out):
    return out.argmax(dim=1)

def metrics(logits, y):
    """
    Calculate regression metrics.
    
    Args:
    - logits (list of tensors): Predicted values, each tensor of size [1].
    - y (list of tensors): True values, each tensor of size [1].
    
    Returns:
    - metrics_dict (dict): A dictionary with keys 'mse', 'mae', and 'r2' containing the respective metric values.
    """
    # Convert list of tensors to single tensors
    logits_concat = torch.cat(logits).cpu().numpy()
    y_concat = torch.cat(y).cpu().numpy()
    
    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(y_concat, logits_concat)
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_concat, logits_concat)
    
    # Calculate R-squared (R²)
    r2 = r2_score(y_concat, logits_concat)
    
    # Compile metrics into a dictionary
    
    return mse, mae, r2

def construct_normalized_adj(edge_index, num_nodes):
    
    edge_index = torch.tensor(edge_index)
    edge_index = torch.transpose(edge_index,0,1)
    edge_index_flip = torch.flip(edge_index,[0]) # re-adds flipped edges that were removed by networkx
    edge_index = torch.cat((edge_index, edge_index_flip), 1)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1], sparse_sizes=(num_nodes,num_nodes))
    adj = adj.set_diag() # adding self loops
    adj = gcn_norm(adj, add_self_loops=False) # normalization

    return adj
