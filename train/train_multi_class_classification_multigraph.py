import torch
import time
import numpy as np
from .metrics import metrics

def full_batch_step(model, optimizer, criterion, device, train_loader, logging=False, flow_flag=False, duo_architecture = False, train_cond_loader = None):
    model.train()
    total_loss = 0
    adjs = []
    scores = []

    if train_cond_loader is None:
        for batch in train_loader:
            x_train, y_train, batch, edge_index = batch.x.to(device), batch.y.to(device), batch.batch.to(device),  batch.edge_index.to(device)
            optimizer.zero_grad()
            if flow_flag:
                out, batch_scores, batch_adjs = model(x_train, edge_index, batch = batch, flow_flag = flow_flag)
            else:
                out = model(x_train, edge_index,batch = batch)
            
            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
    else:
        for batch,batch_cond in zip(train_loader,train_cond_loader):
            x_train, y_train, batch, edge_index = batch.x.to(device), batch.y.to(device), batch.batch.to(device),  batch.edge_index.to(device)
            edge_index_cond = batch_cond.edge_index.to(device)
            optimizer.zero_grad()
                
            out, batch_scores, batch_adjs = model(x_train, edge_index, edge_index_b = edge_index_cond, batch=batch, flow_flag=flow_flag)
            adjs.extend(batch_adjs)
            scores.extend(batch_scores)

            loss = criterion(out, y_train)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

    if logging:
        acc, micro_f1, sens, spec = evaluate(model, train_loader, None, device, flow_flag=flow_flag)
        print(f"Train accuracy: {acc}, Train micro_f1: {micro_f1}, Train Sens: {sens}, Train Spec: {spec}")

    return total_loss / len(train_loader), adjs, scores
    
def evaluate(model, loader, device, flow_flag=False, duo_architecture = False, cond_loader = None):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        if cond_loader is None:
            for batch in loader:
                x, y, batch, edge_index = batch.x.to(device), batch.y.to(device), batch.batch.to(device),  batch.edge_index.to(device)
                if flow_flag:
                    out = model(x, edge_index, batch=batch, flow_flag=flow_flag)[0].squeeze()
                else:
                    out = model(x, edge_index, batch=batch)
                all_preds.append(out.cpu())
                all_labels.append(y.cpu())
        else:
            for batch,batch_cond in zip(loader,cond_loader):
                x, y, batch, edge_index = batch.x.to(device), batch.y.to(device), batch.batch.to(device),  batch.edge_index.to(device)
                edge_index_cond = batch_cond.edge_index.to(device)
                out = model(x, edge_index, edge_index_b = edge_index_cond, batch=batch, flow_flag=flow_flag)[0].squeeze()
                all_preds.append(out.cpu())
                all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc, micro_f1, sens, spec, ap = metrics(all_preds, all_labels)
    return acc, micro_f1, sens, spec, ap

def train(model, device, train_loader, val_loader=None, test_loader=None, train_cond_loader = None, val_cond_loader=None, test_cond_loader=None, multilabel = True, 
          lr = 0.0005, num_epoch = 100, flow_flag = False, flow_control = False, duo_architecture = False, use_accuracy = False):

    # passing model and training data to GPU
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    max_val_ap = 0
    max_val_acc = 0
    max_val_sens = 0
    max_val_spec = 0
    max_val_f1 = 0
    max_val_test_ap = 0
    max_val_test_acc = 0
    max_val_test_sens = 0
    max_val_test_spec = 0
    max_val_test_f1 = 0
    
    time_arr = np.zeros((num_epoch,))

    for epoch in range(num_epoch):
            
        # single mini batch step
        t = time.time()

        loss, adjs, scores = full_batch_step(model, optimizer, criterion, device, train_loader, logging=False, flow_flag=flow_flag, duo_architecture = duo_architecture, train_cond_loader=train_cond_loader)

        time_per_epoch = time.time() - t
        time_arr[epoch] = time_per_epoch

        if epoch == 0:
            # passing validation and test data to GPU (we do it after first forward pass to get)
            # accurate pure training GPU memory usage
            train_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
        
        if epoch % 50 == 0:
            if flow_control:
                model.update_density(max(model.get_density()-0.005,0.85))
                print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, training time: {time_per_epoch:.5f}, Density: {model.get_density()}')
            else:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, training time: {time_per_epoch:.5f}')
            print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
        
        # evaluation
        if val_loader:
            acc, micro_f1, sens, spec, ap = evaluate(model, val_loader, device, flow_flag=flow_flag, duo_architecture=duo_architecture, cond_loader = val_cond_loader)

            if epoch % 100 == 0:
                if not use_accuracy:
                    print(f"Val Average Precision: {ap}, Val accuracy: {acc}, Val micro_f1: {micro_f1}, Val Sens: {sens}, Val Spec: {spec}")
                else:
                    print(f"Val accuracy: {acc}, Val micro_f1: {micro_f1}, Val Sens: {sens}, Val Spec: {spec}, Val Average Precision: {ap}")
                
            if (ap > max_val_ap and not use_accuracy) or (acc > max_val_acc and use_accuracy):
                max_val_ap = ap
                max_val_acc = acc
                max_val_f1 = micro_f1
                max_val_sens = sens
                max_val_spec = spec
                
                if test_loader:
                    
                    acc, micro_f1, sens, spec, ap = evaluate(model, test_loader, device, flow_flag=flow_flag, duo_architecture=duo_architecture, cond_loader = test_cond_loader)
                    
                    max_val_test_ap = ap
                    max_val_test_acc = acc
                    max_val_test_f1 = micro_f1
                    max_val_test_sens = sens
                    max_val_test_spec = spec
                    
                    if not use_accuracy:
                        print("===========================================Best Model Update:=======================================")
                        print(f"Val Average Precision: {max_val_ap}, Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}")
                        print(f"Test Average Precision: {max_val_test_ap}, Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}")
                        print("====================================================================================================")
                    else:
                        print("===========================================Best Model Update:=======================================")
                        print(f"Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}, Val Average Precision: {max_val_ap}")
                        print(f"Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}, Test Average Precision: {max_val_test_ap}")
                        print("====================================================================================================")
    print("Best Model:")
    if not use_accuracy:
        print(f"Val Average Precision: {max_val_ap}, Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}")
        print(f"Test Average Precision: {max_val_test_ap}, Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}")
    else:
        print(f"Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}, Val Average Precision: {max_val_ap}")
        print(f"Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}, Test Average Precision: {max_val_test_ap}")
    print(f"Average time per epoch: {time_arr[10:].mean()}") # don't include the first few epoch (slower due to Torch initialization)
    print(f"Training GPU Memory Usage: {train_memory} MB")
    print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
    
    # cleaning memory and stats
    session_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
    train_time_avg = time_arr[10:].mean()
    model = model.to('cpu')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    return (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc,
            max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
            train_memory, train_time_avg)
