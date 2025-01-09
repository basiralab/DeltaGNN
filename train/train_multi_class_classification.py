import torch
import time
import numpy as np
from .utils import plot_homophily_distributions
from .metrics import metrics
import matplotlib.pyplot as plt


def plot_validation_metrics(validation_acc, validation_spec, num_epoch):
    """
    Plot validation accuracy and specificity over epochs and save the plot as an image file.

    Args:
        validation_acc (list): List of validation accuracies.
        validation_spec (list): List of validation specificities.
        num_epoch (int): Number of epochs.
    """

    validation_acc = [x * 100 for x in validation_acc]
    validation_spec = [x * 100 for x in validation_spec]

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the validation accuracy (red line)
    ax.plot(range(1, num_epoch + 1), validation_acc, color='red', linewidth=2)

    # Plot the validation specificity (blue line)
    ax.plot(range(1, num_epoch + 1), validation_spec, color='blue', linewidth=2)

    # Set axis labels
    ax.set_xlabel('Epochs', color='grey')
    ax.set_ylabel('Percentage (%)', color='grey')

    # Set grid
    ax.grid(True)

    # Set the x-ticks with the calculated spacing
    ax.set_xticks(np.linspace(0, num_epoch, 7))

    # Set the y-axis to show percentages
    ax.set_yticks(np.arange(0, 101, 10))
    ax.set_yticklabels([f'{int(tick)}%' for tick in ax.get_yticks()], color='grey')

    # Customize tick parameters
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')

    # Save the plot with a timestamp
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"results/images/validation_metrics_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')

    # Close the plot to free up memory
    plt.close(fig)


def full_batch_step(model, optimizer, criterion, device, x_train, y_train, 
                    adj_train, train_mask, lr, logging = False, adj_cond = None, flow_flag = False):
    """
    Perform a single optimization step on the model using a full batch of training data.

    Args:
        model (torch.nn.Module): The model to be trained.
        optimizer (torch.optim.Optimizer): The optimizer used for updating model parameters.
        criterion (callable): The loss function.
        device (torch.device): The device (CPU or GPU) to run the computations on.
        x_train (torch.Tensor): The input features for the training data.
        y_train (torch.Tensor): The target labels for the training data.
        adj_train (torch.Tensor): The adjacency matrix for the training data.
        train_mask (torch.Tensor or None): Mask indicating which nodes to consider for training.
        lr (float): The learning rate for adjusting model parameters.
        logging (bool, optional): Whether to log training metrics. Defaults to False.
        adj_cond (torch.Tensor or None, optional): The condenced adjacency matrix. Defaults to None.
        flow_flag (bool, optional): Whether to use flow-based adjustments. Defaults to False.

    Returns:
        loss (torch.Tensor): The computed loss for the current batch.
        density (float): The current density value used in the model.
        adjs (list): A list of adjusted adjacency matrices (if any).

    """
    model.train()
    optimizer.zero_grad()
    adjs = []
    scores = []

    if adj_cond or flow_flag:
        out, scores, adjs = model(x_train, adj_train, device, adj_cond, flow_flag)
    else:
        out = model(x_train, adj_train)
    if train_mask == None:
        loss = criterion(out, y_train)
    else:
        loss = criterion(out[train_mask], y_train[train_mask])
    loss.backward()
    optimizer.step()

    if logging:
        acc,micro_f1,sens,spec, ap = metrics(out,y_train)
        if flow_flag:
            print(f"Train accuracy: {acc}, Train average precision: {ap}, Train micro_f1: {micro_f1},Train Sens: {sens}, Train Spec: {spec}, Last Score: {scores[-1]}")
        else:
            print(f"Train accuracy: {acc}, Train average precision: {ap}, Train micro_f1: {micro_f1},Train Sens: {sens}, Train Spec: {spec}")

    return loss, adjs, scores
    

def evaluate(model, x, y, adj, mask, device, adj_cond = None, flow_flag = False):
    """
    Evaluate the model on the provided data.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        x (torch.Tensor): The input features for the evaluation data.
        y (torch.Tensor): The target labels for the evaluation data.
        adj (torch.Tensor): The adjacency matrix for the evaluation data.
        mask (torch.Tensor): Mask indicating which nodes to consider for evaluation.
        device (torch.device): The device (CPU or GPU) to run the computations on.
        adj_cond (torch.Tensor or None, optional): The condensed adjacency matrix. Defaults to None.
        flow_flag (bool, optional): Whether to use flow-based adjustments. Defaults to False.

    Returns:
        acc (float): The accuracy of the model on the evaluation data.
        micro_f1 (float): The micro-averaged F1 score on the evaluation data.
        sens (float): The sensitivity (recall) of the model on the evaluation data.
        spec (float): The specificity of the model on the evaluation data.
    """
    with torch.no_grad():
        model.eval()
        if adj_cond or flow_flag:
            out = model(x, adj, device, adj_cond,flow_flag)[0].squeeze()
        else:
            out = model(x, adj).squeeze()
        acc,micro_f1,sens,spec, ap = metrics(out[mask],y[mask])
    
    return acc, micro_f1, sens, spec, ap


def train(model, device, x_train, y_train, adj_train, adj_train_cond = None, train_mask = None, x_val = None, 
          y_val = None, adj_val = None, adj_val_cond = None, val_mask = None, x_test = None, 
          y_test = None, adj_test = None, adj_test_cond = None, test_mask = None, multilabel = True, 
          lr = 0.0005, num_epoch = 100, flow_flag = False, plot_distribution = False, flow_control = False):
    """
    Train a model using the provided training data and evaluate it on validation and test data.

    Args:
        model (torch.nn.Module): The model to be trained.
        device (torch.device): The device (CPU or GPU) to run the computations on.
        x_train (torch.Tensor): The input features for the training data.
        y_train (torch.Tensor): The target labels for the training data.
        adj_train (torch.Tensor): The adjacency matrix for the training data.
        adj_train_cond (torch.Tensor or None, optional): The condensed adjacency matrix for training. Defaults to None.
        train_mask (torch.Tensor or None, optional): Mask indicating which nodes to consider for training. Defaults to None.
        x_val (torch.Tensor or None, optional): The input features for the validation data. Defaults to None.
        y_val (torch.Tensor or None, optional): The target labels for the validation data. Defaults to None.
        adj_val (torch.Tensor or None, optional): The adjacency matrix for the validation data. Defaults to None.
        adj_val_cond (torch.Tensor or None, optional): The condensed adjacency matrix for validation. Defaults to None.
        val_mask (torch.Tensor or None, optional): Mask indicating which nodes to consider for validation. Defaults to None.
        x_test (torch.Tensor or None, optional): The input features for the test data. Defaults to None.
        y_test (torch.Tensor or None, optional): The target labels for the test data. Defaults to None.
        adj_test (torch.Tensor or None, optional): The adjacency matrix for the test data. Defaults to None.
        adj_test_cond (torch.Tensor or None, optional): The condensed adjacency matrix for testing. Defaults to None.
        test_mask (torch.Tensor or None, optional): Mask indicating which nodes to consider for testing. Defaults to None.
        multilabel (bool, optional): Whether the task is multilabel classification. Defaults to True.
        lr (float, optional): Learning rate for the optimizer. Defaults to 0.0005.
        num_epoch (int, optional): Number of training epochs. Defaults to 100.
        flow_flag (bool, optional): Whether to use flow-based adjustments. Defaults to False.
        plot_distribution (bool, optional): Whether to plot homophily distributions during training. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - max_val_acc (float): Best validation accuracy achieved.
            - max_val_f1 (float): Best validation F1 score achieved.
            - max_val_sens (float): Best validation sensitivity achieved.
            - max_val_spec (float): Best validation specificity achieved.
            - max_val_test_acc (float): Best test accuracy achieved.
            - max_val_test_f1 (float): Best test F1 score achieved.
            - max_val_test_sens (float): Best test sensitivity achieved.
            - max_val_test_spec (float): Best test specificity achieved.
            - session_memory (float): Peak GPU memory usage during the session (in MB).
            - train_memory (float): Peak GPU memory usage during training (in MB).
            - train_time_avg (float): Average time per epoch during training.
    """

    # passing model and training data to GPU
    model = model.to(device)
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    adj_train = adj_train.to(device)
    if adj_train_cond:
        adj_train_cond = adj_train_cond.to(device)
    if train_mask != None:
        train_mask = train_mask.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    if multilabel:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
    
    max_val_acc = 0
    max_val_sens = 0
    max_val_spec = 0
    max_val_f1 = 0
    max_val_test_acc = 0
    max_val_test_sens = 0
    max_val_test_spec = 0
    max_val_test_f1 = 0

    validation_acc = []
    validation_spec = []
    
    time_arr = np.zeros((num_epoch,))

    for epoch in range(num_epoch):
            
        # single mini batch step
        t = time.time()

        loss, adjs, scores = full_batch_step(model, optimizer, criterion, device, 
                                   x_train, y_train, adj_train, train_mask, lr, 
                                   logging=False, adj_cond=adj_train_cond, flow_flag = flow_flag)
        
        if plot_distribution and epoch == 0:
            plot_homophily_distributions([adj.cpu() for adj in adjs],x_train.cpu(),y_train.cpu(),["Original graph","Homophilic graph - filtered","Heterophilic graph - condensed"],epoch)

        time_per_epoch = time.time() - t
        time_arr[epoch] = time_per_epoch
        
        if epoch == 0:
            train_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
            
            # passing validation and test data to GPU (we do it after first forward pass to get)
            # accurate pure training GPU memory usage
            if x_val != None and y_val != None and adj_val != None and val_mask != None:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                adj_val = adj_val.to(device)
                val_mask = val_mask.to(device)
                if adj_val_cond:
                    adj_val_cond = adj_val_cond.to(device)
                if x_test != None and y_test != None and adj_test != None and test_mask != None:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    adj_test = adj_test.to(device)
                    test_mask = test_mask.to(device)
                    if adj_test_cond:
                        adj_test_cond = adj_test_cond.to(device)
        
        if epoch % 50 == 0:
            if flow_control:
                model.update_density(max(model.get_density()-0.005,0.85))
                print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, training time: {time_per_epoch:.5f}, Density: {model.get_density()}')
            else:
                print(f'Epoch: {epoch:03d}, Loss: {loss:.10f}, training time: {time_per_epoch:.5f}')
            print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
        
        # evaluation
        if x_val != None and y_val != None:
            acc, micro_f1, sens, spec, _ = evaluate(model, x_val, y_val, adj_val, 
                                                 val_mask, device, adj_cond = adj_val_cond, flow_flag=flow_flag)
            validation_acc.append(acc)
            validation_spec.append(spec)
            if epoch % 100 == 0:
                print(f"Val accuracy: {acc}, Val micro_f1: {micro_f1}, Val Sens: {sens}, Val Spec: {spec}")
            
            if acc > max_val_acc:
                max_val_acc = acc
                max_val_f1 = micro_f1
                max_val_sens = sens
                max_val_spec = spec
                
                if (x_test != None and y_test != None):
                    acc, micro_f1, sens, spec, _ = evaluate(model, x_test, y_test, 
                                                         adj_test, test_mask, device, adj_cond = adj_test_cond, flow_flag=flow_flag)
                    max_val_test_acc = acc
                    max_val_test_f1 = micro_f1
                    max_val_test_sens = sens
                    max_val_test_spec = spec
                    
                    print("===========================================Best Model Update:=======================================")
                    print(f"Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}")
                    print(f"Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}")
                    print("====================================================================================================")

    print("Best Model:")
    print(f"Val accuracy: {max_val_acc}, Val f1: {max_val_f1}, Val Sens: {max_val_sens}, Val Spec: {max_val_spec}")
    print(f"Test accuracy: {max_val_test_acc}, Test f1: {max_val_test_f1}, Test Sens: {max_val_test_sens}, Test Spec: {max_val_test_spec}")
    print(f"Average time per epoch: {time_arr[10:].mean()}") # don't include the first few epoch (slower due to Torch initialization)
    print(f"Training GPU Memory Usage: {train_memory} MB")
    print(f"Peak GPU Memory Usage: {torch.cuda.max_memory_allocated(device)*2**(-20)} MB")
    
    # cleaning memory and stats
    session_memory = torch.cuda.max_memory_allocated(device)*2**(-20)
    train_time_avg = time_arr[10:].mean()
    del x_val
    del y_val
    del x_test
    del y_test
    model = model.to('cpu')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    plot_validation_metrics(validation_acc, validation_spec, num_epoch)
    
    return (max_val_acc, max_val_f1, max_val_sens, max_val_spec, max_val_test_acc,
            max_val_test_f1, max_val_test_sens, max_val_test_spec, session_memory, 
            train_memory, train_time_avg)
