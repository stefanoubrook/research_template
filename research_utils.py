import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

def init_experiment(project_name, config):
    """
    Initializes Weights & Biases (W&B) tracking.
    Usage: run = init_experiment("mnist-slt-study", {"lr": 1e-3, "batch_size": 64})
    """
    return wandb.init(project=project_name, config=config)

def get_data_loaders(batch_size=64):
    """
    Creates MNIST loaders that auto-download and normalize data.
    Standard normalization for MNIST uses Mean: 0.1307 and Std: 0.3081.
    This helps the Hessian of the loss function stay well-conditioned.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Download and load the training data
    train_set = datasets.MNIST('./data', train=True, download=True, transform=transform)
    # Download and load the test data
    test_set = datasets.MNIST('./data', train=False, download=True, transform=transform)

    # DataLoader handles batching, shuffling, and multi-processing
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def train_step(model, loader, optimizer, criterion, device, l1_lambda=0.0):
    """
    Performs a single training epoch.
    Includes optional L1 regularization to encourage weight sparsity.
    """
    model.train()
    epoch_loss = 0
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # 1. Calculate the base loss (e.g., CrossEntropy)
        loss = criterion(output, target)
        
        # 2. Add L1 penalty if l1_lambda > 0
        # This sums the absolute values of all parameters (weights and biases)
        if l1_lambda > 0:
            l1_penalty = sum(p.abs().sum() for p in model.parameters())
            loss += l1_lambda * l1_penalty
            
        # 3. Standard backpropagation
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(loader)

def get_flat_gradients(model, criterion, data, target):
    """
    Flattens and concatenates all parameter gradients into a single 1D vector.
    Representing the gradient as a single vector in R^D is essential for 
    SLT geometric analysis and Hessian-vector products.
    """
    # Clear existing gradients
    model.zero_grad()
    
    # Run a forward and backward pass to populate .grad attributes
    output = model(data)
    loss = criterion(output, target)
    loss.backward()
    
    grads = []
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            # Flatten the multi-dimensional parameter (e.g., [64, 784] -> [50176])
            grads.append(param.grad.view(-1))
            
    # Concatenate all flattened gradients into one giant vector
    return torch.cat(grads)

# ==========================================
# FILE: research_utils.py (Added Function)
# ==========================================

def evaluate(model, loader, device):
    """
    Standard evaluation function to calculate accuracy on a dataset.
    
    Args:
        model: The neural network to evaluate.
        loader: DataLoader for the dataset (usually test or validation).
        device: The hardware to use (mps, cuda, or cpu).
    """
    # Set model to evaluation mode (disables dropout, fixes batchnorm stats)
    model.eval()
    
    correct = 0
    total = 0
    
    # torch.no_grad() disables the gradient calculation engine.
    # This reduces memory consumption and speeds up the process significantly.
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            
            # Forward pass only
            output = model(data)
            
            # Get the predicted class by finding the index with the highest logit
            # dim=1 refers to the class dimension
            pred = output.argmax(dim=1, keepdim=True)
            
            # Compare predictions to ground truth labels
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
    accuracy = 100. * correct / total
    return accuracy