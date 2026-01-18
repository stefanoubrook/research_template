import torch
import torch.nn as nn
import torch.optim as optim
from research_utils import get_data_loaders, train_step, evaluate
import random
import numpy as np
from einops.layers.torch import Rearrange
import wandb
from llc_estimation import estimate_llc


def set_seed(seed=42):
    """
    Sets the seed for reproducibility across python, numpy, and pytorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Note: These two lines make training slower but fully deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Specific for your Mac (MPS)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


class SimpleMLP(nn.Module):
    """
    A basic 2-layer Multi-Layer Perceptron (MLP).
    We use this to verify that our training boilerplate can actually 
    learn features from a standard dataset like MNIST.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Einops: "Take Batch, Channel, Height, Width and flatten CHW into one dimension"
            Rearrange('b c h w -> b (c h w)'), 
            nn.Linear(784, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 10)
        )
        
    def forward(self, x):
        """Defines the computation performed at every call."""
        return self.net(x)

def test_repo():
    """
    Main verification function to test the environment and boilerplate.
    """
    set_seed(42)
    wandb.init(
    project="mnist-slt-study",
    config={
        "learning_rate": 1e-3,
        "architecture": "SimpleMLP",
        "dataset": "MNIST",
        "epochs": 10,
        }
    )
    # 1. Hardware Detection
    # On your Mac, this should ideally pick up 'mps' if configured, otherwise 'cpu'.
    # Check for Apple Silicon GPU (MPS)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"--- Verification Started ---")
    print(f"Using device: {device}")

    # 2. Config (Professional Practice: keep params in one place)
    config = {
        "lr": 1e-3,
        "epochs": 20,
        "batch_size": 64,
        "target_accuracy": 98.0
    }

    # 3. Data & Model Initialization
    # We use the standardized loaders from our research_utils
    train_loader, test_loader = get_data_loaders(config["batch_size"])
    model = SimpleMLP().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    criterion = nn.CrossEntropyLoss()
    
    

    wandb.define_metric("epoch")
    wandb.define_metric("llc", step_metric="epoch")
    wandb.define_metric("test_accuracy", step_metric="epoch")
  
    # 4. Training Loop
    for epoch in range(1, config["epochs"] + 1):
        # Execution
        loss = train_step(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        llc = estimate_llc(model, train_loader, criterion, device)
        
        # Terminal Reporting
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.2f}% | LLC: {llc:.2f}")
        
        # W&B Logging - Use standard 'metric': value pairs
        wandb.log({
            "epoch": epoch,
            "train_loss": loss,
            "test_accuracy": acc,
            "llc": llc
         })

        # Early exit if target met
        if acc >= config["target_accuracy"]:
            print(f"Success: Target {config['target_accuracy']}% reached!")
            break

    # Save the weights so the visualization script can find them
    torch.save(model.state_dict(), "mnist_model.pt")
    print("Model saved as mnist_model.pt")

if __name__ == "__main__":
    test_repo()