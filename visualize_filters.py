import torch
import matplotlib.pyplot as plt
from einops import rearrange
from test_template import SimpleMLP # This imports your class definition

def visualize_mnist_filters(model_path="mnist_model.pt"):
    # 1. Initialize the architecture
    model = SimpleMLP()
    
    # 2. Load the saved weights into the architecture
    # map_location ensures it works even if you trained on MPS/GPU but visualize on CPU
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # 3. Extract the weight matrix of the FIRST linear layer
    # model.net[0] is Rearrange, model.net[1] is the first Linear layer
    weights = model.net[1].weight.data # Shape: [512, 784]

    # 4. Rearrange the first 25 neurons into a 5x5 grid of 28x28 images
    # gr=5, gc=5 (5*5=25 neurons)
    # h=28, w=28 (the original image dimensions)
    grid = rearrange(weights[:128], '(gr gc) (h w) -> (gr h) (gc w)', gr=4, gc=4, h=28, w=28)

    # 5. Plotting
    plt.figure(figsize=(10, 10))
    # 'RdBu_r' is Red-Blue-Reversed, making Positive=Blue and Negative=Red
    # vmin and vmax ensure that 0 is exactly in the middle (white)
    v_max = weights[:25].abs().max().item() # Find the strongest weight for scaling
    plt.imshow(grid.numpy(), cmap='RdBu_r', vmin=-v_max, vmax=v_max)
    plt.colorbar(label='Weight Intensity')
    plt.title("What the Neurons See: First Layer Weights")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    visualize_mnist_filters()