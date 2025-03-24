import wandb
import os

wandb.login(key=os.environ.get("WANDB_API_KEY"))


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Initialize Weights and Biases
wandb.init(project="dnn-training", config={
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 100,
    "hidden_size": 128,
    "num_layers": 3
})

# Generate synthetic data with variance
torch.manual_seed(42)
num_samples = 1000
input_size = 10
output_size = 1

# Input data in range (-1, 1) with some variance
X = torch.rand(num_samples, input_size) * 2 - 1  # Range: (-1, 1)
y = torch.randn(num_samples, output_size) * 0.1 + X.mean(dim=1, keepdim=True)  # Add variance

# Create a dataset and dataloader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=wandb.config.batch_size, shuffle=True)

# Define a deep neural network
class DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(DNN, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_size, output_size))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# Initialize the model, loss, and optimizer
model = DNN(input_size, wandb.config.hidden_size, output_size, wandb.config.num_layers)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=wandb.config.learning_rate)

# Watch the model with W&B
wandb.watch(model, log="all", log_freq=10)

# Training loop
for epoch in range(wandb.config.epochs):
    for batch_X, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Log loss and gradients
        wandb.log({"loss": loss.item()})

    print(f"Epoch [{epoch+1}/{wandb.config.epochs}], Loss: {loss.item():.4f}")

wandb.finish()