import torch
import torch.nn as nn
import os

# Define the dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # Output size is 2

    def forward(self, x):
        return self.fc(x)

# Instantiate and save the model weights
model = DummyModel()
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/model_weights.pt")

print("Dummy model weights saved!")
