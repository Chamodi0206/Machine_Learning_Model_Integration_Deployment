# model_wrapper.py
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(2, 3)  # Example: 2 input features -> 3 output classes

    def forward(self, x):
        return self.fc(x)

def load_model(path="model.pth"):
    model = MyModel()
    # If actual model.pth file is available, uncomment below:
    # model.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    model.eval()
    return model
