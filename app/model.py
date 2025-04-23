import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(10, 2)  # Output size is 2

    def forward(self, x):
        return self.fc(x)

class MyModel:
    def __init__(self, model_path: str):
        self.model = DummyModel()
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        self.model.eval()

    def predict(self, data):
        with torch.inference_mode():
            input_tensor = torch.tensor(data, dtype=torch.float32)
            return self.model(input_tensor).tolist()