# train_dummy_model.py
import torch
import torch.nn as nn

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc = nn.Linear(4, 1)  # Simple linear layer: 4 inputs -> 1 output

    def forward(self, x):
        return self.fc(x)

if __name__ == "__main__":
    model = DummyModel()
    dummy_input = torch.rand((10, 4))
    dummy_output = torch.rand((10, 1))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(100):  # dummy training loop
        optimizer.zero_grad()
        outputs = model(dummy_input)
        loss = criterion(outputs, dummy_output)
        loss.backward()
        optimizer.step()

    torch.save(model, "models/model.pt")
    print("Dummy model saved to models/model.pt")
