from fastapi import FastAPI
from pydantic import BaseModel
# import torch
# from app.model import MyModel

app = FastAPI()
# model = MyModel("models/model.pt")  # Temporarily disabled for testing

class InputData(BaseModel):
    values: list[float]

@app.get("/")
def root():
    return {"message": "PyTorch Model Inference API"}

@app.post("/predict")
def predict(data: InputData):
    # tensor_input = torch.tensor(data.values).float().unsqueeze(0)  # add batch dimension
    # result = model.predict(tensor_input)
    # return {"prediction": result.tolist()}
    
    # Return dummy prediction for now
    return {"prediction": 1}
