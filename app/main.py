from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np


app = FastAPI

# Load the model
model = torch.load("app/model/model.pth")
model.eval()

# Define the input format
class ECGPPGInput(BaseModel):
    ecg_raw: list
    ppg_raw: list

# Prediction endpoint
@app.post("/predict_blood_pressure/")
def predict_blood_pressure(input_data: ECGPPGInput):
    # Preprocessing input data
    ecg_data = np.array(input_data.ecg_raw)
    ppg_data = np.array(input_data.ppg_raw)

    # Combine both ECG and PPG data
    combined_data = np.concatenate([ecg_data, ppg_data], axis=-1)

    # Convert to tensor
    input_tensor = torch.tensor(combined_data, dtype=torch.float32)

    # Prediction
    with torch.no_grad():
        output = model(input_tensor)

    # Output twp values (systolic and diastolic BP values)
    systolic, diastolic = output.numpy()

    return {"systolic blood pressure": systolic, "diastolic blood pressure": diastolic}