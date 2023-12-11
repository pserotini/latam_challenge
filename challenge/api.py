import fastapi
from challenge.model import DelayModel
from fastapi import HTTPException  # Import HTTPException
import pandas as pd

app = fastapi.FastAPI()
delay_model = DelayModel()  # Initialize the DelayModel when the application starts

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OKAPA"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    delay_model.load("model_v1_0.mdl")  # Load the pre-trained model when the application starts

    input_data = pd.DataFrame(data['flights'])  # Convert input data to DataFrame
    
    # Preprocess the input data
    try:
        features = delay_model.preprocess(input_data)
    except Exception as preprocess_error:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(preprocess_error)}")
    
    # Make predictions
    try:
        predictions = delay_model.predict(features)
        return {"predictions": predictions}
    except Exception as prediction_error:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(prediction_error)}")
