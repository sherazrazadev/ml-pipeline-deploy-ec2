# main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from joblib import load
import numpy as np
import os

from api_schema import IrisInput, PredictionOutput

# 1. Initialization and Model Loading
MODEL_PATH = 'iris_model.joblib'

# Hardcoding class names based on the Iris dataset documentation
# Target mapping: 0: setosa, 1: versicolor, 2: virginica
CLASS_NAMES = ['setosa', 'versicolor', 'virginica']

app = FastAPI(
    title="Iris Classifier API",
    version="1.0.0",
    description="A simple deployed Scikit-learn model for Iris classification."
)

# Add CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Load the model only once at application startup
# The loaded model (the full pipeline) will be attached to the app object
try:
    if os.path.exists(MODEL_PATH):
        app.model = load(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}. Prediction will fail.")
        app.model = None
except Exception as e:
    # Log the error and exit gracefully if model loading fails
    print(f"FATAL: Could not load the model: {e}")
    app.model = None


# 2. Prediction Endpoint
@app.post("/predict", response_model=PredictionOutput)
def predict(data: IrisInput):
    """
    Accepts Iris features and returns the predicted species.
    """
    if app.model is None:
        raise HTTPException(status_code=500, detail="ML Model not loaded.")

    # Convert Pydantic model to a 2D numpy array for the Scikit-learn pipeline
    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])
    
    # 3. Inference and Output Formatting
    try:
        prediction_id = app.model.predict(features)[0]
        prediction_name = CLASS_NAMES[prediction_id]
        
        return PredictionOutput(
            class_name=prediction_name,
            class_id=int(prediction_id)
        )
    except Exception as e:
        # Catch unexpected model inference errors
        print(f"Inference Error: {e}")
        raise HTTPException(status_code=500, detail="Internal inference error.")
