from fastapi import FastAPI
import joblib
import pandas as pd
from prometheus_client import generate_latest
from fastapi.responses import Response
import time

from prometheus_exporter import (
    REQUEST_COUNT,
    PREDICTION_COUNT,
    REQUEST_LATENCY
)
app = FastAPI(
    title="Heart Disease Prediction API"
)

# Load model terbaik
model = joblib.load(
    "../Membangun_model/best_random_forest.pkl"
)

@app.get("/")
def home():
    return {
        "message": "Heart Disease API Running"
    }
@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type="text/plain"
    )
@app.post("/predict")
def predict(data: dict):

    start_time = time.time()

    REQUEST_COUNT.inc()

    df = pd.DataFrame([data])

    prediction = model.predict(df)

    PREDICTION_COUNT.inc()

    REQUEST_LATENCY.observe(
        time.time() - start_time
    )

    return {
        "prediction": int(prediction[0])
    }                      