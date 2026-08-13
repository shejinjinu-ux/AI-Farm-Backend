from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib

app = FastAPI(title="AI Farm Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML model
model = joblib.load("model.pkl")

# Load encoders
crop_encoder = joblib.load("crop_encoder.pkl")
season_encoder = joblib.load("season_encoder.pkl")
state_encoder = joblib.load("state_encoder.pkl")


@app.get("/")
def home():
    return {
        "message": "AI Farm Backend is running!"
    }


@app.get("/health")
def health():
    return {
        "status": "OK",
        "model": "Loaded"
    }
from pydantic import BaseModel


class PredictionRequest(BaseModel):
    crop: str
    year: int
    season: str
    state: str
    area: float
    fertilizer: float
    pesticide: float
    N: float
    P: float
    K: float
    pH: float
    avg_temp_c: float
    total_rainfall_mm: float
    avg_humidity_percent: float


@app.post("/predict")
def predict(data: PredictionRequest):
    crop_encoded = crop_encoder.transform([data.crop.strip()])[0]

    season_value = data.season.strip()

    season_match = next(
        (x for x in season_encoder.classes_ if x.strip() == season_value),
        None
    )

    if season_match is None:
        raise ValueError(f"Unknown season: {data.season}")

    season_encoded = season_encoder.transform([season_match])[0]

    state_encoded = state_encoder.transform([data.state.strip()])[0]

    features = [[
        crop_encoded,
        data.year,
        season_encoded,
        state_encoded,
        data.area,
        data.fertilizer,
        data.pesticide,
        data.N,
        data.P,
        data.K,
        data.pH,
        data.avg_temp_c,
        data.total_rainfall_mm,
        data.avg_humidity_percent
    ]]

    prediction = model.predict(features)[0]

    return {
        "crop": data.crop,
        "state": data.state,
        "area": data.area,
        "expected_yield_per_hectare": round(float(prediction), 2),
        "estimated_total_production": round(float(prediction) * data.area, 2)
    }