from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import urllib.request
import json


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(title="AI Farm Backend")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# LOAD ML MODEL
# =========================================================

model = joblib.load("crop_yield_model_small.pkl")
preprocessor = joblib.load("preprocessor.pkl")


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "message": "AI Farm Backend is running!",
        "model": "Random Forest - 50 Trees",
        "status": "ready",
        "features": [
            "ML Prediction",
            "Live Weather",
            "Solar Radiation",
            "Last 7 Days Rainfall",
            "7 Day Weather Forecast",
            "Tomorrow Rain Recommendation"
        ]
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "OK",
        "model": "crop_yield_model_small.pkl",
        "preprocessor": "preprocessor.pkl"
    }


# =========================================================
# LIVE CURRENT WEATHER API
# =========================================================

@app.get("/weather")
def get_weather(
    latitude: float,
    longitude: float
):

    try:

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            "&current="
            "temperature_2m,"
            "relative_humidity_2m,"
            "precipitation,"
            "wind_speed_10m,"
            "weather_code,"
            "shortwave_radiation"
            "&timezone=auto"
        )

        with urllib.request.urlopen(
            url,
            timeout=10
        ) as response:

            weather_data = json.loads(
                response.read().decode()
            )

        current = weather_data["current"]

        return {

            "success": True,

            "location": {
                "latitude": latitude,
                "longitude": longitude
            },

            "weather": {

                "temperature": current.get(
                    "temperature_2m"
                ),

                "humidity": current.get(
                    "relative_humidity_2m"
                ),

                "precipitation": current.get(
                    "precipitation"
                ),

                "wind_speed": current.get(
                    "wind_speed_10m"
                ),

                "weather_code": current.get(
                    "weather_code"
                ),

                "solar_radiation_current":
                    current.get(
                        "shortwave_radiation"
                    )
            },

            "timezone": weather_data.get(
                "timezone"
            ),

            "updated_at": current.get(
                "time"
            )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Weather API error: {str(e)}"
        )


# =========================================================
# LAST 7 DAYS RAINFALL API
# =========================================================

@app.get("/rainfall-7days")
def get_last_7_days_rainfall(
    latitude: float,
    longitude: float
):

    try:

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            "&daily=precipitation_sum"
            "&past_days=7"
            "&forecast_days=0"
            "&timezone=auto"
        )

        with urllib.request.urlopen(
            url,
            timeout=10
        ) as response:

            rainfall_data = json.loads(
                response.read().decode()
            )

        daily = rainfall_data.get(
            "daily",
            {}
        )

        dates = daily.get(
            "time",
            []
        )

        rainfall_values = daily.get(
            "precipitation_sum",
            []
        )

        rainfall_history = []

        for date, amount in zip(
            dates,
            rainfall_values
        ):

            rainfall_history.append({

                "date": date,

                "rainfall_mm": round(
                    float(amount or 0),
                    2
                )
            })

        total_rainfall = sum(
            item["rainfall_mm"]
            for item in rainfall_history
        )

        average_rainfall = (
            total_rainfall /
            len(rainfall_history)
            if rainfall_history
            else 0
        )

        return {

            "success": True,

            "location": {
                "latitude": latitude,
                "longitude": longitude
            },

            "rainfall_history":
                rainfall_history,

            "total_rainfall_7_days":
                round(
                    total_rainfall,
                    2
                ),

            "average_daily_rainfall":
                round(
                    average_rainfall,
                    2
                ),

            "unit": "mm",

            "timezone":
                rainfall_data.get(
                    "timezone"
                )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Rainfall API error: {str(e)}"
        )


# =========================================================
# 7 DAY WEATHER FORECAST API
# =========================================================

@app.get("/forecast")
def get_weather_forecast(
    latitude: float,
    longitude: float
):

    try:

        url = (
            "https://api.open-meteo.com/v1/forecast"
            f"?latitude={latitude}"
            f"&longitude={longitude}"
            "&daily="
            "temperature_2m_max,"
            "temperature_2m_min,"
            "precipitation_sum,"
            "precipitation_probability_max,"
            "weather_code"
            "&forecast_days=7"
            "&timezone=auto"
        )

        with urllib.request.urlopen(
            url,
            timeout=10
        ) as response:

            forecast_data = json.loads(
                response.read().decode()
            )

        daily = forecast_data.get(
            "daily",
            {}
        )

        dates = daily.get(
            "time",
            []
        )

        max_temperatures = daily.get(
            "temperature_2m_max",
            []
        )

        min_temperatures = daily.get(
            "temperature_2m_min",
            []
        )

        precipitation = daily.get(
            "precipitation_sum",
            []
        )

        rain_probability = daily.get(
            "precipitation_probability_max",
            []
        )

        weather_codes = daily.get(
            "weather_code",
            []
        )

        forecast = []

        for i in range(len(dates)):

            forecast.append({

                "date": dates[i],

                "max_temperature": (
                    max_temperatures[i]
                    if i < len(max_temperatures)
                    else None
                ),

                "min_temperature": (
                    min_temperatures[i]
                    if i < len(min_temperatures)
                    else None
                ),

                "precipitation_mm": (
                    precipitation[i]
                    if i < len(precipitation)
                    else None
                ),

                "rain_probability": (
                    rain_probability[i]
                    if i < len(rain_probability)
                    else None
                ),

                "weather_code": (
                    weather_codes[i]
                    if i < len(weather_codes)
                    else None
                )
            })


        # =================================================
        # TOMORROW FORECAST
        # =================================================

        tomorrow = (
            forecast[1]
            if len(forecast) > 1
            else None
        )

        rain_expected = False

        if tomorrow:

            precipitation_value = (
                tomorrow["precipitation_mm"]
                or 0
            )

            probability_value = (
                tomorrow["rain_probability"]
                or 0
            )

            rain_expected = (
                precipitation_value > 0
                or probability_value >= 50
            )


        # =================================================
        # RESPONSE
        # =================================================

        return {

            "success": True,

            "location": {
                "latitude": latitude,
                "longitude": longitude
            },

            "tomorrow": tomorrow,

            "tomorrow_rain_expected":
                rain_expected,

            "forecast":
                forecast,

            "timezone":
                forecast_data.get(
                    "timezone"
                )
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Forecast API error: {str(e)}"
        )


# =========================================================
# PREDICTION INPUT
# =========================================================

class PredictionRequest(BaseModel):

    state: str
    district: str
    crop: str

    area: float

    N: float
    P: float
    K: float

    temperature: float
    humidity: float
    ph: float
    rainfall: float
    wind_speed: float
    solar_radiation: float

    soil_type: str


# =========================================================
# ML PREDICTION API
# =========================================================

@app.post("/predict")
def predict(data: PredictionRequest):

    try:

        # -------------------------------------------------
        # CREATE DATAFRAME
        # EXACT TRAINING COLUMN ORDER
        # -------------------------------------------------

        input_data = pd.DataFrame([{

            "State Name":
                data.state.strip(),

            "District Name":
                data.district.strip(),

            "Crop":
                data.crop.strip(),

            "Area":
                data.area,

            "N":
                data.N,

            "P":
                data.P,

            "K":
                data.K,

            "Temperature_C":
                data.temperature,

            "Humidity_%":
                data.humidity,

            "pH":
                data.ph,

            "Rainfall_mm":
                data.rainfall,

            "Wind_Speed_m_s":
                data.wind_speed,

            "Solar_Radiation_MJ_m2_day":
                data.solar_radiation,

            "Soil_Type":
                data.soil_type.strip()
        }])


        # -------------------------------------------------
        # PREPROCESS
        # -------------------------------------------------

        input_encoded = preprocessor.transform(
            input_data
        )


        # -------------------------------------------------
        # AI PREDICTION
        # -------------------------------------------------

        predicted_yield = model.predict(
            input_encoded
        )[0]


        # -------------------------------------------------
        # TOTAL PRODUCTION
        # -------------------------------------------------

        total_production = (
            predicted_yield *
            data.area
        )


        # -------------------------------------------------
        # RESPONSE
        # -------------------------------------------------

        return {

            "success": True,

            "crop":
                data.crop,

            "state":
                data.state,

            "district":
                data.district,

            "soil_type":
                data.soil_type,

            "area":
                data.area,

            "expected_yield_per_hectare":
                round(
                    float(predicted_yield),
                    2
                ),

            "estimated_total_production":
                round(
                    float(total_production),
                    2
                )
        }


    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )