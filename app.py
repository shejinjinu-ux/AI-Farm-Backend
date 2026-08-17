from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import pandas as pd
import urllib.request
import urllib.parse
import json
import csv
import os
from functools import lru_cache


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
# FILE PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "crop_yield_model_small.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "preprocessor.pkl"
)

LOCATION_CSV = os.path.join(
    BASE_DIR,
    "location_data",
    "village-directory.csv"
)


# =========================================================
# LOAD ML MODEL
# =========================================================

print("Loading ML model...")

model = joblib.load(MODEL_PATH)

print("Loading preprocessor...")

preprocessor = joblib.load(PREPROCESSOR_PATH)

print("ML model loaded successfully.")


# =========================================================
# LOCATION CSV HELPER
# =========================================================

def location_file_exists():

    if not os.path.exists(LOCATION_CSV):

        print(
            "WARNING: Location CSV not found:"
        )

        print(LOCATION_CSV)

        return False

    return True


def clean_row(row):

    return {

        "state_code":
            row.get(
                "State code",
                ""
            ).strip(),

        "state":
            row.get(
                "State Name(In English)",
                ""
            ).strip(),

        "district_code":
            row.get(
                "District code",
                ""
            ).strip(),

        "district":
            row.get(
                "District Name(In English)",
                ""
            ).strip(),

        "subdistrict_code":
            row.get(
                "Subdistrict code",
                ""
            ).strip(),

        "subdistrict":
            row.get(
                "Subdistrict Name(In English)",
                ""
            ).strip(),

        "village_code":
            row.get(
                "Village code",
                ""
            ).strip(),

        "village":
            row.get(
                "Village Name(In English)",
                ""
            ).strip()
    }


# =========================================================
# LOCATION CSV STREAM
#
# IMPORTANT:
# We DO NOT load the entire 71 MB CSV into RAM.
# The file is read one row at a time.
# =========================================================

def location_rows():

    if not location_file_exists():

        return

    try:

        with open(
            LOCATION_CSV,
            "r",
            encoding="utf-8-sig",
            newline="",
            buffering=1024 * 1024
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                yield clean_row(row)

    except Exception as e:

        print(
            "Location CSV read error:",
            e
        )


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {

        "message":
            "AI Farm Backend is running!",

        "model":
            "Random Forest - 50 Trees",

        "status":
            "ready",

        "features": [

            "ML Prediction",

            "Live Weather",

            "Solar Radiation",

            "Last 7 Days Rainfall",

            "7 Day Weather Forecast",

            "Tomorrow Rain Recommendation",

            "India State District Village Selection",

            "Village Latitude Longitude"
        ]
    }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health():

    return {

        "status":
            "OK",

        "model":
            "crop_yield_model_small.pkl",

        "preprocessor":
            "preprocessor.pkl",

        "location_data":
            "streaming CSV",

        "location_file_exists":
            location_file_exists()
    }


# =========================================================
# GET ALL INDIA STATES / UNION TERRITORIES
#
# Memory safe:
# Only unique state names are stored.
# =========================================================

@app.get("/locations/states")
def get_states():

    if not location_file_exists():

        raise HTTPException(

            status_code=500,

            detail="Location CSV file not found."
        )

    states = {}

    for row in location_rows():

        state = row["state"]

        if not state:
            continue

        key = state.lower()

        if key not in states:

            states[key] = {

                "name":
                    state,

                "code":
                    row["state_code"]
            }

    result = sorted(

        states.values(),

        key=lambda x:
            x["name"].lower()
    )

    return {

        "success":
            True,

        "count":
            len(result),

        "states":
            result
    }


# =========================================================
# GET ALL DISTRICTS OF SELECTED STATE
# =========================================================

@app.get("/locations/districts")
def get_districts(

    state: str
):

    if not location_file_exists():

        raise HTTPException(

            status_code=500,

            detail="Location CSV file not found."
        )

    state_input = (
        state.strip().lower()
    )

    districts = {}

    for row in location_rows():

        if (
            row["state"]
            .strip()
            .lower()
            != state_input
        ):

            continue

        district = row["district"]

        if not district:
            continue

        key = district.lower()

        if key not in districts:

            districts[key] = {

                "name":
                    district,

                "code":
                    row["district_code"]
            }

    result = sorted(

        districts.values(),

        key=lambda x:
            x["name"].lower()
    )

    return {

        "success":
            True,

        "state":
            state,

        "count":
            len(result),

        "districts":
            result
    }


# =========================================================
# GET ALL VILLAGES OF SELECTED DISTRICT
# =========================================================

@app.get("/locations/villages")
def get_villages(

    state: str,

    district: str
):

    if not location_file_exists():

        raise HTTPException(

            status_code=500,

            detail="Location CSV file not found."
        )

    state_input = (
        state.strip().lower()
    )

    district_input = (
        district.strip().lower()
    )

    villages = {}

    for row in location_rows():

        if (
            row["state"]
            .strip()
            .lower()
            != state_input
        ):

            continue

        if (
            row["district"]
            .strip()
            .lower()
            != district_input
        ):

            continue

        village = row["village"]

        if not village:
            continue

        key = (

            row["village_code"],

            village.lower()
        )

        villages[key] = {

            "name":
                village,

            "code":
                row["village_code"],

            "subdistrict":
                row["subdistrict"],

            "subdistrict_code":
                row["subdistrict_code"]
        }

    result = sorted(

        villages.values(),

        key=lambda x:
            x["name"].lower()
    )

    return {

        "success":
            True,

        "state":
            state,

        "district":
            district,

        "count":
            len(result),

        "villages":
            result
    }


# =========================================================
# VILLAGE GEOCODING
# =========================================================

@lru_cache(maxsize=500)
def geocode_village(

    village: str,

    district: str,

    state: str
):

    query = (

        f"{village}, "
        f"{district}, "
        f"{state}, India"
    )

    try:

        encoded_query = urllib.parse.quote(
            query
        )

        url = (

            "https://nominatim.openstreetmap.org/search"

            f"?q={encoded_query}"

            "&format=json"

            "&limit=1"

            "&countrycodes=in"
        )

        request = urllib.request.Request(

            url,

            headers={

                "User-Agent":
                    "AI-Farm-Assistant/1.0"
            }
        )

        with urllib.request.urlopen(

            request,

            timeout=10

        ) as response:

            data = json.loads(

                response.read().decode()
            )

        if not data:

            return None

        return {

            "latitude":
                float(
                    data[0]["lat"]
                ),

            "longitude":
                float(
                    data[0]["lon"]
                ),

            "display_name":
                data[0].get(
                    "display_name",
                    query
                )
        }

    except Exception as e:

        print(
            "Geocoding error:",
            e
        )

        return None


# =========================================================
# GET LATITUDE AND LONGITUDE FOR VILLAGE
# =========================================================

@app.get("/locations/village-location")
def get_village_location(

    village: str,

    district: str,

    state: str
):

    result = geocode_village(

        village.strip(),

        district.strip(),

        state.strip()
    )

    if result is None:

        return {

            "success":
                False,

            "message":
                "Coordinates not found for this village.",

            "village":
                village,

            "district":
                district,

            "state":
                state
        }

    return {

        "success":
            True,

        "village":
            village,

        "district":
            district,

        "state":
            state,

        "latitude":
            result["latitude"],

        "longitude":
            result["longitude"],

        "display_name":
            result["display_name"]
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

        current = weather_data[
            "current"
        ]

        return {

            "success":
                True,

            "location": {

                "latitude":
                    latitude,

                "longitude":
                    longitude
            },

            "weather": {

                "temperature":
                    current.get(
                        "temperature_2m"
                    ),

                "humidity":
                    current.get(
                        "relative_humidity_2m"
                    ),

                "precipitation":
                    current.get(
                        "precipitation"
                    ),

                "wind_speed":
                    current.get(
                        "wind_speed_10m"
                    ),

                "weather_code":
                    current.get(
                        "weather_code"
                    ),

                "solar_radiation_current":
                    current.get(
                        "shortwave_radiation"
                    )
            },

            "timezone":
                weather_data.get(
                    "timezone"
                ),

            "updated_at":
                current.get(
                    "time"
                )
        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                f"Weather API error: {str(e)}"
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

                "date":
                    date,

                "rainfall_mm":
                    round(
                        float(
                            amount or 0
                        ),
                        2
                    )
            })

        total_rainfall = sum(

            item["rainfall_mm"]

            for item
            in rainfall_history
        )

        average_rainfall = (

            total_rainfall
            /
            len(rainfall_history)

            if rainfall_history

            else 0
        )

        return {

            "success":
                True,

            "location": {

                "latitude":
                    latitude,

                "longitude":
                    longitude
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

            "unit":
                "mm",

            "timezone":
                rainfall_data.get(
                    "timezone"
                )
        }

    except Exception as e:

        raise HTTPException(

            status_code=500,

            detail=
                f"Rainfall API error: {str(e)}"
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

        for i in range(
            len(dates)
        ):

            forecast.append({

                "date":
                    dates[i],

                "max_temperature": (

                    max_temperatures[i]

                    if i <
                    len(max_temperatures)

                    else None
                ),

                "min_temperature": (

                    min_temperatures[i]

                    if i <
                    len(min_temperatures)

                    else None
                ),

                "precipitation_mm": (

                    precipitation[i]

                    if i <
                    len(precipitation)

                    else None
                ),

                "rain_probability": (

                    rain_probability[i]

                    if i <
                    len(rain_probability)

                    else None
                ),

                "weather_code": (

                    weather_codes[i]

                    if i <
                    len(weather_codes)

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

                tomorrow[
                    "precipitation_mm"
                ]

                or 0
            )

            probability_value = (

                tomorrow[
                    "rain_probability"
                ]

                or 0
            )

            rain_expected = (

                precipitation_value > 0

                or

                probability_value >= 50
            )


        return {

            "success":
                True,

            "location": {

                "latitude":
                    latitude,

                "longitude":
                    longitude
            },

            "tomorrow":
                tomorrow,

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

            detail=
                f"Forecast API error: {str(e)}"
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
def predict(

    data: PredictionRequest
):

    try:

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


        # =================================================
        # PREPROCESS
        # =================================================

        input_encoded = (

            preprocessor.transform(
                input_data
            )
        )


        # =================================================
        # AI PREDICTION
        # =================================================

        predicted_yield = (

            model.predict(
                input_encoded
            )[0]
        )


        # =================================================
        # TOTAL PRODUCTION
        # =================================================

        total_production = (

            predicted_yield
            *
            data.area
        )


        # =================================================
        # RESPONSE
        # =================================================

        return {

            "success":
                True,

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
                    float(
                        predicted_yield
                    ),
                    2
                ),

            "estimated_total_production":
                round(
                    float(
                        total_production
                    ),
                    2
                )
        }


    except Exception as e:

        raise HTTPException(

            status_code=400,

            detail=str(e)
        )