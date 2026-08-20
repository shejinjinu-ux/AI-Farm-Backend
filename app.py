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

from google import genai
from dotenv import load_dotenv


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="AGRIGENIE Backend",
    version="1.0.0"
)


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
# BASE DIRECTORY
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


# =========================================================
# FILE PATHS
# =========================================================

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
# GEMINI CONFIGURATION
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

gemini_client = None

if GEMINI_API_KEY:

    try:

        gemini_client = genai.Client(
            api_key=GEMINI_API_KEY
        )

        print(
            "Gemini AI configured successfully."
        )

    except Exception as e:

        print(
            "Gemini configuration error:",
            e
        )

else:

    print(
        "WARNING: GEMINI_API_KEY not found."
    )


# =========================================================
# LOAD ML MODEL
# =========================================================

print(
    "Loading ML model..."
)

try:

    model = joblib.load(
        MODEL_PATH
    )

    print(
        "ML model loaded successfully."
    )

except Exception as e:

    print(
        "ERROR loading ML model:",
        e
    )

    model = None


# =========================================================
# LOAD PREPROCESSOR
# =========================================================

print(
    "Loading preprocessor..."
)

try:

    preprocessor = joblib.load(
        PREPROCESSOR_PATH
    )

    print(
        "Preprocessor loaded successfully."
    )

except Exception as e:

    print(
        "ERROR loading preprocessor:",
        e
    )

    preprocessor = None


# =========================================================
# LOCATION CSV
# =========================================================

def location_file_exists():

    if not os.path.exists(
        LOCATION_CSV
    ):

        print(
            "WARNING: Location CSV not found:"
        )

        print(
            LOCATION_CSV
        )

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
# STREAM LOCATION CSV
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

            reader = csv.DictReader(
                file
            )

            for row in reader:

                yield clean_row(
                    row
                )

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
            "AGRIGENIE Backend is running!",

        "status":
            "ready",

        "model":
            "crop_yield_model_small.pkl",

        "features": [

            "ML Yield Prediction",

            "Live Farm Weather",

            "7 Day Rainfall",

            "7 Day Weather Forecast",

            "India State District Village Selection",

            "Village Coordinates",

            "Smart Irrigation",

            "AGRIGENIE AI Assistant"
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

        "yield_model_loaded":
            model is not None,

        "preprocessor_loaded":
            preprocessor is not None,

        "location_file_exists":
            location_file_exists(),

        "gemini_configured":
            gemini_client is not None
    }


# =========================================================
# STATES
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
# DISTRICTS
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
# VILLAGES
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

        key = village.lower()

        if key not in villages:

            villages[key] = {

                "name":
                    village,

                "code":
                    row["village_code"]
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
# GEOCODE VILLAGE
# =========================================================

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
                    "AGRIGENIE-Farm-Assistant/1.0"
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
# VILLAGE LOCATION
# =========================================================

@app.get(
    "/locations/village-location"
)
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
# LIVE FARM WEATHER
#
# IMPORTANT:
# The frontend must send the SAVED FARM coordinates.
# This endpoint does not request browser GPS.
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

        current = weather_data.get(
            "current",
            {}
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

        print(
            "Weather API error:",
            e
        )

        raise HTTPException(
            status_code=500,
            detail=
                f"Weather API error: {str(e)}"
        )


# =========================================================
# LAST 7 DAYS RAINFALL
# =========================================================

@app.get("/rainfall-7days")
def get_rainfall_7days(
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

            "&forecast_days=1"

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

        for i in range(
            len(dates)
        ):

            value = (

                rainfall_values[i]

                if i <
                len(rainfall_values)

                else 0
            )

            rainfall_history.append({

                "date":
                    dates[i],

                "rainfall_mm":
                    value
            })

        total_rainfall = sum(

            float(
                item["rainfall_mm"] or 0
            )

            for item
            in rainfall_history
        )

        average_rainfall = (

            total_rainfall /
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
# 7 DAY WEATHER FORECAST
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

        max_temperature = daily.get(
            "temperature_2m_max",
            []
        )

        min_temperature = daily.get(
            "temperature_2m_min",
            []
        )

        precipitation = daily.get(
            "precipitation_sum",
            []
        )

        probability = daily.get(
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

                "max_temperature":
                    max_temperature[i]
                    if i < len(max_temperature)
                    else None,

                "min_temperature":
                    min_temperature[i]
                    if i < len(min_temperature)
                    else None,

                "precipitation_mm":
                    precipitation[i]
                    if i < len(precipitation)
                    else None,

                "rain_probability":
                    probability[i]
                    if i < len(probability)
                    else None,

                "weather_code":
                    weather_codes[i]
                    if i < len(weather_codes)
                    else None
            })

        tomorrow = (

            forecast[1]

            if len(forecast) > 1

            else None
        )

        rain_expected = False

        if tomorrow:

            tomorrow_rain = (
                tomorrow["precipitation_mm"]
                or 0
            )

            tomorrow_probability = (
                tomorrow["rain_probability"]
                or 0
            )

            rain_expected = (

                tomorrow_rain > 0

                or

                tomorrow_probability >= 50
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
# YIELD PREDICTION REQUEST
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
# ML YIELD PREDICTION
# =========================================================

@app.post("/predict")
def predict(
    data: PredictionRequest
):

    if model is None:

        raise HTTPException(
            status_code=503,
            detail="Yield model is not loaded."
        )

    if preprocessor is None:

        raise HTTPException(
            status_code=503,
            detail="Preprocessor is not loaded."
        )

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

        input_encoded = (
            preprocessor.transform(
                input_data
            )
        )

        predicted_yield = (
            model.predict(
                input_encoded
            )[0]
        )

        total_production = (
            predicted_yield *
            data.area
        )

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

        print(
            "Prediction Error:",
            repr(e)
        )

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# =========================================================
# IRRIGATION REQUEST
# =========================================================

class IrrigationRequest(BaseModel):

    crop: str

    soil: str

    area_acres: float

    soil_moisture: float

    temperature: float

    rainfall: float

    growth_stage: str = "Unknown"


# =========================================================
# SMART IRRIGATION
#
# Uses:
#   1. Crop
#   2. Soil
#   3. Farm area
#   4. Soil moisture
#   5. Live temperature
#   6. Live rainfall
#   7. Growth stage
#
# This is a decision-support rule engine.
# It does NOT claim exact water requirement.
# =========================================================

@app.post("/irrigation")
def irrigation_recommendation(
    data: IrrigationRequest
):

    # -----------------------------------------------------
    # CLEAN VALUES
    # -----------------------------------------------------

    crop = (
        data.crop.strip()
    )

    soil = (
        data.soil.strip()
    )

    stage = (
        data.growth_stage
        .strip()
        .lower()
    )

    moisture = max(
        0,
        min(
            100,
            float(
                data.soil_moisture
            )
        )
    )

    temperature = float(
        data.temperature
    )

    rainfall = max(
        0,
        float(
            data.rainfall
        )
    )

    area_acres = max(
        0,
        float(
            data.area_acres
        )
    )

    # -----------------------------------------------------
    # STAGE FLAGS
    # -----------------------------------------------------

    is_early_stage = (
        "germination" in stage
        or
        "seedling" in stage
    )

    is_tillering = (
        "tiller" in stage
    )

    is_flowering = (
        "flower" in stage
    )

    is_grain_filling = (
        "grain" in stage
    )

    is_maturity = (
        "maturity" in stage
        or
        "harvest" in stage
    )

    # -----------------------------------------------------
    # WEATHER PRESSURE
    # -----------------------------------------------------

    high_temperature = (
        temperature >= 35
    )

    warm_temperature = (
        temperature >= 30
    )

    significant_rain = (
        rainfall >= 5
    )

    light_rain = (
        rainfall > 0
        and
        rainfall < 5
    )

    # -----------------------------------------------------
    # DEFAULT VALUES
    # -----------------------------------------------------

    status = "Monitor"

    urgency = "Low"

    estimated_action = (
        "Inspect the field before irrigation."
    )

    recommendation = ""

    # -----------------------------------------------------
    # CASE 1
    # HIGH RAINFALL
    # -----------------------------------------------------

    if significant_rain:

        status = (
            "Skip irrigation"
        )

        urgency = "Low"

        estimated_action = (
            "Do not add irrigation immediately."
        )

        recommendation = (

            f"{crop} field has received "
            f"{rainfall:.1f} mm of current rainfall. "
            "Avoid unnecessary irrigation now. "
            "Check field water level and soil condition "
            "before the next irrigation cycle."
        )

    # -----------------------------------------------------
    # CASE 2
    # HIGH MOISTURE
    # -----------------------------------------------------

    elif moisture >= 60:

        status = (
            "No irrigation needed"
        )

        urgency = "Low"

        estimated_action = (
            "Skip the next irrigation cycle."
        )

        recommendation = (

            f"Soil moisture is {moisture:.0f}%, "
            "which is relatively high. "
            "Irrigation is not recommended right now. "
            "Continue monitoring the field and avoid "
            "unnecessary water application."
        )

    # -----------------------------------------------------
    # CASE 3
    # LOW MOISTURE + HIGH TEMPERATURE
    # -----------------------------------------------------

    elif moisture < 30 and high_temperature:

        status = (
            "Irrigation may be required soon"
        )

        urgency = "High"

        estimated_action = (
            "Inspect the crop now and irrigate "
            "if visible water stress is present."
        )

        recommendation = (

            f"Soil moisture is low at {moisture:.0f}% "
            f"and temperature is high at "
            f"{temperature:.1f}°C. "
            "The crop may experience water stress. "
            "Check the field immediately. "
            "If plants or soil show water stress, "
            "start irrigation rather than waiting "
            "for the next routine cycle."
        )

    # -----------------------------------------------------
    # CASE 4
    # LOW MOISTURE
    # -----------------------------------------------------

    elif moisture < 30:

        status = (
            "Irrigation may be required"
        )

        urgency = "High"

        estimated_action = (
            "Inspect the field and consider irrigation."
        )

        recommendation = (

            f"Soil moisture is low at "
            f"{moisture:.0f}%. "
            "Inspect the crop and soil for visible "
            "water stress. If the field is drying, "
            "irrigation may be required."
        )

    # -----------------------------------------------------
    # CASE 5
    # MODERATE MOISTURE + HIGH TEMPERATURE
    # -----------------------------------------------------

    elif moisture < 40 and high_temperature:

        status = (
            "Monitor closely"
        )

        urgency = "Medium"

        estimated_action = (
            "Check moisture again soon."
        )

        recommendation = (

            f"Soil moisture is {moisture:.0f}% "
            f"while temperature is {temperature:.1f}°C. "
            "Water demand can increase under hot "
            "conditions. Inspect the field regularly. "
            "If the soil surface starts drying, "
            "consider light irrigation."
        )

    # -----------------------------------------------------
    # CASE 6
    # MODERATE MOISTURE + RAIN
    # -----------------------------------------------------

    elif moisture < 50 and light_rain:

        status = (
            "Monitor"
        )

        urgency = "Low"

        estimated_action = (
            "Wait and reassess after rainfall."
        )

        recommendation = (

            f"Soil moisture is {moisture:.0f}% "
            f"and current rainfall is "
            f"{rainfall:.1f} mm. "
            "Avoid immediate heavy irrigation. "
            "Allow the rainfall to contribute moisture "
            "and reassess the field before watering."
        )

    # -----------------------------------------------------
    # CASE 7
    # MODERATE MOISTURE
    # -----------------------------------------------------

    elif moisture < 50:

        status = (
            "Monitor"
        )

        urgency = "Medium"

        estimated_action = (
            "Inspect the field before irrigation."
        )

        recommendation = (

            f"Soil moisture is {moisture:.0f}%, "
            "which is in a moderate range. "
            "Do not automatically start heavy irrigation. "
            "Inspect the soil and crop. "
            "If the field surface is drying, "
            "light irrigation may be considered."
        )

    # -----------------------------------------------------
    # CASE 8
    # GOOD MOISTURE
    # -----------------------------------------------------

    else:

        status = (
            "No immediate irrigation"
        )

        urgency = "Low"

        estimated_action = (
            "Continue monitoring."
        )

        recommendation = (

            f"Soil moisture is {moisture:.0f}%, "
            "so immediate irrigation is not indicated "
            "by the supplied conditions. "
            "Continue monitoring the field and weather."
        )

    # -----------------------------------------------------
    # GROWTH-STAGE NOTE
    # -----------------------------------------------------

    if is_flowering:

        stage_note = (

            "The crop is in the flowering stage. "
            "Avoid severe water stress and monitor "
            "soil moisture closely."
        )

    elif is_tillering:

        stage_note = (

            "The crop is in the tillering stage. "
            "Regular moisture monitoring is important "
            "for healthy crop development."
        )

    elif is_grain_filling:

        stage_note = (

            "The crop is in the grain-filling stage. "
            "Avoid unnecessary drying or excessive "
            "water application."
        )

    elif is_early_stage:

        stage_note = (

            "The crop is at an early growth stage. "
            "Maintain suitable moisture and avoid "
            "unnecessary waterlogging."
        )

    elif is_maturity:

        stage_note = (

            "The crop is approaching maturity. "
            "Avoid unnecessary irrigation and follow "
            "crop-specific water management."
        )

    else:

        stage_note = (

            "Growth stage is not fully specified. "
            "Inspect the crop condition before making "
            "major irrigation decisions."
        )

    # -----------------------------------------------------
    # TEMPERATURE NOTE
    # -----------------------------------------------------

    if high_temperature:

        temperature_note = (

            f"Temperature is high at "
            f"{temperature:.1f}°C. "
            "Monitor the crop more frequently "
            "for water stress."
        )

    elif warm_temperature:

        temperature_note = (

            f"Temperature is moderately high at "
            f"{temperature:.1f}°C. "
            "Continue monitoring soil moisture "
            "and crop condition."
        )

    else:

        temperature_note = (

            f"Current temperature is "
            f"{temperature:.1f}°C. "
            "Continue monitoring soil moisture "
            "and weather conditions."
        )

    # -----------------------------------------------------
    # AREA NOTE
    # -----------------------------------------------------

    if area_acres > 0:

        area_note = (

            f"The farm area is approximately "
            f"{area_acres:.2f} acre(s). "
            "Actual water quantity should be decided "
            "from field water level, irrigation method, "
            "soil condition and local crop practice."
        )

    else:

        area_note = (
            "Farm area is unavailable."
        )

    # -----------------------------------------------------
    # SOIL NOTE
    # -----------------------------------------------------

    soil_note = (

        f"Soil type recorded for this recommendation: "
        f"{soil}."
    )

    # -----------------------------------------------------
    # FINAL RESPONSE
    # -----------------------------------------------------

    return {

        "success":
            True,

        "crop":
            crop,

        "soil":
            soil,

        "area_acres":
            round(
                area_acres,
                2
            ),

        "soil_moisture":
            round(
                moisture,
                1
            ),

        "temperature":
            round(
                temperature,
                1
            ),

        "rainfall":
            round(
                rainfall,
                1
            ),

        "growth_stage":
            data.growth_stage,

        "status":
            status,

        "urgency":
            urgency,

        "recommended_action":
            estimated_action,

        "recommendation":
            recommendation,

        "temperature_note":
            temperature_note,

        "stage_note":
            stage_note,

        "area_note":
            area_note,

        "soil_note":
            soil_note,

        "decision_basis": [

            "Soil moisture",

            "Current rainfall",

            "Current temperature",

            "Crop",

            "Soil type",

            "Growth stage",

            "Farm area"
        ],

        "disclaimer":
            (
                "This is a farm decision-support "
                "recommendation, not an exact irrigation "
                "measurement. Inspect actual field "
                "conditions before operating irrigation."
            )
    }
# =========================================================
# AI RECOMMENDATIONS
# =========================================================

class RecommendationRequest(BaseModel):

    crop: str

    soil: str

    area_acres: float

    soil_moisture: float

    temperature: float

    rainfall: float

    growth_stage: str = "Unknown"

    humidity: float | None = None

    wind_speed: float | None = None

    solar_radiation: float | None = None

    predicted_yield: float | None = None


@app.post("/recommendations")
def recommendations(
    data: RecommendationRequest
):

    crop = data.crop.strip()
    soil = data.soil.strip()

    moisture = float(
        data.soil_moisture
    )

    temperature = float(
        data.temperature
    )

    rainfall = float(
        data.rainfall
    )

    growth_stage = (
        data.growth_stage.strip()
    )

    humidity = data.humidity
    wind_speed = data.wind_speed
    solar_radiation = data.solar_radiation

    recommendations_list = []
    priority_actions = []

    # =====================================================
    # 1. IRRIGATION
    # =====================================================

    if rainfall >= 5:

        irrigation_priority = "Low"

        irrigation_advice = (

            f"Rainfall is {rainfall:.1f} mm. "
            "Avoid unnecessary irrigation now. "
            "Allow the rainfall to contribute to soil "
            "moisture and reassess the field afterwards."
        )

        priority_actions.append(
            "Avoid unnecessary irrigation after rainfall."
        )

    elif moisture < 30:

        irrigation_priority = "High"

        irrigation_advice = (

            f"Soil moisture is low at "
            f"{moisture:.1f}%. "
            "Inspect the field and consider irrigation "
            "if the soil is visibly dry."
        )

        priority_actions.append(
            "Check the field and irrigate if soil is dry."
        )

    elif moisture < 45:

        irrigation_priority = "Medium"

        irrigation_advice = (

            f"Soil moisture is {moisture:.1f}%, "
            "which is a moderate level. "
            "Do not apply heavy irrigation immediately. "
            "Monitor the field and provide light irrigation "
            "only if the soil surface is drying."
        )

        priority_actions.append(
            "Monitor soil moisture before irrigation."
        )

    elif moisture <= 70:

        irrigation_priority = "Low"

        irrigation_advice = (

            f"Soil moisture is {moisture:.1f}%, "
            "which is currently within a reasonable range. "
            "Avoid unnecessary watering and continue monitoring."
        )

    else:

        irrigation_priority = "High"

        irrigation_advice = (

            f"Soil moisture is high at "
            f"{moisture:.1f}%. "
            "Avoid additional irrigation because excess "
            "water may cause waterlogging."
        )

        priority_actions.append(
            "Avoid irrigation while soil moisture is high."
        )

    recommendations_list.append({

        "category": "Irrigation",

        "priority": irrigation_priority,

        "recommendation":
            irrigation_advice
    })


    # =====================================================
    # 2. FERTILIZER
    # =====================================================

    stage_lower = growth_stage.lower()

    if "rice" in crop.lower():

        if "tiller" in stage_lower:

            fertilizer_advice = (

                "The crop is in the tillering stage. "
                "Maintain balanced nutrition and use "
                "soil-test-based fertilizer management. "
                "Follow the recommended local crop dose."
            )

        elif "flower" in stage_lower:

            fertilizer_advice = (

                "The crop is in the flowering stage. "
                "Maintain balanced nutrition and avoid "
                "unnecessary excess nitrogen."
            )

        else:

            fertilizer_advice = (

                f"For {crop}, use soil-test-based "
                "fertilizer management and follow the "
                "recommended dose for the current growth stage."
            )

    else:

        fertilizer_advice = (

            f"For {crop}, use soil-test results and "
            "crop-stage requirements to guide fertilizer "
            "management."
        )

    recommendations_list.append({

        "category": "Fertilizer",

        "priority": "Medium",

        "recommendation":
            fertilizer_advice
    })


    # =====================================================
    # 3. WEATHER
    # =====================================================

    if temperature >= 35:

        weather_priority = "High"

        weather_advice = (

            f"Temperature is high at "
            f"{temperature:.1f}°C. "
            "Monitor the crop more frequently for "
            "heat stress and moisture loss."
        )

        priority_actions.append(
            "Monitor the crop for heat stress."
        )

    elif temperature >= 30:

        weather_priority = "Medium"

        weather_advice = (

            f"Temperature is {temperature:.1f}°C. "
            "Warm conditions may increase water loss. "
            "Continue monitoring soil moisture."
        )

    else:

        weather_priority = "Low"

        weather_advice = (

            f"Temperature is {temperature:.1f}°C. "
            "No severe heat condition is indicated "
            "by the supplied temperature."
        )

    recommendations_list.append({

        "category": "Weather",

        "priority": weather_priority,

        "recommendation":
            weather_advice
    })


    # =====================================================
    # 4. RAINFALL
    # =====================================================

    if rainfall >= 10:

        rainfall_advice = (

            f"Rainfall is {rainfall:.1f} mm. "
            "Avoid irrigation and inspect the field "
            "for excess water or waterlogging."
        )

        rainfall_priority = "High"

    elif rainfall > 0:

        rainfall_advice = (

            f"Recent rainfall is {rainfall:.1f} mm. "
            "Consider this rainfall before deciding "
            "the next irrigation."
        )

        rainfall_priority = "Medium"

    else:

        rainfall_advice = (

            "No rainfall is currently reported. "
            "Continue monitoring soil moisture and "
            "the local weather forecast."
        )

        rainfall_priority = "Low"

    recommendations_list.append({

        "category": "Rainfall",

        "priority": rainfall_priority,

        "recommendation":
            rainfall_advice
    })


    # =====================================================
    # 5. CROP GROWTH STAGE
    # =====================================================

    if "tiller" in stage_lower:

        stage_advice = (

            "The crop is in the tillering stage. "
            "Maintain suitable moisture, control weeds "
            "early and monitor crop growth regularly."
        )

    elif "flower" in stage_lower:

        stage_advice = (

            "The crop is in the flowering stage. "
            "Avoid severe water stress and inspect "
            "the crop regularly for pest and disease symptoms."
        )

    elif "seedling" in stage_lower:

        stage_advice = (

            "The crop is at an early growth stage. "
            "Maintain suitable moisture and monitor "
            "weeds and pests."
        )

    elif (
        "maturity" in stage_lower
        or
        "harvest" in stage_lower
    ):

        stage_advice = (

            "The crop is approaching maturity. "
            "Avoid unnecessary irrigation and monitor "
            "crop maturity."
        )

    else:

        stage_advice = (

            f"The crop is currently in the "
            f"{growth_stage} stage. "
            "Monitor crop growth, soil moisture and "
            "field conditions regularly."
        )

    recommendations_list.append({

        "category": "Crop Stage",

        "priority": "Medium",

        "recommendation":
            stage_advice
    })


    # =====================================================
    # 6. PEST & DISEASE
    # =====================================================

    if (
        humidity is not None
        and
        humidity >= 80
    ):

        pest_priority = "High"

        pest_advice = (

            f"Humidity is high at {humidity:.1f}%. "
            "Inspect the crop regularly for fungal "
            "disease symptoms and maintain good field monitoring."
        )

    else:

        pest_priority = "Medium"

        pest_advice = (

            "Inspect the crop regularly for early signs "
            "of pests and diseases. Confirm the problem "
            "before applying any pesticide."
        )

    recommendations_list.append({

        "category": "Pest & Disease",

        "priority": pest_priority,

        "recommendation":
            pest_advice
    })


    # =====================================================
    # 7. WIND
    # =====================================================

    if wind_speed is not None:

        if wind_speed >= 30:

            wind_priority = "High"

            wind_advice = (

                f"Wind speed is {wind_speed:.1f} km/h. "
                "Avoid spraying during strong winds."
            )

            priority_actions.append(
                "Avoid spraying during strong winds."
            )

        else:

            wind_priority = "Low"

            wind_advice = (

                f"Wind speed is {wind_speed:.1f} km/h. "
                "No strong-wind warning is indicated."
            )

    else:

        wind_priority = "Low"

        wind_advice = (

            "Wind information is unavailable. "
            "Check local conditions before spraying."
        )

    recommendations_list.append({

        "category": "Wind",

        "priority": wind_priority,

        "recommendation":
            wind_advice
    })


    # =====================================================
    # 8. SOIL
    # =====================================================

    soil_advice = (

        f"The recorded soil type is {soil}. "
        "Use soil-test results together with crop "
        "requirements for fertilizer and soil management."
    )

    recommendations_list.append({

        "category": "Soil",

        "priority": "Medium",

        "recommendation":
            soil_advice
    })


    # =====================================================
    # 9. YIELD
    # =====================================================

    if data.predicted_yield is not None:

        yield_advice = (

            f"The current supplied yield estimate is "
            f"{data.predicted_yield}. "
            "Continue improving water management, "
            "nutrition and crop health."
        )

    else:

        yield_advice = (

            "Yield depends on crop variety, soil, "
            "weather, irrigation, nutrition and crop health. "
            "Continue monitoring these factors throughout "
            "the crop season."
        )

    recommendations_list.append({

        "category": "Yield",

        "priority": "Medium",

        "recommendation":
            yield_advice
    })


    # =====================================================
    # OVERALL STATUS
    # =====================================================

    if (
        moisture < 30
        or
        temperature >= 35
        or
        (wind_speed is not None and wind_speed >= 30)
    ):

        overall_status = "Attention Needed"

    elif (
        rainfall >= 10
        or
        moisture > 70
    ):

        overall_status = "Monitor"

    else:

        overall_status = "Good"


    # =====================================================
    # WHAT TO DO NOW
    # =====================================================

    if priority_actions:

        what_to_do_now = (
            priority_actions[0]
        )

    else:

        what_to_do_now = (

            "Continue monitoring soil moisture, "
            "weather and crop health."
        )


    # =====================================================
    # SUMMARY
    # =====================================================

    summary = (

        f"For your {crop} crop in the "
        f"{growth_stage} stage, soil moisture is "
        f"{moisture:.1f}% and current rainfall is "
        f"{rainfall:.1f} mm. "
        f"The main recommended action is: "
        f"{what_to_do_now}"
    )


    # =====================================================
    # FINAL RESPONSE
    # =====================================================

    return {

        "success": True,

        "status":
            overall_status,

        "farm": {

            "crop":
                crop,

            "soil":
                soil,

            "area_acres":
                data.area_acres,

            "growth_stage":
                growth_stage
        },

        "current_conditions": {

            "soil_moisture":
                moisture,

            "temperature":
                temperature,

            "rainfall":
                rainfall,

            "humidity":
                humidity,

            "wind_speed":
                wind_speed,

            "solar_radiation":
                solar_radiation
        },

        "summary":
            summary,

        "what_to_do_now":
            what_to_do_now,

        "priority_actions":
            priority_actions,

        "recommendations":
            recommendations_list
    }


# =========================================================
# CHAT REQUEST
# =========================================================

class ChatRequest(BaseModel):

    message: str

    language: str = "English"

    farm_context: dict | None = None

    latitude: float | None = None

    longitude: float | None = None


class ChatResponse(BaseModel):

    success: bool

    reply: str


# =========================================================
# AI FARM ASSISTANT
# =========================================================

@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(
    data: ChatRequest
):

    if gemini_client is None:

        raise HTTPException(
            status_code=503,
            detail=(
                "AI assistant is not configured. "
                "Please configure GEMINI_API_KEY."
            )
        )

    if not data.message.strip():

        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )

    try:

        # =================================================
        # FARM CONTEXT
        # =================================================

        farm_context = (
            data.farm_context
            or {}
        )

        # =================================================
        # LIVE WEATHER
        #
        # These coordinates should be the SAVED FARM
        # coordinates supplied by the frontend.
        # =================================================

        live_weather = None

        if (
            data.latitude is not None
            and
            data.longitude is not None
        ):

            try:

                weather_url = (

                    "https://api.open-meteo.com/v1/forecast"

                    f"?latitude={data.latitude}"

                    f"&longitude={data.longitude}"

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
                    weather_url,
                    timeout=10
                ) as weather_response:

                    weather_data = json.loads(
                        weather_response
                        .read()
                        .decode()
                    )

                current = weather_data.get(
                    "current",
                    {}
                )

                live_weather = {

                    "temperature_c":
                        current.get(
                            "temperature_2m"
                        ),

                    "humidity_percent":
                        current.get(
                            "relative_humidity_2m"
                        ),

                    "precipitation_mm":
                        current.get(
                            "precipitation"
                        ),

                    "wind_speed_kmh":
                        current.get(
                            "wind_speed_10m"
                        ),

                    "weather_code":
                        current.get(
                            "weather_code"
                        ),

                    "solar_radiation":
                        current.get(
                            "shortwave_radiation"
                        ),

                    "updated_at":
                        current.get(
                            "time"
                        )
                }

            except Exception as weather_error:

                print(
                    "Chat weather error:",
                    weather_error
                )

                live_weather = None

        # =================================================
        # SYSTEM INSTRUCTION
        # =================================================

        system_instruction = """

You are AGRIGENIE, an AI-powered
farmer assistance assistant.

Help farmers with:

- crop farming
- soil
- irrigation
- fertilizer
- crop yield
- weather
- farm management
- agricultural planning
- disease awareness
- pest awareness
- farm economics

IMPORTANT RULES:

1. Never guarantee crop yield.

2. Never guarantee profit.

3. Never invent weather information.

4. Never invent market prices.

5. Never invent sensor readings.

6. Never invent fertilizer dosage.

7. Never invent pesticide dosage.

8. Never claim 100 percent accuracy.

9. Use supplied farm context.

10. Never change or invent farm values.

11. Treat demo sensor values as demo values.

12. Use live weather information when supplied.

13. If live weather is unavailable,
say that it is unavailable.

14. Give simple practical answers.

15. Avoid unnecessary technical language.

16. Answer in the farmer's requested language.

17. Do not guarantee that a recommendation
will definitely increase yield.

18. For irrigation advice, consider:
soil moisture, rainfall, temperature,
crop and growth stage together.

19. Never pretend that a demo sensor value
is a real physical sensor reading.

20. If exact fertilizer or pesticide dosage
is requested, recommend following the
appropriate local agricultural guidance.

The application is called AGRIGENIE.

Be clear, practical and farmer-friendly.
"""

        # =================================================
        # FARM INFORMATION
        # =================================================

        farm_information = f"""

CURRENT FARM INFORMATION

Farmer:
{farm_context.get(
    "farmer_name",
    "Unavailable"
)}

Farm:
{farm_context.get(
    "farm_name",
    "Unavailable"
)}

Crop:
{farm_context.get(
    "crop",
    "Unavailable"
)}

Soil:
{farm_context.get(
    "soil",
    "Unavailable"
)}

Farm Area:
{farm_context.get(
    "area",
    "Unavailable"
)}

State:
{farm_context.get(
    "state",
    "Unavailable"
)}

District:
{farm_context.get(
    "district",
    "Unavailable"
)}

Nitrogen:
{farm_context.get(
    "nitrogen",
    "Unavailable"
)}

Phosphorus:
{farm_context.get(
    "phosphorus",
    "Unavailable"
)}

Potassium:
{farm_context.get(
    "potassium",
    "Unavailable"
)}

Moisture:
{farm_context.get(
    "moisture",
    "Unavailable"
)}

pH:
{farm_context.get(
    "ph",
    "Unavailable"
)}
"""

        # =================================================
        # WEATHER INFORMATION
        # =================================================

        weather_information = f"""

LIVE FARM WEATHER

Farm Latitude:
{
    data.latitude
    if data.latitude is not None
    else "Unavailable"
}

Farm Longitude:
{
    data.longitude
    if data.longitude is not None
    else "Unavailable"
}

Temperature:
{
    live_weather.get("temperature_c")
    if live_weather
    else "Unavailable"
} °C

Humidity:
{
    live_weather.get("humidity_percent")
    if live_weather
    else "Unavailable"
} %

Current Rainfall:
{
    live_weather.get("precipitation_mm")
    if live_weather
    else "Unavailable"
} mm

Wind Speed:
{
    live_weather.get("wind_speed_kmh")
    if live_weather
    else "Unavailable"
} km/h

Weather Code:
{
    live_weather.get("weather_code")
    if live_weather
    else "Unavailable"
}

Solar Radiation:
{
    live_weather.get("solar_radiation")
    if live_weather
    else "Unavailable"
}

Updated At:
{
    live_weather.get("updated_at")
    if live_weather
    else "Unavailable"
}
"""

        # =================================================
        # FINAL PROMPT
        # =================================================

        prompt = f"""

{system_instruction}

{farm_information}

{weather_information}

FARMER REQUESTED LANGUAGE:
{data.language}

FARMER QUESTION:
{data.message}

Answer the farmer directly.

Use the current farm information
when relevant.

Use live farm weather when the question
relates to:

- weather
- irrigation
- field operations
- crop management

Do not invent unavailable values.

Keep the answer simple and practical.

Do not guarantee crop yield.

Do not guarantee profit.

Do not invent fertilizer or pesticide dosages.
"""

        # =================================================
        # GEMINI REQUEST
        # =================================================

        response = (

            gemini_client
            .models
            .generate_content(

                model="gemini-3.6-flash",

                contents=prompt
            )
        )

        reply = response.text

        if not reply:

            raise Exception(
                "Gemini returned empty response."
            )

        return {

            "success":
                True,

            "reply":
                reply.strip()
        }

    except Exception as e:

        print(
            "Gemini Chat Error:",
            repr(e)
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "AI assistant is temporarily "
                "unavailable."
            )
        )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )