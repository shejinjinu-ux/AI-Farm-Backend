from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter()


class IrrigationRequest(BaseModel):
    crop: str
    soil: str
    area_acres: float
    soil_moisture: float
    temperature: float
    rainfall: float
    growth_stage: str = "Unknown"


@router.post("/irrigation")
def irrigation_recommendation(data: IrrigationRequest):

    moisture = data.soil_moisture
    rainfall = data.rainfall
    temperature = data.temperature

    # Basic rule-based irrigation recommendation
    # This is intentionally conservative because
    # exact irrigation depends on field conditions.

    if rainfall > 5:
        status = "No irrigation needed"
        recommendation = (
            "Rainfall is currently significant. "
            "Avoid unnecessary irrigation and monitor "
            "the field condition."
        )

    elif moisture >= 60:
        status = "No irrigation needed"
        recommendation = (
            "Soil moisture is relatively high. "
            "Do not irrigate heavily now. "
            "Continue monitoring the field."
        )

    elif moisture >= 40:
        status = "Monitor"
        recommendation = (
            "Soil moisture is in a moderate range. "
            "Inspect the field before irrigation. "
            "If the soil surface is drying, "
            "consider light irrigation."
        )

    else:
        status = "Irrigation may be required"
        recommendation = (
            "Soil moisture is relatively low. "
            "Check the field immediately. "
            "If the crop is showing water stress, "
            "irrigation may be required."
        )

    # Higher temperature increases water demand,
    # but we do not make an exact water-volume claim
    # without crop stage and field-specific data.

    if temperature >= 35:
        extra_note = (
            "Temperature is high, so monitor the field "
            "more frequently for water stress."
        )
    else:
        extra_note = (
            "Continue monitoring soil moisture "
            "and weather conditions."
        )

    return {
        "success": True,
        "crop": data.crop,
        "soil": data.soil,
        "area_acres": data.area_acres,
        "soil_moisture": moisture,
        "rainfall": rainfall,
        "temperature": temperature,
        "growth_stage": data.growth_stage,
        "status": status,
        "recommendation": recommendation,
        "note": extra_note,
    }