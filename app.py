from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# -----------------------------
# Load Models
# -----------------------------
MODEL_DIR = Path("./NasaModels")

reg_model = joblib.load(MODEL_DIR / "weather_reg_model.pkl")
class_model = joblib.load(MODEL_DIR / "weather_class_model.pkl")
unique_weathers = joblib.load(MODEL_DIR / "unique_weathers.pkl")
le_city = joblib.load(MODEL_DIR / "city_encoder.pkl")

# -----------------------------
# FastAPI App
# -----------------------------
app = FastAPI(title="Weather Prediction API", version="1.2.0")

# -----------------------------
# Schemas
# -----------------------------
class WeatherRequest(BaseModel):
    city: str
    datetime: str
    lat: float
    lon: float

class WeatherResponse(BaseModel):
    date: str              # NEW FIELD
    temperature: float
    humidity: float
    wind_speed: float
    weather: str

class WeeklyForecastRequest(BaseModel):
    city: str
    start_date: str
    lat: float
    lon: float

class WeeklyForecastResponse(BaseModel):
    city: str
    predictions: list[WeatherResponse]

# -----------------------------
# Helpers
# -----------------------------
def make_features(city: str, dt: datetime, lat: float, lon: float) -> pd.DataFrame:
    city_encoded = le_city.transform([city])[0]
    return pd.DataFrame({
        'city': [city_encoded],
        'lat': [lat],
        'lon': [lon],
        'hour': [dt.hour],
        'day': [dt.day],
        'month': [dt.month],
        'year': [dt.year],
        'dayofweek': [dt.weekday()]
    })

def predict_once(city: str, dt: datetime, lat: float, lon: float) -> WeatherResponse:
    data = make_features(city, dt, lat, lon)

    # Regression prediction
    reg_pred = reg_model.predict(data)[0]
    temp, humidity, wind_speed = float(reg_pred[0]), float(reg_pred[1]), float(reg_pred[2])

    # Classification prediction
    class_pred = class_model.predict(data)[0]
    weather = unique_weathers[int(class_pred)]

    return WeatherResponse(
        date=dt.strftime("%Y-%m-%d"),   # inject date
        temperature=temp,
        humidity=humidity,
        wind_speed=wind_speed,
        weather=weather
    )

# -----------------------------
# Endpoints
# -----------------------------
@app.post("/predict", response_model=WeatherResponse)
def predict_weather(req: WeatherRequest):
    try:
        dt = pd.to_datetime(req.datetime)
        if req.city not in le_city.classes_:
            raise HTTPException(status_code=400, detail="Unsupported city.")

        return predict_once(req.city, dt, req.lat, req.lon)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_week", response_model=WeeklyForecastResponse)
def predict_week(req: WeeklyForecastRequest):
    try:
        start_dt = pd.to_datetime(req.start_date)
        if req.city not in le_city.classes_:
            raise HTTPException(status_code=400, detail="Unsupported city.")

        predictions = []
        for i in range(7):
            day_dt = start_dt + timedelta(days=i)
            forecast = predict_once(req.city, day_dt, req.lat, req.lon)
            predictions.append(forecast)

        return WeeklyForecastResponse(city=req.city, predictions=predictions)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=2000)

