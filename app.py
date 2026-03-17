# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 17:00:52 2026

@author: marin
"""

from fastapi import FastAPI
import joblib
import pandas as pd

# Cargar el modelo
model = joblib.load("pipeline.joblib")

app = FastAPI()

@app.get("/")
def inicio():
    return {"mensaje": "API funcionando correctamente"}

@app.get("/predict")
def predict(
    Age: int,
    Gender: str,
    Occupation: str,
    Device_Type: str,
    Daily_Phone_Hours: float,
    Social_Media_Hours: float,
    Sleep_Hours: float,
    Stress_Level: int,
    App_Usage_Count: int,
    Caffeine_Intake_Cups: int,
    Weekend_Screen_Time_Hours: float
):
    
    data = {
        "Age": Age,
        "Gender": Gender,
        "Occupation": Occupation,
        "Device_Type": Device_Type,
        "Daily_Phone_Hours": Daily_Phone_Hours,
        "Social_Media_Hours": Social_Media_Hours,
        "Sleep_Hours": Sleep_Hours,
        "Stress_Level": Stress_Level,
        "App_Usage_Count": App_Usage_Count,
        "Caffeine_Intake_Cups": Caffeine_Intake_Cups,
        "Weekend_Screen_Time_Hours": Weekend_Screen_Time_Hours,
    }

    df = pd.DataFrame([data])

    prediction = model.predict(df)[0]

    return {"Productividad_predicha": float(prediction)}