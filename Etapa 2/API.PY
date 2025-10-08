from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score
from pipeline import build_pipeline  # importa tu pipeline.py
from datetime import datetime
import os


def load_model():
    if not os.path.exists("pipeline_model.joblib"):
        raise HTTPException(status_code=404, detail="No se encontró el modelo entrenado.")
    return joblib.load("pipeline_model.joblib")

def save_model(model):
    """Guarda el modelo actualizado y crea una versión con timestamp."""
    joblib.dump(model, "pipeline_model.joblib")  


class TextInput(BaseModel):
    textos: str

class LabeledInput(BaseModel):
    textos: str
    labels: str

class PredictRequest(BaseModel):
    instances: List[TextInput]

class RetrainRequest(BaseModel):
    instances: List[LabeledInput]

app = FastAPI(
    title="API Analítica de Textos – Etapa 2",
    description="Predicción y reentrenamiento del modelo de ODS.",
    version="1.0"
)

@app.get("/")
def root():
    return {"mensaje": "API de Analítica de Textos activa"}

@app.post("/predict/")
def predict(req: PredictRequest):
    try:
        model = load_model()
        df = pd.DataFrame([i.dict() for i in req.instances])
        preds = model.predict(df["textos"])
        return {"predicciones": preds.tolist(), "total": len(preds)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/retrain/")
def retrain(req: RetrainRequest):
    try:
        df_new = pd.DataFrame([i.dict() for i in req.instances])
        if not {"textos", "labels"}.issubset(df_new.columns):
            raise HTTPException(status_code=400, detail="Faltan columnas 'textos' y 'labels'.")

        # Reentrenar desde cero (full batch replacement)
        X = df_new["textos"]
        y = df_new["labels"]
        new_pipeline = build_pipeline()
        new_pipeline.fit(X, y)

        # Evaluar (macro-average)
        preds = new_pipeline.predict(X)
        precision = precision_score(y, preds, average="macro", zero_division=0)
        recall = recall_score(y, preds, average="macro", zero_division=0)
        f1 = f1_score(y, preds, average="macro", zero_division=0)

        # Guardar el modelo actualizado
        version = save_model(new_pipeline)

        return {
            "mensaje": "Modelo reentrenado exitosamente",
            "version": version,
            "muestras": len(df_new),
            "metricas": {
                "precision_macro": round(precision, 3),
                "recall_macro": round(recall, 3),
                "f1_macro": round(f1, 3)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al reentrenar: {str(e)}")