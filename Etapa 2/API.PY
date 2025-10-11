from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from pipeline import build_pipeline  
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipeline_model.joblib")
app = FastAPI(title="API Clasificación ODS", version="2.0")


class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(..., description="Lista de textos para predecir")

class PredictResponse(BaseModel):
    predictions: List[Any]

class RetrainRequest(BaseModel):
    instances: List[Dict[str, Any]] = Field(..., description="Lista con textos y labels (ODS)")

class RetrainResponse(BaseModel):
    metrics: Dict[str, float]

# Funciones auxiliares
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("No se encontró el modelo entrenado (pipeline_model.joblib).")
    return joblib.load(MODEL_PATH)

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def prepare_dataframe(instances: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(instances)
    if "textos" not in df.columns:
        raise ValueError("El campo 'textos' es obligatorio en el JSON de entrada.")
    return df

# ENDPOINTS
@app.get("/health")
def health():

    return {"status": "ok", "model_loaded": os.path.exists(MODEL_PATH)}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        model = load_model()
        df = prepare_dataframe(request.instances)
        preds = model.predict(df["textos"])
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/retrain", response_model=RetrainResponse)
def retrain(request: RetrainRequest):
    try:
        df_new = prepare_dataframe(request.instances)
        if "labels" not in df_new.columns:
            raise ValueError("Se requiere la columna 'labels' para reentrenar el modelo.")

        # 1️ Cargar datos históricos
        base_dir = os.path.dirname(__file__)
        path_proyecto = os.path.join(base_dir, "Datos_proyecto.xlsx")
        path_etapa2 = os.path.join(base_dir, "Datos_etapa 2.xlsx")

        data_frames = []

        for path in [path_proyecto, path_etapa2]:
            if os.path.exists(path):
                df_hist = pd.read_excel(path)
                df_hist.columns = df_hist.columns.str.strip().str.lower()
                df_hist = df_hist.rename(columns={"texto": "textos", "ods": "labels"})
                data_frames.append(df_hist[["textos", "labels"]])

        # Combinar históricos + nuevos datos del JSON
        if data_frames:
            df_all = pd.concat(data_frames + [df_new], ignore_index=True).dropna(subset=["textos", "labels"])
        else:
            df_all = df_new.copy()

        # Eliminar duplicados
        df_all = df_all.drop_duplicates(subset=["textos", "labels"]).reset_index(drop=True)
        print(f"Total de instancias tras combinar: {len(df_all)}")

        X = df_all["textos"]
        y = df_all["labels"]

        # 2️ Dividir datos (70/30 estratificado)
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )

        # 3️ Reentrenar el pipeline
        pipeline = build_pipeline()
        pipeline.fit(X_train, y_train)

        # Guardar modelo actualizado
        save_model(pipeline)

        # 4️ Evaluar métricas de validación
        y_pred = pipeline.predict(X_val)

        from sklearn.metrics import precision_score, recall_score, f1_score
        avg = "macro" if len(y.unique()) > 2 else "binary"
        precision = precision_score(y_val, y_pred, average=avg, zero_division=0)
        recall = recall_score(y_val, y_pred, average=avg, zero_division=0)
        f1 = f1_score(y_val, y_pred, average=avg, zero_division=0)

        metrics = {
            "precision": round(float(precision), 3),
            "recall": round(float(recall), 3),
            "f1_score": round(float(f1), 3),
            "samples": len(y_val)
        }

        print(f"Métricas de validación: {metrics}")
        return {"metrics": metrics}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
