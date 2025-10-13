from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from pipeline import build_pipeline  
import os
from fastapi.middleware.cors import CORSMiddleware
import logging, time
from fastapi import Request
import time, json


#MODEL_PATH = os.path.join(os.path.dirname(__file__), "pipeline_model.joblib")
#app = FastAPI(title="API Clasificación ODS", version="2.0")

MODEL_DIR = os.getenv("MODEL_DIR", os.path.dirname(__file__))  # ./ por defecto
MODEL_FILENAME = os.getenv("MODEL_FILENAME", "pipeline_model.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")


app = FastAPI(title="API Clasificación ODS", version="2.0")

# Normalización de etiquetas (acepta 1/3/4, "ODS 1"/"ods3", "-1", etc)
ODS_SET = {"1", "3", "4"}

def normalize_label(v) -> str:
    s = str(v).strip().upper()
    s = s.replace("ODS", "").strip()   
    s = s.lstrip("+")                   
    if s == "-1": s = "1"            
    if s in {"01","03","04"}: s = s[1:]
    if s not in ODS_SET:
        raise ValueError(f"Etiqueta inválida '{v}'. Use 1/3/4 u 'ODS 1/3/4'.")
    return s


# CORS: Facilita que el front pueda llamar a la API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

# Logging a archivo + consola
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler("logs_api.txt", encoding="utf-8"), logging.StreamHandler()]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration_ms = int((time.time()-start)*1000)
    logging.info(f"{request.method} {request.url.path} -> {response.status_code} [{duration_ms} ms]")
    return response



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

""" def prepare_dataframe(instances: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(instances)
    if "textos" not in df.columns:
        raise ValueError("El campo 'textos' es obligatorio en el JSON de entrada.")
    return df """

def prepare_dataframe(instances: List[Dict[str, Any]]) -> pd.DataFrame:
    if not isinstance(instances, list) or len(instances) == 0:
        raise ValueError("Debes enviar al menos 1 instancia.")
    if len(instances) > 2000:
        raise ValueError("Máximo 2000 instancias por solicitud.")
    df = pd.DataFrame(instances)
    if "textos" not in df.columns:
        raise ValueError("El campo 'textos' es obligatorio.")
    df["textos"] = df["textos"].astype(str).str.strip()
    if df["textos"].str.len().gt(10000).any():
        raise ValueError("Cada 'textos' debe tener ≤ 10,000 caracteres.")
    return df


# ENDPOINTS
""" @app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": os.path.exists(MODEL_PATH),
        "model_path": MODEL_PATH
    } """

def save_model(model):
    joblib.dump(model, MODEL_PATH)
    meta = {"version_ts": int(time.time()), "path": MODEL_PATH}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f)

def load_meta():
    if os.path.exists(META_PATH):
        return json.load(open(META_PATH, "r", encoding="utf-8"))
    return {}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": os.path.exists(MODEL_PATH),
        "meta": load_meta()
    }


os.makedirs(MODEL_DIR, exist_ok=True)  



""" @app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        model = load_model()
        df = prepare_dataframe(request.instances)
        preds = model.predict(df["textos"])
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) """

class PredictResponse(BaseModel):
    predictions: List[Any]
    probabilities: List[Dict[str, float]] | None = None

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    try:
        model = load_model()
        df = prepare_dataframe(request.instances)
        X = df["textos"]
        preds = model.predict(X)
        out = {"predictions": preds.tolist()}

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            classes = getattr(model, "classes_", None)
            if classes is not None:
                out["probabilities"] = [
                    {str(c): float(p[i]) for i, c in enumerate(classes)} for p in probs
                ]
        return out
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
            df_all["labels"] = df_all["labels"].apply(normalize_label)
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
