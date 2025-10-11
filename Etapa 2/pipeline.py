import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from TextPreprocessor import TextPreprocessor  # ✅ Importar desde tu módulo
import os


def build_pipeline():
    """Construye el pipeline completo con TextPreprocessor + RandomForest"""
    pipeline = Pipeline([
        ('vectorizer', TextPreprocessor(max_features=5000, ngram_range=(1, 2))),
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        ))
    ])
    return pipeline


if __name__ == "__main__":
    # === Entrenamiento inicial ===
    path = os.path.join(os.path.dirname(__file__), "Datos_etapa 2.xlsx")
    data = pd.read_excel(path)
    data = data.dropna(subset=['textos', 'labels']).drop_duplicates()

    X = data['textos']
    y = data['labels']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    pipeline = build_pipeline()
    print("Entrenando modelo...")
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    print("\n=== Reporte de Clasificación ===")
    print(classification_report(y_test, preds))
    print("\n=== Matriz de Confusión ===")
    print(confusion_matrix(y_test, preds))

    # Guardar el modelo
    model_path = os.path.join(os.path.dirname(__file__), "pipeline_model.joblib")
    joblib.dump(pipeline, model_path)
    print(f"\nModelo guardado exitosamente en: {model_path}")
