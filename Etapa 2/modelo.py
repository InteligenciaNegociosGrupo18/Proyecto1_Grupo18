def modelo(X_train,y_train,X_test,y_test,y_data,X_data,data_t):
    from imblearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import pandas as pd
    import numpy as np
    from TextPreprocessor import TextPreprocessor
    import matplotlib.pyplot as plt
    from imblearn.over_sampling import SMOTE
    #Definimos el pipeline que vamos a utilizar con Random Forest
    pipeline = Pipeline([
    ('vectorizer', TextPreprocessor(max_features=5000, ngram_range=(1,2))),
    ('model', RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        class_weight="balanced",
        random_state=42
    ))
])

    from sklearn.model_selection import RandomizedSearchCV
    param_grid = {
        "model__n_estimators": [100, 200, 500],
        "model__max_depth": [None, 10, 20, 40],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2"]
    }


    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_grid,
        n_iter=20,
        cv=3,
        scoring="f1_macro",
        n_jobs=1,
        verbose=2,
        random_state=42
    )

    search.fit(X_train, y_train)
    print("Mejores parámetros:", search.best_params_)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)

    #Reporte de métricas (precisión, recall, F1-score)
    print(classification_report(y_test, y_pred))

    #Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_data))
    disp.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig("Matriz_de_Confusion_Modelo_Original.png", dpi=300, bbox_inches="tight")

    pipeline.fit(X_train, y_train)
    import os
    path = os.path.join(os.path.dirname(__file__), "Datos_etapa 2.xlsx")
    datos_etapa2 = pd.read_excel(path)
    #datos_etapa2 = pd.read_excel("Datos_etapa 2.xlsx")

    X_new = datos_etapa2["textos"]
    y_new = datos_etapa2["labels"]

    y_pred_new = pipeline.predict(X_new)

    #Reporte
    report_aum=classification_report(y_new, y_pred_new, output_dict=True)
    print(classification_report(y_new, y_pred_new))

    #Matriz de confusión
    cm = confusion_matrix(y_new, y_pred_new)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_new))
    disp.plot(cmap="Blues", values_format="d")
    plt.tight_layout()
    plt.savefig("Matriz_de_Confusion_Datos_Etapa_2.png", dpi=300, bbox_inches="tight")




    import os
    import json
    #import openai
    from openai import OpenAI


    # Configura tu API Key
    secret = ""
    #openai.api_key = secret

    import os
    path_proyecto = os.path.join(os.path.dirname(__file__), "Datos_proyecto.xlsx")
    path_etapa2 = os.path.join(os.path.dirname(__file__), "Datos_etapa 2.xlsx")

    df_proyecto = pd.read_excel(path_proyecto)
    df_etapa2 = pd.read_excel(path_etapa2)

    df_proyecto.columns = df_proyecto.columns.str.strip().str.lower()
    df_etapa2.columns = df_etapa2.columns.str.strip().str.lower()

    df_proyecto = df_proyecto.rename(columns={"texto": "textos", "ods": "labels"})
    df_etapa2 = df_etapa2.rename(columns={"texto": "textos", "ods": "labels"})

    df = pd.concat([df_proyecto, df_etapa2], ignore_index=True).drop_duplicates(subset=["textos", "labels"])
    print(f"Dataset combinado: {len(df)} filas")

    #Primero, queremos observar la distribucion de clases actual de todo el conjunto 
    import matplotlib.pyplot as plt
    from collections import Counter
    import math

    # Contamos las instancias por clase
    conteo = Counter(df["labels"])
    print("Distribución de clases:", conteo)

    #Graficamos
    plt.figure(figsize=(6, 4))
    plt.bar(conteo.keys(), conteo.values(), color=['#66b3ff', '#99ff99', '#ffcc99'])
    plt.title("Distribución de clases en el conjunto de datos inicial", fontsize=13, fontweight='bold')
    plt.xlabel("Clase ODS", fontsize=11)
    plt.ylabel("Cantidad de ejemplos", fontsize=11)
    plt.xticks(sorted(conteo.keys()))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Distribucion_clases_inicial.png", dpi=300, bbox_inches="tight")

    #Calculamos un estimado de cuantos datos nuevos artificales serian razonables 
    promedio_otras = round((conteo[3] + conteo[4]) / 2, 0)
    faltan = max(0, promedio_otras - conteo[1])

    print(f"\nQueremos que la clase ODS 1 llegue a {conteo[1] + round(faltan/2)} ejemplos.")
    print(f"Necesitamos generar aproximadamente {round(faltan/2)} textos nuevos.")


    TEXTO = "textos"                   
    ODS   = "labels"   

    minoritaria = df[ODS].value_counts().idxmin()
    semillas = (
        df[df[ODS]==minoritaria][TEXTO]      # filtra solo filas de la clase minoritaria y se queda con la columna de texto
        .dropna()                          # quita textos vacíos/NaN
        .astype(str)                       # asegura que todo sea string
        .sample(                           # toma una muestra aleatoria
            min(8, sum(df[ODS]==minoritaria)),  # hasta 8 textos, pero nunca más de los que hay
            random_state=42                      # para que la muestra sea reproducible
        )
        .tolist()                          # lo convierte a lista de strings
    )
    ejemplos = "\n".join(f"- {s}" for s in semillas)

     # columnas clave
    cols = ["textos", "labels"]

    # 1) snapshot de lo que ya tenías
    df_before = df[cols].drop_duplicates().copy()
    print(minoritaria)
    print(len(df))

    # Prompt para generar datos sintéticos
    prompt = f"""
    Genera exactamente 236 opiniones ciudadanas breves (1–2 oraciones), en español de Colombia,
    realistas y respetuosas, sobre problemáticas locales mapeadas SOLO al ODS 1.
    Definición de cada ODS: ODS 1: Fin de la pobreza, ODS 3: Salud y Bienestar, ODS 4: Educación de calidad
    Requisitos:
    - TODAS deben corresponder al ODS 1.
    - Varía zonas (urbano/rural), actores e instituciones; evita datos personales.
    - Mantén neutralidad política y sin contenido sensible.
    - Entrega SOLO JSON válido: una lista de objetos con:
    "textos" (string) y "labels" (entero 1).

    Ejemplos de nuestro dataset (NO copiar, solo inspirarse):
    {ejemplos if ejemplos else '- (sin ejemplos de contexto)'}
    """
    client = OpenAI(api_key=secret)
    # Llamada a la API moderna de OpenAI
    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Eres un generador de datos sintéticos."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
)

    # Procesar la respuesta
    raw = response.choices[0].message.content
    # Permite que venga envuelto en bloques de markdown
    txt = raw.strip()
    if txt.startswith("```"):
        txt = txt.strip("`")
        if "\n" in txt:
            txt = txt.split("\n", 1)[1]

    data = json.loads(txt)  # debe ser una lista de dicts {"opinion":..., "ods":...}
    df_new = pd.DataFrame(data)

    df = pd.concat([df, df_new[["textos", "labels"]]], ignore_index=True)

    print("Filas totales ahora:", len(df))

    # 2) normaliza lo generado y concatena
    df_new = df_new.rename(columns={"opinion": "texto"})[cols].copy()
    df = pd.concat([df, df_new], ignore_index=True).drop_duplicates(subset=cols).reset_index(drop=True)

    # 3) filas que están en df (nuevo) pero no estaban antes
    added_rows = (
        df.merge(df_before, on=cols, how="left", indicator=True)
        .loc[lambda x: x["_merge"] == "left_only", cols]
    )

    print(f"Nuevas filas agregadas: {len(added_rows)}")

    #Graficamos la nueva distribucion de las clases frente a todos los datos
    conteo = Counter(df["labels"])
    print("Distribución de clases:", conteo)

    #Graficamos
    plt.figure(figsize=(6, 4))
    plt.bar(conteo.keys(), conteo.values(), color=['#66b3ff', '#99ff99', '#ffcc99'])
    plt.title("Distribución de clases en el conjunto de datos final", fontsize=13, fontweight='bold')
    plt.xlabel("Clase ODS", fontsize=11)
    plt.ylabel("Cantidad de ejemplos", fontsize=11)
    plt.xticks(sorted(conteo.keys()))
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Distribucion_clases_final.png", dpi=300, bbox_inches="tight")


    # Entrenar de nuevo el modelo con los datos aumentados
    X_data = df["textos"]
    y_data = df["labels"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    pipeline_augmented = Pipeline([
        ('vectorizer', TextPreprocessor(max_features=5000, ngram_range=(1,2))),
        ('smote', SMOTE(random_state=42, sampling_strategy='auto')),  # balancea solo clases minoritarias
        ('model', RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # Entrenamiento
    pipeline_augmented.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    report_reentreno=classification_report(y_test, y_pred, output_dict=True)
    print(classification_report(y_test, y_pred))

    from collections import Counter

    # Ver la nueva distribución
    conteo_actualizado = Counter(y_data)
    print(conteo_actualizado)

    from sklearn.metrics import classification_report
    import pandas as pd
    from IPython.display import display

    #Crear DataFrames a partir de los reportes
    df_orig = pd.DataFrame(report_aum).transpose()
    df_reent = pd.DataFrame(report_reentreno).transpose()

    #Seleccionar solo las clases y métricas principales
    clases = [col for col in df_orig.index if col.isdigit()]  

    #Tabla global (macro averages)
    tabla_global = pd.DataFrame({
        'Métrica': ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1-score (macro)'],
        'Modelo original': [
            report_aum['accuracy'],
            report_aum['macro avg']['precision'],
            report_aum['macro avg']['recall'],
            report_aum['macro avg']['f1-score']
        ],
        'Modelo reentrenado': [
            report_reentreno['accuracy'],
            report_reentreno['macro avg']['precision'],
            report_reentreno['macro avg']['recall'],
            report_reentreno['macro avg']['f1-score']
        ]
    }).round(3)

    #Tabla por clase
    tabla_clases = pd.DataFrame({
        'Clase': clases,
        'Precisión original': [df_orig.loc[c, 'precision'] for c in clases],
        'Precisión reentrenado': [df_reent.loc[c, 'precision'] for c in clases],
        'Recall original': [df_orig.loc[c, 'recall'] for c in clases],
        'Recall reentrenado': [df_reent.loc[c, 'recall'] for c in clases],
        'F1 original': [df_orig.loc[c, 'f1-score'] for c in clases],
        'F1 reentrenado': [df_reent.loc[c, 'f1-score'] for c in clases]
    }).round(3)


    mostrar_tabla_en_plt(tabla_global, "Comparación global")
    mostrar_tabla_en_plt(tabla_clases, "Comparación por clase (ODS 1, ODS 3, ODS 4)")

import matplotlib.pyplot as plt
from io import BytesIO
import pandas as pd

import matplotlib.pyplot as plt
import pandas as pd

def mostrar_tabla_en_plt(estilo, titulo="Tabla"):
    # Convertir el estilo HTML en DataFrame legible por pandas
    html = estilo.to_html()
    df_simple = pd.read_html(html)[0]

    # Crear figura y ejes
    filas, columnas = df_simple.shape
    fig, ax = plt.subplots(figsize=(min(12, columnas * 2), filas * 0.6 + 2))
    ax.axis('off')
    ax.set_title(titulo, fontsize=14, fontweight='bold', pad=15, color='#003366')

    # Crear la tabla en Matplotlib
    tabla = ax.table(
        cellText=df_simple.values,
        colLabels=df_simple.columns,
        cellLoc='center',
        loc='center'
    )

    # Ajustar estilos visuales
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(10)
    tabla.scale(1.2, 1.2)  # escala horizontal, vertical

    # Colorear encabezados de columnas
    for (row, col), cell in tabla.get_celld().items():
        if row == 0:  # fila de encabezado
            cell.set_facecolor('#004c99')
            cell.set_text_props(color='white', fontweight='bold')
        elif col == 0:  # columna de métricas o clase
            cell.set_facecolor('#d9d9d9')
            cell.set_text_props(color='black', fontweight='bold')
        else:
            cell.set_facecolor('white')

        cell.set_edgecolor('#b3b3b3')

    plt.tight_layout()
    plt.tight_layout()
    nombre_archivo = f"Tabla_Comparativa_{titulo.replace(' ', '_')}.png"
    plt.savefig(nombre_archivo, dpi=300, bbox_inches="tight")





