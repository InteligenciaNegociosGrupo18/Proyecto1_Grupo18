

def modelo(X_train,y_train,X_test,y_test,y_data,X_data,data_t):
    from networkx import display
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import pandas as pd
    import numpy as np
    from TextPreprocessor import TextPreprocessor
    import matplotlib.pyplot as plt
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
    plt.show()

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
    plt.show()


    from collections import Counter
    conteo = Counter(y_data)
    print(conteo)


    plt.bar(conteo.keys(), conteo.values(), color='skyblue')
    plt.title("Distribución de clases en el conjunto de entrenamiento")
    plt.xlabel("Clase ODS")
    plt.ylabel("Cantidad de ejemplos")
    plt.show()

    import math
    promedio_otras = round((conteo[3] + conteo[4]) / 2,0)

    faltan = max(0, promedio_otras - conteo[1])  # cantidad de textos a generar
    print(f"Queremos que la clase ODS 1 llegue a {promedio_otras} ejemplos.")
    print(f"Necesitamos generar aproximadamente {faltan} textos nuevos.")



    import os
    import random
    import json
    #import openai
    from openai import OpenAI


    # Configura tu API Key
    secret = ""
    #openai.api_key = secret

    import os
    path = os.path.join(os.path.dirname(__file__), "Datos_etapa 2.xlsx")
    df = pd.read_excel(path)
    #RUTA = "Datos_proyecto.xlsx"  
    TEXTO = "textos"                   
    ODS   = "labels"   
    #df = pd.read_excel(RUTA)

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
    Genera 200 opiniones ciudadanas breves (1–2 oraciones), en español de Colombia,
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
    
   #display(added_rows)           
    # print(added_rows.to_string(index=False))  

    # Entrenar de nuevo el modelo con los datos aumentados
    X_data = df["textos"]
    y_data = df["labels"]
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=0.3, random_state=42, stratify=y_data
    )
    pipeline.fit(X_train, y_train)
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

    # === Crear DataFrames a partir de los reportes ===
    df_orig = pd.DataFrame(report_aum).transpose()
    df_reent = pd.DataFrame(report_reentreno).transpose()

    # === Seleccionar solo las clases y métricas principales ===
    clases = [col for col in df_orig.index if col.isdigit()]  # etiquetas numéricas (ej: '1', '3', '4')

    # === Tabla global (macro averages) ===
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

    # === Tabla por clase ===
    tabla_clases = pd.DataFrame({
        'Clase': clases,
        'Precisión original': [df_orig.loc[c, 'precision'] for c in clases],
        'Precisión reentrenado': [df_reent.loc[c, 'precision'] for c in clases],
        'Recall original': [df_orig.loc[c, 'recall'] for c in clases],
        'Recall reentrenado': [df_reent.loc[c, 'recall'] for c in clases],
        'F1 original': [df_orig.loc[c, 'f1-score'] for c in clases],
        'F1 reentrenado': [df_reent.loc[c, 'f1-score'] for c in clases]
    }).round(3)

    # === Aplicar estilos visuales ===
    estilo_global = tabla_global.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#003366'),
                                    ('color', 'white'),
                                    ('text-align', 'center'),
                                    ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center'),
                                    ('padding', '6px')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f2f2f2')]}
    ]).set_caption("Comparación de métricas globales entre el modelo original y el reentrenado")

    estilo_clases = tabla_clases.style.set_table_styles([
        {'selector': 'th', 'props': [('background-color', '#004c99'),
                                    ('color', 'white'),
                                    ('text-align', 'center'),
                                    ('font-weight', 'bold')]},
        {'selector': 'td', 'props': [('text-align', 'center'),
                                    ('padding', '6px')]},
        {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]}
    ]).set_caption("Comparación por clase (ODS 1, ODS 3, ODS 4)")

    # === Mostrar ambas tablas ===
    display(estilo_global)
    display(estilo_clases)

