import os, requests, pandas as pd, streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import pandas as pd

API_BASE = os.getenv("API_BASE", "http://localhost:8000")
st.set_page_config(page_title="ODS Classifier", page_icon="✅", layout="centered")


ODS_NAMES = {
    "1": "ODS 1 — Fin de la pobreza",
    "3": "ODS 3 — Salud y bienestar",
    "4": "ODS 4 — Educación de calidad",
}

def _normalize_ods_code(x) -> str:
    """Recibe algo como 1, '1', 'ODS 1', '-1' y devuelve '1'|'3'|'4'."""
    s = str(x).strip().upper()
    s = s.replace("ODS", "").strip()
    # por si en algún pipeline quedó '-1' para ODS 1
    if s in {"-1", "1", "01"}:
        return "1"
    if s in {"3", "03"}:
        return "3"
    if s in {"4", "04"}:
        return "4"
    # fallback: deja lo que venga
    return s

def ods_full_name(x) -> str:
    code = _normalize_ods_code(x)
    return ODS_NAMES.get(code, str(x))


st.title("ODS Classifier – UNFPA")
st.caption("Predice ODS (1, 3, 4) para opiniones ciudadanas. Incluye probabilidad y reentrenamiento.")

# --- Tabs
tab_user, tab_expert = st.tabs(["Usuario", "Experto (retrain)"])

with tab_user:
    st.subheader("Clasificar textos")
    mode = st.radio("¿Cómo deseas ingresar?", ["Uno", "Varios (CSV)"], horizontal=True)

    if mode == "Uno":
        txt = st.text_area("Escribe el texto", height=120, placeholder="p.ej., Queremos más acceso a educación de calidad...")
        if st.button("Clasificar", type="primary"):
            if txt.strip():
                with st.spinner("Clasificando..."):
                    r = requests.post(f"{API_BASE}/predict", json={"instances":[{"textos":txt}]}, timeout=60)
                if r.ok:
                    data = r.json()
                    pred = data["predictions"][0]
                    st.success(f"ODS predicho: **{ods_full_name(pred)}**")
                    # ----- Probabilidades con nombres y gráfico formateado
                    if "probabilities" in data and data["probabilities"]:
                        raw = data["probabilities"][0]  # dict {clase: prob}
                        # Mapea claves a nombre completo
                        by_name = {ods_full_name(k): float(v) for k, v in raw.items()}

                        # Orden opcional: por prob desc
                        names = [k for k, _ in sorted(by_name.items(), key=lambda kv: kv[1], reverse=True)]
                        vals  = [by_name[n] for n in names]

                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.bar(names, vals)  # sin forzar colores
                        ax.set_title("Probabilidades por ODS (modelo actual)")
                        ax.set_xlabel("Objetivo de Desarrollo Sostenible")
                        ax.set_ylabel("Probabilidad")
                        plt.xticks(rotation=45, ha="right")
                        ax.set_ylim(0, 1)  # escala 0..1
                        st.pyplot(fig, clear_figure=True)

                else:
                    st.error(r.text)
            else:
                st.warning("Escribe un texto para clasificar.")
    else:
        st.write("Sube un CSV con columna **textos**.")
        file = st.file_uploader("CSV", type=["csv"])
        if file and st.button("Clasificar CSV", type="primary"):
            df = pd.read_csv(file)
            if "textos" not in df.columns:
                st.error("El CSV debe tener columna 'textos'.")
            else:
                rows = [{"textos": t} for t in df["textos"].astype(str).tolist()]
                with st.spinner("Clasificando..."):
                    r = requests.post(f"{API_BASE}/predict", json={"instances": rows}, timeout=120)
                if r.ok:
                    data = r.json()
                    df_out = df.copy()
                    #df_out["prediction"] = data["predictions"]
                    df_out["prediction"] = [ods_full_name(p) for p in data["predictions"]]
                    st.success("Clasificado")
                    st.dataframe(df_out.head(20))
                    st.download_button("Descargar resultados (.csv)",
                        df_out.to_csv(index=False).encode("utf-8"),
                        "predicciones.csv", "text/csv")
                else:
                    st.error(r.text)

with tab_expert:
    st.subheader("Reentrenar modelo")
    st.caption("Sube ejemplos **labeled** (columnas: textos, labels).")
    c1, c2 = st.columns([1,1])
    with c1:
        st.write("Carga CSV (textos,labels)")
        retrain_csv = st.file_uploader("CSV", type=["csv"], key="retrain_csv")
    with c2:
        st.write("O pega JSON válido")
        json_str = st.text_area("JSON", height=160,
            placeholder='{"instances":[{"textos":"...","labels":"ODS 3"}, ...]}')

    payload = None
    if retrain_csv:
        df = pd.read_csv(retrain_csv)
        if not {"textos","labels"}.issubset(df.columns):
            st.error("El CSV debe tener columnas 'textos' y 'labels'.")
        else:
            payload = {"instances": df[["textos","labels"]].to_dict(orient="records")}
    elif json_str.strip():
        try:
            import json
            payload = json.loads(json_str)
        except Exception as e:
            st.error(f"JSON inválido: {e}")

    if st.button("Ejecutar re-train", type="primary"):
        if not payload:
            st.warning("Debes cargar CSV o JSON válido.")
        else:
            with st.spinner("Reentrenando..."):
                r = requests.post(f"{API_BASE}/retrain", json=payload, timeout=300)
            if r.ok:
                st.success("Reentrenado correctamente")

                m = r.json().get("metrics", {})
                precision = float(m.get("precision", 0.0))
                recall    = float(m.get("recall", 0.0))
                f1        = float(m.get("f1_score", 0.0))
                samples   = int(m.get("samples", 0))

                st.subheader("Métricas (validación estratificada 70/30)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Precisión (macro)", f"{precision:.1%}")
                c2.metric("Recall (macro)",    f"{recall:.1%}")
                c3.metric("F1-score (macro)",  f"{f1:.1%}")
                st.caption(f"Muestras de validación: {samples:,}")

                # Tabla compacta
                dfm = pd.DataFrame({
                    "Métrica": ["Precisión (macro)", "Recall (macro)", "F1-score (macro)"],
                    "Valor": [precision, recall, f1]
                })
                dfm["Valor (%)"] = (dfm["Valor"]*100).round(2)
                st.dataframe(dfm[["Métrica","Valor (%)"]], hide_index=True, use_container_width=True)

                # Gráfica rápida de barras
                fig, ax = plt.subplots(figsize=(5.5, 3.2))
                ax.bar(dfm["Métrica"], dfm["Valor"])
                ax.set_ylim(0, 1)
                ax.set_title("Desempeño del modelo")
                ax.set_ylabel("Puntuación (0–1)")
                plt.xticks(rotation=15, ha="right")
                st.pyplot(fig, clear_figure=True)

                # Versión de modelo
                h = requests.get(f"{API_BASE}/health", timeout=30).json()
                meta = h.get("meta", {}) or {}
                ts   = meta.get("version_ts")
                path = meta.get("path", "")
                if ts:
                    dt = datetime.fromtimestamp(int(ts), tz=timezone.utc).astimezone()
                    st.info(f"**Versión del modelo:** {dt:%Y-%m-%d %H:%M}  ·  **Ubicación:** `{path}`")
                else:
                    st.info("**Versión del modelo:** no disponible")
            else:
                st.error(r.text)
