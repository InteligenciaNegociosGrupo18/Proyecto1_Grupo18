FROM python:3.11-slim

# Evita prompts y acelera build
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Variables por defecto; se pueden overridear en el servicio
ENV MODEL_DIR=/data \
    MODEL_FILENAME=pipeline_model.joblib \
    NLTK_DATA=/app/nltk_data

WORKDIR /app

# Dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el código + datos necesarios
COPY . .

# Asegura carpeta de modelo y NLTK
RUN mkdir -p /data && mkdir -p /app/nltk_data


#RUN python -c "import nltk, os; d=os.environ.get('NLTK_DATA','/app/nltk_data'); nltk.download('stopwords', download_dir=d)"
RUN python -c "import nltk, os; d=os.environ.get('NLTK_DATA','/app/nltk_data'); \
[ nltk.download(x, download_dir=d) for x in ['stopwords','punkt','punkt_tab'] ]"
 
EXPOSE 8000

# Uvicorn en modo producción (sin --reload)
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
