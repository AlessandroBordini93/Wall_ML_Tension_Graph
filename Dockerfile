# Usa Python 3.10 (compatibile con openseespy 3.7.1.2)
FROM python:3.10-slim

# Per vedere log in tempo reale
ENV PYTHONUNBUFFERED=1

# Installa dipendenze di sistema richieste da OpenSeesPy
RUN apt-get update && apt-get install -y \
    build-essential \
    gfortran \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Imposta directory di lavoro
WORKDIR /app

# Copia requirements
COPY requirements.txt .

# Installa tutte le dipendenze
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copia il tuo server FastAPI
COPY main.py .

# Avvia FastAPI (Render rileva la porta automaticamente)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
