# --- Estágio de desenvolvimento ---
FROM python:3.11-slim as dev
WORKDIR /app
ENV PYTHONUNBUFFERED=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt uvicorn[standard]
# Código vem do volume bind


FROM python:3.11-slim as prod
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn
COPY . .
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "app:app"]