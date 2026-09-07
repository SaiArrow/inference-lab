# service/Dockerfile
FROM python:3.11-slim AS base

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/app.py .
COPY models/onnx-int8/ models/onnx-int8/

# Cloud Run injects $PORT; default for local runs
ENV PORT=8080 MODEL_PATH=models/onnx-int8/model.onnx
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=40s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8080/healthz')"

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1