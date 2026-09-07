FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/app.py .
COPY models/onnx-int8/ models/onnx-int8/

ENV PORT=8080 MODEL_PATH=models/onnx-int8/model.onnx ORT_THREADS=2
EXPOSE 8080

CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT} --workers 1