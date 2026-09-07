# service/app.py
import os, time
import numpy as np, onnxruntime as ort
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer
from prometheus_client import Counter, Histogram, make_asgi_app

MODEL_PATH = os.getenv("MODEL_PATH", "models/onnx-int8/model.onnx")
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}

app = FastAPI(title="inference-lab", version="1.0.0")
app.mount("/metrics", make_asgi_app())

REQS = Counter("predict_requests_total", "Prediction requests", ["status"])
LAT = Histogram("predict_latency_seconds", "End-to-end predict latency",
                buckets=(.005, .01, .025, .05, .1, .25, .5, 1.0))

TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer")
tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
sess = None

class Req(BaseModel):
    texts: list[str]

@app.on_event("startup")
def load():
    global tok, sess
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
    sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict")
def predict(req: Req):
    t0 = time.perf_counter()
    try:
        enc = tok(req.texts, return_tensors="np", padding="max_length",
                  max_length=128, truncation=True)
        logits = sess.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0]
        e = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = e / e.sum(axis=-1, keepdims=True)
        out = [{"label": LABELS[int(p.argmax())], "score": float(p.max())} for p in probs]
        REQS.labels("ok").inc()
        return {"predictions": out, "latency_ms": (time.perf_counter() - t0) * 1000}
    except Exception:
        REQS.labels("error").inc()
        raise
    finally:
        LAT.observe(time.perf_counter() - t0)