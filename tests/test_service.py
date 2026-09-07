import os, sys, pathlib

REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(REPO / "service"))
os.environ.setdefault("MODEL_PATH", str(REPO / "models/onnx-int8/model.onnx"))
os.environ.setdefault("TOKENIZER_PATH", str(REPO / "tokenizer"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from fastapi.testclient import TestClient
from app import app


def test_healthz():
    with TestClient(app) as c:
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_shape():
    with TestClient(app) as c:
        r = c.post("/predict", json={"texts": ["great film", "awful film"]})
        assert r.status_code == 200
        preds = r.json()["predictions"]
        assert len(preds) == 2
        assert all(p["label"] in {"POSITIVE", "NEGATIVE"} for p in preds)
        assert all(0.0 <= p["score"] <= 1.0 for p in preds)


def test_predict_sentiment():
    with TestClient(app) as c:
        r = c.post("/predict", json={"texts": ["an absolute masterpiece"]})
        assert r.json()["predictions"][0]["label"] == "POSITIVE"


def test_empty_batch():
    with TestClient(app) as c:
        r = c.post("/predict", json={"texts": []})
        assert r.status_code == 200
        assert r.json()["predictions"] == []
