# src/evaluate.py
import numpy as np, pandas as pd, onnxruntime as ort
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from config import MODEL_ID, MAX_LEN, EVAL_SAMPLES

tok = AutoTokenizer.from_pretrained(MODEL_ID)

def accuracy(model_path):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    ds = load_dataset("glue", "sst2", split=f"validation[:{EVAL_SAMPLES}]")
    correct = 0
    for i in range(0, len(ds), 32):
        chunk = ds[i:i + 32]
        enc = tok(chunk["sentence"], return_tensors="np", padding="max_length",
                  max_length=MAX_LEN, truncation=True)
        logits = sess.run(None, {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        })[0]
        correct += (logits.argmax(-1) == np.array(chunk["label"])).sum()
    return correct / len(ds)

if __name__ == "__main__":
    rows = [
        {"variant": "onnx-fp32", "accuracy": accuracy("models/onnx/model.onnx")},
        {"variant": "onnx-int8", "accuracy": accuracy("models/onnx-int8/model.onnx")},
    ]
    for r in rows:
        print(f"{r['variant']:12} acc={r['accuracy']:.4f}")
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/accuracy.csv", index=False)