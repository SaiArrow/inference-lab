# src/benchmark.py
import time, json, statistics as stats
from pathlib import Path
import numpy as np, pandas as pd, torch
import onnxruntime as ort
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_ID, SEQ_LENGTHS, BATCH_SIZES, N_WARMUP, N_RUNS

tok = AutoTokenizer.from_pretrained(MODEL_ID)

def make_batch(batch_size, seq_len):
    text = ["the plot meanders but the performances hold it together"] * batch_size
    enc = tok(text, return_tensors="np", padding="max_length",
              max_length=seq_len, truncation=True)
    return enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)

def timed(fn, ids, mask):
    for _ in range(N_WARMUP):
        fn(ids, mask)
    lat = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        fn(ids, mask)
        lat.append((time.perf_counter() - t0) * 1000)   # ms
    lat.sort()
    return {
        "p50": lat[int(0.50 * len(lat))],
        "p95": lat[int(0.95 * len(lat))],
        "p99": lat[int(0.99 * len(lat))],
        "mean": stats.mean(lat),
    }

def runner_pytorch():
    m = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    def fn(ids, mask):
        with torch.no_grad():
            m(input_ids=torch.from_numpy(ids), attention_mask=torch.from_numpy(mask))
    return fn

def runner_torchscript():
    m = torch.jit.load("models/torchscript/model.pt").eval()
    def fn(ids, mask):
        with torch.no_grad():
            m(torch.from_numpy(ids), torch.from_numpy(mask))
    return fn

def runner_onnx(path):
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1        # pin threads or your numbers are noise
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    def fn(ids, mask):
        sess.run(None, {"input_ids": ids, "attention_mask": mask})
    return fn

VARIANTS = {
    "pytorch-eager":  runner_pytorch,
    "torchscript":    runner_torchscript,
    "onnx-fp32":      lambda: runner_onnx("models/onnx/model.onnx"),
    "onnx-int8":      lambda: runner_onnx("models/onnx-int8/model.onnx"),
}

def main():
    torch.set_num_threads(1)
    rows = []
    for name, build in VARIANTS.items():
        fn = build()
        for bs in BATCH_SIZES:
            for sl in SEQ_LENGTHS:
                ids, mask = make_batch(bs, sl)
                r = timed(fn, ids, mask)
                r.update(variant=name, batch_size=bs, seq_len=sl,
                         throughput=bs / (r["p50"] / 1000))
                rows.append(r)
                print(f"{name:15} bs={bs:<3} seq={sl:<4} "
                      f"p50={r['p50']:7.2f}ms p99={r['p99']:7.2f}ms")

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv("results/latency.csv", index=False)

if __name__ == "__main__":
    main()