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
tok = None
sess = None

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tok, sess
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
    sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    if BATCHING:
        await batcher.start()
    yield
    if BATCHING:
        await batcher.stop()


app = FastAPI(title="inference-lab", version="1.1.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())

class Req(BaseModel):
    texts: list[str]

# @app.on_event("startup")
# def load():
#     global tok, sess
#     tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
#     so = ort.SessionOptions()
#     so.intra_op_num_threads = int(os.getenv("ORT_THREADS", "2"))
#     sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model": MODEL_PATH}

@app.post("/predict")
async def predict(req: Req):
    if not req.texts:
        REQS.labels("ok").inc()
        return {"predictions": [], "latency_ms": 0.0}

    t0 = time.perf_counter()
    try:
        if BATCHING:
            out = list(await asyncio.gather(*[batcher.submit(t) for t in req.texts]))
        else:
            out = _infer_sync(req.texts)
        REQS.labels("ok").inc()
        return {"predictions": out, "latency_ms": (time.perf_counter() - t0) * 1000}
    except Exception:
        REQS.labels("error").inc()
        raise
    finally:
        LAT.observe(time.perf_counter() - t0)

import asyncio
from dataclasses import dataclass

MAX_BATCH = int(os.getenv("MAX_BATCH", "16"))
MAX_DELAY_MS = float(os.getenv("MAX_DELAY_MS", "5"))
BATCHING = os.getenv("BATCHING", "1") == "1"

BATCH_SIZE_HIST = Histogram(
    "inference_batch_size", "Assembled batch size",
    buckets=(1, 2, 4, 8, 16, 32),
)

@dataclass
class Item:
    text: str
    future: asyncio.Future


def _infer_sync(texts: list[str]):
    """One ONNX forward pass over a list of texts. Runs in a worker thread."""
    enc = tok(texts, return_tensors="np", padding="max_length",
              max_length=128, truncation=True)
    logits = sess.run(None, {
        "input_ids": enc["input_ids"].astype(np.int64),
        "attention_mask": enc["attention_mask"].astype(np.int64),
    })[0]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    return [{"label": LABELS[int(p.argmax())], "score": float(p.max())} for p in probs]

class Batcher:
    def __init__(self):
        self.queue: asyncio.Queue[Item] | None = None
        self.task: asyncio.Task | None = None

    async def start(self):
        self.queue = asyncio.Queue()          # bind to the running loop
        self.task = asyncio.create_task(self._worker())
        self.task.add_done_callback(self._on_worker_exit)

    async def stop(self):
        if self.task:
            self.task.cancel()
        self.task = None
        self.queue = None

    @staticmethod
    def _on_worker_exit(task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc:
            import traceback
            print("BATCHER WORKER DIED:", flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)

    async def submit(self, text: str, timeout: float = 30.0):
        fut = asyncio.get_running_loop().create_future()
        await self.queue.put(Item(text, fut))
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            raise RuntimeError("batcher timed out — worker may have died") from None


    async def _worker(self):
        loop = asyncio.get_running_loop()
        while True:
            item = await self.queue.get()
            batch = [item]
            deadline = time.perf_counter() + MAX_DELAY_MS / 1000

            while len(batch) < MAX_BATCH:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    batch.append(await asyncio.wait_for(self.queue.get(), remaining))
                except asyncio.TimeoutError:
                    break

            BATCH_SIZE_HIST.observe(len(batch))
            try:
                results = await loop.run_in_executor(
                    None, _infer_sync, [b.text for b in batch]
                )
                for b, r in zip(batch, results):
                    if not b.future.done():
                        b.future.set_result(r)
            except Exception as exc:
                for b in batch:
                    if not b.future.done():
                        b.future.set_exception(exc)
batcher = Batcher()

