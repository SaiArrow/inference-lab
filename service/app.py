# service/app.py
import asyncio
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from transformers import AutoTokenizer

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_PATH = os.getenv("MODEL_PATH", "models/onnx-int8/model.onnx")
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "tokenizer")
ORT_THREADS = int(os.getenv("ORT_THREADS", "2"))

MAX_BATCH = int(os.getenv("MAX_BATCH", "16"))
MAX_DELAY_MS = float(os.getenv("MAX_DELAY_MS", "5"))
BATCHING = os.getenv("BATCHING", "1") == "1"

LABELS = {0: "NEGATIVE", 1: "POSITIVE"}

# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
REQS = Counter("predict_requests_total", "Prediction requests", ["status"])
LAT = Histogram(
    "predict_latency_seconds",
    "End-to-end predict latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)
BATCH_SIZE_HIST = Histogram(
    "inference_batch_size",
    "Assembled batch size",
    buckets=(1, 2, 4, 8, 16, 32),
)

# --------------------------------------------------------------------------
# Model globals, populated in lifespan
# --------------------------------------------------------------------------
tok = None
sess = None


def _infer_sync(texts: list[str]):
    """One ONNX forward pass over a list of texts. Runs in a worker thread."""
    enc = tok(
        texts,
        return_tensors="np",
        padding="max_length",
        max_length=128,
        truncation=True,
    )
    logits = sess.run(
        None,
        {
            "input_ids": enc["input_ids"].astype(np.int64),
            "attention_mask": enc["attention_mask"].astype(np.int64),
        },
    )[0]
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    return [{"label": LABELS[int(p.argmax())], "score": float(p.max())} for p in probs]


# --------------------------------------------------------------------------
# Dynamic batching
# --------------------------------------------------------------------------
@dataclass
class Item:
    text: str
    future: asyncio.Future


class Batcher:
    """Collects requests for up to MAX_DELAY_MS or MAX_BATCH items,
    then serves the whole group with a single forward pass."""

    def __init__(self):
        self.queue: asyncio.Queue[Item] | None = None
        self.task: asyncio.Task | None = None

    async def start(self):
        # Constructed here, not in __init__, so the queue binds to the
        # running loop. A module-level Queue breaks across test clients,
        # which each create their own loop.
        self.queue = asyncio.Queue()
        self.task = asyncio.create_task(self._worker())
        self.task.add_done_callback(self._on_worker_exit)

    async def stop(self):
        if self.task:
            self.task.cancel()
        self.task = None
        self.queue = None

    @staticmethod
    def _on_worker_exit(task: asyncio.Task):
        # create_task swallows exceptions until GC. Without this the worker
        # can die silently and every request hangs on an unresolved future.
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
            # Block until at least one request arrives, then collect
            # whatever else shows up before the deadline.
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
                # ORT releases the GIL during run(), so the executor thread
                # gives real parallelism. Calling _infer_sync directly here
                # would block the event loop, nothing would queue behind it,
                # and every batch would have size 1.
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


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global tok, sess
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    so = ort.SessionOptions()
    so.intra_op_num_threads = ORT_THREADS
    sess = ort.InferenceSession(MODEL_PATH, so, providers=["CPUExecutionProvider"])
    if BATCHING:
        await batcher.start()
    yield
    if BATCHING:
        await batcher.stop()


app = FastAPI(title="inference-lab", version="1.1.0", lifespan=lifespan)


class Req(BaseModel):
    texts: list[str]


@app.get("/health")
def healthz():
    return {"status": "ok", "model": MODEL_PATH, "batching": BATCHING}


# A plain route, not app.mount(). Mounting a WSGI app at /metrics shadows
# sibling GET routes in Starlette's matching order — that is what made
# /healthz return 404 while /predict (a POST) kept working.
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


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