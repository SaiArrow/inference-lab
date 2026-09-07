# src/export_onnx.py  (also handles TorchScript — rename if you prefer)
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from config import MODEL_ID, MAX_LEN

def export_torchscript(out="models/torchscript/model.pt"):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    sample = tok("a placeholder sentence for tracing",
                 return_tensors="pt", padding="max_length",
                 max_length=MAX_LEN, truncation=True)

    with torch.no_grad():
        traced = torch.jit.trace(
            model, (sample["input_ids"], sample["attention_mask"]), strict=False
        )
    traced = torch.jit.freeze(traced)
    traced.save(out)
    print(f"saved {out}")

if __name__ == "__main__":
    export_torchscript()