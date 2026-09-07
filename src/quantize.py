# src/quantize.py
from pathlib import Path
from onnxruntime.quantization import quantize_dynamic, QuantType
from onnxruntime.quantization.shape_inference import quant_pre_process

SRC = "models/onnx/model.onnx"
PREP = "models/onnx/model_prep.onnx"
DST = "models/onnx-int8/model.onnx"

def main():
    Path(DST).parent.mkdir(parents=True, exist_ok=True)

    # Skipping this pre-pass is the single most common cause of a quantized
    # transformer that is slower than the model it came from.
    quant_pre_process(
        SRC, PREP,
        skip_symbolic_shape=False,
        auto_merge=True,          # merge conflicting symbolic dims instead of failing
        guess_output_rank=True,   # infer rank where it can't be derived
    )

    quantize_dynamic(
        model_input=PREP,
        model_output=DST,
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )

    a = Path(SRC).stat().st_size / 1e6
    b = Path(DST).stat().st_size / 1e6
    print(f"fp32 {a:.1f} MB -> int8 {b:.1f} MB  ({a/b:.2f}x smaller)")

if __name__ == "__main__":
    main()


# fp32 268.0 MB -> int8 67.4 MB  (3.98x smaller)