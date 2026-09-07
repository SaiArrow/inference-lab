# src/config.py
MODEL_ID = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LEN = 128
SEQ_LENGTHS = [32, 64, 128]
BATCH_SIZES = [1, 8, 32]
N_WARMUP = 20
N_RUNS = 200
EVAL_SAMPLES = 872           # full SST-2 validation split
LABELS = {0: "NEGATIVE", 1: "POSITIVE"}