#!/bin/bash
cd "$(dirname "$0")"

echo "=== STEP 3: latency (batch=1, seq=128) ==="
python -c "
import pandas as pd
d = pd.read_csv('results/latency.csv')
s = d[(d.batch_size==1)&(d.seq_len==128)]
print(s[['variant','p50','p95','p99','throughput']].to_string(index=False))
" 2>/dev/null || echo "(results/latency.csv missing)"

echo; echo "=== STEP 3: throughput across batch sizes (seq=128) ==="
python -c "
import pandas as pd
d = pd.read_csv('results/latency.csv')
s = d[d.seq_len==128].pivot(index='variant', columns='batch_size', values='throughput')
print(s.round(1).to_string())
" 2>/dev/null || echo "(missing)"

echo; echo "=== STEP 4: accuracy ==="
cat results/accuracy.csv 2>/dev/null || echo "(results/accuracy.csv missing)"

echo; echo "=== MODEL SIZES ==="
du -h models/onnx/model.onnx models/onnx-int8/model.onnx models/torchscript/model.pt 2>/dev/null

summarize () {
  python -c "
import pandas as pd, sys
try:
    d = pd.read_csv('$1')
except Exception:
    print('  (missing)'); sys.exit()
d = d[d.Name=='Aggregated']
d = d[d['Requests/s'] > 0]
if d.empty:
    print('  (no data)'); sys.exit()
i = d['Requests/s'].idxmax()
print(f\"  peak RPS   : {d.loc[i,'Requests/s']:.1f}\")
print(f\"  p50 at peak: {d.loc[i,'50%']}\")
print(f\"  p95 at peak: {d.loc[i,'95%']}\")
print(f\"  p99 at peak: {d.loc[i,'99%']}\")
print(f\"  failures   : {d['Total Failure Count'].max()}\")
" 2>/dev/null
}

echo; echo "=== 8a: BATCHING A/B ==="
for m in 0 1; do
  echo "--- BATCHING=$m ---"; summarize "results/loadtest_batch${m}_stats_history.csv"
done

echo; echo "=== 8a: delay sweep (if run) ==="
for d in 1 5 20; do
  echo "--- MAX_DELAY_MS=$d ---"; summarize "results/loadtest_delay${d}_stats_history.csv"
done

echo; echo "=== STEP 7: Cloud Run ==="
for f in cloudrun 1inst 3inst; do
  echo "--- $f ---"; summarize "results/loadtest_${f}_stats_history.csv"
done

echo; echo "=== 8b: Kubernetes ==="
summarize "results/loadtest_k8s_stats_history.csv"

echo; echo "=== 8b: HPA scaling ==="
head -25 results/hpa_scaling.txt 2>/dev/null || echo "(missing)"

echo; echo "=== FILES PRESENT ==="
ls results/
