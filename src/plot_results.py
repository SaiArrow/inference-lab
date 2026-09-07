# src/plot_results.py
import pandas as pd, matplotlib.pyplot as plt

lat = pd.read_csv("results/latency.csv")
sub = lat[(lat.batch_size == 1) & (lat.seq_len == 128)]

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].bar(sub.variant, sub.p50, color="#4C72B0")
ax[0].set_ylabel("p50 latency (ms)")
ax[0].set_title("Single-request latency, seq=128")
ax[0].tick_params(axis="x", rotation=20)

for bs in sorted(lat.batch_size.unique()):
    s = lat[(lat.batch_size == bs) & (lat.seq_len == 128)]
    ax[1].plot(s.variant, s.throughput, marker="o", label=f"batch={bs}")
ax[1].set_ylabel("throughput (samples/s)")
ax[1].set_title("Throughput by batch size")
ax[1].legend(); ax[1].tick_params(axis="x", rotation=20)

plt.tight_layout()
plt.savefig("results/latency_vs_accuracy.png", dpi=150)