---
title: "Multi-GPU Training: When Hardware Topology Matters"
description: >-
  On two RTX 4070 Ti SUPER cards that can only exchange data through the CPU's PCIe
  host bridge, data-parallel training made both test models 10–27% slower. What the
  measurements show, why it happens, and how to tell whether a second GPU will help you.
date: 2025-06-17
updated: 2026-09-27
tags: [Distributed Training, GPUs, TensorFlow, Performance]
metrics:
  - value: "0.73–0.90×"
    label: "two-GPU throughput vs. one GPU, across all 10 configurations"
  - value: "≤ 45%"
    label: "scaling efficiency (100% would be a perfect 2× speedup)"
  - value: "No P2P"
    label: "the GPUs can only talk through host memory"
key_findings:
  - "**A second GPU made training slower in every configuration I tested.** Two GPUs delivered 0.73–0.90× the throughput of one, for both a 258K- and a 6.9M-parameter model, at every batch size from 4 to 128."
  - "**The work being split was too small to be worth splitting.** Each training step spent far more time on fixed costs — framework overhead and synchronizing gradients through host memory — than on arithmetic, so halving the arithmetic couldn't pay for the extra synchronization."
  - "**Larger models narrow the gap.** The 6.9M model peaked at 0.90× (batch 32) versus 0.86× for the smaller one. Extrapolating, break-even on this hardware is somewhere around 10–20M parameters — an estimate, not a measurement."
  - "**Check the topology before buying a second card.** Run `nvidia-smi topo -m` and `nvidia-smi topo -p2p r`. If the GPUs can't talk to each other directly, try mixed precision, a faster input pipeline, or a single faster GPU first."
resources:
  - title: Technical appendix
    url: technical-appendix.md
    note: Full result tables with confidence intervals, model definitions and the complete software environment.
  - title: Benchmark code
    url: code/README.md
    note: The TensorFlow scripts used for the measurements (models, profiler, MirroredStrategy setup, runner).
---

Adding a second GPU feels like the obvious fix for slow training: split every batch in two and finish in half the time. I tested that assumption on a workstation with two RTX 4070 Ti SUPER cards and found the opposite. Every configuration trained **more slowly** on two GPUs than on one.

This post walks through the measurements, explains why the second card hurt instead of helped, and turns the result into a checklist you can use before spending money on multi-GPU hardware.

## The question

> At what point does the benefit of parallel computation outweigh the cost of communication between GPUs?

With data parallelism, each GPU runs the forward and backward pass on its share of the batch, then the GPUs **all-reduce** their gradients so every copy of the model applies the same update. The split saves compute time; the all-reduce costs communication time. Whether you win depends on which of the two is bigger, and that depends heavily on how the GPUs are connected.

## Setup

| | |
|---|---|
| **GPUs** | 2 × NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB each (capped at 12 GB per GPU) |
| **Interconnect** | PCIe 4.0 ×16 slots behind the CPU's PCIe host bridge. No NVLink and no peer-to-peer access |
| **Software** | TensorFlow 2.19, CUDA 12.5, cuDNN 9, Python 3.12 ([full environment](technical-appendix.md#reproducibility-information)) |
| **Strategy** | `tf.distribute.MirroredStrategy` with `HierarchicalCopyAllReduce` |
| **Models** | Two fully connected networks: a *medium* one (Dense 256→128→64, 258K parameters) and a *large* one (Dense 1024→1024→512→512→256, 6.9M parameters) |
| **Data** | Synthetic tabular data: 10,000 samples, 50 features, 3 regression targets |
| **Protocol** | 50 runs per configuration; 10 warm-up steps excluded, 100 steps measured; outliers with a modified z-score above 3.5 removed |

The topology is the detail that matters. Without NVLink or peer-to-peer (P2P) access, every byte one GPU sends to the other takes the long way round:

```text
GPU0 ──PCIe──▶ host bridge / system memory ──PCIe──▶ GPU1
```

`nvidia-smi topo -m` reports this kind of link as `PHB` (the path crosses a PCIe host bridge, typically the CPU). It works, but every gradient exchange pays for a round trip through the CPU.

## Results

Throughput is in training samples per second, as the mean ± 95% confidence interval over 50 runs. *Speedup* is two-GPU throughput divided by one-GPU throughput, and *efficiency* is speedup divided by two, so 100% would mean perfect scaling.

### Medium model (258K parameters)

| Batch size | 1 GPU | 2 GPUs | Speedup | Efficiency | GPU utilization, 1 → 2 GPUs |
|---:|---:|---:|---:|---:|---:|
| 8 | 1,234 ± 45 | 1,012 ± 38 | 0.82× | 41% | 82% → 68% |
| 16 | 2,422 ± 67 | 2,039 ± 54 | 0.84× | 42% | 85% → 70% |
| 32 | 4,156 ± 89 | 3,567 ± 76 | 0.86× | 43% | 90% → 75% |
| 64 | 8,234 ± 145 | 6,789 ± 123 | 0.82× | 41% | 92% → 78% |
| 128 | 16,883 ± 234 | 12,345 ± 198 | 0.73× | 37% | 95% → 82% |

### Large model (6.9M parameters)

| Batch size | 1 GPU | 2 GPUs | Speedup | Efficiency | Step time, 1 → 2 GPUs |
|---:|---:|---:|---:|---:|---:|
| 4 | 234 ± 12 | 189 ± 9 | 0.81× | 40% | 17.1 → 21.2 ms |
| 8 | 431 ± 18 | 336 ± 15 | 0.78× | 39% | 18.6 → 23.8 ms |
| 16 | 789 ± 29 | 678 ± 25 | 0.86× | 43% | 20.3 → 23.6 ms |
| 32 | 1,245 ± 41 | 1,123 ± 38 | 0.90× | 45% | 25.7 → 28.5 ms |
| 64 | 2,101 ± 67 | 1,841 ± 59 | 0.88× | 44% | 30.5 → 34.8 ms |

Three things stand out:

1. **No configuration came close to break-even.** The best case, the large model at batch 32, still lost 10% of its throughput.
2. **GPU utilization dropped by 13–15 points on two GPUs.** The cards spent that time waiting on each other rather than computing.
3. **The larger model scaled less badly.** Its efficiency rose from 39–40% at batch sizes 4–8 to 44–45% at 32–64, while the medium model's fell to 37% at batch 128. More compute per step means the synchronization cost is spread over more useful work.

## Why the second GPU hurt

### The arithmetic was never the bottleneck

A forward plus backward pass costs roughly six floating-point operations per parameter per sample. For the large model at batch 64 that's about 6 × 6.9M × 64 ≈ 2.6 GFLOP per step, which an RTX 4070 Ti SUPER can do in well under a millisecond. The measured step took **30.5 ms**.

Nearly all of the step time was fixed cost: launching kernels, running the framework, synchronizing host and device (including the per-step profiling in the benchmark harness). Splitting the batch across two GPUs halves only the part that was already negligible, and it adds a gradient all-reduce through host memory on every step. That trade can't pay off.

### Where the time goes

For the medium model at batch 64, per-step timing shows the trade directly:

```text
One GPU — 7.8 ms per step
  forward pass       3.2 ms  ████████
  backward pass      3.1 ms  ████████
  optimizer update   1.2 ms  ███
  other              0.3 ms  █

Two GPUs — 9.4 ms per step
  forward pass       1.8 ms  ████      (each GPU, half the batch)
  backward pass      1.7 ms  ████
  gradient exchange  5.2 ms  █████████████
  optimizer update   0.6 ms  ██
  other              0.1 ms  █
```

Splitting the batch saved 2.8 ms of forward and backward time. Exchanging gradients through the host bridge cost 5.2 ms. The effective host-bridge bandwidth measured during the study was about 12–15 GB/s, well below what NVLink offers. At this model size, though, the fixed per-exchange latency hurts more than the bandwidth does: 258K float32 gradients are only about 1 MB.

## When a second GPU does make sense

Parameter count is only a rough proxy for what matters: **how much compute each step does compared with how much it has to synchronize**. A convolutional or attention-heavy model with 5M parameters can do far more work per step than a 5M-parameter MLP. With that caveat, this is how I'd read the results for a PCIe host-bridge machine like this one:

| Model size | Recommendation on a host-bridge topology |
|---|---|
| Under 1M parameters | Stay on one GPU. Communication dominates. |
| 1M – 5M | Prefer one GPU. Expect a 15–25% loss on two. |
| 5M – 10M | Benchmark both. It depends on the architecture and batch size. |
| 10M – 50M | Multi-GPU becomes worth testing at batch size 64 or more. |
| Over 50M | Multi-GPU is likely to help, and NVLink helps much more. |

> [!IMPORTANT]
> Only the first and third rows are backed by measurements (258K and 6.9M parameters). The larger size bands are extrapolated from the trend, so treat them as a starting point for your own benchmark, not a result.

### A checklist before you add a GPU

- [ ] **Check the link between the cards.** `nvidia-smi topo -m` shows the path: `NV#` (NVLink) is best and `PIX`/`PXB` are good; `PHB`, `NODE` and `SYS` all route through the CPU. `nvidia-smi topo -p2p r` shows whether peer-to-peer reads are supported at all. Consumer cards may not support P2P even in a good slot layout.
- [ ] **Estimate the compute per step.** If it's small next to your measured step time, a second GPU won't help.
- [ ] **Confirm you can use a larger global batch** (64 or more here) without hurting convergence.
- [ ] **Make sure training time is really the bottleneck**, not data loading, evaluation or iteration speed.
- [ ] **Try the cheap wins first:** mixed precision, a prefetching input pipeline (`tf.data` with `prefetch`), and batch-size tuning. On a single GPU they carry none of the synchronization cost.

### What I saw on production models

Outside this benchmark I run small financial-forecasting models in production. They land in the same regime and behave the same way:

| Model type | Parameters | 1 GPU (samples/s) | On 2 GPUs |
|---|---:|---:|---|
| LSTM | 300K – 500K | 15,000 – 25,000 | 20–28% slower |
| GRU | 200K – 400K | 18,000 – 30,000 | 22–30% slower |
| Small Transformer | 1.5M – 3M | 8,000 – 15,000 | 15–20% slower |

All of them now train on a single GPU.

## If you do go multi-GPU in TensorFlow

This is the setup used for the benchmark. [`code/README.md`](code/README.md) has the full scripts.

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    # Cap each GPU at 12 GB. (TensorFlow does not allow combining a memory
    # limit with set_memory_growth on the same device.)
    tf.config.set_logical_device_configuration(
        gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=12288)]
    )

strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
)
print("Replicas:", strategy.num_replicas_in_sync)

with strategy.scope():
    model = build_model()  # create and compile inside the scope
    model.compile(optimizer="adam", loss="mse")
```

> [!NOTE]
> The benchmark environment also set NCCL variables (`NCCL_ALGO=Tree`, `NCCL_PROTO=Simple`, `NCCL_P2P_DISABLE=1`, and others listed in the [appendix](technical-appendix.md#nccl-configuration)). Those only take effect when the strategy uses NCCL for all-reduce. That's `tf.distribute.NcclAllReduce`, MirroredStrategy's default, not the `HierarchicalCopyAllReduce` used here. Comparing the two on this hardware is the first follow-up experiment I'd run.

## Limitations

- **Two models, one architecture family.** Both are small MLPs on synthetic data. The results describe small, overhead-bound training steps, and the model-size thresholds above are extrapolations.
- **Fixed costs dominate the measured step time**, and they include the benchmark's own per-step profiling (NVML queries and host synchronization). A leaner training loop would shift the absolute numbers, though not the direction of the result.
- **Parameter counts.** With their default 50 input features, the model definitions in [`code/README.md`](code/README.md) produce 54K and 2.0M parameters rather than the 258K and 6.9M recorded for the benchmark runs. I haven't been able to reconcile the two. Either way both models are far below the ~10M range where two GPUs might start to pay off, so the conclusion holds.
- **One machine, one topology.** An NVLink system, or a board where the two slots share a PCIe switch, could behave very differently.

## What I'd test next

- `NcclAllReduce` vs. `HierarchicalCopyAllReduce` on the same hardware.
- A compute-heavy model, such as ResNet-50 on images, to find the real crossover point instead of extrapolating it.
- Mixed precision on one and two GPUs, which changes the compute-to-communication ratio.
- The same experiment on an NVLink-equipped pair of GPUs.

## The bottom line

More GPUs don't automatically mean faster training. On consumer hardware where the cards can only talk through the CPU, a second GPU made small-model training 10–27% slower in every configuration I measured. Measure your own workload, check the topology, and spend on the bottleneck you actually have.

---

*Revision, September 2026:* rewritten for clarity. Confidence intervals and a limitations section were added. A model-size table (small CNN through ViT-Large), a cost/ROI table and a per-operation communication breakdown that appeared on the earlier version of this page were removed, because the recorded measurements don't support them.
