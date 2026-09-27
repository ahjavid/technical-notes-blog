# Technical Appendix: Multi-GPU Training Study

This appendix holds the detailed measurements behind [Multi-GPU Training: When Hardware Topology Matters](index.md): full result tables with confidence intervals, the model definitions and the software environment.

## Test configuration

| | |
|---|---|
| **GPUs** | 2 × NVIDIA GeForce RTX 4070 Ti SUPER, 16 GB each, capped at 12 GB per GPU |
| **Interconnect** | PCIe host bridge; no peer-to-peer access between the GPUs |
| **Software** | Python 3.12.4, TensorFlow 2.19.0, NumPy 2.1.3, CUDA 12.5.1, cuDNN 9 |
| **Strategy** | `tf.distribute.MirroredStrategy` with `HierarchicalCopyAllReduce` |
| **Data** | Synthetic tabular data: 10,000 samples, 50 features, 3 regression targets |
| **Protocol** | 50 runs per configuration; 10 warm-up steps excluded and 100 steps measured per run |

## Models

Both are Keras `Sequential` networks with 50 inputs and 3 outputs, built by `create_medium_model` and `create_large_model` in the [benchmark code](code/README.md).

### Smaller model: 54,403 parameters

```text
Dense(256) + ReLU, Dropout(0.2)
Dense(128) + ReLU, Dropout(0.2)
Dense(64)  + ReLU, Dropout(0.2)
Dense(3)

Weights: 0.2 MB in float32 (0.8 MB including gradients and Adam state)
```

### Larger model: 2,021,379 parameters

```text
Dense(1024) + ReLU, Dropout(0.3)
Dense(1024) + ReLU, Dropout(0.3)
Dense(512)  + ReLU, Dropout(0.3)
Dense(512)  + ReLU, Dropout(0.3)
Dense(256)  + ReLU, Dropout(0.3)
Dense(3)

Weights: 7.7 MB in float32 (31 MB including gradients and Adam state)
```

## Results

Throughput is in samples per second, as the mean ± 95% confidence interval over 50 runs. Speedup is two-GPU throughput divided by one-GPU throughput, and efficiency is speedup divided by two. *Memory/GPU* is the recorded GPU memory in use per GPU. It's far larger than the models themselves (under 8 MB of weights) because it includes TensorFlow's runtime and memory pool.

### Smaller model (54K parameters)

| Batch size | 1 GPU | 2 GPUs | Speedup | Efficiency | Memory/GPU | GPU utilization, 1 → 2 GPUs |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1,234 ± 45 | 1,012 ± 38 | 0.82× | 41% | 2.8 GB | 82% → 68% |
| 16 | 2,422 ± 67 | 2,039 ± 54 | 0.84× | 42% | 3.2 GB | 85% → 70% |
| 32 | 4,156 ± 89 | 3,567 ± 76 | 0.86× | 43% | 3.8 GB | 90% → 75% |
| 64 | 8,234 ± 145 | 6,789 ± 123 | 0.82× | 41% | 4.6 GB | 92% → 78% |
| 128 | 16,883 ± 234 | 12,345 ± 198 | 0.73× | 37% | 6.2 GB | 95% → 82% |

### Larger model (2.0M parameters)

| Batch size | 1 GPU | 2 GPUs | Speedup | Efficiency | Memory/GPU | Step time, 1 → 2 GPUs |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 234 ± 12 | 189 ± 9 | 0.81× | 40% | 3.8 GB | 17.1 → 21.2 ms |
| 8 | 431 ± 18 | 336 ± 15 | 0.78× | 39% | 4.2 GB | 18.6 → 23.8 ms |
| 16 | 789 ± 29 | 678 ± 25 | 0.86× | 43% | 5.8 GB | 20.3 → 23.6 ms |
| 32 | 1,245 ± 41 | 1,123 ± 38 | 0.90× | 45% | 7.4 GB | 25.7 → 28.5 ms |
| 64 | 2,101 ± 67 | 1,841 ± 59 | 0.88× | 44% | 9.8 GB | 30.5 → 34.8 ms |

### Where the time goes

**Smaller model, batch size 64** (7.8 ms per step on one GPU, 9.4 ms on two):

- One GPU: 3.2 ms forward, 3.1 ms backward, 1.2 ms optimizer update, 0.3 ms other
- Two GPUs: 1.8 ms forward and 1.7 ms backward on each GPU, 5.2 ms exchanging gradients, 0.6 ms optimizer update, 0.1 ms other

**Larger model, batch size 32:** 25.7 ms per step on one GPU and 28.5 ms on two.

## Interconnect

- Peer-to-peer access between the two GPUs is not available: `torch.cuda.can_device_access_peer(0, 1)` returns `False`.
- Data moving between the GPUs goes through the CPU's host bridge and system memory, at an effective ~12–15 GB/s.
- `HierarchicalCopyAllReduce` doesn't use NCCL. The NCCL variables set in the benchmark environment (listed in the [benchmark code](code/README.md)) have no effect on these results.

## Production models (outside the benchmark)

Small financial-forecasting models from my trading system, observed in production:

| Model type | Parameters | 1 GPU (samples/s) | 2 GPUs (samples/s) | Change |
|---|---:|---:|---:|---|
| LSTM | 300K – 500K | 15,000 – 25,000 | 12,000 – 18,000 | 20–28% slower |
| GRU | 200K – 400K | 18,000 – 30,000 | 14,000 – 21,000 | 22–30% slower |
| Small Transformer | 1.5M – 3M | 8,000 – 15,000 | 6,800 – 12,000 | 15–20% slower |

## Model size and expected benefit

Only two sizes were measured. Every row above 2M parameters is an extrapolation for this kind of PCIe host-bridge setup, not a result.

| Parameters | Expected benefit from a second GPU | Basis |
|---|---|---|
| Under 1M | None | Measured: 54K model, 0.73–0.86× |
| 1M – 5M | Rarely | Measured: 2.0M model, 0.78–0.90× |
| 5M – 10M | Sometimes; benchmark it | Extrapolated |
| 10M – 50M | Often, at batch size 64 or more | Extrapolated |
| Over 50M | Usually; NVLink helps much more | Extrapolated |

## Cost

- One RTX 4070 Ti SUPER: about $800. Two cards plus the motherboard and power-supply upgrades to run them: about $2,000.
- Both tested models lost throughput on two GPUs (about 19% for the 54K model and 15% for the 2.0M model, averaged over batch sizes), so the second card had a negative return for this workload.
- Break-even on this hardware is estimated at 10M parameters or more. That is not measured.

These alternatives weren't benchmarked, but they avoid multi-GPU synchronization entirely:

1. **Faster storage** (NVMe) if data loading is the bottleneck
2. **More RAM** for caching the dataset in memory
3. **A faster CPU** if preprocessing is the bottleneck
4. **A single higher-end GPU** instead of two mid-range cards

## Try these before multi-GPU

1. **Mixed precision training**: a small code change that uses Tensor Cores
2. **Input pipeline optimization**: `tf.data` with `prefetch` and a parallel `map`
3. **Batch size tuning**: larger batches spread the per-step overhead over more samples
4. **Model architecture**: pruning or quantization where accuracy allows

Only consider multi-GPU when:

- [ ] Each training step does substantial compute (roughly 10M+ parameters for an MLP)
- [ ] A batch size of 64 or more works for your model
- [ ] The GPUs have NVLink or peer-to-peer access
- [ ] Training time is the real bottleneck, not development or debugging
- [ ] The budget covers the supporting hardware

## Reproducibility

### Statistics

- **Runs**: 50 per configuration
- **Warm-up**: 10 training steps, excluded
- **Measurement**: 100 training steps
- **Outliers**: runs with a modified z-score above 3.5 removed
- **Intervals**: 95% confidence intervals from the t-distribution

### Environment

```text
Python      3.12.4
TensorFlow  2.19.0
NumPy       2.1.3
CUDA        12.5.1
cuDNN       9
GPUs        2 × NVIDIA GeForce RTX 4070 Ti SUPER (16 GB)
```

```python
# Seeds and determinism used for every run
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)
tf.config.experimental.enable_op_determinism()
```

## Future work

1. **NVLink comparison**: the same models on GPUs that support NVLink (GeForce RTX 40-series cards don't)
2. **All-reduce comparison**: `NcclAllReduce` vs `HierarchicalCopyAllReduce`
3. **Compute-heavy models**: for example ResNet-50, to measure the real crossover point
4. **Framework comparison**: PyTorch vs TensorFlow vs JAX
5. **Algorithm variants**: local SGD, gradient compression, asynchronous updates

## Questions

Open an issue on [GitHub](https://github.com/ahjavid/technical-notes-blog/issues) for questions about the methodology or the data.
