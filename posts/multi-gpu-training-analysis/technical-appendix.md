# Technical Appendix: Multi-GPU Training Study

This appendix holds the detailed measurements behind [Multi-GPU Training: When Hardware Topology Matters](index.md): full result tables with confidence intervals, model definitions, the communication profile and the software environment.

## Test Configuration

- **Hardware**: 2x NVIDIA GeForce RTX 4070 Ti SUPER (16GB each)
- **Memory Configuration**: 12GB per GPU (safety margin)
- **Connection**: PCIe Host Bridge topology (no P2P)
- **Software**: TensorFlow 2.19.0, CUDA 12.5.1, cuDNN 9, NCCL 2.18.5 (details under [Reproducibility Information](#reproducibility-information))
- **Strategy**: `tf.distribute.MirroredStrategy` with `HierarchicalCopyAllReduce`
- **Test Duration**: 120+ hours of comprehensive testing
- **Statistical Rigor**: 50 runs per configuration, 95% confidence intervals

## Model Specifications

### Medium Model (258,432 parameters)
```
Layer 1: Dense(256) + ReLU + Dropout(0.2)
Layer 2: Dense(128) + ReLU + Dropout(0.2)  
Layer 3: Dense(64) + ReLU + Dropout(0.2)
Output Layer: Dense(n_outputs)

Memory footprint: ~1MB weights, ~54MB total (with optimizer states)
```

> [!NOTE]
> The parameter counts in this section are the ones recorded for the benchmark runs. With the default `input_dim=50` and `output_dim=3` in the [benchmark code](code/README.md), these layer stacks give 54,403 (medium) and 2,021,379 (large) parameters. Both figures are well below the ~10M range discussed in the post, so the conclusions don't depend on which is correct.

### Large Model (6,885,376 parameters)
```
Layer 1: Dense(1024) + ReLU + Dropout(0.3)
Layer 2: Dense(1024) + ReLU + Dropout(0.3)
Layer 3: Dense(512) + ReLU + Dropout(0.3)
Layer 4: Dense(512) + ReLU + Dropout(0.3)
Layer 5: Dense(256) + ReLU + Dropout(0.3)
Output Layer: Dense(n_outputs)

Memory footprint: ~27MB weights, ~108MB total (with optimizer states)
```

## Performance Results

### Medium Model (258K params) - Detailed Results

| Batch Size | Single GPU (samples/sec) | Multi-GPU (samples/sec) | Speedup | Efficiency | Memory/GPU | GPU Util |
|------------|-------------------------|------------------------|---------|------------|------------|----------|
| 8          | 1,234 ± 45              | 1,012 ± 38             | 0.82x   | 41%        | 2.8GB      | 82% → 68% |
| 16         | 2,422 ± 67              | 2,039 ± 54             | 0.84x   | 42%        | 3.2GB      | 85% → 70% |
| 32         | 4,156 ± 89              | 3,567 ± 76             | 0.86x   | 43%        | 3.8GB      | 90% → 75% |
| 64         | 8,234 ± 145             | 6,789 ± 123            | 0.82x   | 41%        | 4.6GB      | 92% → 78% |
| 128        | 16,883 ± 234            | 12,345 ± 198           | 0.73x   | 37%        | 6.2GB      | 95% → 82% |

### Large Model (6.9M params) - Detailed Results

| Batch Size | Single GPU (samples/sec) | Multi-GPU (samples/sec) | Speedup | Efficiency | Memory/GPU | Time/Step |
|------------|-------------------------|------------------------|---------|------------|------------|-----------|
| 4          | 234 ± 12                | 189 ± 9                | 0.81x   | 40%        | 3.8GB      | 17.1ms → 21.2ms |
| 8          | 431 ± 18                | 336 ± 15               | 0.78x   | 39%        | 4.2GB      | 18.6ms → 23.8ms |
| 16         | 789 ± 29                | 678 ± 25               | 0.86x   | 43%        | 5.8GB      | 20.3ms → 23.6ms |
| 32         | 1,245 ± 41              | 1,123 ± 38             | 0.90x   | 45%        | 7.4GB      | 25.7ms → 28.5ms |
| 64         | 2,101 ± 67              | 1,841 ± 59             | 0.88x   | 44%        | 9.8GB      | 30.5ms → 34.8ms |

## Communication Overhead Analysis

### NCCL Configuration
```bash
NCCL_DEBUG=INFO
NCCL_ALGO=Tree
NCCL_PROTO=Simple
NCCL_P2P_DISABLE=1
NCCL_BUFFSIZE=33554432  # 32MB
NCCL_NTHREADS=16
NCCL_MAX_NCHANNELS=8
NCCL_MIN_NCHANNELS=4
```

### Communication Breakdown (per training step)

| Operation | Medium Model | Large Model | Description |
|-----------|--------------|-------------|-------------|
| Gradient Collection | 15-20ms | 25-35ms | Gathering gradients from forward/backward pass |
| AllReduce Operation | 25-35ms | 35-50ms | NCCL synchronization across GPUs |
| Gradient Broadcast | 10-15ms | 15-25ms | Distributing averaged gradients |
| Synchronization | 5-10ms | 8-15ms | GPU coordination and barriers |
| Buffer Management | 3-5ms | 5-10ms | NCCL buffer allocation/cleanup |
| **Total Overhead** | **58-85ms** | **88-135ms** | **Total communication cost** |

> [!WARNING]
> These per-operation ranges can't all be per-step costs: they add up to more than the measured end-to-end step time (for example 9.4 ms for the medium model at batch size 64, derived from the throughput tables above). Treat this breakdown as unverified until it is re-measured. The throughput tables are the reliable result.

### Computation vs Communication Timeline

**Medium Model (258K params, batch size 64):**
- Single GPU: 7.8ms total (3.2ms forward, 3.1ms backward, 1.2ms optimizer, 0.3ms overhead)
- Multi-GPU: 9.4ms total (3.5ms compute per GPU, 5.2ms communication, 0.7ms overhead)

**Large Model (6.9M params, batch size 32):**
- Single GPU: 25.7ms total (11.2ms forward, 10.8ms backward, 3.1ms optimizer, 0.6ms overhead)
- Multi-GPU: 28.5ms total (12.0ms compute per GPU, 7.8ms communication, 1.6ms optimizer, 0.3ms overhead). The listed phases add up to 21.7 ms, so about 6.8 ms of the measured step time is unaccounted for.

## Hardware Topology Impact

### P2P Capability Testing
```python
# P2P access test results
torch.cuda.can_device_access_peer(0, 1) = False
# Reason: PCIe Host Bridge topology prevents direct GPU communication

# Communication bandwidth measurement
PCIe bandwidth (GPU→CPU→GPU): ~12-15 GB/s effective
Theoretical NVLink bandwidth: ~50-112 GB/s (for comparison)
Communication efficiency loss: ~70-85% due to topology
```

### Memory Bandwidth Analysis
- **Single GPU**: Full 512 GB/s memory bandwidth utilized
- **Multi-GPU**: Effective bandwidth reduced to ~350-400 GB/s due to communication overhead
- **Bottleneck**: PCIe 4.0 x16 theoretical 32 GB/s, actual ~25 GB/s for bidirectional traffic

## Production Model Categories

### Financial Trading Models (Real Production Data)
1. **LSTM Models (300K-500K params)**
   - Single GPU: 15,000-25,000 samples/sec
   - Multi-GPU: 12,000-18,000 samples/sec (20-28% slower)
   - **Recommendation**: Single GPU only

2. **GRU Models (200K-400K params)**
   - Single GPU: 18,000-30,000 samples/sec
   - Multi-GPU: 14,000-21,000 samples/sec (22-30% slower)
   - **Recommendation**: Single GPU only

3. **Transformer Models (1.5M-3M params)**
   - Single GPU: 8,000-15,000 samples/sec
   - Multi-GPU: 6,800-12,000 samples/sec (15-20% slower)
   - **Recommendation**: Single GPU preferred

## Parameter Threshold Analysis

Only two model sizes were measured (258K and 6.9M parameters), so every row above 6.9M is an extrapolation and the confidence column is a subjective estimate, not a statistical result.

| Parameter Range | Multi-GPU Benefit | Confidence (subjective) | Sample Models |
|----------------|------------------|------------------|---------------|
| < 500K         | Never            | 99%              | Small NLP, basic neural nets |
| 500K - 1M      | Never            | 95%              | Text classification, simple CNNs |
| 1M - 5M        | Rarely (< 5%)    | 90%              | Medium transformers, ResNet-18 |
| 5M - 10M       | Sometimes (20%)  | 75%              | Large transformers, ResNet-34 |
| 10M - 25M      | Often (60%)      | 85%              | ResNet-50, medium vision models |
| 25M - 50M      | Usually (80%)    | 90%              | Large vision models, BERT-base |
| > 50M          | Always           | 95%              | ResNet-152, BERT-large, GPT models |

## Cost-Benefit Analysis

### Hardware Investment
- Single RTX 4070 Ti SUPER: $800
- Dual RTX 4070 Ti SUPER: $1,600 + motherboard/PSU upgrades (~$2,000 total)
- **Additional investment**: $1,200

### Performance ROI for Tested Models
- Medium model (258K): **Negative ROI** (20% performance loss)
- Large model (6.9M): **Negative ROI** (15% performance loss)
- Break-even model size: **~15-20M parameters** (estimated)

### Alternative Investments
These weren't benchmarked in this study, but they avoid multi-GPU synchronization entirely:
1. **Faster storage** (NVMe) if data loading is the bottleneck
2. **More RAM** for caching the dataset in memory
3. **A faster CPU** if preprocessing is the bottleneck
4. **A single higher-end GPU** instead of two mid-range cards

## Optimization Recommendations

### Try These Before Multi-GPU
1. **Mixed precision training**: small code change, uses Tensor Cores
2. **Input pipeline optimization**: `tf.data` with `prefetch` and parallel map
3. **Batch size tuning**: larger batches amortize per-step overhead
4. **Model architecture**: pruning or quantization where accuracy allows

### Multi-GPU Considerations
Only consider multi-GPU when:
- [ ] Model has >10M parameters
- [ ] Batch size ≥64 achievable
- [ ] NVLink-enabled hardware available
- [ ] Training time is primary bottleneck (not development/debugging)
- [ ] Budget allows for proper hardware infrastructure

## Reproducibility Information

### Statistical Methodology
- **Sample size**: 50 runs per configuration
- **Warmup**: 10 training steps (excluded)
- **Measurement**: 100 training steps (included)
- **Outlier removal**: Modified Z-score > 3.5
- **Confidence intervals**: 95% using t-distribution
- **Significance testing**: Two-tailed t-test for performance differences

### Hardware Validation
- **GPU matching**: Verified identical ASIC quality through stress testing
- **Thermal stability**: Maintained 65-70°C under load
- **Power stability**: <2% voltage ripple, dedicated PSU rails
- **Memory testing**: Full memory test passed on both GPUs

### Software Reproducibility
```python
# Exact environment setup
Python: 3.12.4
TensorFlow: 2.19.0
NumPy: 2.1.3
CUDA: 12.5.1
cuDNN: 9
NCCL: 2.18.5
Driver: 565.77
pandas: 2.0.3

# Random seed control
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

# Deterministic operations
tf.config.experimental.enable_op_determinism()
```

## Future Research Directions

1. **NVLink Comparison**: Test same models with NVLink-enabled hardware
2. **Framework Comparison**: PyTorch vs TensorFlow vs JAX performance
3. **Algorithm Variants**: Local SGD, gradient compression, async updates
4. **Heterogeneous Training**: Mixed GPU types and capabilities
5. **Cloud vs On-Premise**: Cost and performance analysis

## Questions

Open an issue on [GitHub](https://github.com/ahjavid/technical-notes-blog/issues) for questions about the methodology or the data.

---

*This analysis represents 120+ hours of rigorous testing and analysis. All methodology is designed for reproducibility across different hardware configurations.*
