# Data: Vision Transformer Quantization Study

Raw results behind [Vision Transformer Quantization: What FP16 and INT8 Actually Buy You](../index.md). They were recorded on June 20, 2025 on one NVIDIA GeForce RTX 4070 Ti SUPER (16 GB), using PyTorch with CUDA, Hugging Face Transformers, timm and bitsandbytes. The library versions weren't saved with the results.

## Files

| File | Contents |
|---|---|
| [`quantization_results.csv`](quantization_results.csv) | One row per model and precision: 16 models × 4 precisions = 64 rows. |
| [`comprehensive_quantization_study_1750457193.json`](comprehensive_quantization_study_1750457193.json) | The same 64 runs with extra metadata: Hugging Face model ID, input size, layer counts and processing time. The number in the file name is the run's Unix timestamp. |

## Columns in `quantization_results.csv`

| Column | Meaning |
|---|---|
| `model_name` | Short model name used throughout the study |
| `precision` | `fp32` (baseline), `fp16`, `int8_bitsandbytes` or `int4_nf4` |
| `architecture` | Model family label |
| `size_category` | Grouping used in the study: `foundation_transformer`, `self_supervised_2023`, `masked_autoencoder_2021`, `production_ready`, `edge_optimized` or `specialized_efficient` |
| `parameters` | Parameter count |
| `latency_ms` | Mean time per forward pass at batch size 1. Each configuration was timed over a short run of roughly two dozen passes; the JSON's `processing_time_sec` is about 25× the latency |
| `throughput_sps` | Images per second. Because the batch size is 1, this is 1000 ÷ `latency_ms` |
| `peak_memory_mb` | Peak GPU memory during inference |
| `model_size_mb` | Size of the weights at this precision (MiB) |
| `actual_quantization_method` | The method that actually ran. Every `int4_nf4` row says `bitsandbytes_int8_success` (see known issues) |
| `simulated_accuracy` | Placeholder, 0.85 in every row. Accuracy was **not** measured |
| `stability_score` | Placeholder, 0.95 in every row. Not a measurement |
| `speedup` | FP32 latency ÷ this row's latency (above 1 is faster) |
| `memory_reduction_pct` | Peak-memory reduction relative to FP32, in percent |

## Known issues

- **INT4 fell back to INT8.** Every `int4_nf4` configuration records `bitsandbytes_int8_success` as the method used, and has the same memory as the INT8 run. Treat those rows as a repeat of INT8, not as 4-bit results.
- **Accuracy and stability are placeholders.** Both are constants, so don't use them in any analysis.
- **Layer counts.** In the JSON, `quantization_info.quantized_layers` is 0 for the INT8 and INT4 runs even though their memory fell by up to 75%. The counter didn't recognize bitsandbytes layers.

## Loading the data

```python
import pandas as pd

df = pd.read_csv("quantization_results.csv")

fp16 = df[df.precision == "fp16"].sort_values("speedup", ascending=False)
print(fp16[["model_name", "latency_ms", "speedup", "memory_reduction_pct"]])

# Compare precisions side by side
print(df.pivot(index="model_name", columns="precision", values="latency_ms").round(2))
```
