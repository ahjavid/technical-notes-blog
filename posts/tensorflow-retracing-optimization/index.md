---
title: "Eliminating tf.function Retracing in TensorFlow"
description: >-
  Retracing warnings from @tf.function can hide a several-fold slowdown. I benchmarked
  four common patterns on TensorFlow 2.19: three fixes made the code 1.9–3.9× faster,
  and one "optimization" made it 4× slower.
date: 2025-06-17
updated: 2026-09-27
tags: [TensorFlow, Performance]
metrics:
  - value: "1.9–3.9×"
    label: "faster after fixing three common retracing patterns"
  - value: "10 → 4"
    label: "graph traces across those three tests"
  - value: "0.23×"
    label: "the Monte Carlo rewrite made things slower"
key_findings:
  - "**Retracing is easy to trigger by accident.** Calling `model.predict()` inside `@tf.function`, feeding it different input shapes, or passing Python numbers that change between calls each made TensorFlow rebuild the graph."
  - "**Three fixes, three clear wins:** calling the model directly on a tensor (3.85× faster), declaring an `input_signature` (2.54×), and passing tensors instead of Python ints (1.88×). Traces fell from 10 to 4 across the three tests."
  - "**Not every graph-mode rewrite is faster.** A Monte Carlo loop rewritten with `tf.range` and `TensorArray` ran 4.3× slower. Measure before and after every change."
  - "**Memory barely moved.** Only the first fix changed memory noticeably (about 102 MB less); the others stayed within ±1 MB, apart from the Monte Carlo rewrite, which used 28 MB more."
  - "**Make retracing visible.** `fn.experimental_get_tracing_count()` turns a silent slowdown into something a unit test can catch."
resources:
  - title: Raw results (JSON)
    url: data/complete_results.json
    note: Timing, trace counts and memory for every test, plus model sizes and the TensorFlow version.
  - title: Summary table (CSV)
    url: data/performance_summary.csv
  - title: Charts
    url: images/
    note: All five figures generated from the results, including the ones not shown in this post.
---

You've optimized a model, it's accurate, it's deployed, and then the logs start filling with this:

```text
WARNING - 5 out of the last 13 calls to <function> triggered tf.function retracing.
Tracing is expensive and the excessive number of tracings could be due to ...
```

It's easy to dismiss because nothing breaks. But every retrace means TensorFlow is rebuilding and re-optimizing a graph it should have reused. In the production trading models where I first hit this, that turned millisecond predictions into tens of milliseconds. This post measures four common causes and what fixing them buys you.

## What retracing is

`@tf.function` turns a Python function into a TensorFlow graph by **tracing** it: running the Python code once with symbolic inputs and recording the operations. The graph is cached against the call's *signature*, and any call that doesn't match a cached signature triggers a new trace:

- **A new input shape or dtype.** Batch size 16, then 32, then 64 means three traces unless you say the batch dimension can vary.
- **A new Python value.** Python ints, floats, strings and lists are baked into the graph, so every distinct value becomes its own trace.
- **A new function object.** Defining a `@tf.function` inside a loop or a per-request handler creates a fresh cache every time.

Each trace re-runs your Python code, rebuilds the graph and optimizes it again. That's cheap once and expensive on every call.

> [!NOTE]
> Control flow over *tensors* is not the problem. AutoGraph converts `if` and `for` statements that depend on tensor values into graph operations (`tf.cond`, `tf.while_loop`) inside a single trace. Retracing comes from *Python* values that change between calls.

## How I measured

Each test runs the same short sequence of calls through an unoptimized and an optimized version of a function, then records the total wall time (including any tracing), how many times TensorFlow traced, and how much memory grew.

| | |
|---|---|
| **Framework** | TensorFlow 2.19.0, Python 3.12.4, CUDA 12.5.1 |
| **Hardware** | Workstation with 2 × NVIDIA RTX 4070 Ti SUPER |
| **Models** | Two small Keras models with 50 inputs and 3 outputs: 54,403 and 2,021,379 parameters (the same two architectures as in the [multi-GPU study](../multi-gpu-training-analysis/)) |
| **Runs** | One measured run per test (recorded 2025-06-17) |

## Results

| Pattern (before → after) | Traces | Time | Speedup | Memory change |
|---|---:|---:|---:|---:|
| `model.predict()` inside `@tf.function` → direct model call on a tensor | 3 → 2 | 166.6 → 43.3 ms | **3.85×** | −102.1 MB |
| Varying input shapes → `input_signature` | 4 → 1 | 73.5 → 28.9 ms | **2.54×** | +0.25 MB |
| Python `int` argument → tensor argument | 3 → 1 | 88.0 → 46.7 ms | **1.88×** | −0.09 MB |
| Python Monte Carlo loop → `tf.range` + `TensorArray` | 1 → 1 | 120.6 → 523.8 ms | **0.23×** | +28.5 MB |

![Bar charts of execution time and trace counts before and after each optimization](images/performance_comparison.png "**Figure 1.** Execution time (left) and number of traces (right) for each test, before (red) and after (green). The Monte Carlo rewrite is the only regression.")

The first three rows are the patterns that cause retracing, and fixing them paid off every time. The fourth is a cautionary tale, covered [below](#pattern-4-the-rewrite-that-backfired).

## Pattern 1: don't call `model.predict()` inside `@tf.function`

`model.predict()` is a high-level loop that batches its input, builds its own `tf.function` and converts the results back to NumPy. Wrapping it in another `tf.function` nests all of that machinery inside your trace.

```python
# Before: retraces, and wraps a loop inside a graph
@tf.function
def predict_with_retracing(model, X):
    return model.predict(X, verbose=0)

# After: call the model directly on a tensor
@tf.function(reduce_retracing=True)
def predict_optimized(X_tensor):
    return model(X_tensor, training=False)

X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)  # convert once, outside
result = predict_optimized(X_tensor)
```

**Result:** 166.6 → 43.3 ms (**3.85×**), traces 3 → 2, and about 102 MB less memory growth.

## Pattern 2: declare an `input_signature` when batch sizes vary

Without a signature, every new batch size is a new trace. A `TensorSpec` with `None` in the batch dimension tells TensorFlow that one graph fits every batch size.

```python
# Before: one trace per distinct shape
@tf.function
def predict_no_signature(X):
    return model(X, training=False)

predict_no_signature(tf.random.normal([16, 50]))  # trace 1
predict_no_signature(tf.random.normal([32, 50]))  # trace 2
predict_no_signature(tf.random.normal([64, 50]))  # trace 3

# After: one trace for any batch size
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 50], dtype=tf.float32)])
def predict_with_signature(X):
    return model(X, training=False)
```

**Result:** 73.5 → 28.9 ms (**2.54×**), traces 4 → 1.

If you can't pin the signature down, `@tf.function(reduce_retracing=True)` (available since TensorFlow 2.9) asks TensorFlow to generalize shapes by itself after it sees them change.

A separate profiling run shows the same effect across batch sizes. In the bottom-right panel of Figure 2, the retracing version pays 15–20 ms every time it meets a new batch size. The version with a signature pays for one trace on the first call, then handles every later batch size in under a millisecond.

![Four panels: system RAM over time, GPU memory over time, RAM change per measurement point, and execution time by batch size with and without retracing](images/memory_timeline.png "**Figure 2.** Memory and timing profile from a separate run. RAM (top left) jumps once when the model loads and then stays flat. The bottom-right panel shows per-call time by batch size: the retracing version pays for a new trace at every batch size, while the version with a signature only pays for the first.")

## Pattern 3: pass tensors, not Python numbers

A Python `int` is part of the trace key, so `num_steps=10`, `20` and `30` produce three graphs. A tensor argument, together with `tf.range` for the loop, produces one graph that works for any value.

```python
# Before: every new num_steps value retraces
@tf.function
def train_with_python_args(X, y, num_steps):
    for i in range(num_steps):       # Python loop, unrolled into the graph
        ...

# After: one graph for any number of steps
@tf.function
def train_with_tensor_args(X, y, num_steps):
    for i in tf.range(num_steps):    # becomes a tf.while_loop
        ...

train_with_tensor_args(X, y, tf.constant(10))
train_with_tensor_args(X, y, tf.constant(20))  # reuses the graph
```

**Result:** 88.0 → 46.7 ms (**1.88×**), traces 3 → 1.

## Pattern 4: the rewrite that backfired

The last test moved a Monte Carlo simulation (many noisy copies of the input, each run through the model) into a single graph-mode loop:

```python
@tf.function(reduce_retracing=True)
def monte_carlo_optimized(X_base, noise_factor, num_runs):
    results = tf.TensorArray(dtype=tf.float32, size=num_runs, dynamic_size=False)
    for i in tf.range(num_runs):
        noise = tf.random.normal(tf.shape(X_base), stddev=noise_factor)
        results = results.write(i, model(X_base + noise, training=False))
    return results.stack()
```

It traced once, as intended, and still ran **4.3× slower** (120.6 → 523.8 ms) while using 28.5 MB more memory. I haven't isolated the cause. The lesson is the reason this test stays in the post: a rewrite that looks more "graph-friendly" isn't automatically faster, so measure every change.

What I'd try next, untested here: replace the loop with a single batched call. Stack all `num_runs` noisy copies into one tensor, call the model once, and reshape the output. GPUs are much better at one large batch than at many small sequential calls.

## Catching retracing before it reaches production

**Count traces in tests.** Every `tf.function` knows how many times it has traced:

```python
@tf.function(input_signature=[tf.TensorSpec([None, 50], tf.float32)])
def predict(x):
    return model(x, training=False)

for batch in (tf.zeros([16, 50]), tf.zeros([32, 50]), tf.zeros([64, 50])):
    predict(batch)

assert predict.experimental_get_tracing_count() == 1, "predict() is retracing"
```

**Log when tracing happens.** Python code in the function body only runs while TensorFlow is tracing, so a plain `print` or `logging` call fires once per trace. (Use `tf.print` for output on every call.)

```python
@tf.function
def predict(x):
    print("Tracing predict() for", x.shape, x.dtype)  # runs only when tracing
    return model(x, training=False)
```

## A cached predictor for many models with the same shape

In the trading system many small models share an architecture, and creating a `tf.function` per model caused exactly the new-function-object retracing described above. The fix was one cached graph per architecture, with each model's weights swapped into a reference copy for the call:

```python
class OptimizedModelCache:
    def __init__(self):
        self.function_cache = {}
        self.reference_models = {}

    def get_optimized_predictor(self, model_type, input_shape, output_size):
        key = (model_type, tuple(input_shape), output_size)
        if key not in self.function_cache:
            ref_model = self._create_reference_model(model_type, input_shape, output_size)

            @tf.function(
                input_signature=[tf.TensorSpec(shape=[None, *input_shape[1:]], dtype=tf.float32)],
                reduce_retracing=True,
            )
            def optimized_predict(X_tensor):
                return ref_model(X_tensor, training=False)

            self.reference_models[key] = ref_model
            self.function_cache[key] = optimized_predict
        return self.function_cache[key], self.reference_models[key]

    def predict_with_model(self, actual_model, X_tensor, model_type, input_shape, output_size):
        predictor, ref_model = self.get_optimized_predictor(model_type, input_shape, output_size)
        original_weights = ref_model.get_weights()
        ref_model.set_weights(actual_model.get_weights())  # swap weights in
        try:
            return predictor(X_tensor)
        finally:
            ref_model.set_weights(original_weights)        # and back out
```

> [!WARNING]
> The trade-offs: `get_weights` and `set_weights` copy every weight through host memory on each call, so this only pays off for small models. The shared reference model also isn't thread-safe, so serialize access or keep one reference model per worker. For large models, keep one long-lived `tf.function` per model instead.

## What changed in production

These are observations from my trading system rather than benchmark measurements. After applying the patterns above, the retracing warnings (previously 5–13 retraces per prediction cycle) disappeared. Prediction latency dropped from 15–45 ms to 1.4–2.0 ms, and memory stopped growing by 2–3 MB per retracing cycle.

## Checklist

**Do**

- Call `model(x, training=False)` inside `tf.function`, not `model.predict()`.
- Convert NumPy arrays and lists to tensors with one consistent dtype before calling the function.
- Give functions that see varying batch sizes an `input_signature` with `None` in the batch dimension, or use `reduce_retracing=True`.
- Pass values that change between calls as tensors, and use `tf.range` for loops that should stay inside the graph.
- Create `tf.function` objects and `tf.Variable`s once (for example in `__init__`), never per call.
- Assert on `experimental_get_tracing_count()` in tests.

**Avoid**

- Python scalars, strings or lists as arguments that change from call to call.
- Mixing `float64` NumPy inputs with `float32` tensors, since every new dtype is a new signature.
- Assuming a graph-mode rewrite is faster without timing it.

**Retracing is fine** during warm-up, for a handful of genuinely different shapes or architectures, and while debugging.

## Limitations

- **Small models, short call sequences, one run per test.** The timings include tracing and are specific to this machine. Treat the speedups as the size of the effect, not as numbers to expect on your workload.
- **Peak memory didn't change.** Only the per-test memory growth differed. Peak process memory was 1.55–1.58 GB in every test, with or without the fixes.
- **The benchmark scripts aren't published.** They're part of a private trading codebase. This folder has the raw results and the charts, and the post includes the code for each pattern that was tested.

---

*Revision, September 2026:* all figures now come from the committed results in `data/`, which match the charts. Earlier versions quoted 6.18×, 2.97× and 3.85×, a 72.6% overall improvement and 45% lower peak memory, none of which the recorded data supports. The Monte Carlo regression is now reported rather than footnoted, the advice about tensor control flow was corrected, a limitations section was added, and a link to a repository that doesn't exist was removed.
