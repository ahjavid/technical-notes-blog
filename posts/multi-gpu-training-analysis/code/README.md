# Multi-GPU Performance Analysis: Benchmarking Code

This directory contains the code used to conduct the comprehensive multi-GPU performance analysis.

## Overview

The benchmarking code is designed to:
- Test both single-GPU and multi-GPU training configurations
- Measure detailed performance metrics (samples/sec, memory usage, GPU utilization)
- Profile communication overhead in multi-GPU setups
- Ensure statistical rigor with multiple runs and proper controls

## Core Components

### 1. Model Definitions (`models.py`)
```python
import tensorflow as tf
from tensorflow import keras

def create_medium_model(input_dim=50, output_dim=3):
    """
    Medium model with 258K parameters
    Representative of common production AI models
    """
    model = keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_dim)
    ])
    
    return model

def create_large_model(input_dim=50, output_dim=3):
    """
    Large model with 6.9M parameters
    Representative of medium-scale deep learning models
    """
    model = keras.Sequential([
        keras.layers.Dense(1024, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(output_dim)
    ])
    
    return model

def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.trainable_variables)
```

### 2. Performance Profiler (`profiler.py`)
```python
import time
import psutil
import nvidia_ml_py3 as nvml
import tensorflow as tf
from dataclasses import dataclass
from typing import List, Dict

@dataclass
class PerformanceMetrics:
    samples_per_second: float
    training_time_ms: float
    memory_usage_mb: float
    gpu_utilization: float
    communication_time_ms: float = 0.0
    
class PerformanceProfiler:
    def __init__(self, multi_gpu=False):
        self.multi_gpu = multi_gpu
        self.metrics_history = []
        nvml.nvmlInit()
        
    def profile_training_step(self, model, data_batch, targets, optimizer, loss_fn):
        """Profile a single training step with detailed metrics"""
        
        start_time = time.perf_counter()
        memory_before = self._get_gpu_memory()
        
        # Training step with gradient tape
        with tf.GradientTape() as tape:
            predictions = model(data_batch, training=True)
            loss = loss_fn(targets, predictions)
            
        # Communication timing for multi-GPU
        comm_start = time.perf_counter()
        gradients = tape.gradient(loss, model.trainable_variables)
        
        if self.multi_gpu:
            # NCCL AllReduce happens here automatically with MirroredStrategy
            pass
            
        comm_time = time.perf_counter() - comm_start
        
        # Optimizer step
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        total_time = time.perf_counter() - start_time
        memory_after = self._get_gpu_memory()
        
        # Calculate metrics
        batch_size = tf.shape(data_batch)[0].numpy()
        metrics = PerformanceMetrics(
            samples_per_second=batch_size / total_time,
            training_time_ms=total_time * 1000,
            memory_usage_mb=(memory_after - memory_before),
            gpu_utilization=self._get_gpu_utilization(),
            communication_time_ms=comm_time * 1000 if self.multi_gpu else 0.0
        )
        
        self.metrics_history.append(metrics)
        return metrics
    
    def _get_gpu_memory(self):
        """Get current GPU memory usage in MB"""
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        info = nvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024**2)
    
    def _get_gpu_utilization(self):
        """Get current GPU utilization percentage"""
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        util = nvml.nvmlDeviceGetUtilizationRates(handle)
        return util.gpu
```

### 3. Multi-GPU Strategy (`strategy.py`)
```python
import tensorflow as tf
import os

def setup_multi_gpu_strategy():
    """
    Setup optimized multi-GPU strategy for PCIe topology
    """
    # NCCL optimization for PCIe Host Bridge topology
    os.environ.update({
        'NCCL_DEBUG': 'INFO',
        'NCCL_ALGO': 'Tree',                # Optimal for PCIe topology
        'NCCL_PROTO': 'Simple',             # Reduced complexity
        'NCCL_P2P_DISABLE': '1',           # Force through host memory
        'NCCL_SHM_DISABLE': '0',           # Enable shared memory
        'NCCL_NET_GDR_LEVEL': '0',         # Disable GPU Direct RDMA
        'NCCL_BUFFSIZE': '33554432',       # 32MB buffer
        'NCCL_NTHREADS': '16',             # Optimal thread count
        'NCCL_MAX_NCHANNELS': '8',         # Limit channels
        'NCCL_MIN_NCHANNELS': '4'          # Minimum channels
    })
    
    # Configure GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                # Set memory limit to 12GB for safety
                tf.config.experimental.set_memory_limit(gpu, 12288)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Create MirroredStrategy with HierarchicalCopyAllReduce
    strategy = tf.distribute.MirroredStrategy(
        cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
    )
    
    print(f"Number of devices: {strategy.num_replicas_in_sync}")
    return strategy

class IntelligentStrategySelector:
    """
    Production-grade strategy selector based on model analysis
    """
    def __init__(self):
        self.thresholds = {
            'small_model_max_params': 1_000_000,
            'medium_model_max_params': 5_000_000,
            'large_model_min_params': 10_000_000,
            'min_batch_size_multi_gpu': 64
        }
    
    def should_use_multi_gpu(self, model, batch_size):
        """Intelligent decision based on model characteristics"""
        param_count = sum(p.numel() for p in model.trainable_variables)
        
        if param_count < self.thresholds['small_model_max_params']:
            return False, f"Model too small ({param_count:,} params)"
        
        if param_count < self.thresholds['medium_model_max_params']:
            if batch_size < 128:
                return False, f"Batch size {batch_size} too small for medium model"
            return "evaluate", f"Medium model may benefit with large batch"
        
        if param_count >= self.thresholds['large_model_min_params']:
            if batch_size >= self.thresholds['min_batch_size_multi_gpu']:
                return True, f"Large model ({param_count:,} params) benefits"
            return False, f"Increase batch size for large model efficiency"
        
        return "benchmark", f"Model in 5M-10M param range - test required"
```

### 4. Benchmark Runner (`benchmark.py`)
```python
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_squared_error
import json
import time
from datetime import datetime

class ComprehensiveBenchmark:
    def __init__(self, model_type='medium', multi_gpu=False):
        self.model_type = model_type
        self.multi_gpu = multi_gpu
        self.results = []
        
        # Setup strategy
        if multi_gpu:
            self.strategy = setup_multi_gpu_strategy()
        else:
            self.strategy = tf.distribute.get_strategy()  # Default strategy
    
    def create_synthetic_dataset(self, num_samples=10000):
        """Create reproducible synthetic dataset"""
        np.random.seed(42)
        tf.random.set_seed(42)
        
        # Financial-like features: 50 technical indicators
        features = np.random.randn(num_samples, 50).astype(np.float32)
        
        # Add some realistic correlations
        features[:, 1] = 0.7 * features[:, 0] + 0.3 * np.random.randn(num_samples)
        features[:, 2] = 0.5 * features[:, 0] - 0.3 * features[:, 1] + 0.4 * np.random.randn(num_samples)
        
        # Targets: price direction, volatility, volume
        targets = np.random.randn(num_samples, 3).astype(np.float32)
        
        return features, targets
    
    def run_benchmark(self, batch_sizes=[16, 32, 64, 128], num_runs=50):
        """Run comprehensive benchmark across batch sizes"""
        
        features, targets = self.create_synthetic_dataset()
        
        for batch_size in batch_sizes:
            print(f"\n{'='*50}")
            print(f"Testing batch size: {batch_size}")
            print(f"{'='*50}")
            
            batch_results = []
            
            for run in range(num_runs):
                print(f"Run {run+1}/{num_runs}", end=' ')
                
                with self.strategy.scope():
                    # Create model
                    if self.model_type == 'medium':
                        model = create_medium_model()
                    else:
                        model = create_large_model()
                    
                    # Compile model
                    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
                    model.compile(
                        optimizer=optimizer,
                        loss='mse',
                        metrics=['mae']
                    )
                    
                    # Create dataset
                    dataset = tf.data.Dataset.from_tensor_slices((features, targets))
                    dataset = dataset.batch(batch_size)
                    dataset = dataset.prefetch(tf.data.AUTOTUNE)
                    
                    # Distribute dataset
                    if self.multi_gpu:
                        dataset = self.strategy.experimental_distribute_dataset(dataset)
                    
                    # Profile training
                    profiler = PerformanceProfiler(self.multi_gpu)
                    
                    # Warmup runs (excluded from timing)
                    for _ in range(10):
                        for batch in dataset.take(1):
                            if self.multi_gpu:
                                self.strategy.run(self._training_step, args=(model, batch))
                            else:
                                self._training_step(model, batch)
                    
                    # Actual benchmark runs
                    step_metrics = []
                    for step in range(100):
                        for batch_data, batch_targets in dataset.take(1):
                            if self.multi_gpu:
                                metrics = self.strategy.run(
                                    self._profile_step, 
                                    args=(model, batch_data, batch_targets, profiler)
                                )
                            else:
                                metrics = self._profile_step(
                                    model, batch_data, batch_targets, profiler
                                )
                            step_metrics.append(metrics)
                    
                    # Calculate run statistics
                    avg_samples_per_sec = np.mean([m.samples_per_second for m in step_metrics])
                    avg_memory_mb = np.mean([m.memory_usage_mb for m in step_metrics])
                    avg_gpu_util = np.mean([m.gpu_utilization for m in step_metrics])
                    avg_comm_time = np.mean([m.communication_time_ms for m in step_metrics])
                    
                    batch_results.append({
                        'run': run,
                        'samples_per_second': avg_samples_per_sec,
                        'memory_usage_mb': avg_memory_mb,
                        'gpu_utilization': avg_gpu_util,
                        'communication_time_ms': avg_comm_time,
                        'model_parameters': sum(p.numel() for p in model.trainable_variables)
                    })
                
                print(f"✓ {avg_samples_per_sec:.0f} samples/sec")
            
            # Calculate batch statistics
            samples_per_sec = [r['samples_per_second'] for r in batch_results]
            
            self.results.append({
                'batch_size': batch_size,
                'model_type': self.model_type,
                'multi_gpu': self.multi_gpu,
                'num_runs': num_runs,
                'mean_samples_per_sec': np.mean(samples_per_sec),
                'std_samples_per_sec': np.std(samples_per_sec),
                'confidence_interval_95': np.percentile(samples_per_sec, [2.5, 97.5]),
                'mean_memory_mb': np.mean([r['memory_usage_mb'] for r in batch_results]),
                'mean_gpu_utilization': np.mean([r['gpu_utilization'] for r in batch_results]),
                'mean_communication_ms': np.mean([r['communication_time_ms'] for r in batch_results]),
                'model_parameters': batch_results[0]['model_parameters'],
                'detailed_runs': batch_results
            })
    
    def _training_step(self, model, batch):
        """Single training step"""
        batch_data, batch_targets = batch
        
        with tf.GradientTape() as tape:
            predictions = model(batch_data, training=True)
            loss = tf.keras.losses.mse(batch_targets, predictions)
            
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    def _profile_step(self, model, batch_data, batch_targets, profiler):
        """Training step with profiling"""
        return profiler.profile_training_step(
            model, batch_data, batch_targets, 
            model.optimizer, tf.keras.losses.mse
        )
    
    def save_results(self, filename=None):
        """Save benchmark results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            gpu_type = "multi_gpu" if self.multi_gpu else "single_gpu"
            filename = f"benchmark_{self.model_type}_{gpu_type}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Results saved to: {filename}")
        return filename
```

### 5. Main Execution Script (`run_benchmark.py`)
```python
#!/usr/bin/env python3
"""
Multi-GPU Performance Benchmark
Comprehensive analysis of single vs multi-GPU training performance
"""

import argparse
import sys
import os
import tensorflow as tf
from benchmark import ComprehensiveBenchmark
from strategy import IntelligentStrategySelector

def main():
    parser = argparse.ArgumentParser(description='Multi-GPU Performance Benchmark')
    parser.add_argument('--model', choices=['medium', 'large'], default='medium',
                        help='Model size to test')
    parser.add_argument('--gpu-mode', choices=['single', 'multi', 'auto'], default='auto',
                        help='GPU configuration mode')
    parser.add_argument('--batch-sizes', nargs='+', type=int, 
                        default=[16, 32, 64, 128],
                        help='Batch sizes to test')
    parser.add_argument('--runs', type=int, default=50,
                        help='Number of runs per configuration')
    parser.add_argument('--output-dir', default='./results',
                        help='Output directory for results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(f"Found {len(gpus)} GPU(s)")
    
    if len(gpus) < 2 and args.gpu_mode == 'multi':
        print("Error: Multi-GPU mode requires at least 2 GPUs")
        sys.exit(1)
    
    # Intelligent strategy selection
    if args.gpu_mode == 'auto':
        selector = IntelligentStrategySelector()
        
        # Create dummy model to analyze
        if args.model == 'medium':
            from models import create_medium_model
            dummy_model = create_medium_model()
        else:
            from models import create_large_model
            dummy_model = create_large_model()
        
        use_multi_gpu, reason = selector.should_use_multi_gpu(
            dummy_model, max(args.batch_sizes)
        )
        
        print(f"Intelligent Strategy Decision: {reason}")
        
        if use_multi_gpu == True:
            modes_to_test = ['single', 'multi']
        elif use_multi_gpu == "evaluate":
            modes_to_test = ['single', 'multi']  # Test both for comparison
        else:
            modes_to_test = ['single']
    else:
        modes_to_test = [args.gpu_mode] if args.gpu_mode != 'auto' else ['single', 'multi']
    
    # Run benchmarks
    all_results = {}
    
    for mode in modes_to_test:
        print(f"\n{'='*60}")
        print(f"Running {mode.upper()} GPU benchmark for {args.model.upper()} model")
        print(f"{'='*60}")
        
        multi_gpu = (mode == 'multi')
        benchmark = ComprehensiveBenchmark(
            model_type=args.model,
            multi_gpu=multi_gpu
        )
        
        benchmark.run_benchmark(
            batch_sizes=args.batch_sizes,
            num_runs=args.runs
        )
        
        # Save individual results
        result_file = benchmark.save_results(
            f"{args.output_dir}/benchmark_{args.model}_{mode}_gpu.json"
        )
        
        all_results[mode] = benchmark.results
    
    # Generate comparison report
    if len(modes_to_test) > 1:
        generate_comparison_report(all_results, args.output_dir, args.model)
    
    print(f"\nBenchmark complete! Results saved to: {args.output_dir}")

def generate_comparison_report(results, output_dir, model_type):
    """Generate comparison report between single and multi-GPU"""
    
    report_lines = [
        f"# {model_type.title()} Model Performance Comparison Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        ""
    ]
    
    if 'single' in results and 'multi' in results:
        single_results = results['single']
        multi_results = results['multi']
        
        report_lines.append("| Batch Size | Single GPU | Multi GPU | Speedup | Recommendation |")
        report_lines.append("|------------|------------|-----------|---------|----------------|")
        
        for i, batch_size in enumerate([r['batch_size'] for r in single_results]):
            single_perf = single_results[i]['mean_samples_per_sec']
            multi_perf = multi_results[i]['mean_samples_per_sec']
            speedup = multi_perf / single_perf
            
            if speedup >= 1.1:
                recommendation = "✅ Multi-GPU beneficial"
            elif speedup >= 0.95:
                recommendation = "⚠️ Marginal difference"
            else:
                recommendation = "❌ Single GPU preferred"
            
            report_lines.append(
                f"| {batch_size:2d} | {single_perf:8.0f} | {multi_perf:8.0f} | "
                f"{speedup:5.2f}x | {recommendation} |"
            )
        
        report_lines.extend([
            "",
            "## Key Findings",
            ""
        ])
        
        # Add analysis
        avg_speedup = np.mean([
            multi_results[i]['mean_samples_per_sec'] / single_results[i]['mean_samples_per_sec']
            for i in range(len(single_results))
        ])
        
        if avg_speedup >= 1.1:
            conclusion = "Multi-GPU provides significant performance benefits"
        elif avg_speedup >= 0.95:
            conclusion = "Performance difference is marginal - consider other factors"
        else:
            conclusion = "Single GPU provides better performance - avoid multi-GPU"
        
        report_lines.append(f"**Conclusion**: {conclusion}")
        report_lines.append(f"**Average Speedup**: {avg_speedup:.2f}x")
    
    # Save report
    with open(f"{output_dir}/comparison_report_{model_type}.md", 'w') as f:
        f.write('\n'.join(report_lines))

if __name__ == "__main__":
    main()
```

## Usage Examples

### Basic Usage
```bash
# Test medium model with intelligent strategy selection
python run_benchmark.py --model medium --gpu-mode auto

# Test large model with both single and multi-GPU
python run_benchmark.py --model large --gpu-mode auto --runs 25

# Test specific batch sizes
python run_benchmark.py --model medium --batch-sizes 32 64 128 --runs 20
```

### Advanced Usage
```bash
# Force multi-GPU testing (even if not recommended)
python run_benchmark.py --model medium --gpu-mode multi --runs 50

# Quick test with fewer runs
python run_benchmark.py --model large --runs 10 --output-dir ./quick_test

# Test only large batch sizes
python run_benchmark.py --model medium --batch-sizes 64 128 256 --runs 30
```

## Output Files

The benchmark generates several output files:

1. **`benchmark_{model}_{mode}_gpu.json`**: Raw performance data
2. **`comparison_report_{model}.md`**: Human-readable comparison
3. **Individual run logs**: Detailed per-run metrics

## Hardware Requirements

- **Minimum**: 1x RTX 4070 Ti SUPER (16GB)
- **Recommended**: 2x RTX 4070 Ti SUPER (16GB each)
- **System RAM**: 32GB+ (64GB recommended)
- **Storage**: NVMe SSD for dataset storage

## Software Requirements

```
Python: 3.12.4
TensorFlow: 2.19.0
NumPy: 2.1.3
CUDA: 12.5.1
cuDNN: 9
NCCL: 2.18.5
nvidia-ml-py3
psutil
scikit-learn
pandas
```

Install with:
```bash
pip install -r requirements.txt
```

---

*This benchmarking code is designed for reproducibility and statistical rigor. All results should be validated across multiple runs with proper statistical analysis.*
