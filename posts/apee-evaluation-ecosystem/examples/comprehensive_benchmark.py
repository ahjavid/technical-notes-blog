#!/usr/bin/env python3
"""
APEE Comprehensive Model Benchmark.

Phase 3: Proper evaluation following LLM benchmarking best practices:
- Multiple diverse task categories (13 categories)
- Multiple complexity levels (5 levels)
- Multiple runs per scenario for statistical significance
- Ground truth comparison where available
- Structured output validation
- Constraint checking
- Statistical analysis with confidence intervals
"""

import asyncio
import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from apee.benchmarks.datasets import BenchmarkDataset, TaskCategory, Complexity
from apee.benchmarks.runner import BenchmarkRunner, BenchmarkConfig
from apee.benchmarks.analyzer import BenchmarkAnalyzer
from apee.utils.logging import setup_logging, get_logger

logger = get_logger("benchmark")


def print_header(text: str):
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def print_section(text: str):
    """Print a section header."""
    print(f"\n{'-'*50}")
    print(f"  {text}")
    print(f"{'-'*50}")


async def run_comprehensive_benchmark():
    """Run the comprehensive APEE benchmark suite."""
    
    setup_logging()
    
    print_header("APEE COMPREHENSIVE MODEL BENCHMARK")
    print("Phase 3: Statistical Model Evaluation")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize dataset and show summary
    dataset = BenchmarkDataset()
    summary = dataset.summary()
    
    print_section("Benchmark Dataset Summary")
    print(f"Total Scenarios: {summary['total_scenarios']}")
    print("\nBy Category:")
    for cat, count in sorted(summary['by_category'].items()):
        print(f"  • {cat}: {count}")
    print("\nBy Complexity:")
    for comp, count in sorted(summary['by_complexity'].items()):
        print(f"  • {comp}: {count}")
    
    # Configure benchmark
    config = BenchmarkConfig(
        models=[
            "qwen2.5-coder:3b",
            "qwen3:4b",
            "gemma3:4b",
            "granite4:3b",
            "llama3.2:3b",
        ],
        runs_per_scenario=3,  # 3 runs for statistical significance
        categories=None,  # All categories
        complexities=None,  # All complexities
        scenario_limit=None,  # All scenarios
        temperature=0.7,
        max_tokens=1024,
        timeout_seconds=60.0,
        verbose=True,
        save_outputs=True,
    )
    
    total_evals = len(config.models) * summary['total_scenarios'] * config.runs_per_scenario
    print_section("Benchmark Configuration")
    print(f"Models to evaluate: {len(config.models)}")
    for m in config.models:
        print(f"  • {m}")
    print(f"Scenarios: {summary['total_scenarios']}")
    print(f"Runs per scenario: {config.runs_per_scenario}")
    print(f"Total evaluations: {total_evals}")
    print(f"Estimated time: {total_evals * 5 / 60:.1f} - {total_evals * 10 / 60:.1f} minutes")
    
    # Run benchmark
    print_header("RUNNING BENCHMARK")
    
    runner = BenchmarkRunner(dataset=dataset)
    result = await runner.run(config)
    
    # Analyze results
    print_header("BENCHMARK RESULTS")
    
    analyzer = BenchmarkAnalyzer(result)
    
    # Overall rankings
    print_section("Overall Rankings")
    
    print("\n🏆 QUALITY RANKING (by mean quality score)")
    print(f"{'Rank':<6} {'Model':<25} {'Quality':<12} {'±Std':<10} {'95% CI':<20}")
    print("-" * 75)
    for i, model in enumerate(result.quality_ranking, 1):
        stats = next(s for s in result.model_stats if s.model == model)
        analysis = analyzer.get_model_analysis(model)
        ci_str = f"[{analysis.ci_lower:.3f}, {analysis.ci_upper:.3f}]" if analysis else "N/A"
        print(f"{i:<6} {model:<25} {stats.mean_quality:<12.3f} {stats.std_quality:<10.3f} {ci_str:<20}")
    
    print("\n⚡ SPEED RANKING (by mean latency)")
    print(f"{'Rank':<6} {'Model':<25} {'Latency':<12} {'±Std':<10}")
    print("-" * 55)
    for i, model in enumerate(result.speed_ranking, 1):
        stats = next(s for s in result.model_stats if s.model == model)
        print(f"{i:<6} {model:<25} {stats.mean_latency_ms:<12.0f}ms {stats.std_latency_ms:<10.0f}ms")
    
    print("\n📊 EFFICIENCY RANKING (quality per second)")
    print(f"{'Rank':<6} {'Model':<25} {'Efficiency':<12}")
    print("-" * 45)
    for i, model in enumerate(result.efficiency_ranking, 1):
        stats = next(s for s in result.model_stats if s.model == model)
        eff = stats.mean_quality / max(stats.mean_latency_ms, 1) * 1000
        print(f"{i:<6} {model:<25} {eff:<12.4f}")
    
    # Per-category performance
    print_section("Performance by Category")
    
    # Build category comparison table
    categories = sorted(set(
        cat for s in result.model_stats for cat in s.category_scores.keys()
    ))
    
    print(f"\n{'Category':<25}", end="")
    for model in result.quality_ranking[:5]:  # Top 5 models
        short_name = model.split(':')[0][:10]
        print(f"{short_name:<12}", end="")
    print()
    print("-" * (25 + 12 * min(5, len(result.quality_ranking))))
    
    for cat in categories:
        print(f"{cat:<25}", end="")
        for model in result.quality_ranking[:5]:
            stats = next(s for s in result.model_stats if s.model == model)
            score = stats.category_scores.get(cat, 0)
            print(f"{score:<12.3f}", end="")
        print()
    
    # Per-complexity performance
    print_section("Performance by Complexity")
    
    complexities = sorted(set(
        comp for s in result.model_stats for comp in s.complexity_scores.keys()
    ))
    
    print(f"\n{'Complexity':<15}", end="")
    for model in result.quality_ranking[:5]:
        short_name = model.split(':')[0][:10]
        print(f"{short_name:<12}", end="")
    print()
    print("-" * (15 + 12 * min(5, len(result.quality_ranking))))
    
    for comp in complexities:
        print(f"{comp:<15}", end="")
        for model in result.quality_ranking[:5]:
            stats = next(s for s in result.model_stats if s.model == model)
            score = stats.complexity_scores.get(comp, 0)
            print(f"{score:<12.3f}", end="")
        print()
    
    # Head-to-head comparisons
    if len(result.model_stats) >= 2:
        print_section("Head-to-Head Comparisons")
        
        top_model = result.quality_ranking[0]
        for other_model in result.quality_ranking[1:3]:  # Compare top model with next 2
            comparison = analyzer.compare_models(top_model, other_model)
            if comparison:
                print(f"\n{top_model} vs {other_model}:")
                print(f"  Quality: {comparison.quality_winner} wins ({comparison.quality_diff:+.3f})")
                print(f"  Speed: {comparison.speed_winner} wins ({comparison.latency_diff_ms:+.0f}ms)")
                print(f"  Effect Size: {comparison.effect_size:.3f} ({'large' if abs(comparison.effect_size) > 0.8 else 'medium' if abs(comparison.effect_size) > 0.5 else 'small'})")
                print(f"  Statistically Significant: {'Yes' if comparison.is_significant else 'No'}")
    
    # Best/worst scenarios per model
    print_section("Notable Scenarios")
    
    for model in result.quality_ranking[:3]:  # Top 3 models
        print(f"\n{model}:")
        
        best = analyzer.get_best_scenarios(model, n=3)
        print("  Best scenarios:")
        for sid, score in best:
            print(f"    ✓ {sid}: {score:.3f}")
        
        worst = analyzer.get_worst_scenarios(model, n=3)
        print("  Needs improvement:")
        for sid, score in worst:
            print(f"    ✗ {sid}: {score:.3f}")
    
    # Summary statistics
    print_section("Summary Statistics")
    
    print(f"\nBenchmark Duration: {result.duration_seconds:.1f}s")
    print(f"Total Evaluations: {len(result.scenario_results)}")
    print(f"Successful Evaluations: {sum(1 for r in result.scenario_results if r.success)}")
    print(f"Failed Evaluations: {sum(1 for r in result.scenario_results if not r.success)}")
    
    # Save results
    output_dir = Path(__file__).parent.parent / "benchmark_results"
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed JSON report
    json_path = output_dir / f"benchmark_{timestamp}.json"
    report_data = {
        "config": config.model_dump(),
        "summary": {
            "duration_seconds": result.duration_seconds,
            "total_evaluations": len(result.scenario_results),
            "quality_ranking": result.quality_ranking,
            "speed_ranking": result.speed_ranking,
            "efficiency_ranking": result.efficiency_ranking,
        },
        "model_stats": [
            {
                "model": s.model,
                "total_scenarios": s.total_scenarios,
                "total_runs": s.total_runs,
                "success_rate": s.overall_success_rate,
                "mean_quality": s.mean_quality,
                "std_quality": s.std_quality,
                "mean_latency_ms": s.mean_latency_ms,
                "std_latency_ms": s.std_latency_ms,
                "mean_tokens": s.mean_tokens,
                "category_scores": s.category_scores,
                "complexity_scores": s.complexity_scores,
            }
            for s in result.model_stats
        ],
    }
    
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)
    print(f"\n📄 Detailed results saved to: {json_path}")
    
    # Save text report
    text_path = output_dir / f"benchmark_{timestamp}.txt"
    report_text = analyzer.generate_report()
    with open(text_path, "w") as f:
        f.write(report_text)
    print(f"📄 Text report saved to: {text_path}")
    
    print_header("BENCHMARK COMPLETE")
    
    # Print winner
    winner = result.quality_ranking[0]
    winner_stats = next(s for s in result.model_stats if s.model == winner)
    print(f"🏆 WINNER: {winner}")
    print(f"   Quality: {winner_stats.mean_quality:.3f}")
    print(f"   Success Rate: {winner_stats.overall_success_rate:.1%}")
    print(f"   Latency: {winner_stats.mean_latency_ms:.0f}ms")


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())
