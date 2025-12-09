#!/usr/bin/env python3
"""
APEE Command Line Interface.

Usage:
    python -m apee.cli evaluate --models qwen3:4b,gemma3:4b --tasks code,analysis
    python -m apee.cli benchmark --model qwen2.5-coder:3b
    python -m apee.cli list-models
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from apee.models import Task, AgentRole
from apee.agents.ollama import OllamaAgent, MODEL_POOL, OllamaClient
from apee.coordination.coordinator import Coordinator
from apee.evaluation.quality import HeuristicScorer


# Predefined task templates
TASK_TEMPLATES = {
    "code": Task(
        task_id="code_gen",
        description="Write a Python function to implement binary search with type hints.",
        complexity=0.6,
    ),
    "analysis": Task(
        task_id="analysis",
        description="Compare SQL vs NoSQL databases. List 3 key differences and use cases.",
        complexity=0.5,
    ),
    "review": Task(
        task_id="code_review",
        description="Review this code for issues: `def div(a, b): return a / b`",
        complexity=0.4,
    ),
    "synthesis": Task(
        task_id="synthesis",
        description="Summarize the SOLID principles in software design.",
        complexity=0.5,
    ),
    "reasoning": Task(
        task_id="reasoning",
        description="Explain why immutability is important in functional programming.",
        complexity=0.6,
    ),
}


async def check_ollama() -> bool:
    """Check if Ollama is running."""
    client = OllamaClient()
    return await client.check_health()


async def list_models_cmd():
    """List available and configured models."""
    print("\n📦 APEE Model Pool:")
    print("-" * 60)
    
    client = OllamaClient()
    if not await client.check_health():
        print("⚠️  Ollama is not running. Cannot check availability.")
        for model, info in MODEL_POOL.items():
            print(f"  • {model}: {info['type']} ({info['size']})")
        return
    
    available = await client.list_models()
    
    for model, info in MODEL_POOL.items():
        status = "✓" if any(model in m for m in available) else "✗"
        strengths = ", ".join(info['strengths'])
        print(f"  {status} {model}: {info['type']} ({info['size']}) - {strengths}")
    
    print("\n📋 Other Ollama models:")
    for m in available:
        if not any(pool_model in m for pool_model in MODEL_POOL):
            print(f"  • {m}")


async def benchmark_cmd(model: str, iterations: int = 3):
    """Benchmark a single model."""
    print(f"\n🔬 Benchmarking: {model}")
    print("=" * 60)
    
    if not await check_ollama():
        print("❌ Ollama is not running")
        return
    
    agent = OllamaAgent("benchmark_agent", AgentRole.EXECUTOR, model=model)
    scorer = HeuristicScorer()
    
    results = {
        "latencies": [],
        "qualities": [],
        "tokens": [],
    }
    
    task = TASK_TEMPLATES["code"]
    
    print(f"\n📋 Task: {task.description[:60]}...")
    print(f"📊 Iterations: {iterations}\n")
    
    for i in range(iterations):
        print(f"  Run {i+1}/{iterations}...", end=" ", flush=True)
        result = await agent.execute(task)
        
        if result.success:
            score = scorer.score_sync(result, task)
            results["latencies"].append(result.latency_ms)
            results["qualities"].append(score.overall)
            results["tokens"].append(result.tokens_used)
            print(f"✓ {result.latency_ms:.0f}ms, quality: {score.overall:.2f}")
        else:
            print(f"✗ {result.error}")
    
    if results["latencies"]:
        print("\n" + "-" * 60)
        print("📊 BENCHMARK RESULTS")
        print("-" * 60)
        print(f"  Avg Latency: {sum(results['latencies'])/len(results['latencies']):.0f}ms")
        print(f"  Min Latency: {min(results['latencies']):.0f}ms")
        print(f"  Max Latency: {max(results['latencies']):.0f}ms")
        print(f"  Avg Quality: {sum(results['qualities'])/len(results['qualities']):.2f}")
        print(f"  Avg Tokens:  {sum(results['tokens'])/len(results['tokens']):.0f}")


async def evaluate_cmd(models: list[str], tasks: list[str]):
    """Run evaluation across models and tasks."""
    print("\n🚀 APEE Evaluation")
    print("=" * 60)
    
    if not await check_ollama():
        print("❌ Ollama is not running")
        return
    
    print(f"📦 Models: {', '.join(models)}")
    print(f"📋 Tasks: {', '.join(tasks)}")
    
    task_objs = [TASK_TEMPLATES[t] for t in tasks if t in TASK_TEMPLATES]
    scorer = HeuristicScorer()
    
    results_table = []
    
    for model in models:
        print(f"\n🤖 Testing: {model}")
        agent = OllamaAgent(f"agent_{model.split(':')[0]}", AgentRole.EXECUTOR, model=model)
        
        model_results = {"model": model, "tasks": {}}
        
        for task in task_objs:
            result = await agent.execute(task)
            if result.success:
                score = scorer.score_sync(result, task)
                model_results["tasks"][task.task_id] = {
                    "latency": result.latency_ms,
                    "quality": score.overall,
                    "tokens": result.tokens_used,
                }
                print(f"  ✓ {task.task_id}: {result.latency_ms:.0f}ms, quality: {score.overall:.2f}")
            else:
                print(f"  ✗ {task.task_id}: {result.error}")
        
        results_table.append(model_results)
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    print(f"\n{'Model':<25} {'Avg Quality':<12} {'Avg Latency':<12}")
    print("-" * 50)
    
    for mr in results_table:
        if mr["tasks"]:
            avg_q = sum(t["quality"] for t in mr["tasks"].values()) / len(mr["tasks"])
            avg_l = sum(t["latency"] for t in mr["tasks"].values()) / len(mr["tasks"])
            print(f"{mr['model']:<25} {avg_q:<12.2f} {avg_l:<12.0f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="APEE - Adaptive Poly-Agentic Evaluation Ecosystem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # list-models command
    subparsers.add_parser("list-models", help="List available models")
    
    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark a model")
    bench_parser.add_argument("--model", "-m", default="qwen2.5-coder:3b", help="Model to benchmark")
    bench_parser.add_argument("--iterations", "-n", type=int, default=3, help="Number of iterations")
    
    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run evaluation")
    eval_parser.add_argument(
        "--models", "-m", 
        default="qwen2.5-coder:3b,qwen3:4b,gemma3:4b",
        help="Comma-separated list of models"
    )
    eval_parser.add_argument(
        "--tasks", "-t",
        default="code,analysis,review",
        help="Comma-separated list of tasks (code, analysis, review, synthesis, reasoning)"
    )
    
    args = parser.parse_args()
    
    if args.command == "list-models":
        asyncio.run(list_models_cmd())
    elif args.command == "benchmark":
        asyncio.run(benchmark_cmd(args.model, args.iterations))
    elif args.command == "evaluate":
        models = [m.strip() for m in args.models.split(",")]
        tasks = [t.strip() for t in args.tasks.split(",")]
        asyncio.run(evaluate_cmd(models, tasks))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
