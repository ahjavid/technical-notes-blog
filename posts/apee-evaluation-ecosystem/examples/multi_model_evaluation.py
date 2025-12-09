#!/usr/bin/env python3
"""
Multi-Model APEE Evaluation Example.

This example demonstrates evaluating across different models:
- qwen2.5-coder:3b (coding tasks)
- qwen3:4b (reasoning/analysis)
- gemma3:4b (general/clarity)
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from apee.models import Task, AgentRole
from apee.agents.ollama import OllamaAgent, MODEL_POOL, get_recommended_model
from apee.coordination.coordinator import Coordinator
from apee.evaluation.quality import HeuristicScorer, CompositeScorer
from apee.utils.logging import setup_logging, get_logger

logger = get_logger("multi_model")


async def run_multi_model_evaluation():
    """Compare performance across different models."""
    
    setup_logging()
    
    print("\n" + "="*70)
    print("APEE: Multi-Model Comparison Evaluation")
    print("="*70 + "\n")
    
    # Show available models
    print("📦 Model Pool:")
    for model, info in MODEL_POOL.items():
        print(f"  • {model}: {info['type']} ({info['size']}) - {', '.join(info['strengths'])}")
    
    # Test models
    test_models = ["qwen2.5-coder:3b", "qwen3:4b", "gemma3:4b"]
    
    # Check availability
    print("\n🔍 Checking model availability...")
    available_models = []
    for model in test_models:
        agent = OllamaAgent("test", AgentRole.EXECUTOR, model=model)
        if await agent.client.check_health():
            models_list = await agent.client.list_models()
            if any(model in m for m in models_list):
                available_models.append(model)
                print(f"  ✓ {model}")
            else:
                print(f"  ✗ {model} (not pulled)")
        else:
            print(f"  ✗ Ollama not available")
            break
    
    if not available_models:
        print("\n❌ No models available. Please ensure Ollama is running.")
        return
    
    # Define test tasks
    tasks = [
        Task(
            task_id="code_gen",
            description="Write a Python async function that fetches data from multiple URLs concurrently using aiohttp.",
            complexity=0.7,
            context={"type": "coding"},
        ),
        Task(
            task_id="analysis",
            description="Compare microservices vs monolithic architecture. List 3 pros and 3 cons of each.",
            complexity=0.6,
            context={"type": "analysis"},
        ),
        Task(
            task_id="review",
            description="Review this code for bugs and improvements: `async def fetch(url): return await requests.get(url)`",
            complexity=0.5,
            context={"type": "review"},
        ),
    ]
    
    print(f"\n📋 Test Tasks: {len(tasks)}")
    for t in tasks:
        print(f"  • {t.task_id}: {t.description[:50]}...")
    
    # Run evaluation for each model
    model_results = {}
    scorer = HeuristicScorer()
    
    print("\n" + "-"*70)
    print("RUNNING EVALUATIONS")
    print("-"*70)
    
    for model in available_models:
        print(f"\n🤖 Testing: {model}")
        print("-"*40)
        
        agents = [
            OllamaAgent(f"{model.split(':')[0]}_agent", AgentRole.EXECUTOR, model=model)
        ]
        coordinator = Coordinator(agents=agents)
        
        results = []
        scores = []
        total_time = 0
        total_tokens = 0
        
        for task in tasks:
            print(f"  Running {task.task_id}...", end=" ", flush=True)
            task_results = await coordinator.run_parallel(task)
            result = task_results[0]
            results.append(result)
            
            if result.success:
                score = scorer.score_sync(result, task)
                scores.append(score.overall)
                total_time += result.latency_ms
                total_tokens += result.tokens_used
                print(f"✓ {result.latency_ms:.0f}ms, quality: {score.overall:.2f}")
            else:
                print(f"✗ {result.error}")
        
        model_results[model] = {
            "success_rate": sum(1 for r in results if r.success) / len(results),
            "avg_quality": sum(scores) / len(scores) if scores else 0,
            "avg_latency": total_time / len(tasks),
            "total_tokens": total_tokens,
            "results": results,
        }
    
    # Compare results
    print("\n" + "="*70)
    print("MODEL COMPARISON RESULTS")
    print("="*70)
    
    print(f"\n{'Model':<25} {'Success':<10} {'Quality':<10} {'Latency':<12} {'Tokens':<10}")
    print("-"*70)
    
    for model, data in model_results.items():
        print(f"{model:<25} {data['success_rate']:.0%}       {data['avg_quality']:.2f}       {data['avg_latency']:.0f}ms       {data['total_tokens']}")
    
    # Find best model per metric
    print("\n🏆 Best Models:")
    
    if model_results:
        best_quality = max(model_results.items(), key=lambda x: x[1]['avg_quality'])
        best_speed = min(model_results.items(), key=lambda x: x[1]['avg_latency'])
        best_efficiency = max(
            model_results.items(), 
            key=lambda x: x[1]['avg_quality'] / max(x[1]['avg_latency'], 1) * 1000
        )
        
        print(f"  • Highest Quality: {best_quality[0]} ({best_quality[1]['avg_quality']:.2f})")
        print(f"  • Fastest: {best_speed[0]} ({best_speed[1]['avg_latency']:.0f}ms)")
        print(f"  • Most Efficient: {best_efficiency[0]} (quality/latency ratio)")
    
    # Show sample outputs
    print("\n" + "-"*70)
    print("SAMPLE OUTPUTS (code_gen task)")
    print("-"*70)
    
    for model, data in model_results.items():
        code_result = next((r for r in data['results'] if r.task_id == "code_gen"), None)
        if code_result and code_result.success:
            preview = code_result.output[:300].replace("\n", "\n  ")
            print(f"\n📝 {model}:")
            print(f"  {preview}...")
    
    print("\n✅ Multi-model evaluation complete!")


if __name__ == "__main__":
    asyncio.run(run_multi_model_evaluation())
