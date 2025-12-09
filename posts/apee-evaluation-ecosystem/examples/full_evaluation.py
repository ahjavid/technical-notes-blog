#!/usr/bin/env python3
"""
Full APEE Evaluation Example with Ollama.

This example demonstrates the complete APEE workflow:
1. Configure multiple specialized agents
2. Execute tasks across agents
3. Collect and evaluate metrics
4. Generate quality scores
5. Produce comprehensive reports
"""

import asyncio
import sys
from pathlib import Path

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent))

from apee.models import Task, AgentRole
from apee.agents.ollama import OllamaAgent
from apee.coordination.coordinator import Coordinator
from apee.evaluation.evaluator import Evaluator
from apee.evaluation.quality import HeuristicScorer, CompositeScorer
from apee.utils.logging import setup_logging, get_logger

logger = get_logger("full_evaluation")


async def run_full_evaluation():
    """Run a complete APEE evaluation with real LLM agents."""
    
    # Configure logging
    setup_logging()
    
    print("\n" + "="*60)
    print("APEE: Adaptive Poly-Agentic Evaluation Ecosystem")
    print("Full Evaluation Demo with Ollama")
    print("="*60 + "\n")
    
    # Check Ollama availability
    test_agent = OllamaAgent(
        agent_id="test",
        role=AgentRole.EXECUTOR,
        model="qwen2.5-coder:3b"
    )
    
    if not await test_agent.client.check_health():
        print("❌ Ollama is not available. Please start Ollama first.")
        print("   Run: ollama serve")
        return
    
    print("✓ Ollama connection verified\n")
    
    # Create specialized agents
    print("Creating agent ensemble...")
    agents = [
        OllamaAgent(
            agent_id="analyst",
            role=AgentRole.ANALYZER,
            model="qwen2.5-coder:3b",
            temperature=0.3,  # Lower temp for analysis
        ),
        OllamaAgent(
            agent_id="coder",
            role=AgentRole.EXECUTOR,
            model="qwen2.5-coder:3b",
            temperature=0.7,  # Higher creativity for code
        ),
        OllamaAgent(
            agent_id="reviewer",
            role=AgentRole.REVIEWER,
            model="qwen2.5-coder:3b",
            temperature=0.2,  # Low temp for critical review
        ),
        OllamaAgent(
            agent_id="synthesizer",
            role=AgentRole.SYNTHESIZER,
            model="qwen2.5-coder:3b",
            temperature=0.5,
        ),
    ]
    
    for agent in agents:
        print(f"  • {agent.agent_id}: {agent.role.value}")
    
    # Define evaluation tasks
    print("\nPreparing evaluation tasks...")
    tasks = [
        Task(
            task_id="analyze_1",
            description="Analyze the trade-offs between REST and GraphQL APIs. List 3 key differences.",
            complexity=0.6,
        ),
        Task(
            task_id="code_1",
            description="Write a Python function to calculate the Fibonacci sequence using memoization. Include type hints.",
            complexity=0.7,
        ),
        Task(
            task_id="review_1",
            description="Review this code and identify issues: `def add(a,b): return a+b+c`",
            complexity=0.4,
        ),
        Task(
            task_id="synthesize_1",
            description="Summarize the key principles of clean code in 3 bullet points.",
            complexity=0.5,
        ),
    ]
    
    for task in tasks:
        print(f"  • {task.task_id} (complexity: {task.complexity})")
    
    # Initialize coordinator
    print("\nInitializing task coordinator...")
    coordinator = Coordinator(agents=agents)
    
    # Execute evaluation - run all tasks in parallel
    print("\n" + "-"*40)
    print("EXECUTING EVALUATION")
    print("-"*40 + "\n")
    
    all_results = []
    for task in tasks:
        print(f"Running task: {task.task_id}...")
        results = await coordinator.run_parallel(task)
        all_results.extend(results)
    
    # Display results
    print("\n" + "-"*40)
    print("AGENT RESULTS")
    print("-"*40)
    
    for result in all_results:
        status = "✓" if result.success else "✗"
        print(f"\n{status} Task: {result.task_id} | Agent: {result.agent_id}")
        print(f"  Latency: {result.latency_ms:.0f}ms | Tokens: {result.tokens_used}")
        
        # Show truncated response
        response_preview = result.output[:150].replace("\n", " ")
        if len(result.output) > 150:
            response_preview += "..."
        print(f"  Response: {response_preview}")
    
    # Calculate metrics
    print("\n" + "-"*40)
    print("EVALUATION METRICS")
    print("-"*40)
    
    successful = [r for r in all_results if r.success]
    success_rate = len(successful) / len(all_results) if all_results else 0
    avg_latency = sum(r.latency_ms for r in all_results) / len(all_results) if all_results else 0
    total_tokens = sum(r.tokens_used for r in all_results)
    
    print(f"\n📊 Individual Metrics:")
    print(f"  Success Rate: {success_rate:.1%}")
    print(f"  Avg Latency: {avg_latency:.0f}ms")
    print(f"  Total Tokens: {total_tokens}")
    
    print(f"\n🏠 Ecosystem Metrics:")
    print(f"  Total Tasks: {len(tasks)}")
    print(f"  Total Results: {len(all_results)}")
    print(f"  Agents Active: {len(agents)}")
    
    # Quality scoring (Phase 2)
    print("\n" + "-"*40)
    print("QUALITY SCORING (Phase 2)")
    print("-"*40)
    
    # Use heuristic scorer for fast evaluation
    scorer = HeuristicScorer()
    
    for i, (result, task) in enumerate(zip(all_results[:len(tasks)], tasks)):
        if result.success:
            score = scorer.score_sync(result, task)
            print(f"\n📝 {result.task_id} ({result.agent_id}):")
            print(f"  Overall Score: {score.overall:.2f}")
            print(f"    • Relevance: {score.relevance:.2f}")
            print(f"    • Completeness: {score.completeness:.2f}")
            print(f"    • Structure: {score.structure:.2f}")
            print(f"    • Clarity: {score.clarity:.2f}")
            if score.reasoning:
                print(f"  Reasoning: {score.reasoning}")
    
    # Try LLM-based scoring for one result
    print("\n" + "-"*40)
    print("LLM-BASED SCORING (Sample)")
    print("-"*40)
    
    composite_scorer = CompositeScorer(
        heuristic_weight=0.4,
        llm_weight=0.6,
        use_llm=True,
        llm_model="qwen2.5-coder:3b"
    )
    
    if successful:
        sample_result = successful[0]
        sample_task = next(t for t in tasks if t.task_id == sample_result.task_id)
        print(f"\nScoring {sample_result.task_id} with LLM judge...")
        composite_score = await composite_scorer.score(sample_result, sample_task)
        print(f"\n📝 Composite Score (Heuristic + LLM):")
        print(f"  Overall: {composite_score.overall:.2f}")
        print(f"    • Relevance: {composite_score.relevance:.2f}")
        print(f"    • Completeness: {composite_score.completeness:.2f}")
        print(f"    • Accuracy: {composite_score.accuracy:.2f}")
        print(f"  Reasoning: {composite_score.reasoning}")
    
    await composite_scorer.close()
    
    # Final summary
    print("\n" + "="*60)
    print("EVALUATION COMPLETE")
    print("="*60)
    
    print(f"\n✓ {len(successful)}/{len(all_results)} results successful")
    print(f"✓ {len(agents)} agents participated")
    print(f"✓ {len(tasks)} tasks evaluated")
    print(f"✓ Quality scoring applied (Heuristic + LLM)")
    print()


if __name__ == "__main__":
    asyncio.run(run_full_evaluation())
