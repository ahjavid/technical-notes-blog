#!/usr/bin/env python3
"""
Proper APEE Evaluation Demo - LLM-as-a-Judge.

This demonstrates the REAL APEE evaluation approach:
- Uses LLM to evaluate outputs (not heuristics)
- Evaluates goal alignment, semantic quality, collaboration
- Provides meaningful scores that differentiate quality

Following patterns from CrewAI's evaluation framework.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from apee.agents.ollama import OllamaAgent
from apee.coordination.coordinator import Coordinator
from apee.benchmarks.collaborative import MultiAgentDataset, CollaborativeScenario
from apee.models import Task, AgentRole
from apee.evaluation.llm_evaluator import (
    APEEEvaluator,
    EnsembleEvaluator,
    ExecutionTrace,
    CollaborativeTrace,
    MetricCategory,
)

console = Console()


async def run_and_evaluate_scenario(
    scenario: CollaborativeScenario,
    coordinator: Coordinator,
    agents: list[OllamaAgent],
    evaluator: APEEEvaluator | EnsembleEvaluator,
) -> dict:
    """Run a scenario and evaluate it using LLM-as-a-judge."""
    
    # Create the task
    task = Task(
        task_id=f"apee_{scenario.id}",
        description=scenario.task_description,
        context={"scenario": scenario.description},
    )
    
    pattern = scenario.pattern
    start_time = datetime.now()
    agent_ids = list(coordinator.agents.keys())
    
    # Track individual agent outputs
    agent_traces = []
    
    try:
        if pattern.value == "debate":
            results = await coordinator.run_debate(
                task=task, rounds=2, agent_ids=agent_ids[:2]
            )
            final_output = results[-1].output if results else ""
            
            # Build traces for each participating agent
            for i, agent_id in enumerate(agent_ids[:2]):
                agent = coordinator.agents[agent_id]
                trace = ExecutionTrace(
                    agent_id=agent_id,
                    agent_role=agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                    task_description=scenario.task_description,
                    expected_output="Debate contribution with reasoned arguments",
                    final_output=results[i].output if i < len(results) else "",
                    duration_seconds=(datetime.now() - start_time).total_seconds() / 2,
                )
                agent_traces.append(trace)
                
        elif pattern.value in ["sequential", "pipeline"]:
            results = await coordinator.run_pipeline(task, agent_ids[:3])
            final_output = results[-1].output if results else ""
            
            for i, (agent_id, result) in enumerate(zip(agent_ids[:3], results)):
                agent = coordinator.agents[agent_id]
                trace = ExecutionTrace(
                    agent_id=agent_id,
                    agent_role=agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                    task_description=f"Pipeline stage {i+1}: {scenario.task_description[:100]}",
                    expected_output=f"Stage {i+1} output",
                    final_output=result.output,
                    duration_seconds=(datetime.now() - start_time).total_seconds() / 3,
                )
                agent_traces.append(trace)
        else:
            # Parallel execution
            results = await coordinator.run_parallel(task)
            final_output = results[0].output if results else ""
            
            for agent_id, result in zip(agent_ids[:3], results):
                agent = coordinator.agents[agent_id]
                trace = ExecutionTrace(
                    agent_id=agent_id,
                    agent_role=agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                    task_description=scenario.task_description,
                    expected_output="Parallel contribution",
                    final_output=result.output,
                    duration_seconds=(datetime.now() - start_time).total_seconds(),
                )
                agent_traces.append(trace)
                
    except Exception as e:
        console.print(f"    [dim red]Execution error: {e}[/dim red]")
        return {"error": str(e)}
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    # Build collaborative trace for evaluation
    collaborative_trace = CollaborativeTrace(
        scenario_id=scenario.id,
        scenario_description=scenario.description,
        collaboration_pattern=pattern.value,
        participating_agents=[t.agent_role for t in agent_traces],
        agent_traces=agent_traces,
        final_synthesized_output=final_output,
        total_duration_seconds=duration,
    )
    
    # Run LLM-based evaluation
    console.print(f"    [dim]Evaluating with LLM...[/dim]")
    evaluation = evaluator.evaluate_full(collaborative_trace)
    
    return {
        "scenario_id": scenario.id,
        "pattern": pattern.value,
        "duration": duration,
        "evaluation": evaluation,
    }


async def main():
    """Run proper APEE evaluation with LLM-as-a-judge."""
    
    console.print(Panel.fit(
        "[bold cyan]APEE Framework - LLM-as-a-Judge Evaluation[/bold cyan]\n\n"
        "[yellow]Proper evaluation methodology:[/yellow]\n"
        "• Uses LLM to evaluate output quality (not heuristics)\n"
        "• Goal alignment: Did agent achieve the task?\n"
        "• Semantic quality: Is reasoning clear and logical?\n"
        "• Collaboration effectiveness: Did agents work well together?\n"
        "• Synthesis quality: Is combined output coherent?",
        title="🤖 Adaptive Poly-Agentic Evaluation Ecosystem"
    ))
    
    # Initialize agents
    console.print("\n[bold]Initializing agent pool...[/bold]")
    
    # AGENTS: Small models matched to role strengths (based on benchmark data)
    # Gemma reserved for judges only
    agents = [
        OllamaAgent(
            agent_id="coder",
            role=AgentRole.EXECUTOR,
            model="llama3.2:3b",  # Best code_generation: 0.950
        ),
        OllamaAgent(
            agent_id="analyst",
            role=AgentRole.ANALYZER,
            model="qwen2.5-coder:3b",  # Best analysis: 0.964, reasoning: 0.950
        ),
        OllamaAgent(
            agent_id="reviewer",
            role=AgentRole.REVIEWER,
            model="granite4:3b",  # Good code_review: 0.935
        ),
    ]
    
    for agent in agents:
        console.print(f"  ✓ {agent.agent_id} ({agent.role.value}): {agent.model}")
    
    # Initialize coordinator with agents
    coordinator = Coordinator(agents)
    
    # Initialize ENSEMBLE evaluator with DIFFERENT model families than agents
    # This avoids self-preference bias (Qwen agents evaluated by Llama+Gemma judges)
    console.print("\n[bold]Initializing ensemble LLM evaluator...[/bold]")
    console.print("  [dim]Using different model families to avoid self-preference bias[/dim]")
    
    # JUDGES: Large models from DIFFERENT families (evaluators)
    # Must be BIGGER than agents and from DIFFERENT families
    # Agents: Llama 3B, Granite 3B → Judges: Qwen 14B, Gemma 12B
    judge_models = ["qwen3:14b", "gemma3:12b"]
    evaluator = EnsembleEvaluator(
        judge_models=judge_models,
        base_url="http://localhost:11434",
        aggregation="median",  # Median is robust to outlier judges
    )
    for model in judge_models:
        console.print(f"  ✓ Judge: {model}")
    
    # Load scenarios (use subset for demo)
    dataset = MultiAgentDataset()
    scenarios = dataset.scenarios[:3]  # Just 3 for demo (LLM eval is slower)
    
    console.print(f"\n[bold]Running {len(scenarios)} collaborative scenarios...[/bold]\n")
    
    all_results = []
    
    for scenario in scenarios:
        console.print(f"  [cyan]{scenario.name}[/cyan] ({scenario.pattern.value})")
        
        result = await run_and_evaluate_scenario(
            scenario, coordinator, agents, evaluator
        )
        
        if "error" not in result:
            all_results.append(result)
            eval_data = result["evaluation"]
            console.print(f"    L1: {eval_data['level1_individual']['average']:.1f}/10  "
                         f"L2: {eval_data['level2_collaborative']['average']:.1f}/10  "
                         f"Overall: {eval_data['overall_apee_score']:.1f}/10")
        console.print()
    
    # Display results
    if all_results:
        console.print("\n")
        
        table = Table(
            title="APEE Evaluation Results (LLM-as-a-Judge)",
            box=box.ROUNDED,
        )
        table.add_column("Scenario", style="cyan")
        table.add_column("Pattern", style="yellow")
        table.add_column("L1 Individual", justify="right")
        table.add_column("L2 Collaborative", justify="right")
        table.add_column("L3 Ecosystem", justify="right")
        table.add_column("Overall", justify="right", style="bold green")
        
        for result in all_results:
            eval_data = result["evaluation"]
            table.add_row(
                result["scenario_id"],
                result["pattern"],
                f"{eval_data['level1_individual']['average']:.1f}/10",
                f"{eval_data['level2_collaborative']['average']:.1f}/10",
                f"{eval_data['level3_ecosystem']['overall']:.1f}/10",
                f"{eval_data['overall_apee_score']:.1f}/10",
            )
        
        console.print(table)
        
        # Show detailed breakdown for first result
        if all_results:
            first = all_results[0]
            console.print(Panel(
                f"[bold]Detailed Breakdown for: {first['scenario_id']}[/bold]\n\n"
                f"[yellow]Level 1 (Individual - per agent):[/yellow]\n"
                + "\n".join([
                    f"  • {agent}: Goal={scores['goal_alignment']['score']:.1f}, Semantic={scores['semantic_quality']['score']:.1f}"
                    for agent, scores in first['evaluation']['level1_individual']['scores_by_agent'].items()
                ]) + "\n\n"
                f"[yellow]Level 2 (Collaborative):[/yellow]\n"
                f"  • Collaboration: {first['evaluation']['level2_collaborative']['scores']['collaboration_effectiveness']['score']:.1f}/10\n"
                f"  • Synthesis: {first['evaluation']['level2_collaborative']['scores']['synthesis_quality']['score']:.1f}/10\n\n"
                f"[yellow]Level 3 (Ecosystem):[/yellow]\n"
                f"  • Efficiency: {first['evaluation']['level3_ecosystem']['efficiency']:.1f}/10\n"
                f"  • Stability: {first['evaluation']['level3_ecosystem']['stability']:.1f}/10\n"
                f"  • Throughput: {first['evaluation']['level3_ecosystem']['throughput']:.1f}/10\n"
                f"  • Adaptability: {first['evaluation']['level3_ecosystem']['adaptability']:.1f}/10",
                title="📊 Metric Breakdown",
            ))
    
    console.print(Panel.fit(
        "[bold green]✨ APEE Evaluation Complete![/bold green]\n\n"
        "This evaluation used [cyan]ensemble LLM-as-a-judge[/cyan] methodology:\n"
        "• Agents: Small diverse models (Llama/Qwen/Granite 3B)\n"
        "• Judges: Large models (Qwen 14B + Gemma 12B) - BIGGER than agents\n"
        "• Different families avoid self-preference bias\n"
        "• Scores aggregated using median (robust to outliers)",
        title="Summary"
    ))
    
    # Show ensemble disagreement if available
    if all_results and "ensemble_metadata" in all_results[0]["evaluation"]:
        meta = all_results[0]["evaluation"]["ensemble_metadata"]
        disagreement = meta.get("disagreement", {})
        console.print(Panel.fit(
            f"[yellow]Judge Models:[/yellow] {', '.join(meta['judge_models'])}\n"
            f"[yellow]Aggregation:[/yellow] {meta['aggregation_method']}\n\n"
            f"[yellow]Individual Judge Scores (first scenario):[/yellow]\n" +
            "\n".join([
                f"  • {j['model']}: Overall={j['overall']}, L1={j['l1']}, L2={j['l2']}"
                for j in meta['individual_judge_scores']
            ]) + "\n\n"
            f"[yellow]Disagreement Metrics:[/yellow]\n"
            f"  • Overall StdDev: {disagreement.get('overall_stdev', 0):.2f}\n"
            f"  • Overall Range: {disagreement.get('overall_range', 0):.2f}\n"
            f"  • High Disagreement: {'⚠️ Yes' if disagreement.get('high_disagreement') else '✅ No'}",
            title="🔍 Ensemble Evaluation Details",
        ))


if __name__ == "__main__":
    asyncio.run(main())
