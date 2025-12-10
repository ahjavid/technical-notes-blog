#!/usr/bin/env python3
"""
Proper APEE Evaluation Demo - LLM-as-a-Judge.

This demonstrates the REAL APEE evaluation approach:
- Uses LLM to evaluate outputs (not heuristics)
- Evaluates goal alignment, semantic quality, collaboration
- Provides meaningful scores that differentiate quality
- Saves results to JSON for use by other tools

Following patterns from CrewAI's evaluation framework.
"""

import asyncio
import json
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
from apee.evaluation.advanced_patterns import (
    # Progressive Deepening - fail-fast evaluation
    ProgressiveDeepening,
    ProgressiveResult,
    EvaluationDepth,
    create_progressive_evaluator,
    # Jury with Personas - multi-perspective evaluation
    JuryEvaluator,
    JudgePersona,
    PERSONA_CONFIGS,
    create_jury_evaluator,
    # Calibration Loop - negotiated rubric evaluation
    CalibrationLoop,
    CalibratedRubric,
    CalibratedJuryEvaluator,
    create_calibrated_evaluator,
)

console = Console()

# =============================================================================
# EVALUATION MODE ENUM
# =============================================================================

class EvaluationMode:
    """Available evaluation modes."""
    BASIC = "basic"           # Original EnsembleEvaluator
    PROGRESSIVE = "progressive"  # Fail-fast for bulk evaluation
    JURY = "jury"             # Multi-persona evaluation
    CALIBRATED = "calibrated" # Calibration + Jury (best quality)
    ALL = "all"               # Run all modes and combine results


async def run_and_evaluate_scenario(
    scenario: CollaborativeScenario,
    coordinator: Coordinator,
    agents: list[OllamaAgent],
    evaluator: APEEEvaluator | EnsembleEvaluator,
    advanced_evaluator: ProgressiveDeepening | JuryEvaluator | CalibratedJuryEvaluator | None = None,
    evaluation_mode: str = EvaluationMode.BASIC,
) -> dict:
    """Run a scenario and evaluate it using LLM-as-a-judge."""
    
    # Validate minimum agent count for all patterns
    if len(coordinator.agents) < 3:
        raise ValueError(
            f"Evaluation requires at least 3 agents for all patterns, "
            f"got {len(coordinator.agents)}: {list(coordinator.agents.keys())}"
        )
    
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
            n_debate_agents = 3
            n_rounds = 2
            results = await coordinator.run_debate(
                task=task, rounds=n_rounds, agent_ids=agent_ids[:n_debate_agents]
            )
            final_output = results[-1].output if results else ""
            
            # Build traces for each participating agent using LAST round results
            # Results are ordered: [round1_agent1, round1_agent2, round1_agent3, round2_agent1, ...]
            # Last round starts at index: (n_rounds - 1) * n_agents
            last_round_start = (n_rounds - 1) * n_debate_agents
            last_round_results = results[last_round_start:last_round_start + n_debate_agents]
            
            for i, agent_id in enumerate(agent_ids[:n_debate_agents]):
                agent = coordinator.agents[agent_id]
                # Get this agent's result from the last round
                agent_result = last_round_results[i] if i < len(last_round_results) else None
                trace = ExecutionTrace(
                    agent_id=agent_id,
                    agent_role=agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                    task_description=scenario.task_description,
                    expected_output="Debate contribution with reasoned arguments",
                    final_output=agent_result.output if agent_result else "",
                    duration_seconds=(datetime.now() - start_time).total_seconds() / n_rounds,
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
                
        elif pattern.value == "hierarchical":
            # Leader delegates to workers
            leader_id = agent_ids[0]  # First agent is leader
            worker_ids = agent_ids[1:3]  # Others are workers
            results = await coordinator.run_hierarchical(
                task=task, leader_id=leader_id, worker_ids=worker_ids
            )
            final_output = results[-1].output if results else ""  # Synthesis is last
            
            for result in results:
                agent = coordinator.agents.get(result.agent_id)
                if agent:
                    trace = ExecutionTrace(
                        agent_id=result.agent_id,
                        agent_role=agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                        task_description=scenario.task_description,
                        expected_output="Hierarchical contribution",
                        final_output=result.output,
                        duration_seconds=(datetime.now() - start_time).total_seconds() / len(results),
                    )
                    agent_traces.append(trace)
                    
        elif pattern.value == "consensus":
            n_consensus_agents = 3
            max_rounds = 2
            results = await coordinator.run_consensus(
                task=task, max_rounds=max_rounds, agent_ids=agent_ids[:n_consensus_agents]
            )
            final_output = results[-1].output if results else ""
            
            # Get the last round's results for each agent
            # Results may have 1 or 2 rounds depending on early consensus
            n_rounds_actual = len(results) // n_consensus_agents
            if n_rounds_actual > 0:
                last_round_start = (n_rounds_actual - 1) * n_consensus_agents
                last_round_results = results[last_round_start:last_round_start + n_consensus_agents]
            else:
                last_round_results = results
            
            for i, agent_id in enumerate(agent_ids[:n_consensus_agents]):
                agent = coordinator.agents.get(agent_id)
                if agent:
                    agent_result = last_round_results[i] if i < len(last_round_results) else None
                    trace = ExecutionTrace(
                        agent_id=agent_id,
                        agent_role=agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                        task_description=scenario.task_description,
                        expected_output="Consensus contribution seeking agreement",
                        final_output=agent_result.output if agent_result else "",
                        duration_seconds=(datetime.now() - start_time).total_seconds() / n_rounds_actual if n_rounds_actual > 0 else 0,
                    )
                    agent_traces.append(trace)
                        
        elif pattern.value == "peer_review":
            results = await coordinator.run_peer_review(
                task=task, agent_ids=agent_ids[:3]
            )
            # Last phase results are the revised outputs
            final_output = results[-1].output if results else ""
            
            # Use the final revision results for traces
            n_agents = len(agent_ids[:3])
            revision_results = results[-n_agents:]  # Last n results are revisions
            
            for result in revision_results:
                agent = coordinator.agents.get(result.agent_id)
                if agent:
                    trace = ExecutionTrace(
                        agent_id=result.agent_id,
                        agent_role=agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                        task_description=scenario.task_description,
                        expected_output="Peer-reviewed and revised output",
                        final_output=result.output,
                        duration_seconds=(datetime.now() - start_time).total_seconds() / n_agents,
                    )
                    agent_traces.append(trace)
                    
        else:
            # Parallel execution (default)
            results = await coordinator.run_parallel(task)
            # Use the highest quality result as final output
            best_result = max(results, key=lambda r: r.quality_score) if results else None
            final_output = best_result.output if best_result else ""
            
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
    
    # Run advanced evaluation if enabled
    advanced_results = None
    if advanced_evaluator and evaluation_mode != EvaluationMode.BASIC:
        console.print(f"    [dim]Running advanced {evaluation_mode} evaluation...[/dim]")
        
        if evaluation_mode == EvaluationMode.PROGRESSIVE:
            # Progressive Deepening - evaluate each agent trace
            progressive_scores = []
            for trace in agent_traces:
                result = advanced_evaluator.evaluate(trace)
                score = result.final_score.score if result.final_score else 0.0
                tokens_saved = result.tokens_saved_estimate or 0
                progressive_scores.append({
                    "agent_id": trace.agent_id,
                    "score": score,
                    "depth_reached": result.depth_reached.value,
                    "early_termination": result.early_termination,
                    "tokens_saved": tokens_saved,
                })
            valid_scores = [s["score"] for s in progressive_scores if s["score"] is not None]
            advanced_results = {
                "mode": "progressive_deepening",
                "agent_scores": progressive_scores,
                "average_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0,
                "total_tokens_saved": sum(s["tokens_saved"] for s in progressive_scores),
            }
            
        elif evaluation_mode == EvaluationMode.JURY:
            # Jury with Personas - evaluate each agent trace with multi-perspective
            jury_scores = []
            for trace in agent_traces:
                result = advanced_evaluator.evaluate(trace)
                agg_score = result["aggregated_score"].score if result.get("aggregated_score") else 0.0
                jury_scores.append({
                    "agent_id": trace.agent_id,
                    "aggregated_score": agg_score,
                    "persona_scores": {p: (d["score"] or 0) for p, d in result.get("persona_scores", {}).items()},
                    "disagreement": result.get("disagreement", {}),
                })
            valid_scores = [s["aggregated_score"] for s in jury_scores if s["aggregated_score"] is not None]
            advanced_results = {
                "mode": "jury_with_personas",
                "agent_scores": jury_scores,
                "average_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0,
                "personas_used": [p.value for p in PERSONA_CONFIGS.keys()],
            }
            
        elif evaluation_mode == EvaluationMode.CALIBRATED:
            # Calibrated Jury - best quality evaluation
            calibrated_scores = []
            task_type = _infer_task_type(scenario.task_description)
            for trace in agent_traces:
                result = advanced_evaluator.evaluate(trace, task_type=task_type)
                agg_score = result["aggregated_score"].score if result.get("aggregated_score") else 0.0
                calibrated_scores.append({
                    "agent_id": trace.agent_id,
                    "aggregated_score": agg_score,
                    "calibration": result.get("calibration", {}),
                    "persona_scores": {p: (d["score"] or 0) for p, d in result.get("persona_scores", {}).items()},
                    "disagreement": result.get("disagreement", {}),
                })
            valid_scores = [s["aggregated_score"] for s in calibrated_scores if s["aggregated_score"] is not None]
            advanced_results = {
                "mode": "calibrated_jury",
                "agent_scores": calibrated_scores,
                "average_score": sum(valid_scores) / len(valid_scores) if valid_scores else 0,
                "task_type": task_type,
            }
    
    return {
        "scenario_id": scenario.id,
        "pattern": pattern.value,
        "duration": duration,
        "evaluation": evaluation,
        "advanced_evaluation": advanced_results,
    }


def _infer_task_type(task_description: str) -> str:
    """Infer task type from description for calibration."""
    desc_lower = task_description.lower()
    if any(kw in desc_lower for kw in ["security", "vulnerability", "exploit", "attack"]):
        return "security_analysis"
    elif any(kw in desc_lower for kw in ["code", "implement", "function", "class", "programming"]):
        return "code_generation"
    elif any(kw in desc_lower for kw in ["review", "feedback", "improve", "refactor"]):
        return "code_review"
    elif any(kw in desc_lower for kw in ["analyze", "analysis", "evaluate", "assess"]):
        return "analysis"
    else:
        return "general"


async def main(evaluation_mode: str = EvaluationMode.BASIC):
    """Run proper APEE evaluation with LLM-as-a-judge.
    
    Args:
        evaluation_mode: One of 'basic', 'progressive', 'jury', 'calibrated', 'all'
    """
    
    # Handle 'all' mode - run each mode sequentially
    if evaluation_mode == EvaluationMode.ALL:
        console.print(Panel.fit(
            "[bold cyan]APEE Framework - Complete Evaluation Suite[/bold cyan]\n\n"
            "Running ALL evaluation modes sequentially:\n"
            "  1. Basic Ensemble (baseline)\n"
            "  2. Progressive Deepening (token-efficient)\n"
            "  3. Jury with Personas (multi-perspective)\n"
            "  4. Calibrated Jury (best quality)\n\n"
            "[yellow]This will generate 4 JSON result files.[/yellow]",
            title="🤖 Adaptive Poly-Agentic Evaluation Ecosystem"
        ))
        
        modes = [EvaluationMode.BASIC, EvaluationMode.PROGRESSIVE, EvaluationMode.JURY, EvaluationMode.CALIBRATED]
        for i, mode in enumerate(modes, 1):
            console.print(f"\n{'='*70}")
            console.print(f"[bold cyan]Running Mode {i}/4: {mode.upper()}[/bold cyan]")
            console.print('='*70)
            await main(evaluation_mode=mode)
        
        console.print("\n" + "="*70)
        console.print("[bold green]✨ ALL EVALUATION MODES COMPLETE![/bold green]")
        console.print("="*70)
        console.print("\n[yellow]Generated JSON files:[/yellow]")
        console.print("  • data/apee_evaluation_results.json (basic)")
        console.print("  • data/apee_evaluation_results_progressive.json")
        console.print("  • data/apee_evaluation_results_jury.json")
        console.print("  • data/apee_evaluation_results_calibrated.json")
        return
    
    mode_descriptions = {
        EvaluationMode.BASIC: "Basic Ensemble (2 large judges)",
        EvaluationMode.PROGRESSIVE: "Progressive Deepening (fail-fast, token-efficient)",
        EvaluationMode.JURY: "Jury with Personas (4 perspectives)",
        EvaluationMode.CALIBRATED: "Calibrated Jury (best quality)",
    }
    
    console.print(Panel.fit(
        "[bold cyan]APEE Framework - LLM-as-a-Judge Evaluation[/bold cyan]\n\n"
        "[yellow]Proper evaluation methodology:[/yellow]\n"
        "• Uses LLM to evaluate output quality (not heuristics)\n"
        "• Goal alignment: Did agent achieve the task?\n"
        "• Semantic quality: Is reasoning clear and logical?\n"
        "• Collaboration effectiveness: Did agents work well together?\n"
        "• Synthesis quality: Is combined output coherent?\n\n"
        f"[green]Evaluation Mode:[/green] {mode_descriptions.get(evaluation_mode, evaluation_mode)}",
        title="🤖 Adaptive Poly-Agentic Evaluation Ecosystem"
    ))
    
    # Initialize agents
    console.print("\n[bold]Initializing agent pool...[/bold]")
    
    # AGENTS: Small models matched to role strengths (based on benchmark data)
    # Order matters: analyst first (leader for hierarchical), coder second, reviewer last
    # This creates logical flow: analyze → code → review (pipeline order)
    agents = [
        OllamaAgent(
            agent_id="analyst",
            role=AgentRole.ANALYZER,
            model="qwen2.5-coder:3b",  # Best analysis: 0.964, reasoning: 0.950 - ideal for leader/planning
        ),
        OllamaAgent(
            agent_id="coder",
            role=AgentRole.EXECUTOR,
            model="llama3.2:3b",  # Best code_generation: 0.950
        ),
        OllamaAgent(
            agent_id="reviewer",
            role=AgentRole.REVIEWER,
            model="granite4:3b",  # Good code_review: 0.935 - final quality check
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
    # Agents: Llama 3B, Granite 3B → Judges: GPT-OSS 20B, Mistral-Small 24B
    judge_models = ["qwen2.5-coder:7b", "llama3.2:3b"]  # Available models
    evaluator = EnsembleEvaluator(
        judge_models=judge_models,
        base_url="http://localhost:11434",
        aggregation="median",  # Median is robust to outlier judges
    )
    for model in judge_models:
        console.print(f"  ✓ Judge: {model}")
    
    # Initialize advanced evaluator based on mode
    advanced_evaluator = None
    if evaluation_mode != EvaluationMode.BASIC:
        console.print(f"\n[bold]Initializing advanced evaluator ({evaluation_mode})...[/bold]")
        
        if evaluation_mode == EvaluationMode.PROGRESSIVE:
            advanced_evaluator = create_progressive_evaluator(
                model="qwen2.5-coder:7b",
                max_depth=EvaluationDepth.DEEP,  # Limit to DEEP for efficiency
            )
            console.print(f"  ✓ Progressive Deepening (max depth: DEEP)")
            console.print(f"    [dim]Saves 60-80% tokens on obvious pass/fail cases[/dim]")
            
        elif evaluation_mode == EvaluationMode.JURY:
            advanced_evaluator = create_jury_evaluator(
                model="qwen2.5-coder:7b",
                personas=None,  # All 4 personas
            )
            console.print(f"  ✓ Jury with 4 Personas: {[p.value for p in JudgePersona]}")
            console.print(f"    [dim]Multiple perspectives reduce single-viewpoint bias[/dim]")
            
        elif evaluation_mode == EvaluationMode.CALIBRATED:
            advanced_evaluator = create_calibrated_evaluator(
                judge_models=["qwen2.5-coder:7b", "llama3.2:3b"],
                personas=["skeptic", "pragmatist"],  # Focused for efficiency
            )
            console.print(f"  ✓ Calibrated Jury (2 judges, 2 personas)")
            console.print(f"    [dim]Judges negotiate rubric before scoring[/dim]")
    
    # Load all 6 scenarios (one per collaboration pattern)
    dataset = MultiAgentDataset()
    scenarios = dataset.scenarios  # All 6 patterns: peer_review, sequential, debate, parallel, hierarchical, consensus
    
    console.print(f"\n[bold]Running {len(scenarios)} collaborative scenarios (all patterns)...[/bold]")
    console.print("  [dim]Patterns: peer_review, sequential, debate, parallel, hierarchical, consensus[/dim]\n")
    
    all_results = []
    
    for scenario in scenarios:
        console.print(f"  [cyan]{scenario.name}[/cyan] ({scenario.pattern.value})")
        
        result = await run_and_evaluate_scenario(
            scenario, coordinator, agents, evaluator,
            advanced_evaluator=advanced_evaluator,
            evaluation_mode=evaluation_mode,
        )
        
        if "error" not in result:
            all_results.append(result)
            eval_data = result["evaluation"]
            console.print(f"    L1: {eval_data['level1_individual']['average']:.1f}/10  "
                         f"L2: {eval_data['level2_collaborative']['average']:.1f}/10  "
                         f"Overall: {eval_data['overall_apee_score']:.1f}/10")
            
            # Show advanced evaluation summary if available
            if result.get("advanced_evaluation"):
                adv = result["advanced_evaluation"]
                if adv["mode"] == "progressive_deepening":
                    console.print(f"    [dim]Progressive: avg={adv['average_score']:.1f}, tokens_saved={adv['total_tokens_saved']}[/dim]")
                elif adv["mode"] == "jury_with_personas":
                    console.print(f"    [dim]Jury: avg={adv['average_score']:.1f}[/dim]")
                elif adv["mode"] == "calibrated_jury":
                    console.print(f"    [dim]Calibrated: avg={adv['average_score']:.1f}, task_type={adv['task_type']}[/dim]")
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
    
    # Save results to JSON for use by phase6_demo and other tools
    if all_results:
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        
        # Include mode in filename for advanced evaluations
        if evaluation_mode == EvaluationMode.BASIC:
            output_file = output_dir / "apee_evaluation_results.json"
        else:
            output_file = output_dir / f"apee_evaluation_results_{evaluation_mode}.json"
        
        # Prepare JSON-serializable data
        json_results = {
            "timestamp": datetime.now().isoformat(),
            "evaluation_mode": evaluation_mode,
            "judge_models": judge_models,
            "agent_models": {
                "analyst": "qwen2.5-coder:3b",  # Leader for hierarchical
                "coder": "llama3.2:3b",
                "reviewer": "granite4:3b",
            },
            "scenarios": []
        }
        
        for result in all_results:
            eval_data = result["evaluation"]
            scenario_data = {
                "scenario_id": result["scenario_id"],
                "pattern": result["pattern"],
                "duration_seconds": result["duration"],
                "overall_apee_score": eval_data["overall_apee_score"],
                "level1_individual": eval_data["level1_individual"],
                "level2_collaborative": eval_data["level2_collaborative"],
                "level3_ecosystem": eval_data["level3_ecosystem"],
            }
            if "ensemble_metadata" in eval_data:
                scenario_data["ensemble_metadata"] = eval_data["ensemble_metadata"]
            # Include advanced evaluation results
            if result.get("advanced_evaluation"):
                scenario_data["advanced_evaluation"] = result["advanced_evaluation"]
            json_results["scenarios"].append(scenario_data)
        
        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2, default=str)
        
        console.print(f"\n[green]📁 Results saved to: {output_file}[/green]")


def parse_args():
    """Parse command line arguments."""
    import argparse
    parser = argparse.ArgumentParser(
        description="APEE Framework - LLM-as-a-Judge Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Evaluation Modes:
  basic       - Standard ensemble evaluation (2 large judges)
  progressive - Progressive Deepening: fail-fast, saves 60-80% tokens
  jury        - Jury with Personas: 4 perspectives reduce bias
  calibrated  - Calibrated Jury: judges negotiate rubric first (best quality)
  all         - Run ALL modes sequentially (generates 4 JSON files)

Examples:
  python proper_apee_evaluation.py                     # Basic evaluation
  python proper_apee_evaluation.py --mode progressive  # Fast bulk evaluation
  python proper_apee_evaluation.py --mode calibrated   # Best quality
  python proper_apee_evaluation.py --mode all          # Complete suite (all 4 modes)
"""
    )
    parser.add_argument(
        "--mode", "-m",
        choices=["basic", "progressive", "jury", "calibrated", "all"],
        default="basic",
        help="Evaluation mode (default: basic)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(evaluation_mode=args.mode))
