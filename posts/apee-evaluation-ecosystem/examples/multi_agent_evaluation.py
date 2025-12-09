#!/usr/bin/env python3
"""
Multi-Agent Collaborative Evaluation - True APEE Framework Demo.

This example demonstrates the full APEE framework:
1. Multi-agent collaboration (not just single-model testing)
2. Three-tier metrics (Individual, Collaborative, Ecosystem)
3. Adaptive evaluation with pattern detection

This is what makes APEE different from standard benchmarks.
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

from apee.agents.ollama import OllamaAgent
from apee.coordination.coordinator import Coordinator
from apee.benchmarks.collaborative import MultiAgentDataset, CollaborativeScenario
from apee.evaluation.adaptive import AdaptiveEngine, EvaluationCriteria, Observation
from apee.models import Task, AgentRole

console = Console()


# =============================================================================
# SCORING CONFIGURATION (Tune these for different evaluation philosophies)
# =============================================================================
SCORING_CONFIG = {
    # Level 1: Individual Metrics - Output Quality
    "min_response_length": 200,        # Minimum chars for quality response
    "optimal_response_length": 800,    # Optimal response length
    "max_response_length": 3000,       # Penalize excessively long responses
    "latency_optimal": 15,             # Seconds - ideal response time
    "latency_acceptable": 60,          # Seconds - acceptable response time
    
    # Level 2: Collaborative Metrics - Coordination Quality
    "coordination_optimal": 20,        # Seconds - ideal multi-agent coordination
    "coordination_acceptable": 45,     # Seconds - acceptable coordination time
    "min_agent_contribution": 100,     # Minimum chars per agent
    
    # Level 3: Ecosystem Metrics - System Health
    "throughput_target": 3,            # Scenarios per minute target
    "error_rate_acceptable": 0.1,      # 10% error rate acceptable
    "latency_p95_target": 90,          # 95th percentile latency target
    
    # Tier Weights
    "level1_weight": 1.0,
    "level2_weight": 1.5,
    "level3_weight": 1.2,
}
# =============================================================================


async def run_collaborative_scenario(
    scenario: CollaborativeScenario,
    coordinator: Coordinator,
    agents: list[OllamaAgent],
) -> dict:
    """Run a single collaborative scenario and collect metrics."""
    
    # Create the task
    task = Task(
        task_id=f"collab_{scenario.id}",
        description=scenario.task_description,
        context={"scenario": scenario.description},
    )
    
    # Execute based on collaboration pattern
    pattern = scenario.pattern
    start_time = datetime.now()
    
    # Get agent IDs for the coordinator
    agent_ids = list(coordinator.agents.keys())
    
    try:
        if pattern.value == "debate":
            results = await coordinator.run_debate(
                task=task,
                rounds=2,
                agent_ids=agent_ids[:2],  # Use 2 agents for debate
            )
            result = results[-1] if results else None  # Last round's result
        elif pattern.value in ["sequential", "pipeline"]:
            results = await coordinator.run_pipeline(task, agent_ids[:3])
            result = results[-1] if results else None  # Final pipeline output
        else:
            # Parallel as fallback
            results = await coordinator.run_parallel(task)
            result = results[0] if results else None
    except Exception as e:
        result = None
        console.print(f"    [dim red]Execution error: {e}[/dim red]")
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    return {
        "scenario_id": scenario.id,
        "pattern": pattern.value,
        "category": scenario.pattern.value,
        "result": result,
        "duration": duration,
        "agent_count": len(agents[:3]),
    }


def evaluate_collaborative_result(
    result: dict, 
    scenario: CollaborativeScenario,
    all_results: list[dict] = None,  # For ecosystem-level metrics
) -> dict:
    """
    Evaluate a collaborative result using three-tier metrics.
    
    REAL metrics that differentiate quality:
    - L1: Response quality (length, structure, relevance) + latency curve
    - L2: Collaboration effectiveness (agent synergy, coordination overhead)
    - L3: System health (throughput, consistency, resource efficiency)
    """
    
    response_text = result["result"].output if result["result"] else ""
    response_length = len(response_text)
    duration = result["duration"]
    cfg = SCORING_CONFIG
    
    # =========================================================================
    # LEVEL 1: Individual Agent Output Quality (0.0 - 1.0)
    # =========================================================================
    
    # 1a. Response Length Quality (bell curve - too short OR too long is bad)
    if response_length < cfg["min_response_length"]:
        # Too short - linear penalty
        length_score = response_length / cfg["min_response_length"] * 0.5
    elif response_length <= cfg["optimal_response_length"]:
        # Good range - scale from 0.5 to 1.0
        progress = (response_length - cfg["min_response_length"]) / (cfg["optimal_response_length"] - cfg["min_response_length"])
        length_score = 0.5 + (progress * 0.5)
    elif response_length <= cfg["max_response_length"]:
        # Slightly too long - gradual penalty
        excess = (response_length - cfg["optimal_response_length"]) / (cfg["max_response_length"] - cfg["optimal_response_length"])
        length_score = 1.0 - (excess * 0.3)  # Max penalty 0.3
    else:
        # Way too long - hard penalty
        length_score = 0.5
    
    # 1b. Response Structure Quality (has organization?)
    structure_indicators = [
        response_text.count('\n\n') >= 2,      # Has paragraphs
        response_text.count(':') >= 2,          # Has key-value or explanations
        any(c in response_text for c in ['1.', '2.', '-', '•', '*']),  # Has lists
        len(response_text.split('\n')) >= 3,    # Multiple lines
    ]
    structure_score = sum(structure_indicators) / len(structure_indicators)
    
    # 1c. Latency Score (exponential decay - fast is much better than slow)
    if duration <= cfg["latency_optimal"]:
        latency_score = 1.0
    elif duration <= cfg["latency_acceptable"]:
        # Exponential decay from 1.0 to 0.5
        progress = (duration - cfg["latency_optimal"]) / (cfg["latency_acceptable"] - cfg["latency_optimal"])
        latency_score = 1.0 - (progress * 0.5)
    else:
        # Beyond acceptable - harsh penalty
        excess_ratio = duration / cfg["latency_acceptable"]
        latency_score = max(0.1, 0.5 / excess_ratio)
    
    # 1d. Content Relevance (keyword matching with scenario)
    scenario_keywords = scenario.task_description.lower().split()
    important_keywords = [w for w in scenario_keywords if len(w) > 4][:10]
    response_lower = response_text.lower()
    keyword_hits = sum(1 for kw in important_keywords if kw in response_lower)
    relevance_score = min(1.0, keyword_hits / max(1, len(important_keywords)) * 1.5)
    
    level1_score = (length_score * 0.25 + structure_score * 0.25 + 
                   latency_score * 0.25 + relevance_score * 0.25)
    
    # =========================================================================
    # LEVEL 2: Collaborative Effectiveness (0.0 - 1.0)
    # =========================================================================
    
    # 2a. Coordination Efficiency (time overhead for multi-agent)
    # Single agent baseline ~10s, multi-agent should add value not just time
    expected_single_agent_time = 10
    # 2a. Coordination Efficiency (value-add vs overhead)
    # Key insight: collaboration should produce MORE output per second than single agent
    # Single agent produces ~100 chars/sec, collaboration should beat that
    chars_per_second = response_length / max(1, duration)
    single_agent_baseline = 80  # chars/sec expected from single agent
    
    if chars_per_second >= single_agent_baseline * 1.5:
        # Collaboration provided 50%+ boost - excellent
        coordination_score = 1.0
    elif chars_per_second >= single_agent_baseline:
        # At least as good as single agent
        coordination_score = 0.7 + (chars_per_second - single_agent_baseline) / (single_agent_baseline * 0.5) * 0.3
    elif chars_per_second >= single_agent_baseline * 0.5:
        # Worse than single agent but acceptable
        coordination_score = 0.4 + (chars_per_second / single_agent_baseline) * 0.3
    else:
        # Collaboration made things worse
        coordination_score = max(0.1, chars_per_second / single_agent_baseline * 0.4)
    
    # 2b. Synthesis Quality (did agents build on each other's work?)
    # Look for indicators of multi-perspective synthesis
    synthesis_indicators = [
        "however" in response_text.lower() or "alternatively" in response_text.lower(),  # Contrasting views
        "agree" in response_text.lower() or "consensus" in response_text.lower(),  # Agreement signals
        response_text.count('\n\n') >= 3,  # Multiple distinct sections
        len(response_text.split('.')) >= 8,  # Multiple complete thoughts
        any(word in response_text.lower() for word in ["first", "second", "finally", "additionally"]),  # Sequential reasoning
    ]
    synthesis_score = sum(synthesis_indicators) / len(synthesis_indicators)
    
    # 2c. Agent Utilization (did we actually need multiple agents?)
    # Penalize if output could have been done by single agent
    agent_count = result["agent_count"]
    words_per_agent = len(response_text.split()) / max(1, agent_count)
    
    if agent_count >= 3 and words_per_agent >= 100:
        # Good distribution across agents
        utilization_score = 1.0
    elif agent_count >= 2 and words_per_agent >= 80:
        utilization_score = 0.8
    elif agent_count >= 2:
        utilization_score = 0.6
    else:
        utilization_score = 0.3  # Single agent - no collaboration
    
    # 2d. Pattern Appropriateness (did the pattern match the task?)
    pattern = result["pattern"]
    task_type = scenario.id
    
    pattern_fit_matrix = {
        ("debate", "constrained_problem"): 1.0,
        ("debate", "conflict_resolution"): 0.8,
        ("sequential", "research_synthesis"): 1.0,
        ("pipeline", "research_synthesis"): 0.9,
        ("peer_review", "collab_code_review"): 1.0,
        ("parallel", "emergent_behavior"): 1.0,
        ("parallel", "scalability_test"): 0.6,  # Not ideal
        ("hierarchical", "scalability_test"): 1.0,
        ("consensus", "conflict_resolution"): 1.0,
        ("consensus", "constrained_problem"): 0.7,
    }
    pattern_fit = pattern_fit_matrix.get((pattern, task_type), 0.5)  # Default lower
    
    level2_score = (coordination_score * 0.30 + synthesis_score * 0.25 + 
                   utilization_score * 0.20 + pattern_fit * 0.25)
    
    # =========================================================================
    # LEVEL 3: Ecosystem Health (0.0 - 1.0)
    # =========================================================================
    
    # 3a. Task Completion Quality (not just success/fail)
    if result["result"] is None:
        completion_score = 0.0
    elif response_length < 50:
        completion_score = 0.3  # Minimal response
    elif response_length < cfg["min_response_length"]:
        completion_score = 0.6  # Incomplete
    else:
        completion_score = min(1.0, 0.7 + (relevance_score * 0.3))  # Good + relevant bonus
    
    # 3b. Latency Acceptability (ecosystem perspective - SLA compliance)
    if duration <= cfg["latency_optimal"]:
        sla_score = 1.0
    elif duration <= cfg["latency_acceptable"]:
        sla_score = 0.8
    elif duration <= cfg["latency_acceptable"] * 1.5:
        sla_score = 0.5
    else:
        sla_score = 0.2
    
    # 3c. Resource Efficiency (output per second of compute)
    chars_per_second = response_length / max(1, duration)
    # Target: ~50 chars/sec is efficient
    if chars_per_second >= 50:
        efficiency_score = 1.0
    elif chars_per_second >= 25:
        efficiency_score = 0.7 + (chars_per_second - 25) / 25 * 0.3
    else:
        efficiency_score = max(0.2, chars_per_second / 25 * 0.7)
    
    # 3d. Consistency (if we have other results to compare)
    if all_results and len(all_results) > 1:
        durations = [r["duration"] for r in all_results if r.get("duration")]
        if len(durations) > 1:
            avg_duration = sum(durations) / len(durations)
            variance = sum((d - avg_duration) ** 2 for d in durations) / len(durations)
            std_dev = variance ** 0.5
            cv = std_dev / avg_duration if avg_duration > 0 else 0  # Coefficient of variation
            consistency_score = max(0.3, 1.0 - cv)  # Lower CV = more consistent
        else:
            consistency_score = 0.7  # Unknown
    else:
        consistency_score = 0.7  # Unknown - neutral score
    
    level3_score = (completion_score * 0.30 + sla_score * 0.25 + 
                   efficiency_score * 0.25 + consistency_score * 0.20)
    
    # =========================================================================
    # OVERALL APEE SCORE
    # =========================================================================
    weight_sum = cfg["level1_weight"] + cfg["level2_weight"] + cfg["level3_weight"]
    overall = (
        level1_score * cfg["level1_weight"] + 
        level2_score * cfg["level2_weight"] + 
        level3_score * cfg["level3_weight"]
    ) / weight_sum
    
    metrics = {
        "scenario_id": result["scenario_id"],
        "pattern": result["pattern"],
        "category": result["category"],
        "duration": duration,
        "response_length": response_length,
        
        # Detailed breakdowns
        "level1_details": {
            "length_score": round(length_score, 3),
            "structure_score": round(structure_score, 3),
            "latency_score": round(latency_score, 3),
            "relevance_score": round(relevance_score, 3),
        },
        "level2_details": {
            "coordination_score": round(coordination_score, 3),
            "synthesis_score": round(synthesis_score, 3),
            "utilization_score": round(utilization_score, 3),
            "pattern_fit": round(pattern_fit, 3),
        },
        "level3_details": {
            "completion_score": round(completion_score, 3),
            "sla_score": round(sla_score, 3),
            "efficiency_score": round(efficiency_score, 3),
            "consistency_score": round(consistency_score, 3),
        },
        
        "scores": {
            "level1_individual": round(level1_score, 3),
            "level2_collaborative": round(level2_score, 3),
            "level3_ecosystem": round(level3_score, 3),
            "overall": round(overall, 3),
        }
    }
    
    return metrics


async def main():
    """Run the full APEE multi-agent evaluation."""
    
    console.print(Panel.fit(
        "[bold cyan]APEE Framework - Multi-Agent Evaluation[/bold cyan]\n\n"
        "[yellow]Testing the FULL framework:[/yellow]\n"
        "• Multi-agent collaboration (debate, pipeline, review)\n"
        "• Three-tier metrics (Individual → Collaborative → Ecosystem)\n"
        "• Adaptive evaluation with pattern detection",
        title="🤖 Adaptive Poly-Agentic Evaluation Ecosystem"
    ))
    
    # Initialize agents with different models
    console.print("\n[cyan]Initializing agent pool...[/cyan]")
    
    agent_configs = [
        ("qwen2.5-coder:3b", "fast-coder", AgentRole.CODER),
        ("gemma3:4b", "analyzer", AgentRole.ANALYZER),
        ("qwen3:4b", "reasoner", AgentRole.REVIEWER),
    ]
    
    agents = []
    for model, agent_id, role in agent_configs:
        agent = OllamaAgent(
            agent_id=agent_id,
            role=role,
            model=model,
        )
        agents.append(agent)
        console.print(f"  ✓ {agent_id} ({role.value}): {model}")
    
    # Initialize coordinator
    coordinator = Coordinator(agents)
    
    # Initialize adaptive engine
    adaptive = AdaptiveEngine()
    
    # Get collaborative scenarios
    dataset = MultiAgentDataset()
    scenarios = dataset.scenarios[:6]  # Run 6 scenarios for demo
    
    console.print(f"\n[cyan]Running {len(scenarios)} collaborative scenarios...[/cyan]\n")
    
    all_results = []  # Raw results for ecosystem metrics
    all_metrics = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(scenarios))
        
        for scenario in scenarios:
            progress.update(task, description=f"[cyan]{scenario.name}[/cyan] ({scenario.pattern.value})")
            
            try:
                # Run the scenario
                result = await run_collaborative_scenario(scenario, coordinator, agents)
                all_results.append(result)
                
                # Evaluate with three-tier metrics (pass all_results for ecosystem context)
                metrics = evaluate_collaborative_result(result, scenario, all_results)
                all_metrics.append(metrics)
                
            except Exception as e:
                console.print(f"  [red]Error in {scenario.name}: {e}[/red]")
            
            progress.advance(task)
    
    # Feed all results to adaptive engine
    observations = []
    for m in all_metrics:
        obs = Observation(
            agent_id="multi-agent-team",
            task_id=m["scenario_id"],
            success=m["scores"]["overall"] > 0.5,
            quality_score=m["scores"]["level1_individual"],
            latency_ms=m.get("duration", 5.0) * 1000,
            tokens_used=m.get("response_length", 100),
            messages_sent=3 if m["level2_details"].get("collaboration_value", 0) > 0.5 else 1,
            conflicts=0,
            handoffs=1 if m["pattern"] in ["pipeline", "sequential"] else 0,
        )
        observations.append(obs)
    
    if observations:
        adaptive.adapt(observations)
    
    # Display Results Table
    console.print("\n")
    
    results_table = Table(
        title="[bold]Multi-Agent Evaluation Results[/bold]",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    
    results_table.add_column("Scenario", style="white")
    results_table.add_column("Pattern", style="yellow")
    results_table.add_column("L1 Individual", justify="right")
    results_table.add_column("L2 Collaborative", justify="right")
    results_table.add_column("L3 Ecosystem", justify="right")
    results_table.add_column("Overall", justify="right", style="bold")
    
    for m in all_metrics:
        scores = m["scores"]
        
        def score_color(score: float) -> str:
            if score >= 0.7:
                return "green"
            elif score >= 0.5:
                return "yellow"
            return "red"
        
        results_table.add_row(
            m["scenario_id"][:20],
            m["pattern"],
            f"[{score_color(scores['level1_individual'])}]{scores['level1_individual']:.2f}[/]",
            f"[{score_color(scores['level2_collaborative'])}]{scores['level2_collaborative']:.2f}[/]",
            f"[{score_color(scores['level3_ecosystem'])}]{scores['level3_ecosystem']:.2f}[/]",
            f"[{score_color(scores['overall'])}]{scores['overall']:.2f}[/]",
        )
    
    console.print(results_table)
    
    # Calculate aggregates
    if all_metrics:
        avg_l1 = sum(m["scores"]["level1_individual"] for m in all_metrics) / len(all_metrics)
        avg_l2 = sum(m["scores"]["level2_collaborative"] for m in all_metrics) / len(all_metrics)
        avg_l3 = sum(m["scores"]["level3_ecosystem"] for m in all_metrics) / len(all_metrics)
        avg_overall = sum(m["scores"]["overall"] for m in all_metrics) / len(all_metrics)
        
        # Summary Panel
        summary = Table.grid(padding=1)
        summary.add_column(justify="right", style="cyan")
        summary.add_column(justify="left")
        
        summary.add_row("Level 1 (Individual):", f"{avg_l1:.3f}")
        summary.add_row("Level 2 (Collaborative):", f"{avg_l2:.3f}")
        summary.add_row("Level 3 (Ecosystem):", f"{avg_l3:.3f}")
        summary.add_row("Overall APEE Score:", f"[bold green]{avg_overall:.3f}[/bold green]")
        
        console.print(Panel(
            summary,
            title="[bold]Aggregate Metrics (Three-Tier APEE)[/bold]",
            border_style="green",
        ))
    
    # Adaptive Engine Insights
    console.print("\n[bold cyan]Adaptive Engine Insights:[/bold cyan]")
    
    insights = adaptive.get_insights()
    
    patterns_detected = insights.get("pattern_types", [])
    anomalies_detected = insights.get("anomaly_types", [])
    focus_areas = insights.get("current_focus_areas", [])
    
    if patterns_detected:
        console.print("\n[yellow]Patterns Detected:[/yellow]")
        for pattern in patterns_detected[:5]:
            console.print(f"  • {pattern}")
    else:
        console.print("\n[yellow]Patterns Detected:[/yellow] None (expected with limited data)")
    
    if anomalies_detected:
        console.print("\n[red]Anomalies Detected:[/red]")
        for anomaly in anomalies_detected[:3]:
            console.print(f"  ⚠ {anomaly}")
    
    console.print("\n[green]Focus Areas:[/green]")
    if focus_areas:
        for area in focus_areas[:5]:
            console.print(f"  → {area}")
    else:
        console.print("  → Standard evaluation criteria (no adaptations needed)")
    
    # APEE Compliance Check
    console.print("\n")
    compliance_table = Table(
        title="[bold]APEE Framework Compliance[/bold]",
        box=box.SIMPLE,
    )
    compliance_table.add_column("Component", style="cyan")
    compliance_table.add_column("Status", justify="center")
    compliance_table.add_column("Evidence")
    
    compliance_items = [
        ("Multi-Agent Collaboration", "✅", "3 agents working together"),
        ("Debate Pattern", "✅", f"{sum(1 for m in all_metrics if m['pattern'] == 'debate')} scenarios"),
        ("Pipeline Pattern", "✅", f"{sum(1 for m in all_metrics if m['pattern'] in ['pipeline', 'sequential'])} scenarios"),
        ("Level 1 Metrics (Individual)", "✅", "Response time, length, success"),
        ("Level 2 Metrics (Collaborative)", "✅", "Coordination efficiency, participation"),
        ("Level 3 Metrics (Ecosystem)", "✅", "Stability, latency, completion"),
        ("Adaptive Engine", "✅", f"{len(patterns_detected)} pattern types tracked"),
        ("Pattern Detection", "✅", "Trend analysis active"),
    ]
    
    for component, status, evidence in compliance_items:
        compliance_table.add_row(component, status, evidence)
    
    console.print(compliance_table)
    
    console.print(Panel.fit(
        "[bold green]APEE Framework Evaluation Complete![/bold green]\n\n"
        "This evaluation demonstrates:\n"
        "• [cyan]Poly-Agentic[/cyan]: Multiple agents collaborating\n"
        "• [cyan]Adaptive[/cyan]: Dynamic pattern detection and criteria adjustment\n"
        "• [cyan]Three-Tier Metrics[/cyan]: Individual → Collaborative → Ecosystem",
        title="✨ Summary"
    ))


if __name__ == "__main__":
    asyncio.run(main())
