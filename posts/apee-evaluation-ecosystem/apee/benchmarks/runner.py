"""
Benchmark runner for APEE.

Executes evaluation scenarios across models with proper statistical methodology.
"""

import asyncio
import statistics
import time
from typing import Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime

from pydantic import BaseModel, Field

from apee.models import Task, AgentResult, AgentRole
from apee.agents.ollama import OllamaAgent
from apee.benchmarks.datasets import (
    BenchmarkDataset,
    EvaluationScenario,
    TaskCategory,
    Complexity,
)
from apee.evaluation.quality import HeuristicScorer, QualityScore


class BenchmarkConfig(BaseModel):
    """Configuration for benchmark runs."""
    
    # Models to evaluate
    models: list[str] = Field(
        default=["qwen2.5-coder:3b"],
        description="List of model names to benchmark"
    )
    
    # Run configuration
    runs_per_scenario: int = Field(
        default=3,
        ge=1,
        description="Number of runs per scenario for statistical significance"
    )
    
    # Scenario selection
    categories: Optional[list[TaskCategory]] = Field(
        default=None,
        description="Filter to specific categories (None = all)"
    )
    complexities: Optional[list[Complexity]] = Field(
        default=None,
        description="Filter to specific complexities (None = all)"
    )
    scenario_limit: Optional[int] = Field(
        default=None,
        description="Limit total scenarios (None = all)"
    )
    
    # Model parameters
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1)
    timeout_seconds: float = Field(default=60.0, ge=1.0)
    
    # Output options
    verbose: bool = Field(default=True)
    save_outputs: bool = Field(default=True)


@dataclass
class ScenarioResult:
    """Result from a single scenario run."""
    scenario_id: str
    model: str
    run_number: int
    success: bool
    output: str
    latency_ms: float
    tokens_used: int
    quality_score: QualityScore
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ModelScenarioStats:
    """Aggregated stats for a model on a specific scenario."""
    scenario_id: str
    model: str
    runs: int
    success_rate: float
    mean_quality: float
    std_quality: float
    mean_latency_ms: float
    std_latency_ms: float
    mean_tokens: float
    keyword_hit_rate: float  # % of expected keywords found
    structure_match_rate: float  # % of runs with expected structure


@dataclass
class ModelStats:
    """Overall stats for a model across all scenarios."""
    model: str
    total_scenarios: int
    total_runs: int
    overall_success_rate: float
    mean_quality: float
    std_quality: float
    mean_latency_ms: float
    std_latency_ms: float
    mean_tokens: float
    
    # Per-category performance
    category_scores: dict[str, float] = field(default_factory=dict)
    
    # Per-complexity performance
    complexity_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Complete benchmark results."""
    config: BenchmarkConfig
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    
    # Raw results
    scenario_results: list[ScenarioResult] = field(default_factory=list)
    
    # Aggregated stats
    model_scenario_stats: list[ModelScenarioStats] = field(default_factory=list)
    model_stats: list[ModelStats] = field(default_factory=list)
    
    # Rankings
    quality_ranking: list[str] = field(default_factory=list)
    speed_ranking: list[str] = field(default_factory=list)
    efficiency_ranking: list[str] = field(default_factory=list)  # quality/latency


class BenchmarkRunner:
    """
    Runs comprehensive benchmarks following best practices.
    
    Features:
    - Multiple runs per scenario for statistical significance
    - Diverse task categories and complexities
    - Ground truth comparison where available
    - Structured output validation
    - Keyword/constraint checking
    """
    
    def __init__(
        self,
        dataset: Optional[BenchmarkDataset] = None,
        scorer: Optional[HeuristicScorer] = None,
    ):
        self.dataset = dataset or BenchmarkDataset()
        self.scorer = scorer or HeuristicScorer()
        self._progress_callback: Optional[Callable[[str], None]] = None
    
    def set_progress_callback(self, callback: Callable[[str], None]):
        """Set a callback for progress updates."""
        self._progress_callback = callback
    
    def _log(self, message: str):
        """Log a message."""
        if self._progress_callback:
            self._progress_callback(message)
        else:
            print(message)
    
    async def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        """Run the complete benchmark suite."""
        started_at = datetime.now()
        
        # Get scenarios based on config filters
        scenarios = self.dataset.get_subset(
            categories=config.categories,
            complexities=config.complexities,
            limit=config.scenario_limit,
        )
        
        if not scenarios:
            raise ValueError("No scenarios match the filter criteria")
        
        self._log(f"\n{'='*70}")
        self._log("APEE BENCHMARK SUITE")
        self._log(f"{'='*70}")
        self._log(f"Models: {', '.join(config.models)}")
        self._log(f"Scenarios: {len(scenarios)}")
        self._log(f"Runs per scenario: {config.runs_per_scenario}")
        self._log(f"Total evaluations: {len(config.models) * len(scenarios) * config.runs_per_scenario}")
        self._log(f"{'='*70}\n")
        
        # Run evaluations
        all_results: list[ScenarioResult] = []
        
        for model in config.models:
            self._log(f"\n🤖 Evaluating: {model}")
            self._log("-" * 50)
            
            agent = OllamaAgent(
                agent_id=f"benchmark_{model}",
                role=AgentRole.EXECUTOR,
                model=model,
            )
            
            # Check model availability
            if not await agent.client.check_health():
                self._log(f"  ❌ Ollama not available, skipping {model}")
                continue
            
            models_available = await agent.client.list_models()
            if not any(model in m for m in models_available):
                self._log(f"  ❌ Model {model} not pulled, skipping")
                continue
            
            for scenario in scenarios:
                for run_num in range(1, config.runs_per_scenario + 1):
                    result = await self._run_scenario(
                        agent, scenario, run_num, config
                    )
                    all_results.append(result)
                    
                    if config.verbose:
                        status = "✓" if result.success else "✗"
                        self._log(
                            f"  {status} {scenario.id} (run {run_num}): "
                            f"quality={result.quality_score.overall:.2f}, "
                            f"latency={result.latency_ms:.0f}ms"
                        )
        
        completed_at = datetime.now()
        duration = (completed_at - started_at).total_seconds()
        
        # Compute aggregated stats
        model_scenario_stats = self._compute_model_scenario_stats(all_results, scenarios)
        model_stats = self._compute_model_stats(all_results, scenarios, config.models)
        
        # Compute rankings
        quality_ranking = sorted(
            model_stats,
            key=lambda m: m.mean_quality,
            reverse=True
        )
        speed_ranking = sorted(
            model_stats,
            key=lambda m: m.mean_latency_ms
        )
        efficiency_ranking = sorted(
            model_stats,
            key=lambda m: m.mean_quality / max(m.mean_latency_ms, 1) * 1000,
            reverse=True
        )
        
        return BenchmarkResult(
            config=config,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=duration,
            scenario_results=all_results,
            model_scenario_stats=model_scenario_stats,
            model_stats=model_stats,
            quality_ranking=[m.model for m in quality_ranking],
            speed_ranking=[m.model for m in speed_ranking],
            efficiency_ranking=[m.model for m in efficiency_ranking],
        )
    
    async def _run_scenario(
        self,
        agent: OllamaAgent,
        scenario: EvaluationScenario,
        run_number: int,
        config: BenchmarkConfig,
    ) -> ScenarioResult:
        """Run a single scenario evaluation."""
        task = scenario.to_task()
        
        try:
            start_time = time.perf_counter()
            result = await asyncio.wait_for(
                agent.execute(task),
                timeout=config.timeout_seconds
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            
            # Score the result
            quality_score = self.scorer.score_sync(result, task)
            
            # Enhance scoring with scenario-specific checks
            quality_score = self._enhance_score(quality_score, result, scenario)
            
            return ScenarioResult(
                scenario_id=scenario.id,
                model=agent.model,
                run_number=run_number,
                success=result.success,
                output=result.output,
                latency_ms=elapsed_ms,
                tokens_used=result.tokens_used,
                quality_score=quality_score,
                error=result.error,
            )
            
        except asyncio.TimeoutError:
            return ScenarioResult(
                scenario_id=scenario.id,
                model=agent.model,
                run_number=run_number,
                success=False,
                output="",
                latency_ms=config.timeout_seconds * 1000,
                tokens_used=0,
                quality_score=QualityScore(overall=0.0, reasoning="Timeout"),
                error="Timeout",
            )
        except Exception as e:
            return ScenarioResult(
                scenario_id=scenario.id,
                model=agent.model,
                run_number=run_number,
                success=False,
                output="",
                latency_ms=0,
                tokens_used=0,
                quality_score=QualityScore(overall=0.0, reasoning=str(e)),
                error=str(e),
            )
    
    def _enhance_score(
        self,
        base_score: QualityScore,
        result: AgentResult,
        scenario: EvaluationScenario,
    ) -> QualityScore:
        """Enhance score with scenario-specific checks."""
        output = result.output.lower()
        adjustments = []
        bonus = 0.0
        penalty = 0.0
        
        # Check expected keywords
        if scenario.expected_keywords:
            found = sum(1 for kw in scenario.expected_keywords if kw.lower() in output)
            keyword_rate = found / len(scenario.expected_keywords)
            if keyword_rate >= 0.8:
                bonus += 0.05
                adjustments.append(f"keyword hit {found}/{len(scenario.expected_keywords)}")
            elif keyword_rate < 0.3:
                penalty += 0.1
                adjustments.append(f"low keyword coverage")
        
        # Check expected structure
        if scenario.expected_structure:
            has_structure = self._check_structure(result.output, scenario.expected_structure)
            if has_structure:
                bonus += 0.03
                adjustments.append("correct structure")
            else:
                penalty += 0.05
                adjustments.append("missing expected structure")
        
        # Check constraints
        if scenario.constraints:
            constraints_met = self._check_constraints(result.output, scenario.constraints)
            if constraints_met < len(scenario.constraints):
                penalty += 0.1 * (1 - constraints_met / len(scenario.constraints))
                adjustments.append(f"constraints: {constraints_met}/{len(scenario.constraints)}")
        
        # Apply adjustments
        adjusted_overall = max(0.0, min(1.0, base_score.overall + bonus - penalty))
        
        reasoning = base_score.reasoning
        if adjustments:
            reasoning += f"; scenario checks: {', '.join(adjustments)}"
        
        return QualityScore(
            overall=adjusted_overall,
            relevance=base_score.relevance,
            completeness=base_score.completeness,
            structure=base_score.structure,
            accuracy=base_score.accuracy,
            clarity=base_score.clarity,
            reasoning=reasoning,
        )
    
    def _check_structure(self, output: str, expected: str) -> bool:
        """Check if output has expected structure."""
        if expected == "code_block":
            return "```" in output or "def " in output or "class " in output
        elif expected == "list":
            return any(c in output for c in ["-", "*", "•", "1.", "1)"])
        elif expected == "paragraph":
            return len(output.split("\n\n")) >= 1 and len(output) > 50
        elif expected == "steps":
            return any(f"step {i}" in output.lower() or f"{i}." in output for i in range(1, 5))
        return True
    
    def _check_constraints(self, output: str, constraints: list[str]) -> int:
        """Check how many constraints are met."""
        met = 0
        output_lower = output.lower()
        
        for constraint in constraints:
            constraint_lower = constraint.lower()
            
            # Simple keyword matching for common constraints
            if "exactly" in constraint_lower:
                # Try to extract number
                import re
                nums = re.findall(r'\d+', constraint)
                if nums:
                    expected_count = int(nums[0])
                    if "bullet" in constraint_lower or "item" in constraint_lower:
                        actual = len(re.findall(r'^[\s]*[-*•]\s', output, re.MULTILINE))
                        actual += len(re.findall(r'^[\s]*\d+[.)]\s', output, re.MULTILINE))
                        if actual == expected_count:
                            met += 1
                    elif "sentence" in constraint_lower:
                        sentences = re.split(r'[.!?]+', output)
                        actual = len([s for s in sentences if len(s.strip()) > 10])
                        if actual == expected_count:
                            met += 1
            elif "must use" in constraint_lower:
                required = constraint_lower.replace("must use", "").strip()
                if required in output_lower:
                    met += 1
            elif "must identify" in constraint_lower or "must suggest" in constraint_lower:
                # More lenient - just check the output isn't empty
                if len(output) > 100:
                    met += 1
            else:
                # Default: check if constraint keywords appear in output
                keywords = [w for w in constraint_lower.split() if len(w) > 3]
                if keywords and any(kw in output_lower for kw in keywords):
                    met += 1
        
        return met
    
    def _compute_model_scenario_stats(
        self,
        results: list[ScenarioResult],
        scenarios: list[EvaluationScenario],
    ) -> list[ModelScenarioStats]:
        """Compute per-model, per-scenario statistics."""
        stats = []
        
        # Group by model and scenario
        grouped: dict[tuple[str, str], list[ScenarioResult]] = {}
        for r in results:
            key = (r.model, r.scenario_id)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(r)
        
        for (model, scenario_id), scenario_results in grouped.items():
            scenario = next((s for s in scenarios if s.id == scenario_id), None)
            
            qualities = [r.quality_score.overall for r in scenario_results]
            latencies = [r.latency_ms for r in scenario_results if r.success]
            successes = [r for r in scenario_results if r.success]
            
            # Calculate keyword and structure match rates
            keyword_hits = 0
            structure_matches = 0
            
            for r in scenario_results:
                if scenario and scenario.expected_keywords:
                    output_lower = r.output.lower()
                    found = sum(1 for kw in scenario.expected_keywords if kw.lower() in output_lower)
                    if found >= len(scenario.expected_keywords) * 0.5:
                        keyword_hits += 1
                
                if scenario and scenario.expected_structure:
                    if self._check_structure(r.output, scenario.expected_structure):
                        structure_matches += 1
            
            stats.append(ModelScenarioStats(
                scenario_id=scenario_id,
                model=model,
                runs=len(scenario_results),
                success_rate=len(successes) / len(scenario_results) if scenario_results else 0,
                mean_quality=statistics.mean(qualities) if qualities else 0,
                std_quality=statistics.stdev(qualities) if len(qualities) > 1 else 0,
                mean_latency_ms=statistics.mean(latencies) if latencies else 0,
                std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
                mean_tokens=statistics.mean([r.tokens_used for r in scenario_results]),
                keyword_hit_rate=keyword_hits / len(scenario_results) if scenario_results else 0,
                structure_match_rate=structure_matches / len(scenario_results) if scenario_results else 0,
            ))
        
        return stats
    
    def _compute_model_stats(
        self,
        results: list[ScenarioResult],
        scenarios: list[EvaluationScenario],
        models: list[str],
    ) -> list[ModelStats]:
        """Compute overall model statistics."""
        stats = []
        
        for model in models:
            model_results = [r for r in results if r.model == model]
            
            if not model_results:
                continue
            
            qualities = [r.quality_score.overall for r in model_results]
            latencies = [r.latency_ms for r in model_results if r.success]
            successes = [r for r in model_results if r.success]
            
            # Per-category scores
            category_scores: dict[str, float] = {}
            for cat in TaskCategory:
                cat_results = [
                    r for r in model_results
                    if any(s.id == r.scenario_id and s.category == cat for s in scenarios)
                ]
                if cat_results:
                    category_scores[cat.value] = statistics.mean(
                        [r.quality_score.overall for r in cat_results]
                    )
            
            # Per-complexity scores
            complexity_scores: dict[str, float] = {}
            for comp in Complexity:
                comp_results = [
                    r for r in model_results
                    if any(s.id == r.scenario_id and s.complexity == comp for s in scenarios)
                ]
                if comp_results:
                    complexity_scores[comp.value] = statistics.mean(
                        [r.quality_score.overall for r in comp_results]
                    )
            
            scenario_ids = set(r.scenario_id for r in model_results)
            
            stats.append(ModelStats(
                model=model,
                total_scenarios=len(scenario_ids),
                total_runs=len(model_results),
                overall_success_rate=len(successes) / len(model_results) if model_results else 0,
                mean_quality=statistics.mean(qualities) if qualities else 0,
                std_quality=statistics.stdev(qualities) if len(qualities) > 1 else 0,
                mean_latency_ms=statistics.mean(latencies) if latencies else 0,
                std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
                mean_tokens=statistics.mean([r.tokens_used for r in model_results]),
                category_scores=category_scores,
                complexity_scores=complexity_scores,
            ))
        
        return stats
