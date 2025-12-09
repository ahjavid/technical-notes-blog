"""
Benchmark analyzer for APEE.

Provides statistical analysis and model comparisons.
"""

import statistics
from dataclasses import dataclass, field
from typing import Optional

from apee.benchmarks.runner import BenchmarkResult, ModelStats, ScenarioResult
from apee.benchmarks.datasets import TaskCategory, Complexity


@dataclass
class StatisticalAnalysis:
    """Statistical analysis of benchmark results."""
    
    # Sample statistics
    sample_size: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    
    # Confidence interval (95%)
    ci_lower: float
    ci_upper: float
    
    # Distribution
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    iqr: float  # Interquartile range


@dataclass
class ModelComparison:
    """Comparison between two models."""
    model_a: str
    model_b: str
    
    # Quality comparison
    quality_diff: float  # model_a - model_b
    quality_winner: str
    
    # Speed comparison
    latency_diff_ms: float
    speed_winner: str
    
    # Efficiency comparison (quality/latency)
    efficiency_diff: float
    efficiency_winner: str
    
    # Per-category comparison
    category_winners: dict[str, str] = field(default_factory=dict)
    
    # Statistical significance
    is_significant: bool = False  # p < 0.05
    effect_size: float = 0.0  # Cohen's d


class BenchmarkAnalyzer:
    """
    Analyzes benchmark results with statistical rigor.
    
    Provides:
    - Descriptive statistics
    - Model comparisons
    - Category/complexity breakdowns
    - Significance testing
    """
    
    def __init__(self, result: BenchmarkResult):
        self.result = result
    
    def get_model_analysis(self, model: str) -> Optional[StatisticalAnalysis]:
        """Get statistical analysis for a specific model."""
        model_results = [r for r in self.result.scenario_results if r.model == model]
        
        if not model_results:
            return None
        
        qualities = [r.quality_score.overall for r in model_results]
        return self._compute_stats(qualities)
    
    def compare_models(self, model_a: str, model_b: str) -> Optional[ModelComparison]:
        """Compare two models statistically."""
        results_a = [r for r in self.result.scenario_results if r.model == model_a]
        results_b = [r for r in self.result.scenario_results if r.model == model_b]
        
        if not results_a or not results_b:
            return None
        
        stats_a = next((s for s in self.result.model_stats if s.model == model_a), None)
        stats_b = next((s for s in self.result.model_stats if s.model == model_b), None)
        
        if not stats_a or not stats_b:
            return None
        
        # Quality comparison
        quality_diff = stats_a.mean_quality - stats_b.mean_quality
        quality_winner = model_a if quality_diff > 0 else model_b if quality_diff < 0 else "tie"
        
        # Speed comparison
        latency_diff = stats_a.mean_latency_ms - stats_b.mean_latency_ms
        speed_winner = model_b if latency_diff > 0 else model_a if latency_diff < 0 else "tie"
        
        # Efficiency comparison
        eff_a = stats_a.mean_quality / max(stats_a.mean_latency_ms, 1) * 1000
        eff_b = stats_b.mean_quality / max(stats_b.mean_latency_ms, 1) * 1000
        efficiency_diff = eff_a - eff_b
        efficiency_winner = model_a if efficiency_diff > 0 else model_b if efficiency_diff < 0 else "tie"
        
        # Per-category comparison
        category_winners = {}
        all_categories = set(stats_a.category_scores.keys()) | set(stats_b.category_scores.keys())
        for cat in all_categories:
            score_a = stats_a.category_scores.get(cat, 0)
            score_b = stats_b.category_scores.get(cat, 0)
            category_winners[cat] = model_a if score_a > score_b else model_b if score_b > score_a else "tie"
        
        # Effect size (Cohen's d)
        qualities_a = [r.quality_score.overall for r in results_a]
        qualities_b = [r.quality_score.overall for r in results_b]
        effect_size = self._cohens_d(qualities_a, qualities_b)
        
        # Significance (simple threshold based on effect size and sample size)
        is_significant = abs(effect_size) > 0.2 and len(qualities_a) >= 10 and len(qualities_b) >= 10
        
        return ModelComparison(
            model_a=model_a,
            model_b=model_b,
            quality_diff=quality_diff,
            quality_winner=quality_winner,
            latency_diff_ms=latency_diff,
            speed_winner=speed_winner,
            efficiency_diff=efficiency_diff,
            efficiency_winner=efficiency_winner,
            category_winners=category_winners,
            is_significant=is_significant,
            effect_size=effect_size,
        )
    
    def get_category_breakdown(self, model: str) -> dict[str, StatisticalAnalysis]:
        """Get per-category analysis for a model."""
        breakdowns = {}
        
        for cat in TaskCategory:
            cat_results = [
                r for r in self.result.scenario_results
                if r.model == model and self._result_matches_category(r, cat)
            ]
            
            if cat_results:
                qualities = [r.quality_score.overall for r in cat_results]
                breakdowns[cat.value] = self._compute_stats(qualities)
        
        return breakdowns
    
    def get_complexity_breakdown(self, model: str) -> dict[str, StatisticalAnalysis]:
        """Get per-complexity analysis for a model."""
        breakdowns = {}
        
        for comp in Complexity:
            comp_results = [
                r for r in self.result.scenario_results
                if r.model == model and self._result_matches_complexity(r, comp)
            ]
            
            if comp_results:
                qualities = [r.quality_score.overall for r in comp_results]
                breakdowns[comp.value] = self._compute_stats(qualities)
        
        return breakdowns
    
    def get_worst_scenarios(self, model: str, n: int = 5) -> list[tuple[str, float]]:
        """Get the N worst-performing scenarios for a model."""
        model_results = [r for r in self.result.scenario_results if r.model == model]
        
        # Group by scenario and average
        scenario_scores: dict[str, list[float]] = {}
        for r in model_results:
            if r.scenario_id not in scenario_scores:
                scenario_scores[r.scenario_id] = []
            scenario_scores[r.scenario_id].append(r.quality_score.overall)
        
        avg_scores = [
            (sid, statistics.mean(scores))
            for sid, scores in scenario_scores.items()
        ]
        
        return sorted(avg_scores, key=lambda x: x[1])[:n]
    
    def get_best_scenarios(self, model: str, n: int = 5) -> list[tuple[str, float]]:
        """Get the N best-performing scenarios for a model."""
        worst = self.get_worst_scenarios(model, n=1000)
        return list(reversed(worst))[:n]
    
    def generate_report(self) -> str:
        """Generate a comprehensive text report."""
        lines = []
        
        lines.append("=" * 70)
        lines.append("APEE BENCHMARK REPORT")
        lines.append("=" * 70)
        lines.append("")
        
        # Summary
        lines.append("## Summary")
        lines.append(f"- Models evaluated: {len(self.result.model_stats)}")
        lines.append(f"- Total evaluations: {len(self.result.scenario_results)}")
        lines.append(f"- Duration: {self.result.duration_seconds:.1f}s")
        lines.append("")
        
        # Rankings
        lines.append("## Rankings")
        lines.append("")
        lines.append("### Quality (highest to lowest)")
        for i, model in enumerate(self.result.quality_ranking, 1):
            stats = next(s for s in self.result.model_stats if s.model == model)
            lines.append(f"  {i}. {model}: {stats.mean_quality:.3f} (±{stats.std_quality:.3f})")
        
        lines.append("")
        lines.append("### Speed (fastest to slowest)")
        for i, model in enumerate(self.result.speed_ranking, 1):
            stats = next(s for s in self.result.model_stats if s.model == model)
            lines.append(f"  {i}. {model}: {stats.mean_latency_ms:.0f}ms (±{stats.std_latency_ms:.0f}ms)")
        
        lines.append("")
        lines.append("### Efficiency (quality/latency)")
        for i, model in enumerate(self.result.efficiency_ranking, 1):
            stats = next(s for s in self.result.model_stats if s.model == model)
            eff = stats.mean_quality / max(stats.mean_latency_ms, 1) * 1000
            lines.append(f"  {i}. {model}: {eff:.4f}")
        
        # Per-model details
        lines.append("")
        lines.append("## Model Details")
        
        for stats in self.result.model_stats:
            lines.append("")
            lines.append(f"### {stats.model}")
            lines.append(f"- Success Rate: {stats.overall_success_rate:.1%}")
            lines.append(f"- Mean Quality: {stats.mean_quality:.3f}")
            lines.append(f"- Mean Latency: {stats.mean_latency_ms:.0f}ms")
            lines.append(f"- Mean Tokens: {stats.mean_tokens:.0f}")
            
            if stats.category_scores:
                lines.append("")
                lines.append("  Category Scores:")
                for cat, score in sorted(stats.category_scores.items()):
                    lines.append(f"    - {cat}: {score:.3f}")
            
            if stats.complexity_scores:
                lines.append("")
                lines.append("  Complexity Scores:")
                for comp, score in sorted(stats.complexity_scores.items()):
                    lines.append(f"    - {comp}: {score:.3f}")
        
        # Comparisons
        if len(self.result.model_stats) >= 2:
            lines.append("")
            lines.append("## Head-to-Head Comparisons")
            
            models = [s.model for s in self.result.model_stats]
            for i, model_a in enumerate(models):
                for model_b in models[i+1:]:
                    comparison = self.compare_models(model_a, model_b)
                    if comparison:
                        lines.append("")
                        lines.append(f"### {model_a} vs {model_b}")
                        lines.append(f"- Quality Winner: {comparison.quality_winner} ({comparison.quality_diff:+.3f})")
                        lines.append(f"- Speed Winner: {comparison.speed_winner} ({comparison.latency_diff_ms:+.0f}ms)")
                        lines.append(f"- Efficiency Winner: {comparison.efficiency_winner}")
                        lines.append(f"- Effect Size: {comparison.effect_size:.3f}")
                        lines.append(f"- Statistically Significant: {comparison.is_significant}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def _compute_stats(self, values: list[float]) -> StatisticalAnalysis:
        """Compute statistical analysis for a list of values."""
        n = len(values)
        
        if n == 0:
            return StatisticalAnalysis(
                sample_size=0, mean=0, median=0, std_dev=0,
                min_value=0, max_value=0,
                ci_lower=0, ci_upper=0,
                q1=0, q3=0, iqr=0,
            )
        
        mean = statistics.mean(values)
        median = statistics.median(values)
        std_dev = statistics.stdev(values) if n > 1 else 0
        
        sorted_values = sorted(values)
        min_val = sorted_values[0]
        max_val = sorted_values[-1]
        
        # Quartiles
        q1 = sorted_values[n // 4] if n >= 4 else min_val
        q3 = sorted_values[3 * n // 4] if n >= 4 else max_val
        iqr = q3 - q1
        
        # 95% confidence interval (assuming normal distribution)
        import math
        se = std_dev / math.sqrt(n) if n > 0 else 0
        ci_lower = mean - 1.96 * se
        ci_upper = mean + 1.96 * se
        
        return StatisticalAnalysis(
            sample_size=n,
            mean=mean,
            median=median,
            std_dev=std_dev,
            min_value=min_val,
            max_value=max_val,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            q1=q1,
            q3=q3,
            iqr=iqr,
        )
    
    def _cohens_d(self, group_a: list[float], group_b: list[float]) -> float:
        """Calculate Cohen's d effect size."""
        if not group_a or not group_b:
            return 0.0
        
        mean_a = statistics.mean(group_a)
        mean_b = statistics.mean(group_b)
        
        var_a = statistics.variance(group_a) if len(group_a) > 1 else 0
        var_b = statistics.variance(group_b) if len(group_b) > 1 else 0
        
        # Pooled standard deviation
        n_a, n_b = len(group_a), len(group_b)
        pooled_std = ((var_a * (n_a - 1) + var_b * (n_b - 1)) / (n_a + n_b - 2)) ** 0.5
        
        if pooled_std == 0:
            return 0.0
        
        return (mean_a - mean_b) / pooled_std
    
    def _result_matches_category(self, result: ScenarioResult, category: TaskCategory) -> bool:
        """Check if a result matches a category."""
        # Get scenario from dataset to check category
        from apee.benchmarks.datasets import BenchmarkDataset
        dataset = BenchmarkDataset()
        scenario = dataset.get_by_id(result.scenario_id)
        return scenario is not None and scenario.category == category
    
    def _result_matches_complexity(self, result: ScenarioResult, complexity: Complexity) -> bool:
        """Check if a result matches a complexity level."""
        from apee.benchmarks.datasets import BenchmarkDataset
        dataset = BenchmarkDataset()
        scenario = dataset.get_by_id(result.scenario_id)
        return scenario is not None and scenario.complexity == complexity
