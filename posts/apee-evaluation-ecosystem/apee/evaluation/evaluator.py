"""Main evaluator that computes metrics from coordinator data."""

import math
from typing import Optional
from uuid import uuid4

from apee.models import AgentResult
from apee.coordination.coordinator import Coordinator
from apee.evaluation.report import (
    EvaluationReport,
    IndividualMetrics,
    CollaborativeMetrics,
    EcosystemMetrics,
    QualityMetrics,
)


class Evaluator:
    """
    Evaluates multi-agent system performance.
    
    Computes three-level metrics:
    1. Individual - Per-agent performance
    2. Collaborative - Inter-agent interaction quality
    3. Ecosystem - System-wide health and efficiency
    """
    
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
    
    def evaluate(
        self, 
        description: str = "",
        include_individual: bool = True,
        include_collaborative: bool = True,
        include_ecosystem: bool = True,
        include_quality: bool = True
    ) -> EvaluationReport:
        """
        Generate comprehensive evaluation report.
        
        Args:
            description: Optional description for this evaluation
            include_*: Flags to include specific metric categories
        """
        report = EvaluationReport(
            evaluation_id=str(uuid4())[:8],
            description=description
        )
        
        if include_individual:
            report.individual = self._compute_individual_metrics()
        
        if include_collaborative:
            report.collaborative = self._compute_collaborative_metrics()
        
        if include_ecosystem:
            report.ecosystem = self._compute_ecosystem_metrics()
        
        if include_quality:
            report.quality = self._compute_quality_metrics()
        
        # Add metadata
        report.execution_patterns = list(set(
            e["pattern"] for e in self.coordinator.execution_history
        ))
        report.model_info = self._get_model_info()
        
        return report
    
    def _compute_individual_metrics(self) -> list[IndividualMetrics]:
        """Compute metrics for each individual agent."""
        metrics = []
        
        for agent_id, agent in self.coordinator.agents.items():
            m = agent.metrics
            metrics.append(IndividualMetrics(
                agent_id=agent_id,
                role=agent.role.value,
                tasks_completed=m.tasks_completed,
                tasks_failed=m.tasks_failed,
                completion_rate=m.completion_rate,
                avg_latency_ms=m.avg_latency_ms,
                avg_quality=m.avg_quality,
                avg_tokens=m.avg_tokens,
                messages_sent=m.messages_sent,
                messages_received=m.messages_received
            ))
        
        return metrics
    
    def _compute_collaborative_metrics(self) -> CollaborativeMetrics:
        """Compute metrics for multi-agent collaboration."""
        total_messages = len(self.coordinator.message_log)
        total_tasks = len(self.coordinator.execution_history)
        
        avg_messages = total_messages / total_tasks if total_tasks > 0 else 0
        
        # Communication efficiency: useful messages / total messages
        # For now, estimate based on message types
        broadcast_count = sum(
            1 for m in self.coordinator.message_log 
            if m.message_type == "broadcast"
        )
        efficiency = 1.0 - (broadcast_count / max(total_messages, 1)) * 0.3
        
        # Synergy score: combined performance vs individual average
        synergy = self._calculate_synergy()
        
        return CollaborativeMetrics(
            total_messages=total_messages,
            avg_messages_per_task=avg_messages,
            communication_efficiency=efficiency,
            synergy_score=synergy,
            conflict_rate=0.0  # TODO: Implement conflict detection
        )
    
    def _compute_ecosystem_metrics(self) -> EcosystemMetrics:
        """Compute system-wide ecosystem metrics."""
        results = self.coordinator.results
        
        if not results:
            return EcosystemMetrics(
                total_tasks=0,
                total_results=0,
                overall_success_rate=0.0,
                avg_system_latency_ms=0.0,
                total_tokens=0,
                agent_count=len(self.coordinator.agents),
                agent_utilization=0.0
            )
        
        successful = [r for r in results if r.success]
        total_tokens = sum(r.tokens_used for r in results)
        avg_latency = sum(r.latency_ms for r in results) / len(results)
        
        # Agent utilization: % of agents that produced results
        active_agents = len(set(r.agent_id for r in results))
        utilization = active_agents / max(len(self.coordinator.agents), 1)
        
        return EcosystemMetrics(
            total_tasks=len(self.coordinator.execution_history),
            total_results=len(results),
            overall_success_rate=len(successful) / len(results),
            avg_system_latency_ms=avg_latency,
            total_tokens=total_tokens,
            agent_count=len(self.coordinator.agents),
            agent_utilization=utilization
        )
    
    def _compute_quality_metrics(self) -> QualityMetrics:
        """Compute quality-related metrics."""
        results = self.coordinator.results
        quality_scores = [r.quality_score for r in results if r.success]
        
        if not quality_scores:
            return QualityMetrics(
                avg_quality=0.0,
                quality_std=0.0,
                min_quality=0.0,
                max_quality=0.0
            )
        
        avg = sum(quality_scores) / len(quality_scores)
        variance = sum((q - avg) ** 2 for q in quality_scores) / len(quality_scores)
        std = math.sqrt(variance)
        
        # Build distribution buckets
        distribution = {"low": 0, "medium": 0, "high": 0, "excellent": 0}
        for q in quality_scores:
            if q < 0.5:
                distribution["low"] += 1
            elif q < 0.7:
                distribution["medium"] += 1
            elif q < 0.9:
                distribution["high"] += 1
            else:
                distribution["excellent"] += 1
        
        return QualityMetrics(
            avg_quality=avg,
            quality_std=std,
            min_quality=min(quality_scores),
            max_quality=max(quality_scores),
            quality_distribution=distribution
        )
    
    def _calculate_synergy(self) -> float:
        """
        Calculate synergy score.
        
        Synergy > 1.0 means collaboration improves performance.
        Synergy < 1.0 means collaboration degrades performance.
        """
        results = self.coordinator.results
        if not results:
            return 1.0
        
        # Get average quality from collaborative results
        successful = [r for r in results if r.success]
        if not successful:
            return 1.0
        
        combined_avg = sum(r.quality_score for r in successful) / len(successful)
        
        # Get individual agent averages
        individual_avgs = [
            agent.metrics.avg_quality 
            for agent in self.coordinator.agents.values()
            if agent.metrics.tasks_completed > 0
        ]
        
        if not individual_avgs:
            return 1.0
        
        individual_avg = sum(individual_avgs) / len(individual_avgs)
        
        # Synergy is ratio of combined to individual performance
        return combined_avg / individual_avg if individual_avg > 0 else 1.0
    
    def _get_model_info(self) -> dict[str, str]:
        """Extract model information from agents."""
        info = {}
        for agent_id, agent in self.coordinator.agents.items():
            if hasattr(agent, 'model'):
                info[agent_id] = agent.model
        return info
    
    def compare_runs(
        self, 
        reports: list[EvaluationReport]
    ) -> dict:
        """
        Compare multiple evaluation runs.
        
        Useful for A/B testing different configurations.
        """
        if not reports:
            return {}
        
        comparisons = {
            "run_count": len(reports),
            "quality_trend": [],
            "latency_trend": [],
            "success_trend": [],
        }
        
        for r in reports:
            if r.quality:
                comparisons["quality_trend"].append(r.quality.avg_quality)
            if r.ecosystem:
                comparisons["latency_trend"].append(r.ecosystem.avg_system_latency_ms)
                comparisons["success_trend"].append(r.ecosystem.overall_success_rate)
        
        # Calculate improvements
        if len(comparisons["quality_trend"]) >= 2:
            q = comparisons["quality_trend"]
            comparisons["quality_improvement"] = (q[-1] - q[0]) / q[0] if q[0] > 0 else 0
        
        return comparisons
