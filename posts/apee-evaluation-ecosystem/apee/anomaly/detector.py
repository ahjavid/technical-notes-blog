"""
APEE Anomaly Detection Core.

Provides statistical and ML-based anomaly detection for evaluation metrics.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Optional, Callable
from collections import deque


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    PERFORMANCE_SPIKE = "performance_spike"
    PERFORMANCE_DROP = "performance_drop"
    QUALITY_DEGRADATION = "quality_degradation"
    LATENCY_ANOMALY = "latency_anomaly"
    COLLABORATION_BREAKDOWN = "collaboration_breakdown"
    CONSENSUS_FAILURE = "consensus_failure"
    COMMUNICATION_OVERLOAD = "communication_overload"
    AGENT_FAILURE = "agent_failure"
    SCORE_INCONSISTENCY = "score_inconsistency"
    PATTERN_DEVIATION = "pattern_deviation"


class AnomalySeverity(str, Enum):
    """Severity levels for detected anomalies."""
    INFO = "info"           # Interesting but not concerning
    WARNING = "warning"     # Should be monitored
    CRITICAL = "critical"   # Requires attention
    EMERGENCY = "emergency" # Immediate action needed


@dataclass
class AnomalyReport:
    """Report of a detected anomaly."""
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    description: str
    metric_name: str
    observed_value: float
    expected_range: tuple[float, float]
    deviation_score: float  # How far from normal (0-1 scale, 1=max deviation)
    timestamp: datetime = field(default_factory=datetime.now)
    context: dict[str, Any] = field(default_factory=dict)
    recommendations: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"[{self.severity.value.upper()}] {self.anomaly_type.value}: "
            f"{self.description} (deviation: {self.deviation_score:.2f})"
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "metric": self.metric_name,
            "observed": self.observed_value,
            "expected_range": list(self.expected_range),
            "deviation": self.deviation_score,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "recommendations": self.recommendations,
        }


class StatisticalWindow:
    """Rolling window for statistical calculations."""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._values: deque[float] = deque(maxlen=window_size)
    
    def add(self, value: float) -> None:
        """Add a value to the window."""
        self._values.append(value)
    
    @property
    def mean(self) -> float:
        """Calculate mean of window."""
        if not self._values:
            return 0.0
        return sum(self._values) / len(self._values)
    
    @property
    def std(self) -> float:
        """Calculate standard deviation of window."""
        if len(self._values) < 2:
            return 0.0
        mean = self.mean
        variance = sum((x - mean) ** 2 for x in self._values) / len(self._values)
        return math.sqrt(variance)
    
    @property
    def min(self) -> float:
        """Get minimum value."""
        return min(self._values) if self._values else 0.0
    
    @property
    def max(self) -> float:
        """Get maximum value."""
        return max(self._values) if self._values else 0.0
    
    def percentile(self, p: float) -> float:
        """Calculate percentile (0-100)."""
        if not self._values:
            return 0.0
        sorted_values = sorted(self._values)
        index = int((p / 100) * (len(sorted_values) - 1))
        return sorted_values[index]
    
    def is_anomaly(self, value: float, sigma: float = 3.0) -> bool:
        """Check if value is anomalous using Z-score method."""
        if len(self._values) < 10:  # Need enough data
            return False
        std = self.std
        if std == 0:
            return False
        z_score = abs(value - self.mean) / std
        return z_score > sigma
    
    def z_score(self, value: float) -> float:
        """Calculate Z-score for a value."""
        std = self.std
        if std == 0:
            return 0.0
        return (value - self.mean) / std


class AnomalyDetector:
    """
    Main anomaly detector for APEE evaluation metrics.
    
    Uses multiple detection methods:
    1. Statistical (Z-score, IQR)
    2. Threshold-based
    3. Pattern matching
    4. Trend analysis
    """
    
    def __init__(
        self,
        window_size: int = 100,
        z_threshold: float = 3.0,
        enable_learning: bool = True
    ):
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.enable_learning = enable_learning
        
        # Windows for different metrics
        self._metric_windows: dict[str, StatisticalWindow] = {}
        
        # Baseline expectations (can be configured)
        self._baselines: dict[str, tuple[float, float]] = {
            # (expected_min, expected_max)
            "overall_apee_score": (4.0, 9.0),
            "l1_individual": (5.0, 10.0),
            "l2_collaborative": (3.0, 9.0),
            "l3_ecosystem": (4.0, 10.0),
            "latency_ms": (100, 10000),
            "quality_score": (0.5, 1.0),
            "success_rate": (0.8, 1.0),
            "collaboration_score": (3.0, 10.0),
            "synthesis_score": (4.0, 10.0),
        }
        
        # Custom detectors
        self._custom_detectors: list[Callable[[dict, str], Optional[AnomalyReport]]] = []
        
        # History of anomalies
        self._anomaly_history: list[AnomalyReport] = []
    
    def _get_window(self, metric: str) -> StatisticalWindow:
        """Get or create window for metric."""
        if metric not in self._metric_windows:
            self._metric_windows[metric] = StatisticalWindow(self.window_size)
        return self._metric_windows[metric]
    
    def set_baseline(self, metric: str, min_val: float, max_val: float) -> None:
        """Set baseline expectations for a metric."""
        self._baselines[metric] = (min_val, max_val)
    
    def add_custom_detector(
        self,
        detector: Callable[[dict, str], Optional[AnomalyReport]]
    ) -> None:
        """Add a custom anomaly detector function."""
        self._custom_detectors.append(detector)
    
    def check_value(
        self,
        metric: str,
        value: float,
        context: Optional[dict] = None
    ) -> Optional[AnomalyReport]:
        """
        Check a single metric value for anomalies.
        
        Args:
            metric: Name of the metric
            value: Observed value
            context: Additional context information
            
        Returns:
            AnomalyReport if anomaly detected, None otherwise
        """
        context = context or {}
        window = self._get_window(metric)
        
        # Check baseline violations
        if metric in self._baselines:
            min_val, max_val = self._baselines[metric]
            if value < min_val * 0.5:  # 50% below minimum
                report = AnomalyReport(
                    anomaly_type=self._get_anomaly_type(metric, "drop"),
                    severity=AnomalySeverity.CRITICAL if value < min_val * 0.25 else AnomalySeverity.WARNING,
                    description=f"{metric} significantly below expected range",
                    metric_name=metric,
                    observed_value=value,
                    expected_range=(min_val, max_val),
                    deviation_score=min(1.0, (min_val - value) / min_val),
                    context=context,
                    recommendations=self._get_recommendations(metric, "low"),
                )
                self._anomaly_history.append(report)
                if self.enable_learning:
                    window.add(value)
                return report
            
            if value > max_val * 1.5:  # 50% above maximum
                report = AnomalyReport(
                    anomaly_type=self._get_anomaly_type(metric, "spike"),
                    severity=AnomalySeverity.WARNING,
                    description=f"{metric} significantly above expected range",
                    metric_name=metric,
                    observed_value=value,
                    expected_range=(min_val, max_val),
                    deviation_score=min(1.0, (value - max_val) / max_val),
                    context=context,
                    recommendations=self._get_recommendations(metric, "high"),
                )
                self._anomaly_history.append(report)
                if self.enable_learning:
                    window.add(value)
                return report
        
        # Statistical anomaly detection
        if len(window._values) >= 10:
            if window.is_anomaly(value, self.z_threshold):
                z = window.z_score(value)
                direction = "spike" if z > 0 else "drop"
                report = AnomalyReport(
                    anomaly_type=self._get_anomaly_type(metric, direction),
                    severity=self._severity_from_z(abs(z)),
                    description=f"Statistical anomaly in {metric} (Z-score: {z:.2f})",
                    metric_name=metric,
                    observed_value=value,
                    expected_range=(window.mean - 2*window.std, window.mean + 2*window.std),
                    deviation_score=min(1.0, abs(z) / (self.z_threshold * 2)),
                    context=context,
                    recommendations=self._get_recommendations(metric, direction),
                )
                self._anomaly_history.append(report)
                if self.enable_learning:
                    window.add(value)
                return report
        
        # Learn from normal values
        if self.enable_learning:
            window.add(value)
        
        return None
    
    def check_evaluation(
        self,
        evaluation_result: dict[str, Any],
        scenario_id: Optional[str] = None
    ) -> list[AnomalyReport]:
        """
        Check complete evaluation result for anomalies.
        
        Args:
            evaluation_result: Full evaluation result dict
            scenario_id: Optional scenario identifier
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        context = {"scenario_id": scenario_id} if scenario_id else {}
        
        # Check overall score
        if "overall_apee_score" in evaluation_result:
            anomaly = self.check_value(
                "overall_apee_score",
                evaluation_result["overall_apee_score"],
                context
            )
            if anomaly:
                anomalies.append(anomaly)
        
        # Check level averages
        for level in ["l1_average", "l2_average", "l3_average"]:
            if level in evaluation_result:
                metric = level.replace("_average", "_individual" if "l1" in level else "_collaborative" if "l2" in level else "_ecosystem")
                anomaly = self.check_value(metric, evaluation_result[level], context)
                if anomaly:
                    anomalies.append(anomaly)
        
        # Check individual agent scores
        if "individual_scores" in evaluation_result:
            for agent_id, scores in evaluation_result["individual_scores"].items():
                agent_context = {**context, "agent_id": agent_id}
                for metric, value in scores.items():
                    anomaly = self.check_value(f"agent_{metric}", value, agent_context)
                    if anomaly:
                        anomalies.append(anomaly)
        
        # Check collaborative scores
        if "collaborative_scores" in evaluation_result:
            for metric, value in evaluation_result["collaborative_scores"].items():
                anomaly = self.check_value(metric, value, context)
                if anomaly:
                    anomalies.append(anomaly)
        
        # Check ecosystem scores
        if "ecosystem_scores" in evaluation_result:
            for metric, value in evaluation_result["ecosystem_scores"].items():
                anomaly = self.check_value(metric, value, context)
                if anomaly:
                    anomalies.append(anomaly)
        
        # Check for judge disagreement
        if "judge_scores" in evaluation_result:
            judge_scores = list(evaluation_result["judge_scores"].values())
            if len(judge_scores) >= 2:
                score_range = max(judge_scores) - min(judge_scores)
                if score_range > 2.0:  # Judges disagree by more than 2 points
                    anomalies.append(AnomalyReport(
                        anomaly_type=AnomalyType.SCORE_INCONSISTENCY,
                        severity=AnomalySeverity.WARNING if score_range < 3 else AnomalySeverity.CRITICAL,
                        description=f"High disagreement between judges (range: {score_range:.1f})",
                        metric_name="judge_agreement",
                        observed_value=score_range,
                        expected_range=(0, 2.0),
                        deviation_score=min(1.0, score_range / 5.0),
                        context=context,
                        recommendations=[
                            "Review individual judge evaluations",
                            "Consider adding more judges",
                            "Check for evaluation prompt issues",
                        ],
                    ))
        
        # Run custom detectors
        for detector in self._custom_detectors:
            custom_anomaly = detector(evaluation_result, scenario_id or "")
            if custom_anomaly:
                anomalies.append(custom_anomaly)
        
        return anomalies
    
    def check_collaboration_trace(
        self,
        trace: Any,  # CollaborativeTrace
    ) -> list[AnomalyReport]:
        """
        Check collaboration trace for behavioral anomalies.
        
        Args:
            trace: CollaborativeTrace object
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        agents = getattr(trace, 'participating_agents', [])
        messages = getattr(trace, 'coordination_messages', [])
        conflicts = getattr(trace, 'conflicts_detected', [])
        pattern = getattr(trace, 'collaboration_pattern', 'unknown')
        
        context = {
            "pattern": pattern,
            "agent_count": len(agents),
            "message_count": len(messages),
        }
        
        # Check for communication breakdown
        if len(agents) > 1 and len(messages) == 0:
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.COLLABORATION_BREAKDOWN,
                severity=AnomalySeverity.CRITICAL,
                description="No communication between agents in multi-agent scenario",
                metric_name="message_count",
                observed_value=0,
                expected_range=(len(agents), len(agents) * 10),
                deviation_score=1.0,
                context=context,
                recommendations=[
                    "Check agent message passing implementation",
                    "Verify coordinator is routing messages",
                    "Review collaboration pattern configuration",
                ],
            ))
        
        # Check for excessive conflicts
        if len(conflicts) > len(agents) * 2:
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.CONSENSUS_FAILURE,
                severity=AnomalySeverity.WARNING,
                description=f"High conflict rate: {len(conflicts)} conflicts with {len(agents)} agents",
                metric_name="conflict_count",
                observed_value=len(conflicts),
                expected_range=(0, len(agents)),
                deviation_score=min(1.0, len(conflicts) / (len(agents) * 5)),
                context=context,
                recommendations=[
                    "Review conflict resolution logic",
                    "Consider using consensus pattern",
                    "Add mediator agent role",
                ],
            ))
        
        # Check for message overload
        messages_per_agent = len(messages) / max(1, len(agents))
        if messages_per_agent > 50:
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.COMMUNICATION_OVERLOAD,
                severity=AnomalySeverity.WARNING,
                description=f"High message volume: {messages_per_agent:.0f} messages per agent",
                metric_name="messages_per_agent",
                observed_value=messages_per_agent,
                expected_range=(1, 20),
                deviation_score=min(1.0, messages_per_agent / 100),
                context=context,
                recommendations=[
                    "Review communication patterns",
                    "Implement message batching",
                    "Check for infinite loops in collaboration",
                ],
            ))
        
        # Check for agent failures
        agent_traces = getattr(trace, 'agent_traces', [])
        for agent_trace in agent_traces:
            agent_id = getattr(agent_trace, 'agent_id', 'unknown')
            output = getattr(agent_trace, 'final_output', '')
            
            if not output or len(output) < 10:
                anomalies.append(AnomalyReport(
                    anomaly_type=AnomalyType.AGENT_FAILURE,
                    severity=AnomalySeverity.CRITICAL,
                    description=f"Agent {agent_id} produced no meaningful output",
                    metric_name="output_length",
                    observed_value=len(output),
                    expected_range=(50, 10000),
                    deviation_score=1.0,
                    context={**context, "agent_id": agent_id},
                    recommendations=[
                        f"Check {agent_id} agent configuration",
                        "Verify model availability",
                        "Review task clarity",
                    ],
                ))
        
        return anomalies
    
    def get_anomaly_summary(self) -> dict[str, Any]:
        """Get summary of all detected anomalies."""
        by_type: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        by_metric: dict[str, int] = {}
        
        for anomaly in self._anomaly_history:
            by_type[anomaly.anomaly_type.value] = by_type.get(anomaly.anomaly_type.value, 0) + 1
            by_severity[anomaly.severity.value] = by_severity.get(anomaly.severity.value, 0) + 1
            by_metric[anomaly.metric_name] = by_metric.get(anomaly.metric_name, 0) + 1
        
        return {
            "total_anomalies": len(self._anomaly_history),
            "by_type": by_type,
            "by_severity": by_severity,
            "by_metric": by_metric,
            "critical_count": by_severity.get("critical", 0) + by_severity.get("emergency", 0),
            "recent": [a.to_dict() for a in self._anomaly_history[-10:]],
        }
    
    def clear_history(self) -> None:
        """Clear anomaly history."""
        self._anomaly_history.clear()
    
    def _get_anomaly_type(self, metric: str, direction: str) -> AnomalyType:
        """Determine anomaly type based on metric and direction."""
        if "latency" in metric.lower():
            return AnomalyType.LATENCY_ANOMALY
        if "quality" in metric.lower():
            return AnomalyType.QUALITY_DEGRADATION
        if "collaboration" in metric.lower():
            return AnomalyType.COLLABORATION_BREAKDOWN
        if direction == "spike":
            return AnomalyType.PERFORMANCE_SPIKE
        return AnomalyType.PERFORMANCE_DROP
    
    def _severity_from_z(self, z_score: float) -> AnomalySeverity:
        """Determine severity based on Z-score magnitude."""
        if z_score > 5:
            return AnomalySeverity.EMERGENCY
        if z_score > 4:
            return AnomalySeverity.CRITICAL
        if z_score > 3:
            return AnomalySeverity.WARNING
        return AnomalySeverity.INFO
    
    def _get_recommendations(self, metric: str, direction: str) -> list[str]:
        """Get recommendations based on metric and direction."""
        recommendations = []
        
        if "quality" in metric.lower() and direction == "low":
            recommendations.extend([
                "Review agent prompts for clarity",
                "Consider using larger models",
                "Check task complexity appropriateness",
            ])
        elif "latency" in metric.lower() and direction == "spike":
            recommendations.extend([
                "Check model server load",
                "Consider caching responses",
                "Review timeout configurations",
            ])
        elif "collaboration" in metric.lower():
            recommendations.extend([
                "Review collaboration pattern",
                "Check agent role assignments",
                "Verify message passing logic",
            ])
        else:
            recommendations.extend([
                f"Investigate {metric} values",
                "Review recent configuration changes",
                "Compare with baseline performance",
            ])
        
        return recommendations
