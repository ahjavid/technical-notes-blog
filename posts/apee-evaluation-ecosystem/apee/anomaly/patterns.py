"""
APEE Pattern Analyzers.

Specialized analyzers for different types of patterns in evaluation data.
"""

from dataclasses import dataclass
from typing import Any, Optional
from collections import defaultdict
import math

from apee.anomaly.detector import (
    AnomalyType,
    AnomalySeverity,
    AnomalyReport,
    StatisticalWindow,
)


@dataclass
class TrendAnalysis:
    """Result of trend analysis."""
    direction: str  # "increasing", "decreasing", "stable", "volatile"
    slope: float
    confidence: float
    data_points: int


class PerformancePatternAnalyzer:
    """
    Analyzes performance patterns across evaluations.
    
    Detects:
    - Degradation trends
    - Performance regression
    - Cyclical patterns
    - Sudden changes
    """
    
    def __init__(self, min_samples: int = 5):
        self.min_samples = min_samples
        self._history: dict[str, list[tuple[float, float]]] = defaultdict(list)  # metric -> [(timestamp, value)]
    
    def record(self, metric: str, value: float, timestamp: Optional[float] = None) -> None:
        """Record a metric value with timestamp."""
        import time
        ts = timestamp or time.time()
        self._history[metric].append((ts, value))
        
        # Keep only last 1000 points per metric
        if len(self._history[metric]) > 1000:
            self._history[metric] = self._history[metric][-1000:]
    
    def analyze_trend(self, metric: str) -> Optional[TrendAnalysis]:
        """
        Analyze trend for a metric.
        
        Returns:
            TrendAnalysis or None if insufficient data
        """
        data = self._history.get(metric, [])
        if len(data) < self.min_samples:
            return None
        
        # Simple linear regression
        n = len(data)
        x_vals = list(range(n))
        y_vals = [v for _, v in data]
        
        x_mean = sum(x_vals) / n
        y_mean = sum(y_vals) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_vals, y_vals))
        denominator = sum((x - x_mean) ** 2 for x in x_vals)
        
        slope = numerator / denominator if denominator != 0 else 0
        
        # Calculate R-squared for confidence
        y_pred = [y_mean + slope * (x - x_mean) for x in x_vals]
        ss_res = sum((y - yp) ** 2 for y, yp in zip(y_vals, y_pred))
        ss_tot = sum((y - y_mean) ** 2 for y in y_vals)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # Determine direction
        if abs(slope) < 0.01:
            direction = "stable"
        elif slope > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        
        # Check for volatility
        std = math.sqrt(sum((y - y_mean) ** 2 for y in y_vals) / n)
        if std > y_mean * 0.3 and r_squared < 0.5:
            direction = "volatile"
        
        return TrendAnalysis(
            direction=direction,
            slope=slope,
            confidence=max(0, r_squared),
            data_points=n
        )
    
    def detect_anomalies(self, metric: str) -> list[AnomalyReport]:
        """Detect performance anomalies for a metric."""
        anomalies = []
        trend = self.analyze_trend(metric)
        
        if not trend:
            return anomalies
        
        # Detect degradation trend
        if trend.direction == "decreasing" and trend.confidence > 0.7:
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.QUALITY_DEGRADATION,
                severity=AnomalySeverity.WARNING,
                description=f"Consistent performance degradation detected in {metric}",
                metric_name=metric,
                observed_value=trend.slope,
                expected_range=(-0.01, 0.01),
                deviation_score=min(1.0, abs(trend.slope) * 10),
                context={
                    "trend_direction": trend.direction,
                    "confidence": trend.confidence,
                    "data_points": trend.data_points,
                },
                recommendations=[
                    "Review recent changes to models or prompts",
                    "Check for data quality issues",
                    "Consider reverting recent deployments",
                ],
            ))
        
        # Detect high volatility
        if trend.direction == "volatile":
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.PATTERN_DEVIATION,
                severity=AnomalySeverity.INFO,
                description=f"High volatility in {metric} values",
                metric_name=metric,
                observed_value=1 - trend.confidence,
                expected_range=(0, 0.5),
                deviation_score=min(1.0, (1 - trend.confidence)),
                context={
                    "trend_direction": trend.direction,
                    "confidence": trend.confidence,
                },
                recommendations=[
                    "Investigate sources of variation",
                    "Consider stabilizing factors in evaluation",
                ],
            ))
        
        return anomalies
    
    def get_summary(self, metric: str) -> dict[str, Any]:
        """Get summary statistics for a metric."""
        data = self._history.get(metric, [])
        if not data:
            return {"error": "No data for metric"}
        
        values = [v for _, v in data]
        trend = self.analyze_trend(metric)
        
        return {
            "metric": metric,
            "data_points": len(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "latest": values[-1] if values else None,
            "trend": trend.direction if trend else None,
            "trend_confidence": trend.confidence if trend else None,
        }


class CollaborationPatternAnalyzer:
    """
    Analyzes multi-agent collaboration patterns.
    
    Detects:
    - Communication imbalances
    - Role confusion
    - Coordination failures
    - Emergent patterns
    """
    
    def __init__(self):
        self._interaction_matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._role_performance: dict[str, list[float]] = defaultdict(list)
    
    def record_interaction(self, sender: str, receiver: str) -> None:
        """Record an interaction between agents."""
        self._interaction_matrix[sender][receiver] += 1
    
    def record_role_performance(self, role: str, score: float) -> None:
        """Record performance score for a role."""
        self._role_performance[role].append(score)
    
    def analyze_communication_balance(self, agents: list[str]) -> dict[str, Any]:
        """
        Analyze communication balance among agents.
        
        Returns analysis including imbalance metrics.
        """
        if not agents:
            return {"error": "No agents provided"}
        
        # Calculate messages sent/received per agent
        sent_counts = {a: sum(self._interaction_matrix[a].values()) for a in agents}
        received_counts = {a: sum(self._interaction_matrix[s].get(a, 0) for s in agents) for a in agents}
        
        total_messages = sum(sent_counts.values())
        if total_messages == 0:
            return {
                "total_messages": 0,
                "balance_score": 1.0,
                "dominant_agent": None,
                "silent_agents": agents,
            }
        
        # Calculate Gini coefficient for imbalance
        sent_values = sorted(sent_counts.values())
        n = len(sent_values)
        cumulative = sum((i + 1) * v for i, v in enumerate(sent_values))
        gini = (2 * cumulative) / (n * sum(sent_values)) - (n + 1) / n if sum(sent_values) > 0 else 0
        
        # Find dominant and silent agents
        max_sent = max(sent_counts.values())
        dominant = [a for a, c in sent_counts.items() if c == max_sent]
        silent = [a for a, c in sent_counts.items() if c == 0]
        
        return {
            "total_messages": total_messages,
            "balance_score": 1 - gini,  # 1 = perfect balance, 0 = complete imbalance
            "sent_per_agent": sent_counts,
            "received_per_agent": received_counts,
            "dominant_agents": dominant,
            "silent_agents": silent,
            "gini_coefficient": gini,
        }
    
    def detect_anomalies(self, agents: list[str], pattern: str) -> list[AnomalyReport]:
        """Detect collaboration anomalies."""
        anomalies = []
        balance = self.analyze_communication_balance(agents)
        
        # Check for communication imbalance
        if balance.get("balance_score", 1.0) < 0.3:
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.COLLABORATION_BREAKDOWN,
                severity=AnomalySeverity.WARNING,
                description="Severe communication imbalance between agents",
                metric_name="communication_balance",
                observed_value=balance["balance_score"],
                expected_range=(0.5, 1.0),
                deviation_score=1 - balance["balance_score"],
                context={
                    "dominant_agents": balance.get("dominant_agents", []),
                    "silent_agents": balance.get("silent_agents", []),
                    "pattern": pattern,
                },
                recommendations=[
                    "Review agent role assignments",
                    "Check if silent agents are receiving tasks",
                    "Consider rebalancing workload",
                ],
            ))
        
        # Check for silent agents in collaborative patterns
        if balance.get("silent_agents") and pattern not in ["parallel"]:
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.AGENT_FAILURE,
                severity=AnomalySeverity.WARNING,
                description=f"Silent agents detected: {balance['silent_agents']}",
                metric_name="agent_activity",
                observed_value=len(balance.get("silent_agents", [])),
                expected_range=(0, 0),
                deviation_score=len(balance.get("silent_agents", [])) / len(agents),
                context={
                    "silent_agents": balance.get("silent_agents", []),
                    "pattern": pattern,
                },
                recommendations=[
                    "Verify agent connectivity",
                    "Check task routing logic",
                    "Review agent capabilities for assigned tasks",
                ],
            ))
        
        return anomalies
    
    def analyze_role_effectiveness(self) -> dict[str, dict[str, float]]:
        """Analyze effectiveness of each role."""
        effectiveness = {}
        
        for role, scores in self._role_performance.items():
            if scores:
                effectiveness[role] = {
                    "mean": sum(scores) / len(scores),
                    "min": min(scores),
                    "max": max(scores),
                    "count": len(scores),
                    "std": math.sqrt(sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) if len(scores) > 1 else 0,
                }
        
        return effectiveness


class QualityPatternAnalyzer:
    """
    Analyzes quality patterns across evaluations.
    
    Detects:
    - Quality score distributions
    - Consistency issues
    - Outlier patterns
    """
    
    def __init__(self):
        self._quality_history: dict[str, list[float]] = defaultdict(list)
        self._window = StatisticalWindow(window_size=100)
    
    def record(self, category: str, score: float) -> None:
        """Record a quality score for a category."""
        self._quality_history[category].append(score)
        self._window.add(score)
    
    def analyze_distribution(self, category: str) -> dict[str, Any]:
        """Analyze quality score distribution for a category."""
        scores = self._quality_history.get(category, [])
        if not scores:
            return {"error": "No data for category"}
        
        n = len(scores)
        mean = sum(scores) / n
        
        # Calculate standard deviation
        variance = sum((s - mean) ** 2 for s in scores) / n
        std = math.sqrt(variance)
        
        # Calculate percentiles
        sorted_scores = sorted(scores)
        p25 = sorted_scores[int(n * 0.25)] if n >= 4 else sorted_scores[0]
        p50 = sorted_scores[int(n * 0.50)]
        p75 = sorted_scores[int(n * 0.75)] if n >= 4 else sorted_scores[-1]
        
        return {
            "category": category,
            "count": n,
            "mean": mean,
            "std": std,
            "min": min(scores),
            "max": max(scores),
            "p25": p25,
            "p50": p50,
            "p75": p75,
            "iqr": p75 - p25,
        }
    
    def detect_outliers(self, category: str, method: str = "iqr") -> list[tuple[int, float]]:
        """
        Detect outlier scores.
        
        Args:
            category: Category to analyze
            method: "iqr" or "zscore"
            
        Returns:
            List of (index, value) tuples for outliers
        """
        scores = self._quality_history.get(category, [])
        if len(scores) < 5:
            return []
        
        outliers = []
        
        if method == "iqr":
            sorted_scores = sorted(scores)
            n = len(sorted_scores)
            q1 = sorted_scores[int(n * 0.25)]
            q3 = sorted_scores[int(n * 0.75)]
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            
            for i, s in enumerate(scores):
                if s < lower or s > upper:
                    outliers.append((i, s))
        
        elif method == "zscore":
            mean = sum(scores) / len(scores)
            std = math.sqrt(sum((s - mean) ** 2 for s in scores) / len(scores))
            
            if std > 0:
                for i, s in enumerate(scores):
                    z = abs(s - mean) / std
                    if z > 3:
                        outliers.append((i, s))
        
        return outliers
    
    def detect_anomalies(self, category: str) -> list[AnomalyReport]:
        """Detect quality-related anomalies."""
        anomalies = []
        
        dist = self.analyze_distribution(category)
        if "error" in dist:
            return anomalies
        
        # Check for high variance
        cv = dist["std"] / dist["mean"] if dist["mean"] > 0 else 0  # Coefficient of variation
        if cv > 0.5:
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.QUALITY_DEGRADATION,
                severity=AnomalySeverity.INFO,
                description=f"High quality variance in {category}",
                metric_name=f"{category}_variance",
                observed_value=cv,
                expected_range=(0, 0.3),
                deviation_score=min(1.0, cv),
                context=dist,
                recommendations=[
                    "Investigate sources of quality variation",
                    "Review evaluation consistency",
                ],
            ))
        
        # Check for low mean quality
        if dist["mean"] < 5.0:  # Below midpoint on 0-10 scale
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.QUALITY_DEGRADATION,
                severity=AnomalySeverity.WARNING,
                description=f"Below-average quality scores in {category}",
                metric_name=f"{category}_mean",
                observed_value=dist["mean"],
                expected_range=(5.0, 10.0),
                deviation_score=(5.0 - dist["mean"]) / 5.0,
                context=dist,
                recommendations=[
                    "Review agent configurations",
                    "Consider model upgrades",
                    "Analyze low-scoring cases",
                ],
            ))
        
        # Check for outliers
        outliers = self.detect_outliers(category)
        if len(outliers) > dist["count"] * 0.1:  # More than 10% outliers
            anomalies.append(AnomalyReport(
                anomaly_type=AnomalyType.SCORE_INCONSISTENCY,
                severity=AnomalySeverity.INFO,
                description=f"High outlier rate in {category}",
                metric_name=f"{category}_outliers",
                observed_value=len(outliers),
                expected_range=(0, dist["count"] * 0.05),
                deviation_score=min(1.0, len(outliers) / dist["count"]),
                context={
                    "outlier_count": len(outliers),
                    "total_count": dist["count"],
                    "outlier_values": [v for _, v in outliers[:5]],  # First 5
                },
                recommendations=[
                    "Review outlier cases individually",
                    "Check for edge cases in evaluation",
                ],
            ))
        
        return anomalies
