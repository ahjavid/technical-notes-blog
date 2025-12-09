"""
APEE Adaptive Engine.

Core component that dynamically adjusts evaluation criteria based on observed patterns.
Implements Section 3.2.4 of the APEE framework.
"""

import statistics
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

from apee.models import AgentResult


class PatternType(str, Enum):
    """Types of detected patterns."""
    EMERGENT_COLLABORATION = "emergent_collaboration"
    CONSISTENT_FAILURE = "consistent_failure"
    LATENCY_SPIKE = "latency_spike"
    QUALITY_DEGRADATION = "quality_degradation"
    SYNERGY_BOOST = "synergy_boost"
    CONFLICT_ESCALATION = "conflict_escalation"
    CONVERGENT_BEHAVIOR = "convergent_behavior"
    DIVERGENT_BEHAVIOR = "divergent_behavior"


class AnomalyType(str, Enum):
    """Types of detected anomalies."""
    OUTLIER_LATENCY = "outlier_latency"
    OUTLIER_QUALITY = "outlier_quality"
    UNEXPECTED_FAILURE = "unexpected_failure"
    UNUSUAL_TOKEN_USAGE = "unusual_token_usage"
    COMMUNICATION_BURST = "communication_burst"


@dataclass
class DetectedPattern:
    """A detected behavioral pattern."""
    pattern_type: PatternType
    confidence: float  # 0-1
    affected_agents: list[str]
    description: str
    observation_count: int = 1


@dataclass
class DetectedAnomaly:
    """A detected anomaly."""
    anomaly_type: AnomalyType
    severity: float  # 0-1
    agent_id: Optional[str]
    value: float
    expected_range: tuple[float, float]
    description: str


@dataclass
class EvaluationCriteria:
    """Adaptive evaluation criteria with adjustable weights."""
    
    # Level 1: Individual weights
    completion_weight: float = 0.25
    quality_weight: float = 0.35
    latency_weight: float = 0.20
    efficiency_weight: float = 0.20
    
    # Level 2: Collaborative weights
    coordination_weight: float = 0.30
    synergy_weight: float = 0.30
    conflict_resolution_weight: float = 0.20
    handoff_weight: float = 0.20
    
    # Level 3: Ecosystem weights
    stability_weight: float = 0.25
    scalability_weight: float = 0.25
    adaptability_weight: float = 0.25
    resource_weight: float = 0.25
    
    # Focus areas (dynamically added)
    focus_areas: list[str] = field(default_factory=list)
    
    # Thresholds
    quality_threshold: float = 0.7
    latency_threshold_ms: float = 5000
    conflict_threshold: float = 0.3
    
    def normalize_weights(self):
        """Ensure weights sum to 1.0 within each level."""
        # Level 1
        l1_total = self.completion_weight + self.quality_weight + self.latency_weight + self.efficiency_weight
        if l1_total > 0:
            self.completion_weight /= l1_total
            self.quality_weight /= l1_total
            self.latency_weight /= l1_total
            self.efficiency_weight /= l1_total
        
        # Level 2
        l2_total = self.coordination_weight + self.synergy_weight + self.conflict_resolution_weight + self.handoff_weight
        if l2_total > 0:
            self.coordination_weight /= l2_total
            self.synergy_weight /= l2_total
            self.conflict_resolution_weight /= l2_total
            self.handoff_weight /= l2_total
        
        # Level 3
        l3_total = self.stability_weight + self.scalability_weight + self.adaptability_weight + self.resource_weight
        if l3_total > 0:
            self.stability_weight /= l3_total
            self.scalability_weight /= l3_total
            self.adaptability_weight /= l3_total
            self.resource_weight /= l3_total
    
    def add_focus_area(self, area: str):
        """Add a focus area for special attention."""
        if area not in self.focus_areas:
            self.focus_areas.append(area)


@dataclass 
class Observation:
    """An observation from the evaluation process."""
    agent_id: str
    task_id: str
    success: bool
    quality_score: float
    latency_ms: float
    tokens_used: int
    messages_sent: int = 0
    messages_received: int = 0
    conflicts: int = 0
    handoffs: int = 0


class PatternDetector:
    """
    Detects behavioral patterns in multi-agent observations.
    
    Implements pattern recognition for:
    - Emergent collaboration behaviors
    - Consistent performance trends
    - Agent interaction patterns
    """
    
    def __init__(self, sensitivity: float = 0.5):
        self.sensitivity = sensitivity
        self._history: list[Observation] = []
    
    def detect(self, observations: list[Observation]) -> list[DetectedPattern]:
        """Detect patterns in observations."""
        self._history.extend(observations)
        patterns = []
        
        # Detect emergent collaboration
        collab_pattern = self._detect_emergent_collaboration(observations)
        if collab_pattern:
            patterns.append(collab_pattern)
        
        # Detect consistent failure
        failure_pattern = self._detect_consistent_failure(observations)
        if failure_pattern:
            patterns.append(failure_pattern)
        
        # Detect quality trends
        quality_pattern = self._detect_quality_trend(observations)
        if quality_pattern:
            patterns.append(quality_pattern)
        
        # Detect convergent/divergent behavior
        behavior_pattern = self._detect_behavior_convergence(observations)
        if behavior_pattern:
            patterns.append(behavior_pattern)
        
        return patterns
    
    def _detect_emergent_collaboration(self, obs: list[Observation]) -> Optional[DetectedPattern]:
        """Detect if agents are exhibiting emergent collaborative behavior."""
        if len(obs) < 3:
            return None
        
        # Check for high message activity + high quality
        avg_messages = statistics.mean([o.messages_sent + o.messages_received for o in obs])
        avg_quality = statistics.mean([o.quality_score for o in obs])
        
        if avg_messages > 2 and avg_quality > 0.8:
            return DetectedPattern(
                pattern_type=PatternType.EMERGENT_COLLABORATION,
                confidence=min(1.0, avg_quality * (avg_messages / 5)),
                affected_agents=list(set(o.agent_id for o in obs)),
                description=f"High collaboration ({avg_messages:.1f} msgs) with quality ({avg_quality:.2f})",
            )
        return None
    
    def _detect_consistent_failure(self, obs: list[Observation]) -> Optional[DetectedPattern]:
        """Detect if an agent consistently fails."""
        agent_failures: dict[str, int] = {}
        agent_total: dict[str, int] = {}
        
        for o in obs:
            agent_total[o.agent_id] = agent_total.get(o.agent_id, 0) + 1
            if not o.success:
                agent_failures[o.agent_id] = agent_failures.get(o.agent_id, 0) + 1
        
        failing_agents = []
        for agent_id, failures in agent_failures.items():
            total = agent_total[agent_id]
            if total >= 3 and failures / total > 0.5:
                failing_agents.append(agent_id)
        
        if failing_agents:
            return DetectedPattern(
                pattern_type=PatternType.CONSISTENT_FAILURE,
                confidence=0.9,
                affected_agents=failing_agents,
                description=f"Agents with >50% failure rate: {failing_agents}",
                observation_count=len(obs),
            )
        return None
    
    def _detect_quality_trend(self, obs: list[Observation]) -> Optional[DetectedPattern]:
        """Detect quality degradation or improvement trend."""
        if len(obs) < 5:
            return None
        
        qualities = [o.quality_score for o in obs]
        first_half = statistics.mean(qualities[:len(qualities)//2])
        second_half = statistics.mean(qualities[len(qualities)//2:])
        
        diff = second_half - first_half
        
        if diff < -0.1:
            return DetectedPattern(
                pattern_type=PatternType.QUALITY_DEGRADATION,
                confidence=min(1.0, abs(diff) * 2),
                affected_agents=list(set(o.agent_id for o in obs)),
                description=f"Quality dropped from {first_half:.2f} to {second_half:.2f}",
            )
        elif diff > 0.1:
            return DetectedPattern(
                pattern_type=PatternType.SYNERGY_BOOST,
                confidence=min(1.0, diff * 2),
                affected_agents=list(set(o.agent_id for o in obs)),
                description=f"Quality improved from {first_half:.2f} to {second_half:.2f}",
            )
        return None
    
    def _detect_behavior_convergence(self, obs: list[Observation]) -> Optional[DetectedPattern]:
        """Detect if agents are converging or diverging in behavior."""
        if len(obs) < 4:
            return None
        
        # Check variance in quality scores
        qualities = [o.quality_score for o in obs]
        variance = statistics.variance(qualities) if len(qualities) > 1 else 0
        
        if variance < 0.01:  # Very low variance = convergent
            return DetectedPattern(
                pattern_type=PatternType.CONVERGENT_BEHAVIOR,
                confidence=1.0 - variance * 10,
                affected_agents=list(set(o.agent_id for o in obs)),
                description=f"Agents converging on similar quality ({statistics.mean(qualities):.2f}±{variance:.3f})",
            )
        elif variance > 0.1:  # High variance = divergent
            return DetectedPattern(
                pattern_type=PatternType.DIVERGENT_BEHAVIOR,
                confidence=min(1.0, variance * 5),
                affected_agents=list(set(o.agent_id for o in obs)),
                description=f"Agents diverging significantly (variance={variance:.3f})",
            )
        return None


class AnomalyDetector:
    """
    Detects anomalies in agent behavior.
    
    Uses statistical methods to identify:
    - Outlier performance
    - Unexpected failures
    - Unusual resource usage
    """
    
    def __init__(self, z_threshold: float = 2.0):
        self.z_threshold = z_threshold
        self._baseline_latency: list[float] = []
        self._baseline_quality: list[float] = []
        self._baseline_tokens: list[float] = []
    
    def update_baseline(self, observations: list[Observation]):
        """Update baseline statistics."""
        self._baseline_latency.extend([o.latency_ms for o in observations])
        self._baseline_quality.extend([o.quality_score for o in observations])
        self._baseline_tokens.extend([o.tokens_used for o in observations])
        
        # Keep only recent history
        max_history = 100
        self._baseline_latency = self._baseline_latency[-max_history:]
        self._baseline_quality = self._baseline_quality[-max_history:]
        self._baseline_tokens = self._baseline_tokens[-max_history:]
    
    def detect(self, observations: list[Observation]) -> list[DetectedAnomaly]:
        """Detect anomalies in observations."""
        anomalies = []
        
        for obs in observations:
            # Check latency anomaly
            latency_anomaly = self._check_latency_anomaly(obs)
            if latency_anomaly:
                anomalies.append(latency_anomaly)
            
            # Check quality anomaly
            quality_anomaly = self._check_quality_anomaly(obs)
            if quality_anomaly:
                anomalies.append(quality_anomaly)
            
            # Check token usage anomaly
            token_anomaly = self._check_token_anomaly(obs)
            if token_anomaly:
                anomalies.append(token_anomaly)
        
        self.update_baseline(observations)
        return anomalies
    
    def _check_latency_anomaly(self, obs: Observation) -> Optional[DetectedAnomaly]:
        """Check for latency outlier."""
        if len(self._baseline_latency) < 5:
            return None
        
        mean = statistics.mean(self._baseline_latency)
        std = statistics.stdev(self._baseline_latency) if len(self._baseline_latency) > 1 else 1
        
        if std == 0:
            return None
        
        z_score = (obs.latency_ms - mean) / std
        
        if abs(z_score) > self.z_threshold:
            return DetectedAnomaly(
                anomaly_type=AnomalyType.OUTLIER_LATENCY,
                severity=min(1.0, abs(z_score) / 5),
                agent_id=obs.agent_id,
                value=obs.latency_ms,
                expected_range=(mean - 2*std, mean + 2*std),
                description=f"Latency {obs.latency_ms:.0f}ms is {z_score:.1f} std from mean {mean:.0f}ms",
            )
        return None
    
    def _check_quality_anomaly(self, obs: Observation) -> Optional[DetectedAnomaly]:
        """Check for quality outlier."""
        if len(self._baseline_quality) < 5:
            return None
        
        mean = statistics.mean(self._baseline_quality)
        std = statistics.stdev(self._baseline_quality) if len(self._baseline_quality) > 1 else 0.1
        
        if std == 0:
            return None
        
        z_score = (obs.quality_score - mean) / std
        
        # Only flag low quality as anomaly
        if z_score < -self.z_threshold:
            return DetectedAnomaly(
                anomaly_type=AnomalyType.OUTLIER_QUALITY,
                severity=min(1.0, abs(z_score) / 5),
                agent_id=obs.agent_id,
                value=obs.quality_score,
                expected_range=(mean - 2*std, mean + 2*std),
                description=f"Quality {obs.quality_score:.2f} is {z_score:.1f} std below mean {mean:.2f}",
            )
        return None
    
    def _check_token_anomaly(self, obs: Observation) -> Optional[DetectedAnomaly]:
        """Check for unusual token usage."""
        if len(self._baseline_tokens) < 5:
            return None
        
        mean = statistics.mean(self._baseline_tokens)
        std = statistics.stdev(self._baseline_tokens) if len(self._baseline_tokens) > 1 else mean * 0.2
        
        if std == 0:
            return None
        
        z_score = (obs.tokens_used - mean) / std
        
        if abs(z_score) > self.z_threshold:
            return DetectedAnomaly(
                anomaly_type=AnomalyType.UNUSUAL_TOKEN_USAGE,
                severity=min(1.0, abs(z_score) / 5),
                agent_id=obs.agent_id,
                value=obs.tokens_used,
                expected_range=(mean - 2*std, mean + 2*std),
                description=f"Token usage {obs.tokens_used} is {z_score:.1f} std from mean {mean:.0f}",
            )
        return None


class AdaptiveEngine:
    """
    Adapts evaluation criteria based on observed patterns and anomalies.
    
    Core APEE component (Section 3.2.4) that:
    - Detects behavioral patterns
    - Identifies anomalies
    - Dynamically adjusts evaluation weights
    - Generates benchmark recommendations
    """
    
    def __init__(self, baseline_criteria: Optional[EvaluationCriteria] = None):
        self.criteria = baseline_criteria or EvaluationCriteria()
        self.pattern_detector = PatternDetector()
        self.anomaly_detector = AnomalyDetector()
        
        self._adaptation_history: list[dict] = []
        self._patterns_detected: list[DetectedPattern] = []
        self._anomalies_detected: list[DetectedAnomaly] = []
    
    def adapt(self, observations: list[Observation]) -> EvaluationCriteria:
        """
        Adapt evaluation criteria based on observations.
        
        Returns updated criteria with adjusted weights and focus areas.
        """
        # Detect patterns
        patterns = self.pattern_detector.detect(observations)
        self._patterns_detected.extend(patterns)
        
        # Detect anomalies
        anomalies = self.anomaly_detector.detect(observations)
        self._anomalies_detected.extend(anomalies)
        
        # Apply adaptations based on patterns
        for pattern in patterns:
            self._apply_pattern_adaptation(pattern)
        
        # Apply adaptations based on anomalies
        for anomaly in anomalies:
            self._apply_anomaly_adaptation(anomaly)
        
        # Normalize weights
        self.criteria.normalize_weights()
        
        # Record adaptation
        self._adaptation_history.append({
            "observation_count": len(observations),
            "patterns_detected": len(patterns),
            "anomalies_detected": len(anomalies),
            "criteria_snapshot": self._snapshot_criteria(),
        })
        
        return self.criteria
    
    def _apply_pattern_adaptation(self, pattern: DetectedPattern):
        """Adjust criteria based on detected pattern."""
        if pattern.pattern_type == PatternType.EMERGENT_COLLABORATION:
            # Increase collaboration weight when emergence detected
            self.criteria.coordination_weight *= 1.2
            self.criteria.synergy_weight *= 1.3
            self.criteria.add_focus_area("collaboration_dynamics")
        
        elif pattern.pattern_type == PatternType.CONSISTENT_FAILURE:
            # Increase completion weight, add reliability focus
            self.criteria.completion_weight *= 1.5
            self.criteria.stability_weight *= 1.3
            self.criteria.add_focus_area("agent_reliability")
        
        elif pattern.pattern_type == PatternType.QUALITY_DEGRADATION:
            # Increase quality weight, add degradation monitoring
            self.criteria.quality_weight *= 1.4
            self.criteria.add_focus_area("quality_monitoring")
        
        elif pattern.pattern_type == PatternType.SYNERGY_BOOST:
            # Record synergy success, increase synergy weight
            self.criteria.synergy_weight *= 1.2
            self.criteria.add_focus_area("synergy_patterns")
        
        elif pattern.pattern_type == PatternType.CONFLICT_ESCALATION:
            # Increase conflict resolution weight
            self.criteria.conflict_resolution_weight *= 1.5
            self.criteria.add_focus_area("conflict_management")
        
        elif pattern.pattern_type == PatternType.CONVERGENT_BEHAVIOR:
            # Note convergence, may indicate groupthink
            self.criteria.add_focus_area("diversity_check")
        
        elif pattern.pattern_type == PatternType.DIVERGENT_BEHAVIOR:
            # Note divergence, may need consensus mechanisms
            self.criteria.add_focus_area("consensus_building")
    
    def _apply_anomaly_adaptation(self, anomaly: DetectedAnomaly):
        """Adjust criteria based on detected anomaly."""
        if anomaly.anomaly_type == AnomalyType.OUTLIER_LATENCY:
            # Flag latency issues
            if anomaly.severity > 0.7:
                self.criteria.latency_weight *= 1.3
                self.criteria.add_focus_area(f"latency_investigation_{anomaly.agent_id}")
        
        elif anomaly.anomaly_type == AnomalyType.OUTLIER_QUALITY:
            # Flag quality issues
            if anomaly.severity > 0.5:
                self.criteria.quality_weight *= 1.2
                self.criteria.add_focus_area(f"quality_review_{anomaly.agent_id}")
        
        elif anomaly.anomaly_type == AnomalyType.UNUSUAL_TOKEN_USAGE:
            # Flag efficiency issues
            self.criteria.efficiency_weight *= 1.1
            self.criteria.add_focus_area("token_efficiency")
    
    def _snapshot_criteria(self) -> dict:
        """Create snapshot of current criteria."""
        return {
            "completion_weight": self.criteria.completion_weight,
            "quality_weight": self.criteria.quality_weight,
            "latency_weight": self.criteria.latency_weight,
            "coordination_weight": self.criteria.coordination_weight,
            "synergy_weight": self.criteria.synergy_weight,
            "focus_areas": self.criteria.focus_areas.copy(),
        }
    
    def get_insights(self) -> dict:
        """Get insights from adaptation process."""
        return {
            "total_adaptations": len(self._adaptation_history),
            "patterns_detected": len(self._patterns_detected),
            "anomalies_detected": len(self._anomalies_detected),
            "pattern_types": list(set(p.pattern_type.value for p in self._patterns_detected)),
            "anomaly_types": list(set(a.anomaly_type.value for a in self._anomalies_detected)),
            "current_focus_areas": self.criteria.focus_areas,
            "adaptation_history": self._adaptation_history[-5:],  # Last 5
        }
    
    def reset(self):
        """Reset to baseline criteria."""
        self.criteria = EvaluationCriteria()
        self._adaptation_history = []
        self._patterns_detected = []
        self._anomalies_detected = []
