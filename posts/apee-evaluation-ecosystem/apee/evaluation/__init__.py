"""Evaluation module - Metrics and reporting."""

from apee.evaluation.evaluator import Evaluator
from apee.evaluation.report import EvaluationReport
from apee.evaluation.quality import QualityScorer, LLMQualityScorer
from apee.evaluation.adaptive import (
    AdaptiveEngine,
    PatternDetector,
    AnomalyDetector,
    EvaluationCriteria,
    Observation,
)

__all__ = [
    # Core evaluation
    "Evaluator",
    "EvaluationReport",
    "QualityScorer",
    "LLMQualityScorer",
    # Adaptive engine (APEE core)
    "AdaptiveEngine",
    "PatternDetector",
    "AnomalyDetector",
    "EvaluationCriteria",
    "Observation",
]
