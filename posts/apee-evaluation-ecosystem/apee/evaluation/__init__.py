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

# Advanced APEE Evaluation Patterns
from apee.evaluation.advanced_patterns import (
    # Jury with Personas (Independent Pattern)
    JudgePersona,
    JuryEvaluator,
    PersonaConfig,
    PERSONA_CONFIGS,
    # Calibration Loop (Iterative Pattern)
    CalibrationLoop,
    CalibratedRubric,
    RubricCriterion,
    # Progressive Deepening (Sequential Pattern)
    ProgressiveDeepening,
    ProgressiveResult,
    EvaluationDepth,
    DepthConfig,
    DEFAULT_DEPTH_CONFIGS,
    # Combined (Best Practice)
    CalibratedJuryEvaluator,
    # Factory functions
    create_jury_evaluator,
    create_calibrated_evaluator,
    create_progressive_evaluator,
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
    # Advanced APEE Patterns
    "JudgePersona",
    "JuryEvaluator",
    "PersonaConfig",
    "PERSONA_CONFIGS",
    "CalibrationLoop",
    "CalibratedRubric",
    "RubricCriterion",
    "ProgressiveDeepening",
    "ProgressiveResult",
    "EvaluationDepth",
    "DepthConfig",
    "DEFAULT_DEPTH_CONFIGS",
    "CalibratedJuryEvaluator",
    "create_jury_evaluator",
    "create_calibrated_evaluator",
    "create_progressive_evaluator",
]
