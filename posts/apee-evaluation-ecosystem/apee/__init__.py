"""
APEE - Adaptive Poly-Agentic Evaluation Ecosystem

A comprehensive framework for evaluating multi-agent AI systems.
"""

__version__ = "0.2.0"
__author__ = "ahjavid"

from apee.models import Task, AgentResult, AgentRole
from apee.agents.base import Agent
from apee.agents.ollama import OllamaAgent
from apee.coordination.coordinator import Coordinator
from apee.evaluation.evaluator import Evaluator
from apee.evaluation.report import EvaluationReport
from apee.evaluation.quality import (
    QualityScore,
    QualityScorer,
    HeuristicScorer,
    CompositeScorer,
)
from apee.benchmarks.datasets import (
    BenchmarkDataset,
    EvaluationScenario,
    TaskCategory,
    Complexity,
)
from apee.benchmarks.runner import (
    BenchmarkRunner,
    BenchmarkConfig,
    BenchmarkResult,
)
from apee.benchmarks.analyzer import (
    BenchmarkAnalyzer,
    StatisticalAnalysis,
    ModelComparison,
)

__all__ = [
    # Core models
    "Task",
    "AgentResult",
    "AgentRole",
    # Agents
    "Agent",
    "OllamaAgent",
    # Coordination
    "Coordinator",
    # Evaluation
    "Evaluator",
    "EvaluationReport",
    # Quality scoring
    "QualityScore",
    "QualityScorer",
    "HeuristicScorer",
    "CompositeScorer",
    # Benchmarks
    "BenchmarkDataset",
    "EvaluationScenario",
    "TaskCategory",
    "Complexity",
    "BenchmarkRunner",
    "BenchmarkConfig",
    "BenchmarkResult",
    "BenchmarkAnalyzer",
    "StatisticalAnalysis",
    "ModelComparison",
]

