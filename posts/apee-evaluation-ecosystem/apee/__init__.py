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
from apee.benchmarks.collaborative import (
    CollaborationPattern,
    CollaborativeScenario,
    MultiAgentDataset,
)

# Phase 6: New modules
from apee.anomaly.detector import (
    AnomalyDetector,
    AnomalyType,
    AnomalyReport,
    AnomalySeverity,
)
from apee.anomaly.alerts import (
    AnomalyAlert,
    AlertManager,
    ConsoleAlertHandler,
)
from apee.visualization.charts import (
    MetricsVisualizer,
    ChartConfig,
    create_evaluation_chart,
    create_comparison_chart,
    create_anomaly_heatmap,
)
from apee.visualization.export import (
    export_to_html,
    generate_report_html,
)
from apee.dashboard.server import (
    DashboardServer,
    create_dashboard,
)
from apee.dashboard.api import (
    DashboardAPI,
    DashboardIntegration,
    connect_to_dashboard,
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
    # Collaborative
    "CollaborationPattern",
    "CollaborativeScenario",
    "MultiAgentDataset",
    # Phase 6: Anomaly Detection
    "AnomalyDetector",
    "AnomalyType",
    "AnomalyReport",
    "AnomalySeverity",
    "AnomalyAlert",
    "AlertManager",
    "ConsoleAlertHandler",
    # Phase 6: Visualization
    "MetricsVisualizer",
    "ChartConfig",
    "create_evaluation_chart",
    "create_comparison_chart",
    "create_anomaly_heatmap",
    "export_to_html",
    "generate_report_html",
    # Phase 6: Dashboard
    "DashboardServer",
    "create_dashboard",
    "DashboardAPI",
    "DashboardIntegration",
    "connect_to_dashboard",
]

