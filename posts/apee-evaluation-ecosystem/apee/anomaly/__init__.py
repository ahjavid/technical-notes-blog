"""
APEE Anomaly Detection Module.

Detects unusual patterns in multi-agent evaluations including:
- Performance anomalies (sudden drops/spikes)
- Behavioral anomalies (unexpected agent interactions)
- Quality anomalies (inconsistent output quality)
- Collaboration anomalies (breakdown in coordination)
"""

from apee.anomaly.detector import (
    AnomalyDetector,
    AnomalyType,
    AnomalyReport,
    AnomalySeverity,
)
from apee.anomaly.patterns import (
    PerformancePatternAnalyzer,
    CollaborationPatternAnalyzer,
    QualityPatternAnalyzer,
)
from apee.anomaly.alerts import (
    AnomalyAlert,
    AlertHandler,
    ConsoleAlertHandler,
    WebhookAlertHandler,
)

__all__ = [
    "AnomalyDetector",
    "AnomalyType",
    "AnomalyReport",
    "AnomalySeverity",
    "PerformancePatternAnalyzer",
    "CollaborationPatternAnalyzer",
    "QualityPatternAnalyzer",
    "AnomalyAlert",
    "AlertHandler",
    "ConsoleAlertHandler",
    "WebhookAlertHandler",
]
