"""
APEE Anomaly Alerting.

Provides alert handling for detected anomalies.
"""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Callable
import logging

from apee.anomaly.detector import AnomalyReport, AnomalySeverity


logger = logging.getLogger(__name__)


@dataclass
class AnomalyAlert:
    """An alert generated from an anomaly."""
    alert_id: str
    anomaly: AnomalyReport
    created_at: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    notes: str = ""
    
    def acknowledge(self, by: str = "system", notes: str = "") -> None:
        """Mark alert as acknowledged."""
        self.acknowledged = True
        self.acknowledged_at = datetime.now()
        self.acknowledged_by = by
        self.notes = notes
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "anomaly": self.anomaly.to_dict(),
            "created_at": self.created_at.isoformat(),
            "acknowledged": self.acknowledged,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "notes": self.notes,
        }


class AlertHandler(ABC):
    """Base class for alert handlers."""
    
    def __init__(self, min_severity: AnomalySeverity = AnomalySeverity.WARNING):
        self.min_severity = min_severity
        self._severity_order = {
            AnomalySeverity.INFO: 0,
            AnomalySeverity.WARNING: 1,
            AnomalySeverity.CRITICAL: 2,
            AnomalySeverity.EMERGENCY: 3,
        }
    
    def should_handle(self, alert: AnomalyAlert) -> bool:
        """Check if this handler should process the alert."""
        return self._severity_order[alert.anomaly.severity] >= self._severity_order[self.min_severity]
    
    @abstractmethod
    def handle(self, alert: AnomalyAlert) -> bool:
        """
        Handle an alert.
        
        Returns:
            True if handled successfully
        """
        pass
    
    def handle_batch(self, alerts: list[AnomalyAlert]) -> list[bool]:
        """Handle multiple alerts."""
        return [self.handle(alert) for alert in alerts if self.should_handle(alert)]


class ConsoleAlertHandler(AlertHandler):
    """Handler that prints alerts to console with rich formatting."""
    
    def __init__(
        self,
        min_severity: AnomalySeverity = AnomalySeverity.INFO,
        use_rich: bool = True
    ):
        super().__init__(min_severity)
        self.use_rich = use_rich
        self._rich_available = self._check_rich()
    
    def _check_rich(self) -> bool:
        """Check if rich is available."""
        try:
            import rich
            return True
        except ImportError:
            return False
    
    def handle(self, alert: AnomalyAlert) -> bool:
        """Print alert to console."""
        if not self.should_handle(alert):
            return False
        
        if self.use_rich and self._rich_available:
            return self._handle_rich(alert)
        return self._handle_plain(alert)
    
    def _handle_rich(self, alert: AnomalyAlert) -> bool:
        """Handle with rich formatting."""
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        
        console = Console()
        
        # Severity styling
        severity_styles = {
            AnomalySeverity.INFO: ("blue", "ℹ️"),
            AnomalySeverity.WARNING: ("yellow", "⚠️"),
            AnomalySeverity.CRITICAL: ("red", "🚨"),
            AnomalySeverity.EMERGENCY: ("bold red", "🔴"),
        }
        
        style, emoji = severity_styles.get(alert.anomaly.severity, ("white", "?"))
        
        # Build content
        content = Table.grid(padding=(0, 2))
        content.add_column()
        content.add_column()
        
        content.add_row("Type:", alert.anomaly.anomaly_type.value)
        content.add_row("Metric:", alert.anomaly.metric_name)
        content.add_row("Observed:", f"{alert.anomaly.observed_value:.2f}")
        content.add_row("Expected:", f"{alert.anomaly.expected_range[0]:.2f} - {alert.anomaly.expected_range[1]:.2f}")
        content.add_row("Deviation:", f"{alert.anomaly.deviation_score:.2%}")
        
        if alert.anomaly.recommendations:
            content.add_row("", "")
            content.add_row("[bold]Recommendations:[/bold]", "")
            for rec in alert.anomaly.recommendations[:3]:
                content.add_row("  •", rec)
        
        panel = Panel(
            content,
            title=f"{emoji} {alert.anomaly.severity.value.upper()} ALERT",
            subtitle=f"ID: {alert.alert_id}",
            border_style=style,
        )
        
        console.print(panel)
        return True
    
    def _handle_plain(self, alert: AnomalyAlert) -> bool:
        """Handle with plain text formatting."""
        severity_icons = {
            AnomalySeverity.INFO: "[INFO]",
            AnomalySeverity.WARNING: "[WARN]",
            AnomalySeverity.CRITICAL: "[CRIT]",
            AnomalySeverity.EMERGENCY: "[EMRG]",
        }
        
        icon = severity_icons.get(alert.anomaly.severity, "[????]")
        
        print(f"\n{'='*60}")
        print(f"{icon} ANOMALY ALERT: {alert.anomaly.anomaly_type.value}")
        print(f"{'='*60}")
        print(f"Alert ID:    {alert.alert_id}")
        print(f"Metric:      {alert.anomaly.metric_name}")
        print(f"Description: {alert.anomaly.description}")
        print(f"Observed:    {alert.anomaly.observed_value:.2f}")
        print(f"Expected:    {alert.anomaly.expected_range[0]:.2f} - {alert.anomaly.expected_range[1]:.2f}")
        print(f"Deviation:   {alert.anomaly.deviation_score:.2%}")
        
        if alert.anomaly.recommendations:
            print("\nRecommendations:")
            for rec in alert.anomaly.recommendations:
                print(f"  • {rec}")
        
        print(f"{'='*60}\n")
        return True


class WebhookAlertHandler(AlertHandler):
    """Handler that sends alerts to a webhook endpoint."""
    
    def __init__(
        self,
        webhook_url: str,
        min_severity: AnomalySeverity = AnomalySeverity.WARNING,
        headers: Optional[dict[str, str]] = None,
        timeout: float = 10.0
    ):
        super().__init__(min_severity)
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
    
    def handle(self, alert: AnomalyAlert) -> bool:
        """Send alert to webhook."""
        if not self.should_handle(alert):
            return False
        
        try:
            import httpx
            
            payload = {
                "alert": alert.to_dict(),
                "source": "apee",
                "timestamp": datetime.now().isoformat(),
            }
            
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers
                )
                response.raise_for_status()
                logger.info(f"Alert {alert.alert_id} sent to webhook successfully")
                return True
                
        except Exception as e:
            logger.error(f"Failed to send alert to webhook: {e}")
            return False


class LoggingAlertHandler(AlertHandler):
    """Handler that logs alerts using Python logging."""
    
    def __init__(
        self,
        min_severity: AnomalySeverity = AnomalySeverity.INFO,
        logger_name: str = "apee.anomaly"
    ):
        super().__init__(min_severity)
        self.logger = logging.getLogger(logger_name)
    
    def handle(self, alert: AnomalyAlert) -> bool:
        """Log the alert."""
        if not self.should_handle(alert):
            return False
        
        log_levels = {
            AnomalySeverity.INFO: logging.INFO,
            AnomalySeverity.WARNING: logging.WARNING,
            AnomalySeverity.CRITICAL: logging.ERROR,
            AnomalySeverity.EMERGENCY: logging.CRITICAL,
        }
        
        level = log_levels.get(alert.anomaly.severity, logging.INFO)
        
        message = (
            f"ANOMALY [{alert.anomaly.severity.value}]: {alert.anomaly.anomaly_type.value} - "
            f"{alert.anomaly.description} | Metric: {alert.anomaly.metric_name}, "
            f"Value: {alert.anomaly.observed_value:.2f}, Deviation: {alert.anomaly.deviation_score:.2%}"
        )
        
        self.logger.log(level, message, extra={"alert": alert.to_dict()})
        return True


class CallbackAlertHandler(AlertHandler):
    """Handler that calls a custom callback function."""
    
    def __init__(
        self,
        callback: Callable[[AnomalyAlert], bool],
        min_severity: AnomalySeverity = AnomalySeverity.INFO
    ):
        super().__init__(min_severity)
        self.callback = callback
    
    def handle(self, alert: AnomalyAlert) -> bool:
        """Call the callback with the alert."""
        if not self.should_handle(alert):
            return False
        
        try:
            return self.callback(alert)
        except Exception as e:
            logger.error(f"Alert callback failed: {e}")
            return False


class AlertManager:
    """
    Manages alert generation and routing.
    
    Coordinates between anomaly detection and alert handlers.
    """
    
    def __init__(self):
        self.handlers: list[AlertHandler] = []
        self._alert_history: list[AnomalyAlert] = []
        self._alert_counter = 0
    
    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler."""
        self.handlers.append(handler)
    
    def remove_handler(self, handler: AlertHandler) -> None:
        """Remove an alert handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def create_alert(self, anomaly: AnomalyReport) -> AnomalyAlert:
        """Create an alert from an anomaly."""
        self._alert_counter += 1
        alert = AnomalyAlert(
            alert_id=f"APEE-{self._alert_counter:06d}",
            anomaly=anomaly,
        )
        self._alert_history.append(alert)
        return alert
    
    def process_anomaly(self, anomaly: AnomalyReport) -> tuple[AnomalyAlert, list[bool]]:
        """
        Process an anomaly through all handlers.
        
        Returns:
            Tuple of (created alert, list of handler results)
        """
        alert = self.create_alert(anomaly)
        results = [handler.handle(alert) for handler in self.handlers]
        return alert, results
    
    def process_anomalies(self, anomalies: list[AnomalyReport]) -> list[AnomalyAlert]:
        """Process multiple anomalies."""
        alerts = []
        for anomaly in anomalies:
            alert, _ = self.process_anomaly(anomaly)
            alerts.append(alert)
        return alerts
    
    def get_unacknowledged(self) -> list[AnomalyAlert]:
        """Get all unacknowledged alerts."""
        return [a for a in self._alert_history if not a.acknowledged]
    
    def get_by_severity(self, severity: AnomalySeverity) -> list[AnomalyAlert]:
        """Get alerts by severity."""
        return [a for a in self._alert_history if a.anomaly.severity == severity]
    
    def acknowledge_all(self, by: str = "system") -> int:
        """Acknowledge all pending alerts."""
        count = 0
        for alert in self._alert_history:
            if not alert.acknowledged:
                alert.acknowledge(by)
                count += 1
        return count
    
    def get_summary(self) -> dict[str, Any]:
        """Get alert summary."""
        total = len(self._alert_history)
        unacked = len(self.get_unacknowledged())
        
        by_severity = {}
        for sev in AnomalySeverity:
            by_severity[sev.value] = len(self.get_by_severity(sev))
        
        by_type: dict[str, int] = {}
        for alert in self._alert_history:
            t = alert.anomaly.anomaly_type.value
            by_type[t] = by_type.get(t, 0) + 1
        
        return {
            "total_alerts": total,
            "unacknowledged": unacked,
            "by_severity": by_severity,
            "by_type": by_type,
            "handlers_active": len(self.handlers),
        }
