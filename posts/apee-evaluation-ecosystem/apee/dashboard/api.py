"""
APEE Dashboard API.

Provides a programmatic interface to the dashboard server.
"""

from dataclasses import dataclass
from typing import Any, Optional
from enum import Enum


class APIEndpoint(str, Enum):
    """Available API endpoints."""
    SUMMARY = "summary"
    EVALUATIONS = "evaluations"
    ANOMALIES = "anomalies"
    AGENTS = "agents"
    SCENARIOS = "scenarios"
    EXPORT = "export"


@dataclass
class APIResponse:
    """Response from a dashboard API call."""
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    status_code: int = 200


class DashboardAPI:
    """
    Client for interacting with the APEE Dashboard API.
    
    Can be used to push data to a remote dashboard or query data.
    """
    
    def __init__(self, base_url: str = "http://localhost:8765"):
        self.base_url = base_url.rstrip("/")
    
    def _request(
        self,
        endpoint: str,
        method: str = "GET",
        data: Optional[dict] = None
    ) -> APIResponse:
        """Make an API request."""
        try:
            import httpx
            
            url = f"{self.base_url}/api/{endpoint}"
            
            with httpx.Client(timeout=10.0) as client:
                if method == "GET":
                    response = client.get(url)
                elif method == "POST":
                    response = client.post(url, json=data)
                else:
                    return APIResponse(success=False, error=f"Unsupported method: {method}")
                
                return APIResponse(
                    success=response.status_code == 200,
                    data=response.json() if response.status_code == 200 else None,
                    status_code=response.status_code,
                    error=None if response.status_code == 200 else response.text
                )
                
        except Exception as e:
            return APIResponse(success=False, error=str(e), status_code=0)
    
    def get_summary(self) -> APIResponse:
        """Get dashboard summary."""
        return self._request(APIEndpoint.SUMMARY.value)
    
    def get_evaluations(self, limit: int = 10) -> APIResponse:
        """Get recent evaluations."""
        return self._request(f"{APIEndpoint.EVALUATIONS.value}?limit={limit}")
    
    def get_anomalies(self) -> APIResponse:
        """Get all anomalies."""
        return self._request(APIEndpoint.ANOMALIES.value)
    
    def get_agents(self) -> APIResponse:
        """Get registered agents."""
        return self._request(APIEndpoint.AGENTS.value)
    
    def get_scenarios(self) -> APIResponse:
        """Get scenario results."""
        return self._request(APIEndpoint.SCENARIOS.value)
    
    def export_all(self) -> APIResponse:
        """Export all dashboard data."""
        return self._request(APIEndpoint.EXPORT.value)
    
    def push_evaluation(self, result: dict[str, Any]) -> APIResponse:
        """Push an evaluation result to the dashboard."""
        return self._request("evaluation", method="POST", data=result)
    
    def push_anomaly(self, anomaly: dict[str, Any]) -> APIResponse:
        """Push an anomaly to the dashboard."""
        return self._request("anomaly", method="POST", data=anomaly)
    
    def push_agent(
        self,
        agent_id: str,
        role: str,
        model: str,
        extra: Optional[dict] = None
    ) -> APIResponse:
        """Push agent information to the dashboard."""
        data = {
            "agent_id": agent_id,
            "role": role,
            "model": model,
            **(extra or {})
        }
        return self._request("agent", method="POST", data=data)
    
    def is_available(self) -> bool:
        """Check if the dashboard server is available."""
        response = self.get_summary()
        return response.success


class DashboardIntegration:
    """
    Integration helper for connecting APEE components to the dashboard.
    
    Provides automatic pushing of results as they are generated.
    """
    
    def __init__(
        self,
        api: Optional[DashboardAPI] = None,
        auto_push: bool = True
    ):
        self.api = api or DashboardAPI()
        self.auto_push = auto_push
        self._enabled = False
    
    def enable(self) -> bool:
        """Enable dashboard integration if server is available."""
        self._enabled = self.api.is_available()
        return self._enabled
    
    def disable(self) -> None:
        """Disable dashboard integration."""
        self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        """Check if integration is enabled."""
        return self._enabled
    
    def on_evaluation_complete(self, result: dict[str, Any]) -> None:
        """Called when an evaluation completes."""
        if self._enabled and self.auto_push:
            self.api.push_evaluation(result)
    
    def on_anomaly_detected(self, anomaly: dict[str, Any]) -> None:
        """Called when an anomaly is detected."""
        if self._enabled and self.auto_push:
            self.api.push_anomaly(anomaly)
    
    def register_agent(
        self,
        agent_id: str,
        role: str,
        model: str,
        **kwargs
    ) -> None:
        """Register an agent with the dashboard."""
        if self._enabled:
            self.api.push_agent(agent_id, role, model, kwargs)


# Convenience function for creating integration
def connect_to_dashboard(
    url: str = "http://localhost:8765",
    auto_push: bool = True
) -> DashboardIntegration:
    """
    Create and enable dashboard integration.
    
    Args:
        url: Dashboard server URL
        auto_push: Whether to automatically push results
        
    Returns:
        Enabled DashboardIntegration instance
    """
    api = DashboardAPI(url)
    integration = DashboardIntegration(api=api, auto_push=auto_push)
    integration.enable()
    return integration
