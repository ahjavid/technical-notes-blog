"""
APEE Web Dashboard Module.

Provides a simple web interface for viewing evaluation results.
"""

from apee.dashboard.server import (
    DashboardServer,
    create_dashboard,
)
from apee.dashboard.api import (
    DashboardAPI,
    APIEndpoint,
)

__all__ = [
    "DashboardServer",
    "create_dashboard",
    "DashboardAPI",
    "APIEndpoint",
]
