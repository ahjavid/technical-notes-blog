"""Logging utilities for APEE."""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    format_string: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """
    Configure logging for APEE.
    
    Args:
        level: Logging level (default: INFO)
        format_string: Custom format string
        log_file: Optional file path for log output
    """
    if format_string is None:
        format_string = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
    
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
    )
    
    # Reduce noise from httpx
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(f"apee.{name}")


class APEELogger:
    """Structured logger for APEE operations."""
    
    def __init__(self, name: str):
        self._logger = get_logger(name)
    
    def agent_start(self, agent_id: str, task_id: str) -> None:
        """Log agent starting a task."""
        self._logger.info(f"Agent {agent_id} starting task {task_id}")
    
    def agent_complete(
        self, agent_id: str, task_id: str, latency_ms: float
    ) -> None:
        """Log agent completing a task."""
        self._logger.info(
            f"Agent {agent_id} completed task {task_id} in {latency_ms:.1f}ms"
        )
    
    def agent_error(self, agent_id: str, task_id: str, error: str) -> None:
        """Log agent error."""
        self._logger.error(f"Agent {agent_id} failed task {task_id}: {error}")
    
    def evaluation_start(self, num_tasks: int, num_agents: int) -> None:
        """Log evaluation starting."""
        self._logger.info(
            f"Starting evaluation: {num_tasks} tasks, {num_agents} agents"
        )
    
    def evaluation_complete(self, total_time_ms: float) -> None:
        """Log evaluation complete."""
        self._logger.info(f"Evaluation complete in {total_time_ms:.1f}ms")
