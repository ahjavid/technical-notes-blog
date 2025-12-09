"""Core data models for APEE using Pydantic."""

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    """Predefined agent roles in the APEE system."""
    ANALYZER = "analyzer"
    CODER = "coder"
    REVIEWER = "reviewer"
    SYNTHESIZER = "synthesizer"
    PLANNER = "planner"
    EXECUTOR = "executor"
    CUSTOM = "custom"


class Task(BaseModel):
    """A task to be processed by agents."""
    task_id: str
    description: str
    complexity: float = Field(default=0.5, ge=0.0, le=1.0)
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        frozen = False


class AgentResult(BaseModel):
    """Result produced by an agent after processing a task."""
    task_id: str
    agent_id: str
    agent_role: str
    output: str
    quality_score: float = Field(ge=0.0, le=1.0)
    latency_ms: float
    tokens_used: int = 0
    success: bool
    error: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """Inter-agent communication message."""
    message_id: str
    sender_id: str
    receiver_id: str
    content: str
    message_type: str = "info"  # info, request, response, broadcast
    timestamp: datetime = Field(default_factory=datetime.now)


class AgentMetrics(BaseModel):
    """Accumulated metrics for an agent."""
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    quality_scores: list[float] = Field(default_factory=list)
    messages_sent: int = 0
    messages_received: int = 0
    
    @property
    def completion_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0
    
    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.tasks_completed if self.tasks_completed > 0 else 0.0
    
    @property
    def avg_quality(self) -> float:
        return sum(self.quality_scores) / len(self.quality_scores) if self.quality_scores else 0.0
    
    @property
    def avg_tokens(self) -> float:
        return self.total_tokens / self.tasks_completed if self.tasks_completed > 0 else 0.0
    
    def record_success(self, latency_ms: float, quality: float, tokens: int = 0):
        """Record a successful task completion."""
        self.tasks_completed += 1
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens
        self.quality_scores.append(quality)
    
    def record_failure(self):
        """Record a failed task."""
        self.tasks_failed += 1
