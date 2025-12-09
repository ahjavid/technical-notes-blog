"""Evaluation report generation and formatting."""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field

from apee.models import AgentResult


class IndividualMetrics(BaseModel):
    """Metrics for a single agent."""
    agent_id: str
    role: str
    tasks_completed: int
    tasks_failed: int
    completion_rate: float
    avg_latency_ms: float
    avg_quality: float
    avg_tokens: float
    messages_sent: int
    messages_received: int


class CollaborativeMetrics(BaseModel):
    """Metrics for multi-agent collaboration."""
    total_messages: int
    avg_messages_per_task: float
    communication_efficiency: float
    synergy_score: float
    conflict_rate: float = 0.0


class EcosystemMetrics(BaseModel):
    """System-wide ecosystem metrics."""
    total_tasks: int
    total_results: int
    overall_success_rate: float
    avg_system_latency_ms: float
    total_tokens: int
    agent_count: int
    agent_utilization: float


class QualityMetrics(BaseModel):
    """Quality-related metrics."""
    avg_quality: float
    quality_std: float
    min_quality: float
    max_quality: float
    quality_distribution: dict[str, int] = Field(default_factory=dict)


class EvaluationReport(BaseModel):
    """Complete multi-level evaluation report."""
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.now)
    evaluation_id: str = ""
    description: str = ""
    
    # Three-level metrics
    individual: list[IndividualMetrics] = Field(default_factory=list)
    collaborative: Optional[CollaborativeMetrics] = None
    ecosystem: Optional[EcosystemMetrics] = None
    quality: Optional[QualityMetrics] = None
    
    # Raw data references
    execution_patterns: list[str] = Field(default_factory=list)
    model_info: dict[str, str] = Field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return self.model_dump()
    
    def to_json(self, indent: int = 2) -> str:
        """Convert report to JSON string."""
        return self.model_dump_json(indent=indent)
    
    def print_summary(self, show_individual: bool = True) -> None:
        """Print formatted evaluation summary."""
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        
        console = Console()
        
        # Header
        console.print(Panel.fit(
            "[bold blue]APEE Evaluation Report[/bold blue]",
            subtitle=f"Generated: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
        ))
        
        # Individual Agent Metrics
        if show_individual and self.individual:
            table = Table(title="📊 Individual Agent Metrics", show_header=True)
            table.add_column("Agent", style="cyan")
            table.add_column("Role", style="magenta")
            table.add_column("Tasks", justify="right")
            table.add_column("Success", justify="right")
            table.add_column("Avg Latency", justify="right")
            table.add_column("Avg Quality", justify="right")
            table.add_column("Tokens", justify="right")
            
            for m in self.individual:
                table.add_row(
                    m.agent_id,
                    m.role,
                    str(m.tasks_completed),
                    f"{m.completion_rate:.1%}",
                    f"{m.avg_latency_ms:.0f}ms",
                    f"{m.avg_quality:.2f}",
                    f"{m.avg_tokens:.0f}"
                )
            
            console.print(table)
            console.print()
        
        # Collaborative Metrics
        if self.collaborative:
            console.print("[bold]🤝 Collaborative Metrics[/bold]")
            console.print(f"  • Total Messages: {self.collaborative.total_messages}")
            console.print(f"  • Avg Messages/Task: {self.collaborative.avg_messages_per_task:.2f}")
            console.print(f"  • Communication Efficiency: {self.collaborative.communication_efficiency:.1%}")
            console.print(f"  • Synergy Score: {self.collaborative.synergy_score:.2f}x")
            console.print()
        
        # Ecosystem Metrics
        if self.ecosystem:
            console.print("[bold]🌐 Ecosystem Metrics[/bold]")
            console.print(f"  • Total Tasks: {self.ecosystem.total_tasks}")
            console.print(f"  • Success Rate: {self.ecosystem.overall_success_rate:.1%}")
            console.print(f"  • Avg Latency: {self.ecosystem.avg_system_latency_ms:.0f}ms")
            console.print(f"  • Total Tokens: {self.ecosystem.total_tokens:,}")
            console.print(f"  • Agent Utilization: {self.ecosystem.agent_utilization:.1%}")
            console.print()
        
        # Quality Metrics
        if self.quality:
            console.print("[bold]📈 Quality Metrics[/bold]")
            console.print(f"  • Average Quality: {self.quality.avg_quality:.3f}")
            console.print(f"  • Std Deviation: {self.quality.quality_std:.3f}")
            console.print(f"  • Range: [{self.quality.min_quality:.2f}, {self.quality.max_quality:.2f}]")
            if self.quality.quality_distribution:
                console.print(f"  • Distribution: {self.quality.quality_distribution}")
            console.print()
    
    def print_compact(self) -> None:
        """Print compact one-line summary."""
        if self.ecosystem:
            e = self.ecosystem
            q = self.quality
            print(
                f"Tasks: {e.total_tasks} | "
                f"Success: {e.overall_success_rate:.1%} | "
                f"Latency: {e.avg_system_latency_ms:.0f}ms | "
                f"Quality: {q.avg_quality if q else 0:.2f} | "
                f"Tokens: {e.total_tokens:,}"
            )
