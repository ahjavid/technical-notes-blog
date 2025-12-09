"""
APEE Chart Generation.

Creates various charts and visualizations for evaluation metrics.
Uses matplotlib/plotly for rendering.
"""

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional
from pathlib import Path


class ChartType(str, Enum):
    """Available chart types."""
    BAR = "bar"
    LINE = "line"
    RADAR = "radar"
    HEATMAP = "heatmap"
    SANKEY = "sankey"
    SCATTER = "scatter"
    BOX = "box"


@dataclass
class ChartConfig:
    """Configuration for chart generation."""
    title: str = "APEE Evaluation"
    width: int = 800
    height: int = 600
    theme: str = "plotly_white"
    show_legend: bool = True
    show_grid: bool = True
    colors: Optional[list[str]] = None


class MetricsVisualizer:
    """
    Visualizer for APEE evaluation metrics.
    
    Generates interactive charts using plotly (when available)
    or falls back to text-based representations.
    """
    
    def __init__(self, config: Optional[ChartConfig] = None):
        self.config = config or ChartConfig()
        self._plotly_available = self._check_plotly()
    
    def _check_plotly(self) -> bool:
        """Check if plotly is available."""
        try:
            import plotly
            return True
        except ImportError:
            return False
    
    def create_level_comparison(
        self,
        l1_scores: dict[str, float],
        l2_scores: dict[str, float],
        l3_scores: dict[str, float],
        title: str = "Three-Tier APEE Metrics"
    ) -> dict[str, Any]:
        """
        Create comparison chart for L1, L2, L3 metrics.
        
        Args:
            l1_scores: Individual agent scores
            l2_scores: Collaborative metrics
            l3_scores: Ecosystem metrics
            title: Chart title
            
        Returns:
            Chart specification dict (or plotly figure if available)
        """
        if self._plotly_available:
            return self._create_plotly_level_comparison(
                l1_scores, l2_scores, l3_scores, title
            )
        return self._create_text_level_comparison(
            l1_scores, l2_scores, l3_scores, title
        )
    
    def _create_plotly_level_comparison(
        self,
        l1_scores: dict[str, float],
        l2_scores: dict[str, float],
        l3_scores: dict[str, float],
        title: str
    ) -> Any:
        """Create plotly grouped bar chart."""
        import plotly.graph_objects as go
        
        # Prepare data
        categories = []
        l1_values = []
        l2_values = []
        l3_values = []
        
        # Normalize scores to have comparable categories
        all_keys = set(l1_scores.keys()) | set(l2_scores.keys()) | set(l3_scores.keys())
        
        for key in sorted(all_keys):
            categories.append(key)
            l1_values.append(l1_scores.get(key, 0))
            l2_values.append(l2_scores.get(key, 0))
            l3_values.append(l3_scores.get(key, 0))
        
        fig = go.Figure(data=[
            go.Bar(name='L1: Individual', x=categories, y=l1_values, marker_color='#3498db'),
            go.Bar(name='L2: Collaborative', x=categories, y=l2_values, marker_color='#2ecc71'),
            go.Bar(name='L3: Ecosystem', x=categories, y=l3_values, marker_color='#9b59b6'),
        ])
        
        fig.update_layout(
            barmode='group',
            title=title,
            xaxis_title='Metric',
            yaxis_title='Score (0-10)',
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1)
        )
        
        return fig
    
    def _create_text_level_comparison(
        self,
        l1_scores: dict[str, float],
        l2_scores: dict[str, float],
        l3_scores: dict[str, float],
        title: str
    ) -> dict[str, Any]:
        """Create text-based representation of comparison."""
        return {
            "type": "level_comparison",
            "title": title,
            "data": {
                "l1_individual": l1_scores,
                "l2_collaborative": l2_scores,
                "l3_ecosystem": l3_scores,
            },
            "text_representation": self._format_as_text(
                [("L1 Individual", l1_scores),
                 ("L2 Collaborative", l2_scores),
                 ("L3 Ecosystem", l3_scores)],
                title
            )
        }
    
    def _format_as_text(
        self,
        data_groups: list[tuple[str, dict[str, float]]],
        title: str
    ) -> str:
        """Format data as ASCII chart."""
        lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}\n"]
        
        for group_name, scores in data_groups:
            lines.append(f"\n  {group_name}:")
            lines.append(f"  {'-'*40}")
            
            for metric, score in sorted(scores.items()):
                bar_length = int(score * 4)  # Scale to 40 chars for 10 max
                bar = "█" * bar_length + "░" * (40 - bar_length)
                lines.append(f"  {metric:<20} {bar} {score:.1f}/10")
        
        lines.append(f"\n{'='*60}\n")
        return "\n".join(lines)
    
    def create_agent_performance_radar(
        self,
        agent_metrics: dict[str, dict[str, float]],
        title: str = "Agent Performance Comparison"
    ) -> Any:
        """
        Create radar chart comparing agent performance.
        
        Args:
            agent_metrics: {agent_id: {metric: score}}
            title: Chart title
        """
        if self._plotly_available:
            return self._create_plotly_radar(agent_metrics, title)
        return self._create_text_radar(agent_metrics, title)
    
    def _create_plotly_radar(
        self,
        agent_metrics: dict[str, dict[str, float]],
        title: str
    ) -> Any:
        """Create plotly radar chart."""
        import plotly.graph_objects as go
        
        # Get all metrics
        all_metrics = set()
        for metrics in agent_metrics.values():
            all_metrics.update(metrics.keys())
        categories = sorted(all_metrics)
        
        fig = go.Figure()
        
        colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6', '#f39c12']
        
        for i, (agent_id, metrics) in enumerate(agent_metrics.items()):
            values = [metrics.get(cat, 0) for cat in categories]
            values.append(values[0])  # Close the radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories + [categories[0]],
                fill='toself',
                name=agent_id,
                line_color=colors[i % len(colors)],
                opacity=0.7
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 10])
            ),
            showlegend=True,
            title=title,
            template=self.config.theme,
            width=self.config.width,
            height=self.config.height,
        )
        
        return fig
    
    def _create_text_radar(
        self,
        agent_metrics: dict[str, dict[str, float]],
        title: str
    ) -> dict[str, Any]:
        """Create text-based radar representation."""
        text_lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}"]
        
        all_metrics = set()
        for metrics in agent_metrics.values():
            all_metrics.update(metrics.keys())
        categories = sorted(all_metrics)
        
        # Header
        header = f"{'Metric':<20}" + "".join(f"{a[:10]:<12}" for a in agent_metrics.keys())
        text_lines.append(f"\n  {header}")
        text_lines.append(f"  {'-'*len(header)}")
        
        # Data rows
        for cat in categories:
            row = f"{cat:<20}"
            for agent_id, metrics in agent_metrics.items():
                score = metrics.get(cat, 0)
                row += f"{score:>10.1f}  "
            text_lines.append(f"  {row}")
        
        text_lines.append(f"\n{'='*60}\n")
        
        return {
            "type": "radar",
            "title": title,
            "data": agent_metrics,
            "text_representation": "\n".join(text_lines)
        }
    
    def create_timeline(
        self,
        events: list[dict[str, Any]],
        title: str = "Evaluation Timeline"
    ) -> Any:
        """
        Create timeline chart of evaluation events.
        
        Args:
            events: List of {timestamp, agent, event_type, description}
            title: Chart title
        """
        if not events:
            return {"type": "timeline", "title": title, "data": [], "message": "No events to display"}
        
        if self._plotly_available:
            return self._create_plotly_timeline(events, title)
        return self._create_text_timeline(events, title)
    
    def _create_plotly_timeline(self, events: list[dict], title: str) -> Any:
        """Create plotly timeline."""
        import plotly.express as px
        import pandas as pd
        
        df = pd.DataFrame(events)
        
        fig = px.timeline(
            df,
            x_start="start",
            x_end="end",
            y="agent",
            color="event_type",
            title=title,
            template=self.config.theme,
        )
        
        fig.update_layout(
            width=self.config.width,
            height=self.config.height,
        )
        
        return fig
    
    def _create_text_timeline(self, events: list[dict], title: str) -> dict[str, Any]:
        """Create text-based timeline."""
        text_lines = [f"\n{'='*60}", f"  {title}", f"{'='*60}\n"]
        
        for event in sorted(events, key=lambda x: x.get("timestamp", 0)):
            agent = event.get("agent", "unknown")
            event_type = event.get("event_type", "event")
            desc = event.get("description", "")
            ts = event.get("timestamp", "")
            text_lines.append(f"  [{ts}] {agent:<15} | {event_type:<12} | {desc}")
        
        text_lines.append(f"\n{'='*60}\n")
        
        return {
            "type": "timeline",
            "title": title,
            "data": events,
            "text_representation": "\n".join(text_lines)
        }


def create_evaluation_chart(
    evaluation_result: dict[str, Any],
    chart_type: ChartType = ChartType.BAR,
    config: Optional[ChartConfig] = None
) -> Any:
    """
    Create a chart from evaluation results.
    
    Args:
        evaluation_result: Results from EnsembleEvaluator.evaluate_full()
        chart_type: Type of chart to create
        config: Chart configuration
        
    Returns:
        Plotly figure or text representation dict
    """
    visualizer = MetricsVisualizer(config)
    
    # Extract scores from result
    l1_scores = {}
    l2_scores = {}
    l3_scores = {}
    
    if "individual_scores" in evaluation_result:
        for agent_id, scores in evaluation_result["individual_scores"].items():
            for metric, value in scores.items():
                l1_scores[f"{agent_id}_{metric}"] = value
    
    if "collaborative_scores" in evaluation_result:
        l2_scores = evaluation_result["collaborative_scores"]
    
    if "ecosystem_scores" in evaluation_result:
        l3_scores = evaluation_result["ecosystem_scores"]
    
    return visualizer.create_level_comparison(
        l1_scores, l2_scores, l3_scores,
        title="APEE Evaluation Results"
    )


def create_comparison_chart(
    results: list[dict[str, Any]],
    labels: list[str],
    metric: str = "overall_apee_score",
    config: Optional[ChartConfig] = None
) -> Any:
    """
    Compare multiple evaluation results.
    
    Args:
        results: List of evaluation results
        labels: Labels for each result (e.g., scenario names)
        metric: Which metric to compare
        config: Chart configuration
        
    Returns:
        Comparison chart
    """
    visualizer = MetricsVisualizer(config)
    
    scores = {label: result.get(metric, 0) for label, result in zip(labels, results)}
    
    if visualizer._plotly_available:
        import plotly.graph_objects as go
        
        fig = go.Figure(data=[
            go.Bar(
                x=list(scores.keys()),
                y=list(scores.values()),
                marker_color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12'][:len(scores)],
                text=[f"{v:.2f}" for v in scores.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title=f"Comparison: {metric}",
            xaxis_title="Scenario",
            yaxis_title="Score",
            template=visualizer.config.theme,
            width=visualizer.config.width,
            height=visualizer.config.height,
        )
        
        return fig
    
    return {
        "type": "comparison",
        "metric": metric,
        "data": scores,
        "text_representation": "\n".join([
            f"\n{'='*40}",
            f"  Comparison: {metric}",
            f"{'='*40}",
            *[f"  {label:<20} {score:.2f}" for label, score in scores.items()],
            f"{'='*40}\n"
        ])
    }


def create_collaboration_flow(
    trace: Any,  # CollaborativeTrace
    config: Optional[ChartConfig] = None
) -> Any:
    """
    Create Sankey diagram showing collaboration flow.
    
    Args:
        trace: CollaborativeTrace object
        config: Chart configuration
        
    Returns:
        Sankey diagram showing agent interactions
    """
    visualizer = MetricsVisualizer(config)
    
    # Extract flow data from trace
    agents = trace.participating_agents if hasattr(trace, 'participating_agents') else []
    messages = trace.coordination_messages if hasattr(trace, 'coordination_messages') else []
    
    if not agents or not visualizer._plotly_available:
        return {
            "type": "collaboration_flow",
            "agents": agents,
            "message_count": len(messages),
            "pattern": getattr(trace, 'collaboration_pattern', 'unknown'),
        }
    
    import plotly.graph_objects as go
    
    # Build Sankey data
    agent_indices = {agent: i for i, agent in enumerate(agents)}
    
    sources = []
    targets = []
    values = []
    
    # Count messages between agents
    flow_counts: dict[tuple[str, str], int] = {}
    for msg in messages:
        sender = msg.get("sender", "")
        receiver = msg.get("receiver", "")
        if sender in agent_indices and receiver in agent_indices:
            key = (sender, receiver)
            flow_counts[key] = flow_counts.get(key, 0) + 1
    
    for (sender, receiver), count in flow_counts.items():
        sources.append(agent_indices[sender])
        targets.append(agent_indices[receiver])
        values.append(count)
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=agents,
            color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'][:len(agents)]
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
        )
    )])
    
    fig.update_layout(
        title=f"Collaboration Flow ({getattr(trace, 'collaboration_pattern', 'unknown')})",
        width=visualizer.config.width,
        height=visualizer.config.height,
    )
    
    return fig


def create_anomaly_heatmap(
    anomaly_data: dict[str, dict[str, float]],
    config: Optional[ChartConfig] = None
) -> Any:
    """
    Create heatmap showing anomaly scores.
    
    Args:
        anomaly_data: {scenario: {metric: anomaly_score}}
        config: Chart configuration
        
    Returns:
        Heatmap visualization
    """
    visualizer = MetricsVisualizer(config)
    
    if not anomaly_data:
        return {"type": "heatmap", "data": {}, "message": "No anomaly data"}
    
    scenarios = list(anomaly_data.keys())
    metrics = sorted(set(m for scores in anomaly_data.values() for m in scores.keys()))
    
    if visualizer._plotly_available:
        import plotly.graph_objects as go
        import numpy as np
        
        z_data = []
        for scenario in scenarios:
            row = [anomaly_data[scenario].get(m, 0) for m in metrics]
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=metrics,
            y=scenarios,
            colorscale='RdYlGn_r',  # Red for high anomaly
            text=[[f"{v:.2f}" for v in row] for row in z_data],
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="Scenario: %{y}<br>Metric: %{x}<br>Anomaly Score: %{z:.2f}<extra></extra>",
        ))
        
        fig.update_layout(
            title="Anomaly Detection Heatmap",
            xaxis_title="Metric",
            yaxis_title="Scenario",
            template=visualizer.config.theme,
            width=visualizer.config.width,
            height=visualizer.config.height,
        )
        
        return fig
    
    # Text representation
    text_lines = [
        f"\n{'='*60}",
        "  Anomaly Detection Heatmap",
        f"{'='*60}\n",
        f"  {'Scenario':<25}" + "".join(f"{m[:8]:<10}" for m in metrics),
        f"  {'-'*60}"
    ]
    
    for scenario in scenarios:
        row = f"  {scenario:<25}"
        for m in metrics:
            score = anomaly_data[scenario].get(m, 0)
            # Color code: 🟢 <0.3, 🟡 0.3-0.7, 🔴 >0.7
            indicator = "🟢" if score < 0.3 else ("🟡" if score < 0.7 else "🔴")
            row += f"{indicator}{score:.1f}    "
        text_lines.append(row)
    
    text_lines.append(f"\n  Legend: 🟢 Normal | 🟡 Warning | 🔴 Anomaly\n{'='*60}\n")
    
    return {
        "type": "heatmap",
        "data": anomaly_data,
        "text_representation": "\n".join(text_lines)
    }
