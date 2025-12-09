"""
Phase 6 Demo: Visualization, Anomaly Detection, and Dashboard

This example demonstrates the new Phase 6 features:
1. Visualization utilities for evaluation results
2. Anomaly detection for identifying issues
3. Web dashboard for real-time monitoring

Usage:
    python examples/phase6_demo.py
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# Import APEE core
from apee import (
    Task,
    AgentRole,
    OllamaAgent,
    Coordinator,
)

# Import Phase 6 modules
from apee.visualization.charts import (
    MetricsVisualizer,
    ChartConfig,
    create_evaluation_chart,
    create_comparison_chart,
    create_anomaly_heatmap,
)
from apee.visualization.export import generate_report_html
from apee.anomaly.detector import AnomalyDetector, AnomalyType
from apee.anomaly.patterns import (
    PerformancePatternAnalyzer,
    CollaborationPatternAnalyzer,
    QualityPatternAnalyzer,
)
from apee.anomaly.alerts import (
    AlertManager,
    ConsoleAlertHandler,
    AnomalySeverity,
)
from apee.dashboard.server import create_dashboard, DashboardServer
from apee.dashboard.api import DashboardAPI


def demo_visualization():
    """Demonstrate visualization capabilities."""
    print("\n" + "="*60)
    print("📊 VISUALIZATION DEMO")
    print("="*60 + "\n")
    
    # Create visualizer
    config = ChartConfig(
        title="APEE Evaluation Results",
        width=1000,
        height=600,
        theme="plotly_white"
    )
    visualizer = MetricsVisualizer(config)
    
    # Sample evaluation data
    l1_scores = {
        "executor_goal": 9.0,
        "executor_semantic": 8.0,
        "analyzer_goal": 8.0,
        "analyzer_semantic": 7.0,
        "reviewer_goal": 7.0,
        "reviewer_semantic": 4.0,
    }
    
    l2_scores = {
        "collaboration": 6.5,
        "synthesis": 7.5,
    }
    
    l3_scores = {
        "efficiency": 6.0,
        "stability": 9.0,
        "throughput": 8.5,
        "adaptability": 7.0,
    }
    
    # Create level comparison chart
    print("Creating three-tier metrics comparison...")
    chart = visualizer.create_level_comparison(
        l1_scores, l2_scores, l3_scores,
        title="APEE Three-Tier Evaluation"
    )
    
    # Handle both plotly and text output
    if hasattr(chart, 'to_html'):
        print("✅ Plotly chart created (interactive)")
        # Save to file
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        chart.write_html(str(output_dir / "evaluation_chart.html"))
        print(f"   Saved to: data/evaluation_chart.html")
    else:
        print("📝 Text-based chart (plotly not installed):")
        print(chart.get("text_representation", ""))
    
    # Create agent performance radar
    print("\nCreating agent performance radar...")
    agent_metrics = {
        "executor": {"goal": 9.0, "semantic": 8.0, "latency": 7.0, "quality": 8.5},
        "analyzer": {"goal": 8.0, "semantic": 7.0, "latency": 8.0, "quality": 7.5},
        "reviewer": {"goal": 7.0, "semantic": 4.0, "latency": 9.0, "quality": 6.0},
    }
    
    radar = visualizer.create_agent_performance_radar(
        agent_metrics,
        title="Agent Performance Comparison"
    )
    
    if hasattr(radar, 'to_html'):
        print("✅ Radar chart created")
    else:
        print("📝 Text-based radar:")
        print(radar.get("text_representation", ""))
    
    # Generate full HTML report
    print("\nGenerating comprehensive HTML report...")
    evaluation_result = {
        "overall_apee_score": 7.2,
        "l1_average": 7.5,
        "l2_average": 7.0,
        "l3_average": 7.6,
        "individual_scores": {
            "executor": {"goal_alignment": 9.0, "semantic_quality": 8.0},
            "analyzer": {"goal_alignment": 8.0, "semantic_quality": 7.0},
            "reviewer": {"goal_alignment": 7.0, "semantic_quality": 4.0},
        },
        "collaborative_scores": {"collaboration": 6.5, "synthesis": 7.5},
        "ecosystem_scores": {"efficiency": 6.0, "stability": 9.0, "throughput": 8.5},
        "scenario_id": "code_review_demo",
        "pattern": "peer_review",
    }
    
    output_path = Path(__file__).parent.parent / "data" / "evaluation_report.html"
    generate_report_html(
        evaluation_result,
        title="APEE Code Review Evaluation",
        output_path=str(output_path)
    )
    print(f"✅ Report saved to: {output_path}")


def demo_anomaly_detection():
    """Demonstrate anomaly detection capabilities."""
    print("\n" + "="*60)
    print("⚠️ ANOMALY DETECTION DEMO")
    print("="*60 + "\n")
    
    # Create detector
    detector = AnomalyDetector(
        window_size=50,
        z_threshold=2.5,
        enable_learning=True
    )
    
    # Create alert manager with console handler
    alert_manager = AlertManager()
    alert_manager.add_handler(ConsoleAlertHandler(min_severity=AnomalySeverity.INFO))
    
    # Simulate normal evaluations to build baseline
    print("Building baseline from normal evaluations...")
    normal_scores = [7.0, 7.2, 7.5, 6.8, 7.1, 7.3, 7.0, 6.9, 7.4, 7.2]
    for score in normal_scores:
        detector.check_value("overall_apee_score", score)
    print(f"   Baseline established from {len(normal_scores)} evaluations")
    
    # Check an anomalous evaluation
    print("\nChecking potentially anomalous evaluations...")
    
    # Test 1: Normal value
    result = detector.check_value("overall_apee_score", 7.1, {"scenario": "test1"})
    if result:
        alert_manager.process_anomaly(result)
    else:
        print("✅ Score 7.1: Normal (no anomaly)")
    
    # Test 2: Low score
    result = detector.check_value("overall_apee_score", 3.0, {"scenario": "test2"})
    if result:
        print(f"\n🚨 Anomaly detected for score 3.0:")
        alert_manager.process_anomaly(result)
    
    # Test 3: Check complete evaluation
    print("\n\nChecking complete evaluation result...")
    eval_result = {
        "overall_apee_score": 2.5,  # Very low
        "l1_average": 3.0,
        "l2_average": 1.0,  # Very low collaboration
        "l3_average": 4.0,
        "judge_scores": {
            "judge1": 6.0,
            "judge2": 2.0,  # High disagreement
        }
    }
    
    anomalies = detector.check_evaluation(eval_result, "problematic_scenario")
    print(f"\nFound {len(anomalies)} anomalies in evaluation:")
    for anomaly in anomalies:
        alert_manager.process_anomaly(anomaly)
    
    # Print summary
    summary = detector.get_anomaly_summary()
    print(f"\n📊 Anomaly Summary:")
    print(f"   Total anomalies detected: {summary['total_anomalies']}")
    print(f"   Critical/Emergency: {summary['critical_count']}")
    print(f"   By type: {summary['by_type']}")


def demo_pattern_analyzers():
    """Demonstrate pattern analysis capabilities."""
    print("\n" + "="*60)
    print("📈 PATTERN ANALYSIS DEMO")
    print("="*60 + "\n")
    
    # Performance pattern analyzer
    print("Performance Pattern Analysis:")
    perf_analyzer = PerformancePatternAnalyzer(min_samples=5)
    
    # Simulate degrading performance over time
    scores = [8.0, 7.8, 7.5, 7.2, 6.9, 6.5, 6.2, 5.8, 5.5, 5.0]
    for i, score in enumerate(scores):
        perf_analyzer.record("quality_score", score, timestamp=i * 100)
    
    trend = perf_analyzer.analyze_trend("quality_score")
    if trend:
        print(f"   Trend direction: {trend.direction}")
        print(f"   Trend slope: {trend.slope:.4f}")
        print(f"   Confidence: {trend.confidence:.2%}")
        print(f"   Data points: {trend.data_points}")
    
    anomalies = perf_analyzer.detect_anomalies("quality_score")
    if anomalies:
        print(f"\n   ⚠️ Detected {len(anomalies)} performance anomalies")
        for a in anomalies:
            print(f"      - {a.anomaly_type.value}: {a.description}")
    
    # Quality pattern analyzer
    print("\n\nQuality Pattern Analysis:")
    quality_analyzer = QualityPatternAnalyzer()
    
    # Record varied quality scores
    scores = [7.5, 8.0, 7.2, 2.0, 7.8, 7.5, 1.5, 8.2, 7.0, 7.5]  # Some outliers
    for score in scores:
        quality_analyzer.record("semantic_quality", score)
    
    dist = quality_analyzer.analyze_distribution("semantic_quality")
    print(f"   Distribution stats:")
    print(f"     Mean: {dist['mean']:.2f}")
    print(f"     Std: {dist['std']:.2f}")
    print(f"     Min: {dist['min']:.2f}")
    print(f"     Max: {dist['max']:.2f}")
    
    outliers = quality_analyzer.detect_outliers("semantic_quality")
    print(f"   Outliers detected: {len(outliers)}")
    for idx, val in outliers:
        print(f"     - Index {idx}: {val:.2f}")


def demo_dashboard():
    """Demonstrate dashboard capabilities (without starting server)."""
    print("\n" + "="*60)
    print("🖥️ DASHBOARD DEMO")
    print("="*60 + "\n")
    
    print("The APEE Dashboard provides a real-time web interface for:")
    print("  • Viewing evaluation results")
    print("  • Monitoring agent performance")
    print("  • Tracking anomalies")
    print("  • Visualizing metrics")
    
    print("\nTo start the dashboard, use one of these methods:")
    print("\n  Method 1 - Python:")
    print("    from apee import create_dashboard")
    print("    dashboard = create_dashboard(port=8765)")
    print("    # Dashboard runs at http://localhost:8765")
    
    print("\n  Method 2 - Command Line (after installing package):")
    print("    apee-dashboard --port 8765")
    
    print("\n  Method 3 - With API Integration:")
    print("    from apee import DashboardServer, DashboardAPI")
    print("    server = DashboardServer()")
    print("    server.start()")
    print("    # Push results as they come")
    print("    server.add_evaluation({'overall_apee_score': 7.5, ...})")
    
    print("\n📝 Dashboard Features:")
    print("  • Real-time auto-refresh (every 5 seconds)")
    print("  • Summary cards for key metrics")
    print("  • Evaluation history with score visualization")
    print("  • Agent status monitoring")
    print("  • Anomaly alerts display")
    print("  • Detailed JSON data export")


def demo_anomaly_heatmap():
    """Demonstrate anomaly heatmap visualization."""
    print("\n" + "="*60)
    print("🗺️ ANOMALY HEATMAP DEMO")
    print("="*60 + "\n")
    
    # Sample anomaly data across scenarios
    anomaly_data = {
        "code_review": {
            "quality": 0.1,
            "collaboration": 0.3,
            "latency": 0.2,
            "consistency": 0.4,
        },
        "research_synthesis": {
            "quality": 0.2,
            "collaboration": 0.6,
            "latency": 0.8,  # Warning
            "consistency": 0.3,
        },
        "constrained_problem": {
            "quality": 0.9,  # Anomaly!
            "collaboration": 0.4,
            "latency": 0.3,
            "consistency": 0.2,
        },
        "emergent_behavior": {
            "quality": 0.3,
            "collaboration": 0.2,
            "latency": 0.5,
            "consistency": 0.7,  # Warning
        },
    }
    
    heatmap = create_anomaly_heatmap(anomaly_data)
    
    if hasattr(heatmap, 'to_html'):
        print("✅ Interactive heatmap created (plotly)")
        output_dir = Path(__file__).parent.parent / "data"
        heatmap.write_html(str(output_dir / "anomaly_heatmap.html"))
        print(f"   Saved to: data/anomaly_heatmap.html")
    else:
        print("📝 Text-based heatmap:")
        print(heatmap.get("text_representation", ""))


def main():
    """Run all Phase 6 demos."""
    print("\n" + "="*60)
    print("🎯 APEE PHASE 6: FUTURE ENHANCEMENTS DEMO")
    print("="*60)
    print("\nThis demo showcases the new Phase 6 features:")
    print("  1. Visualization utilities")
    print("  2. Advanced anomaly detection")
    print("  3. Pattern analysis")
    print("  4. Web dashboard")
    print("="*60)
    
    # Run demos
    demo_visualization()
    demo_anomaly_detection()
    demo_pattern_analyzers()
    demo_anomaly_heatmap()
    demo_dashboard()
    
    print("\n" + "="*60)
    print("✅ PHASE 6 DEMO COMPLETE")
    print("="*60)
    print("\nFor more information, visit:")
    print("  https://ahjavid.github.io/technical-notes-blog/")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
