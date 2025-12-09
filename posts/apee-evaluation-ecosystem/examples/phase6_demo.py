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


def load_evaluation_results():
    """Load real evaluation results from JSON file."""
    results_file = Path(__file__).parent.parent / "data" / "apee_evaluation_results.json"
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return None


def demo_visualization():
    """Demonstrate visualization capabilities using REAL evaluation data from JSON."""
    print("\n" + "="*60)
    print("📊 VISUALIZATION DEMO")
    print("="*60 + "\n")
    
    # Load real evaluation results
    results = load_evaluation_results()
    
    if not results:
        print("⚠️  No evaluation results found!")
        print("   Run 'python examples/proper_apee_evaluation.py' first to generate results.")
        print("   Then run this demo again.")
        return
    
    print(f"✓ Loaded results from: data/apee_evaluation_results.json")
    print(f"  Timestamp: {results['timestamp']}")
    print(f"  Judge Models: {', '.join(results['judge_models'])}")
    print(f"  Scenarios: {len(results['scenarios'])}")
    
    # Use first scenario for detailed visualization
    first_scenario = results["scenarios"][0]
    scenario_id = first_scenario["scenario_id"]
    
    # Create visualizer
    config = ChartConfig(
        title="APEE Evaluation Results",
        width=1000,
        height=600,
        theme="plotly_white"
    )
    visualizer = MetricsVisualizer(config)
    
    # Extract L1 scores from real data
    l1_data = first_scenario["level1_individual"]
    l1_scores = {}
    for agent_id, scores in l1_data.get("scores_by_agent", {}).items():
        l1_scores[f"{agent_id}_goal"] = scores.get("goal_alignment", {}).get("score", 0)
        l1_scores[f"{agent_id}_semantic"] = scores.get("semantic_quality", {}).get("score", 0)
    
    # Extract L2 scores from real data
    l2_data = first_scenario["level2_collaborative"]
    l2_scores = {
        "collaboration": l2_data.get("scores", {}).get("collaboration_effectiveness", {}).get("score", 0),
        "synthesis": l2_data.get("scores", {}).get("synthesis_quality", {}).get("score", 0),
    }
    
    # Extract L3 scores from real data
    l3_data = first_scenario["level3_ecosystem"]
    l3_scores = {
        "efficiency": l3_data.get("efficiency", 0),
        "stability": l3_data.get("stability", 0),
        "throughput": l3_data.get("throughput", 0),
        "adaptability": l3_data.get("adaptability", 0),
    }
    
    # Create level comparison chart
    print(f"\nCreating three-tier metrics comparison for: {scenario_id}...")
    chart = visualizer.create_level_comparison(
        l1_scores, l2_scores, l3_scores,
        title=f"APEE Three-Tier Evaluation ({scenario_id})"
    )
    
    # Handle both plotly and text output
    if hasattr(chart, 'to_html'):
        print("✅ Plotly chart created (interactive)")
        output_dir = Path(__file__).parent.parent / "data"
        output_dir.mkdir(exist_ok=True)
        chart.write_html(str(output_dir / "evaluation_chart.html"))
        print(f"   Saved to: data/evaluation_chart.html")
    else:
        print("📝 Text-based chart (plotly not installed):")
        print(chart.get("text_representation", ""))
    
    # Create agent performance radar with REAL data
    print("\nCreating agent performance radar...")
    agent_metrics = {}
    for agent_id, scores in l1_data.get("scores_by_agent", {}).items():
        agent_metrics[agent_id] = {
            "goal": scores.get("goal_alignment", {}).get("score", 0),
            "semantic": scores.get("semantic_quality", {}).get("score", 0),
            "latency": 7.0,  # Placeholder - could be calculated from duration
            "quality": (scores.get("goal_alignment", {}).get("score", 0) + 
                       scores.get("semantic_quality", {}).get("score", 0)) / 2,
        }
    
    radar = visualizer.create_agent_performance_radar(
        agent_metrics,
        title="Agent Performance Comparison (Real Data)"
    )
    
    if hasattr(radar, 'to_html'):
        print("✅ Radar chart created")
    else:
        print("📝 Text-based radar:")
        print(radar.get("text_representation", ""))
    
    # Generate full HTML report with REAL data
    print("\nGenerating comprehensive HTML report...")
    evaluation_result = {
        "overall_apee_score": first_scenario["overall_apee_score"],
        "l1_average": l1_data.get("average", 0),
        "l2_average": l2_data.get("average", 0),
        "l3_average": l3_data.get("overall", 0),
        "individual_scores": {
            agent_id: {
                "goal_alignment": scores.get("goal_alignment", {}).get("score", 0),
                "semantic_quality": scores.get("semantic_quality", {}).get("score", 0),
            }
            for agent_id, scores in l1_data.get("scores_by_agent", {}).items()
        },
        "collaborative_scores": l2_scores,
        "ecosystem_scores": l3_scores,
        "scenario_id": scenario_id,
        "pattern": first_scenario["pattern"],
        "judges": results["judge_models"],
        "timestamp": results["timestamp"],
    }
    
    output_path = Path(__file__).parent.parent / "data" / "evaluation_report.html"
    generate_report_html(
        evaluation_result,
        title=f"APEE Evaluation Report ({scenario_id})",
        output_path=str(output_path)
    )
    print(f"✅ Report saved to: {output_path}")
    
    # Show summary of all scenarios
    print(f"\n📊 All Scenario Results:")
    print("-" * 70)
    print(f"{'Scenario':<25} {'Pattern':<15} {'L1':>6} {'L2':>6} {'L3':>6} {'Overall':>8}")
    print("-" * 70)
    for scenario in results["scenarios"]:
        print(f"{scenario['scenario_id']:<25} {scenario['pattern']:<15} "
              f"{scenario['level1_individual']['average']:>5.1f} "
              f"{scenario['level2_collaborative']['average']:>5.1f} "
              f"{scenario['level3_ecosystem']['overall']:>5.1f} "
              f"{scenario['overall_apee_score']:>7.1f}")
    print("-" * 70)


def demo_anomaly_detection():
    """Demonstrate anomaly detection capabilities using REAL evaluation data."""
    print("\n" + "="*60)
    print("⚠️ ANOMALY DETECTION DEMO")
    print("="*60 + "\n")
    
    # Load real evaluation results
    results = load_evaluation_results()
    
    if not results:
        print("⚠️  No evaluation results found!")
        print("   Run 'python examples/proper_apee_evaluation.py' first.")
        return
    
    # Create detector
    detector = AnomalyDetector(
        window_size=50,
        z_threshold=2.5,
        enable_learning=True
    )
    
    # Create alert manager with console handler
    alert_manager = AlertManager()
    alert_manager.add_handler(ConsoleAlertHandler(min_severity=AnomalySeverity.INFO))
    
    scenarios = results["scenarios"]
    
    # Build baseline from all scenarios
    print("Building baseline from real evaluation results...")
    all_scores = [s["overall_apee_score"] for s in scenarios]
    for score in all_scores:
        detector.check_value("overall_apee_score", score)
    
    mean_score = sum(all_scores) / len(all_scores)
    min_score = min(all_scores)
    max_score = max(all_scores)
    print(f"   Baseline from {len(scenarios)} scenarios")
    print(f"   Score range: {min_score:.1f} - {max_score:.1f} (mean: {mean_score:.1f})")
    
    # Check each scenario for anomalies
    print("\nAnalyzing scenarios for anomalies...")
    total_anomalies = 0
    
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        l1_avg = scenario["level1_individual"]["average"]
        l2_avg = scenario["level2_collaborative"]["average"]
        l3_avg = scenario["level3_ecosystem"]["overall"]
        overall = scenario["overall_apee_score"]
        
        # Build evaluation result dict for anomaly checking
        eval_result = {
            "overall_apee_score": overall,
            "l1_average": l1_avg,
            "l2_average": l2_avg,
            "l3_average": l3_avg,
        }
        
        # Check for anomalies (using new detector to avoid baseline pollution)
        check_detector = AnomalyDetector(window_size=50, z_threshold=2.5)
        anomalies = check_detector.check_evaluation(eval_result, scenario_id)
        
        if anomalies:
            total_anomalies += len(anomalies)
            print(f"\n⚠️  {scenario_id}: {len(anomalies)} anomaly(ies)")
            for anomaly in anomalies:
                alert_manager.process_anomaly(anomaly)
        else:
            print(f"✅ {scenario_id}: Normal (L1={l1_avg:.1f}, L2={l2_avg:.1f}, L3={l3_avg:.1f}, Overall={overall:.1f})")
    
    # Print summary
    print(f"\n📊 Anomaly Summary:")
    print(f"   Scenarios analyzed: {len(scenarios)}")
    print(f"   Total anomalies found: {total_anomalies}")
    
    if total_anomalies == 0:
        print("   ✅ All scenarios within normal parameters!")


def demo_pattern_analyzers():
    """Demonstrate pattern analysis capabilities using REAL evaluation data."""
    print("\n" + "="*60)
    print("📈 PATTERN ANALYSIS DEMO")
    print("="*60 + "\n")
    
    # Load real evaluation results
    results = load_evaluation_results()
    
    if not results:
        print("⚠️  No evaluation results found!")
        print("   Run 'python examples/proper_apee_evaluation.py' first.")
        return
    
    scenarios = results["scenarios"]
    
    # Performance pattern analyzer
    print("Performance Pattern Analysis (Real Data):")
    perf_analyzer = PerformancePatternAnalyzer(min_samples=5)
    
    # Use real overall scores from evaluation
    scores = [s["overall_apee_score"] for s in scenarios]
    for i, score in enumerate(scores):
        perf_analyzer.record("overall_score", score, timestamp=i * 100)
    
    print(f"   Analyzed {len(scores)} scenarios")
    print(f"   Scores: {[f'{s:.1f}' for s in scores]}")
    
    trend = perf_analyzer.analyze_trend("overall_score")
    if trend:
        print(f"\n   Trend Analysis:")
        print(f"     Direction: {trend.direction}")
        print(f"     Slope: {trend.slope:.4f}")
        print(f"     Confidence: {trend.confidence:.2%}")
        
        if trend.direction == "stable":
            print("     ✅ Performance is stable across scenarios")
        elif trend.direction == "increasing":
            print("     📈 Performance improving across scenarios")
        else:
            print("     📉 Performance declining - may need investigation")
    
    anomalies = perf_analyzer.detect_anomalies("overall_score")
    if anomalies:
        print(f"\n   ⚠️ Detected {len(anomalies)} performance anomalies")
        for a in anomalies:
            print(f"      - {a.anomaly_type.value}: {a.description}")
    else:
        print(f"\n   ✅ No performance anomalies detected")
    
    # Quality pattern analyzer with L1 scores
    print("\n\nQuality Pattern Analysis (L1 Individual Scores):")
    quality_analyzer = QualityPatternAnalyzer()
    
    # Extract all L1 scores from real data
    l1_scores = [s["level1_individual"]["average"] for s in scenarios]
    for score in l1_scores:
        quality_analyzer.record("l1_quality", score)
    
    dist = quality_analyzer.analyze_distribution("l1_quality")
    print(f"   Distribution stats:")
    print(f"     Mean: {dist['mean']:.2f}")
    print(f"     Std: {dist['std']:.2f}")
    print(f"     Min: {dist['min']:.2f}")
    print(f"     Max: {dist['max']:.2f}")
    
    outliers = quality_analyzer.detect_outliers("l1_quality")
    if outliers:
        print(f"   Outliers detected: {len(outliers)}")
        for idx, val in outliers:
            print(f"     - Index {idx}: {val:.2f}")
    else:
        print(f"   ✅ No outliers detected")
    
    # Pattern by collaboration type
    print("\n\nPerformance by Collaboration Pattern:")
    pattern_scores = {}
    for s in scenarios:
        pattern = s["pattern"]
        if pattern not in pattern_scores:
            pattern_scores[pattern] = []
        pattern_scores[pattern].append(s["overall_apee_score"])
    
    for pattern, scores in sorted(pattern_scores.items()):
        avg = sum(scores) / len(scores)
        print(f"   {pattern}: avg={avg:.1f} (n={len(scores)})")


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
    """Demonstrate anomaly heatmap visualization using REAL evaluation data."""
    print("\n" + "="*60)
    print("🗺️ ANOMALY HEATMAP DEMO")
    print("="*60 + "\n")
    
    # Load real evaluation results
    results = load_evaluation_results()
    
    if not results:
        print("⚠️  No evaluation results found!")
        print("   Run 'python examples/proper_apee_evaluation.py' first.")
        return
    
    scenarios = results["scenarios"]
    
    # Calculate relative concern level for each metric
    # Higher concern = lower score relative to the ideal
    # Values are normalized 0-1 where 0=excellent (at max), 1=poor (at or below min)
    
    def calc_concern(value, ideal_min, ideal_max):
        """
        Calculate concern level (0-1 scale).
        0.0 = excellent (at or above ideal_max)
        0.5 = acceptable (at midpoint)
        1.0 = poor (at or below ideal_min)
        """
        if value >= ideal_max:
            return 0.0  # Excellent - at or above ideal
        if value <= ideal_min:
            return 1.0  # Poor - at or below minimum
        # Linear interpolation between min and max
        # Higher score = lower concern
        range_size = ideal_max - ideal_min
        position = (value - ideal_min) / range_size  # 0 to 1, higher = better
        return 1.0 - position  # Invert: 0 = good, 1 = bad
    
    anomaly_data = {}
    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        l1 = scenario["level1_individual"]["average"]
        l2 = scenario["level2_collaborative"]["average"]
        l3 = scenario["level3_ecosystem"]["overall"]
        overall = scenario["overall_apee_score"]
        
        anomaly_data[scenario_id] = {
            "L1_quality": calc_concern(l1, 5.0, 9.0),
            "L2_collab": calc_concern(l2, 4.0, 8.0),
            "L3_ecosystem": calc_concern(l3, 5.0, 9.0),
            "overall": calc_concern(overall, 5.0, 8.0),
        }
    
    print(f"Generated heatmap data for {len(anomaly_data)} scenarios")
    print("(Values: 0.0=excellent, 0.5=acceptable, 1.0=needs attention)\n")
    
    # Print text summary
    print(f"{'Scenario':<25} {'L1':>8} {'L2':>8} {'L3':>8} {'Overall':>8}")
    print("-" * 60)
    for scenario_id, metrics in anomaly_data.items():
        print(f"{scenario_id:<25} {metrics['L1_quality']:>7.2f} {metrics['L2_collab']:>7.2f} "
              f"{metrics['L3_ecosystem']:>7.2f} {metrics['overall']:>7.2f}")
    
    heatmap = create_anomaly_heatmap(anomaly_data)
    
    if hasattr(heatmap, 'to_html'):
        print("\n✅ Interactive heatmap created (plotly)")
        output_dir = Path(__file__).parent.parent / "data"
        heatmap.write_html(str(output_dir / "anomaly_heatmap.html"))
        print(f"   Saved to: data/anomaly_heatmap.html")
    else:
        print("\n📝 Text-based heatmap:")
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
