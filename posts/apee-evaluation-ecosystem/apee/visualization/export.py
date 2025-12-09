"""
APEE Export Utilities.

Export charts and reports to various formats.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


def export_to_html(
    figure: Any,
    output_path: str,
    include_plotlyjs: bool = True,
    full_html: bool = True
) -> str:
    """
    Export a plotly figure to HTML file.
    
    Args:
        figure: Plotly figure object
        output_path: Output file path
        include_plotlyjs: Include plotly.js in output
        full_html: Create full HTML document
        
    Returns:
        Path to created file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if it's a plotly figure
    if hasattr(figure, 'to_html'):
        html_content = figure.to_html(
            include_plotlyjs=include_plotlyjs,
            full_html=full_html
        )
        path.write_text(html_content)
        return str(path)
    
    # Handle dict representation
    if isinstance(figure, dict):
        html_content = _dict_to_html(figure, full_html)
        path.write_text(html_content)
        return str(path)
    
    raise ValueError(f"Cannot export figure of type {type(figure)}")


def export_to_png(
    figure: Any,
    output_path: str,
    width: int = 800,
    height: int = 600,
    scale: float = 2.0
) -> str:
    """
    Export a plotly figure to PNG image.
    
    Args:
        figure: Plotly figure object
        output_path: Output file path
        width: Image width
        height: Image height
        scale: Scale factor for resolution
        
    Returns:
        Path to created file
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if hasattr(figure, 'write_image'):
        try:
            figure.write_image(
                str(path),
                width=width,
                height=height,
                scale=scale
            )
            return str(path)
        except Exception as e:
            # kaleido not installed
            raise RuntimeError(
                f"Cannot export to PNG. Install kaleido: pip install kaleido. Error: {e}"
            )
    
    raise ValueError(f"Cannot export figure of type {type(figure)} to PNG")


def _dict_to_html(data: dict, full_html: bool = True) -> str:
    """Convert dict chart representation to HTML."""
    text_rep = data.get("text_representation", "")
    chart_type = data.get("type", "unknown")
    title = data.get("title", "APEE Chart")
    
    content = f"""
    <div class="apee-chart" data-type="{chart_type}">
        <h3>{title}</h3>
        <pre class="chart-text">{text_rep}</pre>
        <details>
            <summary>Raw Data</summary>
            <pre class="chart-data">{json.dumps(data.get('data', {}), indent=2)}</pre>
        </details>
    </div>
    """
    
    if full_html:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 20px; background: #f5f5f5; }}
        .apee-chart {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        .chart-text {{ font-family: monospace; white-space: pre; background: #1a1a2e; color: #16f6b0; padding: 15px; border-radius: 4px; overflow-x: auto; }}
        .chart-data {{ font-size: 12px; background: #f0f0f0; padding: 10px; border-radius: 4px; }}
        details {{ margin-top: 15px; }}
        summary {{ cursor: pointer; color: #666; }}
    </style>
</head>
<body>
    {content}
</body>
</html>"""
    
    return content


def generate_report_html(
    evaluation_results: dict[str, Any],
    charts: Optional[list[Any]] = None,
    title: str = "APEE Evaluation Report",
    output_path: Optional[str] = None
) -> str:
    """
    Generate comprehensive HTML report with all evaluation data.
    
    Args:
        evaluation_results: Full evaluation results dict
        charts: Optional list of chart figures to include
        title: Report title
        output_path: If provided, save to this path
        
    Returns:
        HTML content string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Extract key metrics
    overall_score = evaluation_results.get("overall_apee_score", 0)
    l1_avg = evaluation_results.get("l1_average", 0)
    l2_avg = evaluation_results.get("l2_average", 0)
    l3_avg = evaluation_results.get("l3_average", 0)
    
    individual_scores = evaluation_results.get("individual_scores", {})
    collab_scores = evaluation_results.get("collaborative_scores", {})
    ecosystem_scores = evaluation_results.get("ecosystem_scores", {})
    
    # Build agent cards
    agent_cards = ""
    for agent_id, scores in individual_scores.items():
        goal = scores.get("goal_alignment", 0)
        semantic = scores.get("semantic_quality", 0)
        agent_cards += f"""
        <div class="metric-card">
            <h4>{agent_id}</h4>
            <div class="metric-row">
                <span>Goal Alignment</span>
                <div class="metric-bar"><div class="fill" style="width: {goal*10}%"></div></div>
                <span class="score">{goal:.1f}</span>
            </div>
            <div class="metric-row">
                <span>Semantic Quality</span>
                <div class="metric-bar"><div class="fill" style="width: {semantic*10}%"></div></div>
                <span class="score">{semantic:.1f}</span>
            </div>
        </div>
        """
    
    # Build collaborative metrics
    collab_html = ""
    for metric, value in collab_scores.items():
        collab_html += f"""
        <div class="metric-row">
            <span>{metric.replace('_', ' ').title()}</span>
            <div class="metric-bar collaborative"><div class="fill" style="width: {value*10}%"></div></div>
            <span class="score">{value:.1f}</span>
        </div>
        """
    
    # Build ecosystem metrics
    ecosystem_html = ""
    for metric, value in ecosystem_scores.items():
        ecosystem_html += f"""
        <div class="metric-row">
            <span>{metric.replace('_', ' ').title()}</span>
            <div class="metric-bar ecosystem"><div class="fill" style="width: {value*10}%"></div></div>
            <span class="score">{value:.1f}</span>
        </div>
        """
    
    # Include charts
    charts_html = ""
    if charts:
        for i, chart in enumerate(charts):
            if hasattr(chart, 'to_html'):
                charts_html += f"""
                <div class="chart-container">
                    {chart.to_html(include_plotlyjs='cdn' if i == 0 else False, full_html=False)}
                </div>
                """
            elif isinstance(chart, dict) and "text_representation" in chart:
                charts_html += f"""
                <div class="chart-container text-chart">
                    <pre>{chart['text_representation']}</pre>
                </div>
                """
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        :root {{
            --primary: #3498db;
            --success: #2ecc71;
            --warning: #f39c12;
            --danger: #e74c3c;
            --purple: #9b59b6;
            --dark: #2c3e50;
            --light: #ecf0f1;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            background: white;
            border-radius: 16px;
            padding: 30px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
        }}
        
        h1 {{
            color: var(--dark);
            font-size: 2.5rem;
            margin-bottom: 10px;
        }}
        
        .timestamp {{
            color: #666;
            font-size: 0.9rem;
        }}
        
        .score-hero {{
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 60px;
            margin: 30px 0;
        }}
        
        .overall-score {{
            text-align: center;
        }}
        
        .score-circle {{
            width: 150px;
            height: 150px;
            border-radius: 50%;
            background: conic-gradient(
                var(--primary) {overall_score * 10}%,
                var(--light) {overall_score * 10}%
            );
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}
        
        .score-inner {{
            width: 120px;
            height: 120px;
            border-radius: 50%;
            background: white;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }}
        
        .score-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--dark);
        }}
        
        .score-label {{
            font-size: 0.8rem;
            color: #666;
        }}
        
        .level-scores {{
            display: flex;
            flex-direction: column;
            gap: 15px;
        }}
        
        .level-score {{
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .level-badge {{
            width: 60px;
            padding: 5px;
            text-align: center;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.8rem;
            color: white;
        }}
        
        .level-badge.l1 {{ background: var(--primary); }}
        .level-badge.l2 {{ background: var(--success); }}
        .level-badge.l3 {{ background: var(--purple); }}
        
        .level-bar {{
            flex: 1;
            height: 12px;
            background: var(--light);
            border-radius: 6px;
            overflow: hidden;
        }}
        
        .level-fill {{
            height: 100%;
            border-radius: 6px;
            transition: width 0.5s ease;
        }}
        
        .level-fill.l1 {{ background: var(--primary); }}
        .level-fill.l2 {{ background: var(--success); }}
        .level-fill.l3 {{ background: var(--purple); }}
        
        .section {{
            background: white;
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: var(--dark);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--light);
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
        }}
        
        .metric-card {{
            background: var(--light);
            border-radius: 12px;
            padding: 20px;
        }}
        
        .metric-card h4 {{
            color: var(--dark);
            margin-bottom: 15px;
            text-transform: capitalize;
        }}
        
        .metric-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }}
        
        .metric-row span:first-child {{
            width: 140px;
            font-size: 0.85rem;
            color: #555;
        }}
        
        .metric-bar {{
            flex: 1;
            height: 10px;
            background: white;
            border-radius: 5px;
            overflow: hidden;
        }}
        
        .metric-bar .fill {{
            height: 100%;
            background: var(--primary);
            border-radius: 5px;
            transition: width 0.5s ease;
        }}
        
        .metric-bar.collaborative .fill {{
            background: var(--success);
        }}
        
        .metric-bar.ecosystem .fill {{
            background: var(--purple);
        }}
        
        .score {{
            width: 40px;
            text-align: right;
            font-weight: bold;
            color: var(--dark);
        }}
        
        .chart-container {{
            margin-top: 20px;
            padding: 15px;
            background: #fafafa;
            border-radius: 8px;
        }}
        
        .text-chart pre {{
            font-family: 'Fira Code', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
        }}
        
        footer {{
            text-align: center;
            padding: 20px;
            color: white;
            opacity: 0.8;
        }}
        
        footer a {{
            color: white;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 {title}</h1>
            <p class="timestamp">Generated: {timestamp}</p>
            
            <div class="score-hero">
                <div class="overall-score">
                    <div class="score-circle">
                        <div class="score-inner">
                            <span class="score-value">{overall_score:.1f}</span>
                            <span class="score-label">Overall Score</span>
                        </div>
                    </div>
                </div>
                
                <div class="level-scores">
                    <div class="level-score">
                        <span class="level-badge l1">L1</span>
                        <div class="level-bar"><div class="level-fill l1" style="width: {l1_avg * 10}%"></div></div>
                        <span class="score">{l1_avg:.1f}</span>
                    </div>
                    <div class="level-score">
                        <span class="level-badge l2">L2</span>
                        <div class="level-bar"><div class="level-fill l2" style="width: {l2_avg * 10}%"></div></div>
                        <span class="score">{l2_avg:.1f}</span>
                    </div>
                    <div class="level-score">
                        <span class="level-badge l3">L3</span>
                        <div class="level-bar"><div class="level-fill l3" style="width: {l3_avg * 10}%"></div></div>
                        <span class="score">{l3_avg:.1f}</span>
                    </div>
                </div>
            </div>
        </header>
        
        <section class="section">
            <h2>📊 Level 1: Individual Agent Metrics</h2>
            <div class="metrics-grid">
                {agent_cards if agent_cards else '<p>No individual agent data available</p>'}
            </div>
        </section>
        
        <section class="section">
            <h2>🤝 Level 2: Collaborative Metrics</h2>
            {collab_html if collab_html else '<p>No collaborative data available</p>'}
        </section>
        
        <section class="section">
            <h2>🌐 Level 3: Ecosystem Metrics</h2>
            {ecosystem_html if ecosystem_html else '<p>No ecosystem data available</p>'}
        </section>
        
        {f'<section class="section"><h2>📈 Charts</h2>{charts_html}</section>' if charts_html else ''}
        
        <section class="section">
            <h2>📋 Raw Data</h2>
            <details>
                <summary style="cursor: pointer; padding: 10px;">Click to expand JSON data</summary>
                <pre style="background: #1a1a2e; color: #16f6b0; padding: 15px; border-radius: 8px; margin-top: 10px; overflow-x: auto;">{json.dumps(evaluation_results, indent=2, default=str)}</pre>
            </details>
        </section>
        
        <footer>
            <p>APEE - Adaptive Poly-Agentic Evaluation Ecosystem</p>
            <p><a href="https://ahjavid.github.io/technical-notes-blog/">Technical Notes Blog</a></p>
        </footer>
    </div>
</body>
</html>"""
    
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html)
    
    return html
