"""
APEE Dashboard Server.

A lightweight web server for the APEE dashboard using only stdlib.
"""

import http.server
import json
import os
import socketserver
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Callable
from urllib.parse import parse_qs, urlparse
import mimetypes


# Default port for the dashboard
DEFAULT_PORT = 8765


class DashboardState:
    """Shared state for the dashboard."""
    
    def __init__(self):
        self.evaluations: list[dict[str, Any]] = []
        self.anomalies: list[dict[str, Any]] = []
        self.agents: dict[str, dict[str, Any]] = {}
        self.scenarios: list[dict[str, Any]] = []
        self.last_updated: Optional[datetime] = None
    
    def add_evaluation(self, result: dict[str, Any]) -> None:
        """Add an evaluation result."""
        result["_timestamp"] = datetime.now().isoformat()
        self.evaluations.append(result)
        self.last_updated = datetime.now()
        
        # Keep only last 100 evaluations
        if len(self.evaluations) > 100:
            self.evaluations = self.evaluations[-100:]
    
    def add_anomaly(self, anomaly: dict[str, Any]) -> None:
        """Add an anomaly."""
        self.anomalies.append(anomaly)
        self.last_updated = datetime.now()
    
    def update_agent(self, agent_id: str, data: dict[str, Any]) -> None:
        """Update agent data."""
        self.agents[agent_id] = {
            **data,
            "_updated": datetime.now().isoformat()
        }
        self.last_updated = datetime.now()
    
    def add_scenario(self, scenario: dict[str, Any]) -> None:
        """Add a scenario result."""
        self.scenarios.append(scenario)
        self.last_updated = datetime.now()
    
    def get_summary(self) -> dict[str, Any]:
        """Get dashboard summary."""
        return {
            "total_evaluations": len(self.evaluations),
            "total_anomalies": len(self.anomalies),
            "active_agents": len(self.agents),
            "scenarios_run": len(self.scenarios),
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
        }
    
    def to_dict(self) -> dict[str, Any]:
        """Export all data."""
        return {
            "evaluations": self.evaluations,
            "anomalies": self.anomalies,
            "agents": self.agents,
            "scenarios": self.scenarios,
            "summary": self.get_summary(),
        }


# Global dashboard state
_dashboard_state = DashboardState()


def get_dashboard_state() -> DashboardState:
    """Get the global dashboard state."""
    return _dashboard_state


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP request handler for the dashboard."""
    
    def __init__(self, *args, **kwargs):
        self.state = get_dashboard_state()
        super().__init__(*args, **kwargs)
    
    def log_message(self, format: str, *args) -> None:
        """Suppress default logging."""
        pass
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # API endpoints
        if path.startswith("/api/"):
            self._handle_api(path, parsed.query)
            return
        
        # Serve static files or dashboard
        if path == "/" or path == "/index.html":
            self._serve_dashboard()
        elif path == "/style.css":
            self._serve_css()
        elif path == "/app.js":
            self._serve_js()
        else:
            self.send_error(404, "Not Found")
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path
        
        # Read body
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length).decode("utf-8") if content_length > 0 else ""
        
        try:
            data = json.loads(body) if body else {}
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return
        
        if path == "/api/evaluation":
            self.state.add_evaluation(data)
            self._send_json({"status": "ok", "message": "Evaluation added"})
        elif path == "/api/anomaly":
            self.state.add_anomaly(data)
            self._send_json({"status": "ok", "message": "Anomaly added"})
        elif path == "/api/agent":
            agent_id = data.get("agent_id", "unknown")
            self.state.update_agent(agent_id, data)
            self._send_json({"status": "ok", "message": f"Agent {agent_id} updated"})
        else:
            self.send_error(404, "Endpoint not found")
    
    def _handle_api(self, path: str, query: str) -> None:
        """Handle API requests."""
        params = parse_qs(query)
        
        if path == "/api/summary":
            self._send_json(self.state.get_summary())
        elif path == "/api/evaluations":
            limit = int(params.get("limit", [10])[0])
            self._send_json(self.state.evaluations[-limit:])
        elif path == "/api/anomalies":
            self._send_json(self.state.anomalies)
        elif path == "/api/agents":
            self._send_json(self.state.agents)
        elif path == "/api/scenarios":
            self._send_json(self.state.scenarios)
        elif path == "/api/export":
            self._send_json(self.state.to_dict())
        else:
            self.send_error(404, "API endpoint not found")
    
    def _send_json(self, data: Any) -> None:
        """Send JSON response."""
        content = json.dumps(data, indent=2, default=str).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)
    
    def _serve_dashboard(self) -> None:
        """Serve the main dashboard HTML."""
        html = _get_dashboard_html()
        content = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
    
    def _serve_css(self) -> None:
        """Serve dashboard CSS."""
        css = _get_dashboard_css()
        content = css.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/css")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)
    
    def _serve_js(self) -> None:
        """Serve dashboard JavaScript."""
        js = _get_dashboard_js()
        content = js.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/javascript")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


class DashboardServer:
    """
    APEE Dashboard Server.
    
    A simple web server for viewing evaluation results in real-time.
    """
    
    def __init__(self, port: int = DEFAULT_PORT, host: str = "localhost"):
        self.port = port
        self.host = host
        self.server: Optional[socketserver.TCPServer] = None
        self._thread: Optional[threading.Thread] = None
        self.state = get_dashboard_state()
    
    def start(self, open_browser: bool = True) -> None:
        """
        Start the dashboard server.
        
        Args:
            open_browser: Whether to open the dashboard in a browser
        """
        self.server = socketserver.TCPServer(
            (self.host, self.port),
            DashboardHandler
        )
        
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self._thread.start()
        
        url = f"http://{self.host}:{self.port}"
        print(f"🚀 APEE Dashboard running at {url}")
        
        if open_browser:
            webbrowser.open(url)
    
    def stop(self) -> None:
        """Stop the dashboard server."""
        if self.server:
            self.server.shutdown()
            self.server = None
            self._thread = None
            print("Dashboard server stopped")
    
    def add_evaluation(self, result: dict[str, Any]) -> None:
        """Add an evaluation result to the dashboard."""
        self.state.add_evaluation(result)
    
    def add_anomaly(self, anomaly: dict[str, Any]) -> None:
        """Add an anomaly to the dashboard."""
        self.state.add_anomaly(anomaly)
    
    def update_agent(self, agent_id: str, data: dict[str, Any]) -> None:
        """Update agent information."""
        self.state.update_agent(agent_id, data)
    
    @property
    def url(self) -> str:
        """Get dashboard URL."""
        return f"http://{self.host}:{self.port}"


def create_dashboard(
    port: int = DEFAULT_PORT,
    host: str = "localhost",
    open_browser: bool = True
) -> DashboardServer:
    """
    Create and start a dashboard server.
    
    Args:
        port: Port to run on
        host: Host to bind to
        open_browser: Whether to open in browser
        
    Returns:
        Running DashboardServer instance
    """
    server = DashboardServer(port=port, host=host)
    server.start(open_browser=open_browser)
    return server


def _get_dashboard_html() -> str:
    """Get the dashboard HTML content."""
    return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>APEE Dashboard</title>
    <link rel="stylesheet" href="/style.css">
</head>
<body>
    <div class="container">
        <header>
            <h1>🎯 APEE Dashboard</h1>
            <p class="subtitle">Adaptive Poly-Agentic Evaluation Ecosystem</p>
            <div class="refresh-info">
                <span id="last-update">Last updated: -</span>
                <button id="refresh-btn" onclick="refreshData()">🔄 Refresh</button>
            </div>
        </header>
        
        <div class="summary-cards">
            <div class="card">
                <div class="card-icon">📊</div>
                <div class="card-value" id="eval-count">0</div>
                <div class="card-label">Evaluations</div>
            </div>
            <div class="card">
                <div class="card-icon">🤖</div>
                <div class="card-value" id="agent-count">0</div>
                <div class="card-label">Active Agents</div>
            </div>
            <div class="card">
                <div class="card-icon">⚠️</div>
                <div class="card-value" id="anomaly-count">0</div>
                <div class="card-label">Anomalies</div>
            </div>
            <div class="card">
                <div class="card-icon">🎭</div>
                <div class="card-value" id="scenario-count">0</div>
                <div class="card-label">Scenarios</div>
            </div>
        </div>
        
        <div class="grid">
            <section class="panel">
                <h2>📈 Recent Evaluations</h2>
                <div id="evaluations-list" class="list-container">
                    <p class="placeholder">No evaluations yet</p>
                </div>
            </section>
            
            <section class="panel">
                <h2>🤖 Agents</h2>
                <div id="agents-list" class="list-container">
                    <p class="placeholder">No agents registered</p>
                </div>
            </section>
        </div>
        
        <section class="panel full-width">
            <h2>⚠️ Anomalies</h2>
            <div id="anomalies-list" class="list-container">
                <p class="placeholder">No anomalies detected</p>
            </div>
        </section>
        
        <section class="panel full-width">
            <h2>📋 Evaluation Details</h2>
            <div id="eval-details" class="details-container">
                <p class="placeholder">Select an evaluation to view details</p>
            </div>
        </section>
        
        <footer>
            <p>APEE - <a href="https://ahjavid.github.io/technical-notes-blog/" target="_blank">Technical Notes Blog</a></p>
        </footer>
    </div>
    
    <script src="/app.js"></script>
</body>
</html>'''


def _get_dashboard_css() -> str:
    """Get the dashboard CSS content."""
    return '''
:root {
    --primary: #667eea;
    --secondary: #764ba2;
    --success: #2ecc71;
    --warning: #f39c12;
    --danger: #e74c3c;
    --dark: #2c3e50;
    --light: #ecf0f1;
    --bg: #f5f7fa;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg);
    color: var(--dark);
    line-height: 1.6;
}

.container {
    max-width: 1400px;
    margin: 0 auto;
    padding: 20px;
}

header {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    padding: 30px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

header h1 {
    font-size: 2.5rem;
    margin-bottom: 8px;
}

.subtitle {
    opacity: 0.9;
    font-size: 1.1rem;
}

.refresh-info {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-top: 16px;
}

#refresh-btn {
    background: rgba(255,255,255,0.2);
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 8px;
    cursor: pointer;
    font-size: 14px;
    transition: background 0.2s;
}

#refresh-btn:hover {
    background: rgba(255,255,255,0.3);
}

.summary-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 20px;
    margin-bottom: 24px;
}

.card {
    background: white;
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    transition: transform 0.2s, box-shadow 0.2s;
}

.card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}

.card-icon {
    font-size: 2rem;
    margin-bottom: 8px;
}

.card-value {
    font-size: 2.5rem;
    font-weight: bold;
    color: var(--primary);
}

.card-label {
    color: #666;
    font-size: 0.9rem;
    margin-top: 4px;
}

.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
    gap: 24px;
    margin-bottom: 24px;
}

.panel {
    background: white;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

.panel.full-width {
    grid-column: 1 / -1;
}

.panel h2 {
    color: var(--dark);
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 2px solid var(--light);
}

.list-container {
    max-height: 400px;
    overflow-y: auto;
}

.placeholder {
    color: #999;
    text-align: center;
    padding: 40px;
}

.eval-item, .agent-item, .anomaly-item {
    padding: 16px;
    border-radius: 12px;
    margin-bottom: 12px;
    background: var(--light);
    cursor: pointer;
    transition: background 0.2s;
}

.eval-item:hover, .agent-item:hover {
    background: #e0e5ec;
}

.eval-item.selected {
    background: rgba(102, 126, 234, 0.2);
    border-left: 4px solid var(--primary);
}

.eval-score {
    display: flex;
    align-items: center;
    gap: 12px;
}

.score-circle {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    color: white;
    font-size: 1.1rem;
}

.score-good { background: var(--success); }
.score-medium { background: var(--warning); }
.score-poor { background: var(--danger); }

.eval-meta {
    flex: 1;
}

.eval-meta h4 {
    margin-bottom: 4px;
}

.eval-meta small {
    color: #666;
}

.anomaly-item {
    border-left: 4px solid var(--warning);
}

.anomaly-item.critical {
    border-left-color: var(--danger);
    background: rgba(231, 76, 60, 0.1);
}

.anomaly-type {
    font-weight: bold;
    text-transform: uppercase;
    font-size: 0.8rem;
    margin-bottom: 4px;
}

.agent-item {
    display: flex;
    align-items: center;
    gap: 16px;
}

.agent-icon {
    font-size: 1.5rem;
}

.agent-info {
    flex: 1;
}

.agent-role {
    font-size: 0.85rem;
    color: #666;
}

.details-container {
    background: var(--light);
    border-radius: 12px;
    padding: 20px;
    min-height: 200px;
}

.details-container pre {
    background: var(--dark);
    color: #f8f8f2;
    padding: 16px;
    border-radius: 8px;
    overflow-x: auto;
    font-size: 0.9rem;
}

.metric-bar {
    height: 12px;
    background: #ddd;
    border-radius: 6px;
    overflow: hidden;
    margin: 8px 0;
}

.metric-fill {
    height: 100%;
    border-radius: 6px;
    transition: width 0.5s ease;
}

.metric-fill.l1 { background: var(--primary); }
.metric-fill.l2 { background: var(--success); }
.metric-fill.l3 { background: var(--secondary); }

footer {
    text-align: center;
    padding: 24px;
    color: #666;
}

footer a {
    color: var(--primary);
}

@media (max-width: 768px) {
    .grid {
        grid-template-columns: 1fr;
    }
    
    header h1 {
        font-size: 1.8rem;
    }
}
'''


def _get_dashboard_js() -> str:
    """Get the dashboard JavaScript content."""
    return '''
// APEE Dashboard JavaScript

let selectedEvalId = null;

async function fetchAPI(endpoint) {
    try {
        const response = await fetch('/api/' + endpoint);
        return await response.json();
    } catch (error) {
        console.error('API Error:', error);
        return null;
    }
}

async function refreshData() {
    const summary = await fetchAPI('summary');
    const evaluations = await fetchAPI('evaluations?limit=20');
    const agents = await fetchAPI('agents');
    const anomalies = await fetchAPI('anomalies');
    
    if (summary) {
        document.getElementById('eval-count').textContent = summary.total_evaluations || 0;
        document.getElementById('agent-count').textContent = summary.active_agents || 0;
        document.getElementById('anomaly-count').textContent = summary.total_anomalies || 0;
        document.getElementById('scenario-count').textContent = summary.scenarios_run || 0;
        document.getElementById('last-update').textContent = 
            'Last updated: ' + (summary.last_updated ? new Date(summary.last_updated).toLocaleTimeString() : '-');
    }
    
    renderEvaluations(evaluations || []);
    renderAgents(agents || {});
    renderAnomalies(anomalies || []);
}

function renderEvaluations(evaluations) {
    const container = document.getElementById('evaluations-list');
    
    if (!evaluations.length) {
        container.innerHTML = '<p class="placeholder">No evaluations yet</p>';
        return;
    }
    
    container.innerHTML = evaluations.reverse().map((eval, idx) => {
        const score = eval.overall_apee_score || 0;
        const scoreClass = score >= 7 ? 'score-good' : score >= 5 ? 'score-medium' : 'score-poor';
        const scenario = eval.scenario_id || 'Evaluation ' + (evaluations.length - idx);
        const time = eval._timestamp ? new Date(eval._timestamp).toLocaleTimeString() : '';
        const selected = selectedEvalId === idx ? 'selected' : '';
        
        return `
            <div class="eval-item ${selected}" onclick="showEvalDetails(${idx}, ${JSON.stringify(eval).replace(/"/g, '&quot;')})">
                <div class="eval-score">
                    <div class="score-circle ${scoreClass}">${score.toFixed(1)}</div>
                    <div class="eval-meta">
                        <h4>${scenario}</h4>
                        <small>${time}</small>
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function renderAgents(agents) {
    const container = document.getElementById('agents-list');
    const agentList = Object.entries(agents);
    
    if (!agentList.length) {
        container.innerHTML = '<p class="placeholder">No agents registered</p>';
        return;
    }
    
    container.innerHTML = agentList.map(([id, data]) => {
        const role = data.role || 'unknown';
        const model = data.model || 'unknown';
        return `
            <div class="agent-item">
                <div class="agent-icon">🤖</div>
                <div class="agent-info">
                    <strong>${id}</strong>
                    <div class="agent-role">${role} | ${model}</div>
                </div>
            </div>
        `;
    }).join('');
}

function renderAnomalies(anomalies) {
    const container = document.getElementById('anomalies-list');
    
    if (!anomalies.length) {
        container.innerHTML = '<p class="placeholder">No anomalies detected ✓</p>';
        return;
    }
    
    container.innerHTML = anomalies.slice(-10).reverse().map(anomaly => {
        const isCritical = anomaly.severity === 'critical' || anomaly.severity === 'emergency';
        return `
            <div class="anomaly-item ${isCritical ? 'critical' : ''}">
                <div class="anomaly-type">${anomaly.severity || 'warning'}: ${anomaly.type || 'anomaly'}</div>
                <div>${anomaly.description || 'Unknown anomaly'}</div>
                <small>Metric: ${anomaly.metric || '-'} | Deviation: ${(anomaly.deviation * 100 || 0).toFixed(0)}%</small>
            </div>
        `;
    }).join('');
}

function showEvalDetails(idx, evalData) {
    selectedEvalId = idx;
    const container = document.getElementById('eval-details');
    
    // Re-render list to show selection
    fetchAPI('evaluations?limit=20').then(evals => renderEvaluations(evals || []));
    
    if (!evalData) {
        container.innerHTML = '<p class="placeholder">No data available</p>';
        return;
    }
    
    const l1 = evalData.l1_average || 0;
    const l2 = evalData.l2_average || 0;
    const l3 = evalData.l3_average || 0;
    
    container.innerHTML = `
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin-bottom: 20px;">
            <div>
                <strong>L1 Individual</strong>
                <div class="metric-bar"><div class="metric-fill l1" style="width: ${l1 * 10}%"></div></div>
                <span>${l1.toFixed(1)}/10</span>
            </div>
            <div>
                <strong>L2 Collaborative</strong>
                <div class="metric-bar"><div class="metric-fill l2" style="width: ${l2 * 10}%"></div></div>
                <span>${l2.toFixed(1)}/10</span>
            </div>
            <div>
                <strong>L3 Ecosystem</strong>
                <div class="metric-bar"><div class="metric-fill l3" style="width: ${l3 * 10}%"></div></div>
                <span>${l3.toFixed(1)}/10</span>
            </div>
        </div>
        <details>
            <summary style="cursor: pointer; padding: 8px;">View Raw JSON</summary>
            <pre>${JSON.stringify(evalData, null, 2)}</pre>
        </details>
    `;
}

// Initial load
refreshData();

// Auto-refresh every 5 seconds
setInterval(refreshData, 5000);
'''


def main():
    """CLI entry point for starting the dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="APEE Dashboard - Web interface for evaluation results"
    )
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port to run the dashboard on (default: {DEFAULT_PORT})"
    )
    parser.add_argument(
        "--host", "-H",
        type=str,
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    
    args = parser.parse_args()
    
    server = DashboardServer(port=args.port, host=args.host)
    server.start(open_browser=not args.no_browser)
    
    print("\nPress Ctrl+C to stop the dashboard...\n")
    
    try:
        # Keep running until interrupted
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == "__main__":
    main()
