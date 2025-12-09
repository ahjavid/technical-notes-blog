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
    
    def load_from_json(self, filepath: str) -> bool:
        """Load evaluation results from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load scenarios as evaluations
            if 'scenarios' in data:
                for scenario in data['scenarios']:
                    eval_entry = {
                        'scenario_id': scenario.get('scenario_id', 'unknown'),
                        'pattern': scenario.get('pattern', 'unknown'),
                        'overall_score': scenario.get('overall_apee_score', 0),
                        'l1_average': scenario.get('level1_individual', {}).get('average', 0),
                        'l2_average': scenario.get('level2_collaborative', {}).get('average', 0),
                        'l3_average': scenario.get('level3_ecosystem', {}).get('overall', 0),
                        'duration': scenario.get('duration_seconds', 0),
                        'details': scenario,
                        '_timestamp': data.get('timestamp', datetime.now().isoformat())
                    }
                    self.evaluations.append(eval_entry)
                    self.scenarios.append(scenario)
            
            # Load agents from agent_models
            if 'agent_models' in data:
                for role, model in data['agent_models'].items():
                    self.agents[role] = {
                        'role': role,
                        'model': model,
                        'status': 'active',
                        '_updated': data.get('timestamp', datetime.now().isoformat())
                    }
            
            # Load summary stats
            if 'summary_statistics' in data:
                stats = data['summary_statistics']
                # Check for any anomalies based on low scores
                if stats.get('min_score', 10) < 5.0:
                    self.anomalies.append({
                        'type': 'low_score',
                        'severity': 'warning',
                        'description': f"Low score detected: {stats.get('min_score', 0):.1f}/10",
                        'metric': 'overall_score',
                        'deviation': (5.0 - stats.get('min_score', 0)) / 5.0
                    })
            
            self.last_updated = datetime.now()
            return True
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return False


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
        # Use ThreadingTCPServer for better concurrency
        class ThreadedServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
            allow_reuse_address = True
        
        self.server = ThreadedServer(
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
            <div class="header-row">
                <div>
                    <h1>🎯 APEE Dashboard</h1>
                    <span class="subtitle">Adaptive Poly-Agentic Evaluation Ecosystem</span>
                </div>
                <div class="header-stats">
                    <span class="stat"><strong id="eval-count">0</strong> evals</span>
                    <span class="stat"><strong id="agent-count">0</strong> agents</span>
                    <span class="stat"><strong id="anomaly-count">0</strong> anomalies</span>
                    <button id="refresh-btn" onclick="refreshData()">🔄</button>
                </div>
            </div>
        </header>
        
        <div class="main-grid">
            <section class="panel eval-panel">
                <h2>📈 Evaluations</h2>
                <div id="evaluations-list" class="list-container">
                    <p class="placeholder">No evaluations yet</p>
                </div>
            </section>
            
            <section class="panel details-panel">
                <h2>📋 Details</h2>
                <div id="eval-details" class="details-container">
                    <p class="placeholder">Click an evaluation</p>
                </div>
            </section>
            
            <section class="panel agents-panel">
                <h2>🤖 Agents</h2>
                <div id="agents-list" class="list-container small">
                    <p class="placeholder">No agents</p>
                </div>
                <h2 style="margin-top:12px;">⚠️ Anomalies <span id="anomaly-count" style="font-size:12px;color:#888;">(0)</span></h2>
                <div id="anomalies-list" class="list-container small">
                    <p class="placeholder">None ✓</p>
                </div>
            </section>
        </div>
        
        <footer>
            <span id="last-update">-</span> | <a href="https://ahjavid.github.io/technical-notes-blog/" target="_blank">APEE Blog</a>
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
    --light: #f0f2f5;
    --bg: #f5f7fa;
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--dark);
    font-size: 13px;
    height: 100vh;
    overflow: hidden;
}

.container {
    height: 100vh;
    display: flex;
    flex-direction: column;
    padding: 10px;
    gap: 10px;
}

header {
    background: linear-gradient(135deg, var(--primary), var(--secondary));
    color: white;
    padding: 12px 20px;
    border-radius: 10px;
    flex-shrink: 0;
}

.header-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
}

header h1 { font-size: 1.3rem; display: inline; }
.subtitle { opacity: 0.8; font-size: 0.85rem; margin-left: 10px; }

.header-stats {
    display: flex;
    gap: 15px;
    align-items: center;
}

.stat { background: rgba(255,255,255,0.2); padding: 4px 10px; border-radius: 15px; font-size: 12px; }
.stat strong { font-size: 14px; }

#refresh-btn {
    background: rgba(255,255,255,0.2);
    border: none;
    padding: 6px 12px;
    border-radius: 6px;
    cursor: pointer;
    font-size: 14px;
}
#refresh-btn:hover { background: rgba(255,255,255,0.3); }

.main-grid {
    display: grid;
    grid-template-columns: 260px 1fr 240px;
    gap: 10px;
    flex: 1;
    min-height: 0;
}

.panel {
    background: white;
    border-radius: 10px;
    padding: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    display: flex;
    flex-direction: column;
    min-height: 0;
}

.panel h2 {
    font-size: 0.95rem;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--light);
    flex-shrink: 0;
}

.list-container {
    flex: 1;
    overflow-y: auto;
    min-height: 0;
}

.list-container.small { font-size: 13px; }

.placeholder { color: #999; text-align: center; padding: 15px; font-size: 12px; }

.eval-item {
    padding: 8px 10px;
    border-radius: 6px;
    margin-bottom: 6px;
    background: var(--light);
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 10px;
}
.eval-item:hover { background: #e4e7eb; }
.eval-item.selected { background: rgba(102,126,234,0.15); border-left: 3px solid var(--primary); }

.agent-item {
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 8px;
    background: var(--light);
}
.agent-item strong { font-size: 13px; display: block; margin-bottom: 2px; }
.agent-role { color: #666; font-size: 12px; }

.score-circle {
    width: 38px;
    height: 38px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 12px;
    color: white;
    flex-shrink: 0;
}
.score-good { background: var(--success); }
.score-medium { background: var(--warning); }
.score-poor { background: var(--danger); }

.eval-meta h4 { font-size: 12px; font-weight: 600; }
.eval-meta small { color: #888; font-size: 11px; }

.details-container { flex: 1; overflow-y: auto; }

.metric-bar {
    height: 8px;
    background: #e0e0e0;
    border-radius: 4px;
    margin: 4px 0;
    overflow: hidden;
}
.metric-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s;
}
.metric-fill.l1 { background: var(--primary); }
.metric-fill.l2 { background: var(--secondary); }
.metric-fill.l3 { background: var(--success); }

footer {
    text-align: center;
    padding: 8px;
    font-size: 11px;
    color: #888;
    flex-shrink: 0;
}
footer a { color: var(--primary); }

@media (max-width: 900px) {
    .main-grid { grid-template-columns: 1fr; }
}
'''


def _get_dashboard_js() -> str:
    """Get the dashboard JavaScript content."""
    return '''
// APEE Dashboard JavaScript

let selectedEvalId = null;
let evalDataCache = [];

// Format snake_case to Title Case
function formatName(str) {
    if (!str) return '';
    return str.replace(/_/g, ' ').replace(/\\b\\w/g, c => c.toUpperCase());
}

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
        document.getElementById('anomaly-count').textContent = '(' + (summary.total_anomalies || 0) + ')';
        document.getElementById('last-update').textContent = summary.last_updated ? new Date(summary.last_updated).toLocaleTimeString() : '-';
    }
    
    renderEvaluations(evaluations || []);
    renderAgents(agents || {});
    renderAnomalies(anomalies || []);
}

function renderEvaluations(evaluations) {
    const container = document.getElementById('evaluations-list');
    
    if (!evaluations.length) {
        container.innerHTML = '<p class="placeholder">No evaluations</p>';
        return;
    }
    
    evalDataCache = [...evaluations].reverse();
    
    container.innerHTML = evalDataCache.map((ev, idx) => {
        const score = ev.overall_score || ev.overall_apee_score || 0;
        const scoreClass = score >= 7 ? 'score-good' : score >= 5 ? 'score-medium' : 'score-poor';
        const scenario = formatName(ev.scenario_id || 'Eval ' + idx);
        const pattern = formatName(ev.pattern || '');
        const selected = selectedEvalId === idx ? 'selected' : '';
        
        return `<div class="eval-item ${selected}" onclick="showEvalDetails(${idx})">
            <div class="score-circle ${scoreClass}">${score.toFixed(1)}</div>
            <div class="eval-meta"><h4>${scenario}</h4><small>${pattern}</small></div>
        </div>`;
    }).join('');
}

function renderAgents(agents) {
    const container = document.getElementById('agents-list');
    const list = Object.entries(agents);
    
    if (!list.length) {
        container.innerHTML = '<p class="placeholder">No agents</p>';
        return;
    }
    
    container.innerHTML = list.map(([id, d]) => `
        <div class="agent-item">
            <strong>🤖 ${formatName(id)}</strong>
            <div class="agent-role">${d.model || ''}</div>
        </div>
    `).join('');
}

function renderAnomalies(anomalies) {
    const container = document.getElementById('anomalies-list');
    
    if (!anomalies || !anomalies.length) {
        container.innerHTML = '<p class="placeholder" style="padding:10px;">None ✓</p>';
        return;
    }
    
    container.innerHTML = anomalies.map(a => {
        const severity = a.severity || 'warning';
        const color = severity === 'critical' ? 'var(--danger)' : 'var(--warning)';
        return `<div style="padding:6px 8px;margin-bottom:4px;background:${color}22;border-left:3px solid ${color};border-radius:4px;font-size:11px;">
            <strong>${formatName(a.type || 'Anomaly')}</strong>
            <div style="color:#666;">${a.description || ''}</div>
        </div>`;
    }).join('');
}

function showEvalDetails(idx) {
    selectedEvalId = idx;
    const container = document.getElementById('eval-details');
    const ev = evalDataCache[idx];
    
    document.querySelectorAll('.eval-item').forEach((el, i) => el.classList.toggle('selected', i === idx));
    
    if (!ev) { container.innerHTML = '<p class="placeholder">No data</p>'; return; }
    
    const score = ev.overall_score || ev.overall_apee_score || 0;
    const l1 = ev.l1_average || 0, l2 = ev.l2_average || 0, l3 = ev.l3_average || 0;
    const scoreColor = score >= 7 ? 'var(--success)' : score >= 5 ? 'var(--warning)' : 'var(--danger)';
    
    container.innerHTML = `
        <div style="margin-bottom:12px;">
            <h3 style="color:var(--primary);font-size:1rem;margin-bottom:6px;">${formatName(ev.scenario_id || 'Evaluation')}</h3>
            <div style="font-size:12px;color:#666;">
                <span style="margin-right:12px;"><b>Pattern:</b> ${formatName(ev.pattern || '-')}</span>
                <span style="margin-right:12px;"><b>Time:</b> ${(ev.duration || 0).toFixed(1)}s</span>
                <span><b>Score:</b> <span style="color:${scoreColor};font-weight:bold;font-size:14px;">${score.toFixed(2)}</span>/10</span>
            </div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:12px;">
            <div><div style="font-size:11px;font-weight:600;">L1 Individual</div>
                <div class="metric-bar"><div class="metric-fill l1" style="width:${l1*10}%"></div></div>
                <div style="font-size:12px;">${l1.toFixed(1)}</div></div>
            <div><div style="font-size:11px;font-weight:600;">L2 Collaborative</div>
                <div class="metric-bar"><div class="metric-fill l2" style="width:${l2*10}%"></div></div>
                <div style="font-size:12px;">${l2.toFixed(1)}</div></div>
            <div><div style="font-size:11px;font-weight:600;">L3 Ecosystem</div>
                <div class="metric-bar"><div class="metric-fill l3" style="width:${l3*10}%"></div></div>
                <div style="font-size:12px;">${l3.toFixed(1)}</div></div>
        </div>
        <details><summary style="cursor:pointer;padding:6px 10px;background:var(--light);border-radius:4px;font-size:11px;">📄 Raw JSON</summary>
            <pre style="background:#1e1e1e;color:#d4d4d4;padding:10px;border-radius:0 0 4px 4px;font-size:10px;max-height:250px;overflow:auto;">${JSON.stringify(ev, null, 2)}</pre>
        </details>`;
}

// Initial load
refreshData();

// Auto-refresh every 10 seconds
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
    parser.add_argument(
        "--data", "-d",
        type=str,
        default=None,
        help="Path to JSON file with evaluation results to load"
    )
    
    args = parser.parse_args()
    
    # Try to auto-detect data file if not specified
    data_file = args.data
    if not data_file:
        # Look for default data files
        default_paths = [
            Path.cwd() / "data" / "apee_evaluation_results.json",
            Path.cwd() / "apee_evaluation_results.json",
            Path(__file__).parent.parent.parent / "data" / "apee_evaluation_results.json",
        ]
        for path in default_paths:
            if path.exists():
                data_file = str(path)
                break
    
    server = DashboardServer(port=args.port, host=args.host)
    
    # Load data if available
    if data_file:
        state = get_dashboard_state()
        if state.load_from_json(data_file):
            print(f"📊 Loaded {len(state.evaluations)} evaluations from {data_file}")
        else:
            print(f"⚠️  Failed to load data from {data_file}")
    
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
