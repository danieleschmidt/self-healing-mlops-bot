#!/usr/bin/env python3
"""
Autonomous Monitoring Dashboard - Real-time system observability and alerting
Advanced monitoring with AI-powered anomaly detection and predictive insights
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import structlog
from collections import defaultdict, deque
import threading
import psutil
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

logger = structlog.get_logger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class MetricPoint:
    """Single metric data point"""
    timestamp: datetime
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Metric:
    """Metric definition and data"""
    name: str
    metric_type: MetricType
    description: str
    unit: str = ""
    data_points: deque = field(default_factory=lambda: deque(maxlen=1000))
    labels: Dict[str, str] = field(default_factory=dict)


@dataclass
class Alert:
    """Alert definition"""
    id: str
    name: str
    description: str
    severity: AlertSeverity
    condition: str
    threshold: float
    timestamp: datetime
    active: bool = True
    acknowledged: bool = False
    resolved: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Dashboard:
    """Dashboard configuration"""
    name: str
    title: str
    metrics: List[str]
    refresh_interval: int = 30  # seconds
    auto_refresh: bool = True
    layout: Dict[str, Any] = field(default_factory=dict)


class AutonomousMonitoringDashboard:
    """Advanced monitoring dashboard with AI-powered insights"""
    
    def __init__(self, project_root: str = "/root/repo", port: int = 8090):
        self.project_root = Path(project_root)
        self.port = port
        self.metrics: Dict[str, Metric] = {}
        self.alerts: Dict[str, Alert] = {}
        self.dashboards: Dict[str, Dashboard] = {}
        self.websocket_connections: List[WebSocket] = []
        self.data_collection_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.app = FastAPI(title="Autonomous Monitoring Dashboard")
        
        # Historical data storage
        self.historical_data = defaultdict(lambda: deque(maxlen=10000))
        self.anomaly_scores = defaultdict(lambda: deque(maxlen=1000))
        
        # AI-powered insights
        self.trend_analysis = {}
        self.prediction_models = {}
        self.anomaly_thresholds = {}
        
        self._initialize_monitoring()
        self._setup_routes()
    
    def _initialize_monitoring(self) -> None:
        """Initialize monitoring components"""
        logger.info("🖥️ Initializing autonomous monitoring dashboard")
        
        # Core system metrics
        self._register_metric("cpu_usage", MetricType.GAUGE, "CPU utilization percentage", "%")
        self._register_metric("memory_usage", MetricType.GAUGE, "Memory usage percentage", "%")
        self._register_metric("disk_usage", MetricType.GAUGE, "Disk usage percentage", "%")
        self._register_metric("network_io", MetricType.GAUGE, "Network I/O rate", "MB/s")
        
        # Application metrics
        self._register_metric("response_time", MetricType.HISTOGRAM, "Response time", "ms")
        self._register_metric("request_count", MetricType.COUNTER, "Request count", "requests")
        self._register_metric("error_rate", MetricType.GAUGE, "Error rate", "%")
        self._register_metric("active_connections", MetricType.GAUGE, "Active connections", "connections")
        
        # MLOps-specific metrics
        self._register_metric("pipeline_success_rate", MetricType.GAUGE, "Pipeline success rate", "%")
        self._register_metric("model_accuracy", MetricType.GAUGE, "Model accuracy", "score")
        self._register_metric("data_drift_score", MetricType.GAUGE, "Data drift detection score", "score")
        self._register_metric("webhook_processing_time", MetricType.HISTOGRAM, "Webhook processing time", "ms")
        
        # Create default dashboards
        self._create_default_dashboards()
        
        # Set up default alerts
        self._setup_default_alerts()
        
        logger.info("Monitoring dashboard initialized", 
                   metrics=len(self.metrics),
                   alerts=len(self.alerts),
                   dashboards=len(self.dashboards))
    
    def _register_metric(self, name: str, metric_type: MetricType, description: str, unit: str = "") -> None:
        """Register a new metric"""
        self.metrics[name] = Metric(
            name=name,
            metric_type=metric_type,
            description=description,
            unit=unit
        )
    
    def _create_default_dashboards(self) -> None:
        """Create default monitoring dashboards"""
        # System Overview Dashboard
        self.dashboards["system"] = Dashboard(
            name="system",
            title="System Overview",
            metrics=["cpu_usage", "memory_usage", "disk_usage", "network_io"],
            layout={
                "rows": 2,
                "cols": 2,
                "charts": [
                    {"metric": "cpu_usage", "type": "gauge", "position": [0, 0]},
                    {"metric": "memory_usage", "type": "gauge", "position": [0, 1]},
                    {"metric": "disk_usage", "type": "gauge", "position": [1, 0]},
                    {"metric": "network_io", "type": "line", "position": [1, 1]}
                ]
            }
        )
        
        # Application Performance Dashboard
        self.dashboards["performance"] = Dashboard(
            name="performance",
            title="Application Performance",
            metrics=["response_time", "request_count", "error_rate", "active_connections"],
            layout={
                "rows": 2,
                "cols": 2,
                "charts": [
                    {"metric": "response_time", "type": "histogram", "position": [0, 0]},
                    {"metric": "request_count", "type": "line", "position": [0, 1]},
                    {"metric": "error_rate", "type": "gauge", "position": [1, 0]},
                    {"metric": "active_connections", "type": "line", "position": [1, 1]}
                ]
            }
        )
        
        # MLOps Dashboard
        self.dashboards["mlops"] = Dashboard(
            name="mlops",
            title="MLOps Metrics",
            metrics=["pipeline_success_rate", "model_accuracy", "data_drift_score", "webhook_processing_time"],
            layout={
                "rows": 2,
                "cols": 2,
                "charts": [
                    {"metric": "pipeline_success_rate", "type": "gauge", "position": [0, 0]},
                    {"metric": "model_accuracy", "type": "line", "position": [0, 1]},
                    {"metric": "data_drift_score", "type": "line", "position": [1, 0]},
                    {"metric": "webhook_processing_time", "type": "histogram", "position": [1, 1]}
                ]
            }
        )
    
    def _setup_default_alerts(self) -> None:
        """Setup default alert rules"""
        alerts_config = [
            {
                "name": "high_cpu_usage",
                "description": "CPU usage is high",
                "severity": AlertSeverity.WARNING,
                "condition": "cpu_usage > 80",
                "threshold": 80.0
            },
            {
                "name": "critical_cpu_usage", 
                "description": "CPU usage is critically high",
                "severity": AlertSeverity.CRITICAL,
                "condition": "cpu_usage > 95",
                "threshold": 95.0
            },
            {
                "name": "high_memory_usage",
                "description": "Memory usage is high",
                "severity": AlertSeverity.WARNING,
                "condition": "memory_usage > 85",
                "threshold": 85.0
            },
            {
                "name": "high_error_rate",
                "description": "Error rate is high",
                "severity": AlertSeverity.ERROR,
                "condition": "error_rate > 5",
                "threshold": 5.0
            },
            {
                "name": "data_drift_detected",
                "description": "Data drift detected",
                "severity": AlertSeverity.WARNING,
                "condition": "data_drift_score > 0.3",
                "threshold": 0.3
            }
        ]
        
        for alert_config in alerts_config:
            alert_id = f"alert_{len(self.alerts)}"
            self.alerts[alert_id] = Alert(
                id=alert_id,
                name=alert_config["name"],
                description=alert_config["description"],
                severity=alert_config["severity"],
                condition=alert_config["condition"],
                threshold=alert_config["threshold"],
                timestamp=datetime.now(timezone.utc)
            )
    
    def _setup_routes(self) -> None:
        """Setup FastAPI routes"""
        @self.app.get("/")
        async def dashboard_home():
            return HTMLResponse(self._generate_dashboard_html())
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            return {
                name: {
                    "name": metric.name,
                    "type": metric.metric_type.value,
                    "description": metric.description,
                    "unit": metric.unit,
                    "current_value": list(metric.data_points)[-1].value if metric.data_points else 0,
                    "data_points": len(metric.data_points)
                }
                for name, metric in self.metrics.items()
            }
        
        @self.app.get("/api/metrics/{metric_name}/data")
        async def get_metric_data(metric_name: str, hours: int = 1):
            if metric_name not in self.metrics:
                return {"error": "Metric not found"}
            
            metric = self.metrics[metric_name]
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
            
            data = [
                {
                    "timestamp": point.timestamp.isoformat(),
                    "value": point.value,
                    "labels": point.labels
                }
                for point in metric.data_points
                if point.timestamp > cutoff_time
            ]
            
            return {
                "metric_name": metric_name,
                "data": data,
                "count": len(data)
            }
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            return {
                alert_id: {
                    "id": alert.id,
                    "name": alert.name,
                    "description": alert.description,
                    "severity": alert.severity.value,
                    "condition": alert.condition,
                    "threshold": alert.threshold,
                    "active": alert.active,
                    "acknowledged": alert.acknowledged,
                    "resolved": alert.resolved,
                    "timestamp": alert.timestamp.isoformat(),
                    "metadata": alert.metadata
                }
                for alert_id, alert in self.alerts.items()
            }
        
        @self.app.get("/api/dashboards")
        async def get_dashboards():
            return {
                name: {
                    "name": dashboard.name,
                    "title": dashboard.title,
                    "metrics": dashboard.metrics,
                    "refresh_interval": dashboard.refresh_interval,
                    "layout": dashboard.layout
                }
                for name, dashboard in self.dashboards.items()
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.websocket_connections.append(websocket)
            try:
                while True:
                    data = await websocket.receive_text()
                    # Handle websocket messages
                    await websocket.send_text(f"Echo: {data}")
            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
    
    def _generate_dashboard_html(self) -> str:
        """Generate dashboard HTML"""
        return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #1a1a1a;
            color: #ffffff;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
        }
        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        .metric-title {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #4CAF50;
        }
        .metric-value {
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .metric-unit {
            font-size: 14px;
            color: #888;
        }
        .status-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-healthy { background-color: #4CAF50; }
        .status-warning { background-color: #FF9800; }
        .status-critical { background-color: #F44336; }
        .alerts-section {
            background-color: #2d2d2d;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
        }
        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 4px solid;
        }
        .alert-info { border-left-color: #2196F3; background-color: #1e3a8a20; }
        .alert-warning { border-left-color: #FF9800; background-color: #f5941620; }
        .alert-error { border-left-color: #F44336; background-color: #dc262620; }
        .alert-critical { border-left-color: #9C27B0; background-color: #7c2d9a20; }
        .refresh-button {
            background-color: #4CAF50;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            margin: 10px;
        }
        .refresh-button:hover {
            background-color: #45a049;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🤖 Autonomous Monitoring Dashboard</h1>
        <p>Real-time system observability and AI-powered insights</p>
        <button class="refresh-button" onclick="refreshDashboard()">🔄 Refresh</button>
        <button class="refresh-button" onclick="toggleAutoRefresh()">⏱️ Auto Refresh</button>
    </div>
    
    <div id="dashboard-content">
        <div class="dashboard-grid" id="metrics-grid">
            <!-- Metrics will be loaded here -->
        </div>
        
        <div class="alerts-section">
            <h2>🚨 Active Alerts</h2>
            <div id="alerts-container">
                <!-- Alerts will be loaded here -->
            </div>
        </div>
    </div>

    <script>
        let autoRefresh = true;
        let refreshInterval;
        
        async function loadMetrics() {
            try {
                const response = await fetch('/api/metrics');
                const metrics = await response.json();
                
                const grid = document.getElementById('metrics-grid');
                grid.innerHTML = '';
                
                Object.values(metrics).forEach(metric => {
                    const card = document.createElement('div');
                    card.className = 'metric-card';
                    
                    const status = getMetricStatus(metric.name, metric.current_value);
                    
                    card.innerHTML = `
                        <div class="metric-title">
                            <span class="status-indicator status-${status}"></span>
                            ${metric.description}
                        </div>
                        <div class="metric-value">${metric.current_value.toFixed(2)}</div>
                        <div class="metric-unit">${metric.unit}</div>
                        <small>Type: ${metric.type} | Points: ${metric.data_points}</small>
                    `;
                    
                    grid.appendChild(card);
                });
            } catch (error) {
                console.error('Error loading metrics:', error);
            }
        }
        
        async function loadAlerts() {
            try {
                const response = await fetch('/api/alerts');
                const alerts = await response.json();
                
                const container = document.getElementById('alerts-container');
                container.innerHTML = '';
                
                const activeAlerts = Object.values(alerts).filter(alert => alert.active);
                
                if (activeAlerts.length === 0) {
                    container.innerHTML = '<p>✅ No active alerts</p>';
                } else {
                    activeAlerts.forEach(alert => {
                        const alertDiv = document.createElement('div');
                        alertDiv.className = `alert-item alert-${alert.severity}`;
                        alertDiv.innerHTML = `
                            <strong>${alert.name}</strong> - ${alert.description}
                            <br><small>Threshold: ${alert.threshold} | ${new Date(alert.timestamp).toLocaleString()}</small>
                        `;
                        container.appendChild(alertDiv);
                    });
                }
            } catch (error) {
                console.error('Error loading alerts:', error);
            }
        }
        
        function getMetricStatus(metricName, value) {
            const thresholds = {
                'cpu_usage': { warning: 80, critical: 95 },
                'memory_usage': { warning: 85, critical: 95 },
                'disk_usage': { warning: 80, critical: 90 },
                'error_rate': { warning: 2, critical: 5 }
            };
            
            const threshold = thresholds[metricName];
            if (!threshold) return 'healthy';
            
            if (value >= threshold.critical) return 'critical';
            if (value >= threshold.warning) return 'warning';
            return 'healthy';
        }
        
        function refreshDashboard() {
            loadMetrics();
            loadAlerts();
        }
        
        function toggleAutoRefresh() {
            autoRefresh = !autoRefresh;
            if (autoRefresh) {
                refreshInterval = setInterval(refreshDashboard, 30000);
                document.querySelector('button[onclick="toggleAutoRefresh()"]').textContent = '⏸️ Stop Auto';
            } else {
                clearInterval(refreshInterval);
                document.querySelector('button[onclick="toggleAutoRefresh()"]').textContent = '▶️ Start Auto';
            }
        }
        
        // Initialize dashboard
        refreshDashboard();
        if (autoRefresh) {
            refreshInterval = setInterval(refreshDashboard, 30000);
        }
        
        // WebSocket connection for real-time updates
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        ws.onmessage = function(event) {
            console.log('WebSocket message:', event.data);
            // Handle real-time updates
            refreshDashboard();
        };
    </script>
</body>
</html>
        """
    
    def record_metric(self, metric_name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a metric value"""
        if metric_name not in self.metrics:
            logger.warning(f"Unknown metric: {metric_name}")
            return
        
        metric = self.metrics[metric_name]
        point = MetricPoint(
            timestamp=datetime.now(timezone.utc),
            value=value,
            labels=labels or {}
        )
        
        metric.data_points.append(point)
        
        # Store in historical data for analysis
        self.historical_data[metric_name].append({
            'timestamp': point.timestamp,
            'value': value
        })
        
        # Check for alert conditions
        asyncio.create_task(self._check_alert_conditions(metric_name, value))
        
        # Perform anomaly detection
        self._detect_anomalies(metric_name, value)
        
        # Broadcast to WebSocket connections
        asyncio.create_task(self._broadcast_metric_update(metric_name, value))
    
    async def _check_alert_conditions(self, metric_name: str, value: float) -> None:
        """Check if metric value triggers any alerts"""
        for alert_id, alert in self.alerts.items():
            if not alert.active or alert.resolved:
                continue
            
            # Simple condition evaluation (in production, use more sophisticated parser)
            condition = alert.condition.replace(metric_name, str(value))
            try:
                if eval(condition):
                    if not alert.acknowledged:
                        logger.warning(f"Alert triggered: {alert.name} - {alert.description}")
                        alert.metadata['triggered_at'] = datetime.now(timezone.utc).isoformat()
                        alert.metadata['triggered_value'] = value
                        await self._send_alert_notification(alert)
            except Exception as e:
                logger.error(f"Error evaluating alert condition: {e}")
    
    def _detect_anomalies(self, metric_name: str, value: float) -> None:
        """Detect anomalies using statistical methods"""
        if metric_name not in self.historical_data:
            return
        
        data = [point['value'] for point in list(self.historical_data[metric_name])[-100:]]
        if len(data) < 10:
            return
        
        # Simple anomaly detection using z-score
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        if std_val > 0:
            z_score = abs((value - mean_val) / std_val)
            self.anomaly_scores[metric_name].append(z_score)
            
            # Threshold for anomaly (3 standard deviations)
            if z_score > 3.0:
                logger.warning(f"Anomaly detected in {metric_name}: value={value}, z-score={z_score:.2f}")
    
    async def _broadcast_metric_update(self, metric_name: str, value: float) -> None:
        """Broadcast metric update to WebSocket connections"""
        if not self.websocket_connections:
            return
        
        message = json.dumps({
            'type': 'metric_update',
            'metric': metric_name,
            'value': value,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        # Remove disconnected connections
        active_connections = []
        for ws in self.websocket_connections:
            try:
                await ws.send_text(message)
                active_connections.append(ws)
            except Exception:
                pass
        
        self.websocket_connections = active_connections
    
    async def _send_alert_notification(self, alert: Alert) -> None:
        """Send alert notification"""
        logger.info(f"Sending alert notification: {alert.name}")
        
        # In production, this would integrate with notification systems
        # (Slack, PagerDuty, email, etc.)
        
        # Broadcast to WebSocket connections
        message = json.dumps({
            'type': 'alert',
            'alert': {
                'id': alert.id,
                'name': alert.name,
                'description': alert.description,
                'severity': alert.severity.value,
                'timestamp': alert.timestamp.isoformat()
            }
        })
        
        for ws in self.websocket_connections:
            try:
                await ws.send_text(message)
            except Exception:
                pass
    
    def start_data_collection(self) -> None:
        """Start automatic data collection"""
        if self.data_collection_thread and self.data_collection_thread.is_alive():
            return
        
        logger.info("Starting data collection")
        self.shutdown_event.clear()
        self.data_collection_thread = threading.Thread(target=self._data_collection_loop, daemon=True)
        self.data_collection_thread.start()
    
    def _data_collection_loop(self) -> None:
        """Data collection loop"""
        while not self.shutdown_event.is_set():
            try:
                # Collect system metrics
                self.record_metric("cpu_usage", psutil.cpu_percent())
                self.record_metric("memory_usage", psutil.virtual_memory().percent)
                self.record_metric("disk_usage", psutil.disk_usage('/').percent)
                
                # Simulate application metrics
                import random
                self.record_metric("response_time", random.uniform(50, 200))
                self.record_metric("request_count", random.randint(10, 100))
                self.record_metric("error_rate", random.uniform(0, 10))
                self.record_metric("active_connections", random.randint(50, 300))
                
                # Simulate MLOps metrics
                self.record_metric("pipeline_success_rate", random.uniform(85, 100))
                self.record_metric("model_accuracy", random.uniform(0.85, 0.98))
                self.record_metric("data_drift_score", random.uniform(0, 0.5))
                self.record_metric("webhook_processing_time", random.uniform(100, 500))
                
                time.sleep(5)  # Collect every 5 seconds
                
            except Exception as e:
                logger.exception("Error in data collection loop", error=str(e))
                time.sleep(10)
    
    def stop_data_collection(self) -> None:
        """Stop data collection"""
        logger.info("Stopping data collection")
        self.shutdown_event.set()
        if self.data_collection_thread:
            self.data_collection_thread.join(timeout=10)
    
    async def start_dashboard(self) -> None:
        """Start the monitoring dashboard"""
        logger.info(f"🖥️ Starting monitoring dashboard on port {self.port}")
        
        # Start data collection
        self.start_data_collection()
        
        # Start web server
        config = uvicorn.Config(self.app, host="0.0.0.0", port=self.port, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get dashboard status"""
        return {
            'metrics_registered': len(self.metrics),
            'alerts_configured': len(self.alerts),
            'active_alerts': len([a for a in self.alerts.values() if a.active and not a.resolved]),
            'dashboards': len(self.dashboards),
            'websocket_connections': len(self.websocket_connections),
            'data_collection_active': self.data_collection_thread is not None and self.data_collection_thread.is_alive(),
            'historical_data_points': {name: len(data) for name, data in self.historical_data.items()}
        }


async def main():
    """Demo the autonomous monitoring dashboard"""
    dashboard = AutonomousMonitoringDashboard(port=8090)
    
    print("🖥️ AUTONOMOUS MONITORING DASHBOARD - DEMO")
    print("=" * 60)
    print("Dashboard will be available at: http://localhost:8090")
    print("Press Ctrl+C to stop")
    
    try:
        await dashboard.start_dashboard()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        dashboard.stop_data_collection()
        print("✅ Dashboard stopped")


if __name__ == "__main__":
    asyncio.run(main())