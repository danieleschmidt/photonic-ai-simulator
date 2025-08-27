"""
Real-Time Monitoring Dashboard for Photonic AI Systems

Provides web-based dashboard for real-time monitoring of photonic neural networks,
including performance metrics, system health, and operational insights.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
import logging
from collections import deque, defaultdict
import threading
import numpy as np

# Web framework imports
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    import uvicorn
    WEB_FRAMEWORK_AVAILABLE = True
except ImportError:
    WEB_FRAMEWORK_AVAILABLE = False

try:
    from .utils.monitoring import SystemMetrics, AlertLevel, MetricType
    from .robust_error_handling import RobustErrorHandler
except ImportError:
    from utils.monitoring import SystemMetrics, AlertLevel, MetricType
    from robust_error_handling import RobustErrorHandler

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the monitoring dashboard."""
    host: str = "0.0.0.0"
    port: int = 8080
    update_interval_seconds: float = 1.0
    max_data_points: int = 1000
    enable_alerts: bool = True
    enable_authentication: bool = False
    api_key: Optional[str] = None


@dataclass 
class DashboardMetrics:
    """Dashboard-specific metrics aggregation."""
    timestamp: float
    system_health_score: float
    active_alerts: int
    total_inference_count: int
    avg_latency_ms: float
    power_consumption_mw: float
    temperature_c: float
    accuracy_percent: float
    error_rate_percent: float
    throughput_samples_per_second: float


class RealtimeDataBuffer:
    """Thread-safe buffer for real-time data storage."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self._lock = threading.Lock()
        
    def add_data_point(self, data_point: DashboardMetrics):
        """Add new data point to buffer."""
        with self._lock:
            self.data.append(data_point)
    
    def get_recent_data(self, count: int = 100) -> List[DashboardMetrics]:
        """Get recent data points."""
        with self._lock:
            return list(self.data)[-count:] if self.data else []
    
    def get_all_data(self) -> List[DashboardMetrics]:
        """Get all stored data points."""
        with self._lock:
            return list(self.data)


class PhotonicDashboard:
    """Real-time monitoring dashboard for photonic AI systems."""
    
    def __init__(self, config: DashboardConfig = None):
        """Initialize the monitoring dashboard."""
        self.config = config or DashboardConfig()
        self.app = None
        self.data_buffer = RealtimeDataBuffer(self.config.max_data_points)
        self.connected_clients = set()
        self.metrics_collector = None
        self.is_running = False
        self.update_task = None
        
        # Initialize web framework if available
        if WEB_FRAMEWORK_AVAILABLE:
            self.app = self._create_fastapi_app()
        else:
            logger.warning("Web framework not available, dashboard will run in headless mode")
    
    def _create_fastapi_app(self) -> FastAPI:
        """Create FastAPI application."""
        app = FastAPI(
            title="Photonic AI Monitoring Dashboard",
            description="Real-time monitoring for photonic neural networks",
            version="1.0.0"
        )
        
        # Add middleware for authentication if enabled
        if self.config.enable_authentication:
            app.middleware("http")(self._auth_middleware)
        
        # Main dashboard route
        @app.get("/", response_class=HTMLResponse)
        async def dashboard():
            return self._get_dashboard_html()
        
        # API endpoints
        @app.get("/api/health")
        async def health_check():
            return {"status": "healthy", "timestamp": time.time()}
        
        @app.get("/api/metrics")
        async def get_metrics():
            recent_data = self.data_buffer.get_recent_data(100)
            return {"data": [asdict(point) for point in recent_data]}
        
        @app.get("/api/system-status")
        async def get_system_status():
            if not self.data_buffer.data:
                return {"status": "no_data"}
                
            latest = self.data_buffer.data[-1]
            return {
                "health_score": latest.system_health_score,
                "status": "healthy" if latest.system_health_score > 0.8 else "degraded",
                "active_alerts": latest.active_alerts,
                "uptime_hours": (time.time() - self.data_buffer.data[0].timestamp) / 3600
            }
        
        # WebSocket endpoint for real-time updates
        @app.websocket("/ws/metrics")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.connected_clients.add(websocket)
            
            try:
                while True:
                    # Keep connection alive
                    await websocket.receive_text()
            except WebSocketDisconnect:
                self.connected_clients.remove(websocket)
        
        return app
    
    async def _auth_middleware(self, request: Request, call_next):
        """Authentication middleware."""
        if self.config.api_key:
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                return JSONResponse(
                    status_code=401, 
                    content={"error": "Authentication required"}
                )
            
            token = auth_header.split(" ")[1]
            if token != self.config.api_key:
                return JSONResponse(
                    status_code=403,
                    content={"error": "Invalid API key"}
                )
        
        response = await call_next(request)
        return response
    
    def set_metrics_collector(self, collector: Any):
        """Set the metrics collector for data gathering."""
        self.metrics_collector = collector
    
    async def start_monitoring(self):
        """Start the monitoring system."""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info("Starting photonic AI monitoring dashboard")
        
        # Start metrics collection task
        self.update_task = asyncio.create_task(self._metrics_update_loop())
        
        if self.app:
            # Start web server
            config = uvicorn.Config(
                self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info"
            )
            server = uvicorn.Server(config)
            await server.serve()
    
    async def stop_monitoring(self):
        """Stop the monitoring system."""
        if not self.is_running:
            return
            
        self.is_running = False
        logger.info("Stopping photonic AI monitoring dashboard")
        
        if self.update_task:
            self.update_task.cancel()
            try:
                await self.update_task
            except asyncio.CancelledError:
                pass
    
    async def _metrics_update_loop(self):
        """Main metrics update loop."""
        while self.is_running:
            try:
                # Collect metrics if collector is available
                if self.metrics_collector:
                    system_metrics = self.metrics_collector.collect()
                    dashboard_metrics = self._process_system_metrics(system_metrics)
                    self.data_buffer.add_data_point(dashboard_metrics)
                    
                    # Broadcast to connected WebSocket clients
                    await self._broadcast_metrics(dashboard_metrics)
                
                # Wait for next update
                await asyncio.sleep(self.config.update_interval_seconds)
                
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(1.0)  # Brief pause before retry
    
    def _process_system_metrics(self, metrics: SystemMetrics) -> DashboardMetrics:
        """Process system metrics into dashboard format."""
        # Calculate system health score
        health_score = self._calculate_health_score(metrics)
        
        # Calculate derived metrics
        error_rate = 0.0
        if metrics.inference_count and metrics.inference_count > 0:
            error_rate = (metrics.error_count / metrics.inference_count) * 100
        
        throughput = 0.0
        if hasattr(metrics, 'throughput_samples_per_second'):
            throughput = metrics.throughput_samples_per_second
        
        return DashboardMetrics(
            timestamp=metrics.timestamp,
            system_health_score=health_score,
            active_alerts=0,  # Would be populated by alert system
            total_inference_count=metrics.inference_count or 0,
            avg_latency_ms=(metrics.latency_ns / 1_000_000) if metrics.latency_ns else 0.0,
            power_consumption_mw=metrics.power_mw or 0.0,
            temperature_c=(metrics.temperature_k - 273.15) if metrics.temperature_k else 25.0,
            accuracy_percent=(metrics.accuracy * 100) if metrics.accuracy else 0.0,
            error_rate_percent=error_rate,
            throughput_samples_per_second=throughput
        )
    
    def _calculate_health_score(self, metrics: SystemMetrics) -> float:
        """Calculate overall system health score (0-1)."""
        score_components = []
        
        # Accuracy component
        if metrics.accuracy is not None:
            score_components.append(min(metrics.accuracy / 0.95, 1.0))  # Target 95%
        
        # Latency component  
        if metrics.latency_ns is not None:
            target_latency_ns = 1_000_000  # 1ms target
            latency_score = max(0, 1.0 - (metrics.latency_ns / target_latency_ns - 1.0))
            score_components.append(max(0, latency_score))
        
        # Power component
        if metrics.power_mw is not None:
            target_power_mw = 500  # 500mW target
            power_score = max(0, 1.0 - max(0, metrics.power_mw / target_power_mw - 1.0))
            score_components.append(power_score)
        
        # Temperature component
        if metrics.temperature_k is not None:
            target_temp_k = 300  # 27°C target
            temp_diff = abs(metrics.temperature_k - target_temp_k)
            temp_score = max(0, 1.0 - temp_diff / 20.0)  # Allow 20K deviation
            score_components.append(temp_score)
        
        # Error rate component
        if metrics.inference_count and metrics.error_count is not None:
            error_rate = metrics.error_count / max(metrics.inference_count, 1)
            error_score = max(0, 1.0 - error_rate * 10)  # Penalize errors heavily
            score_components.append(error_score)
        
        # Average of all components, or 1.0 if no metrics available
        return np.mean(score_components) if score_components else 1.0
    
    async def _broadcast_metrics(self, metrics: DashboardMetrics):
        """Broadcast metrics to all connected WebSocket clients."""
        if not self.connected_clients:
            return
            
        message = json.dumps(asdict(metrics))
        disconnected_clients = []
        
        for client in self.connected_clients:
            try:
                await client.send_text(message)
            except Exception:
                disconnected_clients.append(client)
        
        # Remove disconnected clients
        for client in disconnected_clients:
            self.connected_clients.discard(client)
    
    def _get_dashboard_html(self) -> str:
        """Generate HTML for the monitoring dashboard."""
        return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Photonic AI Monitoring Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0; padding: 20px; background: #f5f5f5;
        }}
        .header {{ 
            background: #2c3e50; color: white; padding: 20px; margin-bottom: 20px;
            border-radius: 8px; text-align: center;
        }}
        .metrics-grid {{ 
            display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px; margin-bottom: 20px;
        }}
        .metric-card {{ 
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-value {{ 
            font-size: 2em; font-weight: bold; margin: 10px 0;
        }}
        .metric-label {{ 
            color: #666; font-size: 0.9em;
        }}
        .health-excellent {{ color: #27ae60; }}
        .health-good {{ color: #f39c12; }}
        .health-poor {{ color: #e74c3c; }}
        .chart-container {{ 
            background: white; padding: 20px; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); height: 400px;
        }}
        .status-indicator {{
            display: inline-block; width: 12px; height: 12px;
            border-radius: 50%; margin-right: 8px;
        }}
        .status-healthy {{ background: #27ae60; }}
        .status-warning {{ background: #f39c12; }}
        .status-error {{ background: #e74c3c; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔬 Photonic AI Monitoring Dashboard</h1>
        <div>
            <span class="status-indicator status-healthy" id="status-indicator"></span>
            <span id="system-status">System Healthy</span>
        </div>
    </div>

    <div class="metrics-grid">
        <div class="metric-card">
            <div class="metric-label">System Health Score</div>
            <div class="metric-value health-excellent" id="health-score">--</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Inference Latency</div>
            <div class="metric-value" id="latency">-- ms</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Power Consumption</div>
            <div class="metric-value" id="power">-- mW</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Temperature</div>
            <div class="metric-value" id="temperature">-- °C</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-value" id="accuracy">-- %</div>
        </div>
        
        <div class="metric-card">
            <div class="metric-label">Throughput</div>
            <div class="metric-value" id="throughput">-- samples/s</div>
        </div>
    </div>

    <div class="chart-container">
        <canvas id="metricsChart"></canvas>
    </div>

    <script>
        // Initialize WebSocket connection
        const ws = new WebSocket('ws://localhost:{self.config.port}/ws/metrics');
        
        // Initialize chart
        const ctx = document.getElementById('metricsChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: [],
                datasets: [
                    {{
                        label: 'Health Score',
                        data: [],
                        borderColor: '#27ae60',
                        backgroundColor: 'rgba(39, 174, 96, 0.1)',
                        tension: 0.4
                    }},
                    {{
                        label: 'Accuracy (%)',
                        data: [],
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        tension: 0.4
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    title: {{
                        display: true,
                        text: 'Real-time Performance Metrics'
                    }}
                }},
                scales: {{
                    x: {{ display: true }},
                    y: {{ display: true, min: 0, max: 100 }}
                }}
            }}
        }});

        // Update dashboard with new metrics
        function updateDashboard(metrics) {{
            // Update metric cards
            document.getElementById('health-score').textContent = 
                (metrics.system_health_score * 100).toFixed(1) + '%';
            document.getElementById('latency').textContent = 
                metrics.avg_latency_ms.toFixed(2) + ' ms';
            document.getElementById('power').textContent = 
                metrics.power_consumption_mw.toFixed(1) + ' mW';
            document.getElementById('temperature').textContent = 
                metrics.temperature_c.toFixed(1) + ' °C';
            document.getElementById('accuracy').textContent = 
                metrics.accuracy_percent.toFixed(1) + '%';
            document.getElementById('throughput').textContent = 
                Math.round(metrics.throughput_samples_per_second) + ' samples/s';
            
            // Update health score color
            const healthElement = document.getElementById('health-score');
            const score = metrics.system_health_score;
            healthElement.className = 'metric-value ' + 
                (score > 0.8 ? 'health-excellent' : score > 0.6 ? 'health-good' : 'health-poor');
            
            // Update system status
            const statusIndicator = document.getElementById('status-indicator');
            const statusText = document.getElementById('system-status');
            if (score > 0.8) {{
                statusIndicator.className = 'status-indicator status-healthy';
                statusText.textContent = 'System Healthy';
            }} else if (score > 0.6) {{
                statusIndicator.className = 'status-indicator status-warning';
                statusText.textContent = 'System Warning';
            }} else {{
                statusIndicator.className = 'status-indicator status-error';
                statusText.textContent = 'System Error';
            }}
            
            // Update chart
            const time = new Date(metrics.timestamp * 1000).toLocaleTimeString();
            chart.data.labels.push(time);
            chart.data.datasets[0].data.push(score * 100);
            chart.data.datasets[1].data.push(metrics.accuracy_percent);
            
            // Keep only last 50 data points
            if (chart.data.labels.length > 50) {{
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }}
            
            chart.update('none');
        }}

        // WebSocket event handlers
        ws.onmessage = function(event) {{
            const metrics = JSON.parse(event.data);
            updateDashboard(metrics);
        }};

        ws.onopen = function(event) {{
            console.log('Connected to monitoring dashboard');
        }};

        ws.onclose = function(event) {{
            console.log('Disconnected from monitoring dashboard');
            setTimeout(() => location.reload(), 5000); // Reconnect after 5s
        }};

        ws.onerror = function(error) {{
            console.error('WebSocket error:', error);
        }};
    </script>
</body>
</html>"""

    def get_system_summary(self) -> Dict[str, Any]:
        """Get comprehensive system summary for reporting."""
        if not self.data_buffer.data:
            return {"status": "no_data"}
        
        recent_data = self.data_buffer.get_recent_data(100)
        latest = recent_data[-1] if recent_data else None
        
        if not latest:
            return {"status": "no_data"}
        
        # Calculate statistics over recent period
        health_scores = [d.system_health_score for d in recent_data[-10:]]
        latencies = [d.avg_latency_ms for d in recent_data[-10:] if d.avg_latency_ms > 0]
        accuracies = [d.accuracy_percent for d in recent_data[-10:] if d.accuracy_percent > 0]
        
        return {
            "system_status": {
                "current_health_score": latest.system_health_score,
                "avg_health_score_10min": np.mean(health_scores) if health_scores else 0,
                "status": "healthy" if latest.system_health_score > 0.8 else 
                         "warning" if latest.system_health_score > 0.6 else "error"
            },
            "performance": {
                "current_latency_ms": latest.avg_latency_ms,
                "avg_latency_ms": np.mean(latencies) if latencies else 0,
                "current_accuracy_percent": latest.accuracy_percent,
                "avg_accuracy_percent": np.mean(accuracies) if accuracies else 0,
                "throughput_samples_per_second": latest.throughput_samples_per_second
            },
            "resources": {
                "power_consumption_mw": latest.power_consumption_mw,
                "temperature_c": latest.temperature_c,
                "total_inference_count": latest.total_inference_count,
                "error_rate_percent": latest.error_rate_percent
            },
            "alerts": {
                "active_alerts": latest.active_alerts
            },
            "timestamp": latest.timestamp
        }


# Factory function
def create_photonic_dashboard(host: str = "0.0.0.0", 
                            port: int = 8080,
                            update_interval: float = 1.0,
                            **kwargs) -> PhotonicDashboard:
    """
    Create a photonic AI monitoring dashboard.
    
    Args:
        host: Dashboard host address
        port: Dashboard port
        update_interval: Metrics update interval in seconds
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured PhotonicDashboard instance
    """
    config = DashboardConfig(
        host=host,
        port=port,
        update_interval_seconds=update_interval,
        **kwargs
    )
    
    return PhotonicDashboard(config)