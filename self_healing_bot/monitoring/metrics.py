"""Prometheus metrics collection."""

import time
from typing import Dict, Any, Optional
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import logging

logger = logging.getLogger(__name__)


class BotMetrics:
    """Prometheus metrics for the self-healing bot."""
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or CollectorRegistry()
        self._initialize_metrics()
    
    def _initialize_metrics(self):
        """Initialize Prometheus metrics."""
        # Event processing metrics
        self.events_processed_total = Counter(
            'bot_events_processed_total',
            'Total number of events processed',
            ['event_type', 'repo', 'status'],
            registry=self.registry
        )
        
        self.event_processing_duration = Histogram(
            'bot_event_processing_duration_seconds',
            'Time spent processing events',
            ['event_type', 'repo'],
            registry=self.registry
        )
        
        # Issue detection metrics
        self.issues_detected_total = Counter(
            'bot_issues_detected_total',
            'Total number of issues detected',
            ['issue_type', 'severity', 'detector'],
            registry=self.registry
        )
        
        # Repair metrics
        self.repairs_attempted_total = Counter(
            'bot_repairs_attempted_total',
            'Total number of repair attempts',
            ['playbook', 'repo', 'status'],
            registry=self.registry
        )
        
        self.repair_duration = Histogram(
            'bot_repair_duration_seconds',
            'Time spent on repairs',
            ['playbook', 'repo'],
            registry=self.registry
        )
        
        # System health metrics
        self.active_executions = Gauge(
            'bot_active_executions',
            'Number of currently active executions',
            registry=self.registry
        )
        
        self.github_api_requests_total = Counter(
            'bot_github_api_requests_total',
            'Total GitHub API requests',
            ['endpoint', 'status_code'],
            registry=self.registry
        )
        
        self.github_api_rate_limit = Gauge(
            'bot_github_api_rate_limit_remaining',
            'GitHub API rate limit remaining',
            registry=self.registry
        )
        
        # Error metrics
        self.errors_total = Counter(
            'bot_errors_total',
            'Total number of errors',
            ['error_type', 'component'],
            registry=self.registry
        )
    
    def record_event_processed(self, event_type: str, repo: str, status: str, duration: float):
        """Record event processing metrics."""
        self.events_processed_total.labels(
            event_type=event_type,
            repo=repo,
            status=status
        ).inc()
        
        self.event_processing_duration.labels(
            event_type=event_type,
            repo=repo
        ).observe(duration)
    
    def record_issue_detected(self, issue_type: str, severity: str, detector: str):
        """Record issue detection metrics."""
        self.issues_detected_total.labels(
            issue_type=issue_type,
            severity=severity,
            detector=detector
        ).inc()
    
    def record_repair_attempted(self, playbook: str, repo: str, status: str, duration: float):
        """Record repair attempt metrics."""
        self.repairs_attempted_total.labels(
            playbook=playbook,
            repo=repo,
            status=status
        ).inc()
        
        self.repair_duration.labels(
            playbook=playbook,
            repo=repo
        ).observe(duration)
    
    def set_active_executions(self, count: int):
        """Set active executions count."""
        self.active_executions.set(count)
    
    def record_github_api_request(self, endpoint: str, status_code: int):
        """Record GitHub API request."""
        self.github_api_requests_total.labels(
            endpoint=endpoint,
            status_code=status_code
        ).inc()
    
    def set_github_rate_limit(self, remaining: int):
        """Set GitHub API rate limit remaining."""
        self.github_api_rate_limit.set(remaining)
    
    def record_error(self, error_type: str, component: str):
        """Record error occurrence."""
        self.errors_total.labels(
            error_type=error_type,
            component=component
        ).inc()
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)


# Global metrics instance
metrics = BotMetrics()


class MetricsMiddleware:
    """Middleware for automatic metrics collection."""
    
    def __init__(self, metrics_instance: BotMetrics):
        self.metrics = metrics_instance
    
    async def __call__(self, request, call_next):
        """Process request and collect metrics."""
        start_time = time.time()
        
        try:
            response = await call_next(request)
            duration = time.time() - start_time
            
            # Record successful request
            self.metrics.record_github_api_request(
                endpoint=str(request.url.path),
                status_code=response.status_code
            )
            
            return response
            
        except Exception as e:
            duration = time.time() - start_time
            
            # Record error
            self.metrics.record_error(
                error_type=type(e).__name__,
                component="web_api"
            )
            
            raise