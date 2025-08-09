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
        
        # Web API metrics
        self.webhook_requests_total = Counter(
            'bot_webhook_requests_total',
            'Total webhook requests received',
            ['event_type', 'status'],
            registry=self.registry
        )
        
        self.webhook_events_processed = Counter(
            'bot_webhook_events_processed_total',
            'Total webhook events processed',
            ['event_type', 'status'],
            registry=self.registry
        )
        
        self.http_requests_total = Counter(
            'bot_http_requests_total',
            'Total HTTP requests',
            ['method', 'path', 'status_code'],
            registry=self.registry
        )
        
        self.http_errors_total = Counter(
            'bot_http_errors_total',
            'Total HTTP errors',
            ['method', 'status_code'],
            registry=self.registry
        )
        
        self.request_duration = Histogram(
            'bot_http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'path'],
            registry=self.registry
        )
        
        self.rate_limit_hits = Counter(
            'bot_rate_limit_hits_total',
            'Rate limit violations',
            ['client_type', 'endpoint'],
            registry=self.registry
        )
        
        self.security_violations = Counter(
            'bot_security_violations_total',
            'Security violations detected',
            ['violation_type', 'severity'],
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
    
    def record_webhook_request(self, event_type: str, status: str):
        """Record webhook request."""
        self.webhook_requests_total.labels(
            event_type=event_type,
            status=status
        ).inc()
    
    def record_webhook_event_processed(self, event_type: str, status: str):
        """Record webhook event processing."""
        self.webhook_events_processed.labels(
            event_type=event_type,
            status=status
        ).inc()
    
    def record_http_request(self, method: str, path: str, status_code: int, duration: float):
        """Record HTTP request."""
        self.http_requests_total.labels(
            method=method,
            path=path,
            status_code=status_code
        ).inc()
        
        self.request_duration.labels(
            method=method,
            path=path
        ).observe(duration)
    
    def record_http_error(self, method: str, status_code: int):
        """Record HTTP error."""
        self.http_errors_total.labels(
            method=method,
            status_code=status_code
        ).inc()
    
    def record_rate_limit_hit(self, client_type: str, endpoint: str):
        """Record rate limit violation."""
        self.rate_limit_hits.labels(
            client_type=client_type,
            endpoint=endpoint
        ).inc()
    
    def record_security_violation(self, violation_type: str, severity: str):
        """Record security violation."""
        self.security_violations.labels(
            violation_type=violation_type,
            severity=severity
        ).inc()
    
    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(self.registry)


# Global metrics instance
prometheus_metrics = BotMetrics()

# Backward compatibility alias
metrics = prometheus_metrics


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