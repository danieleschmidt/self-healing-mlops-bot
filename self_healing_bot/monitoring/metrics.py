"""Metrics collection and monitoring for the self-healing bot."""

import time
from typing import Dict, Any, List
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
import threading
from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class MetricValue:
    """Individual metric value with timestamp."""
    value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Advanced metrics collection system with Prometheus integration."""
    
    def __init__(self):
        self.registry = CollectorRegistry()
        self._lock = threading.Lock()
        
        # Prometheus metrics
        self.counters: Dict[str, Counter] = {}
        self.histograms: Dict[str, Histogram] = {}
        self.gauges: Dict[str, Gauge] = {}
        
        # In-memory storage for custom metrics
        self.custom_metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
        # System metrics
        self._initialize_system_metrics()
    
    def _initialize_system_metrics(self):
        """Initialize system-level metrics."""
        # Event processing metrics
        self.counters["events_processed_total"] = Counter(
            "events_processed_total",
            "Total number of events processed",
            ["event_type"],
            registry=self.registry
        )
        
        self.counters["events_processed_successfully_total"] = Counter(
            "events_processed_successfully_total",
            "Total number of events processed successfully",
            registry=self.registry
        )
        
        self.counters["events_failed_total"] = Counter(
            "events_failed_total",
            "Total number of events that failed processing",
            registry=self.registry
        )
        
        self.counters["events_timeout_total"] = Counter(
            "events_timeout_total",
            "Total number of events that timed out",
            registry=self.registry
        )
        
        self.counters["events_no_issues_total"] = Counter(
            "events_no_issues_total",
            "Total number of events with no issues detected",
            registry=self.registry
        )
        
        # Processing time metrics
        self.histograms["event_processing_duration_seconds"] = Histogram(
            "event_processing_duration_seconds",
            "Time spent processing events",
            buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0],
            registry=self.registry
        )
        
        # System health metrics
        self.gauges["active_executions"] = Gauge(
            "active_executions",
            "Number of active executions",
            registry=self.registry
        )
        
        self.gauges["health_score"] = Gauge(
            "health_score",
            "Overall system health score (0-100)",
            registry=self.registry
        )
    
    def increment_counter(self, name: str, labels: Dict[str, str] = None) -> None:
        """Increment a counter metric."""
        labels = labels or {}
        
        with self._lock:
            if name in self.counters:
                counter = self.counters[name]
                if counter._labelnames:
                    counter.labels(**labels).inc()
                else:
                    counter.inc()
            else:
                logger.warning("Counter not found", metric_name=name)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Record a value in a histogram metric."""
        labels = labels or {}
        
        with self._lock:
            if name in self.histograms:
                histogram = self.histograms[name]
                if histogram._labelnames:
                    histogram.labels(**labels).observe(value)
                else:
                    histogram.observe(value)
            else:
                logger.warning("Histogram not found", metric_name=name)
    
    def set_gauge(self, name: str, value: float, labels: Dict[str, str] = None) -> None:
        """Set a gauge metric value."""
        labels = labels or {}
        
        with self._lock:
            if name in self.gauges:
                gauge = self.gauges[name]
                if gauge._labelnames:
                    gauge.labels(**labels).set(value)
                else:
                    gauge.set(value)
            else:
                logger.warning("Gauge not found", metric_name=name)
    
    def get_prometheus_metrics(self) -> str:
        """Get Prometheus-formatted metrics."""
        return generate_latest(self.registry).decode('utf-8')
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get a summary of all metrics."""
        summary = {
            "counters": {},
            "histograms": {},
            "gauges": {}
        }
        
        with self._lock:
            # Get counter values
            for name, counter in self.counters.items():
                try:
                    # Get the total value across all label combinations
                    summary["counters"][name] = counter._value.sum()
                except:
                    summary["counters"][name] = 0
            
            # Get gauge values
            for name, gauge in self.gauges.items():
                try:
                    summary["gauges"][name] = gauge._value.sum()
                except:
                    summary["gauges"][name] = 0
        
        return summary


# Global metrics collector instance
metrics_collector = MetricsCollector()