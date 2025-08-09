"""Infrastructure health monitoring and detection."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import json
import socket
import subprocess
import requests
import psutil
from datetime import datetime, timedelta
from collections import defaultdict, deque
import concurrent.futures
import asyncio
import time

from .base import BaseDetector
from ..core.context import Context

logger = logging.getLogger(__name__)


class InfrastructureHealthDetector(BaseDetector):
    """Detect infrastructure health issues including network, database, and service availability."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Service monitoring configuration
        self.monitored_services = self.config.get("monitored_services", [])
        self.database_connections = self.config.get("database_connections", [])
        self.external_apis = self.config.get("external_apis", [])
        
        # Health check thresholds
        self.response_timeout = self.config.get("response_timeout_seconds", 30)
        self.max_response_time = self.config.get("max_response_time_ms", 5000)
        self.min_success_rate = self.config.get("min_success_rate", 0.95)
        
        # Network monitoring
        self.network_monitoring_enabled = self.config.get("network_monitoring_enabled", True)
        self.dns_servers = self.config.get("dns_servers", ["8.8.8.8", "1.1.1.1"])
        self.connectivity_hosts = self.config.get("connectivity_hosts", [
            "google.com", "github.com", "pypi.org"
        ])
        
        # System health monitoring
        self.system_monitoring_enabled = self.config.get("system_monitoring_enabled", True)
        self.disk_space_threshold = self.config.get("disk_space_threshold", 0.90)
        self.memory_threshold = self.config.get("memory_threshold", 0.85)
        self.cpu_threshold = self.config.get("cpu_threshold", 0.80)
        
        # Service discovery and container monitoring
        self.container_monitoring = self.config.get("container_monitoring_enabled", True)
        self.kubernetes_monitoring = self.config.get("kubernetes_monitoring_enabled", True)
        
        # Historical data and alerting
        self.health_history = defaultdict(lambda: deque(maxlen=100))
        self.alert_cooldown_minutes = self.config.get("alert_cooldown_minutes", 15)
        self.last_alerts = {}
        
        # Load balancer and CDN monitoring
        self.lb_monitoring_enabled = self.config.get("load_balancer_monitoring", True)
        self.cdn_monitoring_enabled = self.config.get("cdn_monitoring", True)
        
        # SSL/TLS certificate monitoring
        self.ssl_monitoring_enabled = self.config.get("ssl_monitoring", True)
        self.ssl_expiry_warning_days = self.config.get("ssl_expiry_warning_days", 30)
    
    def get_supported_events(self) -> List[str]:
        return ["schedule", "infrastructure_alert", "service_deployment", "health_check", "monitoring"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """Detect infrastructure health issues across multiple dimensions."""
        issues = []
        
        try:
            # Service health monitoring
            service_issues = await self._monitor_service_health(context)
            issues.extend(service_issues)
            
            # Database connectivity monitoring
            db_issues = await self._monitor_database_health(context)
            issues.extend(db_issues)
            
            # Network connectivity monitoring
            if self.network_monitoring_enabled:
                network_issues = await self._monitor_network_health(context)
                issues.extend(network_issues)
            
            # System resource monitoring
            if self.system_monitoring_enabled:
                system_issues = await self._monitor_system_health(context)
                issues.extend(system_issues)
            
            # Container and orchestration monitoring
            if self.container_monitoring:
                container_issues = await self._monitor_container_health(context)
                issues.extend(container_issues)
            
            if self.kubernetes_monitoring:
                k8s_issues = await self._monitor_kubernetes_health(context)
                issues.extend(k8s_issues)
            
            # Load balancer and CDN monitoring
            if self.lb_monitoring_enabled:
                lb_issues = await self._monitor_load_balancer_health(context)
                issues.extend(lb_issues)
            
            if self.cdn_monitoring_enabled:
                cdn_issues = await self._monitor_cdn_health(context)
                issues.extend(cdn_issues)
            
            # SSL/TLS certificate monitoring
            if self.ssl_monitoring_enabled:
                ssl_issues = await self._monitor_ssl_certificates(context)
                issues.extend(ssl_issues)
            
            # External dependency monitoring
            external_issues = await self._monitor_external_dependencies(context)
            issues.extend(external_issues)
            
        except Exception as e:
            logger.exception(f"Error in infrastructure health detection: {e}")
            issues.append(self.create_issue(
                issue_type="infrastructure_monitoring_error",
                severity="medium",
                message=f"Infrastructure health monitoring failed: {str(e)}",
                data={"error_details": str(e)}
            ))
        
        return issues
    
    async def _monitor_service_health(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor health of configured services."""
        issues = []
        
        if not self.monitored_services:
            # Auto-discover services if none configured
            self.monitored_services = await self._discover_services(context)
        
        for service_config in self.monitored_services:
            service_name = service_config["name"]
            health_endpoint = service_config.get("health_endpoint", f"http://localhost:{service_config.get('port', 8080)}/health")
            expected_status = service_config.get("expected_status", 200)
            
            try:
                health_result = await self._check_service_health(health_endpoint, expected_status)
                
                # Track health history
                self.health_history[service_name].append({
                    "timestamp": datetime.utcnow(),
                    "healthy": health_result["healthy"],
                    "response_time": health_result["response_time_ms"],
                    "status_code": health_result.get("status_code", 0)
                })
                
                if not health_result["healthy"]:
                    severity = self._calculate_service_health_severity(service_name, health_result)
                    
                    if self._should_alert(f"service_{service_name}"):
                        issues.append(self.create_issue(
                            issue_type="service_health_failure",
                            severity=severity,
                            message=f"Service '{service_name}' health check failed",
                            data={
                                "service_name": service_name,
                                "health_endpoint": health_endpoint,
                                "status_code": health_result.get("status_code", 0),
                                "response_time_ms": health_result["response_time_ms"],
                                "error_message": health_result.get("error", ""),
                                "success_rate_24h": self._calculate_success_rate(service_name),
                                "recommendation": self._get_service_health_recommendation(service_name, health_result)
                            }
                        ))
                        self.last_alerts[f"service_{service_name}"] = datetime.utcnow()
                
                # Check response time degradation
                elif health_result["response_time_ms"] > self.max_response_time:
                    if self._should_alert(f"service_{service_name}_latency"):
                        issues.append(self.create_issue(
                            issue_type="service_high_latency",
                            severity="medium",
                            message=f"Service '{service_name}' response time degraded: {health_result['response_time_ms']}ms",
                            data={
                                "service_name": service_name,
                                "response_time_ms": health_result["response_time_ms"],
                                "threshold_ms": self.max_response_time,
                                "recommendation": "Investigate performance bottlenecks and consider scaling"
                            }
                        ))
                        self.last_alerts[f"service_{service_name}_latency"] = datetime.utcnow()
            
            except Exception as e:
                logger.exception(f"Error checking health for service {service_name}: {e}")
                issues.append(self.create_issue(
                    issue_type="service_health_check_error",
                    severity="medium",
                    message=f"Failed to check health of service '{service_name}'",
                    data={
                        "service_name": service_name,
                        "error_details": str(e),
                        "recommendation": "Verify service configuration and network connectivity"
                    }
                ))
        
        return issues
    
    async def _monitor_database_health(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor database connectivity and performance."""
        issues = []
        
        for db_config in self.database_connections:
            db_name = db_config["name"]
            db_type = db_config.get("type", "postgresql")
            connection_string = db_config.get("connection_string", "")
            
            try:
                db_health = await self._check_database_health(db_config)
                
                if not db_health["connected"]:
                    if self._should_alert(f"database_{db_name}"):
                        issues.append(self.create_issue(
                            issue_type="database_connection_failure",
                            severity="critical",
                            message=f"Database '{db_name}' connection failed",
                            data={
                                "database_name": db_name,
                                "database_type": db_type,
                                "connection_time_ms": db_health.get("connection_time_ms", 0),
                                "error_message": db_health.get("error", ""),
                                "recommendation": "Check database availability and connection configuration"
                            }
                        ))
                        self.last_alerts[f"database_{db_name}"] = datetime.utcnow()
                
                # Check connection pool health
                elif db_health.get("pool_exhausted", False):
                    issues.append(self.create_issue(
                        issue_type="database_pool_exhausted",
                        severity="high",
                        message=f"Database '{db_name}' connection pool exhausted",
                        data={
                            "database_name": db_name,
                            "active_connections": db_health.get("active_connections", 0),
                            "max_connections": db_health.get("max_connections", 0),
                            "recommendation": "Increase connection pool size or investigate connection leaks"
                        }
                    ))
                
                # Check query performance
                elif db_health.get("slow_queries", 0) > 10:
                    issues.append(self.create_issue(
                        issue_type="database_slow_queries",
                        severity="medium",
                        message=f"Database '{db_name}' has {db_health['slow_queries']} slow queries",
                        data={
                            "database_name": db_name,
                            "slow_query_count": db_health["slow_queries"],
                            "avg_query_time_ms": db_health.get("avg_query_time_ms", 0),
                            "recommendation": "Optimize slow queries and consider database indexing"
                        }
                    ))
            
            except Exception as e:
                logger.exception(f"Error checking database health for {db_name}: {e}")
        
        return issues
    
    async def _monitor_network_health(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor network connectivity and DNS resolution."""
        issues = []
        
        # DNS resolution checks
        dns_issues = await self._check_dns_health()
        issues.extend(dns_issues)
        
        # Internet connectivity checks
        connectivity_issues = await self._check_internet_connectivity()
        issues.extend(connectivity_issues)
        
        # Network latency monitoring
        latency_issues = await self._monitor_network_latency()
        issues.extend(latency_issues)
        
        return issues
    
    async def _monitor_system_health(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor system-level health metrics."""
        issues = []
        
        try:
            # CPU usage monitoring
            cpu_usage = psutil.cpu_percent(interval=1)
            if cpu_usage > self.cpu_threshold * 100:
                if self._should_alert("system_cpu"):
                    issues.append(self.create_issue(
                        issue_type="high_cpu_usage",
                        severity="high" if cpu_usage > 90 else "medium",
                        message=f"High CPU usage: {cpu_usage:.1f}%",
                        data={
                            "cpu_usage": cpu_usage,
                            "threshold": self.cpu_threshold * 100,
                            "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [],
                            "recommendation": "Investigate high CPU usage and consider scaling"
                        }
                    ))
                    self.last_alerts["system_cpu"] = datetime.utcnow()
            
            # Memory usage monitoring
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_threshold * 100:
                if self._should_alert("system_memory"):
                    issues.append(self.create_issue(
                        issue_type="high_memory_usage",
                        severity="high" if memory.percent > 95 else "medium",
                        message=f"High memory usage: {memory.percent:.1f}%",
                        data={
                            "memory_usage": memory.percent,
                            "memory_available_gb": memory.available / (1024**3),
                            "memory_total_gb": memory.total / (1024**3),
                            "threshold": self.memory_threshold * 100,
                            "recommendation": "Check for memory leaks and consider adding memory"
                        }
                    ))
                    self.last_alerts["system_memory"] = datetime.utcnow()
            
            # Disk space monitoring
            disk_issues = await self._check_disk_space()
            issues.extend(disk_issues)
            
            # System load monitoring
            if hasattr(psutil, 'getloadavg'):
                load_avg = psutil.getloadavg()
                cpu_count = psutil.cpu_count()
                
                if load_avg[0] > cpu_count * 2:  # Load average > 2x CPU count
                    issues.append(self.create_issue(
                        issue_type="high_system_load",
                        severity="medium",
                        message=f"High system load: {load_avg[0]:.2f} (CPU count: {cpu_count})",
                        data={
                            "load_1min": load_avg[0],
                            "load_5min": load_avg[1],
                            "load_15min": load_avg[2],
                            "cpu_count": cpu_count,
                            "recommendation": "Investigate system bottlenecks and consider scaling"
                        }
                    ))
        
        except Exception as e:
            logger.exception(f"Error monitoring system health: {e}")
        
        return issues
    
    async def _monitor_container_health(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor container health and status."""
        issues = []
        
        try:
            # Get container information (mock implementation)
            containers = await self._get_container_info()
            
            for container in containers:
                container_id = container["id"]
                container_name = container["name"]
                status = container["status"]
                
                # Check container status
                if status not in ["running", "healthy"]:
                    severity = "critical" if status in ["exited", "failed"] else "high"
                    
                    issues.append(self.create_issue(
                        issue_type="container_unhealthy",
                        severity=severity,
                        message=f"Container '{container_name}' is {status}",
                        data={
                            "container_id": container_id,
                            "container_name": container_name,
                            "status": status,
                            "restart_count": container.get("restart_count", 0),
                            "last_restart": container.get("last_restart", ""),
                            "recommendation": f"Investigate container '{container_name}' failure and restart if necessary"
                        }
                    ))
                
                # Check restart frequency
                restart_count = container.get("restart_count", 0)
                if restart_count > 5:  # More than 5 restarts
                    issues.append(self.create_issue(
                        issue_type="container_frequent_restarts",
                        severity="medium",
                        message=f"Container '{container_name}' has restarted {restart_count} times",
                        data={
                            "container_id": container_id,
                            "container_name": container_name,
                            "restart_count": restart_count,
                            "recommendation": "Investigate container stability issues"
                        }
                    ))
        
        except Exception as e:
            logger.exception(f"Error monitoring container health: {e}")
        
        return issues
    
    async def _monitor_kubernetes_health(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor Kubernetes cluster health."""
        issues = []
        
        try:
            # Mock Kubernetes monitoring - in production would use kubectl or K8s API
            k8s_status = await self._get_kubernetes_status()
            
            # Check node health
            unhealthy_nodes = [node for node in k8s_status.get("nodes", []) 
                             if node["status"] != "Ready"]
            
            if unhealthy_nodes:
                issues.append(self.create_issue(
                    issue_type="kubernetes_unhealthy_nodes",
                    severity="high",
                    message=f"{len(unhealthy_nodes)} Kubernetes nodes are unhealthy",
                    data={
                        "unhealthy_nodes": unhealthy_nodes,
                        "total_nodes": len(k8s_status.get("nodes", [])),
                        "recommendation": "Investigate node issues and consider node replacement"
                    }
                ))
            
            # Check pod health
            failed_pods = [pod for pod in k8s_status.get("pods", []) 
                         if pod["status"] in ["Failed", "CrashLoopBackOff", "ImagePullBackOff"]]
            
            if failed_pods:
                issues.append(self.create_issue(
                    issue_type="kubernetes_failed_pods",
                    severity="medium",
                    message=f"{len(failed_pods)} Kubernetes pods are failing",
                    data={
                        "failed_pods": failed_pods,
                        "total_pods": len(k8s_status.get("pods", [])),
                        "recommendation": "Investigate pod failures and restart if necessary"
                    }
                ))
            
            # Check cluster resource utilization
            resource_usage = k8s_status.get("resource_usage", {})
            if resource_usage.get("cpu_usage", 0) > 80:
                issues.append(self.create_issue(
                    issue_type="kubernetes_high_resource_usage",
                    severity="medium",
                    message=f"Kubernetes cluster CPU usage: {resource_usage['cpu_usage']:.1f}%",
                    data={
                        "cpu_usage": resource_usage["cpu_usage"],
                        "memory_usage": resource_usage.get("memory_usage", 0),
                        "recommendation": "Consider scaling cluster or optimizing resource requests"
                    }
                ))
        
        except Exception as e:
            logger.exception(f"Error monitoring Kubernetes health: {e}")
        
        return issues
    
    async def _monitor_load_balancer_health(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor load balancer health and distribution."""
        issues = []
        
        # Mock load balancer monitoring
        lb_health = await self._check_load_balancer_health()
        
        for lb_name, lb_data in lb_health.items():
            # Check backend health
            unhealthy_backends = [backend for backend in lb_data.get("backends", []) 
                                if not backend["healthy"]]
            
            if unhealthy_backends:
                issues.append(self.create_issue(
                    issue_type="load_balancer_unhealthy_backends",
                    severity="high",
                    message=f"Load balancer '{lb_name}' has {len(unhealthy_backends)} unhealthy backends",
                    data={
                        "load_balancer": lb_name,
                        "unhealthy_backends": unhealthy_backends,
                        "total_backends": len(lb_data.get("backends", [])),
                        "recommendation": "Investigate backend health and remove unhealthy instances"
                    }
                ))
            
            # Check traffic distribution
            if lb_data.get("traffic_imbalance", False):
                issues.append(self.create_issue(
                    issue_type="load_balancer_traffic_imbalance",
                    severity="medium",
                    message=f"Load balancer '{lb_name}' has uneven traffic distribution",
                    data={
                        "load_balancer": lb_name,
                        "traffic_distribution": lb_data.get("traffic_distribution", {}),
                        "recommendation": "Review load balancing algorithm and backend capacity"
                    }
                ))
        
        return issues
    
    async def _monitor_cdn_health(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor CDN performance and availability."""
        issues = []
        
        # Mock CDN monitoring
        cdn_health = await self._check_cdn_health()
        
        for cdn_name, cdn_data in cdn_health.items():
            # Check cache hit ratio
            hit_ratio = cdn_data.get("cache_hit_ratio", 0)
            if hit_ratio < 0.7:  # Less than 70% hit ratio
                issues.append(self.create_issue(
                    issue_type="cdn_low_cache_hit_ratio",
                    severity="medium",
                    message=f"CDN '{cdn_name}' has low cache hit ratio: {hit_ratio:.1%}",
                    data={
                        "cdn_name": cdn_name,
                        "cache_hit_ratio": hit_ratio,
                        "recommendation": "Review caching policies and content optimization"
                    }
                ))
            
            # Check edge server availability
            unavailable_edges = cdn_data.get("unavailable_edges", [])
            if unavailable_edges:
                issues.append(self.create_issue(
                    issue_type="cdn_edge_servers_unavailable",
                    severity="medium",
                    message=f"CDN '{cdn_name}' has {len(unavailable_edges)} unavailable edge servers",
                    data={
                        "cdn_name": cdn_name,
                        "unavailable_edges": unavailable_edges,
                        "total_edges": cdn_data.get("total_edges", 0),
                        "recommendation": "Monitor edge server health and failover mechanisms"
                    }
                ))
        
        return issues
    
    async def _monitor_ssl_certificates(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor SSL certificate expiration and health."""
        issues = []
        
        # Get SSL certificates to monitor
        ssl_endpoints = self.config.get("ssl_endpoints", [])
        
        for endpoint in ssl_endpoints:
            try:
                cert_info = await self._check_ssl_certificate(endpoint)
                
                days_until_expiry = cert_info.get("days_until_expiry", 0)
                
                if days_until_expiry <= 0:
                    issues.append(self.create_issue(
                        issue_type="ssl_certificate_expired",
                        severity="critical",
                        message=f"SSL certificate expired for {endpoint}",
                        data={
                            "endpoint": endpoint,
                            "expiry_date": cert_info.get("expiry_date", ""),
                            "days_expired": abs(days_until_expiry),
                            "recommendation": "Renew SSL certificate immediately"
                        }
                    ))
                
                elif days_until_expiry <= self.ssl_expiry_warning_days:
                    severity = "high" if days_until_expiry <= 7 else "medium"
                    
                    issues.append(self.create_issue(
                        issue_type="ssl_certificate_expiring",
                        severity=severity,
                        message=f"SSL certificate expiring in {days_until_expiry} days for {endpoint}",
                        data={
                            "endpoint": endpoint,
                            "expiry_date": cert_info.get("expiry_date", ""),
                            "days_until_expiry": days_until_expiry,
                            "issuer": cert_info.get("issuer", ""),
                            "recommendation": f"Renew SSL certificate for {endpoint}"
                        }
                    ))
            
            except Exception as e:
                logger.exception(f"Error checking SSL certificate for {endpoint}: {e}")
        
        return issues
    
    async def _monitor_external_dependencies(self, context: Context) -> List[Dict[str, Any]]:
        """Monitor external API and service dependencies."""
        issues = []
        
        for api_config in self.external_apis:
            api_name = api_config["name"]
            api_url = api_config["url"]
            expected_status = api_config.get("expected_status", 200)
            timeout = api_config.get("timeout", self.response_timeout)
            
            try:
                api_health = await self._check_external_api_health(api_url, expected_status, timeout)
                
                if not api_health["available"]:
                    if self._should_alert(f"external_api_{api_name}"):
                        issues.append(self.create_issue(
                            issue_type="external_api_unavailable",
                            severity="high",
                            message=f"External API '{api_name}' is unavailable",
                            data={
                                "api_name": api_name,
                                "api_url": api_url,
                                "status_code": api_health.get("status_code", 0),
                                "response_time_ms": api_health.get("response_time_ms", 0),
                                "error_message": api_health.get("error", ""),
                                "recommendation": f"Check external API status and implement fallback for {api_name}"
                            }
                        ))
                        self.last_alerts[f"external_api_{api_name}"] = datetime.utcnow()
                
                elif api_health.get("response_time_ms", 0) > 10000:  # 10 second threshold
                    issues.append(self.create_issue(
                        issue_type="external_api_slow_response",
                        severity="medium",
                        message=f"External API '{api_name}' responding slowly: {api_health['response_time_ms']}ms",
                        data={
                            "api_name": api_name,
                            "response_time_ms": api_health["response_time_ms"],
                            "recommendation": f"Monitor {api_name} performance and consider caching"
                        }
                    ))
            
            except Exception as e:
                logger.exception(f"Error checking external API {api_name}: {e}")
        
        return issues
    
    def _should_alert(self, alert_key: str) -> bool:
        """Check if enough time has passed since last alert."""
        last_alert = self.last_alerts.get(alert_key)
        if not last_alert:
            return True
        
        time_since_alert = datetime.utcnow() - last_alert
        return time_since_alert.total_seconds() > (self.alert_cooldown_minutes * 60)
    
    async def _discover_services(self, context: Context) -> List[Dict[str, Any]]:
        """Auto-discover services to monitor."""
        # Mock service discovery
        return [
            {"name": "web-service", "port": 8080, "health_endpoint": "http://localhost:8080/health"},
            {"name": "api-service", "port": 8081, "health_endpoint": "http://localhost:8081/health"},
            {"name": "worker-service", "port": 8082, "health_endpoint": "http://localhost:8082/health"}
        ]
    
    async def _check_service_health(self, health_endpoint: str, expected_status: int) -> Dict[str, Any]:
        """Check health of a service endpoint."""
        start_time = time.time()
        
        try:
            response = requests.get(health_endpoint, timeout=self.response_timeout)
            response_time_ms = (time.time() - start_time) * 1000
            
            return {
                "healthy": response.status_code == expected_status,
                "status_code": response.status_code,
                "response_time_ms": response_time_ms,
                "response_body": response.text[:200]  # Truncate for logging
            }
        
        except requests.exceptions.RequestException as e:
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "healthy": False,
                "status_code": 0,
                "response_time_ms": response_time_ms,
                "error": str(e)
            }
    
    async def _check_database_health(self, db_config: Dict[str, Any]) -> Dict[str, Any]:
        """Check database connectivity and health."""
        # Mock database health check - in production would use actual DB connections
        db_type = db_config.get("type", "postgresql")
        
        # Simulate database health check results
        return {
            "connected": True,  # Mock successful connection
            "connection_time_ms": 50,
            "active_connections": 15,
            "max_connections": 100,
            "pool_exhausted": False,
            "slow_queries": 2,
            "avg_query_time_ms": 120
        }
    
    async def _check_dns_health(self) -> List[Dict[str, Any]]:
        """Check DNS resolution health."""
        issues = []
        
        for dns_server in self.dns_servers:
            try:
                # Test DNS resolution (simplified)
                start_time = time.time()
                socket.gethostbyname_ex("google.com")
                resolution_time_ms = (time.time() - start_time) * 1000
                
                if resolution_time_ms > 5000:  # 5 second threshold
                    issues.append(self.create_issue(
                        issue_type="dns_slow_resolution",
                        severity="medium",
                        message=f"Slow DNS resolution via {dns_server}: {resolution_time_ms:.0f}ms",
                        data={
                            "dns_server": dns_server,
                            "resolution_time_ms": resolution_time_ms,
                            "recommendation": "Check DNS server performance or switch to alternative"
                        }
                    ))
            
            except Exception as e:
                issues.append(self.create_issue(
                    issue_type="dns_resolution_failure",
                    severity="high",
                    message=f"DNS resolution failed via {dns_server}",
                    data={
                        "dns_server": dns_server,
                        "error": str(e),
                        "recommendation": "Check DNS server availability and configuration"
                    }
                ))
        
        return issues
    
    async def _check_internet_connectivity(self) -> List[Dict[str, Any]]:
        """Check internet connectivity to key hosts."""
        issues = []
        
        failed_hosts = []
        
        for host in self.connectivity_hosts:
            try:
                # Simple connectivity test
                response = requests.get(f"https://{host}", timeout=10, stream=True)
                if response.status_code != 200:
                    failed_hosts.append(host)
            
            except Exception:
                failed_hosts.append(host)
        
        if failed_hosts:
            if len(failed_hosts) == len(self.connectivity_hosts):
                severity = "critical"
                message = "Complete internet connectivity failure"
            else:
                severity = "medium"
                message = f"Partial internet connectivity failure: {len(failed_hosts)} hosts unreachable"
            
            issues.append(self.create_issue(
                issue_type="internet_connectivity_failure",
                severity=severity,
                message=message,
                data={
                    "failed_hosts": failed_hosts,
                    "total_hosts": len(self.connectivity_hosts),
                    "recommendation": "Check internet connection and routing"
                }
            ))
        
        return issues
    
    async def _monitor_network_latency(self) -> List[Dict[str, Any]]:
        """Monitor network latency to key endpoints."""
        issues = []
        
        # Mock network latency monitoring
        high_latency_endpoints = [
            {"endpoint": "api.example.com", "latency_ms": 2500},
            {"endpoint": "cdn.example.com", "latency_ms": 1800}
        ]
        
        for endpoint_info in high_latency_endpoints:
            if endpoint_info["latency_ms"] > 2000:  # 2 second threshold
                issues.append(self.create_issue(
                    issue_type="high_network_latency",
                    severity="medium",
                    message=f"High network latency to {endpoint_info['endpoint']}: {endpoint_info['latency_ms']}ms",
                    data={
                        "endpoint": endpoint_info["endpoint"],
                        "latency_ms": endpoint_info["latency_ms"],
                        "recommendation": "Investigate network routing and consider CDN optimization"
                    }
                ))
        
        return issues
    
    async def _check_disk_space(self) -> List[Dict[str, Any]]:
        """Check disk space across mounted filesystems."""
        issues = []
        
        try:
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    usage_percent = (usage.used / usage.total) * 100
                    
                    if usage_percent > self.disk_space_threshold * 100:
                        severity = "critical" if usage_percent > 95 else "high"
                        
                        issues.append(self.create_issue(
                            issue_type="disk_space_low",
                            severity=severity,
                            message=f"Low disk space on {partition.mountpoint}: {usage_percent:.1f}% used",
                            data={
                                "mountpoint": partition.mountpoint,
                                "device": partition.device,
                                "usage_percent": usage_percent,
                                "free_gb": usage.free / (1024**3),
                                "total_gb": usage.total / (1024**3),
                                "recommendation": f"Free up space on {partition.mountpoint} or expand storage"
                            }
                        ))
                
                except (PermissionError, OSError):
                    continue
        
        except Exception as e:
            logger.exception(f"Error checking disk space: {e}")
        
        return issues
    
    async def _get_container_info(self) -> List[Dict[str, Any]]:
        """Get container information (mock implementation)."""
        # Mock container data - in production would use Docker API
        return [
            {
                "id": "container_1",
                "name": "web-app",
                "status": "running",
                "restart_count": 2,
                "last_restart": "2023-01-15T10:30:00Z"
            },
            {
                "id": "container_2", 
                "name": "database",
                "status": "exited",
                "restart_count": 8,
                "last_restart": "2023-01-15T11:45:00Z"
            }
        ]
    
    async def _get_kubernetes_status(self) -> Dict[str, Any]:
        """Get Kubernetes cluster status (mock implementation)."""
        # Mock Kubernetes data
        return {
            "nodes": [
                {"name": "node-1", "status": "Ready"},
                {"name": "node-2", "status": "NotReady"},
                {"name": "node-3", "status": "Ready"}
            ],
            "pods": [
                {"name": "app-pod-1", "status": "Running"},
                {"name": "app-pod-2", "status": "CrashLoopBackOff"},
                {"name": "db-pod-1", "status": "Running"}
            ],
            "resource_usage": {
                "cpu_usage": 75.5,
                "memory_usage": 68.2
            }
        }
    
    async def _check_load_balancer_health(self) -> Dict[str, Dict[str, Any]]:
        """Check load balancer health (mock implementation)."""
        return {
            "main-lb": {
                "backends": [
                    {"name": "backend-1", "healthy": True, "response_time_ms": 120},
                    {"name": "backend-2", "healthy": False, "response_time_ms": 0},
                    {"name": "backend-3", "healthy": True, "response_time_ms": 150}
                ],
                "traffic_imbalance": True,
                "traffic_distribution": {
                    "backend-1": 60,
                    "backend-2": 0,
                    "backend-3": 40
                }
            }
        }
    
    async def _check_cdn_health(self) -> Dict[str, Dict[str, Any]]:
        """Check CDN health (mock implementation)."""
        return {
            "main-cdn": {
                "cache_hit_ratio": 0.85,
                "total_edges": 50,
                "unavailable_edges": ["edge-us-west-2", "edge-eu-central-1"],
                "avg_response_time_ms": 45
            }
        }
    
    async def _check_ssl_certificate(self, endpoint: str) -> Dict[str, Any]:
        """Check SSL certificate information."""
        # Mock SSL certificate check
        import random
        
        days_until_expiry = random.randint(-5, 90)  # Random expiry for testing
        
        return {
            "days_until_expiry": days_until_expiry,
            "expiry_date": "2024-03-15",
            "issuer": "Let's Encrypt Authority X3",
            "subject": endpoint
        }
    
    async def _check_external_api_health(self, api_url: str, expected_status: int, timeout: int) -> Dict[str, Any]:
        """Check external API health."""
        start_time = time.time()
        
        try:
            response = requests.get(api_url, timeout=timeout)
            response_time_ms = (time.time() - start_time) * 1000
            
            return {
                "available": response.status_code == expected_status,
                "status_code": response.status_code,
                "response_time_ms": response_time_ms
            }
        
        except requests.exceptions.RequestException as e:
            response_time_ms = (time.time() - start_time) * 1000
            return {
                "available": False,
                "status_code": 0,
                "response_time_ms": response_time_ms,
                "error": str(e)
            }
    
    def _calculate_service_health_severity(self, service_name: str, health_result: Dict[str, Any]) -> str:
        """Calculate severity of service health failure."""
        status_code = health_result.get("status_code", 0)
        
        if status_code == 0 or status_code >= 500:
            return "critical"
        elif status_code >= 400:
            return "high"
        else:
            return "medium"
    
    def _calculate_success_rate(self, service_name: str) -> float:
        """Calculate 24-hour success rate for a service."""
        history = list(self.health_history[service_name])
        
        if not history:
            return 1.0
        
        # Filter last 24 hours
        cutoff_time = datetime.utcnow() - timedelta(hours=24)
        recent_checks = [h for h in history if h["timestamp"] > cutoff_time]
        
        if not recent_checks:
            return 1.0
        
        successful_checks = sum(1 for check in recent_checks if check["healthy"])
        return successful_checks / len(recent_checks)
    
    def _get_service_health_recommendation(self, service_name: str, health_result: Dict[str, Any]) -> str:
        """Get recommendation for service health issues."""
        status_code = health_result.get("status_code", 0)
        
        if status_code == 0:
            return f"Service {service_name} is unreachable - check if service is running and network connectivity"
        elif status_code >= 500:
            return f"Service {service_name} has internal errors - check application logs and restart if necessary"
        elif status_code >= 400:
            return f"Service {service_name} has client errors - review request format and authentication"
        else:
            return f"Investigate {service_name} health check response and service status"