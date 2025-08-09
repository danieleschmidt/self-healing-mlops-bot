"""Resource constraint detection for MLOps systems."""

from typing import List, Dict, Any, Optional, Tuple
import logging
import re
import psutil
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json

from .base import BaseDetector
from ..core.context import Context

logger = logging.getLogger(__name__)


class ResourceConstraintDetector(BaseDetector):
    """Detect resource constraints including GPU OOM, memory issues, and CPU bottlenecks."""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        
        # Resource thresholds
        self.memory_threshold = self.config.get("memory_threshold", 0.85)  # 85%
        self.cpu_threshold = self.config.get("cpu_threshold", 0.80)  # 80%
        self.disk_threshold = self.config.get("disk_threshold", 0.90)  # 90%
        self.gpu_memory_threshold = self.config.get("gpu_memory_threshold", 0.90)  # 90%
        
        # Alert thresholds
        self.critical_memory_threshold = self.config.get("critical_memory_threshold", 0.95)
        self.critical_cpu_threshold = self.config.get("critical_cpu_threshold", 0.95)
        self.critical_disk_threshold = self.config.get("critical_disk_threshold", 0.95)
        
        # OOM detection patterns
        self.oom_patterns = self.config.get("oom_patterns", [
            r"CUDA out of memory",
            r"OutOfMemoryError",
            r"RuntimeError.*out of memory",
            r"GPU memory.*insufficient",
            r"OOMKilled",
            r"Killed.*memory",
            r"MemoryError",
            r"Cannot allocate memory"
        ])
        
        # Resource monitoring
        self.resource_history = defaultdict(lambda: deque(maxlen=100))
        self.alert_cooldown = self.config.get("alert_cooldown_minutes", 15)
        self.last_alerts = {}
        
        # Process monitoring
        self.monitor_processes = self.config.get("monitor_processes", True)
        self.process_memory_threshold = self.config.get("process_memory_threshold", 0.50)  # 50% of total memory
        self.process_cpu_threshold = self.config.get("process_cpu_threshold", 0.80)  # 80% CPU
        
        # Container resource monitoring
        self.container_monitoring_enabled = self.config.get("container_monitoring_enabled", True)
        self.container_memory_threshold = self.config.get("container_memory_threshold", 0.85)
        self.container_cpu_threshold = self.config.get("container_cpu_threshold", 0.80)
    
    def get_supported_events(self) -> List[str]:
        return ["schedule", "workflow_run", "push", "resource_alert", "container_event"]
    
    async def detect(self, context: Context) -> List[Dict[str, Any]]:
        """Detect resource constraints across system components."""
        issues = []
        
        try:
            # System resource monitoring
            system_issues = await self._detect_system_resource_issues(context)
            issues.extend(system_issues)
            
            # GPU resource monitoring
            gpu_issues = await self._detect_gpu_issues(context)
            issues.extend(gpu_issues)
            
            # OOM detection from logs
            oom_issues = await self._detect_oom_from_logs(context)
            issues.extend(oom_issues)
            
            # Process resource monitoring
            if self.monitor_processes:
                process_issues = await self._detect_process_resource_issues(context)
                issues.extend(process_issues)
            
            # Container resource monitoring
            if self.container_monitoring_enabled:
                container_issues = await self._detect_container_resource_issues(context)
                issues.extend(container_issues)
            
            # Resource trend analysis
            trend_issues = await self._analyze_resource_trends(context)
            issues.extend(trend_issues)
            
            # Resource capacity planning
            capacity_issues = await self._detect_capacity_planning_issues(context)
            issues.extend(capacity_issues)
            
        except Exception as e:
            logger.exception(f"Error in resource constraint detection: {e}")
            issues.append(self.create_issue(
                issue_type="resource_detection_error",
                severity="medium",
                message=f"Resource constraint detection failed: {str(e)}",
                data={"error_details": str(e)}
            ))
        
        return issues
    
    async def _detect_system_resource_issues(self, context: Context) -> List[Dict[str, Any]]:
        """Detect system-wide resource constraint issues."""
        issues = []
        
        # Get current system metrics
        system_metrics = await self._get_system_metrics()
        
        # Track metrics in history
        timestamp = datetime.utcnow()
        for metric_name, value in system_metrics.items():
            if isinstance(value, (int, float)):
                self.resource_history[metric_name].append({
                    "value": value,
                    "timestamp": timestamp
                })
        
        # Memory usage detection
        memory_usage = system_metrics.get("memory_percent", 0)
        if memory_usage > self.critical_memory_threshold:
            if self._should_alert("system_memory"):
                issues.append(self.create_issue(
                    issue_type="critical_memory_usage",
                    severity="critical",
                    message=f"Critical memory usage: {memory_usage:.1f}%",
                    data={
                        "memory_percent": memory_usage,
                        "memory_available_gb": system_metrics.get("memory_available_gb", 0),
                        "memory_total_gb": system_metrics.get("memory_total_gb", 0),
                        "threshold": self.critical_memory_threshold,
                        "recommendation": "Immediate action required - scale memory or reduce workload"
                    }
                ))
                self.last_alerts["system_memory"] = timestamp
        
        elif memory_usage > self.memory_threshold:
            if self._should_alert("system_memory"):
                issues.append(self.create_issue(
                    issue_type="high_memory_usage",
                    severity="high",
                    message=f"High memory usage: {memory_usage:.1f}%",
                    data={
                        "memory_percent": memory_usage,
                        "memory_available_gb": system_metrics.get("memory_available_gb", 0),
                        "threshold": self.memory_threshold,
                        "recommendation": "Monitor closely and consider scaling memory resources"
                    }
                ))
                self.last_alerts["system_memory"] = timestamp
        
        # CPU usage detection
        cpu_usage = system_metrics.get("cpu_percent", 0)
        if cpu_usage > self.critical_cpu_threshold:
            if self._should_alert("system_cpu"):
                issues.append(self.create_issue(
                    issue_type="critical_cpu_usage",
                    severity="critical",
                    message=f"Critical CPU usage: {cpu_usage:.1f}%",
                    data={
                        "cpu_percent": cpu_usage,
                        "cpu_count": system_metrics.get("cpu_count", 0),
                        "load_average": system_metrics.get("load_average", []),
                        "threshold": self.critical_cpu_threshold,
                        "recommendation": "Immediate scaling required - add CPU resources or reduce workload"
                    }
                ))
                self.last_alerts["system_cpu"] = timestamp
        
        elif cpu_usage > self.cpu_threshold:
            if self._should_alert("system_cpu"):
                issues.append(self.create_issue(
                    issue_type="high_cpu_usage",
                    severity="high",
                    message=f"High CPU usage: {cpu_usage:.1f}%",
                    data={
                        "cpu_percent": cpu_usage,
                        "threshold": self.cpu_threshold,
                        "recommendation": "Consider scaling CPU resources"
                    }
                ))
                self.last_alerts["system_cpu"] = timestamp
        
        # Disk usage detection
        for disk_info in system_metrics.get("disk_usage", []):
            disk_usage = disk_info["percent"]
            mount_point = disk_info["mountpoint"]
            
            if disk_usage > self.critical_disk_threshold:
                if self._should_alert(f"disk_{mount_point}"):
                    issues.append(self.create_issue(
                        issue_type="critical_disk_usage",
                        severity="critical",
                        message=f"Critical disk usage on {mount_point}: {disk_usage:.1f}%",
                        data={
                            "mountpoint": mount_point,
                            "disk_percent": disk_usage,
                            "free_gb": disk_info["free_gb"],
                            "total_gb": disk_info["total_gb"],
                            "threshold": self.critical_disk_threshold,
                            "recommendation": "Immediate cleanup required or expand storage"
                        }
                    ))
                    self.last_alerts[f"disk_{mount_point}"] = timestamp
            
            elif disk_usage > self.disk_threshold:
                if self._should_alert(f"disk_{mount_point}"):
                    issues.append(self.create_issue(
                        issue_type="high_disk_usage",
                        severity="high",
                        message=f"High disk usage on {mount_point}: {disk_usage:.1f}%",
                        data={
                            "mountpoint": mount_point,
                            "disk_percent": disk_usage,
                            "free_gb": disk_info["free_gb"],
                            "threshold": self.disk_threshold,
                            "recommendation": "Plan for storage expansion or cleanup"
                        }
                    ))
                    self.last_alerts[f"disk_{mount_point}"] = timestamp
        
        return issues
    
    async def _detect_gpu_issues(self, context: Context) -> List[Dict[str, Any]]:
        """Detect GPU-related resource constraints."""
        issues = []
        
        gpu_metrics = await self._get_gpu_metrics()
        
        if not gpu_metrics:
            return issues
        
        for gpu_id, metrics in gpu_metrics.items():
            # GPU memory usage
            gpu_memory_percent = metrics.get("memory_percent", 0)
            if gpu_memory_percent > self.gpu_memory_threshold:
                severity = "critical" if gpu_memory_percent > 95 else "high"
                
                if self._should_alert(f"gpu_{gpu_id}_memory"):
                    issues.append(self.create_issue(
                        issue_type="gpu_memory_constraint",
                        severity=severity,
                        message=f"GPU {gpu_id} memory usage: {gpu_memory_percent:.1f}%",
                        data={
                            "gpu_id": gpu_id,
                            "memory_percent": gpu_memory_percent,
                            "memory_used_mb": metrics.get("memory_used_mb", 0),
                            "memory_total_mb": metrics.get("memory_total_mb", 0),
                            "gpu_utilization": metrics.get("utilization_percent", 0),
                            "temperature": metrics.get("temperature", 0),
                            "threshold": self.gpu_memory_threshold,
                            "recommendation": self._get_gpu_memory_recommendation(gpu_memory_percent)
                        }
                    ))
                    self.last_alerts[f"gpu_{gpu_id}_memory"] = datetime.utcnow()
            
            # GPU temperature monitoring
            temperature = metrics.get("temperature", 0)
            if temperature > 85:  # Critical temperature
                if self._should_alert(f"gpu_{gpu_id}_temp"):
                    issues.append(self.create_issue(
                        issue_type="gpu_overheating",
                        severity="critical" if temperature > 90 else "high",
                        message=f"GPU {gpu_id} overheating: {temperature}°C",
                        data={
                            "gpu_id": gpu_id,
                            "temperature": temperature,
                            "recommendation": "Check cooling system and reduce workload"
                        }
                    ))
                    self.last_alerts[f"gpu_{gpu_id}_temp"] = datetime.utcnow()
            
            # GPU utilization issues
            utilization = metrics.get("utilization_percent", 0)
            if utilization > 98:  # Near 100% utilization
                if self._should_alert(f"gpu_{gpu_id}_util"):
                    issues.append(self.create_issue(
                        issue_type="gpu_utilization_maxed",
                        severity="medium",
                        message=f"GPU {gpu_id} at maximum utilization: {utilization:.1f}%",
                        data={
                            "gpu_id": gpu_id,
                            "utilization_percent": utilization,
                            "recommendation": "Consider load balancing or adding GPU capacity"
                        }
                    ))
                    self.last_alerts[f"gpu_{gpu_id}_util"] = datetime.utcnow()
        
        return issues
    
    async def _detect_oom_from_logs(self, context: Context) -> List[Dict[str, Any]]:
        """Detect Out of Memory errors from application logs."""
        issues = []
        
        # Mock log analysis - in production, this would analyze actual logs
        log_entries = await self._get_recent_logs(context)
        
        oom_detections = []
        for log_entry in log_entries:
            log_content = log_entry.get("message", "")
            timestamp = log_entry.get("timestamp", datetime.utcnow())
            
            for pattern in self.oom_patterns:
                if re.search(pattern, log_content, re.IGNORECASE):
                    oom_detections.append({
                        "pattern": pattern,
                        "log_message": log_content[:200],  # Truncate for display
                        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else timestamp,
                        "log_source": log_entry.get("source", "unknown"),
                        "process_id": log_entry.get("pid"),
                        "container_id": log_entry.get("container_id")
                    })
                    break
        
        if oom_detections:
            # Group by pattern for better analysis
            pattern_counts = defaultdict(int)
            for detection in oom_detections:
                pattern_counts[detection["pattern"]] += 1
            
            most_common_pattern = max(pattern_counts.items(), key=lambda x: x[1])
            
            issues.append(self.create_issue(
                issue_type="out_of_memory_detected",
                severity="high",
                message=f"Out of Memory errors detected: {len(oom_detections)} occurrences",
                data={
                    "total_detections": len(oom_detections),
                    "most_common_pattern": most_common_pattern[0],
                    "pattern_frequency": most_common_pattern[1],
                    "recent_detections": oom_detections[-5:],  # Last 5 detections
                    "recommendation": self._get_oom_recommendation(oom_detections)
                }
            ))
        
        return issues
    
    async def _detect_process_resource_issues(self, context: Context) -> List[Dict[str, Any]]:
        """Detect resource issues at the process level."""
        issues = []
        
        process_metrics = await self._get_process_metrics()
        
        for process_info in process_metrics:
            pid = process_info["pid"]
            name = process_info["name"]
            memory_percent = process_info["memory_percent"]
            cpu_percent = process_info["cpu_percent"]
            
            # High memory usage by single process
            if memory_percent > self.process_memory_threshold:
                severity = "critical" if memory_percent > 0.70 else "high"
                
                if self._should_alert(f"process_{pid}_memory"):
                    issues.append(self.create_issue(
                        issue_type="process_high_memory",
                        severity=severity,
                        message=f"Process '{name}' (PID: {pid}) using {memory_percent:.1f}% of system memory",
                        data={
                            "pid": pid,
                            "process_name": name,
                            "memory_percent": memory_percent,
                            "memory_mb": process_info.get("memory_mb", 0),
                            "threshold": self.process_memory_threshold,
                            "recommendation": f"Investigate memory usage in process '{name}'"
                        }
                    ))
                    self.last_alerts[f"process_{pid}_memory"] = datetime.utcnow()
            
            # High CPU usage by single process
            if cpu_percent > self.process_cpu_threshold:
                if self._should_alert(f"process_{pid}_cpu"):
                    issues.append(self.create_issue(
                        issue_type="process_high_cpu",
                        severity="medium",
                        message=f"Process '{name}' (PID: {pid}) using {cpu_percent:.1f}% CPU",
                        data={
                            "pid": pid,
                            "process_name": name,
                            "cpu_percent": cpu_percent,
                            "threshold": self.process_cpu_threshold,
                            "recommendation": f"Monitor CPU usage pattern for process '{name}'"
                        }
                    ))
                    self.last_alerts[f"process_{pid}_cpu"] = datetime.utcnow()
        
        return issues
    
    async def _detect_container_resource_issues(self, context: Context) -> List[Dict[str, Any]]:
        """Detect resource constraints in containerized environments."""
        issues = []
        
        container_metrics = await self._get_container_metrics()
        
        for container_info in container_metrics:
            container_id = container_info["id"]
            container_name = container_info["name"]
            
            # Container memory usage
            memory_usage = container_info.get("memory_usage_percent", 0)
            memory_limit = container_info.get("memory_limit_mb", 0)
            
            if memory_usage > self.container_memory_threshold:
                severity = "critical" if memory_usage > 0.95 else "high"
                
                if self._should_alert(f"container_{container_id}_memory"):
                    issues.append(self.create_issue(
                        issue_type="container_memory_constraint",
                        severity=severity,
                        message=f"Container '{container_name}' memory usage: {memory_usage:.1f}%",
                        data={
                            "container_id": container_id,
                            "container_name": container_name,
                            "memory_usage_percent": memory_usage,
                            "memory_limit_mb": memory_limit,
                            "memory_used_mb": container_info.get("memory_used_mb", 0),
                            "threshold": self.container_memory_threshold,
                            "recommendation": self._get_container_memory_recommendation(memory_usage, memory_limit)
                        }
                    ))
                    self.last_alerts[f"container_{container_id}_memory"] = datetime.utcnow()
            
            # Container CPU usage
            cpu_usage = container_info.get("cpu_usage_percent", 0)
            if cpu_usage > self.container_cpu_threshold:
                if self._should_alert(f"container_{container_id}_cpu"):
                    issues.append(self.create_issue(
                        issue_type="container_cpu_constraint",
                        severity="medium",
                        message=f"Container '{container_name}' CPU usage: {cpu_usage:.1f}%",
                        data={
                            "container_id": container_id,
                            "container_name": container_name,
                            "cpu_usage_percent": cpu_usage,
                            "cpu_limit": container_info.get("cpu_limit", 0),
                            "threshold": self.container_cpu_threshold,
                            "recommendation": f"Consider increasing CPU limit for container '{container_name}'"
                        }
                    ))
                    self.last_alerts[f"container_{container_id}_cpu"] = datetime.utcnow()
            
            # Container restart detection
            restart_count = container_info.get("restart_count", 0)
            if restart_count > 5:  # Multiple restarts indicate issues
                if self._should_alert(f"container_{container_id}_restarts"):
                    issues.append(self.create_issue(
                        issue_type="container_frequent_restarts",
                        severity="high",
                        message=f"Container '{container_name}' has restarted {restart_count} times",
                        data={
                            "container_id": container_id,
                            "container_name": container_name,
                            "restart_count": restart_count,
                            "recommendation": "Investigate container stability and resource allocation"
                        }
                    ))
                    self.last_alerts[f"container_{container_id}_restarts"] = datetime.utcnow()
        
        return issues
    
    async def _analyze_resource_trends(self, context: Context) -> List[Dict[str, Any]]:
        """Analyze resource usage trends for predictive alerts."""
        issues = []
        
        # Analyze trends for key resources
        key_metrics = ["memory_percent", "cpu_percent", "disk_usage_percent"]
        
        for metric_name in key_metrics:
            history = list(self.resource_history[metric_name])
            if len(history) < 20:  # Need sufficient history
                continue
            
            values = [entry["value"] for entry in history[-20:]]
            trend_info = ResourceTrendAnalyzer.analyze_trend(values)
            
            if trend_info["trend"] == "increasing" and trend_info["confidence"] > 0.7:
                # Predict when threshold will be crossed
                current_value = values[-1] if values else 0
                threshold = self._get_threshold_for_metric(metric_name)
                
                if trend_info["slope"] > 0:
                    time_to_threshold = (threshold - current_value) / trend_info["slope"]
                    
                    if 0 < time_to_threshold <= 60:  # Within 60 data points
                        issues.append(self.create_issue(
                            issue_type="resource_trend_warning",
                            severity="medium",
                            message=f"Increasing trend in {metric_name} - threshold may be exceeded soon",
                            data={
                                "metric_name": metric_name,
                                "current_value": current_value,
                                "threshold": threshold,
                                "trend_slope": trend_info["slope"],
                                "confidence": trend_info["confidence"],
                                "estimated_time_to_threshold": time_to_threshold,
                                "recommendation": f"Proactively scale {metric_name.replace('_percent', '')} resources"
                            }
                        ))
        
        return issues
    
    async def _detect_capacity_planning_issues(self, context: Context) -> List[Dict[str, Any]]:
        """Detect issues requiring capacity planning."""
        issues = []
        
        # Analyze resource utilization patterns
        resource_analysis = await self._analyze_resource_capacity(context)
        
        for resource_type, analysis in resource_analysis.items():
            if analysis["utilization_trend"] == "high_sustained":
                issues.append(self.create_issue(
                    issue_type="capacity_planning_needed",
                    severity="medium",
                    message=f"Sustained high {resource_type} utilization detected",
                    data={
                        "resource_type": resource_type,
                        "average_utilization": analysis["average_utilization"],
                        "peak_utilization": analysis["peak_utilization"],
                        "utilization_trend": analysis["utilization_trend"],
                        "recommendation": f"Plan for {resource_type} capacity expansion"
                    }
                ))
        
        return issues
    
    def _should_alert(self, resource_key: str) -> bool:
        """Check if enough time has passed since last alert."""
        last_alert = self.last_alerts.get(resource_key)
        if not last_alert:
            return True
        
        time_since_alert = datetime.utcnow() - last_alert
        return time_since_alert.total_seconds() > (self.alert_cooldown * 60)
    
    def _get_threshold_for_metric(self, metric_name: str) -> float:
        """Get threshold value for specific metric."""
        thresholds = {
            "memory_percent": self.memory_threshold * 100,
            "cpu_percent": self.cpu_threshold * 100,
            "disk_usage_percent": self.disk_threshold * 100
        }
        return thresholds.get(metric_name, 80.0)
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics."""
        try:
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # Disk metrics
            disk_usage = []
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage.append({
                        "mountpoint": partition.mountpoint,
                        "device": partition.device,
                        "fstype": partition.fstype,
                        "total_gb": usage.total / (1024**3),
                        "used_gb": usage.used / (1024**3),
                        "free_gb": usage.free / (1024**3),
                        "percent": (usage.used / usage.total) * 100
                    })
                except (PermissionError, OSError):
                    continue
            
            return {
                "memory_percent": memory.percent,
                "memory_total_gb": memory.total / (1024**3),
                "memory_available_gb": memory.available / (1024**3),
                "memory_used_gb": memory.used / (1024**3),
                "cpu_percent": cpu_percent,
                "cpu_count": psutil.cpu_count(),
                "load_average": list(load_avg),
                "disk_usage": disk_usage
            }
        except Exception as e:
            logger.exception(f"Error getting system metrics: {e}")
            return {}
    
    async def _get_gpu_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get GPU metrics (mock implementation)."""
        # Mock GPU metrics - in production, would use nvidia-ml-py or similar
        import random
        
        gpu_metrics = {}
        for gpu_id in range(2):  # Mock 2 GPUs
            gpu_metrics[f"gpu_{gpu_id}"] = {
                "memory_percent": random.uniform(70, 95),
                "memory_used_mb": random.randint(8000, 15000),
                "memory_total_mb": 16000,
                "utilization_percent": random.uniform(80, 100),
                "temperature": random.randint(65, 88),
                "power_draw": random.randint(200, 300),
                "driver_version": "470.82.01"
            }
        
        return gpu_metrics
    
    async def _get_recent_logs(self, context: Context) -> List[Dict[str, Any]]:
        """Get recent log entries for OOM detection."""
        # Mock log entries with some OOM patterns
        import random
        
        log_entries = []
        oom_messages = [
            "CUDA out of memory. Tried to allocate 2.00 GiB (GPU 0; 15.78 GiB total capacity)",
            "RuntimeError: CUDA out of memory. Tried to allocate 1.50 GiB",
            "OutOfMemoryError: Unable to allocate array with shape and data type",
            "Process killed due to OOMKilled by cgroup memory controller",
            "MemoryError: Unable to allocate 8.00 GiB for an array"
        ]
        
        for i in range(10):  # Generate 10 mock log entries
            if random.random() < 0.3:  # 30% chance of OOM message
                message = random.choice(oom_messages)
            else:
                message = f"INFO: Processing batch {i} with model inference"
            
            log_entries.append({
                "timestamp": datetime.utcnow() - timedelta(minutes=random.randint(1, 60)),
                "message": message,
                "source": "model_service",
                "pid": random.randint(1000, 9999),
                "container_id": f"container_{random.randint(1, 5)}"
            })
        
        return log_entries
    
    async def _get_process_metrics(self) -> List[Dict[str, Any]]:
        """Get process-level resource metrics."""
        try:
            process_metrics = []
            
            # Get top processes by memory and CPU usage
            for proc in psutil.process_iter(['pid', 'name', 'memory_percent', 'cpu_percent']):
                try:
                    proc_info = proc.info
                    if proc_info['memory_percent'] > 5 or proc_info['cpu_percent'] > 10:  # Filter significant processes
                        memory_info = proc.memory_info()
                        process_metrics.append({
                            "pid": proc_info['pid'],
                            "name": proc_info['name'],
                            "memory_percent": proc_info['memory_percent'],
                            "memory_mb": memory_info.rss / (1024**2),
                            "cpu_percent": proc_info['cpu_percent']
                        })
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            # Sort by memory usage and return top 20
            process_metrics.sort(key=lambda x: x['memory_percent'], reverse=True)
            return process_metrics[:20]
            
        except Exception as e:
            logger.exception(f"Error getting process metrics: {e}")
            return []
    
    async def _get_container_metrics(self) -> List[Dict[str, Any]]:
        """Get container resource metrics (mock implementation)."""
        # Mock container metrics - in production, would use Docker API or Kubernetes API
        import random
        
        containers = []
        for i in range(3):  # Mock 3 containers
            containers.append({
                "id": f"container_{i}",
                "name": f"ml_service_{i}",
                "memory_usage_percent": random.uniform(60, 95),
                "memory_limit_mb": 8192,
                "memory_used_mb": random.randint(4000, 7800),
                "cpu_usage_percent": random.uniform(50, 90),
                "cpu_limit": 2.0,
                "restart_count": random.randint(0, 8),
                "status": "running"
            })
        
        return containers
    
    async def _analyze_resource_capacity(self, context: Context) -> Dict[str, Dict[str, Any]]:
        """Analyze resource capacity and utilization patterns."""
        capacity_analysis = {}
        
        # Analyze memory capacity
        memory_history = list(self.resource_history["memory_percent"])
        if len(memory_history) > 10:
            values = [entry["value"] for entry in memory_history[-50:]]  # Last 50 points
            
            avg_utilization = sum(values) / len(values)
            peak_utilization = max(values)
            
            if avg_utilization > 70 and peak_utilization > 85:
                utilization_trend = "high_sustained"
            elif avg_utilization > 60:
                utilization_trend = "moderate_sustained"
            else:
                utilization_trend = "normal"
            
            capacity_analysis["memory"] = {
                "average_utilization": avg_utilization,
                "peak_utilization": peak_utilization,
                "utilization_trend": utilization_trend
            }
        
        # Similar analysis for CPU
        cpu_history = list(self.resource_history["cpu_percent"])
        if len(cpu_history) > 10:
            values = [entry["value"] for entry in cpu_history[-50:]]
            
            avg_utilization = sum(values) / len(values)
            peak_utilization = max(values)
            
            if avg_utilization > 70 and peak_utilization > 85:
                utilization_trend = "high_sustained"
            elif avg_utilization > 60:
                utilization_trend = "moderate_sustained"
            else:
                utilization_trend = "normal"
            
            capacity_analysis["cpu"] = {
                "average_utilization": avg_utilization,
                "peak_utilization": peak_utilization,
                "utilization_trend": utilization_trend
            }
        
        return capacity_analysis
    
    def _get_gpu_memory_recommendation(self, memory_percent: float) -> str:
        """Get recommendation for GPU memory issues."""
        if memory_percent > 95:
            return "CRITICAL: GPU memory exhausted. Reduce batch size, enable gradient checkpointing, or use model parallelism"
        elif memory_percent > 90:
            return "HIGH: GPU memory very high. Consider reducing batch size or optimizing memory usage"
        else:
            return "Monitor GPU memory usage and optimize if possible"
    
    def _get_oom_recommendation(self, oom_detections: List[Dict[str, Any]]) -> str:
        """Get recommendation based on OOM pattern analysis."""
        if not oom_detections:
            return "Monitor memory usage patterns"
        
        # Analyze patterns
        cuda_oom = any("CUDA" in d["pattern"] for d in oom_detections)
        system_oom = any("MemoryError" in d["pattern"] or "OOMKilled" in d["pattern"] for d in oom_detections)
        
        if cuda_oom and system_oom:
            return "Multiple OOM types detected - review both GPU and system memory allocation"
        elif cuda_oom:
            return "GPU OOM detected - reduce batch size, enable gradient checkpointing, or use model parallelism"
        elif system_oom:
            return "System OOM detected - increase memory allocation or optimize memory usage"
        else:
            return "Review memory allocation patterns and optimize usage"
    
    def _get_container_memory_recommendation(self, memory_usage: float, memory_limit: int) -> str:
        """Get recommendation for container memory issues."""
        if memory_usage > 95:
            return f"CRITICAL: Container near memory limit ({memory_limit}MB). Increase memory limit immediately"
        elif memory_usage > 85:
            return f"HIGH: Container memory usage high. Consider increasing memory limit from {memory_limit}MB"
        else:
            return "Monitor container memory usage trends"


class ResourceTrendAnalyzer:
    """Utility class for analyzing resource usage trends."""
    
    @staticmethod
    def analyze_trend(values: List[float]) -> Dict[str, Any]:
        """Analyze trend in resource usage values."""
        if len(values) < 5:
            return {"trend": "insufficient_data", "confidence": 0.0}
        
        n = len(values)
        x_values = list(range(n))
        
        # Calculate linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x_values[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return {"trend": "stable", "confidence": 0.0, "slope": 0.0}
        
        slope = numerator / denominator
        
        # Calculate R-squared for confidence
        y_pred = [slope * (x - x_mean) + y_mean for x in x_values]
        ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
        ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))
        
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        confidence = max(0, min(1, r_squared))
        
        # Classify trend
        if abs(slope) < 0.1:
            trend = "stable"
        elif slope > 0.1:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        return {
            "trend": trend,
            "slope": slope,
            "confidence": confidence,
            "r_squared": r_squared
        }
    
    @staticmethod
    def detect_resource_spikes(values: List[float], threshold_std: float = 2.0) -> List[int]:
        """Detect resource usage spikes using statistical methods."""
        if len(values) < 10:
            return []
        
        import statistics
        
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values)
        
        spikes = []
        for i, value in enumerate(values):
            if std_val > 0:
                z_score = (value - mean_val) / std_val
                if z_score > threshold_std:
                    spikes.append(i)
        
        return spikes