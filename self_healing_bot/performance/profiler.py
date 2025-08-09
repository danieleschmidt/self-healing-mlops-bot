"""Comprehensive performance profiling and benchmarking system."""

import asyncio
import time
import cProfile
import pstats
import io
import sys
import gc
import threading
import traceback
import psutil
import logging
import json
import statistics
from typing import Dict, List, Any, Optional, Callable, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
from contextlib import contextmanager, asynccontextmanager
from functools import wraps
from pathlib import Path
import resource
import linecache
import inspect
import memory_profiler
import py_spy

logger = logging.getLogger(__name__)


@dataclass
class ProfileMetric:
    """Individual profiling metric."""
    name: str
    value: float
    unit: str
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    duration: float
    iterations: int
    ops_per_second: float
    memory_usage_mb: float
    cpu_percent: float
    success_rate: float
    error_count: int
    percentiles: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "duration": self.duration,
            "iterations": self.iterations,
            "ops_per_second": self.ops_per_second,
            "memory_usage_mb": self.memory_usage_mb,
            "cpu_percent": self.cpu_percent,
            "success_rate": self.success_rate,
            "error_count": self.error_count,
            "percentiles": self.percentiles,
            "metadata": self.metadata
        }


@dataclass
class ProfileSession:
    """A profiling session with comprehensive metrics."""
    session_id: str
    start_time: float
    end_time: Optional[float] = None
    metrics: List[ProfileMetric] = field(default_factory=list)
    function_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    memory_timeline: List[Dict[str, float]] = field(default_factory=list)
    cpu_timeline: List[Dict[str, float]] = field(default_factory=list)
    call_graph: Dict[str, List[str]] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time
    
    def add_metric(self, name: str, value: float, unit: str, metadata: Dict[str, Any] = None):
        """Add a metric to the session."""
        self.metrics.append(ProfileMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            metadata=metadata or {}
        ))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "metrics": [m.to_dict() for m in self.metrics],
            "function_stats": self.function_stats,
            "memory_timeline": self.memory_timeline,
            "cpu_timeline": self.cpu_timeline,
            "call_graph": self.call_graph
        }


class FunctionProfiler:
    """Decorator-based function profiler."""
    
    def __init__(self):
        self.function_stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "call_count": 0,
            "total_time": 0.0,
            "min_time": float('inf'),
            "max_time": 0.0,
            "avg_time": 0.0,
            "memory_usage": [],
            "error_count": 0,
            "last_called": None
        })
        self.active_calls: Dict[str, float] = {}
    
    def profile(self, include_memory: bool = True, include_args: bool = False):
        """Decorator to profile function performance."""
        def decorator(func: Callable) -> Callable:
            func_name = f"{func.__module__}.{func.__qualname__}"
            
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = None
                
                if include_memory:
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                self.active_calls[func_name] = start_time
                
                try:
                    result = await func(*args, **kwargs)
                    success = True
                except Exception as e:
                    self.function_stats[func_name]["error_count"] += 1
                    success = False
                    raise
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # Update statistics
                    stats = self.function_stats[func_name]
                    stats["call_count"] += 1
                    stats["total_time"] += duration
                    stats["min_time"] = min(stats["min_time"], duration)
                    stats["max_time"] = max(stats["max_time"], duration)
                    stats["avg_time"] = stats["total_time"] / stats["call_count"]
                    stats["last_called"] = datetime.now().isoformat()
                    
                    if include_memory and start_memory:
                        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        memory_delta = end_memory - start_memory
                        stats["memory_usage"].append(memory_delta)
                    
                    if include_args and success:
                        # Store sanitized args info
                        args_info = {
                            "arg_count": len(args),
                            "kwargs_count": len(kwargs),
                            "arg_types": [type(arg).__name__ for arg in args[:5]]  # First 5 only
                        }
                        if "args_patterns" not in stats:
                            stats["args_patterns"] = []
                        stats["args_patterns"].append(args_info)
                        
                        # Keep only last 10 patterns
                        if len(stats["args_patterns"]) > 10:
                            stats["args_patterns"] = stats["args_patterns"][-10:]
                    
                    self.active_calls.pop(func_name, None)
                
                return result
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                start_time = time.time()
                start_memory = None
                
                if include_memory:
                    start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                self.active_calls[func_name] = start_time
                
                try:
                    result = func(*args, **kwargs)
                    success = True
                except Exception as e:
                    self.function_stats[func_name]["error_count"] += 1
                    success = False
                    raise
                finally:
                    end_time = time.time()
                    duration = end_time - start_time
                    
                    # Update statistics (same as async)
                    stats = self.function_stats[func_name]
                    stats["call_count"] += 1
                    stats["total_time"] += duration
                    stats["min_time"] = min(stats["min_time"], duration)
                    stats["max_time"] = max(stats["max_time"], duration)
                    stats["avg_time"] = stats["total_time"] / stats["call_count"]
                    stats["last_called"] = datetime.now().isoformat()
                    
                    if include_memory and start_memory:
                        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                        memory_delta = end_memory - start_memory
                        stats["memory_usage"].append(memory_delta)
                    
                    self.active_calls.pop(func_name, None)
                
                return result
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive function profiling statistics."""
        stats = {}
        
        for func_name, func_stats in self.function_stats.items():
            # Calculate additional metrics
            processed_stats = func_stats.copy()
            
            if func_stats["memory_usage"]:
                processed_stats["avg_memory_delta"] = statistics.mean(func_stats["memory_usage"])
                processed_stats["max_memory_delta"] = max(func_stats["memory_usage"])
                processed_stats["min_memory_delta"] = min(func_stats["memory_usage"])
            
            # Calculate success rate
            total_calls = func_stats["call_count"]
            error_count = func_stats["error_count"]
            processed_stats["success_rate"] = (total_calls - error_count) / total_calls if total_calls > 0 else 0
            
            # Calculate calls per second (rough estimate)
            if func_stats["last_called"] and func_stats["call_count"] > 1:
                # This is a simplification - would need better time tracking
                processed_stats["estimated_calls_per_second"] = func_stats["call_count"] / max(1, func_stats["total_time"])
            
            stats[func_name] = processed_stats
        
        return {
            "functions": stats,
            "active_calls": len(self.active_calls),
            "total_functions_tracked": len(self.function_stats)
        }
    
    def reset_stats(self):
        """Reset all profiling statistics."""
        self.function_stats.clear()
        self.active_calls.clear()


class MemoryProfiler:
    """Advanced memory profiling and leak detection."""
    
    def __init__(self):
        self.snapshots: List[Dict[str, Any]] = []
        self.leak_candidates: Set[str] = set()
        self.monitoring_enabled = False
        self.monitor_task: Optional[asyncio.Task] = None
        
    def take_snapshot(self, label: str = None) -> Dict[str, Any]:
        """Take a memory snapshot."""
        import tracemalloc
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        snapshot = tracemalloc.take_snapshot()
        
        # Analyze top memory allocations
        top_stats = snapshot.statistics('lineno')
        
        snapshot_data = {
            "timestamp": time.time(),
            "label": label or f"snapshot_{len(self.snapshots)}",
            "total_memory_mb": psutil.Process().memory_info().rss / 1024 / 1024,
            "top_allocations": []
        }
        
        # Get top 20 allocations
        for index, stat in enumerate(top_stats[:20]):
            snapshot_data["top_allocations"].append({
                "filename": stat.traceback.format()[0],
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count,
                "average_size": stat.size / stat.count if stat.count > 0 else 0
            })
        
        self.snapshots.append(snapshot_data)
        
        # Detect potential leaks
        if len(self.snapshots) >= 3:
            self._detect_memory_leaks()
        
        return snapshot_data
    
    def _detect_memory_leaks(self):
        """Detect potential memory leaks by analyzing snapshots."""
        if len(self.snapshots) < 3:
            return
        
        # Compare recent snapshots
        recent = self.snapshots[-3:]
        
        # Check for consistently growing memory
        memory_growth = []
        for i in range(1, len(recent)):
            growth = recent[i]["total_memory_mb"] - recent[i-1]["total_memory_mb"]
            memory_growth.append(growth)
        
        if all(growth > 5 for growth in memory_growth):  # Growing by more than 5MB each snapshot
            logger.warning(f"Potential memory leak detected: {memory_growth}")
            
            # Analyze which files/functions are growing
            for snapshot in recent:
                for allocation in snapshot["top_allocations"][:5]:
                    filename = allocation["filename"]
                    if allocation["size_mb"] > 10:  # Large allocation
                        self.leak_candidates.add(filename)
    
    async def start_monitoring(self, interval: int = 60):
        """Start continuous memory monitoring."""
        if self.monitor_task and not self.monitor_task.done():
            return
        
        self.monitoring_enabled = True
        self.monitor_task = asyncio.create_task(self._monitoring_loop(interval))
    
    async def stop_monitoring(self):
        """Stop memory monitoring."""
        self.monitoring_enabled = False
        if self.monitor_task:
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitoring_loop(self, interval: int):
        """Background memory monitoring loop."""
        while self.monitoring_enabled:
            try:
                self.take_snapshot(f"auto_{datetime.now().strftime('%H%M%S')}")
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(interval)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """Get comprehensive memory analysis report."""
        if not self.snapshots:
            return {"error": "No memory snapshots available"}
        
        latest = self.snapshots[-1]
        
        report = {
            "current_memory_mb": latest["total_memory_mb"],
            "total_snapshots": len(self.snapshots),
            "leak_candidates": list(self.leak_candidates),
            "memory_trend": [],
            "top_current_allocations": latest["top_allocations"][:10],
            "recommendations": []
        }
        
        # Memory trend analysis
        if len(self.snapshots) >= 5:
            recent_snapshots = self.snapshots[-10:]  # Last 10 snapshots
            memory_values = [s["total_memory_mb"] for s in recent_snapshots]
            
            report["memory_trend"] = {
                "values": memory_values,
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": statistics.mean(memory_values),
                "growth_rate": (memory_values[-1] - memory_values[0]) / len(memory_values)
            }
            
            # Generate recommendations
            if report["memory_trend"]["growth_rate"] > 2:  # Growing by >2MB per snapshot
                report["recommendations"].append({
                    "type": "memory_leak",
                    "message": "Memory is growing consistently. Check for potential leaks.",
                    "action": "Review top allocations and leak candidates"
                })
            
            if latest["total_memory_mb"] > 1000:  # >1GB memory usage
                report["recommendations"].append({
                    "type": "high_memory",
                    "message": f"High memory usage: {latest['total_memory_mb']:.1f}MB",
                    "action": "Consider optimizing memory-intensive operations"
                })
        
        return report


class PerformanceBenchmarker:
    """Comprehensive benchmarking system."""
    
    def __init__(self):
        self.benchmark_results: List[BenchmarkResult] = []
        self.baseline_results: Dict[str, BenchmarkResult] = {}
        
    async def run_benchmark(
        self,
        name: str,
        benchmark_func: Callable,
        iterations: int = 100,
        warmup_iterations: int = 10,
        timeout_per_operation: float = 30.0,
        include_memory_profiling: bool = True,
        **kwargs
    ) -> BenchmarkResult:
        """Run a comprehensive benchmark."""
        logger.info(f"Starting benchmark: {name} ({iterations} iterations)")
        
        # Warmup
        logger.debug(f"Warming up with {warmup_iterations} iterations...")
        for _ in range(warmup_iterations):
            try:
                if asyncio.iscoroutinefunction(benchmark_func):
                    await asyncio.wait_for(benchmark_func(**kwargs), timeout=timeout_per_operation)
                else:
                    benchmark_func(**kwargs)
            except Exception as e:
                logger.warning(f"Warmup iteration failed: {e}")
        
        # Force garbage collection before benchmark
        gc.collect()
        
        # Actual benchmark
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_cpu_time = psutil.Process().cpu_times()
        
        iteration_times = []
        error_count = 0
        memory_samples = []
        
        for i in range(iterations):
            iteration_start = time.time()
            
            try:
                if asyncio.iscoroutinefunction(benchmark_func):
                    await asyncio.wait_for(benchmark_func(**kwargs), timeout=timeout_per_operation)
                else:
                    benchmark_func(**kwargs)
                
                iteration_time = time.time() - iteration_start
                iteration_times.append(iteration_time)
                
                # Sample memory usage periodically
                if include_memory_profiling and i % max(1, iterations // 10) == 0:
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    memory_samples.append(current_memory)
                
            except Exception as e:
                error_count += 1
                logger.debug(f"Benchmark iteration {i} failed: {e}")
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        end_cpu_time = psutil.Process().cpu_times()
        
        # Calculate metrics
        total_duration = end_time - start_time
        successful_iterations = iterations - error_count
        
        if not iteration_times:
            raise ValueError(f"All benchmark iterations failed for {name}\")\n        \n        ops_per_second = successful_iterations / total_duration if total_duration > 0 else 0\n        success_rate = successful_iterations / iterations\n        \n        # Calculate percentiles\n        percentiles = {}\n        if iteration_times:\n            sorted_times = sorted(iteration_times)\n            percentiles = {\n                \"p50\": np.percentile(sorted_times, 50),\n                \"p75\": np.percentile(sorted_times, 75),\n                \"p90\": np.percentile(sorted_times, 90),\n                \"p95\": np.percentile(sorted_times, 95),\n                \"p99\": np.percentile(sorted_times, 99)\n            }\n        \n        # CPU usage calculation\n        cpu_time_used = (end_cpu_time.user - start_cpu_time.user) + (end_cpu_time.system - start_cpu_time.system)\n        cpu_percent = (cpu_time_used / total_duration) * 100 if total_duration > 0 else 0\n        \n        # Memory usage\n        avg_memory = statistics.mean(memory_samples) if memory_samples else end_memory\n        \n        result = BenchmarkResult(\n            name=name,\n            duration=total_duration,\n            iterations=successful_iterations,\n            ops_per_second=ops_per_second,\n            memory_usage_mb=avg_memory,\n            cpu_percent=cpu_percent,\n            success_rate=success_rate,\n            error_count=error_count,\n            percentiles=percentiles,\n            metadata={\n                \"start_memory_mb\": start_memory,\n                \"end_memory_mb\": end_memory,\n                \"memory_delta_mb\": end_memory - start_memory,\n                \"avg_iteration_time_ms\": statistics.mean(iteration_times) * 1000 if iteration_times else 0,\n                \"min_iteration_time_ms\": min(iteration_times) * 1000 if iteration_times else 0,\n                \"max_iteration_time_ms\": max(iteration_times) * 1000 if iteration_times else 0\n            }\n        )\n        \n        self.benchmark_results.append(result)\n        logger.info(f\"Benchmark {name} completed: {ops_per_second:.2f} ops/sec, {success_rate:.1%} success rate\")\n        \n        return result\n    \n    def set_baseline(self, benchmark_name: str, result: BenchmarkResult = None):\n        \"\"\"Set a benchmark result as baseline for comparison.\"\"\"\n        if result:\n            self.baseline_results[benchmark_name] = result\n        else:\n            # Find most recent result for this benchmark\n            for result in reversed(self.benchmark_results):\n                if result.name == benchmark_name:\n                    self.baseline_results[benchmark_name] = result\n                    break\n    \n    def compare_with_baseline(self, result: BenchmarkResult) -> Dict[str, Any]:\n        \"\"\"Compare a benchmark result with its baseline.\"\"\"\n        baseline = self.baseline_results.get(result.name)\n        if not baseline:\n            return {\"error\": f\"No baseline found for {result.name}\"}\n        \n        comparison = {\n            \"benchmark_name\": result.name,\n            \"baseline_ops_per_second\": baseline.ops_per_second,\n            \"current_ops_per_second\": result.ops_per_second,\n            \"performance_change_percent\": ((result.ops_per_second - baseline.ops_per_second) / baseline.ops_per_second) * 100,\n            \"memory_change_mb\": result.memory_usage_mb - baseline.memory_usage_mb,\n            \"success_rate_change\": result.success_rate - baseline.success_rate,\n            \"improvements\": [],\n            \"regressions\": []\n        }\n        \n        # Identify improvements and regressions\n        if comparison[\"performance_change_percent\"] > 5:\n            comparison[\"improvements\"].append(f\"Performance improved by {comparison['performance_change_percent']:.1f}%\")\n        elif comparison[\"performance_change_percent\"] < -5:\n            comparison[\"regressions\"].append(f\"Performance degraded by {abs(comparison['performance_change_percent']):.1f}%\")\n        \n        if comparison[\"memory_change_mb\"] < -10:\n            comparison[\"improvements\"].append(f\"Memory usage reduced by {abs(comparison['memory_change_mb']):.1f}MB\")\n        elif comparison[\"memory_change_mb\"] > 10:\n            comparison[\"regressions\"].append(f\"Memory usage increased by {comparison['memory_change_mb']:.1f}MB\")\n        \n        if comparison[\"success_rate_change\"] > 0.05:\n            comparison[\"improvements\"].append(f\"Success rate improved by {comparison['success_rate_change']:.1%}\")\n        elif comparison[\"success_rate_change\"] < -0.05:\n            comparison[\"regressions\"].append(f\"Success rate degraded by {abs(comparison['success_rate_change']):.1%}\")\n        \n        return comparison\n    \n    def get_benchmark_summary(self) -> Dict[str, Any]:\n        \"\"\"Get summary of all benchmark results.\"\"\"\n        if not self.benchmark_results:\n            return {\"error\": \"No benchmark results available\"}\n        \n        # Group by benchmark name\n        by_name = defaultdict(list)\n        for result in self.benchmark_results:\n            by_name[result.name].append(result)\n        \n        summary = {\n            \"total_benchmarks_run\": len(self.benchmark_results),\n            \"unique_benchmarks\": len(by_name),\n            \"benchmarks\": {},\n            \"overall_trends\": {}\n        }\n        \n        # Analyze each benchmark\n        for name, results in by_name.items():\n            recent_results = sorted(results, key=lambda r: r.metadata.get('timestamp', 0))[-5:]  # Last 5 runs\n            \n            ops_per_sec = [r.ops_per_second for r in recent_results]\n            memory_usage = [r.memory_usage_mb for r in recent_results]\n            \n            summary[\"benchmarks\"][name] = {\n                \"runs_count\": len(results),\n                \"latest_result\": recent_results[-1].to_dict(),\n                \"trends\": {\n                    \"avg_ops_per_second\": statistics.mean(ops_per_sec),\n                    \"ops_per_second_trend\": \"improving\" if len(ops_per_sec) > 1 and ops_per_sec[-1] > ops_per_sec[0] else \"stable\",\n                    \"avg_memory_usage_mb\": statistics.mean(memory_usage),\n                    \"memory_trend\": \"increasing\" if len(memory_usage) > 1 and memory_usage[-1] > memory_usage[0] + 10 else \"stable\"\n                }\n            }\n        \n        return summary\n\n\nclass PerformanceProfiler:\n    \"\"\"Main performance profiler integrating all profiling tools.\"\"\"\n    \n    def __init__(self):\n        self.function_profiler = FunctionProfiler()\n        self.memory_profiler = MemoryProfiler()\n        self.benchmarker = PerformanceBenchmarker()\n        self.active_sessions: Dict[str, ProfileSession] = {}\n        self.profiling_enabled = True\n        \n        # Global profiler instance for easy decoration\n        global profiler\n        profiler = self.function_profiler\n    \n    def start_session(self, session_id: str = None) -> str:\n        \"\"\"Start a new profiling session.\"\"\"\n        session_id = session_id or f\"session_{int(time.time() * 1000)}\"\n        \n        session = ProfileSession(\n            session_id=session_id,\n            start_time=time.time()\n        )\n        \n        self.active_sessions[session_id] = session\n        \n        # Take initial memory snapshot\n        self.memory_profiler.take_snapshot(f\"session_start_{session_id}\")\n        \n        logger.info(f\"Started profiling session: {session_id}\")\n        return session_id\n    \n    def end_session(self, session_id: str) -> ProfileSession:\n        \"\"\"End a profiling session and return results.\"\"\"\n        if session_id not in self.active_sessions:\n            raise ValueError(f\"Session {session_id} not found\")\n        \n        session = self.active_sessions[session_id]\n        session.end_time = time.time()\n        \n        # Take final memory snapshot\n        self.memory_profiler.take_snapshot(f\"session_end_{session_id}\")\n        \n        # Collect function statistics\n        session.function_stats = self.function_profiler.get_stats()\n        \n        # Remove from active sessions\n        del self.active_sessions[session_id]\n        \n        logger.info(f\"Ended profiling session: {session_id} (duration: {session.duration:.2f}s)\")\n        return session\n    \n    @asynccontextmanager\n    async def profile_context(self, session_id: str = None):\n        \"\"\"Context manager for profiling a block of code.\"\"\"\n        session_id = self.start_session(session_id)\n        try:\n            yield session_id\n        finally:\n            session = self.end_session(session_id)\n            logger.info(f\"Profile context completed: {session_id}\")\n    \n    async def run_comprehensive_benchmark(self, target_functions: List[Callable]) -> Dict[str, Any]:\n        \"\"\"Run comprehensive benchmarks on target functions.\"\"\"\n        logger.info(f\"Running comprehensive benchmark on {len(target_functions)} functions\")\n        \n        results = {}\n        \n        for func in target_functions:\n            func_name = f\"{func.__module__}.{func.__qualname__}\"\n            \n            try:\n                # Create a simple benchmark wrapper\n                async def benchmark_wrapper():\n                    if asyncio.iscoroutinefunction(func):\n                        return await func()\n                    else:\n                        return func()\n                \n                result = await self.benchmarker.run_benchmark(\n                    name=func_name,\n                    benchmark_func=benchmark_wrapper,\n                    iterations=50,\n                    warmup_iterations=5\n                )\n                \n                results[func_name] = result.to_dict()\n                \n            except Exception as e:\n                logger.error(f\"Benchmark failed for {func_name}: {e}\")\n                results[func_name] = {\"error\": str(e)}\n        \n        return results\n    \n    def get_comprehensive_report(self) -> Dict[str, Any]:\n        \"\"\"Get a comprehensive performance report.\"\"\"\n        return {\n            \"timestamp\": datetime.now().isoformat(),\n            \"function_profiling\": self.function_profiler.get_stats(),\n            \"memory_analysis\": self.memory_profiler.get_memory_report(),\n            \"benchmark_summary\": self.benchmarker.get_benchmark_summary(),\n            \"active_sessions\": {\n                session_id: {\n                    \"duration\": session.duration,\n                    \"metrics_count\": len(session.metrics)\n                }\n                for session_id, session in self.active_sessions.items()\n            },\n            \"system_resources\": {\n                \"cpu_percent\": psutil.cpu_percent(),\n                \"memory_percent\": psutil.virtual_memory().percent,\n                \"memory_available_mb\": psutil.virtual_memory().available / 1024 / 1024,\n                \"disk_usage_percent\": psutil.disk_usage('/').percent if psutil.disk_usage('/') else 0\n            }\n        }\n    \n    async def export_report(self, file_path: str, format: str = \"json\"):\n        \"\"\"Export comprehensive report to file.\"\"\"\n        report = self.get_comprehensive_report()\n        \n        path = Path(file_path)\n        path.parent.mkdir(parents=True, exist_ok=True)\n        \n        if format.lower() == \"json\":\n            with open(path, 'w') as f:\n                json.dump(report, f, indent=2, default=str)\n        else:\n            raise ValueError(f\"Unsupported format: {format}\")\n        \n        logger.info(f\"Performance report exported to {file_path}\")\n    \n    def enable_profiling(self):\n        \"\"\"Enable performance profiling.\"\"\"\n        self.profiling_enabled = True\n    \n    def disable_profiling(self):\n        \"\"\"Disable performance profiling.\"\"\"\n        self.profiling_enabled = False\n\n\n# Global performance profiler instance\nperformance_profiler = PerformanceProfiler()\n\n# Global function profiler for easy access\nprofiler = performance_profiler.function_profiler\n\n# Convenience decorators\ndef profile_function(include_memory: bool = True, include_args: bool = False):\n    \"\"\"Convenience decorator for function profiling.\"\"\"\n    return profiler.profile(include_memory=include_memory, include_args=include_args)\n\n\n@contextmanager\ndef profile_block(name: str):\n    \"\"\"Context manager for profiling a block of code.\"\"\"\n    start_time = time.time()\n    start_memory = psutil.Process().memory_info().rss / 1024 / 1024\n    \n    try:\n        yield\n    finally:\n        end_time = time.time()\n        end_memory = psutil.Process().memory_info().rss / 1024 / 1024\n        \n        duration = end_time - start_time\n        memory_delta = end_memory - start_memory\n        \n        logger.info(f\"Profile block '{name}': {duration:.3f}s, memory delta: {memory_delta:.2f}MB\")\n\n\n# Example benchmark functions for bot components\nasync def benchmark_cache_operations():\n    \"\"\"Benchmark cache operations.\"\"\"\n    # This would benchmark actual cache operations\n    await asyncio.sleep(0.001)  # Simulate work\n\n\nasync def benchmark_detector_execution():\n    \"\"\"Benchmark detector execution.\"\"\"\n    # This would benchmark actual detector execution\n    await asyncio.sleep(0.005)  # Simulate work\n\n\nasync def benchmark_concurrent_tasks():\n    \"\"\"Benchmark concurrent task execution.\"\"\"\n    # This would benchmark actual concurrent task processing\n    await asyncio.sleep(0.002)  # Simulate work"