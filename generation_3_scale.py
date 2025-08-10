#!/usr/bin/env python3
"""
TERRAGON SDLC - Generation 3: MAKE IT SCALE
Performance optimization, caching, concurrent processing, auto-scaling
"""

import asyncio
import logging
import sys
import time
import random
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import wraps, lru_cache
from collections import defaultdict, deque
import hashlib
import json

# Enhanced logging for performance tracking
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PerformanceCache:
    """High-performance caching system with TTL and LRU eviction."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict] = {}
        self.access_times: deque = deque()
        self.lock = threading.RLock()
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "size": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                if time.time() < entry["expires_at"]:
                    self.stats["hits"] += 1
                    self._update_access_time(key)
                    return entry["value"]
                else:
                    # Expired entry
                    del self.cache[key]
                    self.stats["evictions"] += 1
            
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        with self.lock:
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            expires_at = time.time() + (ttl or self.default_ttl)
            self.cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": time.time()
            }
            self._update_access_time(key)
            self.stats["size"] = len(self.cache)
    
    def _update_access_time(self, key: str):
        # Remove old entries for this key
        self.access_times = deque([entry for entry in self.access_times if entry[1] != key])
        # Add new entry
        self.access_times.append((time.time(), key))
    
    def _evict_lru(self):
        if self.access_times:
            _, lru_key = self.access_times.popleft()
            if lru_key in self.cache:
                del self.cache[lru_key]
                self.stats["evictions"] += 1

class ConcurrencyManager:
    """Advanced concurrency management with adaptive pool sizing."""
    
    def __init__(self):
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        self.process_pool = ProcessPoolExecutor(max_workers=2)
        self.semaphore = asyncio.Semaphore(10)
        self.request_queue = asyncio.Queue(maxsize=100)
        self.worker_stats = defaultdict(int)
        self.adaptive_scaling = True
    
    async def execute_concurrent(self, tasks: List[Callable], max_concurrent: int = 5) -> List[Any]:
        """Execute tasks concurrently with rate limiting."""
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def limited_task(task):
            async with semaphore:
                start_time = time.time()
                try:
                    if asyncio.iscoroutinefunction(task):
                        result = await task()
                    else:
                        result = task()
                    
                    duration = time.time() - start_time
                    self.worker_stats["successful_tasks"] += 1
                    self.worker_stats["total_duration"] += duration
                    return result
                except Exception as e:
                    self.worker_stats["failed_tasks"] += 1
                    logger.error(f"Task failed: {e}")
                    return None
        
        results = await asyncio.gather(*[limited_task(task) for task in tasks])
        return results
    
    async def cpu_intensive_task(self, func: Callable, *args) -> Any:
        """Execute CPU-intensive tasks in process pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.process_pool, func, *args)
    
    async def io_bound_task(self, func: Callable, *args) -> Any:
        """Execute I/O-bound tasks in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.thread_pool, func, *args)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        total_tasks = self.worker_stats["successful_tasks"] + self.worker_stats["failed_tasks"]
        avg_duration = 0
        if self.worker_stats["successful_tasks"] > 0:
            avg_duration = self.worker_stats["total_duration"] / self.worker_stats["successful_tasks"]
        
        return {
            "total_tasks": total_tasks,
            "success_rate": self.worker_stats["successful_tasks"] / max(1, total_tasks),
            "average_duration": avg_duration,
            "thread_pool_size": self.thread_pool._max_workers,
            "process_pool_size": self.process_pool._max_workers
        }

class AutoScaler:
    """Auto-scaling system based on load and performance metrics."""
    
    def __init__(self):
        self.metrics_history = deque(maxlen=100)
        self.scaling_decisions = []
        self.current_capacity = 1.0
        self.target_utilization = 0.75
        self.scale_up_threshold = 0.85
        self.scale_down_threshold = 0.5
    
    def record_metrics(self, cpu_usage: float, memory_usage: float, request_rate: float):
        timestamp = time.time()
        metrics = {
            "timestamp": timestamp,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "request_rate": request_rate,
            "utilization": max(cpu_usage, memory_usage)
        }
        self.metrics_history.append(metrics)
    
    def should_scale_up(self) -> bool:
        if len(self.metrics_history) < 5:
            return False
        
        recent_metrics = list(self.metrics_history)[-5:]
        avg_utilization = sum(m["utilization"] for m in recent_metrics) / len(recent_metrics)
        
        return avg_utilization > self.scale_up_threshold
    
    def should_scale_down(self) -> bool:
        if len(self.metrics_history) < 10:
            return False
        
        recent_metrics = list(self.metrics_history)[-10:]
        avg_utilization = sum(m["utilization"] for m in recent_metrics) / len(recent_metrics)
        
        return avg_utilization < self.scale_down_threshold and self.current_capacity > 1.0
    
    def make_scaling_decision(self) -> Optional[str]:
        if self.should_scale_up():
            self.current_capacity = min(5.0, self.current_capacity * 1.5)
            decision = f"scale_up_to_{self.current_capacity:.1f}"
            self.scaling_decisions.append((time.time(), decision))
            return decision
        elif self.should_scale_down():
            self.current_capacity = max(1.0, self.current_capacity * 0.8)
            decision = f"scale_down_to_{self.current_capacity:.1f}"
            self.scaling_decisions.append((time.time(), decision))
            return decision
        
        return None

class PerformanceOptimizer:
    """Advanced performance optimization and monitoring."""
    
    def __init__(self):
        self.cache = PerformanceCache(max_size=2000, default_ttl=1800)
        self.concurrency = ConcurrencyManager()
        self.auto_scaler = AutoScaler()
        self.performance_metrics = {
            "total_requests": 0,
            "total_response_time": 0.0,
            "cache_hit_rate": 0.0,
            "concurrency_utilization": 0.0,
            "optimization_events": 0
        }
    
    def cached_operation(self, ttl: int = 3600):
        """Decorator for caching expensive operations."""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                # Create cache key from function name and arguments
                key_data = f"{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
                cache_key = hashlib.md5(key_data.encode()).hexdigest()
                
                # Try to get from cache
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function and cache result
                start_time = time.time()
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                execution_time = time.time() - start_time
                self.cache.set(cache_key, result, ttl)
                
                # Update metrics
                self.performance_metrics["total_requests"] += 1
                self.performance_metrics["total_response_time"] += execution_time
                
                return result
            return wrapper
        return decorator
    
    async def optimize_workload(self, workload_func: Callable, data: List[Any]) -> List[Any]:
        """Optimize workload execution with batching and concurrency."""
        batch_size = min(50, len(data))
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        
        start_time = time.time()
        
        # Process batches concurrently
        async def process_batch(batch):
            tasks = [workload_func(item) for item in batch]
            return await self.concurrency.execute_concurrent(tasks, max_concurrent=8)
        
        all_results = []
        for batch in batches:
            batch_results = await process_batch(batch)
            all_results.extend(batch_results)
        
        # Record performance metrics
        execution_time = time.time() - start_time
        throughput = len(data) / execution_time if execution_time > 0 else 0
        
        # Simulate resource usage for auto-scaling
        cpu_usage = min(0.9, 0.3 + (len(data) / 1000))
        memory_usage = min(0.85, 0.2 + (len(data) / 2000))
        request_rate = len(data) / max(1, execution_time)
        
        self.auto_scaler.record_metrics(cpu_usage, memory_usage, request_rate)
        
        # Make scaling decision
        scaling_decision = self.auto_scaler.make_scaling_decision()
        if scaling_decision:
            logger.info(f"Auto-scaling decision: {scaling_decision}")
            self.performance_metrics["optimization_events"] += 1
        
        logger.info(f"Processed {len(data)} items in {execution_time:.2f}s (throughput: {throughput:.1f} items/s)")
        
        return all_results
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cache_stats = self.cache.stats
        concurrency_stats = self.concurrency.get_performance_stats()
        
        # Calculate cache hit rate
        total_cache_requests = cache_stats["hits"] + cache_stats["misses"]
        cache_hit_rate = cache_stats["hits"] / max(1, total_cache_requests)
        
        # Calculate average response time
        avg_response_time = 0
        if self.performance_metrics["total_requests"] > 0:
            avg_response_time = (
                self.performance_metrics["total_response_time"] / 
                self.performance_metrics["total_requests"]
            )
        
        return {
            "cache": {
                "hit_rate": f"{cache_hit_rate:.1%}",
                "size": cache_stats["size"],
                "evictions": cache_stats["evictions"]
            },
            "concurrency": concurrency_stats,
            "auto_scaling": {
                "current_capacity": self.auto_scaler.current_capacity,
                "decisions_made": len(self.auto_scaler.scaling_decisions),
                "metrics_collected": len(self.auto_scaler.metrics_history)
            },
            "performance": {
                "avg_response_time": f"{avg_response_time:.3f}s",
                "total_requests": self.performance_metrics["total_requests"],
                "optimization_events": self.performance_metrics["optimization_events"]
            }
        }

class ScalingDemo:
    """Generation 3 scaling demonstration."""
    
    def __init__(self):
        self.optimizer = PerformanceOptimizer()
        
    async def demonstrate_scaling(self):
        """Demonstrate all scaling and performance features."""
        
        print("\n⚡ TERRAGON SDLC - Generation 3: MAKE IT SCALE")
        print("=" * 70)
        
        try:
            # 1. Caching Performance
            print("\n1️⃣ High-Performance Caching")
            await self._demo_caching()
            
            # 2. Concurrent Processing
            print("\n2️⃣ Concurrent Processing")
            await self._demo_concurrency()
            
            # 3. Auto-Scaling
            print("\n3️⃣ Auto-Scaling System")
            await self._demo_auto_scaling()
            
            # 4. Load Balancing
            print("\n4️⃣ Load Balancing")
            await self._demo_load_balancing()
            
            # 5. Resource Pooling
            print("\n5️⃣ Resource Pooling")
            await self._demo_resource_pooling()
            
            # 6. Performance Optimization
            print("\n6️⃣ Performance Optimization")
            await self._demo_performance_optimization()
            
            # Generate final performance report
            print("\n📊 FINAL PERFORMANCE REPORT")
            report = self.optimizer.get_performance_report()
            self._print_performance_report(report)
            
            print(f"\n🎉 GENERATION 3 COMPLETE - System optimized for scale!")
            print(f"🚀 Ready for Quality Gates and Global Features!")
            
            return True
            
        except Exception as e:
            logger.error(f"Generation 3 failed: {e}", exc_info=True)
            return False
    
    async def _demo_caching(self):
        """Demonstrate high-performance caching."""
        
        @self.optimizer.cached_operation(ttl=60)
        async def expensive_computation(x: int) -> int:
            await asyncio.sleep(0.1)  # Simulate expensive operation
            return x * x * x
        
        # Test caching performance
        start_time = time.time()
        
        # First run (cache misses)
        tasks = [expensive_computation(i) for i in range(20)]
        results1 = await asyncio.gather(*tasks)
        first_run_time = time.time() - start_time
        
        # Second run (cache hits)
        start_time = time.time()
        tasks = [expensive_computation(i) for i in range(20)]
        results2 = await asyncio.gather(*tasks)
        second_run_time = time.time() - start_time
        
        speedup = first_run_time / max(0.001, second_run_time)
        
        print(f"   ✅ First run (cache misses): {first_run_time:.2f}s")
        print(f"   ✅ Second run (cache hits): {second_run_time:.2f}s")
        print(f"   🚀 Speedup: {speedup:.1f}x")
    
    async def _demo_concurrency(self):
        """Demonstrate concurrent processing capabilities."""
        
        async def cpu_task():
            # Simulate CPU-intensive work
            await asyncio.sleep(0.05)
            return sum(i*i for i in range(1000))
        
        async def io_task():
            # Simulate I/O-bound work
            await asyncio.sleep(0.02)
            return random.randint(1, 100)
        
        # Test concurrent execution
        start_time = time.time()
        
        cpu_tasks = [cpu_task for _ in range(20)]
        io_tasks = [io_task for _ in range(30)]
        all_tasks = cpu_tasks + io_tasks
        
        results = await self.optimizer.concurrency.execute_concurrent(all_tasks, max_concurrent=10)
        
        execution_time = time.time() - start_time
        successful_results = [r for r in results if r is not None]
        
        print(f"   ✅ Processed {len(all_tasks)} tasks concurrently")
        print(f"   ✅ Success rate: {len(successful_results)}/{len(all_tasks)} ({len(successful_results)/len(all_tasks):.1%})")
        print(f"   ⏱️ Total time: {execution_time:.2f}s")
        print(f"   🏎️ Throughput: {len(all_tasks)/execution_time:.1f} tasks/sec")
    
    async def _demo_auto_scaling(self):
        """Demonstrate auto-scaling capabilities."""
        
        # Simulate varying load patterns
        load_patterns = [
            {"requests": 50, "complexity": 0.01},   # Low load
            {"requests": 200, "complexity": 0.02},  # Medium load
            {"requests": 500, "complexity": 0.03},  # High load
            {"requests": 100, "complexity": 0.01},  # Back to normal
        ]
        
        scaling_events = 0
        
        for i, pattern in enumerate(load_patterns):
            print(f"   📈 Load pattern {i+1}: {pattern['requests']} requests")
            
            # Simulate workload
            async def workload_item(item):
                await asyncio.sleep(pattern["complexity"])
                return random.randint(1, 100)
            
            # Generate workload data
            workload_data = list(range(pattern["requests"]))
            
            # Process with optimization
            results = await self.optimizer.optimize_workload(workload_item, workload_data)
            
            # Check if scaling occurred
            if self.optimizer.performance_metrics["optimization_events"] > scaling_events:
                scaling_events = self.optimizer.performance_metrics["optimization_events"]
                print(f"      🔧 Auto-scaling triggered!")
        
        print(f"   ✅ Auto-scaling events: {scaling_events}")
        print(f"   📊 Current capacity: {self.optimizer.auto_scaler.current_capacity:.1f}")
    
    async def _demo_load_balancing(self):
        """Demonstrate load balancing capabilities."""
        
        # Simulate multiple service endpoints
        services = ["service_a", "service_b", "service_c"]
        service_loads = {service: 0 for service in services}
        
        async def call_service(service_name: str, request_id: int):
            service_loads[service_name] += 1
            await asyncio.sleep(0.01 + random.random() * 0.02)
            return f"{service_name}:{request_id}"
        
        # Round-robin load balancing
        requests = 150
        tasks = []
        
        for i in range(requests):
            selected_service = services[i % len(services)]
            tasks.append(call_service(selected_service, i))
        
        start_time = time.time()
        results = await self.optimizer.concurrency.execute_concurrent(tasks, max_concurrent=15)
        execution_time = time.time() - start_time
        
        print(f"   ✅ Distributed {requests} requests across {len(services)} services")
        for service, load in service_loads.items():
            print(f"      {service}: {load} requests ({load/requests:.1%})")
        print(f"   ⚡ Total time: {execution_time:.2f}s")
    
    async def _demo_resource_pooling(self):
        """Demonstrate advanced resource pooling."""
        
        # Simulate database connections pool
        class ConnectionPool:
            def __init__(self, max_size: int = 10):
                self.max_size = max_size
                self.available = asyncio.Queue(maxsize=max_size)
                self.in_use = set()
                self._initialize_pool()
            
            def _initialize_pool(self):
                for i in range(self.max_size):
                    self.available.put_nowait(f"conn_{i}")
            
            async def acquire(self):
                conn = await self.available.get()
                self.in_use.add(conn)
                return conn
            
            async def release(self, conn):
                if conn in self.in_use:
                    self.in_use.remove(conn)
                    await self.available.put(conn)
        
        pool = ConnectionPool(max_size=8)
        
        async def database_operation(op_id: int):
            conn = await pool.acquire()
            try:
                # Simulate database work
                await asyncio.sleep(0.02 + random.random() * 0.01)
                return f"result_{op_id}"
            finally:
                await pool.release(conn)
        
        # Test resource pooling under high concurrency
        operations = [database_operation(i) for i in range(50)]
        
        start_time = time.time()
        results = await asyncio.gather(*operations)
        execution_time = time.time() - start_time
        
        print(f"   ✅ Completed {len(operations)} database operations")
        print(f"   🏊 Pool size: {pool.max_size} connections")
        print(f"   ⏱️ Time: {execution_time:.2f}s")
        print(f"   🔄 Pool utilization: {len(pool.in_use)}/{pool.max_size} active")
    
    async def _demo_performance_optimization(self):
        """Demonstrate performance optimization techniques."""
        
        # Memory-efficient batch processing
        async def process_large_dataset(size: int = 1000):
            # Simulate memory-efficient processing
            batch_size = 100
            processed = 0
            
            for batch_start in range(0, size, batch_size):
                batch_end = min(batch_start + batch_size, size)
                batch_data = list(range(batch_start, batch_end))
                
                # Process batch
                batch_results = []
                for item in batch_data:
                    # Simulate processing
                    result = item * 2 + 1
                    batch_results.append(result)
                
                processed += len(batch_results)
                
                # Yield control to allow other operations
                if processed % 300 == 0:
                    await asyncio.sleep(0)
            
            return processed
        
        start_time = time.time()
        processed_count = await process_large_dataset(2000)
        execution_time = time.time() - start_time
        
        print(f"   ✅ Processed {processed_count} items efficiently")
        print(f"   📈 Processing rate: {processed_count/execution_time:.0f} items/sec")
        print(f"   💾 Memory-efficient batching used")
    
    def _print_performance_report(self, report: Dict[str, Any]):
        """Print formatted performance report."""
        print("   " + "="*50)
        print(f"   🏆 Cache Hit Rate: {report['cache']['hit_rate']}")
        print(f"   📦 Cache Size: {report['cache']['size']} items")
        print(f"   🔄 Cache Evictions: {report['cache']['evictions']}")
        print(f"   🏎️ Avg Response Time: {report['performance']['avg_response_time']}")
        print(f"   📊 Total Requests: {report['performance']['total_requests']}")
        print(f"   🚀 Auto-scaling Events: {report['auto_scaling']['decisions_made']}")
        print(f"   📈 Current Capacity: {report['auto_scaling']['current_capacity']}")
        print(f"   🎯 Success Rate: {report['concurrency']['success_rate']:.1%}")
        print("   " + "="*50)

async def main():
    """Main execution for Generation 3."""
    demo = ScalingDemo()
    
    start_time = time.time()
    success = await demo.demonstrate_scaling()
    duration = time.time() - start_time
    
    print(f"\n⏱️ Total execution time: {duration:.2f} seconds")
    
    if success:
        print("🎉 Generation 3 COMPLETED - System optimized for massive scale!")
        print("🔜 Proceeding to Quality Gates and Global Features...")
        return True
    else:
        print("❌ Generation 3 FAILED - Need additional performance improvements")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)