"""Enhanced concurrent processing with advanced patterns, connection pooling, and deadlock detection."""

import asyncio
import time
import threading
import queue
import heapq
import weakref
import psutil
import logging
from typing import Dict, List, Any, Optional, Callable, Union, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, deque
from enum import Enum, IntEnum
import resource
import signal
import gc
import inspect
import traceback
from contextlib import asynccontextmanager
import socket
import ssl
import aiohttp
import aiofiles

logger = logging.getLogger(__name__)


class TaskPriority(IntEnum):
    """Task priority levels (lower number = higher priority)."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskState(Enum):
    """Task execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


class ConnectionPoolType(Enum):
    """Types of connection pools."""
    HTTP = "http"
    DATABASE = "database"
    REDIS = "redis"
    CUSTOM = "custom"


@dataclass
class TaskMetrics:
    """Enhanced metrics for tracking task performance."""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    cancelled_tasks: int = 0
    retried_tasks: int = 0
    avg_execution_time: float = 0.0
    max_execution_time: float = 0.0
    min_execution_time: float = float('inf')
    total_execution_time: float = 0.0
    avg_queue_time: float = 0.0
    total_queue_time: float = 0.0
    active_tasks: int = 0
    peak_active_tasks: int = 0
    
    # Priority-specific metrics
    priority_metrics: Dict[TaskPriority, Dict[str, float]] = field(default_factory=dict)
    
    def update(self, execution_time: float, queue_time: float, success: bool, 
               priority: TaskPriority = TaskPriority.NORMAL, cancelled: bool = False):
        """Update metrics with new task completion."""
        self.total_tasks += 1
        self.total_execution_time += execution_time
        self.total_queue_time += queue_time
        
        if cancelled:
            self.cancelled_tasks += 1
        elif success:
            self.completed_tasks += 1
        else:
            self.failed_tasks += 1
        
        if execution_time > 0:
            self.max_execution_time = max(self.max_execution_time, execution_time)
            self.min_execution_time = min(self.min_execution_time, execution_time)
            
        if self.total_tasks > 0:
            self.avg_execution_time = self.total_execution_time / self.total_tasks
            self.avg_queue_time = self.total_queue_time / self.total_tasks
        
        # Update priority-specific metrics
        if priority not in self.priority_metrics:
            self.priority_metrics[priority] = {
                "count": 0, "success_rate": 0.0, "avg_time": 0.0
            }
        
        pm = self.priority_metrics[priority]
        pm["count"] += 1
        if success and not cancelled:
            pm["success_rate"] = (pm["success_rate"] * (pm["count"] - 1) + 1) / pm["count"]
        else:
            pm["success_rate"] = (pm["success_rate"] * (pm["count"] - 1)) / pm["count"]
        pm["avg_time"] = (pm["avg_time"] * (pm["count"] - 1) + execution_time) / pm["count"]
    
    def update_active_tasks(self, active_count: int):
        """Update active task metrics."""
        self.active_tasks = active_count
        self.peak_active_tasks = max(self.peak_active_tasks, active_count)


@dataclass
class PriorityTask:
    """Priority-based task wrapper."""
    priority: TaskPriority
    task_id: str
    created_at: float
    func: Callable
    args: tuple
    kwargs: dict
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    callback: Optional[Callable] = None
    dependencies: Set[str] = field(default_factory=set)
    
    def __lt__(self, other):
        """Priority comparison for heap queue."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at


class ConnectionPool:
    """Generic connection pool with health monitoring."""
    
    def __init__(
        self,
        pool_type: ConnectionPoolType,
        min_size: int = 5,
        max_size: int = 50,
        max_idle_time: float = 300.0,
        health_check_interval: float = 60.0,
        connection_factory: Optional[Callable] = None,
        health_check_func: Optional[Callable] = None
    ):
        self.pool_type = pool_type
        self.min_size = min_size
        self.max_size = max_size
        self.max_idle_time = max_idle_time
        self.health_check_interval = health_check_interval
        self.connection_factory = connection_factory
        self.health_check_func = health_check_func
        
        self._pool: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._active_connections: Set = set()
        self._connection_created_at: Dict = {}
        self._connection_last_used: Dict = {}
        self._lock = asyncio.Lock()
        self._health_check_task: Optional[asyncio.Task] = None
        self._metrics = {
            "created": 0,
            "destroyed": 0,
            "borrowed": 0,
            "returned": 0,
            "health_checks": 0,
            "failed_health_checks": 0
        }
    
    async def initialize(self):
        """Initialize the connection pool."""
        # Create minimum number of connections
        for _ in range(self.min_size):
            try:
                conn = await self._create_connection()
                await self._pool.put(conn)
            except Exception as e:
                logger.error(f"Failed to create initial connection: {e}")
        
        # Start health check task
        if self.health_check_func:
            self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Connection pool initialized: {self.pool_type.value}")
    
    async def _create_connection(self):
        """Create a new connection."""
        if not self.connection_factory:
            raise ValueError("Connection factory not provided")
        
        conn = await self.connection_factory()
        conn_id = id(conn)
        self._connection_created_at[conn_id] = time.time()
        self._connection_last_used[conn_id] = time.time()
        self._metrics["created"] += 1
        return conn
    
    @asynccontextmanager
    async def acquire(self, timeout: float = 30.0):
        """Acquire a connection from the pool."""
        start_time = time.time()
        conn = None
        
        try:
            # Try to get existing connection
            try:
                conn = await asyncio.wait_for(self._pool.get(), timeout=timeout)
                self._metrics["borrowed"] += 1
            except asyncio.TimeoutError:
                # Create new connection if pool is empty and under max size
                async with self._lock:
                    if len(self._active_connections) < self.max_size:
                        conn = await self._create_connection()
                        self._metrics["borrowed"] += 1
                    else:
                        raise Exception("Connection pool exhausted")
            
            # Add to active connections
            self._active_connections.add(conn)
            self._connection_last_used[id(conn)] = time.time()
            
            yield conn
            
        finally:
            if conn:
                # Remove from active connections
                self._active_connections.discard(conn)
                
                # Return to pool if healthy and not too old
                if await self._is_connection_healthy(conn):
                    try:
                        self._pool.put_nowait(conn)
                        self._metrics["returned"] += 1
                    except asyncio.QueueFull:
                        # Pool is full, destroy connection
                        await self._destroy_connection(conn)
                else:
                    await self._destroy_connection(conn)
    
    async def _is_connection_healthy(self, conn) -> bool:
        """Check if connection is healthy."""
        if not self.health_check_func:
            return True
        
        try:
            return await self.health_check_func(conn)
        except Exception as e:
            logger.debug(f"Health check failed: {e}")
            return False
    
    async def _destroy_connection(self, conn):
        """Destroy a connection."""
        try:
            if hasattr(conn, 'close'):
                if asyncio.iscoroutinefunction(conn.close):
                    await conn.close()
                else:
                    conn.close()
            
            conn_id = id(conn)
            self._connection_created_at.pop(conn_id, None)
            self._connection_last_used.pop(conn_id, None)
            self._metrics["destroyed"] += 1
            
        except Exception as e:
            logger.debug(f"Error destroying connection: {e}")
    
    async def _health_check_loop(self):
        """Background health check loop."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._perform_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def _perform_health_checks(self):
        """Perform health checks on idle connections."""
        current_time = time.time()
        connections_to_check = []
        
        # Get connections from pool for health checking
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                connections_to_check.append(conn)
            except asyncio.QueueEmpty:
                break
        
        healthy_connections = []
        
        for conn in connections_to_check:
            conn_id = id(conn)
            last_used = self._connection_last_used.get(conn_id, 0)
            
            # Check if connection is too old or idle
            if (current_time - last_used > self.max_idle_time or
                current_time - self._connection_created_at.get(conn_id, 0) > 3600):  # 1 hour max age
                await self._destroy_connection(conn)
                continue
            
            # Perform health check
            self._metrics["health_checks"] += 1
            if await self._is_connection_healthy(conn):
                healthy_connections.append(conn)
            else:
                self._metrics["failed_health_checks"] += 1
                await self._destroy_connection(conn)
        
        # Return healthy connections to pool
        for conn in healthy_connections:
            try:
                self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                await self._destroy_connection(conn)
        
        # Ensure minimum pool size
        current_size = self._pool.qsize()
        if current_size < self.min_size:
            for _ in range(self.min_size - current_size):
                try:
                    conn = await self._create_connection()
                    await self._pool.put(conn)
                except Exception as e:
                    logger.error(f"Failed to create connection for minimum pool size: {e}")
                    break
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        return {
            "pool_type": self.pool_type.value,
            "pool_size": self._pool.qsize(),
            "active_connections": len(self._active_connections),
            "max_size": self.max_size,
            "min_size": self.min_size,
            "metrics": self._metrics.copy()
        }
    
    async def shutdown(self):
        """Shutdown the connection pool."""
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        # Close all connections
        connections = []
        
        # Get connections from pool
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                connections.append(conn)
            except asyncio.QueueEmpty:
                break
        
        # Add active connections
        connections.extend(list(self._active_connections))
        
        # Close all connections
        for conn in connections:
            await self._destroy_connection(conn)
        
        logger.info(f"Connection pool shutdown: {self.pool_type.value}")


class DeadlockDetector:
    """Deadlock detection and prevention system."""
    
    def __init__(self):
        self._lock_graph: Dict[str, Set[str]] = defaultdict(set)  # resource -> waiting tasks
        self._task_resources: Dict[str, Set[str]] = defaultdict(set)  # task -> held resources
        self._resource_owners: Dict[str, str] = {}  # resource -> owning task
        self._lock = asyncio.Lock()
        self._detection_enabled = True
        self._deadlock_timeout = 30.0  # seconds
        
    async def acquire_resource(self, task_id: str, resource_id: str, timeout: float = 30.0) -> bool:
        """Attempt to acquire a resource with deadlock detection."""
        if not self._detection_enabled:
            return True
        
        async with self._lock:
            # Check if resource is available
            if resource_id not in self._resource_owners:
                self._resource_owners[resource_id] = task_id
                self._task_resources[task_id].add(resource_id)
                return True
            
            # Resource is held by another task
            owner_task = self._resource_owners[resource_id]
            
            # Check for potential deadlock
            if await self._would_cause_deadlock(task_id, owner_task):
                logger.warning(f"Potential deadlock detected: task {task_id} waiting for resource {resource_id} held by {owner_task}")
                return False
            
            # Add to waiting graph
            self._lock_graph[resource_id].add(task_id)
        
        # Wait for resource to become available
        start_time = time.time()
        while time.time() - start_time < timeout:
            async with self._lock:
                if resource_id not in self._resource_owners:
                    self._resource_owners[resource_id] = task_id
                    self._task_resources[task_id].add(resource_id)
                    self._lock_graph[resource_id].discard(task_id)
                    return True
            
            await asyncio.sleep(0.1)  # Short wait
        
        # Timeout - remove from waiting graph
        async with self._lock:
            self._lock_graph[resource_id].discard(task_id)
        
        return False
    
    async def release_resource(self, task_id: str, resource_id: str):
        """Release a resource."""
        if not self._detection_enabled:
            return
        
        async with self._lock:
            if self._resource_owners.get(resource_id) == task_id:
                del self._resource_owners[resource_id]
                self._task_resources[task_id].discard(resource_id)
    
    async def _would_cause_deadlock(self, requesting_task: str, resource_owner: str) -> bool:
        """Check if granting the resource would cause a deadlock."""
        # Use DFS to detect cycles in the wait-for graph
        visited = set()
        recursion_stack = set()
        
        return self._has_cycle_dfs(requesting_task, resource_owner, visited, recursion_stack)
    
    def _has_cycle_dfs(self, current_task: str, target_task: str, visited: Set[str], recursion_stack: Set[str]) -> bool:
        """Detect cycle using depth-first search."""
        if current_task == target_task:
            return True
        
        if current_task in recursion_stack:
            return True
        
        if current_task in visited:
            return False
        
        visited.add(current_task)
        recursion_stack.add(current_task)
        
        # Check resources held by current task
        for resource in self._task_resources.get(current_task, set()):
            # Check who is waiting for this resource
            for waiting_task in self._lock_graph.get(resource, set()):
                if self._has_cycle_dfs(waiting_task, target_task, visited, recursion_stack):
                    return True
        
        recursion_stack.remove(current_task)
        return False
    
    def enable_detection(self):
        """Enable deadlock detection."""
        self._detection_enabled = True
    
    def disable_detection(self):
        """Disable deadlock detection."""
        self._detection_enabled = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get deadlock detector statistics."""
        return {
            "enabled": self._detection_enabled,
            "active_resources": len(self._resource_owners),
            "waiting_tasks": sum(len(waiters) for waiters in self._lock_graph.values()),
            "resource_graph_size": len(self._lock_graph)
        }


class SmartTaskQueue:
    """Priority-based task queue with intelligent scheduling."""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._heap: List[PriorityTask] = []
        self._task_map: Dict[str, PriorityTask] = {}
        self._dependency_graph: Dict[str, Set[str]] = defaultdict(set)
        self._completed_tasks: Set[str] = set()
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        
        # Metrics
        self.queue_metrics = {
            "enqueued": 0,
            "dequeued": 0,
            "cancelled": 0,
            "current_size": 0,
            "max_size_reached": 0
        }
    
    async def put(self, task: PriorityTask) -> bool:
        """Add task to queue with priority and dependency handling."""
        async with self._not_empty:
            if len(self._heap) >= self.max_size:
                self.queue_metrics["max_size_reached"] += 1
                return False
            
            # Check dependencies
            unmet_dependencies = task.dependencies - self._completed_tasks
            if unmet_dependencies:
                logger.debug(f"Task {task.task_id} waiting for dependencies: {unmet_dependencies}")
                self._dependency_graph[task.task_id] = unmet_dependencies
                # Don't add to heap yet, will be added when dependencies are met
                self._task_map[task.task_id] = task
                return True
            
            # Add to heap and notify waiters
            heapq.heappush(self._heap, task)
            self._task_map[task.task_id] = task
            self.queue_metrics["enqueued"] += 1
            self.queue_metrics["current_size"] = len(self._heap)
            
            self._not_empty.notify()
            return True
    
    async def get(self, timeout: Optional[float] = None) -> Optional[PriorityTask]:
        """Get highest priority task from queue."""
        async with self._not_empty:
            while not self._heap:
                if timeout is not None:
                    try:
                        await asyncio.wait_for(self._not_empty.wait(), timeout=timeout)
                    except asyncio.TimeoutError:
                        return None
                else:
                    await self._not_empty.wait()
            
            task = heapq.heappop(self._heap)
            self.queue_metrics["dequeued"] += 1
            self.queue_metrics["current_size"] = len(self._heap)
            
            return task
    
    async def complete_task(self, task_id: str):
        """Mark task as completed and check for dependent tasks."""
        async with self._not_empty:
            self._completed_tasks.add(task_id)
            
            # Check for tasks waiting on this dependency
            ready_tasks = []
            for waiting_task_id, dependencies in list(self._dependency_graph.items()):
                dependencies.discard(task_id)
                if not dependencies:  # All dependencies met
                    task = self._task_map.get(waiting_task_id)
                    if task:
                        ready_tasks.append(task)
                        del self._dependency_graph[waiting_task_id]
            
            # Add ready tasks to heap
            for task in ready_tasks:
                heapq.heappush(self._heap, task)
                self.queue_metrics["enqueued"] += 1
                logger.debug(f"Task {task.task_id} dependencies met, added to queue")
            
            if ready_tasks:
                self.queue_metrics["current_size"] = len(self._heap)
                self._not_empty.notify_all()
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task."""
        async with self._not_empty:
            if task_id in self._task_map:
                task = self._task_map[task_id]
                
                # Remove from heap if present
                if task in self._heap:
                    self._heap.remove(task)
                    heapq.heapify(self._heap)
                
                # Remove from dependency graph
                if task_id in self._dependency_graph:
                    del self._dependency_graph[task_id]
                
                del self._task_map[task_id]
                self.queue_metrics["cancelled"] += 1
                self.queue_metrics["current_size"] = len(self._heap)
                return True
            
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics."""
        return {
            "metrics": self.queue_metrics.copy(),
            "heap_size": len(self._heap),
            "pending_dependencies": len(self._dependency_graph),
            "completed_tasks": len(self._completed_tasks)
        }


class EnhancedConcurrencyManager:
    """Advanced concurrency manager with intelligent resource management."""
    
    def __init__(
        self,
        max_workers: int = None,
        max_io_workers: int = None,
        max_cpu_workers: int = None,
        enable_deadlock_detection: bool = True,
        enable_resource_monitoring: bool = True
    ):
        # Configuration
        cpu_count = psutil.cpu_count() or 1
        self.max_workers = max_workers or min(64, cpu_count * 8)
        self.max_io_workers = max_io_workers or min(100, cpu_count * 16)
        self.max_cpu_workers = max_cpu_workers or cpu_count
        
        # Executors
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.max_io_workers,
            thread_name_prefix="bot-io"
        )
        self.process_executor = ProcessPoolExecutor(
            max_workers=self.max_cpu_workers
        )
        
        # Task management
        self.task_queue = SmartTaskQueue()
        self.active_tasks: Dict[str, asyncio.Task] = {}
        self.task_metrics: Dict[str, TaskMetrics] = defaultdict(TaskMetrics)
        
        # Advanced features
        self.deadlock_detector = DeadlockDetector() if enable_deadlock_detection else None
        self.connection_pools: Dict[str, ConnectionPool] = {}
        self.semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Resource monitoring
        self.enable_resource_monitoring = enable_resource_monitoring
        self.resource_monitor_task: Optional[asyncio.Task] = None
        self.resource_limits = {
            "cpu_percent": 85.0,
            "memory_percent": 90.0,
            "open_files": 1000
        }
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.shutdown_event = asyncio.Event()
        
        # Rate limiting
        self.rate_limiters: Dict[str, Dict] = {}
        
    async def initialize(self):
        """Initialize the concurrency manager."""
        # Start background tasks
        await self._start_background_tasks()
        
        # Initialize default HTTP connection pool
        await self.create_connection_pool(
            "http",
            ConnectionPoolType.HTTP,
            min_size=5,
            max_size=50,
            connection_factory=self._create_http_connection,
            health_check_func=self._http_health_check
        )
        
        logger.info("Enhanced concurrency manager initialized")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # Task processor
        task = asyncio.create_task(self._task_processor_loop())
        self.background_tasks.add(task)
        
        # Resource monitor
        if self.enable_resource_monitoring:
            task = asyncio.create_task(self._resource_monitor_loop())
            self.background_tasks.add(task)
        
        # Cleanup task
        task = asyncio.create_task(self._cleanup_loop())
        self.background_tasks.add(task)
    
    async def submit_task(
        self,
        func: Callable,
        *args,
        priority: TaskPriority = TaskPriority.NORMAL,
        timeout: Optional[float] = None,
        retry_count: int = 3,
        dependencies: Set[str] = None,
        task_id: Optional[str] = None,
        callback: Optional[Callable] = None,
        resource_requirements: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> str:
        """Submit a task for execution with advanced options."""
        
        task_id = task_id or f"task_{int(time.time() * 1000000)}"
        dependencies = dependencies or set()
        
        # Check resource requirements
        if resource_requirements and not await self._check_resource_requirements(resource_requirements):
            raise Exception("Resource requirements not met")
        
        # Create priority task
        priority_task = PriorityTask(
            priority=priority,
            task_id=task_id,
            created_at=time.time(),
            func=func,
            args=args,
            kwargs=kwargs,
            timeout=timeout,
            max_retries=retry_count,
            callback=callback,
            dependencies=dependencies
        )
        
        # Add to queue
        if await self.task_queue.put(priority_task):
            logger.debug(f"Task {task_id} queued with priority {priority.name}")
            return task_id
        else:
            raise Exception("Task queue is full")
    
    async def _task_processor_loop(self):
        """Main task processing loop."""
        while not self.shutdown_event.is_set():
            try:
                # Get task from queue
                priority_task = await self.task_queue.get(timeout=1.0)
                if not priority_task:
                    continue
                
                # Check resource limits before processing
                if not await self._check_resource_limits():
                    # Put task back in queue and wait
                    await self.task_queue.put(priority_task)
                    await asyncio.sleep(1.0)
                    continue
                
                # Create and start asyncio task
                task = asyncio.create_task(
                    self._execute_priority_task(priority_task)
                )
                self.active_tasks[priority_task.task_id] = task
                
                # Update metrics
                self.task_metrics["global"].update_active_tasks(len(self.active_tasks))
                
            except Exception as e:
                logger.error(f"Error in task processor loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _execute_priority_task(self, priority_task: PriorityTask):
        """Execute a priority task with full lifecycle management."""
        task_id = priority_task.task_id
        queue_time = time.time() - priority_task.created_at
        start_time = time.time()
        success = False
        
        try:
            # Acquire resources if needed
            resource_id = f"task_resource_{task_id}"
            if (self.deadlock_detector and 
                not await self.deadlock_detector.acquire_resource(task_id, resource_id)):
                raise Exception("Could not acquire resources (potential deadlock)")
            
            try:
                # Execute the function
                if asyncio.iscoroutinefunction(priority_task.func):
                    if priority_task.timeout:
                        result = await asyncio.wait_for(
                            priority_task.func(*priority_task.args, **priority_task.kwargs),
                            timeout=priority_task.timeout
                        )
                    else:
                        result = await priority_task.func(*priority_task.args, **priority_task.kwargs)
                else:
                    # Run in thread pool for synchronous functions
                    if priority_task.timeout:
                        result = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(
                                self.thread_executor,
                                priority_task.func,
                                *priority_task.args
                            ),
                            timeout=priority_task.timeout
                        )
                    else:
                        result = await asyncio.get_event_loop().run_in_executor(
                            self.thread_executor,
                            priority_task.func,
                            *priority_task.args
                        )
                
                success = True
                
                # Call callback if provided
                if priority_task.callback:
                    try:
                        if asyncio.iscoroutinefunction(priority_task.callback):
                            await priority_task.callback(result)
                        else:
                            priority_task.callback(result)
                    except Exception as e:
                        logger.error(f"Callback error for task {task_id}: {e}")
                
                logger.debug(f"Task {task_id} completed successfully")
                
            finally:
                # Release resources
                if self.deadlock_detector:
                    await self.deadlock_detector.release_resource(task_id, resource_id)
                
        except asyncio.TimeoutError:
            logger.warning(f"Task {task_id} timed out")
            
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            
            # Retry if configured
            if priority_task.retry_count < priority_task.max_retries:
                priority_task.retry_count += 1
                logger.info(f"Retrying task {task_id} (attempt {priority_task.retry_count}/{priority_task.max_retries})")
                
                # Add delay before retry
                await asyncio.sleep(min(2 ** priority_task.retry_count, 30))
                
                # Re-queue the task
                await self.task_queue.put(priority_task)
                self.task_metrics["global"].retried_tasks += 1
                return
        
        finally:
            # Clean up
            execution_time = time.time() - start_time
            
            # Update metrics
            self.task_metrics["global"].update(
                execution_time, queue_time, success, priority_task.priority
            )
            
            # Mark task as completed
            await self.task_queue.complete_task(task_id)
            
            # Remove from active tasks
            self.active_tasks.pop(task_id, None)
            self.task_metrics["global"].update_active_tasks(len(self.active_tasks))
    
    async def _check_resource_limits(self) -> bool:
        """Check if system resource limits allow task execution."""
        if not self.enable_resource_monitoring:
            return True
        
        try:
            # Check CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            if cpu_percent > self.resource_limits["cpu_percent"]:
                return False
            
            # Check memory usage
            memory = psutil.virtual_memory()
            if memory.percent > self.resource_limits["memory_percent"]:
                return False
            
            # Check open files
            process = psutil.Process()
            open_files = len(process.open_files())
            if open_files > self.resource_limits["open_files"]:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking resource limits: {e}")
            return True  # Allow execution on error
    
    async def _check_resource_requirements(self, requirements: Dict[str, Any]) -> bool:
        """Check if resource requirements can be met."""
        try:
            # Check memory requirement
            if "memory_mb" in requirements:
                available_mb = psutil.virtual_memory().available / (1024 * 1024)
                if available_mb < requirements["memory_mb"]:
                    return False
            
            # Check CPU requirement
            if "cpu_cores" in requirements:
                if psutil.cpu_count() < requirements["cpu_cores"]:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking resource requirements: {e}")
            return False
    
    async def create_connection_pool(
        self,
        pool_name: str,
        pool_type: ConnectionPoolType,
        min_size: int = 5,
        max_size: int = 50,
        connection_factory: Optional[Callable] = None,
        health_check_func: Optional[Callable] = None,
        **kwargs
    ):
        """Create a named connection pool."""
        pool = ConnectionPool(
            pool_type=pool_type,
            min_size=min_size,
            max_size=max_size,
            connection_factory=connection_factory,
            health_check_func=health_check_func,
            **kwargs
        )
        
        await pool.initialize()
        self.connection_pools[pool_name] = pool
        
        logger.info(f"Created connection pool '{pool_name}' of type {pool_type.value}")
    
    def get_connection_pool(self, pool_name: str) -> Optional[ConnectionPool]:
        """Get a connection pool by name."""
        return self.connection_pools.get(pool_name)
    
    async def _create_http_connection(self):
        """Create HTTP connection for default pool."""
        connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            use_dns_cache=True,
        )
        session = aiohttp.ClientSession(connector=connector)
        return session
    
    async def _http_health_check(self, session) -> bool:
        """Health check for HTTP connections."""
        try:
            # Simple health check - verify session is not closed
            return not session.closed
        except Exception:
            return False
    
    async def _resource_monitor_loop(self):
        """Background resource monitoring."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Monitor system resources
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                
                # Log resource usage
                logger.debug(f"Resource usage - CPU: {cpu_percent:.1f}%, Memory: {memory.percent:.1f}%")
                
                # Adjust resource limits dynamically
                if cpu_percent > 90:
                    self.resource_limits["cpu_percent"] = max(70, self.resource_limits["cpu_percent"] - 5)
                elif cpu_percent < 50:
                    self.resource_limits["cpu_percent"] = min(90, self.resource_limits["cpu_percent"] + 5)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup of completed tasks and resources."""
        while not self.shutdown_event.is_set():
            try:
                await asyncio.sleep(300)  # Clean up every 5 minutes
                
                # Clean up completed tasks
                completed_tasks = []
                for task_id, task in list(self.active_tasks.items()):
                    if task.done():
                        completed_tasks.append(task_id)
                
                for task_id in completed_tasks:
                    del self.active_tasks[task_id]
                
                if completed_tasks:
                    logger.debug(f"Cleaned up {len(completed_tasks)} completed tasks")
                
                # Force garbage collection
                collected = gc.collect()
                if collected > 0:
                    logger.debug(f"Garbage collected {collected} objects")
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
    
    def create_semaphore(self, name: str, limit: int) -> asyncio.Semaphore:
        """Create a named semaphore for resource limiting."""
        semaphore = asyncio.Semaphore(limit)
        self.semaphores[name] = semaphore
        return semaphore
    
    def get_semaphore(self, name: str) -> Optional[asyncio.Semaphore]:
        """Get a semaphore by name."""
        return self.semaphores.get(name)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a task by ID."""
        # Try to cancel from queue first
        if await self.task_queue.cancel_task(task_id):
            return True
        
        # Try to cancel active task
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.cancel()
            return True
        
        return False
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive concurrency statistics."""
        return {
            "task_queue": self.task_queue.get_stats(),
            "task_metrics": {
                name: {
                    "total_tasks": metrics.total_tasks,
                    "completed_tasks": metrics.completed_tasks,
                    "failed_tasks": metrics.failed_tasks,
                    "cancelled_tasks": metrics.cancelled_tasks,
                    "retried_tasks": metrics.retried_tasks,
                    "active_tasks": metrics.active_tasks,
                    "peak_active_tasks": metrics.peak_active_tasks,
                    "avg_execution_time": metrics.avg_execution_time,
                    "avg_queue_time": metrics.avg_queue_time,
                    "priority_metrics": {
                        priority.name: pm for priority, pm in metrics.priority_metrics.items()
                    }
                }
                for name, metrics in self.task_metrics.items()
            },
            "connection_pools": {
                name: pool.get_stats() for name, pool in self.connection_pools.items()
            },
            "deadlock_detector": self.deadlock_detector.get_stats() if self.deadlock_detector else None,
            "system_resources": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "open_files": len(psutil.Process().open_files()) if hasattr(psutil.Process(), 'open_files') else 0
            },
            "executors": {
                "thread_pool_size": self.thread_executor._max_workers,
                "process_pool_size": self.process_executor._max_workers,
                "active_background_tasks": len(self.background_tasks)
            }
        }
    
    async def shutdown(self):
        """Gracefully shutdown the concurrency manager."""
        logger.info("Shutting down concurrency manager...")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for background tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Cancel all active tasks
        for task in list(self.active_tasks.values()):
            task.cancel()
        
        # Wait for active tasks to complete
        if self.active_tasks:
            await asyncio.gather(*self.active_tasks.values(), return_exceptions=True)
        
        # Shutdown connection pools
        for pool in self.connection_pools.values():
            await pool.shutdown()
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Concurrency manager shutdown complete")


# Global concurrency manager instance
concurrency_manager = EnhancedConcurrencyManager()