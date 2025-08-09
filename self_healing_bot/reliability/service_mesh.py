"""Service mesh integration for advanced traffic management, load balancing, and resilience patterns."""

import asyncio
import time
import logging
import random
import hashlib
from typing import Dict, Any, Optional, List, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
from functools import wraps
from contextlib import asynccontextmanager
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    CONSISTENT_HASH = "consistent_hash"
    IP_HASH = "ip_hash"
    HEALTH_BASED = "health_based"


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies for canary deployments."""
    PERCENTAGE = "percentage"
    HEADER_BASED = "header_based"
    USER_BASED = "user_based"
    GEOGRAPHIC = "geographic"
    TIME_BASED = "time_based"


class ServiceHealthStatus(Enum):
    """Service instance health status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    DRAINING = "draining"
    UNKNOWN = "unknown"


@dataclass
class ServiceEndpoint:
    """Service endpoint configuration."""
    id: str
    host: str
    port: int
    weight: int = 100
    health_status: ServiceHealthStatus = ServiceHealthStatus.UNKNOWN
    
    # Connection tracking
    active_connections: int = 0
    max_connections: int = 1000
    
    # Performance metrics
    avg_response_time: float = 0.0
    success_rate: float = 100.0
    total_requests: int = 0
    failed_requests: int = 0
    
    # Health check
    last_health_check: float = 0.0
    health_check_failures: int = 0
    
    # Metadata
    version: str = "v1"
    region: str = "default"
    zone: str = "default"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"{self.host}:{self.port}"
    
    @property
    def url(self) -> str:
        """Get the full URL for this endpoint."""
        return f"http://{self.host}:{self.port}"
    
    @property
    def is_healthy(self) -> bool:
        """Check if endpoint is healthy."""
        return self.health_status == ServiceHealthStatus.HEALTHY
    
    @property
    def is_available(self) -> bool:
        """Check if endpoint is available for traffic."""
        return self.health_status in {
            ServiceHealthStatus.HEALTHY,
            ServiceHealthStatus.DEGRADED
        }


@dataclass
class TrafficSplitRule:
    """Traffic splitting rule for canary deployments."""
    name: str
    strategy: TrafficSplitStrategy
    target_service: str
    percentage: float = 0.0
    
    # Rule conditions
    headers: Dict[str, str] = field(default_factory=dict)
    user_groups: Set[str] = field(default_factory=set)
    regions: Set[str] = field(default_factory=set)
    time_range: Optional[Tuple[str, str]] = None  # (start_time, end_time)
    
    # Metadata
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    enabled: bool = True


@dataclass
class ServiceMeshConfig:
    """Configuration for service mesh behavior."""
    service_name: str
    
    # Load balancing
    load_balancing_strategy: LoadBalancingStrategy = LoadBalancingStrategy.ROUND_ROBIN
    enable_health_based_lb: bool = True
    health_check_interval: float = 30.0
    
    # Circuit breaking
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    
    # Retries
    enable_retries: bool = True
    max_retries: int = 3
    retry_timeout: float = 30.0
    
    # Timeouts
    connect_timeout: float = 5.0
    request_timeout: float = 30.0
    
    # Health checks
    health_check_path: str = "/health"
    health_check_timeout: float = 5.0
    health_check_retries: int = 2
    
    # Traffic management
    enable_traffic_splitting: bool = False
    max_concurrent_requests: int = 1000
    
    # Observability
    enable_metrics: bool = True
    enable_tracing: bool = True
    enable_logging: bool = True


@dataclass
class RequestContext:
    """Context for a service mesh request."""
    request_id: str
    service_name: str
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    
    # User/session info
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    # Geographic info
    region: Optional[str] = None
    zone: Optional[str] = None
    
    # Request metadata
    timestamp: float = field(default_factory=time.time)
    source_service: Optional[str] = None
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    
    # Request properties
    size: int = 0
    priority: int = 0


@dataclass
class ServiceMeshMetrics:
    """Comprehensive service mesh metrics."""
    service_name: str
    
    # Request metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    avg_response_time: float = 0.0
    p50_response_time: float = 0.0
    p95_response_time: float = 0.0
    p99_response_time: float = 0.0
    
    # Load balancing metrics
    requests_by_endpoint: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    
    # Circuit breaker metrics
    circuit_breaker_trips: int = 0
    circuit_breaker_recoveries: int = 0
    
    # Traffic splitting metrics
    canary_requests: int = 0
    canary_success_rate: float = 100.0
    
    # Health check metrics
    health_checks_performed: int = 0
    health_check_failures: int = 0
    
    last_updated: float = 0.0


class ServiceRegistry:
    """Registry for service endpoints and their health status."""
    
    def __init__(self):
        self.services: Dict[str, List[ServiceEndpoint]] = defaultdict(list)
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self.health_check_running = False
    
    def register_endpoint(self, service_name: str, endpoint: ServiceEndpoint):
        """Register a service endpoint."""
        # Remove existing endpoint with same ID
        self.services[service_name] = [
            ep for ep in self.services[service_name]
            if ep.id != endpoint.id
        ]
        
        # Add new endpoint
        self.services[service_name].append(endpoint)
        
        logger.info(
            f"Registered endpoint {endpoint.id} for service {service_name}"
        )
        
        # Start health checks if not already running
        if not self.health_check_running:
            asyncio.create_task(self._start_health_checks())
    
    def deregister_endpoint(self, service_name: str, endpoint_id: str):
        """Deregister a service endpoint."""
        self.services[service_name] = [
            ep for ep in self.services[service_name]
            if ep.id != endpoint_id
        ]
        
        logger.info(
            f"Deregistered endpoint {endpoint_id} from service {service_name}"
        )
    
    def get_endpoints(self, service_name: str, healthy_only: bool = False) -> List[ServiceEndpoint]:
        """Get endpoints for a service."""
        endpoints = self.services.get(service_name, [])
        
        if healthy_only:
            endpoints = [ep for ep in endpoints if ep.is_available]
        
        return endpoints
    
    def get_endpoint_by_id(self, service_name: str, endpoint_id: str) -> Optional[ServiceEndpoint]:
        """Get a specific endpoint by ID."""
        for endpoint in self.services.get(service_name, []):
            if endpoint.id == endpoint_id:
                return endpoint
        return None
    
    def update_endpoint_health(self, service_name: str, endpoint_id: str, status: ServiceHealthStatus):
        """Update endpoint health status."""
        endpoint = self.get_endpoint_by_id(service_name, endpoint_id)
        if endpoint:
            old_status = endpoint.health_status
            endpoint.health_status = status
            endpoint.last_health_check = time.time()
            
            if old_status != status:
                logger.info(
                    f"Endpoint {endpoint_id} health changed: {old_status.value} -> {status.value}"
                )
    
    async def _start_health_checks(self):
        """Start health check monitoring."""
        self.health_check_running = True
        
        try:
            while self.health_check_running:
                await self._perform_health_checks()
                await asyncio.sleep(30)  # Check every 30 seconds
        except Exception as e:
            logger.error(f"Health check monitor error: {e}")
        finally:
            self.health_check_running = False
    
    async def _perform_health_checks(self):
        """Perform health checks on all registered endpoints."""
        for service_name, endpoints in self.services.items():
            for endpoint in endpoints:
                try:
                    is_healthy = await self._check_endpoint_health(endpoint)
                    
                    if is_healthy:
                        endpoint.health_check_failures = 0
                        if endpoint.health_status != ServiceHealthStatus.HEALTHY:
                            self.update_endpoint_health(
                                service_name, endpoint.id, ServiceHealthStatus.HEALTHY
                            )
                    else:
                        endpoint.health_check_failures += 1
                        
                        if endpoint.health_check_failures >= 3:
                            self.update_endpoint_health(
                                service_name, endpoint.id, ServiceHealthStatus.UNHEALTHY
                            )
                        elif endpoint.health_check_failures >= 1:
                            self.update_endpoint_health(
                                service_name, endpoint.id, ServiceHealthStatus.DEGRADED
                            )
                
                except Exception as e:
                    logger.warning(f"Health check failed for {endpoint.id}: {e}")
                    endpoint.health_check_failures += 1
    
    async def _check_endpoint_health(self, endpoint: ServiceEndpoint) -> bool:
        """Check if an endpoint is healthy."""
        # This is a basic implementation - in practice, you'd make HTTP requests
        # to the health check endpoint
        
        try:
            # Simulate health check - in reality, would be an HTTP request
            await asyncio.sleep(0.1)  # Simulate network delay
            
            # For demonstration, assume endpoint is healthy if not overloaded
            is_healthy = (
                endpoint.active_connections < endpoint.max_connections * 0.9 and
                endpoint.success_rate > 50.0
            )
            
            return is_healthy
            
        except Exception:
            return False


class LoadBalancer:
    """Load balancer with multiple strategies."""
    
    def __init__(self, service_registry: ServiceRegistry):
        self.service_registry = service_registry
        self.round_robin_counters: Dict[str, int] = defaultdict(int)
        self.consistent_hash_ring: Dict[str, List[Tuple[int, str]]] = {}
    
    def select_endpoint(
        self,
        service_name: str,
        strategy: LoadBalancingStrategy,
        context: Optional[RequestContext] = None
    ) -> Optional[ServiceEndpoint]:
        """Select an endpoint based on load balancing strategy."""
        endpoints = self.service_registry.get_endpoints(service_name, healthy_only=True)
        
        if not endpoints:
            logger.warning(f"No healthy endpoints available for service {service_name}")
            return None
        
        if strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin(service_name, endpoints)
        elif strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin(service_name, endpoints)
        elif strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections(endpoints)
        elif strategy == LoadBalancingStrategy.WEIGHTED_LEAST_CONNECTIONS:
            return self._weighted_least_connections(endpoints)
        elif strategy == LoadBalancingStrategy.RANDOM:
            return random.choice(endpoints)
        elif strategy == LoadBalancingStrategy.WEIGHTED_RANDOM:
            return self._weighted_random(endpoints)
        elif strategy == LoadBalancingStrategy.CONSISTENT_HASH:
            return self._consistent_hash(service_name, endpoints, context)
        elif strategy == LoadBalancingStrategy.IP_HASH:
            return self._ip_hash(endpoints, context)
        elif strategy == LoadBalancingStrategy.HEALTH_BASED:
            return self._health_based(endpoints)
        else:
            return self._round_robin(service_name, endpoints)
    
    def _round_robin(self, service_name: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Round-robin load balancing."""
        counter = self.round_robin_counters[service_name]
        selected = endpoints[counter % len(endpoints)]
        self.round_robin_counters[service_name] = (counter + 1) % len(endpoints)
        return selected
    
    def _weighted_round_robin(self, service_name: str, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted round-robin load balancing."""
        # Create weighted list
        weighted_endpoints = []
        for endpoint in endpoints:
            weighted_endpoints.extend([endpoint] * max(1, endpoint.weight // 10))
        
        if not weighted_endpoints:
            return endpoints[0]
        
        return self._round_robin(f"{service_name}_weighted", weighted_endpoints)
    
    def _least_connections(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Least connections load balancing."""
        return min(endpoints, key=lambda ep: ep.active_connections)
    
    def _weighted_least_connections(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted least connections load balancing."""
        def score(ep: ServiceEndpoint) -> float:
            return ep.active_connections / max(1, ep.weight)
        
        return min(endpoints, key=score)
    
    def _weighted_random(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Weighted random load balancing."""
        total_weight = sum(max(1, ep.weight) for ep in endpoints)
        if total_weight == 0:
            return random.choice(endpoints)
        
        rand_val = random.randint(1, total_weight)
        current_weight = 0
        
        for endpoint in endpoints:
            current_weight += max(1, endpoint.weight)
            if rand_val <= current_weight:
                return endpoint
        
        return endpoints[-1]  # Fallback
    
    def _consistent_hash(
        self,
        service_name: str,
        endpoints: List[ServiceEndpoint],
        context: Optional[RequestContext]
    ) -> ServiceEndpoint:
        """Consistent hash load balancing."""
        if not context or not context.user_id:
            return random.choice(endpoints)
        
        # Build hash ring if not exists or endpoints changed
        ring_key = f"{service_name}_{len(endpoints)}"
        if ring_key not in self.consistent_hash_ring:
            self.consistent_hash_ring[ring_key] = self._build_hash_ring(endpoints)
        
        # Hash the user ID
        user_hash = int(hashlib.md5(context.user_id.encode()).hexdigest(), 16)
        
        # Find the appropriate endpoint on the ring
        ring = self.consistent_hash_ring[ring_key]
        for ring_hash, endpoint_id in ring:
            if user_hash <= ring_hash:
                # Find the endpoint by ID
                for endpoint in endpoints:
                    if endpoint.id == endpoint_id:
                        return endpoint
        
        # If we didn't find one (shouldn't happen), return first
        return endpoints[0]
    
    def _build_hash_ring(self, endpoints: List[ServiceEndpoint]) -> List[Tuple[int, str]]:
        """Build consistent hash ring."""
        ring = []
        
        for endpoint in endpoints:
            # Create multiple virtual nodes for better distribution
            for i in range(100):  # 100 virtual nodes per endpoint
                virtual_key = f"{endpoint.id}_{i}"
                hash_val = int(hashlib.md5(virtual_key.encode()).hexdigest(), 16)
                ring.append((hash_val, endpoint.id))
        
        # Sort by hash value
        ring.sort(key=lambda x: x[0])
        return ring
    
    def _ip_hash(self, endpoints: List[ServiceEndpoint], context: Optional[RequestContext]) -> ServiceEndpoint:
        """IP hash load balancing."""
        if not context or not context.headers.get('X-Forwarded-For'):
            return random.choice(endpoints)
        
        ip = context.headers['X-Forwarded-For'].split(',')[0].strip()
        ip_hash = int(hashlib.md5(ip.encode()).hexdigest(), 16)
        
        return endpoints[ip_hash % len(endpoints)]
    
    def _health_based(self, endpoints: List[ServiceEndpoint]) -> ServiceEndpoint:
        """Health-based load balancing (best performing endpoint)."""
        def score(ep: ServiceEndpoint) -> float:
            # Lower is better: combine response time and failure rate
            response_time_score = ep.avg_response_time
            failure_rate = 100.0 - ep.success_rate
            connection_load = ep.active_connections / max(1, ep.max_connections)
            
            return response_time_score + failure_rate + (connection_load * 1000)
        
        return min(endpoints, key=score)


class TrafficSplitter:
    """Traffic splitter for canary deployments and A/B testing."""
    
    def __init__(self):
        self.rules: Dict[str, List[TrafficSplitRule]] = defaultdict(list)
    
    def add_rule(self, service_name: str, rule: TrafficSplitRule):
        """Add a traffic split rule."""
        self.rules[service_name].append(rule)
        logger.info(f"Added traffic split rule '{rule.name}' for service {service_name}")
    
    def remove_rule(self, service_name: str, rule_name: str):
        """Remove a traffic split rule."""
        self.rules[service_name] = [
            rule for rule in self.rules[service_name]
            if rule.name != rule_name
        ]
        logger.info(f"Removed traffic split rule '{rule_name}' for service {service_name}")
    
    def should_split_traffic(self, service_name: str, context: RequestContext) -> Optional[str]:
        """Check if traffic should be split and return target service."""
        rules = self.rules.get(service_name, [])
        current_time = time.time()
        
        for rule in rules:
            if not rule.enabled:
                continue
            
            # Check expiration
            if rule.expires_at and current_time > rule.expires_at:
                continue
            
            if self._rule_matches(rule, context):
                return rule.target_service
        
        return None
    
    def _rule_matches(self, rule: TrafficSplitRule, context: RequestContext) -> bool:
        """Check if a traffic split rule matches the request context."""
        if rule.strategy == TrafficSplitStrategy.PERCENTAGE:
            return random.random() < (rule.percentage / 100.0)
        
        elif rule.strategy == TrafficSplitStrategy.HEADER_BASED:
            for header, value in rule.headers.items():
                if context.headers.get(header) != value:
                    return False
            return True
        
        elif rule.strategy == TrafficSplitStrategy.USER_BASED:
            return (context.user_id and 
                   any(group in rule.user_groups for group in context.headers.get('user-groups', '').split(',')))
        
        elif rule.strategy == TrafficSplitStrategy.GEOGRAPHIC:
            return context.region in rule.regions
        
        elif rule.strategy == TrafficSplitStrategy.TIME_BASED:
            if rule.time_range:
                current_time = time.strftime('%H:%M')
                start_time, end_time = rule.time_range
                return start_time <= current_time <= end_time
        
        return False


class ServiceMesh:
    """Main service mesh coordinator."""
    
    def __init__(self, config: ServiceMeshConfig):
        self.config = config
        self.service_registry = ServiceRegistry()
        self.load_balancer = LoadBalancer(self.service_registry)
        self.traffic_splitter = TrafficSplitter()
        self.metrics = ServiceMeshMetrics(service_name=config.service_name)
        
        # Request tracking
        self.active_requests: Dict[str, RequestContext] = {}
        self.response_times = deque(maxlen=1000)
        
        # Circuit breaker state (simplified)
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = 0.0
        self.circuit_breaker_open = False
    
    def register_endpoint(self, endpoint: ServiceEndpoint):
        """Register a service endpoint."""
        self.service_registry.register_endpoint(self.config.service_name, endpoint)
    
    def deregister_endpoint(self, endpoint_id: str):
        """Deregister a service endpoint."""
        self.service_registry.deregister_endpoint(self.config.service_name, endpoint_id)
    
    async def make_request(
        self,
        method: str,
        path: str,
        context: Optional[RequestContext] = None,
        **kwargs
    ) -> Any:
        """Make a request through the service mesh."""
        if not context:
            context = RequestContext(
                request_id=f"req_{int(time.time() * 1000)}",
                service_name=self.config.service_name,
                method=method,
                path=path
            )
        
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            raise CircuitBreakerOpenError("Service circuit breaker is open")
        
        # Check for traffic splitting
        target_service = self.traffic_splitter.should_split_traffic(
            self.config.service_name, context
        )
        
        service_name = target_service or self.config.service_name
        
        # Select endpoint
        endpoint = self.load_balancer.select_endpoint(
            service_name,
            self.config.load_balancing_strategy,
            context
        )
        
        if not endpoint:
            raise ServiceUnavailableError(f"No healthy endpoints for service {service_name}")
        
        # Execute request with retries
        return await self._execute_request_with_retries(endpoint, context, **kwargs)
    
    async def _execute_request_with_retries(
        self,
        endpoint: ServiceEndpoint,
        context: RequestContext,
        **kwargs
    ) -> Any:
        """Execute request with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_retries + 1):
            try:
                start_time = time.time()
                
                # Track active request
                endpoint.active_connections += 1
                self.active_requests[context.request_id] = context
                
                # Simulate request execution (in real implementation, this would be HTTP client)
                result = await self._execute_request(endpoint, context, **kwargs)
                
                # Record success
                execution_time = time.time() - start_time
                self._record_success(endpoint, execution_time)
                
                return result
                
            except Exception as e:
                last_exception = e
                execution_time = time.time() - start_time
                
                # Record failure
                self._record_failure(endpoint, execution_time, e)
                
                # Don't retry on last attempt
                if attempt < self.config.max_retries:
                    delay = min(2 ** attempt, 10)  # Exponential backoff, max 10s
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.config.max_retries + 1}), "
                        f"retrying in {delay}s: {e}"
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"All retry attempts failed for {endpoint.id}: {e}")
            
            finally:
                # Clean up tracking
                endpoint.active_connections = max(0, endpoint.active_connections - 1)
                self.active_requests.pop(context.request_id, None)
        
        raise RequestFailedException(
            f"Request failed after {self.config.max_retries + 1} attempts"
        ) from last_exception
    
    async def _execute_request(self, endpoint: ServiceEndpoint, context: RequestContext, **kwargs) -> Any:
        """Execute the actual request (simulated)."""
        # This is a simulation - in real implementation, this would use an HTTP client
        
        # Simulate request time based on endpoint performance
        base_time = 0.1 + (endpoint.avg_response_time / 1000.0)
        jitter = random.uniform(0.8, 1.2)
        request_time = base_time * jitter
        
        # Simulate timeout
        if request_time > self.config.request_timeout:
            raise asyncio.TimeoutError("Request timed out")
        
        await asyncio.sleep(request_time)
        
        # Simulate occasional failures based on endpoint health
        failure_probability = (100.0 - endpoint.success_rate) / 100.0
        if random.random() < failure_probability:
            raise ConnectionError("Simulated request failure")
        
        return {
            "status": "success",
            "endpoint": endpoint.id,
            "response_time": request_time * 1000,  # Convert to ms
            "timestamp": time.time()
        }
    
    def _record_success(self, endpoint: ServiceEndpoint, execution_time: float):
        """Record successful request."""
        endpoint.total_requests += 1
        
        # Update endpoint metrics
        if endpoint.total_requests == 1:
            endpoint.avg_response_time = execution_time * 1000
        else:
            # Exponential moving average
            alpha = 0.1
            endpoint.avg_response_time = (
                (1 - alpha) * endpoint.avg_response_time + 
                alpha * (execution_time * 1000)
            )
        
        # Update success rate
        endpoint.success_rate = (
            (endpoint.total_requests - endpoint.failed_requests) / 
            endpoint.total_requests * 100
        )
        
        # Update global metrics
        self.metrics.total_requests += 1
        self.metrics.successful_requests += 1
        self.response_times.append(execution_time)
        
        # Reset circuit breaker failures on success
        self.circuit_breaker_failures = 0
    
    def _record_failure(self, endpoint: ServiceEndpoint, execution_time: float, error: Exception):
        """Record failed request."""
        endpoint.total_requests += 1
        endpoint.failed_requests += 1
        
        # Update success rate
        endpoint.success_rate = (
            (endpoint.total_requests - endpoint.failed_requests) / 
            endpoint.total_requests * 100
        )
        
        # Update global metrics
        self.metrics.total_requests += 1
        self.metrics.failed_requests += 1
        
        # Update circuit breaker state
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        
        # Open circuit breaker if threshold exceeded
        if (self.config.enable_circuit_breaker and
            self.circuit_breaker_failures >= self.config.circuit_breaker_threshold):
            self.circuit_breaker_open = True
            logger.warning(
                f"Circuit breaker opened for service {self.config.service_name} "
                f"after {self.circuit_breaker_failures} failures"
            )
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        if not self.config.enable_circuit_breaker:
            return False
        
        if not self.circuit_breaker_open:
            return False
        
        # Check if circuit breaker should be closed (half-open)
        current_time = time.time()
        if (current_time - self.circuit_breaker_last_failure) > self.config.circuit_breaker_timeout:
            self.circuit_breaker_open = False
            self.circuit_breaker_failures = 0
            logger.info(f"Circuit breaker closed for service {self.config.service_name}")
            return False
        
        return True
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get comprehensive service status."""
        endpoints = self.service_registry.get_endpoints(self.config.service_name)
        healthy_endpoints = [ep for ep in endpoints if ep.is_healthy]
        
        # Calculate percentile response times
        if self.response_times:
            sorted_times = sorted(self.response_times)
            length = len(sorted_times)
            
            self.metrics.p50_response_time = sorted_times[int(length * 0.5)] * 1000
            self.metrics.p95_response_time = sorted_times[int(length * 0.95)] * 1000
            self.metrics.p99_response_time = sorted_times[int(length * 0.99)] * 1000
            
            self.metrics.avg_response_time = sum(sorted_times) / length * 1000
        
        # Update success rate
        if self.metrics.total_requests > 0:
            success_rate = (
                self.metrics.successful_requests / self.metrics.total_requests * 100
            )
        else:
            success_rate = 100.0
        
        return {
            "service_name": self.config.service_name,
            "total_endpoints": len(endpoints),
            "healthy_endpoints": len(healthy_endpoints),
            "circuit_breaker_open": self.circuit_breaker_open,
            "active_requests": len(self.active_requests),
            "load_balancing_strategy": self.config.load_balancing_strategy.value,
            "endpoints": [
                {
                    "id": ep.id,
                    "status": ep.health_status.value,
                    "active_connections": ep.active_connections,
                    "success_rate": ep.success_rate,
                    "avg_response_time": ep.avg_response_time
                }
                for ep in endpoints
            ],
            "metrics": {
                "total_requests": self.metrics.total_requests,
                "successful_requests": self.metrics.successful_requests,
                "failed_requests": self.metrics.failed_requests,
                "success_rate": success_rate,
                "avg_response_time": self.metrics.avg_response_time,
                "p50_response_time": self.metrics.p50_response_time,
                "p95_response_time": self.metrics.p95_response_time,
                "p99_response_time": self.metrics.p99_response_time
            },
            "traffic_split_rules": len(self.traffic_splitter.rules.get(self.config.service_name, []))
        }


# Custom exceptions
class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


class ServiceUnavailableError(Exception):
    """Exception raised when no healthy endpoints are available."""
    pass


class RequestFailedException(Exception):
    """Exception raised when request fails after all retries."""
    pass


# Global service mesh manager
class ServiceMeshManager:
    """Manager for multiple service meshes."""
    
    def __init__(self):
        self.meshes: Dict[str, ServiceMesh] = {}
    
    def create_mesh(self, config: ServiceMeshConfig) -> ServiceMesh:
        """Create a new service mesh."""
        mesh = ServiceMesh(config)
        self.meshes[config.service_name] = mesh
        
        logger.info(f"Created service mesh for {config.service_name}")
        return mesh
    
    def get_mesh(self, service_name: str) -> Optional[ServiceMesh]:
        """Get service mesh by name."""
        return self.meshes.get(service_name)
    
    def get_all_status(self) -> Dict[str, Any]:
        """Get status for all service meshes."""
        return {
            "total_services": len(self.meshes),
            "services": {
                name: mesh.get_service_status()
                for name, mesh in self.meshes.items()
            }
        }


# Global instance
service_mesh_manager = ServiceMeshManager()


# Decorator for service mesh integration
def service_mesh_integration(
    service_name: str,
    config: Optional[ServiceMeshConfig] = None,
    endpoints: Optional[List[ServiceEndpoint]] = None
):
    """Decorator to integrate function with service mesh."""
    def decorator(func: Callable) -> Callable:
        # Create or get mesh
        mesh_config = config or ServiceMeshConfig(service_name=service_name)
        mesh = service_mesh_manager.create_mesh(mesh_config)
        
        # Register endpoints if provided
        if endpoints:
            for endpoint in endpoints:
                mesh.register_endpoint(endpoint)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Create request context
            context = RequestContext(
                request_id=f"req_{int(time.time() * 1000)}",
                service_name=service_name,
                method="CALL",
                path=func.__name__
            )
            
            # Execute through service mesh
            return await mesh.make_request("CALL", func.__name__, context, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            if asyncio.iscoroutinefunction(func):
                return asyncio.run(async_wrapper(*args, **kwargs))
            else:
                # For sync functions, execute directly but still track metrics
                try:
                    result = func(*args, **kwargs)
                    # Update metrics
                    mesh.metrics.total_requests += 1
                    mesh.metrics.successful_requests += 1
                    return result
                except Exception as e:
                    mesh.metrics.total_requests += 1
                    mesh.metrics.failed_requests += 1
                    raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


# Predefined configurations
GITHUB_API_MESH_CONFIG = ServiceMeshConfig(
    service_name="github_api",
    load_balancing_strategy=LoadBalancingStrategy.HEALTH_BASED,
    enable_circuit_breaker=True,
    circuit_breaker_threshold=5,
    max_retries=3,
    request_timeout=10.0
)

DATABASE_MESH_CONFIG = ServiceMeshConfig(
    service_name="database",
    load_balancing_strategy=LoadBalancingStrategy.LEAST_CONNECTIONS,
    enable_circuit_breaker=True,
    circuit_breaker_threshold=3,
    max_retries=2,
    request_timeout=5.0
)

ML_INFERENCE_MESH_CONFIG = ServiceMeshConfig(
    service_name="ml_inference",
    load_balancing_strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN,
    enable_circuit_breaker=True,
    circuit_breaker_threshold=10,
    max_retries=2,
    request_timeout=30.0,
    enable_traffic_splitting=True
)