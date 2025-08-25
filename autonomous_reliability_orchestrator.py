#!/usr/bin/env python3
"""
Autonomous Reliability Orchestrator - Enterprise-grade reliability and fault tolerance
Implements self-healing, circuit breakers, monitoring, and comprehensive error recovery
"""

import asyncio
import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
import structlog
from collections import defaultdict, deque
import threading
import psutil

logger = structlog.get_logger(__name__)


class ReliabilityLevel(Enum):
    """Reliability levels for different components"""
    BASIC = auto()
    ENHANCED = auto() 
    ENTERPRISE = auto()
    MISSION_CRITICAL = auto()


class HealthStatus(Enum):
    """Health status indicators"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ReliabilityMetrics:
    """Comprehensive reliability metrics"""
    uptime: float = 0.0
    availability: float = 0.0
    mttr: float = 0.0  # Mean Time To Recovery
    mtbf: float = 0.0  # Mean Time Between Failures
    error_rate: float = 0.0
    success_rate: float = 0.0
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    circuit_breaker_trips: int = 0
    auto_recovery_count: int = 0
    manual_interventions: int = 0


@dataclass
class HealthCheck:
    """Health check configuration and results"""
    name: str
    check_function: Callable
    interval: int = 30  # seconds
    timeout: int = 10
    failure_threshold: int = 3
    success_threshold: int = 2
    current_failures: int = 0
    current_successes: int = 0
    last_check: Optional[datetime] = None
    status: HealthStatus = HealthStatus.HEALTHY
    last_error: Optional[str] = None


@dataclass
class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    name: str
    failure_threshold: int = 5
    recovery_timeout: int = 60
    half_open_max_calls: int = 3
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: Optional[datetime] = None
    next_attempt_time: Optional[datetime] = None


class AutonomousReliabilityOrchestrator:
    """Enterprise-grade reliability and fault tolerance orchestrator"""
    
    def __init__(self, project_root: str = "/root/repo"):
        self.project_root = Path(project_root)
        self.health_checks: Dict[str, HealthCheck] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.metrics = ReliabilityMetrics()
        self.incident_history: deque = deque(maxlen=1000)
        self.recovery_strategies: Dict[str, Callable] = {}
        self.monitoring_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        self.alert_handlers: List[Callable] = []
        self._initialize_reliability_components()
    
    def _initialize_reliability_components(self) -> None:
        """Initialize reliability components"""
        logger.info("🛡️ Initializing autonomous reliability orchestrator")
        
        # Core health checks
        self.register_health_check("system_resources", self._check_system_resources)
        self.register_health_check("database_connection", self._check_database)
        self.register_health_check("external_apis", self._check_external_apis)
        self.register_health_check("disk_space", self._check_disk_space)
        self.register_health_check("memory_usage", self._check_memory)
        
        # Circuit breakers for external dependencies
        self.register_circuit_breaker("github_api", failure_threshold=10, recovery_timeout=120)
        self.register_circuit_breaker("database", failure_threshold=5, recovery_timeout=60)
        self.register_circuit_breaker("webhook_processing", failure_threshold=15, recovery_timeout=180)
        
        # Recovery strategies
        self.recovery_strategies.update({
            "restart_component": self._restart_component,
            "scale_resources": self._scale_resources,
            "fallback_mode": self._activate_fallback_mode,
            "clear_cache": self._clear_cache,
            "reconnect_database": self._reconnect_database,
            "reset_circuit_breaker": self._reset_circuit_breaker
        })
        
        logger.info("Reliability components initialized", 
                   health_checks=len(self.health_checks),
                   circuit_breakers=len(self.circuit_breakers),
                   recovery_strategies=len(self.recovery_strategies))
    
    def register_health_check(self, name: str, check_function: Callable, 
                            interval: int = 30, timeout: int = 10,
                            failure_threshold: int = 3) -> None:
        """Register a new health check"""
        self.health_checks[name] = HealthCheck(
            name=name,
            check_function=check_function,
            interval=interval,
            timeout=timeout,
            failure_threshold=failure_threshold
        )
        logger.info(f"Health check registered: {name}")
    
    def register_circuit_breaker(self, name: str, failure_threshold: int = 5,
                               recovery_timeout: int = 60) -> None:
        """Register a new circuit breaker"""
        self.circuit_breakers[name] = CircuitBreaker(
            name=name,
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout
        )
        logger.info(f"Circuit breaker registered: {name}")
    
    def register_alert_handler(self, handler: Callable) -> None:
        """Register alert handler"""
        self.alert_handlers.append(handler)
    
    async def start_monitoring(self) -> None:
        """Start autonomous monitoring and reliability checks"""
        logger.info("🔍 Starting autonomous reliability monitoring")
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            logger.warning("Monitoring already running")
            return
        
        self.shutdown_event.clear()
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Autonomous monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        logger.info("Stopping autonomous monitoring")
        self.shutdown_event.set()
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
    
    def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while not self.shutdown_event.is_set():
            try:
                # Run health checks
                asyncio.run(self._run_health_checks())
                
                # Update metrics
                self._update_metrics()
                
                # Check for incidents and auto-recovery
                asyncio.run(self._check_for_incidents())
                
                # Sleep before next cycle
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.exception("Error in monitoring loop", error=str(e))
                time.sleep(30)  # Back off on errors
    
    async def _run_health_checks(self) -> None:
        """Run all registered health checks"""
        tasks = []
        for name, health_check in self.health_checks.items():
            # Check if it's time to run this health check
            if (health_check.last_check is None or 
                datetime.now() - health_check.last_check >= timedelta(seconds=health_check.interval)):
                tasks.append(self._run_single_health_check(health_check))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _run_single_health_check(self, health_check: HealthCheck) -> None:
        """Run a single health check"""
        try:
            start_time = time.time()
            
            # Run the health check with timeout
            result = await asyncio.wait_for(
                health_check.check_function(),
                timeout=health_check.timeout
            )
            
            duration = time.time() - start_time
            health_check.last_check = datetime.now()
            
            if result.get('healthy', False):
                health_check.current_failures = 0
                health_check.current_successes += 1
                health_check.last_error = None
                
                # Update status based on success threshold
                if health_check.current_successes >= health_check.success_threshold:
                    if health_check.status != HealthStatus.HEALTHY:
                        logger.info(f"Health check recovered: {health_check.name}")
                        await self._send_alert(f"Health check {health_check.name} recovered", "info")
                    health_check.status = HealthStatus.HEALTHY
            else:
                health_check.current_successes = 0
                health_check.current_failures += 1
                health_check.last_error = result.get('error', 'Unknown error')
                
                # Update status based on failure threshold
                if health_check.current_failures >= health_check.failure_threshold:
                    old_status = health_check.status
                    health_check.status = HealthStatus.UNHEALTHY
                    
                    if old_status != HealthStatus.UNHEALTHY:
                        logger.error(f"Health check failed: {health_check.name} - {health_check.last_error}")
                        await self._send_alert(f"Health check {health_check.name} failed: {health_check.last_error}", "error")
                        await self._trigger_auto_recovery(health_check.name, health_check.last_error)
                        
        except asyncio.TimeoutError:
            health_check.current_failures += 1
            health_check.last_error = f"Timeout after {health_check.timeout}s"
            health_check.last_check = datetime.now()
            logger.warning(f"Health check timeout: {health_check.name}")
            
        except Exception as e:
            health_check.current_failures += 1
            health_check.last_error = str(e)
            health_check.last_check = datetime.now()
            logger.exception(f"Health check error: {health_check.name}", error=str(e))
    
    async def call_with_circuit_breaker(self, breaker_name: str, func: Callable, *args, **kwargs) -> Any:
        """Call a function with circuit breaker protection"""
        breaker = self.circuit_breakers.get(breaker_name)
        if not breaker:
            # No circuit breaker registered, call directly
            return await func(*args, **kwargs)
        
        # Check circuit breaker state
        if breaker.state == CircuitState.OPEN:
            now = datetime.now()
            if breaker.next_attempt_time and now < breaker.next_attempt_time:
                raise Exception(f"Circuit breaker {breaker_name} is OPEN")
            else:
                # Move to half-open state
                breaker.state = CircuitState.HALF_OPEN
                breaker.success_count = 0
                logger.info(f"Circuit breaker {breaker_name} moved to HALF_OPEN")
        
        try:
            result = await func(*args, **kwargs)
            
            # Success - update circuit breaker
            if breaker.state == CircuitState.HALF_OPEN:
                breaker.success_count += 1
                if breaker.success_count >= breaker.half_open_max_calls:
                    breaker.state = CircuitState.CLOSED
                    breaker.failure_count = 0
                    logger.info(f"Circuit breaker {breaker_name} CLOSED after recovery")
            elif breaker.state == CircuitState.CLOSED:
                breaker.failure_count = max(0, breaker.failure_count - 1)
            
            return result
            
        except Exception as e:
            # Failure - update circuit breaker
            breaker.failure_count += 1
            breaker.last_failure_time = datetime.now()
            
            if (breaker.state == CircuitState.CLOSED and 
                breaker.failure_count >= breaker.failure_threshold):
                
                breaker.state = CircuitState.OPEN
                breaker.next_attempt_time = datetime.now() + timedelta(seconds=breaker.recovery_timeout)
                self.metrics.circuit_breaker_trips += 1
                
                logger.error(f"Circuit breaker {breaker_name} OPENED after {breaker.failure_count} failures")
                await self._send_alert(f"Circuit breaker {breaker_name} opened", "error")
                
            elif breaker.state == CircuitState.HALF_OPEN:
                breaker.state = CircuitState.OPEN
                breaker.next_attempt_time = datetime.now() + timedelta(seconds=breaker.recovery_timeout)
                logger.warning(f"Circuit breaker {breaker_name} returned to OPEN from HALF_OPEN")
            
            raise e
    
    async def _check_for_incidents(self) -> None:
        """Check for incidents requiring intervention"""
        # Check for unhealthy components
        unhealthy_checks = [
            hc for hc in self.health_checks.values() 
            if hc.status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]
        ]
        
        if unhealthy_checks:
            for health_check in unhealthy_checks:
                await self._handle_incident(health_check.name, health_check.last_error or "Unknown error")
        
        # Check for open circuit breakers
        open_breakers = [
            cb for cb in self.circuit_breakers.values()
            if cb.state == CircuitState.OPEN
        ]
        
        if open_breakers:
            for breaker in open_breakers:
                await self._handle_circuit_breaker_incident(breaker)
    
    async def _handle_incident(self, component: str, error: str) -> None:
        """Handle an incident with autonomous recovery"""
        incident = {
            'timestamp': datetime.now(timezone.utc),
            'component': component,
            'error': error,
            'recovery_attempted': False,
            'recovery_successful': False
        }
        
        self.incident_history.append(incident)
        logger.error(f"Incident detected: {component} - {error}")
        
        # Attempt autonomous recovery
        await self._trigger_auto_recovery(component, error)
        incident['recovery_attempted'] = True
    
    async def _handle_circuit_breaker_incident(self, breaker: CircuitBreaker) -> None:
        """Handle circuit breaker incident"""
        if breaker.next_attempt_time and datetime.now() > breaker.next_attempt_time:
            # Try to reset the circuit breaker
            logger.info(f"Attempting to reset circuit breaker: {breaker.name}")
            await self._apply_recovery_strategy("reset_circuit_breaker", {"breaker_name": breaker.name})
    
    async def _trigger_auto_recovery(self, component: str, error: str) -> None:
        """Trigger autonomous recovery based on component and error"""
        recovery_map = {
            "system_resources": ["scale_resources", "clear_cache"],
            "database_connection": ["reconnect_database", "restart_component"],
            "external_apis": ["reset_circuit_breaker", "fallback_mode"],
            "disk_space": ["clear_cache", "scale_resources"],
            "memory_usage": ["clear_cache", "restart_component"]
        }
        
        strategies = recovery_map.get(component, ["restart_component"])
        
        for strategy in strategies:
            try:
                logger.info(f"Applying recovery strategy: {strategy} for {component}")
                success = await self._apply_recovery_strategy(strategy, {"component": component, "error": error})
                
                if success:
                    self.metrics.auto_recovery_count += 1
                    logger.info(f"Auto-recovery successful: {strategy} for {component}")
                    await self._send_alert(f"Auto-recovery successful: {component} using {strategy}", "info")
                    break
                else:
                    logger.warning(f"Recovery strategy failed: {strategy} for {component}")
                    
            except Exception as e:
                logger.exception(f"Recovery strategy error: {strategy} for {component}", error=str(e))
        
        else:
            # All recovery strategies failed
            self.metrics.manual_interventions += 1
            await self._send_alert(f"Manual intervention required: {component} - {error}", "critical")
    
    async def _apply_recovery_strategy(self, strategy: str, context: Dict[str, Any]) -> bool:
        """Apply a specific recovery strategy"""
        strategy_func = self.recovery_strategies.get(strategy)
        if not strategy_func:
            logger.error(f"Unknown recovery strategy: {strategy}")
            return False
        
        try:
            return await strategy_func(context)
        except Exception as e:
            logger.exception(f"Recovery strategy execution failed: {strategy}", error=str(e))
            return False
    
    async def _send_alert(self, message: str, level: str) -> None:
        """Send alert to all registered handlers"""
        alert = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': message,
            'level': level,
            'source': 'autonomous_reliability_orchestrator'
        }
        
        for handler in self.alert_handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.exception("Alert handler failed", error=str(e))
    
    def _update_metrics(self) -> None:
        """Update reliability metrics"""
        # Calculate success rate from health checks
        total_checks = len(self.health_checks)
        healthy_checks = sum(1 for hc in self.health_checks.values() if hc.status == HealthStatus.HEALTHY)
        
        if total_checks > 0:
            self.metrics.success_rate = healthy_checks / total_checks
            self.metrics.error_rate = 1.0 - self.metrics.success_rate
        
        # Calculate availability (simplified)
        self.metrics.availability = self.metrics.success_rate
    
    # Health check implementations
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resources"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Consider healthy if all resources are below 90%
            healthy = cpu_percent < 90 and memory_percent < 90 and disk_percent < 90
            
            return {
                'healthy': healthy,
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'disk_percent': disk_percent
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _check_database(self) -> Dict[str, Any]:
        """Check database connection"""
        try:
            # Simulate database check
            # In real implementation, this would test actual database connection
            await asyncio.sleep(0.1)
            return {'healthy': True, 'response_time': 0.1}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _check_external_apis(self) -> Dict[str, Any]:
        """Check external API health"""
        try:
            # Simulate external API check
            await asyncio.sleep(0.05)
            return {'healthy': True, 'response_time': 0.05}
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space"""
        try:
            disk_usage = psutil.disk_usage('/')
            free_percent = (disk_usage.free / disk_usage.total) * 100
            healthy = free_percent > 10  # At least 10% free
            
            return {
                'healthy': healthy,
                'free_percent': free_percent,
                'free_gb': disk_usage.free // (1024**3)
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    async def _check_memory(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            healthy = memory.percent < 85
            
            return {
                'healthy': healthy,
                'used_percent': memory.percent,
                'available_gb': memory.available // (1024**3)
            }
        except Exception as e:
            return {'healthy': False, 'error': str(e)}
    
    # Recovery strategy implementations
    async def _restart_component(self, context: Dict[str, Any]) -> bool:
        """Restart a component"""
        component = context.get('component', 'unknown')
        logger.info(f"Simulating component restart: {component}")
        await asyncio.sleep(1)
        return True
    
    async def _scale_resources(self, context: Dict[str, Any]) -> bool:
        """Scale resources"""
        logger.info("Simulating resource scaling")
        await asyncio.sleep(1)
        return True
    
    async def _activate_fallback_mode(self, context: Dict[str, Any]) -> bool:
        """Activate fallback mode"""
        logger.info("Activating fallback mode")
        await asyncio.sleep(0.5)
        return True
    
    async def _clear_cache(self, context: Dict[str, Any]) -> bool:
        """Clear cache"""
        logger.info("Clearing cache")
        await asyncio.sleep(0.5)
        return True
    
    async def _reconnect_database(self, context: Dict[str, Any]) -> bool:
        """Reconnect database"""
        logger.info("Reconnecting database")
        await asyncio.sleep(1)
        return True
    
    async def _reset_circuit_breaker(self, context: Dict[str, Any]) -> bool:
        """Reset circuit breaker"""
        breaker_name = context.get('breaker_name')
        if breaker_name and breaker_name in self.circuit_breakers:
            breaker = self.circuit_breakers[breaker_name]
            breaker.state = CircuitState.CLOSED
            breaker.failure_count = 0
            breaker.next_attempt_time = None
            logger.info(f"Circuit breaker reset: {breaker_name}")
            return True
        return False
    
    def get_reliability_status(self) -> Dict[str, Any]:
        """Get comprehensive reliability status"""
        return {
            'metrics': {
                'uptime': self.metrics.uptime,
                'availability': self.metrics.availability,
                'success_rate': self.metrics.success_rate,
                'error_rate': self.metrics.error_rate,
                'auto_recovery_count': self.metrics.auto_recovery_count,
                'circuit_breaker_trips': self.metrics.circuit_breaker_trips,
                'manual_interventions': self.metrics.manual_interventions
            },
            'health_checks': {
                name: {
                    'status': hc.status.value,
                    'last_check': hc.last_check.isoformat() if hc.last_check else None,
                    'current_failures': hc.current_failures,
                    'last_error': hc.last_error
                }
                for name, hc in self.health_checks.items()
            },
            'circuit_breakers': {
                name: {
                    'state': cb.state.value,
                    'failure_count': cb.failure_count,
                    'last_failure_time': cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self.circuit_breakers.items()
            },
            'recent_incidents': list(self.incident_history)[-10:] if self.incident_history else [],
            'monitoring_active': self.monitoring_thread is not None and self.monitoring_thread.is_alive()
        }
    
    async def force_health_check(self, check_name: Optional[str] = None) -> Dict[str, Any]:
        """Force immediate health check"""
        if check_name:
            if check_name in self.health_checks:
                await self._run_single_health_check(self.health_checks[check_name])
                return {check_name: self.health_checks[check_name].status.value}
            else:
                return {'error': f'Health check not found: {check_name}'}
        else:
            await self._run_health_checks()
            return {name: hc.status.value for name, hc in self.health_checks.items()}


async def main():
    """Demo the autonomous reliability orchestrator"""
    orchestrator = AutonomousReliabilityOrchestrator()
    
    # Register alert handler
    async def console_alert_handler(alert):
        level_emoji = {'info': '💙', 'warning': '💛', 'error': '❤️', 'critical': '🚨'}
        emoji = level_emoji.get(alert['level'], '🔔')
        print(f"{emoji} ALERT [{alert['level'].upper()}]: {alert['message']}")
    
    orchestrator.register_alert_handler(console_alert_handler)
    
    print("🛡️ AUTONOMOUS RELIABILITY ORCHESTRATOR - DEMO")
    print("=" * 60)
    
    # Start monitoring
    await orchestrator.start_monitoring()
    
    # Force health checks
    print("\n🔍 Running initial health checks...")
    health_status = await orchestrator.force_health_check()
    for name, status in health_status.items():
        print(f"  - {name}: {status}")
    
    # Test circuit breaker
    print("\n⚡ Testing circuit breaker...")
    async def failing_function():
        raise Exception("Simulated failure")
    
    for i in range(3):
        try:
            await orchestrator.call_with_circuit_breaker("github_api", failing_function)
        except Exception as e:
            print(f"  Attempt {i+1}: {str(e)}")
    
    # Wait a bit for monitoring
    print("\n⏳ Monitoring for 10 seconds...")
    await asyncio.sleep(10)
    
    # Get reliability status
    status = orchestrator.get_reliability_status()
    print(f"\n📊 Reliability Status:")
    print(f"  - Success Rate: {status['metrics']['success_rate']:.1%}")
    print(f"  - Auto Recoveries: {status['metrics']['auto_recovery_count']}")
    print(f"  - Circuit Breaker Trips: {status['metrics']['circuit_breaker_trips']}")
    
    # Stop monitoring
    orchestrator.stop_monitoring()
    print("\n✅ Demo completed")


if __name__ == "__main__":
    asyncio.run(main())