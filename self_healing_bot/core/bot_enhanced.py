"""Enhanced bot with Generation 2 reliability features."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import uuid
import time

from .context import Context
from .playbook import Playbook, PlaybookRegistry, ActionResult
from ..detectors.registry import DetectorRegistry
from ..reliability.circuit_breaker import CircuitBreaker, CircuitBreakerConfig
from ..reliability.retry_handler import RetryHandler, RetryConfig
from ..reliability.health_monitor import HealthMonitor, HealthCheck

logger = logging.getLogger(__name__)


class EnhancedSelfHealingBot:
    """Enhanced self-healing MLOps bot with Generation 2 reliability features."""
    
    def __init__(self):
        self.detector_registry = DetectorRegistry()
        self.playbook_registry = PlaybookRegistry()
        self._active_executions: Dict[str, Context] = {}
        
        # Reliability components
        self.circuit_breakers = self._setup_circuit_breakers()
        self.retry_handlers = self._setup_retry_handlers()
        self.health_monitor = HealthMonitor()
        
        # Metrics and monitoring
        self._execution_metrics = {
            "total_events": 0,
            "successful_events": 0,
            "failed_events": 0,
            "average_processing_time": 0.0,
            "last_error": None,
            "last_error_time": None
        }
        
        self._setup_health_checks()
    
    def _setup_circuit_breakers(self) -> Dict[str, CircuitBreaker]:
        """Setup circuit breakers for critical operations."""
        return {
            "event_processing": CircuitBreaker(
                "event_processing",
                CircuitBreakerConfig(failure_threshold=5, recovery_timeout=120)
            ),
            "detector_execution": CircuitBreaker(
                "detector_execution", 
                CircuitBreakerConfig(failure_threshold=3, recovery_timeout=60)
            ),
            "playbook_execution": CircuitBreaker(
                "playbook_execution",
                CircuitBreakerConfig(failure_threshold=3, recovery_timeout=90)
            )
        }
    
    def _setup_retry_handlers(self) -> Dict[str, RetryHandler]:
        """Setup retry handlers for different operations."""
        return {
            "detector": RetryHandler(RetryConfig(
                max_retries=2,
                base_delay=1.0,
                max_delay=10.0,
                retriable_exceptions=(ConnectionError, TimeoutError)
            )),
            "playbook": RetryHandler(RetryConfig(
                max_retries=1,
                base_delay=2.0,
                max_delay=15.0,
                retriable_exceptions=(ConnectionError, TimeoutError)
            ))
        }
    
    def _setup_health_checks(self):
        """Setup health monitoring checks."""
        # Bot component health check
        async def bot_health_check():
            active_count = len(self._active_executions)
            if active_count > 20:  # Too many concurrent executions
                return {
                    "status": "unhealthy",
                    "message": f"Too many active executions: {active_count}"
                }
            elif active_count > 10:
                return {
                    "status": "degraded", 
                    "message": f"High active executions: {active_count}"
                }
            else:
                return {
                    "status": "healthy",
                    "message": f"Active executions: {active_count}"
                }
        
        # Circuit breaker health check
        def circuit_breaker_health_check():
            unhealthy_breakers = []
            for name, breaker in self.circuit_breakers.items():
                state = breaker.get_state()
                if state["state"] == "open":
                    unhealthy_breakers.append(name)
            
            if unhealthy_breakers:
                return {
                    "status": "unhealthy",
                    "message": f"Open circuit breakers: {unhealthy_breakers}"
                }
            else:
                return {"status": "healthy", "message": "All circuit breakers healthy"}
        
        # Metrics health check
        def metrics_health_check():
            if self._execution_metrics["total_events"] == 0:
                return {"status": "healthy", "message": "No events processed yet"}
            
            success_rate = (self._execution_metrics["successful_events"] / 
                          self._execution_metrics["total_events"])
            
            if success_rate < 0.8:
                return {
                    "status": "unhealthy",
                    "message": f"Low success rate: {success_rate:.2%}"
                }
            elif success_rate < 0.9:
                return {
                    "status": "degraded",
                    "message": f"Moderate success rate: {success_rate:.2%}"
                }
            else:
                return {
                    "status": "healthy", 
                    "message": f"Good success rate: {success_rate:.2%}"
                }
        
        # Register health checks
        self.health_monitor.add_check(
            "bot_execution", bot_health_check, interval=30, critical=True
        )
        self.health_monitor.add_check(
            "circuit_breakers", circuit_breaker_health_check, interval=60, critical=True
        )
        self.health_monitor.add_check(
            "success_metrics", metrics_health_check, interval=120, critical=False
        )
    
    async def process_event(self, event_type: str, event_data: Dict[str, Any]) -> Optional[Context]:
        """Process a GitHub webhook event with full reliability protection."""
        context = None
        execution_start_time = time.time()
        
        # Circuit breaker protection for entire event processing
        try:
            return await self.circuit_breakers["event_processing"].call(
                self._process_event_protected, event_type, event_data, execution_start_time
            )
        except Exception as e:
            # Update metrics on failure
            self._update_execution_metrics(False, execution_start_time, str(e))
            
            # Clean up if context was created
            if context and context.execution_id in self._active_executions:
                del self._active_executions[context.execution_id]
            
            raise
    
    async def _process_event_protected(
        self, event_type: str, event_data: Dict[str, Any], execution_start_time: float
    ) -> Optional[Context]:
        """Internal event processing with timeout and error handling."""
        context = None
        
        try:
            # Create execution context
            context = self._create_context(event_type, event_data)
            
            logger.info(f"Processing {event_type} event for {context.repo_full_name}")
            
            # Store active execution
            self._active_executions[context.execution_id] = context
            
            # Add execution timeout protection (5 minutes)
            async with asyncio.timeout(300):
                # Detect issues with circuit breaker and retry protection
                issues = await self._detect_issues_with_protection(context)
                
                if not issues:
                    logger.info(f"No issues detected for {context.repo_full_name}")
                    self._update_execution_metrics(True, execution_start_time)
                    return context
                
                logger.info(f"Detected {len(issues)} issues: {[i['type'] for i in issues]}")
                
                # Execute repair playbooks with protection
                repair_results = await self._execute_repairs_with_protection(context, issues)
                
                # Log results
                self._log_repair_results(context, repair_results)
                
                # Update success metrics
                self._update_execution_metrics(True, execution_start_time)
                
                return context
                
        except asyncio.TimeoutError:
            logger.error(f"Event processing timed out for {context.repo_full_name if context else 'unknown'}")
            if context:
                context.set_error("ProcessingTimeout", "Event processing exceeded 5 minute timeout")
            self._update_execution_metrics(False, execution_start_time, "timeout")
            raise
            
        except Exception as e:
            logger.exception(f"Error processing {event_type} event: {e}")
            if context:
                context.set_error("ProcessingError", str(e))
            self._update_execution_metrics(False, execution_start_time, str(e))
            raise
        finally:
            # Clean up active execution
            if context and context.execution_id in self._active_executions:
                del self._active_executions[context.execution_id]
    
    async def _detect_issues_with_protection(self, context: Context) -> List[Dict[str, Any]]:
        """Detect issues with circuit breaker and retry protection."""
        return await self.circuit_breakers["detector_execution"].call(
            self._detect_issues_with_retry, context
        )
    
    async def _detect_issues_with_retry(self, context: Context) -> List[Dict[str, Any]]:
        """Detect issues with retry logic."""
        return await self.retry_handlers["detector"].execute_async(
            self._detect_issues, context
        )
    
    async def _detect_issues(self, context: Context) -> List[Dict[str, Any]]:
        """Detect issues using registered detectors."""
        issues = []
        
        # Get available detectors
        detectors = self.detector_registry.get_detectors_for_event(context.event_type)
        
        # Process detectors concurrently with individual timeouts
        detector_tasks = []
        for detector in detectors:
            task = asyncio.create_task(
                asyncio.wait_for(detector.detect(context), timeout=30)
            )
            detector_tasks.append((detector.__class__.__name__, task))
        
        # Gather results with error handling
        for detector_name, task in detector_tasks:
            try:
                detector_issues = await task
                issues.extend(detector_issues)
            except asyncio.TimeoutError:
                logger.warning(f"Detector {detector_name} timed out")
            except Exception as e:
                logger.exception(f"Error running detector {detector_name}: {e}")
        
        return issues
    
    async def _execute_repairs_with_protection(
        self, context: Context, issues: List[Dict[str, Any]]
    ) -> List[ActionResult]:
        """Execute repairs with circuit breaker and retry protection."""
        return await self.circuit_breakers["playbook_execution"].call(
            self._execute_repairs_with_retry, context, issues
        )
    
    async def _execute_repairs_with_retry(
        self, context: Context, issues: List[Dict[str, Any]]
    ) -> List[ActionResult]:
        """Execute repairs with retry logic."""
        return await self.retry_handlers["playbook"].execute_async(
            self._execute_repairs, context, issues
        )
    
    async def _execute_repairs(self, context: Context, issues: List[Dict[str, Any]]) -> List[ActionResult]:
        """Execute repair playbooks for detected issues."""
        all_results = []
        
        # Get available playbooks
        playbook_names = self.playbook_registry.list_playbooks()
        
        # Execute playbooks with individual timeouts
        playbook_tasks = []
        for playbook_name in playbook_names:
            playbook_class = self.playbook_registry.get_playbook(playbook_name)
            if not playbook_class:
                continue
            
            try:
                playbook = playbook_class()
                
                # Check if playbook should trigger
                if playbook.should_trigger(context):
                    logger.info(f"Triggering playbook: {playbook_name}")
                    
                    # Execute with timeout
                    task = asyncio.create_task(
                        asyncio.wait_for(playbook.execute(context), timeout=180)
                    )
                    playbook_tasks.append((playbook_name, task))
                    
            except Exception as e:
                logger.exception(f"Error setting up playbook {playbook_name}: {e}")
                all_results.append(ActionResult(
                    success=False,
                    message=f"Playbook {playbook_name} setup failed: {str(e)}"
                ))
        
        # Gather playbook results
        for playbook_name, task in playbook_tasks:
            try:
                results = await task
                all_results.extend(results)
            except asyncio.TimeoutError:
                logger.error(f"Playbook {playbook_name} timed out")
                all_results.append(ActionResult(
                    success=False,
                    message=f"Playbook {playbook_name} timed out after 3 minutes"
                ))
            except Exception as e:
                logger.exception(f"Error executing playbook {playbook_name}: {e}")
                all_results.append(ActionResult(
                    success=False,
                    message=f"Playbook {playbook_name} failed: {str(e)}"
                ))
        
        return all_results
    
    def _create_context(self, event_type: str, event_data: Dict[str, Any]) -> Context:
        """Create execution context from event data."""
        # Extract repository information
        repo_data = event_data.get("repository", {})
        repo_full_name = repo_data.get("full_name", "unknown/unknown")
        repo_owner, repo_name = repo_full_name.split("/", 1) if "/" in repo_full_name else ("unknown", "unknown")
        
        context = Context(
            repo_owner=repo_owner,
            repo_name=repo_name,
            repo_full_name=repo_full_name,
            event_type=event_type,
            event_data=event_data,
            execution_id=str(uuid.uuid4()),
            started_at=datetime.utcnow()
        )
        
        # Extract error information if present
        if event_type == "workflow_run" and event_data.get("conclusion") == "failure":
            context.set_error(
                "WorkflowFailure",
                f"Workflow {event_data.get('name', 'unknown')} failed"
            )
        
        return context
    
    def _update_execution_metrics(
        self, success: bool, start_time: float, error: Optional[str] = None
    ):
        """Update execution metrics."""
        execution_time = time.time() - start_time
        
        self._execution_metrics["total_events"] += 1
        
        if success:
            self._execution_metrics["successful_events"] += 1
        else:
            self._execution_metrics["failed_events"] += 1
            self._execution_metrics["last_error"] = error
            self._execution_metrics["last_error_time"] = datetime.utcnow().isoformat()
        
        # Update rolling average processing time
        current_avg = self._execution_metrics["average_processing_time"]
        total_events = self._execution_metrics["total_events"]
        self._execution_metrics["average_processing_time"] = (
            (current_avg * (total_events - 1) + execution_time) / total_events
        )
    
    def _log_repair_results(self, context: Context, results: List[ActionResult]) -> None:
        """Log the results of repair actions."""
        successful_actions = sum(1 for r in results if r.success)
        failed_actions = len(results) - successful_actions
        
        logger.info(
            f"Repair completed for {context.repo_full_name}: "
            f"{successful_actions} successful, {failed_actions} failed actions"
        )
        
        for result in results:
            if result.success:
                logger.info(f"✅ {result.message}")
            else:
                logger.error(f"❌ {result.message}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Comprehensive health check with reliability components."""
        health_report = await self.health_monitor.run_all_checks()
        
        # Add circuit breaker states
        circuit_breaker_states = {
            name: breaker.get_state()
            for name, breaker in self.circuit_breakers.items()
        }
        
        # Determine overall status
        overall_status = "healthy"
        if health_report.overall_status.value in ["unhealthy", "degraded"]:
            overall_status = health_report.overall_status.value
        
        # Check for open circuit breakers
        open_breakers = [
            name for name, state in circuit_breaker_states.items()
            if state["state"] == "open"
        ]
        if open_breakers:
            overall_status = "degraded"
        
        return {
            "status": overall_status,
            "timestamp": health_report.timestamp,
            "active_executions": len(self._active_executions),
            "execution_metrics": self._execution_metrics.copy(),
            "circuit_breakers": circuit_breaker_states,
            "health_checks": health_report.checks,
            "health_summary": health_report.summary,
            "components": {
                "detectors": f"loaded: {len(self.detector_registry.list_detectors())}",
                "playbooks": f"loaded: {len(self.playbook_registry.list_playbooks())}",
                "circuit_breakers": f"configured: {len(self.circuit_breakers)}",
                "health_monitor": "active" if self.health_monitor.monitoring else "inactive"
            }
        }
    
    async def start_monitoring(self):
        """Start health monitoring."""
        await self.health_monitor.start_monitoring()
        logger.info("Enhanced bot monitoring started")
    
    async def stop_monitoring(self):
        """Stop health monitoring."""
        await self.health_monitor.stop_monitoring()
        logger.info("Enhanced bot monitoring stopped")
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get status of an active execution."""
        context = self._active_executions.get(execution_id)
        if not context:
            return None
        
        return {
            "execution_id": execution_id,
            "repo": context.repo_full_name,
            "event_type": context.event_type,
            "started_at": context.started_at.isoformat(),
            "has_error": context.has_error(),
            "state": context.state,
            "duration_seconds": (datetime.utcnow() - context.started_at).total_seconds()
        }
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active executions."""
        return [
            self.get_execution_status(execution_id)
            for execution_id in self._active_executions.keys()
        ]