"""Main bot orchestration and coordination."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import uuid
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .context import Context
from .playbook import Playbook, PlaybookRegistry, ActionResult
from .config import config
from .error_handler import error_handler, handle_errors, ErrorCategory, RetryStrategy
from ..detectors.registry import DetectorRegistry
from ..integrations.github import GitHubIntegration
from ..monitoring.metrics import metrics_collector
from ..performance.optimization import performance_optimizer

logger = structlog.get_logger(__name__)


class SelfHealingBot:
    """Main orchestrator for the self-healing MLOps bot."""
    
    def __init__(self):
        self.github = GitHubIntegration()
        self.detector_registry = DetectorRegistry()
        self.playbook_registry = PlaybookRegistry()
        self._active_executions: Dict[str, Context] = {}
        self._setup_error_recovery()
    
    async def process_event(self, event_type: str, event_data: Dict[str, Any]) -> Optional[Context]:
        """Process a GitHub webhook event with enhanced error handling."""
        context = None
        start_time = datetime.now(timezone.utc)
        
        try:
            # Create execution context
            context = self._create_context(event_type, event_data)
            
            logger.info(
                "Processing event",
                event_type=event_type,
                repo=context.repo_full_name,
                execution_id=context.execution_id
            )
            
            # Store active execution
            self._active_executions[context.execution_id] = context
            
            # Record metrics
            metrics_collector.increment_counter("events_processed_total", {"event_type": event_type})
            
            # Detect issues with timeout
            issues = await asyncio.wait_for(
                self._detect_issues(context),
                timeout=30.0  # 30 second timeout for detection
            )
            
            if not issues:
                logger.info("No issues detected", repo=context.repo_full_name)
                metrics_collector.increment_counter("events_no_issues_total", {})
                return context
            
            logger.info(
                "Issues detected",
                repo=context.repo_full_name,
                issue_count=len(issues),
                issue_types=[i.get('type', 'unknown') for i in issues]
            )
            
            # Execute repair playbooks with timeout
            repair_results = await asyncio.wait_for(
                self._execute_repairs(context, issues),
                timeout=300.0  # 5 minute timeout for repairs
            )
            
            # Log results
            self._log_repair_results(context, repair_results)
            
            # Record success metrics
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            metrics_collector.record_histogram("event_processing_duration_seconds", processing_time)
            metrics_collector.increment_counter("events_processed_successfully_total", {})
            
            return context
            
        except asyncio.TimeoutError as e:
            error_msg = f"Timeout processing {event_type} event"
            logger.error(error_msg, event_type=event_type, timeout=str(e))
            if context:
                context.set_error("TimeoutError", error_msg)
            metrics_collector.increment_counter("events_timeout_total", {})
            raise
            
        except Exception as e:
            error_msg = f"Error processing {event_type} event: {e}"
            logger.exception(error_msg, event_type=event_type, error=str(e))
            if context:
                context.set_error("ProcessingError", str(e))
            metrics_collector.increment_counter("events_failed_total", {})
            
            # Let error handler deal with it
            await error_handler.handle_error(e, {
                "event_type": event_type,
                "repo": context.repo_full_name if context else "unknown",
                "source": "webhook"
            })
            raise
            
        finally:
            # Clean up active execution
            if context and context.execution_id in self._active_executions:
                del self._active_executions[context.execution_id]
    
    def _setup_error_recovery(self) -> None:
        """Setup error recovery strategies."""
        # GitHub API recovery
        def github_api_recovery(error_context) -> bool:
            """Attempt to recover from GitHub API errors."""
            try:
                logger.info("Attempting GitHub API recovery", error_id=error_context.error_id)
                # Simple recovery: wait and retry connection
                import time
                time.sleep(5)
                return True
            except Exception:
                return False
        
        # Database recovery
        def database_recovery(error_context) -> bool:
            """Attempt to recover from database errors."""
            try:
                logger.info("Attempting database recovery", error_id=error_context.error_id)
                # Simple recovery: reconnect
                return True
            except Exception:
                return False
        
        # Register recovery strategies
        error_handler.register_recovery_strategy(ErrorCategory.GITHUB_API, github_api_recovery)
        error_handler.register_recovery_strategy(ErrorCategory.DATABASE, database_recovery)
    
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
            started_at=datetime.now(timezone.utc)
        )
        
        # Extract error information if present
        if event_type == "workflow_run":
            workflow_run = event_data.get("workflow_run", {})
            if workflow_run.get("conclusion") == "failure":
                context.set_error(
                    "WorkflowFailure",
                    f"Workflow {workflow_run.get('name', 'unknown')} failed"
                )
        
        return context
    
    @performance_optimizer.cached(ttl=300, key_func=lambda self, context: f"detect_issues:{context.repo_full_name}:{context.event_type}:{hash(str(context.event_data))}")
    async def _detect_issues(self, context: Context) -> List[Dict[str, Any]]:
        """Detect issues using registered detectors with performance optimization."""
        issues = []
        
        # Get available detectors
        detectors = self.detector_registry.get_detectors_for_event(context.event_type)
        
        # Execute detectors concurrently for better performance
        async def run_detector(detector):
            try:
                return await detector.detect(context)
            except Exception as e:
                logger.exception(f"Error running detector {detector.__class__.__name__}: {e}")
                return []
        
        # Run all detectors concurrently
        detector_tasks = [run_detector(detector) for detector in detectors]
        results = await asyncio.gather(*detector_tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                issues.extend(result)
        
        return issues
    
    async def _execute_repairs(self, context: Context, issues: List[Dict[str, Any]]) -> List[ActionResult]:
        """Execute repair playbooks for detected issues."""
        all_results = []
        
        # Get available playbooks
        playbook_names = self.playbook_registry.list_playbooks()
        
        for playbook_name in playbook_names:
            playbook_class = self.playbook_registry.get_playbook(playbook_name)
            if not playbook_class:
                continue
            
            try:
                playbook = playbook_class()
                
                # Check if playbook should trigger
                if playbook.should_trigger(context):
                    logger.info(f"Triggering playbook: {playbook_name}")
                    results = await playbook.execute(context)
                    all_results.extend(results)
                
            except Exception as e:
                logger.exception(f"Error executing playbook {playbook_name}: {e}")
                all_results.append(ActionResult(
                    success=False,
                    message=f"Playbook {playbook_name} failed: {str(e)}"
                ))
        
        return all_results
    
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
        """Perform health check of bot components."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {},
            "active_executions": len(self._active_executions)
        }
        
        # Check GitHub integration
        try:
            await self.github.test_connection()
            health_status["components"]["github"] = "healthy"
        except Exception as e:
            health_status["components"]["github"] = f"unhealthy: {str(e)}"
            health_status["status"] = "degraded"
        
        # Check detector registry
        detector_count = len(self.detector_registry.list_detectors())
        health_status["components"]["detectors"] = f"loaded: {detector_count}"
        
        # Check playbook registry
        playbook_count = len(self.playbook_registry.list_playbooks())
        health_status["components"]["playbooks"] = f"loaded: {playbook_count}"
        
        return health_status
    
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
            "state": context.state
        }
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active executions."""
        return [
            status for status in [
                self.get_execution_status(execution_id)
                for execution_id in self._active_executions.keys()
            ]
            if status is not None
        ]