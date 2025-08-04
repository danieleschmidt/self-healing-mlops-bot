"""Main bot orchestration and coordination."""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

from .context import Context
from .playbook import Playbook, PlaybookRegistry, ActionResult
from .config import config
from ..detectors.registry import DetectorRegistry
from ..integrations.github import GitHubIntegration

logger = logging.getLogger(__name__)


class SelfHealingBot:
    """Main orchestrator for the self-healing MLOps bot."""
    
    def __init__(self):
        self.github = GitHubIntegration()
        self.detector_registry = DetectorRegistry()
        self.playbook_registry = PlaybookRegistry()
        self._active_executions: Dict[str, Context] = {}
    
    async def process_event(self, event_type: str, event_data: Dict[str, Any]) -> Optional[Context]:
        """Process a GitHub webhook event."""
        try:
            # Create execution context
            context = self._create_context(event_type, event_data)
            
            logger.info(f"Processing {event_type} event for {context.repo_full_name}")
            
            # Store active execution
            self._active_executions[context.execution_id] = context
            
            # Detect issues
            issues = await self._detect_issues(context)
            
            if not issues:
                logger.info(f"No issues detected for {context.repo_full_name}")
                return context
            
            logger.info(f"Detected {len(issues)} issues: {[i['type'] for i in issues]}")
            
            # Execute repair playbooks
            repair_results = await self._execute_repairs(context, issues)
            
            # Log results
            self._log_repair_results(context, repair_results)
            
            return context
            
        except Exception as e:
            logger.exception(f"Error processing {event_type} event: {e}")
            if context.execution_id in self._active_executions:
                context.set_error("ProcessingError", str(e))
            raise
        finally:
            # Clean up active execution
            if context.execution_id in self._active_executions:
                del self._active_executions[context.execution_id]
    
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
    
    async def _detect_issues(self, context: Context) -> List[Dict[str, Any]]:
        """Detect issues using registered detectors."""
        issues = []
        
        # Get available detectors
        detectors = self.detector_registry.get_detectors_for_event(context.event_type)
        
        for detector in detectors:
            try:
                detector_issues = await detector.detect(context)
                issues.extend(detector_issues)
            except Exception as e:
                logger.exception(f"Error running detector {detector.__class__.__name__}: {e}")
        
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
            "timestamp": datetime.utcnow().isoformat(),
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
            self.get_execution_status(execution_id)
            for execution_id in self._active_executions.keys()
        ]