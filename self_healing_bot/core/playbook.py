"""Playbook system for defining automated repair sequences."""

from typing import Dict, Any, List, Callable, Optional, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import logging
from functools import wraps

from .context import Context

logger = logging.getLogger(__name__)


@dataclass
class ActionResult:
    """Result of an action execution."""
    success: bool
    message: str
    data: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0


def Action(order: int = 0, timeout: int = 300, retry_count: int = 1):
    """Decorator to mark methods as playbook actions."""
    def decorator(func: Callable) -> Callable:
        func._is_action = True
        func._action_order = order
        func._action_timeout = timeout
        func._action_retry_count = retry_count
        return func
    return decorator


class PlaybookRegistry:
    """Registry for managing playbook classes."""
    
    _playbooks: Dict[str, Type["Playbook"]] = {}
    
    @classmethod
    def register(cls, name: str) -> Callable:
        """Decorator to register a playbook class."""
        def decorator(playbook_class: Type["Playbook"]) -> Type["Playbook"]:
            cls._playbooks[name] = playbook_class
            return playbook_class
        return decorator
    
    @classmethod
    def get_playbook(cls, name: str) -> Optional[Type["Playbook"]]:
        """Get a playbook class by name."""
        return cls._playbooks.get(name)
    
    @classmethod
    def list_playbooks(cls) -> List[str]:
        """List all registered playbook names."""
        return list(cls._playbooks.keys())


class Playbook(ABC):
    """Base class for all playbooks."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.actions = self._discover_actions()
    
    def _discover_actions(self) -> List[Callable]:
        """Discover and sort action methods."""
        actions = []
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_is_action"):
                actions.append(attr)
        
        # Sort by action order
        actions.sort(key=lambda x: getattr(x, "_action_order", 0))
        return actions
    
    @abstractmethod
    def should_trigger(self, context: Context) -> bool:
        """Determine if this playbook should be triggered for the given context."""
        pass
    
    async def execute(self, context: Context) -> List[ActionResult]:
        """Execute all actions in the playbook."""
        results = []
        
        logger.info(f"Executing playbook {self.__class__.__name__} for {context.repo_full_name}")
        
        for action in self.actions:
            try:
                result = await self._execute_action(action, context)
                results.append(result)
                
                # Stop execution if action failed and no retry configured
                if not result.success and getattr(action, "_action_retry_count", 1) <= 1:
                    logger.error(f"Action {action.__name__} failed, stopping playbook execution")
                    break
                    
            except Exception as e:
                logger.exception(f"Error executing action {action.__name__}: {e}")
                results.append(ActionResult(
                    success=False,
                    message=f"Exception in {action.__name__}: {str(e)}"
                ))
                break
        
        return results
    
    async def _execute_action(self, action: Callable, context: Context) -> ActionResult:
        """Execute a single action with timeout and retry logic."""
        timeout = getattr(action, "_action_timeout", 300)
        retry_count = getattr(action, "_action_retry_count", 1)
        
        for attempt in range(retry_count):
            try:
                # Execute with timeout
                result = await asyncio.wait_for(
                    self._run_action(action, context),
                    timeout=timeout
                )
                
                if isinstance(result, str):
                    return ActionResult(success=True, message=result)
                elif isinstance(result, ActionResult):
                    return result
                else:
                    return ActionResult(success=True, message="Action completed successfully")
                    
            except asyncio.TimeoutError:
                logger.warning(f"Action {action.__name__} timed out (attempt {attempt + 1}/{retry_count})")
                if attempt == retry_count - 1:
                    return ActionResult(success=False, message=f"Action timed out after {timeout}s")
            except Exception as e:
                logger.exception(f"Action {action.__name__} failed (attempt {attempt + 1}/{retry_count}): {e}")
                if attempt == retry_count - 1:
                    return ActionResult(success=False, message=str(e))
        
        return ActionResult(success=False, message="All retry attempts failed")
    
    async def _run_action(self, action: Callable, context: Context) -> Any:
        """Run the action method."""
        if asyncio.iscoroutinefunction(action):
            return await action(context)
        else:
            return action(context)


# Register the base Playbook class
Playbook.register = PlaybookRegistry.register


# Example playbook implementations
@Playbook.register("test_failure_handler")
class TestFailurePlaybook(Playbook):
    """Handle test failures in CI/CD pipelines."""
    
    def should_trigger(self, context: Context) -> bool:
        return (
            context.event_type == "workflow_run" and
            context.event_data.get("conclusion") == "failure"
        )
    
    @Action(order=1)
    def analyze_logs(self, context: Context) -> str:
        """Analyze test failure logs."""
        # Mock log analysis
        context.set_state("failure_type", "import_error")
        context.set_state("affected_file", "src/model.py")
        return "Identified import error in src/model.py"
    
    @Action(order=2)
    def fix_common_errors(self, context: Context) -> str:
        """Apply common fixes for test failures."""
        failure_type = context.get_state("failure_type")
        
        if failure_type == "import_error":
            # Mock fix
            context.write_file("src/model.py", "# Fixed import error\nimport numpy as np\n")
            return "Fixed import error"
        
        return "No common fix available"
    
    @Action(order=3)
    def create_pr(self, context: Context) -> str:
        """Create pull request with fixes."""
        pr = context.create_pull_request(
            title="🤖 Fix test failures",
            body="Automated fixes for test failures detected by self-healing bot.",
            branch="fix/test-failures-auto"
        )
        return f"Created PR #{pr.number}"


@Playbook.register("gpu_oom_handler")
class GPUOOMPlaybook(Playbook):
    """Handle GPU out-of-memory errors."""
    
    def should_trigger(self, context: Context) -> bool:
        return (
            context.has_error() and
            "CUDA out of memory" in (context.error_message or "")
        )
    
    @Action(order=1)
    def reduce_batch_size(self, context: Context) -> str:
        """Reduce batch size to fit in GPU memory."""
        config = context.load_config("training_config.yaml")
        
        old_batch_size = config.get("batch_size", 32)
        new_batch_size = max(1, old_batch_size // 2)
        
        config["batch_size"] = new_batch_size
        context.save_config("training_config.yaml", config)
        context.set_state("new_batch_size", new_batch_size)
        
        return f"Reduced batch size from {old_batch_size} to {new_batch_size}"
    
    @Action(order=2)
    def enable_gradient_checkpointing(self, context: Context) -> str:
        """Enable gradient checkpointing for memory efficiency."""
        training_script = context.read_file("train.py")
        
        if "gradient_checkpointing" not in training_script:
            modified = training_script + "\n# Enable gradient checkpointing\nmodel.gradient_checkpointing_enable()\n"
            context.write_file("train.py", modified)
            return "Enabled gradient checkpointing"
        
        return "Gradient checkpointing already enabled"
    
    @Action(order=3)
    def create_fix_pr(self, context: Context) -> str:
        """Create PR with memory optimization fixes."""
        new_batch_size = context.get_state("new_batch_size", "unknown")
        
        pr = context.create_pull_request(
            title="🤖 Fix GPU OOM error",
            body=f"""## Automated Fix for GPU Out-of-Memory Error

The self-healing bot detected a GPU OOM error and applied the following fixes:

1. ✅ Reduced batch size to {new_batch_size}
2. ✅ Enabled gradient checkpointing
3. 📊 Estimated memory reduction: ~40%

### Error Details
```
{context.error_message}
```

---
*This PR was automatically generated by the self-healing MLOps bot*
            """,
            branch="fix/gpu-oom-auto-repair"
        )
        return f"Created PR #{pr.number}"