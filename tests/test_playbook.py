"""Tests for playbook functionality."""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio

from self_healing_bot.core.playbook import (
    Playbook, Action, ActionResult, PlaybookRegistry,
    TestFailurePlaybook, GPUOOMPlaybook
)
from self_healing_bot.core.context import Context


class TestPlaybook:
    """Test cases for base Playbook functionality."""
    
    def test_action_decorator(self):
        """Test Action decorator functionality."""
        @Action(order=1, timeout=60, retry_count=2)
        def test_action(context):
            return "success"
        
        assert hasattr(test_action, '_is_action')
        assert test_action._is_action is True
        assert test_action._action_order == 1
        assert test_action._action_timeout == 60
        assert test_action._action_retry_count == 2
    
    def test_playbook_registry(self):
        """Test PlaybookRegistry functionality."""
        @PlaybookRegistry.register("test_playbook")
        class TestPlaybook(Playbook):
            def should_trigger(self, context):
                return True
            
            @Action(order=1)
            def test_action(self, context):
                return "test result"
        
        # Test registration
        assert "test_playbook" in PlaybookRegistry.list_playbooks()
        
        # Test retrieval
        playbook_class = PlaybookRegistry.get_playbook("test_playbook")
        assert playbook_class is TestPlaybook
        
        # Test instantiation
        playbook = playbook_class()
        assert isinstance(playbook, TestPlaybook)
    
    def test_discover_actions(self):
        """Test action discovery in playbooks."""
        class TestPlaybook(Playbook):
            def should_trigger(self, context):
                return True
            
            @Action(order=2)
            def second_action(self, context):
                return "second"
            
            @Action(order=1)
            def first_action(self, context):
                return "first"
            
            def not_an_action(self, context):
                return "not an action"
        
        playbook = TestPlaybook()
        
        # Should have 2 actions, sorted by order
        assert len(playbook.actions) == 2
        assert playbook.actions[0].__name__ == "first_action"
        assert playbook.actions[1].__name__ == "second_action"
    
    @pytest.mark.asyncio
    async def test_execute_success(self, mock_context):
        """Test successful playbook execution."""
        class TestPlaybook(Playbook):
            def should_trigger(self, context):
                return True
            
            @Action(order=1)
            def action1(self, context):
                return "action1 result"
            
            @Action(order=2)
            def action2(self, context):
                return ActionResult(success=True, message="action2 result")
        
        playbook = TestPlaybook()
        results = await playbook.execute(mock_context)
        
        assert len(results) == 2
        assert all(result.success for result in results)
        assert results[0].message == "action1 result"
        assert results[1].message == "action2 result"
    
    @pytest.mark.asyncio
    async def test_execute_with_failure(self, mock_context):
        """Test playbook execution with action failure."""
        class TestPlaybook(Playbook):
            def should_trigger(self, context):
                return True
            
            @Action(order=1)
            def action1(self, context):
                raise Exception("Action failed")
            
            @Action(order=2)
            def action2(self, context):
                return "should not execute"
        
        playbook = TestPlaybook()
        results = await playbook.execute(mock_context)
        
        # Should have 1 result (failed action)
        assert len(results) == 1
        assert not results[0].success
        assert "Action failed" in results[0].message
    
    @pytest.mark.asyncio
    async def test_execute_with_timeout(self, mock_context):
        """Test playbook execution with timeout."""
        class TestPlaybook(Playbook):
            def should_trigger(self, context):
                return True
            
            @Action(order=1, timeout=0.1)  # Very short timeout
            async def slow_action(self, context):
                await asyncio.sleep(1)  # Will timeout
                return "completed"
        
        playbook = TestPlaybook()
        results = await playbook.execute(mock_context)
        
        assert len(results) == 1
        assert not results[0].success
        assert "timed out" in results[0].message


class TestBuiltinPlaybooks:
    """Test cases for built-in playbooks."""
    
    def test_test_failure_playbook_trigger(self, mock_context):
        """Test TestFailurePlaybook trigger conditions."""
        playbook = TestFailurePlaybook()
        
        # Should trigger on workflow failure
        mock_context.event_type = "workflow_run"
        mock_context.event_data = {"conclusion": "failure"}
        assert playbook.should_trigger(mock_context)
        
        # Should not trigger on success
        mock_context.event_data = {"conclusion": "success"}
        assert not playbook.should_trigger(mock_context)
        
        # Should not trigger on different event type
        mock_context.event_type = "push"
        assert not playbook.should_trigger(mock_context)
    
    @pytest.mark.asyncio
    async def test_test_failure_playbook_execution(self, mock_context):
        """Test TestFailurePlaybook execution."""
        playbook = TestFailurePlaybook()
        
        # Setup context for workflow failure
        mock_context.event_type = "workflow_run"
        mock_context.event_data = {"conclusion": "failure"}
        
        results = await playbook.execute(mock_context)
        
        # Should have 3 results (analyze_logs, fix_common_errors, create_pr)
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Check that state was set
        assert mock_context.get_state("failure_type") == "import_error"
        assert mock_context.get_state("affected_file") == "src/model.py"
        
        # Check that files were modified
        file_changes = mock_context.get_file_changes()
        assert "src/model.py" in file_changes
    
    def test_gpu_oom_playbook_trigger(self, mock_context):
        """Test GPUOOMPlaybook trigger conditions."""
        playbook = GPUOOMPlaybook()
        
        # Should trigger on CUDA OOM error
        mock_context.set_error("RuntimeError", "CUDA out of memory")
        assert playbook.should_trigger(mock_context)
        
        # Should not trigger on different error
        mock_context.set_error("ImportError", "No module named 'torch'")
        assert not playbook.should_trigger(mock_context)
        
        # Should not trigger without error
        mock_context.clear_error()
        assert not playbook.should_trigger(mock_context)
    
    @pytest.mark.asyncio
    async def test_gpu_oom_playbook_execution(self, mock_context):
        """Test GPUOOMPlaybook execution."""
        playbook = GPUOOMPlaybook()
        
        # Setup context with GPU OOM error
        mock_context.set_error("RuntimeError", "CUDA out of memory")
        
        results = await playbook.execute(mock_context)
        
        # Should have 3 results
        assert len(results) == 3
        assert all(result.success for result in results)
        
        # Check that batch size was reduced
        assert mock_context.get_state("new_batch_size") == 16  # Half of default 32
        
        # Check that files were modified
        file_changes = mock_context.get_file_changes()
        assert "training_config.yaml" in file_changes
        assert "train.py" in file_changes


class TestActionResult:
    """Test cases for ActionResult."""
    
    def test_action_result_creation(self):
        """Test ActionResult creation."""
        result = ActionResult(
            success=True,
            message="Test message",
            data={"key": "value"},
            execution_time=1.5
        )
        
        assert result.success is True
        assert result.message == "Test message"
        assert result.data == {"key": "value"}
        assert result.execution_time == 1.5
    
    def test_action_result_default_data(self):
        """Test ActionResult with default data."""
        result = ActionResult(success=False, message="Error")
        
        assert result.data == {}
        assert result.execution_time == 0.0