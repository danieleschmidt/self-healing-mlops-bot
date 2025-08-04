"""Tests for the main bot functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from self_healing_bot.core.bot import SelfHealingBot
from self_healing_bot.core.context import Context
from self_healing_bot.core.playbook import ActionResult


class TestSelfHealingBot:
    """Test cases for SelfHealingBot."""
    
    @pytest.mark.asyncio
    async def test_process_event_success(self, sample_webhook_payload):
        """Test successful event processing."""
        bot = SelfHealingBot()
        
        # Mock dependencies
        with patch.object(bot.detector_registry, 'get_detectors_for_event') as mock_detectors, \
             patch.object(bot.playbook_registry, 'list_playbooks') as mock_playbooks, \
             patch.object(bot.playbook_registry, 'get_playbook') as mock_get_playbook:
            
            # Setup mocks
            mock_detector = AsyncMock()
            mock_detector.detect.return_value = [{
                "type": "test_failure",
                "severity": "high",
                "message": "Test failed",
                "data": {}
            }]
            mock_detectors.return_value = [mock_detector]
            
            mock_playbooks.return_value = ["test_failure_handler"]
            
            mock_playbook_class = Mock()
            mock_playbook = Mock()
            mock_playbook.should_trigger.return_value = True
            mock_playbook.execute.return_value = [
                ActionResult(success=True, message="Fixed test failure")
            ]
            mock_playbook_class.return_value = mock_playbook
            mock_get_playbook.return_value = mock_playbook_class
            
            # Process event
            context = await bot.process_event("workflow_run", sample_webhook_payload)
            
            # Assertions
            assert context is not None
            assert context.repo_full_name == "testowner/testrepo"
            assert context.event_type == "workflow_run"
            assert context.has_error()  # Should have error due to workflow failure
            
            # Verify detector was called
            mock_detector.detect.assert_called_once()
            
            # Verify playbook was executed
            mock_playbook.should_trigger.assert_called_once()
            mock_playbook.execute.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_event_no_issues(self, sample_webhook_payload):
        """Test event processing when no issues are detected."""
        bot = SelfHealingBot()
        
        # Make workflow successful
        sample_webhook_payload["workflow_run"]["conclusion"] = "success"
        
        with patch.object(bot.detector_registry, 'get_detectors_for_event') as mock_detectors:
            mock_detector = AsyncMock()
            mock_detector.detect.return_value = []  # No issues
            mock_detectors.return_value = [mock_detector]
            
            context = await bot.process_event("workflow_run", sample_webhook_payload)
            
            assert context is not None
            assert not context.has_error()
    
    @pytest.mark.asyncio
    async def test_process_event_detector_failure(self, sample_webhook_payload):
        """Test event processing when detector fails."""
        bot = SelfHealingBot()
        
        with patch.object(bot.detector_registry, 'get_detectors_for_event') as mock_detectors:
            mock_detector = AsyncMock()
            mock_detector.detect.side_effect = Exception("Detector failed")
            mock_detectors.return_value = [mock_detector]
            
            context = await bot.process_event("workflow_run", sample_webhook_payload)
            
            assert context is not None
            # Should continue processing despite detector failure
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """Test health check functionality."""
        bot = SelfHealingBot()
        
        with patch.object(bot.github, 'test_connection') as mock_github_test:
            mock_github_test.return_value = True
            
            health_data = await bot.health_check()
            
            assert health_data["status"] == "healthy"
            assert "timestamp" in health_data
            assert "components" in health_data
            assert health_data["active_executions"] == 0
            assert health_data["components"]["github"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_health_check_github_failure(self):
        """Test health check when GitHub is unavailable."""
        bot = SelfHealingBot()
        
        with patch.object(bot.github, 'test_connection') as mock_github_test:
            mock_github_test.side_effect = Exception("GitHub unavailable")
            
            health_data = await bot.health_check()
            
            assert health_data["status"] == "degraded"
            assert "GitHub unavailable" in health_data["components"]["github"]
    
    def test_create_context(self):
        """Test context creation from event data."""
        bot = SelfHealingBot()
        
        event_data = {
            "repository": {
                "full_name": "owner/repo",
                "name": "repo"
            },
            "workflow_run": {
                "conclusion": "failure",
                "name": "CI"
            }
        }
        
        context = bot._create_context("workflow_run", event_data)
        
        assert context.repo_owner == "owner"
        assert context.repo_name == "repo"
        assert context.repo_full_name == "owner/repo"
        assert context.event_type == "workflow_run"
        assert context.has_error()
        assert "CI" in context.error_message
    
    def test_get_execution_status(self):
        """Test getting execution status."""
        bot = SelfHealingBot()
        
        # Create mock context
        context = Context(
            repo_owner="owner",
            repo_name="repo",
            repo_full_name="owner/repo",
            event_type="test",
            event_data={}
        )
        
        # Add to active executions
        bot._active_executions[context.execution_id] = context
        
        status = bot.get_execution_status(context.execution_id)
        
        assert status is not None
        assert status["execution_id"] == context.execution_id
        assert status["repo"] == "owner/repo"
        assert status["event_type"] == "test"
        assert not status["has_error"]
    
    def test_get_execution_status_not_found(self):
        """Test getting status for non-existent execution."""
        bot = SelfHealingBot()
        
        status = bot.get_execution_status("non-existent-id")
        
        assert status is None
    
    def test_list_active_executions(self):
        """Test listing active executions."""
        bot = SelfHealingBot()
        
        # Create multiple mock contexts
        contexts = []
        for i in range(3):
            context = Context(
                repo_owner="owner",
                repo_name=f"repo{i}",
                repo_full_name=f"owner/repo{i}",
                event_type="test",
                event_data={}
            )
            contexts.append(context)
            bot._active_executions[context.execution_id] = context
        
        executions = bot.list_active_executions()
        
        assert len(executions) == 3
        assert all(exec_data is not None for exec_data in executions)
        
        repo_names = [exec_data["repo"] for exec_data in executions]
        assert "owner/repo0" in repo_names
        assert "owner/repo1" in repo_names
        assert "owner/repo2" in repo_names