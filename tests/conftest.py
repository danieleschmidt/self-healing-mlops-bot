"""Pytest configuration and fixtures."""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock
from typing import Dict, Any

from self_healing_bot.core.context import Context
from self_healing_bot.core.bot import SelfHealingBot
from self_healing_bot.core.config import BotConfig


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock(spec=BotConfig)
    config.github_app_id = "12345"
    config.github_private_key = "mock-private-key"
    config.github_webhook_secret = "mock-webhook-secret"
    config.database_url = "postgresql://test:test@localhost:5432/test"
    config.redis_url = "redis://localhost:6379/0"
    config.log_level = "INFO"
    config.environment = "test"
    config.debug = True
    return config


@pytest.fixture
def mock_context():
    """Mock execution context for testing."""
    return Context(
        repo_owner="testowner",
        repo_name="testrepo",
        repo_full_name="testowner/testrepo",
        event_type="workflow_run",
        event_data={
            "workflow_run": {
                "id": 123456,
                "name": "CI",
                "conclusion": "failure",
                "html_url": "https://github.com/testowner/testrepo/actions/runs/123456"
            },
            "repository": {
                "full_name": "testowner/testrepo",
                "name": "testrepo",
                "owner": {"login": "testowner"}
            }
        }
    )


@pytest.fixture
def mock_github_integration():
    """Mock GitHub integration for testing."""
    mock = AsyncMock()
    mock.test_connection.return_value = True
    mock.get_installation_token.return_value = "mock-token"
    mock.create_pull_request.return_value = {
        "number": 123,
        "url": "https://github.com/testowner/testrepo/pull/123",
        "title": "Test PR",
        "state": "open"
    }
    return mock


@pytest.fixture
def mock_bot(mock_github_integration):
    """Mock bot instance for testing."""
    bot = Mock(spec=SelfHealingBot)
    bot.github = mock_github_integration
    bot.process_event = AsyncMock()
    bot.health_check = AsyncMock(return_value={
        "status": "healthy",
        "timestamp": "2025-01-01T00:00:00Z",
        "components": {
            "github": "healthy",
            "detectors": "loaded: 3",
            "playbooks": "loaded: 2"
        },
        "active_executions": 0
    })
    return bot


@pytest.fixture
def sample_webhook_payload():
    """Sample GitHub webhook payload for testing."""
    return {
        "action": "completed",
        "workflow_run": {
            "id": 123456789,
            "name": "CI",
            "head_branch": "main",
            "head_sha": "abc123def456",
            "status": "completed",
            "conclusion": "failure",
            "html_url": "https://github.com/testowner/testrepo/actions/runs/123456789",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:05:00Z"
        },
        "repository": {
            "id": 123456,
            "name": "testrepo",
            "full_name": "testowner/testrepo",
            "owner": {
                "login": "testowner",
                "id": 12345
            },
            "private": False,
            "html_url": "https://github.com/testowner/testrepo"
        },
        "installation": {
            "id": 12345
        }
    }


@pytest.fixture
def mock_issue_data():
    """Mock issue data for testing."""
    return {
        "type": "workflow_failure_test_failure",
        "severity": "high",
        "message": "Workflow 'CI' failed with test_failure",
        "data": {
            "workflow_id": 123456,
            "workflow_name": "CI",
            "failure_type": "test_failure",
            "run_url": "https://github.com/testowner/testrepo/actions/runs/123456",
            "head_sha": "abc123def456"
        },
        "detector": "PipelineFailureDetector",
        "timestamp": "2025-01-01T00:00:00Z"
    }


@pytest.fixture
def mock_file_changes():
    """Mock file changes for testing."""
    return {
        "src/model.py": "# Fixed import error\nimport numpy as np\n",
        "requirements.txt": "numpy>=1.24.0\npandas>=2.1.0\n",
        "config.yaml": "batch_size: 16\nlearning_rate: 0.001\n"
    }


class MockAsyncIterator:
    """Mock async iterator for testing."""
    
    def __init__(self, items):
        self.items = items
        self.index = 0
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


@pytest.fixture
def mock_async_iterator():
    """Mock async iterator factory."""
    return MockAsyncIterator