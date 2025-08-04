"""Tests for integration functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx
import jwt
import time

from self_healing_bot.integrations.github import GitHubIntegration
from self_healing_bot.core.config import config


class TestGitHubIntegration:
    """Test cases for GitHubIntegration."""
    
    def test_generate_jwt(self):
        """Test JWT generation for GitHub App."""
        integration = GitHubIntegration()
        
        token = integration.generate_jwt()
        
        # Verify it's a valid JWT structure
        assert len(token.split('.')) == 3
        
        # Decode and verify payload (without verification for testing)
        payload = jwt.decode(token, options={"verify_signature": False})
        assert payload["iss"] == config.github_app_id
        assert "iat" in payload
        assert "exp" in payload
    
    @pytest.mark.asyncio
    async def test_get_installation_token_success(self):
        """Test successful installation token retrieval."""
        integration = GitHubIntegration()
        
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "token": "ghs_testfaketoken",
            "expires_at": "2025-01-01T01:00:00Z"
        }
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            token = await integration.get_installation_token(12345)
            
            assert token == "ghs_testfaketoken"
            # Verify token is cached
            assert 12345 in integration._installation_tokens
    
    @pytest.mark.asyncio
    async def test_get_installation_token_cached(self):
        """Test cached installation token retrieval."""
        integration = GitHubIntegration()
        
        # Manually add cached token
        future_expiry = time.time() + 3600  # 1 hour from now
        integration._installation_tokens[12345] = {
            "token": "cached_token",
            "expires_at": future_expiry
        }
        
        token = await integration.get_installation_token(12345)
        
        assert token == "cached_token"
    
    @pytest.mark.asyncio
    async def test_get_installation_token_failure(self):
        """Test installation token retrieval failure."""
        integration = GitHubIntegration()
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            
            with pytest.raises(Exception) as exc_info:
                await integration.get_installation_token(12345)
            
            assert "Failed to get installation token" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_get_github_client(self):
        """Test GitHub client creation."""
        integration = GitHubIntegration()
        
        with patch.object(integration, 'get_installation_token') as mock_get_token:
            mock_get_token.return_value = "test_token"
            
            client = await integration.get_github_client(12345)
            
            assert client is not None
            mock_get_token.assert_called_once_with(12345)
    
    @pytest.mark.asyncio
    async def test_create_pull_request_success(self):
        """Test successful pull request creation."""
        integration = GitHubIntegration()
        
        mock_pr = Mock()
        mock_pr.number = 123
        mock_pr.html_url = "https://github.com/owner/repo/pull/123"
        mock_pr.title = "Test PR"
        mock_pr.state = "open"
        
        mock_repo = Mock()
        mock_repo.create_pull.return_value = mock_pr
        mock_repo.get_git_ref.return_value.object.sha = "abc123"
        mock_repo.create_git_ref.return_value = None
        mock_repo.create_file.return_value = None
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        
        with patch.object(integration, 'get_github_client') as mock_get_client:
            mock_get_client.return_value = mock_github
            
            result = await integration.create_pull_request(
                installation_id=12345,
                repo_full_name="owner/repo",
                title="Test PR",
                body="Test body",
                head_branch="feature-branch",
                file_changes={"test.txt": "test content"}
            )
            
            assert result["number"] == 123
            assert result["url"] == "https://github.com/owner/repo/pull/123"
            assert result["title"] == "Test PR"
            assert result["state"] == "open"
            
            # Verify repository operations were called
            mock_repo.create_pull.assert_called_once()
            mock_repo.create_file.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_create_issue_success(self):
        """Test successful issue creation."""
        integration = GitHubIntegration()
        
        mock_issue = Mock()
        mock_issue.number = 456
        mock_issue.html_url = "https://github.com/owner/repo/issues/456"
        mock_issue.title = "Test Issue"
        mock_issue.state = "open"
        
        mock_repo = Mock()
        mock_repo.create_issue.return_value = mock_issue
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        
        with patch.object(integration, 'get_github_client') as mock_get_client:
            mock_get_client.return_value = mock_github
            
            result = await integration.create_issue(
                installation_id=12345,
                repo_full_name="owner/repo",
                title="Test Issue",
                body="Test body",
                labels=["bug", "help wanted"]
            )
            
            assert result["number"] == 456
            assert result["url"] == "https://github.com/owner/repo/issues/456"
            assert result["title"] == "Test Issue"
            assert result["state"] == "open"
            
            mock_repo.create_issue.assert_called_once_with(
                title="Test Issue",
                body="Test body",
                labels=["bug", "help wanted"]
            )
    
    @pytest.mark.asyncio
    async def test_add_comment_success(self):
        """Test successful comment addition."""
        integration = GitHubIntegration()
        
        mock_comment = Mock()
        mock_comment.id = 789
        mock_comment.html_url = "https://github.com/owner/repo/issues/456#issuecomment-789"
        mock_comment.body = "Test comment"
        
        mock_issue = Mock()
        mock_issue.create_comment.return_value = mock_comment
        
        mock_repo = Mock()
        mock_repo.get_issue.return_value = mock_issue
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        
        with patch.object(integration, 'get_github_client') as mock_get_client:
            mock_get_client.return_value = mock_github
            
            result = await integration.add_comment(
                installation_id=12345,
                repo_full_name="owner/repo",
                issue_number=456,
                comment="Test comment"
            )
            
            assert result["id"] == 789
            assert result["url"] == "https://github.com/owner/repo/issues/456#issuecomment-789"
            assert result["body"] == "Test comment"
            
            mock_issue.create_comment.assert_called_once_with("Test comment")
    
    @pytest.mark.asyncio
    async def test_get_workflow_runs_success(self):
        """Test successful workflow runs retrieval."""
        integration = GitHubIntegration()
        
        from datetime import datetime
        
        mock_run1 = Mock()
        mock_run1.id = 123
        mock_run1.name = "CI"
        mock_run1.status = "completed"
        mock_run1.conclusion = "success"
        mock_run1.html_url = "https://github.com/owner/repo/actions/runs/123"
        mock_run1.created_at = datetime(2025, 1, 1, 0, 0, 0)
        mock_run1.updated_at = datetime(2025, 1, 1, 0, 5, 0)
        
        mock_run2 = Mock()
        mock_run2.id = 124
        mock_run2.name = "Tests"
        mock_run2.status = "completed"
        mock_run2.conclusion = "failure"
        mock_run2.html_url = "https://github.com/owner/repo/actions/runs/124"
        mock_run2.created_at = datetime(2025, 1, 1, 1, 0, 0)
        mock_run2.updated_at = datetime(2025, 1, 1, 1, 5, 0)
        
        mock_repo = Mock()
        mock_repo.get_workflow_runs.return_value = [mock_run1, mock_run2]
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        
        with patch.object(integration, 'get_github_client') as mock_get_client:
            mock_get_client.return_value = mock_github
            
            results = await integration.get_workflow_runs(
                installation_id=12345,
                repo_full_name="owner/repo",
                limit=10
            )
            
            assert len(results) == 2
            
            assert results[0]["id"] == 123
            assert results[0]["name"] == "CI"
            assert results[0]["status"] == "completed"
            assert results[0]["conclusion"] == "success"
            
            assert results[1]["id"] == 124
            assert results[1]["name"] == "Tests"
            assert results[1]["conclusion"] == "failure"
    
    @pytest.mark.asyncio
    async def test_get_file_content_success(self):
        """Test successful file content retrieval."""
        integration = GitHubIntegration()
        
        mock_file = Mock()
        mock_file.decoded_content = b"print('Hello, World!')"
        
        mock_repo = Mock()
        mock_repo.get_contents.return_value = mock_file
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        
        with patch.object(integration, 'get_github_client') as mock_get_client:
            mock_get_client.return_value = mock_github
            
            content = await integration.get_file_content(
                installation_id=12345,
                repo_full_name="owner/repo",
                file_path="main.py",
                ref="main"
            )
            
            assert content == "print('Hello, World!')"
            mock_repo.get_contents.assert_called_once_with("main.py", ref="main")
    
    @pytest.mark.asyncio
    async def test_get_file_content_not_found(self):
        """Test file content retrieval when file not found."""
        integration = GitHubIntegration()
        
        mock_repo = Mock()
        mock_repo.get_contents.side_effect = Exception("File not found")
        
        mock_github = Mock()
        mock_github.get_repo.return_value = mock_repo
        
        with patch.object(integration, 'get_github_client') as mock_get_client:
            mock_get_client.return_value = mock_github
            
            content = await integration.get_file_content(
                installation_id=12345,
                repo_full_name="owner/repo",
                file_path="nonexistent.py"
            )
            
            assert content is None
    
    @pytest.mark.asyncio
    async def test_test_connection_success(self):
        """Test successful connection test."""
        integration = GitHubIntegration()
        
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await integration.test_connection()
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_test_connection_failure(self):
        """Test connection test failure."""
        integration = GitHubIntegration()
        
        mock_response = Mock()
        mock_response.status_code = 401
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(return_value=mock_response)
            
            result = await integration.test_connection()
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_test_connection_exception(self):
        """Test connection test with exception."""
        integration = GitHubIntegration()
        
        with patch('httpx.AsyncClient') as mock_client:
            mock_client.return_value.__aenter__.return_value.get = AsyncMock(side_effect=Exception("Network error"))
            
            result = await integration.test_connection()
            
            assert result is False