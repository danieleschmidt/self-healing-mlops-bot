"""GitHub API integration."""

import asyncio
import logging
from typing import Dict, Any, Optional, List
import jwt
import time
import httpx
from github import Github, Auth

from ..core.config import config

logger = logging.getLogger(__name__)


class GitHubIntegration:
    """GitHub API integration for the self-healing bot."""
    
    def __init__(self):
        self.app_id = config.github_app_id
        self.private_key = config.github_private_key
        self.webhook_secret = config.github_webhook_secret
        self._installation_tokens: Dict[int, Dict[str, Any]] = {}
    
    def generate_jwt(self) -> str:
        """Generate JWT for GitHub App authentication."""
        now = int(time.time())
        payload = {
            "iat": now - 60,  # issued at (60 seconds ago to account for clock skew)
            "exp": now + (10 * 60),  # expires in 10 minutes
            "iss": self.app_id
        }
        
        return jwt.encode(payload, self.private_key, algorithm="RS256")
    
    async def get_installation_token(self, installation_id: int) -> str:
        """Get installation access token."""
        # Check if we have a cached token that's still valid
        if installation_id in self._installation_tokens:
            token_data = self._installation_tokens[installation_id]
            expires_at = token_data.get("expires_at", 0)
            if time.time() < expires_at - 300:  # 5 minute buffer
                return token_data["token"]
        
        # Generate new token
        jwt_token = self.generate_jwt()
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"https://api.github.com/app/installations/{installation_id}/access_tokens",
                headers={
                    "Authorization": f"Bearer {jwt_token}",
                    "Accept": "application/vnd.github.v3+json",
                    "User-Agent": "self-healing-mlops-bot/1.0"
                }
            )
            
            if response.status_code != 201:
                raise Exception(f"Failed to get installation token: {response.text}")
            
            token_data = response.json()
            
            # Cache the token
            self._installation_tokens[installation_id] = {
                "token": token_data["token"],
                "expires_at": time.mktime(
                    time.strptime(token_data["expires_at"], "%Y-%m-%dT%H:%M:%SZ")
                )
            }
            
            return token_data["token"]
    
    async def get_github_client(self, installation_id: int) -> Github:
        """Get authenticated GitHub client for installation."""
        token = await self.get_installation_token(installation_id)
        auth = Auth.Token(token)
        return Github(auth=auth)
    
    async def create_pull_request(
        self,
        installation_id: int,
        repo_full_name: str,
        title: str,
        body: str,
        head_branch: str,
        base_branch: str = "main",
        file_changes: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Create a pull request with file changes."""
        try:
            github = await self.get_github_client(installation_id)
            repo = github.get_repo(repo_full_name)
            
            # Create new branch if file changes provided
            if file_changes:
                # Get base branch reference
                base_ref = repo.get_git_ref(f"heads/{base_branch}")
                base_sha = base_ref.object.sha
                
                # Create new branch
                try:
                    repo.create_git_ref(f"refs/heads/{head_branch}", base_sha)
                except Exception as e:
                    # Branch might already exist
                    logger.warning(f"Branch {head_branch} might already exist: {e}")
                
                # Update files
                for file_path, content in file_changes.items():
                    try:
                        # Try to get existing file
                        file_obj = repo.get_contents(file_path, ref=head_branch)
                        repo.update_file(
                            path=file_path,
                            message=f"🤖 Update {file_path}",
                            content=content,
                            sha=file_obj.sha,
                            branch=head_branch
                        )
                    except Exception:
                        # File doesn't exist, create it
                        repo.create_file(
                            path=file_path,
                            message=f"🤖 Create {file_path}",
                            content=content,
                            branch=head_branch
                        )
            
            # Create pull request
            pr = repo.create_pull(
                title=title,
                body=body,
                head=head_branch,
                base=base_branch
            )
            
            return {
                "number": pr.number,
                "url": pr.html_url,
                "title": pr.title,
                "state": pr.state
            }
            
        except Exception as e:
            logger.exception(f"Error creating pull request: {e}")
            raise
    
    async def create_issue(
        self,
        installation_id: int,
        repo_full_name: str,
        title: str,
        body: str,
        labels: List[str] = None
    ) -> Dict[str, Any]:
        """Create an issue in the repository."""
        try:
            github = await self.get_github_client(installation_id)
            repo = github.get_repo(repo_full_name)
            
            issue = repo.create_issue(
                title=title,
                body=body,
                labels=labels or []
            )
            
            return {
                "number": issue.number,
                "url": issue.html_url,
                "title": issue.title,
                "state": issue.state
            }
            
        except Exception as e:
            logger.exception(f"Error creating issue: {e}")
            raise
    
    async def add_comment(
        self,
        installation_id: int,
        repo_full_name: str,
        issue_number: int,
        comment: str
    ) -> Dict[str, Any]:
        """Add comment to an issue or pull request."""
        try:
            github = await self.get_github_client(installation_id)
            repo = github.get_repo(repo_full_name)
            
            issue = repo.get_issue(issue_number)
            comment_obj = issue.create_comment(comment)
            
            return {
                "id": comment_obj.id,
                "url": comment_obj.html_url,
                "body": comment_obj.body
            }
            
        except Exception as e:
            logger.exception(f"Error adding comment: {e}")
            raise
    
    async def get_workflow_runs(
        self,
        installation_id: int,
        repo_full_name: str,
        workflow_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get workflow runs for a repository."""
        try:
            github = await self.get_github_client(installation_id)
            repo = github.get_repo(repo_full_name)
            
            if workflow_id:
                workflow = repo.get_workflow(workflow_id)
                runs = workflow.get_runs()
            else:
                runs = repo.get_workflow_runs()
            
            results = []
            count = 0
            for run in runs:
                if count >= limit:
                    break
                
                if status and run.status != status:
                    continue
                
                results.append({
                    "id": run.id,
                    "name": run.name,
                    "status": run.status,
                    "conclusion": run.conclusion,
                    "html_url": run.html_url,
                    "created_at": run.created_at.isoformat(),
                    "updated_at": run.updated_at.isoformat()
                })
                count += 1
            
            return results
            
        except Exception as e:
            logger.exception(f"Error getting workflow runs: {e}")
            raise
    
    async def get_file_content(
        self,
        installation_id: int,
        repo_full_name: str,
        file_path: str,
        ref: str = "main"
    ) -> Optional[str]:
        """Get content of a file from repository."""
        try:
            github = await self.get_github_client(installation_id)
            repo = github.get_repo(repo_full_name)
            
            file_obj = repo.get_contents(file_path, ref=ref)
            return file_obj.decoded_content.decode("utf-8")
            
        except Exception as e:
            logger.exception(f"Error getting file content: {e}")
            return None
    
    async def test_connection(self) -> bool:
        """Test GitHub API connection."""
        try:
            jwt_token = self.generate_jwt()
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.github.com/app",
                    headers={
                        "Authorization": f"Bearer {jwt_token}",
                        "Accept": "application/vnd.github.v3+json",
                        "User-Agent": "self-healing-mlops-bot/1.0"
                    }
                )
                
                return response.status_code == 200
                
        except Exception as e:
            logger.exception(f"GitHub connection test failed: {e}")
            return False