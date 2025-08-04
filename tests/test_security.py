"""Tests for security functionality."""

import pytest
from unittest.mock import Mock, patch

from self_healing_bot.security.validation import (
    InputValidator, ValidationError, RateLimiter
)
from self_healing_bot.security.secrets import (
    SecretsManager, SecretScanner, EnvironmentProtector
)


class TestInputValidator:
    """Test cases for InputValidator."""
    
    def test_validate_repo_name_valid(self):
        """Test valid repository name validation."""
        valid_names = [
            "owner/repo",
            "user123/my-repo",
            "org_name/repo.name",
            "a/b"
        ]
        
        for name in valid_names:
            result = InputValidator.validate_repo_name(name)
            assert result == name.strip()
    
    def test_validate_repo_name_invalid(self):
        """Test invalid repository name validation."""
        invalid_names = [
            "",
            "owner",  # Missing repo name
            "owner/",  # Empty repo name
            "/repo",  # Empty owner
            "owner/repo/extra",  # Too many parts
            "owner repo",  # Space not allowed
            "owner@repo",  # @ not allowed
            "a" * 101,  # Too long
        ]
        
        for name in invalid_names:
            with pytest.raises(ValidationError):
                InputValidator.validate_repo_name(name)
    
    def test_validate_branch_name_valid(self):
        """Test valid branch name validation."""
        valid_names = [
            "main",
            "feature/new-feature",
            "hotfix-123",
            "release/v1.0.0",
            "user/feature-branch"
        ]
        
        for name in valid_names:
            result = InputValidator.validate_branch_name(name)
            assert result == name.strip()
    
    def test_validate_branch_name_invalid(self):
        """Test invalid branch name validation."""
        invalid_names = [
            "",
            "branch with spaces",
            "branch@invalid",
            "branch#invalid",
            "../dangerous",
            "a" * 251,  # Too long
        ]
        
        for name in invalid_names:
            with pytest.raises(ValidationError):
                InputValidator.validate_branch_name(name)
    
    def test_validate_file_path_valid(self):
        """Test valid file path validation."""
        valid_paths = [
            "src/main.py",
            "docs/README.md",
            "config/app.yaml",
            "tests/test_file.py"
        ]
        
        for path in valid_paths:
            result = InputValidator.validate_file_path(path)
            assert result == path.strip()
    
    def test_validate_file_path_invalid(self):
        """Test invalid file path validation."""
        invalid_paths = [
            "",
            "../../../etc/passwd",  # Path traversal
            "/absolute/path",  # Absolute path
            "path/../file.txt",  # Path traversal
            "path with spaces/file.txt",  # Spaces not allowed
            "path@invalid/file.txt",  # Invalid characters
            "a" * 501,  # Too long
        ]
        
        for path in invalid_paths:
            with pytest.raises(ValidationError):
                InputValidator.validate_file_path(path)
    
    def test_validate_commit_message_valid(self):
        """Test valid commit message validation."""
        valid_messages = [
            "Fix bug in authentication",
            "Add new feature for user management",
            "Update documentation",
            "🚀 Deploy version 1.0.0"
        ]
        
        for message in valid_messages:
            result = InputValidator.validate_commit_message(message)
            assert result == message.strip()
    
    def test_validate_commit_message_invalid(self):
        """Test invalid commit message validation."""
        invalid_messages = [
            "",
            "a" * 501,  # Too long
            "<script>alert('xss')</script>",  # XSS attempt
            "javascript:void(0)",  # JavaScript injection
        ]
        
        for message in invalid_messages:
            with pytest.raises(ValidationError):
                InputValidator.validate_commit_message(message)
    
    def test_validate_url_valid(self):
        """Test valid URL validation."""
        valid_urls = [
            "https://api.github.com/repos/owner/repo",
            "https://example.com/webhook",
            "https://secure-api.example.org/endpoint"
        ]
        
        for url in valid_urls:
            result = InputValidator.validate_url(url)
            assert result == url.strip()
    
    def test_validate_url_invalid(self):
        """Test invalid URL validation."""
        invalid_urls = [
            "",
            "http://insecure.com",  # HTTP not allowed
            "ftp://file.server.com",  # FTP not allowed
            "https://localhost/local",  # Localhost not allowed
            "https://127.0.0.1/local",  # Local IP not allowed
            "not-a-url",  # Invalid format
        ]
        
        for url in invalid_urls:
            with pytest.raises(ValidationError):
                InputValidator.validate_url(url)
    
    def test_validate_webhook_payload_valid(self):
        """Test valid webhook payload validation."""
        valid_payload = {
            "action": "opened",
            "repository": {
                "full_name": "owner/repo",
                "name": "repo"
            },
            "pull_request": {
                "number": 123,
                "title": "Test PR"
            }
        }
        
        result = InputValidator.validate_webhook_payload(valid_payload)
        assert result == valid_payload
    
    def test_validate_webhook_payload_invalid(self):
        """Test invalid webhook payload validation."""
        # Non-dictionary payload
        with pytest.raises(ValidationError):
            InputValidator.validate_webhook_payload("not a dict")
        
        # Invalid repository name
        with pytest.raises(ValidationError):
            InputValidator.validate_webhook_payload({
                "repository": {"full_name": "invalid@repo"}
            })
    
    def test_sanitize_log_message(self):
        """Test log message sanitization."""
        # Test newline removal
        message = "Line 1\nLine 2\rLine 3\tTab"
        sanitized = InputValidator.sanitize_log_message(message)
        assert "\n" not in sanitized
        assert "\r" not in sanitized
        assert "\t" not in sanitized
        
        # Test length truncation
        long_message = "a" * 1500
        sanitized = InputValidator.sanitize_log_message(long_message)
        assert len(sanitized) == 1000
        assert sanitized.endswith("...")
        
        # Test non-string input
        sanitized = InputValidator.sanitize_log_message(123)
        assert sanitized == "123"
    
    def test_dangerous_patterns_detection(self):
        """Test dangerous pattern detection."""
        dangerous_inputs = [
            "../etc/passwd",
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "eval(malicious_code)",
            "import os; os.system('rm -rf /')"
        ]
        
        for dangerous_input in dangerous_inputs:
            with pytest.raises(ValidationError):
                InputValidator.validate_commit_message(dangerous_input)


class TestRateLimiter:
    """Test cases for RateLimiter."""
    
    def test_rate_limiting_allowed(self):
        """Test requests within rate limit."""
        limiter = RateLimiter()
        
        # First request should be allowed
        assert limiter.is_allowed("127.0.0.1", "default") is True
        
        # Subsequent requests within limit should be allowed
        for _ in range(50):  # Default limit is 60/minute
            assert limiter.is_allowed("127.0.0.1", "default") is True
    
    def test_rate_limiting_exceeded(self):
        """Test requests exceeding rate limit."""
        limiter = RateLimiter()
        
        # Exhaust the rate limit
        for _ in range(60):  # Default limit is 60/minute
            limiter.is_allowed("127.0.0.1", "default")
        
        # Next request should be denied
        assert limiter.is_allowed("127.0.0.1", "default") is False
    
    def test_rate_limiting_different_identifiers(self):
        """Test rate limiting with different client identifiers."""
        limiter = RateLimiter()
        
        # Exhaust limit for first client
        for _ in range(60):
            limiter.is_allowed("192.168.1.1", "default")
        
        # Second client should not be affected
        assert limiter.is_allowed("192.168.1.2", "default") is True
    
    def test_rate_limiting_different_endpoints(self):
        """Test rate limiting for different endpoints."""
        limiter = RateLimiter()
        
        # Exhaust limit for webhook endpoint
        for _ in range(100):  # Webhook limit is 100/minute
            limiter.is_allowed("127.0.0.1", "webhook")
        
        # Default endpoint should not be affected
        assert limiter.is_allowed("127.0.0.1", "default") is True


class TestSecretsManager:
    """Test cases for SecretsManager."""
    
    def test_encryption_decryption(self):
        """Test value encryption and decryption."""
        manager = SecretsManager()
        
        original_value = "super_secret_password"
        encrypted = manager.encrypt_value(original_value)
        decrypted = manager.decrypt_value(encrypted)
        
        assert encrypted != original_value
        assert decrypted == original_value
    
    def test_mask_sensitive_data(self):
        """Test sensitive data masking."""
        manager = SecretsManager()
        
        sensitive_data = {
            "username": "user123",
            "password": "secret123",
            "api_key": "sk-testfakekey123456",
            "token": "ghp_testfaketoken123456789012345678901234567890",
            "normal_field": "not_sensitive",
            "nested": {
                "secret": "nested_secret",
                "public": "public_data"
            }
        }
        
        masked = manager.mask_sensitive_data(sensitive_data)
        
        # Sensitive fields should be masked
        assert masked["password"] != "secret123"
        assert "*" in masked["password"]
        assert masked["api_key"] != "sk-testfakekey123456"
        assert masked["token"] != "ghp_testfaketoken123456789012345678901234567890"
        
        # Non-sensitive fields should be unchanged
        assert masked["username"] == "user123"
        assert masked["normal_field"] == "not_sensitive"
        assert masked["nested"]["public"] == "public_data"
        
        # Nested sensitive fields should be masked
        assert masked["nested"]["secret"] != "nested_secret"
    
    def test_mask_short_values(self):
        """Test masking of short sensitive values."""
        manager = SecretsManager()
        
        data = {
            "short_password": "123",
            "medium_key": "abcd1234"
        }
        
        masked = manager.mask_sensitive_data(data)
        
        # Short values should be completely masked
        assert masked["short_password"] == "***"
        assert masked["medium_key"] == "****1234"  # Show last 4 chars


class TestSecretScanner:
    """Test cases for SecretScanner."""
    
    def test_scan_github_tokens(self):
        """Test scanning for GitHub tokens."""
        scanner = SecretScanner()
        
        text_with_secrets = """
        const token = "ghp_testfaketoken123456789012345678901234567890";
        const installation_token = "ghs_testfaketoken123456789012345678901234567890";
        """
        
        findings = scanner.scan_text(text_with_secrets)
        
        assert len(findings) >= 2
        token_types = [f["type"] for f in findings]
        assert "GitHub Personal Access Token" in token_types
        assert "GitHub App Installation Token" in token_types
    
    def test_scan_aws_credentials(self):
        """Test scanning for AWS credentials."""
        scanner = SecretScanner()
        
        text_with_secrets = """
        AWS_ACCESS_KEY_ID = "AKIATEST1234EXAMPLE"
        AWS_SECRET_ACCESS_KEY = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        """
        
        findings = scanner.scan_text(text_with_secrets)
        
        assert len(findings) >= 1
        aws_findings = [f for f in findings if "AWS" in f["type"]]
        assert len(aws_findings) > 0
    
    def test_scan_api_keys(self):
        """Test scanning for API keys."""
        scanner = SecretScanner()
        
        text_with_secrets = """
        OPENAI_API_KEY = "sk-test123fake456key789test123fake456test123"
        SLACK_BOT_TOKEN = "xoxb-test123-fake456-testfakekeytestfakekey"
        """
        
        findings = scanner.scan_text(text_with_secrets)
        
        assert len(findings) >= 2
        token_types = [f["type"] for f in findings]
        assert "OpenAI API Key" in token_types
        assert "Slack Bot Token" in token_types
    
    def test_scan_password_assignments(self):
        """Test scanning for password assignments."""
        scanner = SecretScanner()
        
        text_with_secrets = """
        password = "super_secret_password"
        api_key = "my_secret_api_key"
        """
        
        findings = scanner.scan_text(text_with_secrets)
        
        password_findings = [f for f in findings if "password" in f["type"].lower()]
        assert len(password_findings) > 0
    
    def test_scan_no_secrets(self):
        """Test scanning text with no secrets."""
        scanner = SecretScanner()
        
        clean_text = """
        def hello_world():
            print("Hello, World!")
            return "success"
        """
        
        findings = scanner.scan_text(clean_text)
        
        # Should have no findings or only very generic ones
        assert len(findings) == 0
    
    def test_validate_pr_changes(self):
        """Test validating PR changes for secrets."""
        scanner = SecretScanner()
        
        file_changes = {
            "config.py": 'API_KEY = "sk-testfakekey123456"',
            "README.md": "# This is a README file",
            "secrets.yml": 'github_token: "ghp_testfaketoken123456789012345678901234567890"'
        }
        
        findings = scanner.validate_pr_changes(file_changes)
        
        # Should find secrets in config.py and secrets.yml
        assert "config.py" in findings
        assert "secrets.yml" in findings
        assert "README.md" not in findings
        
        # Verify findings structure
        assert len(findings["config.py"]) > 0
        assert len(findings["secrets.yml"]) > 0


class TestEnvironmentProtector:
    """Test cases for EnvironmentProtector."""
    
    def test_filter_environment(self):
        """Test environment variable filtering."""
        env_vars = {
            "HOME": "/home/user",
            "PATH": "/usr/bin:/bin",
            "GITHUB_TOKEN": "ghp_testfaketoken",
            "SECRET_KEY": "super_secret_key",
            "DATABASE_URL": "postgresql://user:pass@host:5432/db",
            "DEBUG": "true",
            "MY_API_KEY": "secret_api_key"
        }
        
        filtered = EnvironmentProtector.filter_environment(env_vars)
        
        # Non-sensitive variables should be unchanged
        assert filtered["HOME"] == "/home/user"
        assert filtered["PATH"] == "/usr/bin:/bin"
        assert filtered["DEBUG"] == "true"
        
        # Sensitive variables should be redacted
        assert filtered["GITHUB_TOKEN"] == "***REDACTED***"
        assert filtered["SECRET_KEY"] == "***REDACTED***"
        assert filtered["DATABASE_URL"] == "***REDACTED***"
        assert filtered["MY_API_KEY"] == "***REDACTED***"
    
    def test_safe_environment_dump(self):
        """Test safe environment dump."""
        with patch('os.environ', {
            "USER": "testuser",
            "SECRET_TOKEN": "secret123",
            "API_KEY": "key123"
        }):
            safe_env = EnvironmentProtector.safe_environment_dump()
            
            assert safe_env["USER"] == "testuser"
            assert safe_env["SECRET_TOKEN"] == "***REDACTED***"
            assert safe_env["API_KEY"] == "***REDACTED***"