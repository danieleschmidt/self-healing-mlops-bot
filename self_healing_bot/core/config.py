"""Configuration management for the self-healing bot."""

import os
from typing import Optional, Dict, Any, Annotated
from pydantic import Field, field_validator, ConfigDict
from pydantic_settings import BaseSettings
from pathlib import Path


class BotConfig(BaseSettings):
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )
    """Main configuration for the self-healing bot."""
    
    # GitHub App Configuration
    github_app_id: Annotated[str, Field(description="GitHub App ID")] = "test"
    github_private_key_path: Annotated[str, Field(description="Path to GitHub private key")] = "/tmp/test-key.pem"
    github_webhook_secret: Annotated[str, Field(description="GitHub webhook secret")] = "test-secret"
    github_token: Annotated[Optional[str], Field(description="GitHub token")] = None
    
    # Server Configuration
    host: Annotated[str, Field(description="Server host")] = "0.0.0.0"
    port: Annotated[int, Field(description="Server port")] = 8080
    debug: Annotated[bool, Field(description="Debug mode")] = False
    environment: Annotated[str, Field(description="Environment")] = "production"
    
    # Database Configuration
    database_url: Annotated[str, Field(description="Database URL")] = "sqlite:///test.db"
    redis_url: Annotated[str, Field(description="Redis URL")] = "redis://localhost:6379/0"
    
    # Celery Configuration
    celery_broker_url: Annotated[str, Field(description="Celery broker URL")] = "redis://localhost:6379/1"
    celery_result_backend: Annotated[str, Field(description="Celery result backend")] = "redis://localhost:6379/2"
    
    # Monitoring Configuration
    prometheus_port: Annotated[int, Field(description="Prometheus port")] = 9090
    log_level: Annotated[str, Field(description="Log level")] = "INFO"
    
    # Notification Configuration
    slack_webhook_url: Annotated[Optional[str], Field(description="Slack webhook URL")] = None
    slack_bot_token: Annotated[Optional[str], Field(description="Slack bot token")] = None
    
    # ML Platform Integrations
    mlflow_tracking_uri: Annotated[Optional[str], Field(description="MLflow tracking URI")] = None
    wandb_api_key: Annotated[Optional[str], Field(description="Weights & Biases API key")] = None
    
    # Security
    secret_key: Annotated[str, Field(description="Secret key")] = "test-secret-key-for-development"
    encryption_key: Annotated[str, Field(description="Encryption key")] = "test-encryption-key-32-bytes-long"
    
    # Rate Limiting
    rate_limit_per_minute: Annotated[int, Field(description="Rate limit per minute")] = 60
    
    # Health Check Configuration
    health_check_interval: Annotated[int, Field(description="Health check interval")] = 300
    
    @field_validator("github_private_key_path")
    @classmethod
    def validate_private_key_path(cls, v: str) -> str:
        """Validate that the private key file exists."""
        path = Path(v)
        if not path.exists():
            # For development, create a mock key file if it doesn't exist
            if v == "/tmp/test-key.pem":
                path.parent.mkdir(parents=True, exist_ok=True)
                with open(path, 'w') as f:
                    f.write("-----BEGIN RSA PRIVATE KEY-----\nMOCK_KEY_FOR_DEVELOPMENT\n-----END RSA PRIVATE KEY-----\n")
                return v
            raise ValueError(f"GitHub private key file not found: {v}")
        return v
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()
    
    @property
    def github_private_key(self) -> str:
        """Read and return the GitHub private key."""
        with open(self.github_private_key_path, "r") as f:
            return f.read()
    


class PlaybookConfig:
    """Configuration for individual playbooks."""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self.config = config_dict
    
    @property
    def name(self) -> str:
        return self.config.get("name", "unknown")
    
    @property
    def version(self) -> str:
        return self.config.get("version", "1.0")
    
    @property
    def triggers(self) -> list:
        return self.config.get("triggers", [])
    
    @property
    def actions(self) -> list:
        return self.config.get("actions", [])
    
    @property
    def timeout(self) -> int:
        return self.config.get("timeout", 3600)  # 1 hour default
    
    @property
    def retry_count(self) -> int:
        return self.config.get("retry_count", 3)


# Global configuration instance
config = BotConfig()