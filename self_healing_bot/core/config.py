"""Configuration management for the self-healing bot."""

import os
from typing import Optional, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from pathlib import Path


class BotConfig(BaseSettings):
    """Main configuration for the self-healing bot."""
    
    # GitHub App Configuration
    github_app_id: str = Field("test", env="GITHUB_APP_ID")
    github_private_key_path: str = Field("/tmp/test-key.pem", env="GITHUB_PRIVATE_KEY_PATH")
    github_webhook_secret: str = Field("test-secret", env="GITHUB_WEBHOOK_SECRET")
    github_token: Optional[str] = Field(None, env="GITHUB_TOKEN")
    
    # Server Configuration
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8080, env="PORT")
    debug: bool = Field(False, env="DEBUG")
    environment: str = Field("production", env="ENVIRONMENT")
    
    # Database Configuration
    database_url: str = Field("sqlite:///test.db", env="DATABASE_URL")
    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    
    # Celery Configuration
    celery_broker_url: str = Field("redis://localhost:6379/1", env="CELERY_BROKER_URL")
    celery_result_backend: str = Field("redis://localhost:6379/2", env="CELERY_RESULT_BACKEND")
    
    # Monitoring Configuration
    prometheus_port: int = Field(9090, env="PROMETHEUS_PORT")
    log_level: str = Field("INFO", env="LOG_LEVEL")
    
    # Notification Configuration
    slack_webhook_url: Optional[str] = Field(None, env="SLACK_WEBHOOK_URL")
    slack_bot_token: Optional[str] = Field(None, env="SLACK_BOT_TOKEN")
    
    # ML Platform Integrations
    mlflow_tracking_uri: Optional[str] = Field(None, env="MLFLOW_TRACKING_URI")
    wandb_api_key: Optional[str] = Field(None, env="WANDB_API_KEY")
    
    # Security
    secret_key: str = Field("test-secret-key-for-development", env="SECRET_KEY")
    encryption_key: str = Field("test-encryption-key-32-bytes-long", env="ENCRYPTION_KEY")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(60, env="RATE_LIMIT_PER_MINUTE")
    
    # Health Check Configuration
    health_check_interval: int = Field(300, env="HEALTH_CHECK_INTERVAL")
    
    @validator("github_private_key_path")
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
    
    @validator("log_level")
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


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