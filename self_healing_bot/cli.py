"""Command-line interface for the self-healing bot."""

import asyncio
import click
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

from .core.config import config
from .core.bot import SelfHealingBot
from .core.context import Context
from .web.app import app


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("self_healing_bot.log")
        ]
    )


@click.group()
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--debug", is_flag=True, help="Enable debug mode")
def cli(log_level, debug):
    """Self-Healing MLOps Bot CLI."""
    if debug:
        config.debug = True
        log_level = "DEBUG"
    
    setup_logging(log_level)
    
    if debug:
        click.echo("🐛 Debug mode enabled")


@cli.command()
@click.option("--host", default=None, help="Host to bind to")
@click.option("--port", default=None, type=int, help="Port to bind to")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development")
@click.option("--production", is_flag=True, help="Run in production mode with Gunicorn")
def server(host, port, reload, production):
    """Start the bot server with production-ready configuration."""
    from .server import run_server, run_production_server
    
    # Override config if specified
    if host:
        config.host = host
    if port:
        config.port = port
    if reload:
        config.debug = True
    
    click.echo(f"🚀 Starting Self-Healing MLOps Bot server")
    click.echo(f"📡 Server: {config.host}:{config.port}")
    click.echo(f"🌍 Environment: {config.environment}")
    click.echo(f"🔧 Debug: {config.debug}")
    click.echo(f"🔄 Reload: {reload}")
    click.echo(f"🏭 Production mode: {production}")
    
    if production or config.environment == "production":
        click.echo("Starting production server with enhanced features...")
        run_production_server()
    else:
        click.echo("Starting development server...")
        run_server()


@cli.command()
async def health():
    """Check bot health status."""
    bot = SelfHealingBot()
    
    try:
        health_data = await bot.health_check()
        
        click.echo(f"Bot Status: {health_data['status']}")
        click.echo(f"Timestamp: {health_data['timestamp']}")
        click.echo(f"Active Executions: {health_data['active_executions']}")
        
        click.echo("\nComponents:")
        for component, status in health_data['components'].items():
            click.echo(f"  {component}: {status}")
            
    except Exception as e:
        click.echo(f"Health check failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("repo_full_name")
@click.option("--event-type", default="manual", help="Event type to simulate")
def trigger(repo_full_name, event_type):
    """Manually trigger bot processing for a repository."""
    async def run_trigger():
        bot = SelfHealingBot()
        
        mock_payload = {
            "repository": {
                "full_name": repo_full_name,
                "name": repo_full_name.split("/")[-1],
                "owner": {"login": repo_full_name.split("/")[0]}
            }
        }
        
        try:
            context = await bot.process_event(event_type, mock_payload)
            
            if context:
                click.echo(f"✅ Successfully processed {event_type} event for {repo_full_name}")
                click.echo(f"Execution ID: {context.execution_id}")
                
                if context.has_error():
                    click.echo(f"⚠️  Error: {context.error_message}")
                
                file_changes = context.get_file_changes()
                if file_changes:
                    click.echo(f"📝 File changes: {list(file_changes.keys())}")
            else:
                click.echo("No context returned from bot processing")
                
        except Exception as e:
            click.echo(f"❌ Error processing event: {e}", err=True)
            sys.exit(1)
    
    asyncio.run(run_trigger())


@cli.command()
def list_playbooks():
    """List available playbooks."""
    from .core.playbook import PlaybookRegistry
    
    playbooks = PlaybookRegistry.list_playbooks()
    
    click.echo("Available Playbooks:")
    for playbook_name in playbooks:
        playbook_class = PlaybookRegistry.get_playbook(playbook_name)
        if playbook_class:
            click.echo(f"  • {playbook_name}: {playbook_class.__doc__ or 'No description'}")


@cli.command()
def list_detectors():
    """List available detectors."""
    from .detectors.registry import DetectorRegistry
    
    registry = DetectorRegistry()
    detectors = registry.list_detectors()
    
    click.echo("Available Detectors:")
    for detector_name in detectors:
        detector = registry.get_detector(detector_name)
        if detector:
            supported_events = detector.get_supported_events()
            click.echo(f"  • {detector_name}: {detector.__class__.__doc__ or 'No description'}")
            click.echo(f"    Events: {', '.join(supported_events)}")


@cli.command()
@click.option("--config-file", help="Path to configuration file")
def validate_config(config_file):
    """Validate bot configuration."""
    try:
        # Test configuration loading
        click.echo("✅ Configuration loaded successfully")
        
        # Test GitHub integration
        from .integrations.github import GitHubIntegration
        github = GitHubIntegration()
        
        # Test JWT generation
        jwt_token = github.generate_jwt()
        click.echo("✅ GitHub JWT generation successful")
        
        click.echo("Configuration validation completed successfully")
        
    except Exception as e:
        click.echo(f"❌ Configuration validation failed: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--output-dir", default="./example-configs", help="Output directory for examples")
def generate_examples(output_dir):
    """Generate example configuration files."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Generate example bot configuration
    bot_config = '''# Self-healing bot configuration
version: 1.0

monitoring:
  # Pipeline monitoring
  pipelines:
    - name: "training-pipeline"
      type: "github-actions"
      workflow: ".github/workflows/train.yml"
      success_rate_threshold: 0.95
      
  # Model performance monitoring  
  models:
    - name: "production-model"
      endpoint: "https://api.mycompany.com/predict"
      metrics:
        - name: "accuracy"
          threshold: 0.92
          window: "7d"
        - name: "latency_p95"
          threshold: 200  # ms
          
  # Data drift monitoring
  data:
    - name: "training-data"
      path: "data/processed/"
      drift_threshold: 0.1
      check_frequency: "daily"

# Repair playbooks
playbooks:
  - trigger: "test_failure"
    actions:
      - "analyze_logs"
      - "fix_common_errors"
      - "create_pr"
      
  - trigger: "data_drift"
    actions:
      - "validate_data_quality"
      - "trigger_retraining"
      - "notify_team"

# Notification settings
notifications:
  slack:
    webhook: "${SLACK_WEBHOOK}"
    channels:
      failures: "#ml-ops-alerts"
      repairs: "#ml-ops-activity"
'''
    
    with open(output_path / "self-healing-bot.yml", "w") as f:
        f.write(bot_config)
    
    click.echo(f"✅ Generated example configuration in {output_dir}")


def main():
    """Main CLI entry point."""
    cli()


if __name__ == "__main__":
    main()