"""Extended CLI commands for the self-healing bot."""

import asyncio
import click
import json
import sys
import uuid
from datetime import datetime

from .core.bot import SelfHealingBot
from .core.config import config
from .core.context import Context


@click.command()
@click.argument("repo")
@click.option("--event-type", default="manual", help="Event type to simulate")
@click.option("--event-data", help="JSON event data")
@click.option("--with-error", help="Simulate error with given message")
def trigger(repo: str, event_type: str, event_data: str, with_error: str):
    """Manually trigger bot processing for a repository."""
    click.echo(f"🔧 Triggering {event_type} event for repository: {repo}")
    
    bot = SelfHealingBot()
    
    # Parse event data
    event_payload = {}
    if event_data:
        try:
            event_payload = json.loads(event_data)
        except json.JSONDecodeError:
            click.echo("❌ Error: Invalid JSON in event data", err=True)
            sys.exit(1)
    
    # Create mock repository data
    repo_owner, repo_name = repo.split("/") if "/" in repo else ("unknown", repo)
    event_payload.update({
        "repository": {
            "full_name": repo,
            "name": repo_name,
            "owner": {"login": repo_owner}
        }
    })
    
    # Add error simulation if requested
    if with_error:
        if event_type == "workflow_run":
            event_payload.update({
                "workflow_run": {
                    "name": "Test Workflow",
                    "conclusion": "failure",
                    "id": 12345,
                    "html_url": f"https://github.com/{repo}/actions/runs/12345"
                }
            })
        event_payload["error_message"] = with_error
    
    async def run_trigger():
        try:
            click.echo("⏳ Processing event...")
            context = await bot.process_event(event_type, event_payload)
            
            if context:
                click.echo(f"✅ Successfully processed {event_type} event for {repo}")
                click.echo(f"🆔 Execution ID: {context.execution_id}")
                click.echo(f"⏰ Started at: {context.started_at.isoformat()}")
                
                # Display error info
                if context.has_error():
                    click.echo(f"❌ Error detected: {context.error_message}")
                    click.echo(f"🏷️  Error type: {context.error_type}")
                else:
                    click.echo("✅ No errors detected")
                
                # Show state information
                if context.state:
                    click.echo("📊 Context state:")
                    for key, value in context.state.items():
                        click.echo(f"    {key}: {value}")
                
                # Show file changes
                file_changes = context.get_file_changes()
                if file_changes:
                    click.echo("📝 Files that would be changed:")
                    for file_path, content in file_changes.items():
                        click.echo(f"    • {file_path} ({len(content)} bytes)")
                        if config.debug:
                            click.echo(f"      Content preview: {content[:100]}...")
                else:
                    click.echo("📝 No file changes proposed")
                    
            else:
                click.echo("❌ No context returned from event processing")
                
        except Exception as e:
            click.echo(f"❌ Error processing event: {e}", err=True)
            if config.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    asyncio.run(run_trigger())


@click.command()
def health():
    """Check bot health status."""
    click.echo("🏥 Checking bot health status...")
    
    bot = SelfHealingBot()
    
    async def check_health():
        try:
            health_data = await bot.health_check()
            
            # Overall status
            status_icon = "✅" if health_data['status'] == 'healthy' else "⚠️"
            click.echo(f"{status_icon} Overall Status: {health_data['status']}")
            click.echo(f"🔄 Active executions: {health_data['active_executions']}")
            click.echo(f"⏰ Timestamp: {health_data['timestamp']}")
            
            # Component status
            click.echo("\n🔧 Component Health:")
            components = health_data.get('components', {})
            for component, status in components.items():
                if "healthy" in status.lower():
                    icon = "✅"
                elif "loaded" in status.lower():
                    icon = "📦"
                else:
                    icon = "❌"
                click.echo(f"  {icon} {component}: {status}")
                
            if health_data['status'] != 'healthy':
                click.echo("\n⚠️  Some components are unhealthy. Check logs for details.")
                sys.exit(1)
                
        except Exception as e:
            click.echo(f"❌ Error checking health: {e}", err=True)
            if config.debug:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    asyncio.run(check_health())


@click.command()
def demo():
    """Run a demonstration of the bot capabilities."""
    click.echo("🎭 Self-healing MLOps Bot Demo")
    click.echo("=" * 50)
    
    # Demo scenarios
    scenarios = [
        {
            "name": "GPU OOM Error Fix",
            "repo": "demo/ml-training",
            "event_type": "workflow_run",
            "error": "CUDA out of memory. Tried to allocate 2.00 GiB"
        },
        {
            "name": "Test Failure Fix", 
            "repo": "demo/ml-pipeline",
            "event_type": "workflow_run",
            "error": "ImportError: No module named 'torch'"
        },
        {
            "name": "Data Drift Detection",
            "repo": "demo/production-model",
            "event_type": "schedule",
            "error": None
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        click.echo(f"\n🎬 Scenario {i}: {scenario['name']}")
        click.echo("-" * 30)
        
        # Run the scenario
        bot = SelfHealingBot()
        
        # Create event payload
        repo_owner, repo_name = scenario["repo"].split("/")
        event_payload = {
            "repository": {
                "full_name": scenario["repo"],
                "name": repo_name,
                "owner": {"login": repo_owner}
            }
        }
        
        if scenario["error"]:
            event_payload["workflow_run"] = {
                "name": "ML Training",
                "conclusion": "failure",
                "id": 12345 + i,
                "html_url": f"https://github.com/{scenario['repo']}/actions/runs/{12345 + i}"
            }
        
        async def run_demo_scenario():
            try:
                context = await bot.process_event(scenario["event_type"], event_payload)
                
                if scenario["error"]:
                    context.set_error("DemoError", scenario["error"])
                
                if context:
                    click.echo(f"  ✅ Execution ID: {context.execution_id}")
                    click.echo(f"  🎯 Repository: {scenario['repo']}")
                    click.echo(f"  📅 Event type: {scenario['event_type']}")
                    
                    if context.has_error():
                        click.echo(f"  🐛 Error simulated: {context.error_message}")
                    
                    file_changes = context.get_file_changes()
                    if file_changes:
                        click.echo(f"  📝 Files changed: {len(file_changes)}")
                        for file_path in file_changes.keys():
                            click.echo(f"      • {file_path}")
                    else:
                        click.echo(f"  📝 No file changes")
                else:
                    click.echo("  ❌ Failed to create context")
                        
            except Exception as e:
                click.echo(f"  ❌ Error: {e}")
        
        asyncio.run(run_demo_scenario())
    
    click.echo(f"\n🎉 Demo completed! Try running individual scenarios with:")
    click.echo(f"  self-healing-bot trigger demo/ml-training --event-type workflow_run --with-error 'CUDA out of memory'")


# Add these to the main CLI
def extend_cli(cli_group):
    """Add extended commands to the CLI group."""
    cli_group.add_command(trigger)
    cli_group.add_command(health)
    cli_group.add_command(demo)