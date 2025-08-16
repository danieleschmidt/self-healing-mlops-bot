#!/usr/bin/env python3
"""
Generation 2 Demo: MAKE IT ROBUST (Reliable Implementation)
Demonstrates enhanced error handling, monitoring, and reliability features.
"""

import asyncio
import sys
import os
import structlog

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from self_healing_bot.core.bot import SelfHealingBot
from self_healing_bot.core.error_handler import error_handler, ErrorCategory
from self_healing_bot.monitoring.metrics import metrics_collector


async def demo_enhanced_reliability():
    """Demonstrate enhanced reliability features."""
    print("🤖 Self-Healing MLOps Bot - Generation 2 Demo")
    print("=" * 60)
    print("Enhanced Features: Error Handling, Monitoring, Reliability")
    print("=" * 60)
    
    # Initialize bot
    bot = SelfHealingBot()
    print("✅ Bot initialized with enhanced error handling")
    
    # Test metrics system
    print("\n📊 Testing Metrics System:")
    metrics_collector.increment_counter("demo_events", {"type": "test"})
    metrics_collector.record_histogram("demo_duration", 1.5)
    metrics_collector.set_gauge("demo_status", 100)
    
    summary = metrics_collector.get_metrics_summary()
    print(f"   Counters: {len(summary['counters'])} metrics")
    print(f"   Histograms: {len(summary['histograms'])} metrics") 
    print(f"   Gauges: {len(summary['gauges'])} metrics")
    
    # Test error handling system
    print("\n🛡️ Testing Error Handling System:")
    
    # Simulate different types of errors
    test_errors = [
        (Exception("Network timeout"), {"source": "network"}),
        (ValueError("Invalid configuration"), {"source": "config"}),
        (ConnectionError("GitHub API unavailable"), {"source": "github"}),
    ]
    
    for error, context in test_errors:
        error_context = await error_handler.handle_error(error, context)
        print(f"   ✅ Handled {error_context.category.value} error: {error_context.error_id[:8]}")
    
    # Get error statistics
    stats = error_handler.get_error_statistics()
    print(f"   Total errors handled: {stats['total_errors']}")
    print(f"   Recovery attempts: {stats['recovery_attempts']}")
    print(f"   Recovery rate: {stats['recovery_rate']:.2%}")
    
    # Test enhanced health check
    print("\n💚 Testing Enhanced Health Check:")
    health = await bot.health_check()
    print(f"   System status: {health['status']}")
    print(f"   Active executions: {health['active_executions']}")
    print(f"   Components health:")
    for component, status in health['components'].items():
        print(f"     {component}: {status}")
    
    # Test event processing with enhanced features
    print("\n🔄 Testing Enhanced Event Processing:")
    
    event_data = {
        "action": "completed",
        "workflow_run": {
            "id": 987654321,
            "name": "Enhanced CI Pipeline",
            "head_branch": "feature/reliability",
            "head_sha": "def456abc789",
            "status": "completed",
            "conclusion": "failure",
            "html_url": "https://github.com/testowner/testrepo/actions/runs/987654321",
            "created_at": "2025-01-01T00:00:00Z",
            "updated_at": "2025-01-01T00:05:00Z"
        },
        "repository": {
            "id": 654321,
            "name": "testrepo",
            "full_name": "testowner/testrepo",
            "owner": {"login": "testowner", "id": 54321},
            "private": False,
            "html_url": "https://github.com/testowner/testrepo"
        },
        "installation": {"id": 54321}
    }
    
    # Process event with enhanced monitoring
    context = await bot.process_event("workflow_run", event_data)
    
    if context:
        print(f"   ✅ Enhanced event processed")
        print(f"   Execution ID: {context.execution_id}")
        print(f"   Repository: {context.repo_full_name}")
        print(f"   Error detected: {context.has_error()}")
        if context.has_error():
            print(f"   Error type: {context.error_type}")
    
    # Show final metrics
    print("\n📈 Final Metrics Summary:")
    final_summary = metrics_collector.get_metrics_summary()
    for metric_type, metrics in final_summary.items():
        print(f"   {metric_type.title()}:")
        for name, value in metrics.items():
            if value > 0:
                print(f"     {name}: {value}")
    
    # Test Prometheus metrics export
    print("\n🔧 Prometheus Metrics:")
    prometheus_metrics = metrics_collector.get_prometheus_metrics()
    metric_lines = prometheus_metrics.strip().split('\n')
    metric_count = len([line for line in metric_lines if not line.startswith('#') and line.strip()])
    print(f"   Exported {metric_count} metric values")
    
    print("\n🎉 Generation 2 demo completed successfully!")
    print("   Enhanced Features:")
    print("   ✅ Advanced error handling and recovery")
    print("   ✅ Comprehensive metrics collection")
    print("   ✅ Prometheus integration")
    print("   ✅ Structured logging")
    print("   ✅ Timeout protection")
    print("   ✅ Automatic retry logic")
    print("   ✅ Health monitoring")


if __name__ == "__main__":
    # Configure structured logging
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )
    
    asyncio.run(demo_enhanced_reliability())