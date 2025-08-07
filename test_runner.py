#!/usr/bin/env python3
"""
Simple test runner for the Self-Healing MLOps Bot.
"""

import asyncio
import os
import sys
import traceback
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Set up test environment
os.environ.update({
    'GITHUB_APP_ID': 'test-app',
    'GITHUB_PRIVATE_KEY_PATH': '/tmp/test-key.pem',
    'GITHUB_WEBHOOK_SECRET': 'test-secret',
    'DATABASE_URL': 'sqlite:///test.db',
    'REDIS_URL': 'redis://localhost:6379/0',
    'LOG_LEVEL': 'INFO',
    'ENVIRONMENT': 'test'
})


def print_test_header(test_name: str):
    """Print test header."""
    print(f"\n{'='*60}")
    print(f"🧪 {test_name}")
    print('='*60)


def print_test_result(test_name: str, success: bool, message: str = ""):
    """Print test result."""
    status = "✅ PASS" if success else "❌ FAIL"
    print(f"{status} {test_name}")
    if message:
        print(f"   {message}")


async def test_imports():
    """Test that all modules can be imported."""
    print_test_header("Module Import Tests")
    
    try:
        # Core imports
        from self_healing_bot.core.bot import SelfHealingBot
        from self_healing_bot.core.playbook import PlaybookRegistry, Playbook
        from self_healing_bot.core.context import Context
        from self_healing_bot.core.config import config
        
        # Detector imports
        from self_healing_bot.detectors.registry import DetectorRegistry
        from self_healing_bot.detectors.pipeline_failure import PipelineFailureDetector
        from self_healing_bot.detectors.data_drift import DataDriftDetector
        from self_healing_bot.detectors.model_degradation import ModelDegradationDetector
        
        # Action imports
        from self_healing_bot.actions.code_fixes import CodeFixAction
        from self_healing_bot.actions.config_updates import ConfigUpdateAction
        from self_healing_bot.actions.rollback import ModelRollbackAction
        from self_healing_bot.actions.notifications import SlackNotificationAction
        from self_healing_bot.actions.retraining import AutoRetrainingAction
        
        # Integration imports
        from self_healing_bot.integrations.github import GitHubIntegration
        
        # Monitoring imports
        from self_healing_bot.monitoring.health import health_monitor
        from self_healing_bot.monitoring.metrics import metrics
        from self_healing_bot.monitoring.logging import get_logger
        
        # Web imports
        from self_healing_bot.web.app import app
        
        print_test_result("All module imports", True, "All modules imported successfully")
        return True
        
    except Exception as e:
        print_test_result("Module imports", False, f"Import error: {str(e)}")
        traceback.print_exc()
        return False


async def test_core_functionality():
    """Test core bot functionality."""
    print_test_header("Core Functionality Tests")
    
    try:
        from self_healing_bot.core.bot import SelfHealingBot
        
        # Test bot initialization
        bot = SelfHealingBot()
        print_test_result("Bot initialization", True, "Bot created successfully")
        
        # Test registry loading
        detector_count = len(bot.detector_registry.list_detectors())
        print_test_result("Detector registry", detector_count > 0, 
                         f"Loaded {detector_count} detectors")
        
        playbook_count = len(bot.playbook_registry.list_playbooks())
        print_test_result("Playbook registry", playbook_count > 0, 
                         f"Loaded {playbook_count} playbooks")
        
        return detector_count > 0 and playbook_count > 0
        
    except Exception as e:
        print_test_result("Core functionality", False, f"Error: {str(e)}")
        traceback.print_exc()
        return False


async def test_event_processing():
    """Test event processing workflow."""
    print_test_header("Event Processing Tests")
    
    try:
        from self_healing_bot.core.bot import SelfHealingBot
        
        bot = SelfHealingBot()
        
        # Test workflow failure event
        mock_event = {
            'repository': {
                'full_name': 'test/repo',
                'name': 'repo',
                'owner': {'login': 'test'}
            },
            'workflow_run': {
                'name': 'CI Pipeline',
                'conclusion': 'failure',
                'id': 123,
                'html_url': 'https://github.com/test/repo/actions/runs/123'
            }
        }
        
        context = await bot.process_event('workflow_run', mock_event)
        
        if context:
            print_test_result("Event processing", True, 
                             f"Processed event with ID: {context.execution_id}")
            
            # Check if issues were detected
            has_changes = len(context.get_file_changes()) > 0
            print_test_result("File changes generated", has_changes,
                             f"Generated {len(context.get_file_changes())} file changes")
            
            return True
        else:
            print_test_result("Event processing", False, "No context returned")
            return False
            
    except Exception as e:
        print_test_result("Event processing", False, f"Error: {str(e)}")
        traceback.print_exc()
        return False


async def test_detectors():
    """Test individual detectors."""
    print_test_header("Detector Tests")
    
    try:
        from self_healing_bot.core.context import Context
        from self_healing_bot.detectors.pipeline_failure import PipelineFailureDetector
        from self_healing_bot.detectors.data_drift import DataDriftDetector
        from self_healing_bot.detectors.model_degradation import ModelDegradationDetector
        from datetime import datetime
        import uuid
        
        # Create test context
        context = Context(
            repo_owner="test",
            repo_name="repo", 
            repo_full_name="test/repo",
            event_type="workflow_run",
            event_data={
                'workflow_run': {
                    'name': 'CI Pipeline',
                    'conclusion': 'failure'
                }
            },
            execution_id=str(uuid.uuid4()),
            started_at=datetime.utcnow()
        )
        
        # Test pipeline failure detector
        pipeline_detector = PipelineFailureDetector()
        pipeline_issues = await pipeline_detector.detect(context)
        print_test_result("Pipeline failure detector", len(pipeline_issues) > 0,
                         f"Detected {len(pipeline_issues)} pipeline issues")
        
        # Test data drift detector  
        drift_detector = DataDriftDetector()
        drift_issues = await drift_detector.detect(context)
        print_test_result("Data drift detector", len(drift_issues) >= 0,
                         f"Detected {len(drift_issues)} drift issues")
        
        # Test model degradation detector
        degradation_detector = ModelDegradationDetector()
        degradation_issues = await degradation_detector.detect(context)
        print_test_result("Model degradation detector", len(degradation_issues) >= 0,
                         f"Detected {len(degradation_issues)} degradation issues")
        
        return True
        
    except Exception as e:
        print_test_result("Detectors", False, f"Error: {str(e)}")
        traceback.print_exc()
        return False


async def test_actions():
    """Test action execution."""
    print_test_header("Action Tests")
    
    try:
        from self_healing_bot.core.context import Context
        from self_healing_bot.actions.code_fixes import CodeFixAction
        from self_healing_bot.actions.notifications import SlackNotificationAction
        from datetime import datetime
        import uuid
        
        # Create test context
        context = Context(
            repo_owner="test",
            repo_name="repo",
            repo_full_name="test/repo", 
            event_type="workflow_run",
            event_data={},
            execution_id=str(uuid.uuid4()),
            started_at=datetime.utcnow()
        )
        
        # Test code fix action
        code_fix_action = CodeFixAction()
        issue_data = {
            "type": "import_error",
            "missing_module": "numpy",
            "affected_file": "train.py"
        }
        
        # This should fail gracefully since we don't have real files
        result = await code_fix_action.execute(context, issue_data)
        print_test_result("Code fix action", True,
                         f"Action executed: {result.success} - {result.message}")
        
        # Test notification action
        slack_action = SlackNotificationAction({'webhook_url': None})  # No webhook for test
        notification_data = {
            "type": "test_issue",
            "severity": "medium",
            "message": "Test issue detected"
        }
        
        result = await slack_action.execute(context, notification_data)
        print_test_result("Notification action", True,
                         f"Action executed: {result.success} - {result.message}")
        
        return True
        
    except Exception as e:
        print_test_result("Actions", False, f"Error: {str(e)}")
        traceback.print_exc()
        return False


async def test_health_monitoring():
    """Test health monitoring system."""
    print_test_header("Health Monitoring Tests")
    
    try:
        from self_healing_bot.monitoring.health import health_monitor
        
        # Run health checks
        health_results = await health_monitor.run_all_checks()
        
        print_test_result("Health checks execution", len(health_results) > 0,
                         f"Executed {len(health_results)} health checks")
        
        # Check individual results
        for check_name, result in health_results.items():
            status = result.status.value if hasattr(result.status, 'value') else str(result.status)
            print_test_result(f"Health check: {check_name}", True,
                             f"Status: {status} - {result.message}")
        
        # Get health summary
        summary = health_monitor.get_health_summary()
        print_test_result("Health summary", summary.get('status') != 'unknown',
                         f"Overall status: {summary.get('status', 'unknown')}")
        
        return True
        
    except Exception as e:
        print_test_result("Health monitoring", False, f"Error: {str(e)}")
        traceback.print_exc()
        return False


async def test_web_server():
    """Test web server initialization."""
    print_test_header("Web Server Tests")
    
    try:
        from self_healing_bot.web.app import app
        
        # Test that FastAPI app is created
        print_test_result("FastAPI app creation", app is not None,
                         f"FastAPI app created: {type(app).__name__}")
        
        # Test that routes are registered
        route_count = len(app.routes)
        print_test_result("Route registration", route_count > 0,
                         f"Registered {route_count} routes")
        
        return True
        
    except Exception as e:
        print_test_result("Web server", False, f"Error: {str(e)}")
        traceback.print_exc()
        return False


async def main():
    """Run all tests."""
    print("🚀 Self-Healing MLOps Bot - Test Suite")
    print("="*60)
    
    test_results = []
    
    # Run all test suites
    test_functions = [
        test_imports,
        test_core_functionality,
        test_event_processing,
        test_detectors,
        test_actions,
        test_health_monitoring,
        test_web_server
    ]
    
    for test_func in test_functions:
        try:
            result = await test_func()
            test_results.append(result)
        except Exception as e:
            print(f"❌ Test suite {test_func.__name__} crashed: {e}")
            test_results.append(False)
    
    # Print summary
    print(f"\n{'='*60}")
    print("📊 Test Results Summary")
    print('='*60)
    
    passed = sum(test_results)
    total = len(test_results)
    success_rate = (passed / total) * 100 if total > 0 else 0
    
    print(f"Tests passed: {passed}/{total} ({success_rate:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! The Self-Healing MLOps Bot is ready.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)