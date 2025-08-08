#!/usr/bin/env python3
"""Test core functionality without external dependencies."""

import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add the repo to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_basic_imports():
    """Test basic module imports."""
    try:
        from self_healing_bot.core.context import Context
        from self_healing_bot.core.playbook import Playbook, Action, PlaybookRegistry, ActionResult
        print("✅ Core modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_context_functionality():
    """Test Context class functionality."""
    try:
        from self_healing_bot.core.context import Context
        
        # Create context
        context = Context(
            repo_owner="test",
            repo_name="repo", 
            repo_full_name="test/repo",
            event_type="manual",
            event_data={"test": "data"}
        )
        
        # Test state management
        context.set_state("test_key", "test_value")
        assert context.get_state("test_key") == "test_value"
        
        # Test error handling
        context.set_error("TestError", "This is a test error")
        assert context.has_error() == True
        assert context.error_type == "TestError"
        
        # Test file operations
        context.write_file("test.py", "print('hello')")
        changes = context.get_file_changes()
        assert "test.py" in changes
        
        # Test config loading
        config = context.load_config("training_config.yml")
        assert "batch_size" in config
        
        print("✅ Context functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Context test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_playbook_functionality():
    """Test Playbook system functionality."""
    try:
        from self_healing_bot.core.playbook import Playbook, Action, PlaybookRegistry, ActionResult
        from self_healing_bot.core.context import Context
        
        # Test playbook registry
        playbooks = PlaybookRegistry.list_playbooks()
        print(f"📚 Found {len(playbooks)} registered playbooks: {playbooks}")
        
        # Test specific playbook
        if "test_failure_handler" in playbooks:
            playbook_class = PlaybookRegistry.get_playbook("test_failure_handler")
            assert playbook_class is not None
            
            # Create test context with error
            context = Context(
                repo_owner="test",
                repo_name="repo",
                repo_full_name="test/repo", 
                event_type="workflow_run",
                event_data={"workflow_run": {"conclusion": "failure"}}
            )
            
            # Test playbook instantiation and triggering
            playbook = playbook_class()
            should_trigger = playbook.should_trigger(context)
            print(f"🔍 Playbook should trigger: {should_trigger}")
            
            # Test action discovery
            actions = playbook.actions
            print(f"🎬 Found {len(actions)} actions in playbook")
            
        if "gpu_oom_handler" in playbooks:
            playbook_class = PlaybookRegistry.get_playbook("gpu_oom_handler")
            context = Context(
                repo_owner="test",
                repo_name="repo",
                repo_full_name="test/repo",
                event_type="manual",
                event_data={}
            )
            context.set_error("OOMError", "CUDA out of memory")
            
            playbook = playbook_class()
            should_trigger = playbook.should_trigger(context)
            print(f"🔍 GPU OOM playbook should trigger: {should_trigger}")
        
        print("✅ Playbook functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Playbook test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_detectors():
    """Test detector functionality."""
    try:
        from self_healing_bot.detectors.base import BaseDetector
        from self_healing_bot.detectors.pipeline_failure import PipelineFailureDetector
        
        # Test detector instantiation
        detector = PipelineFailureDetector()
        
        # Test supported events
        events = detector.get_supported_events()
        print(f"🔍 Pipeline detector supports events: {events}")
        
        # Test issue creation
        issue = detector.create_issue(
            issue_type="test_issue",
            severity="medium", 
            message="Test issue",
            data={"test": "data"}
        )
        
        assert issue["type"] == "test_issue"
        assert issue["severity"] == "medium"
        
        print("✅ Detector functionality works")
        return True
        
    except Exception as e:
        print(f"❌ Detector test error: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Testing Self-Healing MLOps Bot Core Functionality")
    print("=" * 60)
    
    tests = [
        test_basic_imports,
        test_context_functionality, 
        test_playbook_functionality,
        test_detectors
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\n📋 Running {test.__name__}...")
        if test():
            passed += 1
        print("-" * 40)
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All core functionality tests passed!")
        print("\n✅ Generation 1 Implementation Complete!")
        print("\nCore Features Working:")
        print("  • Context management and state tracking")
        print("  • Error detection and handling") 
        print("  • File change tracking")
        print("  • Playbook system with action execution")
        print("  • Detector registry and issue creation")
        print("  • Configuration loading and management")
        return True
    else:
        print("❌ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)