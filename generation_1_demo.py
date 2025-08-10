#!/usr/bin/env python3
"""
TERRAGON SDLC - Generation 1: MAKE IT WORK
Simple demonstration of core self-healing functionality
"""

import asyncio
import logging
from pathlib import Path

# Set up basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def demo_generation_1():
    """Demonstrate Generation 1 - Basic Working Functionality"""
    
    print("🚀 TERRAGON SDLC - Generation 1: MAKE IT WORK")
    print("=" * 60)
    
    try:
        # Import core components
        from self_healing_bot.core.bot import SelfHealingBot
        from self_healing_bot.core.context import Context
        from self_healing_bot.detectors.pipeline_failure import PipelineFailureDetector
        from self_healing_bot.actions.code_fixes import CodeFixAction
        
        print("✅ Core imports successful")
        
        # Initialize bot
        bot = SelfHealingBot()
        print("✅ Bot initialized")
        
        # Create sample context for testing
        sample_event = {
            "repository": {"full_name": "test/repo", "name": "repo"},
            "workflow_run": {
                "id": 123,
                "status": "failed",
                "conclusion": "failure",
                "html_url": "https://github.com/test/repo/actions/runs/123"
            },
            "action": "workflow_run"
        }
        
        # Process a simulated failure event
        print("\n🔍 Processing simulated pipeline failure...")
        
        # Simulate event processing (simplified for Generation 1)
        context = Context(
            repo_owner="test",
            repo_name="repo", 
            repo_full_name="test/repo",
            event_type="workflow_run",
            event_data=sample_event
        )
        print("✅ Context created")
        
        # Test detector
        detector = PipelineFailureDetector()
        issues = await detector.detect(context)
        print(f"✅ Issues detected: {len(issues) if issues else 0}")
        
        # Test action
        fixer = CodeFixAction()
        if issues:
            result = await fixer.execute(context, issues[0])
            print(f"✅ Repair attempted: {result.success}")
        
        print("\n🎉 Generation 1 Complete - Basic functionality works!")
        print("✨ Ready for Generation 2: Robustness & Error Handling")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in Generation 1: {e}")
        logger.error(f"Generation 1 failed: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = asyncio.run(demo_generation_1())
    exit(0 if success else 1)