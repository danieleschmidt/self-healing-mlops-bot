"""Tests for detector functionality."""

import pytest
from unittest.mock import Mock, patch
import numpy as np

from self_healing_bot.detectors.pipeline_failure import PipelineFailureDetector, ErrorPatternDetector
from self_healing_bot.detectors.data_drift import DataDriftDetector  
from self_healing_bot.detectors.model_degradation import ModelDegradationDetector
from self_healing_bot.detectors.registry import DetectorRegistry
from self_healing_bot.core.context import Context


class TestPipelineFailureDetector:
    """Test cases for PipelineFailureDetector."""
    
    def test_supported_events(self):
        """Test supported event types."""
        detector = PipelineFailureDetector()
        events = detector.get_supported_events()
        
        assert "workflow_run" in events
        assert "check_run" in events
        assert "status" in events
    
    @pytest.mark.asyncio
    async def test_detect_workflow_failure(self, mock_context):
        """Test detection of workflow failures."""
        detector = PipelineFailureDetector()
        
        # Setup workflow failure context
        mock_context.event_type = "workflow_run"
        mock_context.event_data = {
            "workflow_run": {
                "id": 123,
                "name": "CI Tests",
                "conclusion": "failure",
                "html_url": "https://github.com/test/repo/actions/runs/123",
                "head_sha": "abc123"
            }
        }
        
        issues = await detector.detect(mock_context)
        
        assert len(issues) == 1
        issue = issues[0]
        assert issue["type"] == "workflow_failure_test_failure"
        assert issue["severity"] == "high"
        assert "CI Tests" in issue["message"]
        assert issue["data"]["workflow_id"] == 123
        assert issue["data"]["failure_type"] == "test_failure"
    
    @pytest.mark.asyncio
    async def test_detect_workflow_success(self, mock_context):
        """Test no detection on successful workflow."""
        detector = PipelineFailureDetector()
        
        mock_context.event_type = "workflow_run"
        mock_context.event_data = {
            "workflow_run": {
                "conclusion": "success"
            }
        }
        
        issues = await detector.detect(mock_context)
        
        assert len(issues) == 0
    
    @pytest.mark.asyncio
    async def test_detect_check_run_failure(self, mock_context):
        """Test detection of check run failures."""
        detector = PipelineFailureDetector()
        
        mock_context.event_type = "check_run"
        mock_context.event_data = {
            "check_run": {
                "id": 456,
                "name": "Unit Tests",
                "conclusion": "failure",
                "details_url": "https://github.com/test/repo/runs/456"
            }
        }
        
        issues = await detector.detect(mock_context)
        
        assert len(issues) == 1
        issue = issues[0]
        assert issue["type"] == "check_failure"
        assert issue["severity"] == "medium"
        assert "Unit Tests" in issue["message"]
    
    def test_categorize_workflow_failure(self):
        """Test workflow failure categorization."""
        detector = PipelineFailureDetector()
        
        # Test training failure
        context = Mock()
        workflow_data = {"name": "ML Training Pipeline"}
        result = detector._categorize_workflow_failure(context, workflow_data)
        assert result == "training_failure"
        
        # Test deployment failure
        workflow_data = {"name": "Deploy to Production"}
        result = detector._categorize_workflow_failure(context, workflow_data)
        assert result == "deployment_failure"
        
        # Test unknown failure
        workflow_data = {"name": "Custom Workflow"}
        result = detector._categorize_workflow_failure(context, workflow_data)
        assert result == "unknown_failure"
    
    def test_get_failure_severity(self):
        """Test failure severity assignment."""
        detector = PipelineFailureDetector()
        
        assert detector._get_failure_severity("deployment_failure") == "critical"
        assert detector._get_failure_severity("training_failure") == "high"
        assert detector._get_failure_severity("test_failure") == "high"
        assert detector._get_failure_severity("code_quality_failure") == "medium"
        assert detector._get_failure_severity("unknown_failure") == "medium"


class TestErrorPatternDetector:
    """Test cases for ErrorPatternDetector."""
    
    def test_detect_gpu_oom_pattern(self):
        """Test GPU OOM pattern detection."""
        log_content = "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB"
        patterns = ErrorPatternDetector.detect_patterns(log_content)
        
        assert "gpu_oom" in patterns
    
    def test_detect_import_error_pattern(self):
        """Test import error pattern detection."""
        log_content = "ModuleNotFoundError: No module named 'torch'"
        patterns = ErrorPatternDetector.detect_patterns(log_content)
        
        assert "import_error" in patterns
    
    def test_detect_dependency_error_pattern(self):
        """Test dependency error pattern detection."""
        log_content = "Could not find a version that satisfies the requirement torch==2.0.0"
        patterns = ErrorPatternDetector.detect_patterns(log_content)
        
        assert "dependency_error" in patterns
    
    def test_detect_multiple_patterns(self):
        """Test detection of multiple error patterns."""
        log_content = """
        ImportError: cannot import name 'torch'
        RuntimeError: CUDA out of memory
        TimeoutError: Request timeout after 30 seconds
        """
        patterns = ErrorPatternDetector.detect_patterns(log_content)
        
        assert "import_error" in patterns
        assert "gpu_oom" in patterns
        assert "timeout_error" in patterns
    
    def test_no_patterns_detected(self):
        """Test when no patterns are detected."""
        log_content = "Everything is working fine!"
        patterns = ErrorPatternDetector.detect_patterns(log_content)
        
        assert len(patterns) == 0


class TestDataDriftDetector:
    """Test cases for DataDriftDetector."""
    
    def test_supported_events(self):
        """Test supported event types."""
        detector = DataDriftDetector()
        events = detector.get_supported_events()
        
        assert "push" in events
        assert "schedule" in events
        assert "workflow_run" in events
    
    @pytest.mark.asyncio
    async def test_detect_data_drift(self, mock_context):
        """Test data drift detection."""
        detector = DataDriftDetector(config={"drift_threshold": 0.05})
        
        issues = await detector.detect(mock_context)
        
        # Should detect drift in feature_3 (significant drift in mock data)
        drift_issues = [issue for issue in issues if issue["type"] == "data_drift"]
        assert len(drift_issues) > 0
        
        # Check issue structure
        if drift_issues:
            issue = drift_issues[0]
            assert "feature_name" in issue["data"]
            assert "drift_score" in issue["data"]
            assert "test_method" in issue["data"]
            assert issue["severity"] in ["low", "medium", "high", "critical"]
    
    def test_calculate_psi(self):
        """Test Population Stability Index calculation."""
        detector = DataDriftDetector()
        
        # Test with identical distributions
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(0, 1, 1000)
        psi = detector._calculate_psi(expected, actual)
        
        # PSI should be low for similar distributions
        assert psi < 0.1
        
        # Test with different distributions
        expected = np.random.normal(0, 1, 1000)
        actual = np.random.normal(2, 1, 1000)  # Shifted distribution
        psi = detector._calculate_psi(expected, actual)
        
        # PSI should be higher for different distributions
        assert psi > 0.1
    
    def test_calculate_js_divergence(self):
        """Test Jensen-Shannon divergence calculation."""
        detector = DataDriftDetector()
        
        # Test with identical distributions
        p = np.random.normal(0, 1, 1000)
        q = np.random.normal(0, 1, 1000)
        js_div = detector._calculate_js_divergence(p, q)
        
        # JS divergence should be low for similar distributions
        assert js_div < 0.1
        
        # Test with different distributions
        p = np.random.normal(0, 1, 1000)
        q = np.random.exponential(1, 1000)  # Different distribution
        js_div = detector._calculate_js_divergence(p, q)
        
        # JS divergence should be higher for different distributions
        assert js_div > 0.1
    
    def test_get_drift_severity(self):
        """Test drift severity classification."""
        detector = DataDriftDetector()
        
        assert detector._get_drift_severity(0.6) == "critical"
        assert detector._get_drift_severity(0.3) == "high"
        assert detector._get_drift_severity(0.15) == "medium"
        assert detector._get_drift_severity(0.05) == "low"
    
    def test_get_drift_recommendation(self):
        """Test drift recommendation generation."""
        detector = DataDriftDetector()
        
        rec_critical = detector._get_drift_recommendation(0.6)
        assert "Immediate retraining required" in rec_critical
        
        rec_high = detector._get_drift_recommendation(0.3)
        assert "Schedule retraining within 24 hours" in rec_high
        
        rec_medium = detector._get_drift_recommendation(0.15)
        assert "Monitor closely" in rec_medium
        
        rec_low = detector._get_drift_recommendation(0.05)
        assert "Continue monitoring" in rec_low


class TestModelDegradationDetector:
    """Test cases for ModelDegradationDetector."""
    
    def test_supported_events(self):
        """Test supported event types."""
        detector = ModelDegradationDetector()
        events = detector.get_supported_events()
        
        assert "schedule" in events
        assert "workflow_run" in events
        assert "push" in events
    
    @pytest.mark.asyncio
    async def test_detect_model_degradation(self, mock_context):
        """Test model performance degradation detection."""
        detector = ModelDegradationDetector(config={"performance_threshold": 0.05})
        
        issues = await detector.detect(mock_context)
        
        # Should detect degradation in several metrics based on mock data
        degradation_issues = [issue for issue in issues if issue["type"] == "model_degradation"]
        assert len(degradation_issues) > 0
        
        # Check issue structure
        if degradation_issues:
            issue = degradation_issues[0]
            assert "metric_name" in issue["data"]
            assert "current_value" in issue["data"]
            assert "baseline_value" in issue["data"]
            assert "degradation_percentage" in issue["data"]
            assert issue["severity"] in ["low", "medium", "high", "critical"]
    
    @pytest.mark.asyncio
    async def test_compare_metrics(self, mock_context):
        """Test metric comparison logic."""
        detector = ModelDegradationDetector()
        
        current = {
            "accuracy": 0.85,
            "latency_p95": 250.0
        }
        
        baseline = {
            "accuracy": 0.90,
            "latency_p95": 200.0
        }
        
        results = await detector._compare_metrics(current, baseline)
        
        # Accuracy degraded (lower is worse)
        assert results["accuracy"]["degradation_detected"] is True
        assert results["accuracy"]["degradation_percentage"] > 5.0
        
        # Latency degraded (higher is worse)
        assert results["latency_p95"]["degradation_detected"] is True
        assert results["latency_p95"]["degradation_percentage"] > 5.0
    
    def test_get_degradation_severity(self):
        """Test degradation severity classification."""
        detector = ModelDegradationDetector()
        
        assert detector._get_degradation_severity(25.0) == "critical"
        assert detector._get_degradation_severity(15.0) == "high"
        assert detector._get_degradation_severity(7.0) == "medium"
        assert detector._get_degradation_severity(3.0) == "low"
    
    def test_get_degradation_recommendation(self):
        """Test degradation recommendation generation."""
        detector = ModelDegradationDetector()
        
        rec_critical = detector._get_degradation_recommendation(25.0)
        assert "Immediate rollback" in rec_critical
        
        rec_high = detector._get_degradation_recommendation(15.0)
        assert "Consider rollback" in rec_high
        
        rec_medium = detector._get_degradation_recommendation(7.0)
        assert "Monitor closely" in rec_medium
        
        rec_low = detector._get_degradation_recommendation(3.0)
        assert "Continue monitoring" in rec_low


class TestDetectorRegistry:
    """Test cases for DetectorRegistry."""
    
    def test_detector_registration(self):
        """Test detector registration."""
        registry = DetectorRegistry()
        
        # Should have built-in detectors
        detectors = registry.list_detectors()
        assert "pipeline_failure" in detectors
        assert "data_drift" in detectors
        assert "model_degradation" in detectors
    
    def test_get_detector(self):
        """Test getting detector by name."""
        registry = DetectorRegistry()
        
        detector = registry.get_detector("pipeline_failure")
        assert detector is not None
        assert isinstance(detector, PipelineFailureDetector)
    
    def test_get_detectors_for_event(self):
        """Test getting detectors for specific event types."""
        registry = DetectorRegistry()
        
        # Workflow run events
        detectors = registry.get_detectors_for_event("workflow_run")
        assert len(detectors) >= 1
        
        # Should include pipeline failure detector
        detector_types = [type(d).__name__ for d in detectors]
        assert "PipelineFailureDetector" in detector_types
    
    def test_get_nonexistent_detector(self):
        """Test getting non-existent detector."""
        registry = DetectorRegistry()
        
        detector = registry.get_detector("nonexistent")
        assert detector is None
    
    def test_get_detectors_for_unsupported_event(self):
        """Test getting detectors for unsupported event type."""
        registry = DetectorRegistry()
        
        detectors = registry.get_detectors_for_event("unsupported_event")
        assert len(detectors) == 0