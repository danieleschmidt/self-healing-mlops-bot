"""Registry for managing detector instances."""

from typing import Dict, List, Type, Optional
import logging

from .base import BaseDetector

logger = logging.getLogger(__name__)


class DetectorRegistry:
    """Registry for managing detector instances."""
    
    def __init__(self):
        self._detectors: Dict[str, BaseDetector] = {}
        self._event_mapping: Dict[str, List[str]] = {}
        self._initialize_builtin_detectors()
    
    def register_detector(self, name: str, detector: BaseDetector) -> None:
        """Register a detector instance."""
        if not detector.is_enabled():
            logger.info(f"Detector {name} is disabled, skipping registration")
            return
        
        self._detectors[name] = detector
        
        # Update event mapping
        for event_type in detector.get_supported_events():
            if event_type not in self._event_mapping:
                self._event_mapping[event_type] = []
            self._event_mapping[event_type].append(name)
        
        logger.info(f"Registered detector: {name}")
    
    def get_detector(self, name: str) -> Optional[BaseDetector]:
        """Get detector by name."""
        return self._detectors.get(name)
    
    def get_detectors_for_event(self, event_type: str) -> List[BaseDetector]:
        """Get all detectors that support the given event type."""
        detector_names = self._event_mapping.get(event_type, [])
        return [self._detectors[name] for name in detector_names if name in self._detectors]
    
    def list_detectors(self) -> List[str]:
        """List all registered detector names."""
        return list(self._detectors.keys())
    
    def _initialize_builtin_detectors(self) -> None:
        """Initialize built-in detectors."""
        from .pipeline_failure import PipelineFailureDetector
        from .data_drift import DataDriftDetector
        from .model_degradation import ModelDegradationDetector
        from .resource_constraint import ResourceConstraintDetector
        from .security_vulnerability import SecurityVulnerabilityDetector
        from .dependency_conflict import DependencyConflictDetector
        from .infrastructure_health import InfrastructureHealthDetector
        
        # Register built-in detectors
        self.register_detector("pipeline_failure", PipelineFailureDetector())
        self.register_detector("data_drift", DataDriftDetector())
        self.register_detector("model_degradation", ModelDegradationDetector())
        self.register_detector("resource_constraint", ResourceConstraintDetector())
        self.register_detector("security_vulnerability", SecurityVulnerabilityDetector())
        self.register_detector("dependency_conflict", DependencyConflictDetector())
        self.register_detector("infrastructure_health", InfrastructureHealthDetector())