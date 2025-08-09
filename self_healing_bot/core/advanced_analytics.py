"""Advanced analytics and prediction engine for Generation 1 enhancement."""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class PredictionConfidence(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class PipelinePattern:
    """Pattern detected in pipeline behavior."""
    pattern_id: str
    pattern_type: str
    frequency: float
    confidence: PredictionConfidence
    description: str
    metadata: Dict[str, Any]
    first_seen: datetime
    last_seen: datetime


@dataclass
class FailurePrediction:
    """Prediction of potential pipeline failure."""
    prediction_id: str
    repo_full_name: str
    failure_type: str
    probability: float
    confidence: PredictionConfidence
    predicted_failure_time: datetime
    contributing_factors: List[str]
    prevention_actions: List[str]
    created_at: datetime


@dataclass
class RepairRecommendation:
    """AI-powered repair recommendation."""
    recommendation_id: str
    issue_type: str
    recommended_action: str
    success_probability: float
    estimated_time: int  # minutes
    prerequisites: List[str]
    alternative_actions: List[str]
    created_at: datetime


class AdvancedAnalyticsEngine:
    """Advanced analytics engine for pipeline intelligence."""
    
    def __init__(self, max_history_days: int = 30):
        self.max_history_days = max_history_days
        
        # Data storage
        self._pipeline_history = defaultdict(deque)  # repo -> events
        self._failure_patterns = defaultdict(list)   # repo -> patterns
        self._repair_success_rates = defaultdict(dict)  # action_type -> success_rate
        
        # Analytics state
        self._pattern_detection_enabled = True
        self._prediction_enabled = True
        self._recommendation_cache: Dict[str, RepairRecommendation] = {}
        
        # Statistics
        self._analytics_stats = {
            "patterns_detected": 0,
            "predictions_made": 0,
            "recommendations_generated": 0,
            "accuracy_rate": 0.0
        }
    
    async def record_pipeline_event(
        self,
        repo_full_name: str,
        event_type: str,
        event_data: Dict[str, Any],
        outcome: str  # success, failure, timeout
    ):
        """Record a pipeline event for analysis."""
        try:
            event_record = {
                "timestamp": datetime.utcnow(),
                "event_type": event_type,
                "outcome": outcome,
                "data": event_data
            }
            
            # Add to history
            history = self._pipeline_history[repo_full_name]
            history.append(event_record)
            
            # Maintain history size
            cutoff_date = datetime.utcnow() - timedelta(days=self.max_history_days)
            while history and history[0]["timestamp"] < cutoff_date:
                history.popleft()
            
            # Trigger pattern detection
            if self._pattern_detection_enabled:
                await self._detect_patterns(repo_full_name)
                
        except Exception as e:
            logger.exception(f"Error recording pipeline event: {e}")
    
    async def _detect_patterns(self, repo_full_name: str):
        """Detect patterns in pipeline behavior."""
        try:
            history = self._pipeline_history[repo_full_name]
            if len(history) < 10:  # Need minimum events for pattern detection
                return
            
            patterns = []
            
            # Pattern 1: Failure frequency patterns
            failure_pattern = await self._detect_failure_frequency_pattern(history)
            if failure_pattern:
                patterns.append(failure_pattern)
            
            # Pattern 2: Time-based patterns (e.g., failures on specific days/times)
            time_pattern = await self._detect_temporal_pattern(history)
            if time_pattern:
                patterns.append(time_pattern)
            
            # Pattern 3: Error cascade patterns
            cascade_pattern = await self._detect_cascade_pattern(history)
            if cascade_pattern:
                patterns.append(cascade_pattern)
            
            # Update patterns for repository
            self._failure_patterns[repo_full_name] = patterns
            self._analytics_stats["patterns_detected"] += len(patterns)
            
            logger.info(f"Detected {len(patterns)} patterns for {repo_full_name}")
            
        except Exception as e:
            logger.exception(f"Error detecting patterns: {e}")
    
    async def _detect_failure_frequency_pattern(
        self, history: deque
    ) -> Optional[PipelinePattern]:
        """Detect failure frequency patterns."""
        try:
            recent_events = [e for e in history if 
                           e["timestamp"] > datetime.utcnow() - timedelta(days=7)]
            
            if len(recent_events) < 5:
                return None
            
            failure_count = sum(1 for e in recent_events if e["outcome"] == "failure")
            failure_rate = failure_count / len(recent_events)
            
            if failure_rate > 0.3:  # More than 30% failure rate
                confidence = (
                    PredictionConfidence.HIGH if failure_rate > 0.6 else
                    PredictionConfidence.MEDIUM if failure_rate > 0.4 else
                    PredictionConfidence.LOW
                )
                
                return PipelinePattern(
                    pattern_id=f"failure_freq_{datetime.utcnow().timestamp()}",
                    pattern_type="high_failure_rate",
                    frequency=failure_rate,
                    confidence=confidence,
                    description=f"High failure rate detected: {failure_rate:.1%}",
                    metadata={
                        "failure_count": failure_count,
                        "total_events": len(recent_events),
                        "time_period": "7_days"
                    },
                    first_seen=recent_events[0]["timestamp"],
                    last_seen=recent_events[-1]["timestamp"]
                )
        
        except Exception as e:
            logger.exception(f"Error detecting failure frequency pattern: {e}")
            return None
    
    async def _detect_temporal_pattern(
        self, history: deque
    ) -> Optional[PipelinePattern]:
        """Detect time-based patterns in failures."""
        try:
            failures = [e for e in history if e["outcome"] == "failure"]
            if len(failures) < 5:
                return None
            
            # Check for patterns by hour of day
            hour_failures = defaultdict(int)
            for failure in failures:
                hour = failure["timestamp"].hour
                hour_failures[hour] += 1
            
            if hour_failures:
                peak_hour = max(hour_failures, key=hour_failures.get)
                peak_count = hour_failures[peak_hour]
                total_failures = len(failures)
                
                # If more than 40% of failures happen in a specific hour
                if peak_count / total_failures > 0.4:
                    return PipelinePattern(
                        pattern_id=f"temporal_{datetime.utcnow().timestamp()}",
                        pattern_type="temporal_clustering",
                        frequency=peak_count / total_failures,
                        confidence=PredictionConfidence.MEDIUM,
                        description=f"Failures cluster around hour {peak_hour}:00",
                        metadata={
                            "peak_hour": peak_hour,
                            "peak_count": peak_count,
                            "hour_distribution": dict(hour_failures)
                        },
                        first_seen=failures[0]["timestamp"],
                        last_seen=failures[-1]["timestamp"]
                    )
        
        except Exception as e:
            logger.exception(f"Error detecting temporal pattern: {e}")
            return None
    
    async def _detect_cascade_pattern(
        self, history: deque
    ) -> Optional[PipelinePattern]:
        """Detect cascade failure patterns."""
        try:
            events = list(history)
            cascade_sequences = 0
            
            for i in range(len(events) - 2):
                # Look for 3+ consecutive failures within 1 hour
                if (events[i]["outcome"] == "failure" and
                    events[i+1]["outcome"] == "failure" and
                    events[i+2]["outcome"] == "failure"):
                    
                    time_diff = (events[i+2]["timestamp"] - events[i]["timestamp"]).total_seconds()
                    if time_diff <= 3600:  # Within 1 hour
                        cascade_sequences += 1
            
            if cascade_sequences > 0:
                return PipelinePattern(
                    pattern_id=f"cascade_{datetime.utcnow().timestamp()}",
                    pattern_type="failure_cascade",
                    frequency=cascade_sequences / max(len(events) - 2, 1),
                    confidence=PredictionConfidence.MEDIUM,
                    description=f"Detected {cascade_sequences} cascade failure sequences",
                    metadata={
                        "cascade_count": cascade_sequences,
                        "total_events": len(events)
                    },
                    first_seen=events[0]["timestamp"],
                    last_seen=events[-1]["timestamp"]
                )
        
        except Exception as e:
            logger.exception(f"Error detecting cascade pattern: {e}")
            return None
    
    async def predict_failure(
        self, repo_full_name: str, hours_ahead: int = 24
    ) -> Optional[FailurePrediction]:
        """Predict potential pipeline failures."""
        if not self._prediction_enabled:
            return None
        
        try:
            patterns = self._failure_patterns.get(repo_full_name, [])
            if not patterns:
                return None
            
            # Calculate failure probability based on detected patterns
            base_probability = 0.1  # 10% base probability
            
            for pattern in patterns:
                if pattern.pattern_type == "high_failure_rate":
                    base_probability += pattern.frequency * 0.5
                elif pattern.pattern_type == "temporal_clustering":
                    # Check if we're approaching the high-risk time
                    current_hour = datetime.utcnow().hour
                    peak_hour = pattern.metadata.get("peak_hour", -1)
                    
                    if abs(current_hour - peak_hour) <= 2:  # Within 2 hours
                        base_probability += 0.3
                elif pattern.pattern_type == "failure_cascade":
                    # Check recent failures
                    history = self._pipeline_history[repo_full_name]
                    recent_failures = [
                        e for e in history 
                        if (e["outcome"] == "failure" and
                            e["timestamp"] > datetime.utcnow() - timedelta(hours=1))
                    ]
                    if len(recent_failures) >= 2:
                        base_probability += 0.4
            
            # Cap probability at 95%
            probability = min(base_probability, 0.95)
            
            if probability > 0.3:  # Only predict if probability > 30%
                confidence = (
                    PredictionConfidence.HIGH if probability > 0.7 else
                    PredictionConfidence.MEDIUM if probability > 0.5 else
                    PredictionConfidence.LOW
                )
                
                contributing_factors = [p.description for p in patterns]
                
                prevention_actions = self._generate_prevention_actions(patterns)
                
                prediction = FailurePrediction(
                    prediction_id=f"pred_{datetime.utcnow().timestamp()}",
                    repo_full_name=repo_full_name,
                    failure_type="pipeline_failure",
                    probability=probability,
                    confidence=confidence,
                    predicted_failure_time=datetime.utcnow() + timedelta(hours=hours_ahead),
                    contributing_factors=contributing_factors,
                    prevention_actions=prevention_actions,
                    created_at=datetime.utcnow()
                )
                
                self._analytics_stats["predictions_made"] += 1
                logger.info(f"Generated failure prediction for {repo_full_name}: {probability:.1%} probability")
                
                return prediction
        
        except Exception as e:
            logger.exception(f"Error predicting failure: {e}")
            return None
    
    def _generate_prevention_actions(self, patterns: List[PipelinePattern]) -> List[str]:
        """Generate prevention actions based on detected patterns."""
        actions = []
        
        for pattern in patterns:
            if pattern.pattern_type == "high_failure_rate":
                actions.extend([
                    "Review recent code changes for potential issues",
                    "Check infrastructure health and resource availability",
                    "Increase monitoring frequency for early detection"
                ])
            elif pattern.pattern_type == "temporal_clustering":
                peak_hour = pattern.metadata.get("peak_hour")
                actions.extend([
                    f"Schedule maintenance outside of peak failure time (hour {peak_hour})",
                    "Implement proactive health checks before high-risk periods",
                    "Consider load balancing during peak failure times"
                ])
            elif pattern.pattern_type == "failure_cascade":
                actions.extend([
                    "Implement circuit breakers to prevent cascade failures",
                    "Add automated rollback triggers after 2 consecutive failures",
                    "Enable emergency notification for rapid response"
                ])
        
        return list(set(actions))  # Remove duplicates
    
    async def recommend_repair_action(
        self,
        issue_type: str,
        error_message: str,
        context: Dict[str, Any]
    ) -> Optional[RepairRecommendation]:
        """Generate AI-powered repair recommendations."""
        try:
            # Check cache first
            cache_key = f"{issue_type}_{hash(error_message)}"
            if cache_key in self._recommendation_cache:
                return self._recommendation_cache[cache_key]
            
            recommendation = await self._generate_repair_recommendation(
                issue_type, error_message, context
            )
            
            if recommendation:
                # Cache recommendation
                self._recommendation_cache[cache_key] = recommendation
                self._analytics_stats["recommendations_generated"] += 1
                
                # Limit cache size
                if len(self._recommendation_cache) > 100:
                    # Remove oldest entries
                    oldest_key = min(
                        self._recommendation_cache.keys(),
                        key=lambda k: self._recommendation_cache[k].created_at
                    )
                    del self._recommendation_cache[oldest_key]
            
            return recommendation
        
        except Exception as e:
            logger.exception(f"Error generating repair recommendation: {e}")
            return None
    
    async def _generate_repair_recommendation(
        self,
        issue_type: str,
        error_message: str,
        context: Dict[str, Any]
    ) -> Optional[RepairRecommendation]:
        """Generate specific repair recommendations based on issue analysis."""
        
        # Common issue patterns and their solutions
        repair_rules = {
            "ImportError": {
                "action": "Fix missing dependency",
                "success_rate": 0.85,
                "time_estimate": 5,
                "prerequisites": ["Package manager access"],
                "alternatives": ["Update requirements.txt", "Use virtual environment"]
            },
            "TimeoutError": {
                "action": "Increase timeout limits",
                "success_rate": 0.75,
                "time_estimate": 10,
                "prerequisites": ["Configuration access"],
                "alternatives": ["Optimize slow operations", "Add retry logic"]
            },
            "OutOfMemoryError": {
                "action": "Reduce batch size or increase memory allocation",
                "success_rate": 0.90,
                "time_estimate": 15,
                "prerequisites": ["Resource configuration access"],
                "alternatives": ["Enable memory optimization", "Use data streaming"]
            },
            "ConnectionError": {
                "action": "Check network connectivity and retry",
                "success_rate": 0.70,
                "time_estimate": 8,
                "prerequisites": ["Network access"],
                "alternatives": ["Use backup endpoints", "Implement exponential backoff"]
            }
        }
        
        # Determine issue type from error message
        detected_issue = None
        for pattern, info in repair_rules.items():
            if pattern.lower() in error_message.lower() or pattern.lower() in issue_type.lower():
                detected_issue = pattern
                break
        
        if not detected_issue:
            # Generic recommendation
            detected_issue = "UnknownError"
            repair_rules["UnknownError"] = {
                "action": "Analyze logs and apply standard troubleshooting",
                "success_rate": 0.60,
                "time_estimate": 20,
                "prerequisites": ["Log access"],
                "alternatives": ["Restart services", "Check system health"]
            }
        
        rule = repair_rules[detected_issue]
        
        # Adjust success probability based on historical data
        historical_rate = self._repair_success_rates.get(detected_issue, {}).get("success_rate", rule["success_rate"])
        adjusted_rate = (rule["success_rate"] + historical_rate) / 2 if historical_rate else rule["success_rate"]
        
        recommendation = RepairRecommendation(
            recommendation_id=f"rec_{datetime.utcnow().timestamp()}",
            issue_type=detected_issue,
            recommended_action=rule["action"],
            success_probability=adjusted_rate,
            estimated_time=rule["time_estimate"],
            prerequisites=rule["prerequisites"],
            alternative_actions=rule["alternatives"],
            created_at=datetime.utcnow()
        )
        
        logger.info(f"Generated repair recommendation for {issue_type}: {rule['action']}")
        
        return recommendation
    
    async def record_repair_outcome(
        self,
        action_type: str,
        success: bool,
        execution_time: int
    ):
        """Record the outcome of a repair action for learning."""
        try:
            if action_type not in self._repair_success_rates:
                self._repair_success_rates[action_type] = {
                    "total_attempts": 0,
                    "successful_attempts": 0,
                    "success_rate": 0.0,
                    "avg_execution_time": 0.0
                }
            
            stats = self._repair_success_rates[action_type]
            stats["total_attempts"] += 1
            
            if success:
                stats["successful_attempts"] += 1
            
            stats["success_rate"] = stats["successful_attempts"] / stats["total_attempts"]
            
            # Update rolling average execution time
            current_avg = stats["avg_execution_time"]
            total_attempts = stats["total_attempts"]
            stats["avg_execution_time"] = (
                (current_avg * (total_attempts - 1) + execution_time) / total_attempts
            )
            
            # Update overall accuracy
            total_successful = sum(
                s["successful_attempts"] for s in self._repair_success_rates.values()
            )
            total_attempts = sum(
                s["total_attempts"] for s in self._repair_success_rates.values()
            )
            
            if total_attempts > 0:
                self._analytics_stats["accuracy_rate"] = total_successful / total_attempts
            
        except Exception as e:
            logger.exception(f"Error recording repair outcome: {e}")
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary."""
        try:
            return {
                "statistics": self._analytics_stats.copy(),
                "tracked_repositories": len(self._pipeline_history),
                "total_patterns": sum(len(patterns) for patterns in self._failure_patterns.values()),
                "pattern_types": self._get_pattern_type_distribution(),
                "repair_success_rates": {
                    action: stats["success_rate"]
                    for action, stats in self._repair_success_rates.items()
                },
                "top_failure_patterns": self._get_top_failure_patterns(),
                "recommendations_cached": len(self._recommendation_cache),
                "prediction_accuracy": self._calculate_prediction_accuracy()
            }
        except Exception as e:
            logger.exception(f"Error generating analytics summary: {e}")
            return {"error": str(e)}
    
    def _get_pattern_type_distribution(self) -> Dict[str, int]:
        """Get distribution of pattern types across all repositories."""
        distribution = defaultdict(int)
        for patterns in self._failure_patterns.values():
            for pattern in patterns:
                distribution[pattern.pattern_type] += 1
        return dict(distribution)
    
    def _get_top_failure_patterns(self) -> List[Dict[str, Any]]:
        """Get the most common failure patterns."""
        all_patterns = []
        for repo, patterns in self._failure_patterns.items():
            for pattern in patterns:
                all_patterns.append({
                    "repo": repo,
                    "type": pattern.pattern_type,
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence.value
                })
        
        # Sort by frequency and return top 10
        all_patterns.sort(key=lambda x: x["frequency"], reverse=True)
        return all_patterns[:10]
    
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy (placeholder - would need actual validation data)."""
        # This would require tracking predictions vs actual outcomes
        # For now, return estimated accuracy based on pattern confidence
        if not self._failure_patterns:
            return 0.0
        
        total_confidence = 0.0
        pattern_count = 0
        
        for patterns in self._failure_patterns.values():
            for pattern in patterns:
                confidence_score = {
                    PredictionConfidence.LOW: 0.3,
                    PredictionConfidence.MEDIUM: 0.6,
                    PredictionConfidence.HIGH: 0.85
                }[pattern.confidence]
                total_confidence += confidence_score
                pattern_count += 1
        
        return total_confidence / pattern_count if pattern_count > 0 else 0.0