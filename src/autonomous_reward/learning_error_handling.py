"""
Learning system error handling for the Autonomous State-Derived Reward System.

This module provides specialized error handling for learning-related failures,
including learning rate instability, complexity overload, and meta-learning
fallback mechanisms.
"""

print("Starting learning_error_handling import...")

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
from enum import Enum
import math

print("Basic imports successful...")

try:
    from .models import LearningState, Experience
    print("Models import successful...")
except ImportError as e:
    print(f"Models import failed: {e}")

try:
    from .error_handling import RewardSystemError, ErrorType, ErrorSeverity
    print("Error handling import successful...")
except ImportError as e:
    print(f"Error handling import failed: {e}")

logger = logging.getLogger(__name__)

print("About to define classes...")

class LearningErrorType(Enum):
    """Specific types of learning system errors."""
    LEARNING_RATE_INSTABILITY = "learning_rate_instability"
    COMPLEXITY_OVERLOAD = "complexity_overload"
    META_LEARNING_FAILURE = "meta_learning_failure"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONSOLIDATION_FAILURE = "consolidation_failure"
    ADAPTATION_OVERFLOW = "adaptation_overflow"

print("LearningErrorType defined...")

class LearningRateInstabilityError(RewardSystemError):
    """Error raised when learning rates become unstable."""
    
    def __init__(self, message: str, unstable_rate: float, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.LEARNING_RATE_INSTABILITY, ErrorSeverity.MEDIUM, context)
        self.unstable_rate = unstable_rate

print("LearningRateInstabilityError defined...")

class ComplexityOverloadError(RewardSystemError):
    """Error raised when complexity exceeds agent's current capacity."""
    
    def __init__(self, message: str, overload_level: float, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.COMPLEXITY_OVERLOAD, ErrorSeverity.HIGH, context)
        self.overload_level = overload_level

print("ComplexityOverloadError defined...")

class MetaLearningFailureError(RewardSystemError):
    """Error raised when meta-learning mechanisms fail."""
    
    def __init__(self, message: str, failure_type: str, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.META_LEARNING_FAILURE, ErrorSeverity.MEDIUM, context)
        self.failure_type = failure_type

print("MetaLearningFailureError defined...")

class LearningSystemErrorHandler:
    """
    Specialized error handler for learning system failures.
    
    Provides learning rate stability controls, complexity overload detection,
    and meta-learning fallback mechanisms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the learning system error handler.
        
        Args:
            config: Optional configuration for learning error handling
        """
        self.config = config or {}
        
        # Learning rate stability parameters
        self.min_learning_rate = self.config.get('min_learning_rate', 0.0001)
        self.max_learning_rate = self.config.get('max_learning_rate', 0.5)
        self.learning_rate_variance_threshold = self.config.get('learning_rate_variance_threshold', 0.1)
        self.learning_rate_change_limit = self.config.get('learning_rate_change_limit', 0.5)
        
        # Complexity overload parameters
        self.max_complexity_level = self.config.get('max_complexity_level', 1.0)
        self.complexity_overload_threshold = self.config.get('complexity_overload_threshold', 0.9)
        self.complexity_reduction_factor = self.config.get('complexity_reduction_factor', 0.7)
        self.performance_degradation_threshold = self.config.get('performance_degradation_threshold', 0.3)
        
        # Meta-learning fallback parameters
        self.meta_learning_timeout = self.config.get('meta_learning_timeout', 60.0)  # seconds
        self.adaptation_rate_bounds = self.config.get('adaptation_rate_bounds', (0.0001, 0.01))
        self.stability_score_bounds = self.config.get('stability_score_bounds', (0.1, 1.0))
        self.learning_efficiency_bounds = self.config.get('learning_efficiency_bounds', (0.1, 3.0))
        
        # Error tracking
        self.learning_error_history: List[RewardSystemError] = []
        self.learning_rate_history: List[float] = []
        self.complexity_history: List[float] = []
        self.performance_history: List[float] = []
        self.stability_interventions: List[Dict[str, Any]] = []
        
        # Fallback learning state
        self.fallback_learning_state = LearningState(
            learning_rate=0.01,
            complexity_level=0.3,
            readiness_score=0.5,
            consolidation_needed=False,
            meta_learning_params={
                'adaptation_rate': 0.001,
                'performance_trend': 0.0,
                'stability_score': 0.5,
                'learning_efficiency': 1.0
            }
        )
        
        logger.info("Initialized LearningSystemErrorHandler")
    
    def handle_learning_rate_instability(self, current_rate: float, 
                                       rate_history: List[float],
                                       performance_history: List[float]) -> float:
        """
        Handle learning rate instability by applying stability controls.
        
        Args:
            current_rate: Current learning rate
            rate_history: Recent learning rate history
            performance_history: Recent performance history
            
        Returns:
            Stabilized learning rate
        """
        try:
            # Check for rate bounds violation
            if current_rate < self.min_learning_rate:
                logger.warning(f"Learning rate below minimum: {current_rate} < {self.min_learning_rate}")
                
                error = LearningRateInstabilityError(
                    f"Learning rate below minimum bound: {current_rate}",
                    current_rate,
                    context={'violation_type': 'underflow', 'min_bound': self.min_learning_rate}
                )
                self._record_learning_error(error)
                
                return self.min_learning_rate
            
            if current_rate > self.max_learning_rate:
                logger.warning(f"Learning rate above maximum: {current_rate} > {self.max_learning_rate}")
                
                error = LearningRateInstabilityError(
                    f"Learning rate above maximum bound: {current_rate}",
                    current_rate,
                    context={'violation_type': 'overflow', 'max_bound': self.max_learning_rate}
                )
                self._record_learning_error(error)
                
                return self.max_learning_rate
            
            # Check for excessive variance in learning rate
            if len(rate_history) >= 5:
                rate_variance = np.var(rate_history[-5:])
                rate_mean = np.mean(rate_history[-5:])
                coefficient_of_variation = rate_variance / (rate_mean + 1e-8)
                
                if coefficient_of_variation > self.learning_rate_variance_threshold:
                    logger.warning(f"Learning rate instability detected: CV={coefficient_of_variation:.4f}")
                    
                    # Apply smoothing to stabilize
                    smoothed_rate = self._apply_learning_rate_smoothing(current_rate, rate_history)
                    
                    error = LearningRateInstabilityError(
                        f"Learning rate variance too high: CV={coefficient_of_variation:.4f}",
                        current_rate,
                        context={
                            'coefficient_of_variation': coefficient_of_variation,
                            'smoothed_rate': smoothed_rate,
                            'stabilization_applied': True
                        }
                    )
                    self._record_learning_error(error)
                    
                    return smoothed_rate
            
            # Check for excessive rate changes
            if len(rate_history) >= 2:
                previous_rate = rate_history[-1]
                rate_change = abs(current_rate - previous_rate) / (previous_rate + 1e-8)
                
                if rate_change > self.learning_rate_change_limit:
                    logger.warning(f"Excessive learning rate change: {rate_change:.4f}")
                    
                    # Limit the change
                    max_change = previous_rate * self.learning_rate_change_limit
                    if current_rate > previous_rate:
                        limited_rate = previous_rate + max_change
                    else:
                        limited_rate = previous_rate - max_change
                    
                    # Ensure within bounds
                    limited_rate = max(self.min_learning_rate, min(self.max_learning_rate, limited_rate))
                    
                    error = LearningRateInstabilityError(
                        f"Learning rate change too large: {rate_change:.4f}",
                        current_rate,
                        context={
                            'rate_change': rate_change,
                            'limited_rate': limited_rate,
                            'change_limiting_applied': True
                        }
                    )
                    self._record_learning_error(error)
                    
                    return limited_rate
            
            # Check for performance-based instability
            if len(performance_history) >= 5:
                recent_performance = performance_history[-3:]
                performance_variance = np.var(recent_performance)
                
                if performance_variance > 0.2:  # High performance variance
                    # Reduce learning rate to stabilize
                    stabilized_rate = current_rate * 0.8
                    stabilized_rate = max(self.min_learning_rate, stabilized_rate)
                    
                    logger.warning(f"Performance instability detected, reducing learning rate: "
                                 f"{current_rate:.6f} -> {stabilized_rate:.6f}")
                    
                    error = LearningRateInstabilityError(
                        f"Performance instability: variance={performance_variance:.4f}",
                        current_rate,
                        context={
                            'performance_variance': performance_variance,
                            'stabilized_rate': stabilized_rate,
                            'performance_stabilization_applied': True
                        }
                    )
                    self._record_learning_error(error)
                    
                    return stabilized_rate
            
            # Rate is stable
            return current_rate
            
        except Exception as e:
            logger.error(f"Error in learning rate stability handling: {e}")
            
            error = LearningRateInstabilityError(
                f"Critical error in learning rate stability: {str(e)}",
                current_rate,
                context={'critical_failure': True, 'exception': str(e)}
            )
            self._record_learning_error(error)
            
            # Return safe fallback rate
            return self.fallback_learning_state.learning_rate
    
    def detect_complexity_overload(self, current_complexity: float,
                                 performance_history: List[float],
                                 readiness_score: float,
                                 recent_experiences: List[Experience]) -> Tuple[bool, float]:
        """
        Detect complexity overload and suggest appropriate complexity level.
        
        Args:
            current_complexity: Current complexity level
            performance_history: Recent performance data
            readiness_score: Current readiness assessment
            recent_experiences: Recent learning experiences
            
        Returns:
            Tuple of (overload_detected, suggested_complexity)
        """
        try:
            overload_indicators = []
            
            # Indicator 1: Performance degradation
            if len(performance_history) >= 5:
                recent_performance = np.mean(performance_history[-3:])
                older_performance = np.mean(performance_history[-5:-3])
                
                if recent_performance < older_performance * (1 - self.performance_degradation_threshold):
                    performance_degradation = (older_performance - recent_performance) / older_performance
                    overload_indicators.append(('performance_degradation', performance_degradation))
                    logger.warning(f"Performance degradation detected: {performance_degradation:.4f}")
            
            # Indicator 2: Readiness score vs complexity mismatch
            if current_complexity > readiness_score + 0.2:  # Complexity significantly exceeds readiness
                readiness_mismatch = current_complexity - readiness_score
                overload_indicators.append(('readiness_mismatch', readiness_mismatch))
                logger.warning(f"Complexity-readiness mismatch: {readiness_mismatch:.4f}")
            
            # Indicator 3: High complexity level
            if current_complexity > self.complexity_overload_threshold:
                overload_indicators.append(('high_complexity', current_complexity))
                logger.warning(f"High complexity level: {current_complexity:.4f}")
            
            # Determine if overload exists
            overload_detected = len(overload_indicators) >= 2  # Multiple indicators suggest overload
            
            if overload_detected:
                # Calculate suggested complexity reduction
                suggested_complexity = current_complexity * self.complexity_reduction_factor
                suggested_complexity = max(0.1, min(suggested_complexity, self.max_complexity_level))
                
                error = ComplexityOverloadError(
                    f"Complexity overload detected with {len(overload_indicators)} indicators",
                    current_complexity,
                    context={
                        'overload_indicators': overload_indicators,
                        'suggested_complexity': suggested_complexity,
                        'readiness_score': readiness_score
                    }
                )
                self._record_learning_error(error)
                
                return True, suggested_complexity
            else:
                return False, current_complexity
                
        except Exception as e:
            logger.error(f"Error in complexity overload detection: {e}")
            
            error = ComplexityOverloadError(
                f"Critical error in complexity overload detection: {str(e)}",
                current_complexity,
                context={'critical_failure': True, 'exception': str(e)}
            )
            self._record_learning_error(error)
            
            # Return safe fallback
            return True, self.fallback_learning_state.complexity_level
    
    def handle_meta_learning_failure(self, failure_type: str, 
                                   meta_params: Dict[str, float],
                                   context: Dict[str, Any]) -> Dict[str, float]:
        """
        Handle meta-learning system failures with fallback mechanisms.
        
        Args:
            failure_type: Type of meta-learning failure
            meta_params: Current meta-learning parameters
            context: Additional context about the failure
            
        Returns:
            Fallback meta-learning parameters
        """
        try:
            logger.warning(f"Meta-learning failure detected: {failure_type}")
            
            # Create fallback parameters based on failure type
            fallback_params = self.fallback_learning_state.meta_learning_params.copy()
            
            if failure_type == "adaptation_overflow":
                # Reduce adaptation rates
                fallback_params['adaptation_rate'] = min(
                    fallback_params.get('adaptation_rate', 0.001),
                    self.adaptation_rate_bounds[1]
                )
            elif failure_type == "stability_collapse":
                # Reset to stable values
                fallback_params['stability_score'] = self.stability_score_bounds[0]
                fallback_params['learning_efficiency'] = 1.0
            elif failure_type == "efficiency_degradation":
                # Reset efficiency parameters
                fallback_params['learning_efficiency'] = self.learning_efficiency_bounds[0]
                fallback_params['performance_trend'] = 0.0
            else:
                # General failure - use all fallback values
                fallback_params = self.fallback_learning_state.meta_learning_params.copy()
            
            # Ensure all parameters are within bounds
            fallback_params['adaptation_rate'] = max(
                self.adaptation_rate_bounds[0],
                min(self.adaptation_rate_bounds[1], fallback_params.get('adaptation_rate', 0.001))
            )
            fallback_params['stability_score'] = max(
                self.stability_score_bounds[0],
                min(self.stability_score_bounds[1], fallback_params.get('stability_score', 0.5))
            )
            fallback_params['learning_efficiency'] = max(
                self.learning_efficiency_bounds[0],
                min(self.learning_efficiency_bounds[1], fallback_params.get('learning_efficiency', 1.0))
            )
            
            error = MetaLearningFailureError(
                f"Meta-learning failure: {failure_type}",
                failure_type,
                context={
                    'original_params': meta_params,
                    'fallback_params': fallback_params,
                    'failure_context': context
                }
            )
            self._record_learning_error(error)
            
            return fallback_params
            
        except Exception as e:
            logger.error(f"Critical error in meta-learning failure handling: {e}")
            
            error = MetaLearningFailureError(
                f"Critical meta-learning failure handling error: {str(e)}",
                failure_type,
                context={
                    'critical_failure': True,
                    'exception': str(e),
                    'original_params': meta_params
                }
            )
            self._record_learning_error(error)
            
            # Return safe fallback
            return self.fallback_learning_state.meta_learning_params.copy()
    
    def get_learning_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive learning error statistics.
        
        Returns:
            Dictionary containing learning error statistics
        """
        total_errors = len(self.learning_error_history)
        
        # Count errors by type
        error_type_counts = {}
        for error in self.learning_error_history:
            error_type = error.error_type.value if hasattr(error.error_type, 'value') else str(error.error_type)
            error_type_counts[error_type] = error_type_counts.get(error_type, 0) + 1
        
        # Recent errors (last hour)
        recent_errors = [
            e for e in self.learning_error_history 
            if (datetime.now() - e.timestamp).total_seconds() < 3600
        ]
        
        return {
            'total_learning_errors': total_errors,
            'error_type_counts': error_type_counts,
            'recent_learning_errors': len(recent_errors),
            'learning_rate_history_length': len(self.learning_rate_history),
            'complexity_history_length': len(self.complexity_history),
            'performance_history_length': len(self.performance_history),
            'stability_interventions': len(self.stability_interventions)
        }
    
    def reset_learning_error_tracking(self) -> None:
        """Reset learning error tracking for fresh start."""
        self.learning_error_history.clear()
        self.learning_rate_history.clear()
        self.complexity_history.clear()
        self.performance_history.clear()
        self.stability_interventions.clear()
        logger.info("Reset learning error tracking")
    
    # Private helper methods
    
    def _apply_learning_rate_smoothing(self, current_rate: float, 
                                     rate_history: List[float]) -> float:
        """Apply smoothing to stabilize learning rate."""
        if len(rate_history) < 3:
            return current_rate
        
        # Use exponential moving average for smoothing
        weights = np.array([0.5, 0.3, 0.2])  # More weight on recent values
        recent_rates = np.array(rate_history[-3:])
        smoothed_rate = np.average(recent_rates, weights=weights)
        
        # Blend with current rate
        blended_rate = 0.7 * smoothed_rate + 0.3 * current_rate
        
        # Ensure within bounds
        return max(self.min_learning_rate, min(self.max_learning_rate, blended_rate))
    
    def _record_learning_error(self, error: RewardSystemError) -> None:
        """Record a learning error in the tracking system."""
        self.learning_error_history.append(error)
        
        # Limit error history size
        if len(self.learning_error_history) > 500:
            self.learning_error_history = self.learning_error_history[-250:]  # Keep last 250
        
        # Log error based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL LEARNING ERROR: {error}")
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY LEARNING ERROR: {error}")
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY LEARNING ERROR: {error}")
        else:
            logger.info(f"LOW SEVERITY LEARNING ERROR: {error}")