"""
Error handling utilities for the Autonomous State-Derived Reward System.

This module provides comprehensive error handling, recovery mechanisms, and
failure isolation for all components of the autonomous reward system.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
import logging
from datetime import datetime, timedelta
from enum import Enum
import traceback
import warnings

from .models import StateReward, IntrinsicMotivation, LearningState, Experience

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels for autonomous reward system failures."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Types of errors that can occur in the autonomous reward system."""
    STATE_CORRUPTION = "state_corruption"
    REWARD_OVERFLOW = "reward_overflow"
    INTEGRATION_FAILURE = "integration_failure"
    LEARNING_RATE_INSTABILITY = "learning_rate_instability"
    COMPLEXITY_OVERLOAD = "complexity_overload"
    META_LEARNING_FAILURE = "meta_learning_failure"
    NUMERICAL_INSTABILITY = "numerical_instability"
    MEMORY_OVERFLOW = "memory_overflow"
    PATTERN_DETECTION_FAILURE = "pattern_detection_failure"


class RewardSystemError(Exception):
    """Base exception for autonomous reward system errors."""
    
    def __init__(self, message: str, error_type: ErrorType, severity: ErrorSeverity, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}
        self.timestamp = datetime.now()


class StateCorruptionError(RewardSystemError):
    """Error raised when pathos state becomes corrupted."""
    
    def __init__(self, message: str, corrupted_state: np.ndarray, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.STATE_CORRUPTION, ErrorSeverity.HIGH, context)
        self.corrupted_state = corrupted_state


class RewardOverflowError(RewardSystemError):
    """Error raised when reward values become infinite or cause numerical instability."""
    
    def __init__(self, message: str, overflow_value: float, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.REWARD_OVERFLOW, ErrorSeverity.MEDIUM, context)
        self.overflow_value = overflow_value


class IntegrationFailureError(RewardSystemError):
    """Error raised when cross-layer integration fails."""
    
    def __init__(self, message: str, failed_layers: List[str], 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message, ErrorType.INTEGRATION_FAILURE, ErrorSeverity.HIGH, context)
        self.failed_layers = failed_layers


class RewardSystemErrorHandler:
    """
    Comprehensive error handler for the autonomous reward system.
    
    Provides state corruption recovery, reward overflow protection,
    integration failure isolation, and fallback mechanisms.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the error handler.
        
        Args:
            config: Optional configuration for error handling parameters
        """
        self.config = config or {}
        
        # Error handling parameters
        self.max_reward_value = self.config.get('max_reward_value', 10.0)
        self.min_reward_value = self.config.get('min_reward_value', -10.0)
        self.state_corruption_threshold = self.config.get('state_corruption_threshold', 100.0)
        self.numerical_stability_epsilon = self.config.get('numerical_stability_epsilon', 1e-8)
        
        # Recovery parameters
        self.max_recovery_attempts = self.config.get('max_recovery_attempts', 3)
        self.recovery_timeout = self.config.get('recovery_timeout', 30.0)  # seconds
        self.fallback_state_size = self.config.get('fallback_state_size', 64)
        
        # Error tracking
        self.error_history: List[RewardSystemError] = []
        self.recovery_attempts: Dict[str, int] = {}
        self.last_recovery_time: Dict[str, datetime] = {}
        self.error_counts: Dict[ErrorType, int] = {error_type: 0 for error_type in ErrorType}
        
        # Fallback values
        self.fallback_state_reward = StateReward(
            coherence_reward=0.1,
            growth_reward=0.1,
            integration_reward=0.1,
            elegance_reward=0.1,
            emergence_reward=0.0,
            total_reward=0.4
        )
        
        self.fallback_intrinsic_motivation = IntrinsicMotivation(
            curiosity_drive=0.3,
            mastery_drive=0.3,
            autonomy_drive=0.3,
            growth_drive=0.3,
            combined_motivation=0.3
        )
        
        logger.info("Initialized RewardSystemErrorHandler")
    
    def handle_state_corruption(self, corrupted_state: np.ndarray, 
                              previous_states: List[np.ndarray]) -> np.ndarray:
        """
        Handle state corruption by attempting recovery or providing fallback state.
        
        Args:
            corrupted_state: The corrupted pathos state
            previous_states: Recent historical states for recovery
            
        Returns:
            Recovered or fallback state
            
        Raises:
            StateCorruptionError: If recovery fails and no fallback is possible
        """
        error_key = "state_corruption"
        
        try:
            # Check if state is actually corrupted
            if not self._is_state_corrupted(corrupted_state):
                return corrupted_state
            
            logger.warning(f"State corruption detected: shape={corrupted_state.shape}, "
                          f"norm={np.linalg.norm(corrupted_state):.4f}")
            
            # Attempt recovery
            recovered_state = self._attempt_state_recovery(corrupted_state, previous_states)
            
            if recovered_state is not None:
                logger.info("Successfully recovered from state corruption")
                
                # Record error even when recovery is successful
                error = StateCorruptionError(
                    "State corruption detected and successfully recovered",
                    corrupted_state,
                    context={
                        'recovery_attempted': True,
                        'recovery_successful': True,
                        'fallback_used': False,
                        'original_shape': corrupted_state.shape,
                        'original_norm': float(np.linalg.norm(corrupted_state))
                    }
                )
                self._record_error(error)
                
                return recovered_state
            
            # Recovery failed, use fallback
            fallback_state = self._generate_fallback_state(corrupted_state.shape[0])
            
            logger.warning("State recovery failed, using fallback state")
            
            # Record error
            error = StateCorruptionError(
                "State corruption detected and fallback state generated",
                corrupted_state,
                context={
                    'recovery_attempted': True,
                    'fallback_used': True,
                    'original_shape': corrupted_state.shape,
                    'original_norm': float(np.linalg.norm(corrupted_state))
                }
            )
            self._record_error(error)
            
            return fallback_state
            
        except Exception as e:
            logger.error(f"Critical error in state corruption handling: {e}")
            
            # Last resort: generate minimal fallback state
            fallback_state = np.random.normal(0, 0.1, size=self.fallback_state_size)
            
            error = StateCorruptionError(
                f"Critical state corruption recovery failure: {str(e)}",
                corrupted_state,
                context={
                    'critical_failure': True,
                    'exception': str(e),
                    'traceback': traceback.format_exc()
                }
            )
            self._record_error(error)
            
            return fallback_state
    
    def handle_reward_overflow(self, reward_value: float, 
                             reward_type: str = "unknown") -> float:
        """
        Handle reward overflow by clamping values and preventing numerical instability.
        
        Args:
            reward_value: The potentially overflowing reward value
            reward_type: Type of reward for logging purposes
            
        Returns:
            Clamped reward value within safe bounds
        """
        try:
            # Check for NaN or infinite values
            if np.isnan(reward_value) or np.isinf(reward_value):
                logger.warning(f"Invalid reward value detected: {reward_value} ({reward_type})")
                
                error = RewardOverflowError(
                    f"Invalid reward value: {reward_value}",
                    reward_value,
                    context={'reward_type': reward_type, 'invalid_type': 'nan_or_inf'}
                )
                self._record_error(error)
                
                return 0.0  # Safe fallback
            
            # Check for overflow
            if reward_value > self.max_reward_value:
                logger.warning(f"Reward overflow detected: {reward_value} > {self.max_reward_value} ({reward_type})")
                
                error = RewardOverflowError(
                    f"Reward overflow: {reward_value} exceeds maximum {self.max_reward_value}",
                    reward_value,
                    context={'reward_type': reward_type, 'overflow_type': 'positive'}
                )
                self._record_error(error)
                
                return self.max_reward_value
            
            # Check for underflow
            if reward_value < self.min_reward_value:
                logger.warning(f"Reward underflow detected: {reward_value} < {self.min_reward_value} ({reward_type})")
                
                error = RewardOverflowError(
                    f"Reward underflow: {reward_value} below minimum {self.min_reward_value}",
                    reward_value,
                    context={'reward_type': reward_type, 'overflow_type': 'negative'}
                )
                self._record_error(error)
                
                return self.min_reward_value
            
            return reward_value
            
        except Exception as e:
            logger.error(f"Error in reward overflow handling: {e}")
            
            error = RewardOverflowError(
                f"Critical error in reward overflow handling: {str(e)}",
                reward_value,
                context={
                    'reward_type': reward_type,
                    'exception': str(e),
                    'traceback': traceback.format_exc()
                }
            )
            self._record_error(error)
            
            return 0.0  # Safe fallback
    
    def handle_integration_failure(self, failed_components: List[str], 
                                 error_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle integration failure by isolating failed components and providing fallbacks.
        
        Args:
            failed_components: List of component names that failed
            error_details: Details about the integration failure
            
        Returns:
            Fallback integration results
        """
        try:
            logger.warning(f"Integration failure detected in components: {failed_components}")
            
            # Create fallback results for failed components
            fallback_results = {}
            
            for component in failed_components:
                if 'reward' in component.lower():
                    fallback_results[component] = self.fallback_state_reward
                elif 'motivation' in component.lower():
                    fallback_results[component] = self.fallback_intrinsic_motivation
                elif 'synergy' in component.lower():
                    fallback_results[component] = 0.1  # Minimal synergy
                else:
                    fallback_results[component] = None  # Component disabled
            
            # Record error
            error = IntegrationFailureError(
                f"Integration failure in components: {failed_components}",
                failed_components,
                context={
                    'error_details': error_details,
                    'fallback_results': {k: str(v) for k, v in fallback_results.items()},
                    'isolation_applied': True
                }
            )
            self._record_error(error)
            
            logger.info(f"Applied fallback results for failed components: {failed_components}")
            
            return fallback_results
            
        except Exception as e:
            logger.error(f"Critical error in integration failure handling: {e}")
            
            error = IntegrationFailureError(
                f"Critical integration failure handling error: {str(e)}",
                failed_components,
                context={
                    'critical_failure': True,
                    'exception': str(e),
                    'traceback': traceback.format_exc()
                }
            )
            self._record_error(error)
            
            # Return minimal fallback
            return {component: None for component in failed_components}
    
    def safe_execute(self, operation: Callable, *args, 
                    fallback_value: Any = None, 
                    error_type: ErrorType = ErrorType.NUMERICAL_INSTABILITY,
                    **kwargs) -> Any:
        """
        Safely execute an operation with error handling and fallback.
        
        Args:
            operation: The operation to execute
            *args: Arguments for the operation
            fallback_value: Value to return if operation fails
            error_type: Type of error for logging
            **kwargs: Keyword arguments for the operation
            
        Returns:
            Result of operation or fallback value
        """
        try:
            result = operation(*args, **kwargs)
            
            # Validate result for numerical stability
            if isinstance(result, (int, float)):
                if np.isnan(result) or np.isinf(result):
                    logger.warning(f"Operation returned invalid numerical result: {result}")
                    return fallback_value
            elif isinstance(result, np.ndarray):
                if np.any(np.isnan(result)) or np.any(np.isinf(result)):
                    logger.warning(f"Operation returned array with invalid values")
                    return fallback_value
            
            return result
            
        except Exception as e:
            logger.warning(f"Operation failed: {str(e)}, using fallback value")
            
            error = RewardSystemError(
                f"Operation failed: {str(e)}",
                error_type,
                ErrorSeverity.MEDIUM,
                context={
                    'operation': operation.__name__ if hasattr(operation, '__name__') else str(operation),
                    'args': str(args)[:200],  # Truncate for logging
                    'kwargs': str(kwargs)[:200],
                    'exception': str(e),
                    'fallback_used': True
                }
            )
            self._record_error(error)
            
            return fallback_value
    
    def validate_state_reward(self, state_reward: StateReward) -> StateReward:
        """
        Validate and sanitize a StateReward object.
        
        Args:
            state_reward: StateReward to validate
            
        Returns:
            Validated and sanitized StateReward
        """
        try:
            # Validate individual reward components
            coherence_reward = self.handle_reward_overflow(state_reward.coherence_reward, "coherence")
            growth_reward = self.handle_reward_overflow(state_reward.growth_reward, "growth")
            integration_reward = self.handle_reward_overflow(state_reward.integration_reward, "integration")
            elegance_reward = self.handle_reward_overflow(state_reward.elegance_reward, "elegance")
            emergence_reward = self.handle_reward_overflow(state_reward.emergence_reward, "emergence")
            
            # Validate total_reward (preserve the original which may include bias)
            # Only recalculate if the original total_reward is invalid
            total_reward = self.handle_reward_overflow(state_reward.total_reward, "total")
            
            return StateReward(
                coherence_reward=coherence_reward,
                growth_reward=growth_reward,
                integration_reward=integration_reward,
                elegance_reward=elegance_reward,
                emergence_reward=emergence_reward,
                total_reward=total_reward
            )
            
        except Exception as e:
            logger.error(f"Error validating StateReward: {e}")
            
            error = RewardSystemError(
                f"StateReward validation failed: {str(e)}",
                ErrorType.NUMERICAL_INSTABILITY,
                ErrorSeverity.MEDIUM,
                context={
                    'original_reward': str(state_reward),
                    'exception': str(e)
                }
            )
            self._record_error(error)
            
            return self.fallback_state_reward
    
    def validate_intrinsic_motivation(self, motivation: IntrinsicMotivation) -> IntrinsicMotivation:
        """
        Validate and sanitize an IntrinsicMotivation object.
        
        Args:
            motivation: IntrinsicMotivation to validate
            
        Returns:
            Validated and sanitized IntrinsicMotivation
        """
        try:
            # Validate individual motivation components
            curiosity_drive = self.handle_reward_overflow(motivation.curiosity_drive, "curiosity")
            mastery_drive = self.handle_reward_overflow(motivation.mastery_drive, "mastery")
            autonomy_drive = self.handle_reward_overflow(motivation.autonomy_drive, "autonomy")
            growth_drive = self.handle_reward_overflow(motivation.growth_drive, "growth")
            
            # Recalculate combined motivation
            combined_motivation = (curiosity_drive + mastery_drive + autonomy_drive + growth_drive) / 4.0
            combined_motivation = self.handle_reward_overflow(combined_motivation, "combined_motivation")
            
            return IntrinsicMotivation(
                curiosity_drive=curiosity_drive,
                mastery_drive=mastery_drive,
                autonomy_drive=autonomy_drive,
                growth_drive=growth_drive,
                combined_motivation=combined_motivation
            )
            
        except Exception as e:
            logger.error(f"Error validating IntrinsicMotivation: {e}")
            
            error = RewardSystemError(
                f"IntrinsicMotivation validation failed: {str(e)}",
                ErrorType.NUMERICAL_INSTABILITY,
                ErrorSeverity.MEDIUM,
                context={
                    'original_motivation': str(motivation),
                    'exception': str(e)
                }
            )
            self._record_error(error)
            
            return self.fallback_intrinsic_motivation
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive error statistics.
        
        Returns:
            Dictionary containing error statistics
        """
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_counts_by_type': dict(self.error_counts),
            'recent_errors': len([e for e in self.error_history 
                                if (datetime.now() - e.timestamp).total_seconds() < 3600]),  # Last hour
            'recovery_attempts': dict(self.recovery_attempts),
            'last_recovery_times': {k: v.isoformat() for k, v in self.last_recovery_time.items()},
            'error_history_length': len(self.error_history)
        }
    
    def reset_error_tracking(self) -> None:
        """Reset error tracking for fresh start."""
        self.error_history.clear()
        self.recovery_attempts.clear()
        self.last_recovery_time.clear()
        self.error_counts = {error_type: 0 for error_type in ErrorType}
        logger.info("Reset error tracking")
    
    # Private helper methods
    
    def _is_state_corrupted(self, state: np.ndarray) -> bool:
        """Check if a state is corrupted."""
        if len(state) == 0:
            return True
        
        # Check for NaN or infinite values
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            return True
        
        # Check for excessive magnitude
        state_norm = np.linalg.norm(state)
        if state_norm > self.state_corruption_threshold:
            return True
        
        # Check for all-zero state (might indicate corruption)
        if np.allclose(state, 0.0, atol=self.numerical_stability_epsilon):
            return True
        
        return False
    
    def _attempt_state_recovery(self, corrupted_state: np.ndarray, 
                              previous_states: List[np.ndarray]) -> Optional[np.ndarray]:
        """Attempt to recover a corrupted state using historical data."""
        if not previous_states:
            return None
        
        # Filter valid previous states
        valid_states = [s for s in previous_states 
                       if len(s) == len(corrupted_state) and not self._is_state_corrupted(s)]
        
        if not valid_states:
            return None
        
        try:
            # Method 1: Use most recent valid state
            if len(valid_states) >= 1:
                recent_state = valid_states[-1]
                
                # Add small amount of noise to avoid exact repetition
                noise = np.random.normal(0, 0.01, size=recent_state.shape)
                recovered_state = recent_state + noise
                
                if not self._is_state_corrupted(recovered_state):
                    return recovered_state
            
            # Method 2: Average of recent valid states
            if len(valid_states) >= 2:
                recent_states = valid_states[-3:]  # Last 3 valid states
                averaged_state = np.mean(recent_states, axis=0)
                
                if not self._is_state_corrupted(averaged_state):
                    return averaged_state
            
            # Method 3: Interpolation between valid states
            if len(valid_states) >= 2:
                state1, state2 = valid_states[-2], valid_states[-1]
                interpolated_state = 0.7 * state2 + 0.3 * state1
                
                if not self._is_state_corrupted(interpolated_state):
                    return interpolated_state
            
        except Exception as e:
            logger.warning(f"State recovery attempt failed: {e}")
        
        return None
    
    def _generate_fallback_state(self, state_size: int) -> np.ndarray:
        """Generate a safe fallback state."""
        # Generate state with small random values around zero
        fallback_state = np.random.normal(0, 0.1, size=state_size)
        
        # Ensure it's not corrupted
        if self._is_state_corrupted(fallback_state):
            # Last resort: uniform small values
            fallback_state = np.full(state_size, 0.01)
        
        return fallback_state
    
    def _record_error(self, error: RewardSystemError) -> None:
        """Record an error in the error tracking system."""
        self.error_history.append(error)
        self.error_counts[error.error_type] += 1
        
        # Limit error history size
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-500:]  # Keep last 500
        
        # Log error based on severity
        if error.severity == ErrorSeverity.CRITICAL:
            logger.critical(f"CRITICAL ERROR: {error}")
        elif error.severity == ErrorSeverity.HIGH:
            logger.error(f"HIGH SEVERITY ERROR: {error}")
        elif error.severity == ErrorSeverity.MEDIUM:
            logger.warning(f"MEDIUM SEVERITY ERROR: {error}")
        else:
            logger.info(f"LOW SEVERITY ERROR: {error}")