"""
Progressive Learning Controller for the Autonomous State-Derived Reward System.

This module implements adaptive learning pace and complexity management based on
internal state readiness, enabling the agent to learn at its own natural rhythm
without external pressure.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime, timedelta
import math

from .interfaces import ProgressiveLearningControllerInterface
from .models import LearningState, Experience
from ..core.config import AgentConfig

logger = logging.getLogger(__name__)


class ProgressiveLearningController(ProgressiveLearningControllerInterface):
    """
    Adapts learning pace and complexity based on internal state readiness.
    
    This controller implements progressive learning that derives optimal learning
    pace from state stability and energy levels, enabling meta-learning capabilities
    that improve learning efficiency over time.
    """
    
    def __init__(self, config: AgentConfig):
        """
        Initialize the progressive learning controller.
        
        Args:
            config: Agent configuration containing learning parameters
        """
        self.config = config
        
        # Learning rate parameters
        self.base_learning_rate = getattr(config, 'base_learning_rate', 0.01)
        self.min_learning_rate = getattr(config, 'min_learning_rate', 0.001)
        self.max_learning_rate = getattr(config, 'max_learning_rate', 0.1)
        self.confidence_threshold = getattr(config, 'confidence_threshold', 0.7)
        
        # Complexity management parameters
        self.base_complexity = getattr(config, 'base_complexity', 0.3)
        self.min_complexity = getattr(config, 'min_complexity', 0.1)
        self.max_complexity = getattr(config, 'max_complexity', 1.0)
        self.readiness_threshold = getattr(config, 'readiness_threshold', 0.8)
        self.complexity_increment = getattr(config, 'complexity_increment', 0.1)
        
        # Meta-learning parameters
        self.meta_learning_rate = getattr(config, 'meta_learning_rate', 0.001)
        self.adaptation_window = getattr(config, 'adaptation_window', 20)
        self.performance_smoothing = getattr(config, 'performance_smoothing', 0.9)
        
        # Consolidation parameters
        self.consolidation_threshold = getattr(config, 'consolidation_threshold', 0.6)
        self.experience_window = getattr(config, 'experience_window', 50)
        self.stability_requirement = getattr(config, 'stability_requirement', 0.8)
        
        # Internal state tracking
        self.current_learning_state = LearningState(
            learning_rate=self.base_learning_rate,
            complexity_level=self.base_complexity,
            readiness_score=0.5,
            consolidation_needed=False,
            meta_learning_params={
                'adaptation_rate': self.meta_learning_rate,
                'performance_trend': 0.0,
                'stability_score': 0.5,
                'learning_efficiency': 1.0
            }
        )
        
        # Performance tracking
        self.performance_history: List[float] = []
        self.learning_rate_history: List[float] = []
        self.complexity_history: List[float] = []
        self.confidence_history: List[float] = []
        self.readiness_history: List[float] = []
        
        # Meta-learning tracking
        self.learning_outcomes: List[Dict[str, Any]] = []
        self.adaptation_history: List[Dict[str, float]] = []
        
        # Consolidation tracking
        self.last_consolidation: Optional[datetime] = None
        self.consolidation_frequency = timedelta(hours=1)  # Minimum time between consolidations
        
        logger.info("Initialized ProgressiveLearningController")
    
    def compute_adaptive_learning_rate(self, state_confidence: float, 
                                     recent_performance: List[float]) -> float:
        """
        Compute adaptive learning rate based on state confidence.
        
        The learning rate adapts inversely to confidence: lower confidence leads to
        higher learning rates to accelerate learning, while higher confidence leads
        to lower learning rates for fine-tuning.
        
        Args:
            state_confidence: Confidence level from pathos state (0.0 to 1.0)
            recent_performance: Recent learning performance data
            
        Returns:
            Adaptive learning rate
        """
        # Clamp confidence to valid range
        state_confidence = max(0.0, min(1.0, state_confidence))
        
        # Store confidence for tracking
        self.confidence_history.append(state_confidence)
        if len(self.confidence_history) > self.adaptation_window:
            self.confidence_history.pop(0)
        
        # Store performance data
        if recent_performance:
            self.performance_history.extend(recent_performance)
            if len(self.performance_history) > self.adaptation_window:
                self.performance_history = self.performance_history[-self.adaptation_window:]
        
        # Base adaptive rate: inverse relationship with confidence
        # High confidence (0.9) -> low learning rate (closer to min)
        # Low confidence (0.1) -> high learning rate (closer to max)
        confidence_factor = 1.0 - state_confidence
        base_adaptive_rate = (self.min_learning_rate + 
                             confidence_factor * (self.max_learning_rate - self.min_learning_rate))
        
        # Performance-based adjustment
        performance_adjustment = 1.0
        if len(self.performance_history) >= 3:
            recent_avg = np.mean(self.performance_history[-3:])
            overall_avg = np.mean(self.performance_history)
            
            if recent_avg > overall_avg:
                # Recent performance is good, can afford to be more conservative
                performance_adjustment = 0.9
            elif recent_avg < overall_avg * 0.8:
                # Recent performance is poor, need to learn faster
                performance_adjustment = 1.2
        
        # Meta-learning adjustment based on learning efficiency
        meta_efficiency = self.current_learning_state.meta_learning_params.get('learning_efficiency', 1.0)
        meta_adjustment = meta_efficiency
        
        # Stability adjustment - more stable states can handle higher learning rates
        stability_score = self.current_learning_state.meta_learning_params.get('stability_score', 0.5)
        stability_adjustment = 0.8 + (stability_score * 0.4)  # Range: 0.8 to 1.2
        
        # Combine all adjustments
        adaptive_rate = (base_adaptive_rate * performance_adjustment * 
                        meta_adjustment * stability_adjustment)
        
        # Clamp to valid range
        adaptive_rate = max(self.min_learning_rate, min(self.max_learning_rate, adaptive_rate))
        
        # Update learning state
        self.current_learning_state.learning_rate = adaptive_rate
        self.learning_rate_history.append(adaptive_rate)
        if len(self.learning_rate_history) > self.adaptation_window:
            self.learning_rate_history.pop(0)
        
        logger.debug(f"Computed adaptive learning rate: {adaptive_rate:.6f} "
                    f"(confidence={state_confidence:.3f}, performance_adj={performance_adjustment:.3f}, "
                    f"meta_adj={meta_adjustment:.3f}, stability_adj={stability_adjustment:.3f})")
        
        return adaptive_rate
    
    def assess_complexity_readiness(self, current_state: np.ndarray, 
                                  capability_level: float) -> float:
        """
        Assess readiness for increased complexity challenges.
        
        Readiness is determined by state stability, recent performance trends,
        and current capability levels. Higher readiness indicates the agent
        can handle more complex challenges.
        
        Args:
            current_state: Current pathos state
            capability_level: Current capability assessment (0.0 to 1.0)
            
        Returns:
            Complexity readiness score (0.0 to 1.0)
        """
        # Clamp capability level to valid range
        capability_level = max(0.0, min(1.0, capability_level))
        
        # Assess state stability
        state_energy = float(np.linalg.norm(current_state))
        state_variance = float(np.var(current_state))
        
        # Normalize state metrics
        normalized_energy = min(state_energy / 10.0, 1.0)  # Assume max energy ~10
        normalized_variance = min(state_variance / 1.0, 1.0)  # Assume max variance ~1
        
        # State stability: high energy with low variance indicates readiness
        state_stability = normalized_energy * (1.0 - normalized_variance)
        
        # Performance trend assessment
        performance_trend = 0.5  # Default neutral trend
        if len(self.performance_history) >= 5:
            recent_performance = self.performance_history[-5:]
            older_performance = self.performance_history[-10:-5] if len(self.performance_history) >= 10 else []
            
            if older_performance:
                recent_avg = np.mean(recent_performance)
                older_avg = np.mean(older_performance)
                trend_ratio = recent_avg / (older_avg + 1e-8)
                performance_trend = min(trend_ratio, 2.0) / 2.0  # Normalize to 0-1
            else:
                # Use absolute performance level if no trend available
                performance_trend = min(np.mean(recent_performance), 1.0)
        
        # Learning rate stability - consistent learning rates indicate readiness
        learning_rate_stability = 0.5
        if len(self.learning_rate_history) >= 5:
            rate_variance = np.var(self.learning_rate_history[-5:])
            rate_mean = np.mean(self.learning_rate_history[-5:])
            coefficient_of_variation = rate_variance / (rate_mean + 1e-8)
            learning_rate_stability = 1.0 / (1.0 + coefficient_of_variation)
        
        # Confidence trend - increasing confidence indicates readiness
        confidence_trend = 0.5
        if len(self.confidence_history) >= 5:
            recent_confidence = np.mean(self.confidence_history[-3:])
            older_confidence = np.mean(self.confidence_history[-5:-3])
            confidence_improvement = (recent_confidence - older_confidence + 1.0) / 2.0
            confidence_trend = max(0.0, min(1.0, confidence_improvement))
        
        # Combine readiness factors
        readiness_components = {
            'state_stability': state_stability * 0.25,
            'performance_trend': performance_trend * 0.30,
            'learning_rate_stability': learning_rate_stability * 0.20,
            'confidence_trend': confidence_trend * 0.15,
            'capability_level': capability_level * 0.10
        }
        
        readiness_score = sum(readiness_components.values())
        readiness_score = max(0.0, min(1.0, readiness_score))
        
        # Update learning state
        self.current_learning_state.readiness_score = readiness_score
        self.readiness_history.append(readiness_score)
        if len(self.readiness_history) > self.adaptation_window:
            self.readiness_history.pop(0)
        
        # Adjust complexity level based on readiness
        current_complexity = self.current_learning_state.complexity_level
        
        if readiness_score > self.readiness_threshold:
            # Ready for increased complexity
            new_complexity = min(current_complexity + self.complexity_increment, self.max_complexity)
        elif readiness_score < (self.readiness_threshold - 0.2):
            # Need to reduce complexity
            new_complexity = max(current_complexity - self.complexity_increment, self.min_complexity)
        else:
            # Maintain current complexity
            new_complexity = current_complexity
        
        self.current_learning_state.complexity_level = new_complexity
        self.complexity_history.append(new_complexity)
        if len(self.complexity_history) > self.adaptation_window:
            self.complexity_history.pop(0)
        
        logger.debug(f"Assessed complexity readiness: {readiness_score:.3f} "
                    f"(state_stability={state_stability:.3f}, performance_trend={performance_trend:.3f}, "
                    f"lr_stability={learning_rate_stability:.3f}, confidence_trend={confidence_trend:.3f}, "
                    f"capability={capability_level:.3f}) -> complexity={new_complexity:.3f}")
        
        return readiness_score
    
    def update_meta_learning_parameters(self, learning_outcomes: List[Dict[str, Any]]) -> None:
        """
        Update meta-learning parameters based on learning outcomes.
        
        Meta-learning enables the system to learn how to learn more effectively
        by analyzing patterns in learning outcomes and adjusting parameters
        accordingly.
        
        Args:
            learning_outcomes: Recent learning outcome data
        """
        if not learning_outcomes:
            return
        
        # Store learning outcomes
        self.learning_outcomes.extend(learning_outcomes)
        if len(self.learning_outcomes) > self.adaptation_window * 2:
            self.learning_outcomes = self.learning_outcomes[-self.adaptation_window * 2:]
        
        # Extract performance metrics from outcomes
        recent_outcomes = learning_outcomes[-self.adaptation_window:] if len(learning_outcomes) > self.adaptation_window else learning_outcomes
        
        performance_scores = []
        learning_speeds = []
        stability_scores = []
        
        for outcome in recent_outcomes:
            # Extract metrics (with defaults for missing data)
            performance = outcome.get('performance', 0.5)
            learning_speed = outcome.get('learning_speed', 1.0)
            stability = outcome.get('stability', 0.5)
            
            performance_scores.append(performance)
            learning_speeds.append(learning_speed)
            stability_scores.append(stability)
        
        # Compute meta-learning metrics
        avg_performance = np.mean(performance_scores) if performance_scores else 0.5
        avg_learning_speed = np.mean(learning_speeds) if learning_speeds else 1.0
        avg_stability = np.mean(stability_scores) if stability_scores else 0.5
        
        # Compute performance trend
        performance_trend = 0.0
        if len(performance_scores) >= 5:
            recent_perf = np.mean(performance_scores[-3:])
            older_perf = np.mean(performance_scores[-5:-3])
            performance_trend = recent_perf - older_perf
        
        # Update meta-learning parameters
        current_params = self.current_learning_state.meta_learning_params
        
        # Adaptation rate: increase if performance is improving, decrease if declining
        adaptation_rate = current_params.get('adaptation_rate', self.meta_learning_rate)
        if performance_trend > 0.1:
            adaptation_rate = min(adaptation_rate * 1.1, self.meta_learning_rate * 2.0)
        elif performance_trend < -0.1:
            adaptation_rate = max(adaptation_rate * 0.9, self.meta_learning_rate * 0.5)
        
        # Learning efficiency: based on ratio of performance to learning speed
        learning_efficiency = avg_performance / (avg_learning_speed + 1e-8)
        learning_efficiency = max(0.1, min(2.0, learning_efficiency))
        
        # Stability score: exponential moving average
        stability_ema = current_params.get('stability_score', 0.5)
        stability_ema = (self.performance_smoothing * stability_ema + 
                        (1.0 - self.performance_smoothing) * avg_stability)
        
        # Update parameters
        updated_params = {
            'adaptation_rate': adaptation_rate,
            'performance_trend': performance_trend,
            'stability_score': stability_ema,
            'learning_efficiency': learning_efficiency,
            'avg_performance': avg_performance,
            'avg_learning_speed': avg_learning_speed,
            'last_update': datetime.now().isoformat()
        }
        
        self.current_learning_state.meta_learning_params.update(updated_params)
        
        # Store adaptation history
        self.adaptation_history.append({
            'timestamp': datetime.now().isoformat(),
            'adaptation_rate': adaptation_rate,
            'learning_efficiency': learning_efficiency,
            'stability_score': stability_ema,
            'performance_trend': performance_trend
        })
        
        if len(self.adaptation_history) > self.adaptation_window:
            self.adaptation_history.pop(0)
        
        logger.debug(f"Updated meta-learning parameters: "
                    f"adaptation_rate={adaptation_rate:.6f}, "
                    f"learning_efficiency={learning_efficiency:.3f}, "
                    f"stability_score={stability_ema:.3f}, "
                    f"performance_trend={performance_trend:.3f}")
    
    def detect_consolidation_need(self, recent_experiences: List[Experience]) -> bool:
        """
        Detect when consolidation time is needed for integration.
        
        Consolidation is needed when the agent has accumulated significant
        new experiences that need to be integrated, or when performance
        indicates cognitive overload.
        
        Args:
            recent_experiences: Recent experience data
            
        Returns:
            True if consolidation is needed, False otherwise
        """
        if not recent_experiences:
            return False
        
        # Check if minimum time has passed since last consolidation
        now = datetime.now()
        if (self.last_consolidation and 
            (now - self.last_consolidation) < self.consolidation_frequency):
            return False
        
        # Experience volume indicator
        experience_volume = len(recent_experiences)
        volume_threshold = self.experience_window * 0.8  # 80% of window size
        volume_pressure = experience_volume / volume_threshold if volume_threshold > 0 else 0
        
        # Experience diversity indicator
        action_types = set([exp.action_taken for exp in recent_experiences])
        context_types = set([str(exp.context.get('type', 'unknown')) for exp in recent_experiences])
        diversity_score = (len(action_types) + len(context_types)) / (2.0 * max(1, len(recent_experiences) / 10))
        
        # Performance instability indicator
        recent_rewards = [exp.reward_received for exp in recent_experiences[-10:]]
        reward_variance = np.var(recent_rewards) if len(recent_rewards) > 1 else 0.0
        reward_mean = np.mean(recent_rewards) if recent_rewards else 0.0
        instability_score = reward_variance / (reward_mean + 1e-8) if reward_mean != 0 else 0.0
        
        # Learning rate instability
        lr_instability = 0.0
        if len(self.learning_rate_history) >= 5:
            lr_variance = np.var(self.learning_rate_history[-5:])
            lr_mean = np.mean(self.learning_rate_history[-5:])
            lr_instability = lr_variance / (lr_mean + 1e-8)
        
        # State transition intensity
        state_transitions = []
        for i in range(1, len(recent_experiences)):
            prev_state = recent_experiences[i-1].state_after
            curr_state = recent_experiences[i].state_before
            transition_magnitude = float(np.linalg.norm(curr_state - prev_state))
            state_transitions.append(transition_magnitude)
        
        transition_intensity = np.mean(state_transitions) if state_transitions else 0.0
        
        # Consolidation indicators
        consolidation_indicators = {
            'volume_pressure': min(volume_pressure, 2.0),  # Cap at 2.0
            'diversity_score': min(diversity_score, 2.0),  # Cap at 2.0
            'instability_score': min(instability_score, 2.0),  # Cap at 2.0
            'lr_instability': min(lr_instability, 2.0),  # Cap at 2.0
            'transition_intensity': min(transition_intensity, 2.0)  # Cap at 2.0
        }
        
        # Weighted consolidation score
        consolidation_weights = {
            'volume_pressure': 0.25,
            'diversity_score': 0.20,
            'instability_score': 0.25,
            'lr_instability': 0.15,
            'transition_intensity': 0.15
        }
        
        consolidation_score = sum(
            consolidation_indicators[key] * consolidation_weights[key]
            for key in consolidation_indicators
        )
        
        # Determine if consolidation is needed
        consolidation_needed = consolidation_score > self.consolidation_threshold
        
        # Update learning state
        self.current_learning_state.consolidation_needed = consolidation_needed
        
        if consolidation_needed:
            self.last_consolidation = now
            logger.info(f"Consolidation needed: score={consolidation_score:.3f} > threshold={self.consolidation_threshold:.3f} "
                       f"(volume={volume_pressure:.2f}, diversity={diversity_score:.2f}, "
                       f"instability={instability_score:.2f}, lr_instability={lr_instability:.2f}, "
                       f"transition_intensity={transition_intensity:.2f})")
        else:
            logger.debug(f"No consolidation needed: score={consolidation_score:.3f} <= threshold={self.consolidation_threshold:.3f}")
        
        return consolidation_needed
    
    def get_current_learning_state(self) -> LearningState:
        """
        Get the current learning state configuration.
        
        Returns:
            Current LearningState
        """
        return self.current_learning_state
    
    def get_learning_history(self) -> Dict[str, List[float]]:
        """
        Get comprehensive learning history for analysis.
        
        Returns:
            Dictionary containing various learning metrics over time
        """
        return {
            'performance_history': self.performance_history.copy(),
            'learning_rate_history': self.learning_rate_history.copy(),
            'complexity_history': self.complexity_history.copy(),
            'confidence_history': self.confidence_history.copy(),
            'readiness_history': self.readiness_history.copy()
        }
    
    def get_meta_learning_status(self) -> Dict[str, Any]:
        """
        Get current meta-learning status and parameters.
        
        Returns:
            Dictionary containing meta-learning status
        """
        return {
            'current_params': self.current_learning_state.meta_learning_params.copy(),
            'adaptation_history': self.adaptation_history.copy(),
            'learning_outcomes_count': len(self.learning_outcomes),
            'last_consolidation': self.last_consolidation.isoformat() if self.last_consolidation else None,
            'consolidation_needed': self.current_learning_state.consolidation_needed
        }
    
    def update_learning_parameters(self, current_capabilities: Dict[str, float], 
                                 performance_feedback: Dict[str, Any]) -> None:
        """
        Update learning parameters based on current capabilities and performance feedback.
        
        Args:
            current_capabilities: Dictionary of current capability levels by skill domain
            performance_feedback: Dictionary containing performance metrics and feedback
        """
        # Extract performance metrics
        success_rate = performance_feedback.get('success_rate', 0.5)
        complexity_handled = performance_feedback.get('complexity_handled', self.current_learning_state.complexity_level)
        learning_efficiency = performance_feedback.get('learning_efficiency', 0.6)
        
        # Compute overall capability level
        capability_level = np.mean(list(current_capabilities.values())) if current_capabilities else 0.5
        
        # Update performance history
        self.performance_history.append(success_rate)
        if len(self.performance_history) > self.adaptation_window:
            self.performance_history.pop(0)
        
        # Compute adaptive learning rate based on success rate (inverse of confidence)
        state_confidence = success_rate  # Use success rate as proxy for confidence
        adaptive_learning_rate = self.compute_adaptive_learning_rate(state_confidence, [success_rate])
        
        # Assess complexity readiness based on current capabilities
        # Create a dummy state for readiness assessment
        dummy_state = np.random.normal(0.5, 0.2, 32)  # Placeholder state
        readiness_score = self.assess_complexity_readiness(dummy_state, capability_level)
        
        # Create learning outcome for meta-learning
        learning_outcome = {
            'performance': success_rate,
            'learning_speed': learning_efficiency,
            'stability': min(success_rate, learning_efficiency),
            'capability_level': capability_level,
            'complexity_handled': complexity_handled,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update meta-learning parameters
        self.update_meta_learning_parameters([learning_outcome])
        
        logger.debug(f"Updated learning parameters: lr={adaptive_learning_rate:.6f}, "
                    f"complexity={self.current_learning_state.complexity_level:.3f}, "
                    f"readiness={readiness_score:.3f}, capability={capability_level:.3f}")

    def reset_learning_state(self) -> None:
        """
        Reset the learning state to initial configuration.
        
        This can be useful for starting fresh learning cycles or
        recovering from problematic learning states.
        """
        self.current_learning_state = LearningState(
            learning_rate=self.base_learning_rate,
            complexity_level=self.base_complexity,
            readiness_score=0.5,
            consolidation_needed=False,
            meta_learning_params={
                'adaptation_rate': self.meta_learning_rate,
                'performance_trend': 0.0,
                'stability_score': 0.5,
                'learning_efficiency': 1.0
            }
        )
        
        # Clear histories
        self.performance_history.clear()
        self.learning_rate_history.clear()
        self.complexity_history.clear()
        self.confidence_history.clear()
        self.readiness_history.clear()
        self.learning_outcomes.clear()
        self.adaptation_history.clear()
        
        self.last_consolidation = None
        
        logger.info("Reset learning state to initial configuration")