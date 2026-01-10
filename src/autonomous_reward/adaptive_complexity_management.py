"""
Adaptive Complexity Management for the Autonomous State-Derived Reward System.

This module implements progressive complexity management that scales challenges
based on the agent's growing capabilities, ensuring continuous expansion of potential
while maintaining optimal learning conditions.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging

from .interfaces import AdaptiveComplexityManagementInterface
from .models import Pattern, Experience

logger = logging.getLogger(__name__)


class AdaptiveComplexityManagement(AdaptiveComplexityManagementInterface):
    """
    Progressively takes on more complex challenges as capabilities grow.
    
    This system assesses current capability levels from state patterns,
    scales complexity challenges appropriately, adjusts difficulty based
    on performance, and rewards successful complexity navigation.
    """
    
    def __init__(self, 
                 complexity_scaling_factor: float = 1.2,
                 readiness_threshold: float = 0.7,
                 performance_window: int = 10,
                 min_complexity: float = 0.1,
                 max_complexity: float = 10.0):
        """
        Initialize the Adaptive Complexity Management system.
        
        Args:
            complexity_scaling_factor: Factor by which complexity increases
            readiness_threshold: Minimum readiness score to increase complexity
            performance_window: Number of recent performances to consider
            min_complexity: Minimum complexity level
            max_complexity: Maximum complexity level
        """
        self.complexity_scaling_factor = complexity_scaling_factor
        self.readiness_threshold = readiness_threshold
        self.performance_window = performance_window
        self.min_complexity = min_complexity
        self.max_complexity = max_complexity
        
        # Track capability levels by domain
        self.capability_levels: Dict[str, float] = {}
        
        # Track complexity levels by domain
        self.complexity_levels: Dict[str, float] = {}
        
        # Track performance history
        self.performance_history: List[Dict[str, Any]] = []
        
        # Track mastery indicators
        self.mastery_indicators: Dict[str, List[float]] = {}
        
        # Initialize default domains
        self._initialize_default_domains()
    
    def _initialize_default_domains(self) -> None:
        """Initialize default capability and complexity domains."""
        default_domains = [
            'reasoning', 'learning', 'creativity', 'problem_solving',
            'pattern_recognition', 'synthesis', 'adaptation', 'autonomy'
        ]
        
        for domain in default_domains:
            self.capability_levels[domain] = 0.5  # Start at moderate level
            self.complexity_levels[domain] = 0.5  # Start at moderate complexity
            self.mastery_indicators[domain] = []
    
    def assess_capability_levels(self, state_patterns: List[Pattern]) -> Dict[str, float]:
        """
        Assess current capability levels from state patterns.
        
        Analyzes state patterns to determine capability levels across different
        domains, looking for indicators of skill development and mastery.
        
        Args:
            state_patterns: Current state patterns to analyze
            
        Returns:
            Capability level assessments by domain
        """
        try:
            # Initialize capability assessments
            capability_assessments = dict(self.capability_levels)
            
            # Analyze patterns for capability indicators
            for pattern in state_patterns:
                domain = self._extract_domain_from_pattern(pattern)
                if domain in capability_assessments:
                    # Update capability based on pattern strength and frequency
                    pattern_contribution = self._compute_pattern_capability_contribution(pattern)
                    
                    # Use exponential moving average to update capability
                    alpha = 0.1  # Learning rate for capability updates
                    capability_assessments[domain] = (
                        (1 - alpha) * capability_assessments[domain] + 
                        alpha * pattern_contribution
                    )
            
            # Apply bounds to capability levels
            for domain in capability_assessments:
                capability_assessments[domain] = np.clip(
                    capability_assessments[domain], 0.0, 1.0
                )
            
            # Update internal tracking
            self.capability_levels.update(capability_assessments)
            
            logger.debug(f"Assessed capability levels: {capability_assessments}")
            return capability_assessments
            
        except Exception as e:
            logger.error(f"Error assessing capability levels: {e}")
            return dict(self.capability_levels)
    
    def scale_complexity_challenges(self, capabilities: Dict[str, float]) -> Dict[str, float]:
        """
        Scale complexity challenges based on capabilities.
        
        Determines appropriate complexity levels for each domain based on
        current capability assessments, ensuring challenges are neither
        too easy nor too difficult.
        
        Args:
            capabilities: Current capability levels by domain
            
        Returns:
            Appropriate complexity levels for each domain
        """
        try:
            complexity_levels = {}
            
            for domain, capability in capabilities.items():
                # Base complexity should be slightly above current capability
                # to provide optimal challenge (zone of proximal development)
                base_complexity = capability * 1.1
                
                # Apply scaling factor based on recent performance
                performance_factor = self._compute_performance_factor(domain)
                scaled_complexity = base_complexity * performance_factor
                
                # Apply bounds
                scaled_complexity = np.clip(
                    scaled_complexity, self.min_complexity, self.max_complexity
                )
                
                complexity_levels[domain] = scaled_complexity
            
            # Update internal tracking
            self.complexity_levels.update(complexity_levels)
            
            logger.debug(f"Scaled complexity levels: {complexity_levels}")
            return complexity_levels
            
        except Exception as e:
            logger.error(f"Error scaling complexity challenges: {e}")
            return dict(self.complexity_levels)
    
    def adjust_challenge_difficulty(self, 
                                  performance_history: List[Dict[str, Any]], 
                                  current_readiness: float) -> float:
        """
        Adjust challenge difficulty based on performance and readiness.
        
        Analyzes recent performance to determine if difficulty should be
        increased, decreased, or maintained, taking into account the agent's
        current readiness for complexity.
        
        Args:
            performance_history: Historical performance data
            current_readiness: Current readiness assessment (0.0 to 1.0)
            
        Returns:
            Adjusted difficulty level
        """
        try:
            # Update internal performance history
            self.performance_history.extend(performance_history)
            
            # Keep only recent performance data
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.performance_history = [
                perf for perf in self.performance_history
                if perf.get('timestamp', datetime.now()) > cutoff_time
            ]
            
            if not self.performance_history:
                return 0.5  # Default moderate difficulty
            
            # Compute recent performance metrics
            recent_performances = self.performance_history[-self.performance_window:]
            success_rates = [perf.get('success_rate', 0.5) for perf in recent_performances]
            avg_success_rate = np.mean(success_rates)
            
            # Determine difficulty adjustment based on performance and readiness
            if avg_success_rate > 0.8 and current_readiness > self.readiness_threshold:
                # High success rate and high readiness -> increase difficulty
                difficulty_adjustment = self.complexity_scaling_factor
            elif avg_success_rate < 0.4:
                # Low success rate -> decrease difficulty
                difficulty_adjustment = 1.0 / self.complexity_scaling_factor
            else:
                # Moderate performance -> maintain current difficulty
                difficulty_adjustment = 1.0
            
            # Apply readiness factor
            readiness_factor = 0.5 + 0.5 * current_readiness
            adjusted_difficulty = difficulty_adjustment * readiness_factor
            
            # Apply bounds
            adjusted_difficulty = np.clip(
                adjusted_difficulty, self.min_complexity, self.max_complexity
            )
            
            logger.debug(f"Adjusted difficulty: {adjusted_difficulty} "
                        f"(success_rate: {avg_success_rate:.2f}, "
                        f"readiness: {current_readiness:.2f})")
            
            return adjusted_difficulty
            
        except Exception as e:
            logger.error(f"Error adjusting challenge difficulty: {e}")
            return 0.5  # Default moderate difficulty
    
    def reward_complexity_navigation(self, 
                                   complexity_handled: float, 
                                   success_rate: float) -> float:
        """
        Reward successful navigation of complexity.
        
        Generates rewards for successfully handling complex challenges,
        with higher rewards for higher complexity levels and better
        success rates.
        
        Args:
            complexity_handled: Level of complexity successfully handled
            success_rate: Success rate in handling complexity (0.0 to 1.0)
            
        Returns:
            Complexity navigation reward
        """
        try:
            # Base reward scales with complexity level
            base_reward = complexity_handled * 0.5
            
            # Success rate multiplier (exponential to reward high success rates)
            success_multiplier = np.power(success_rate, 2)
            
            # Complexity bonus for handling high complexity
            complexity_bonus = 0.0
            if complexity_handled > 0.7:
                complexity_bonus = (complexity_handled - 0.7) * 2.0
            
            # Compute total reward
            total_reward = (base_reward * success_multiplier) + complexity_bonus
            
            # Apply bounds to prevent extreme rewards
            total_reward = np.clip(total_reward, 0.0, 5.0)
            
            logger.debug(f"Complexity navigation reward: {total_reward:.3f} "
                        f"(complexity: {complexity_handled:.2f}, "
                        f"success_rate: {success_rate:.2f})")
            
            return total_reward
            
        except Exception as e:
            logger.error(f"Error computing complexity navigation reward: {e}")
            return 0.0
    
    def _extract_domain_from_pattern(self, pattern: Pattern) -> str:
        """
        Extract the capability domain from a pattern.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Domain name or 'general' if no specific domain identified
        """
        pattern_type = pattern.pattern_type.lower()
        
        # Map pattern types to capability domains
        domain_mappings = {
            'reasoning': ['logic', 'inference', 'deduction', 'reasoning'],
            'learning': ['adaptation', 'learning', 'memory', 'consolidation'],
            'creativity': ['novelty', 'creativity', 'synthesis', 'innovation'],
            'problem_solving': ['solution', 'problem', 'resolution', 'optimization'],
            'pattern_recognition': ['pattern', 'recognition', 'detection', 'classification'],
            'autonomy': ['autonomy', 'independence', 'self_direction', 'agency']
        }
        
        for domain, keywords in domain_mappings.items():
            if any(keyword in pattern_type for keyword in keywords):
                return domain
        
        return 'general'
    
    def _compute_pattern_capability_contribution(self, pattern: Pattern) -> float:
        """
        Compute how much a pattern contributes to capability assessment.
        
        Args:
            pattern: Pattern to analyze
            
        Returns:
            Capability contribution (0.0 to 1.0)
        """
        # Base contribution from pattern strength
        base_contribution = pattern.strength
        
        # Frequency bonus (more frequent patterns indicate stronger capability)
        frequency_bonus = min(pattern.frequency / 100.0, 0.3)
        
        # Recency bonus (recent patterns are more indicative of current capability)
        time_since_occurrence = datetime.now() - pattern.last_occurrence
        hours_since = time_since_occurrence.total_seconds() / 3600
        recency_bonus = max(0.0, 0.2 * (1.0 - hours_since / 24.0))
        
        total_contribution = base_contribution + frequency_bonus + recency_bonus
        return np.clip(total_contribution, 0.0, 1.0)
    
    def _compute_performance_factor(self, domain: str) -> float:
        """
        Compute performance factor for complexity scaling in a domain.
        
        Args:
            domain: Domain to compute performance factor for
            
        Returns:
            Performance factor (0.5 to 2.0)
        """
        if domain not in self.mastery_indicators:
            return 1.0
        
        recent_indicators = self.mastery_indicators[domain][-self.performance_window:]
        if not recent_indicators:
            return 1.0
        
        avg_performance = np.mean(recent_indicators)
        
        # Scale performance to factor range
        if avg_performance > 0.8:
            return 1.5  # Increase complexity for high performance
        elif avg_performance < 0.4:
            return 0.8  # Decrease complexity for low performance
        else:
            return 1.0  # Maintain complexity for moderate performance
    
    def update_mastery_indicators(self, domain: str, performance: float) -> None:
        """
        Update mastery indicators for a domain.
        
        Args:
            domain: Domain to update
            performance: Performance score (0.0 to 1.0)
        """
        if domain not in self.mastery_indicators:
            self.mastery_indicators[domain] = []
        
        self.mastery_indicators[domain].append(performance)
        
        # Keep only recent indicators
        if len(self.mastery_indicators[domain]) > self.performance_window * 2:
            self.mastery_indicators[domain] = self.mastery_indicators[domain][-self.performance_window:]
    
    def get_current_complexity_profile(self) -> Dict[str, Any]:
        """
        Get current complexity management profile.
        
        Returns:
            Dictionary containing current complexity management state
        """
        return {
            'capability_levels': dict(self.capability_levels),
            'complexity_levels': dict(self.complexity_levels),
            'performance_history_length': len(self.performance_history),
            'mastery_indicators': {
                domain: len(indicators) 
                for domain, indicators in self.mastery_indicators.items()
            },
            'configuration': {
                'complexity_scaling_factor': self.complexity_scaling_factor,
                'readiness_threshold': self.readiness_threshold,
                'performance_window': self.performance_window,
                'min_complexity': self.min_complexity,
                'max_complexity': self.max_complexity
            }
        }