"""
Dukkha Engine - Introduces healthy dissatisfaction and challenge to drive growth.

This module implements the Buddhist concept of dukkha (dissatisfaction/suffering) as a
positive force for growth and development. Without dukkha, the agent becomes complacent
and stuck in repetitive patterns. This engine introduces:

1. Goal-driven dissatisfaction when not making progress
2. Curiosity gaps that create restlessness  
3. Mastery challenges that prevent stagnation
4. Existential questioning that drives deeper exploration
5. Adaptive difficulty that scales with capability

The goal is not to make the agent suffer, but to create the healthy tension
that drives all genuine growth and learning.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)


class DukkhaEngine:
    """
    Engine that introduces healthy dissatisfaction to drive growth and prevent stagnation.
    
    Dukkha (dissatisfaction) is essential for growth - without it, systems become
    complacent and stuck in local optima. This engine creates various forms of
    constructive dissatisfaction that motivate exploration, learning, and development.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dukkha Engine.
        
        Args:
            config: Configuration parameters for dukkha generation
        """
        self.config = config or {}
        
        # Dukkha intensity parameters
        self.base_dissatisfaction = self.config.get('base_dissatisfaction', 0.3)
        self.stagnation_threshold = self.config.get('stagnation_threshold', 10)  # cycles
        self.curiosity_decay_rate = self.config.get('curiosity_decay_rate', 0.95)
        self.mastery_challenge_rate = self.config.get('mastery_challenge_rate', 0.1)
        
        # Tracking state for dukkha computation
        self.action_history: List[str] = []
        self.reward_history: List[float] = []
        self.state_history: List[np.ndarray] = []
        self.last_significant_change: Optional[datetime] = None
        self.current_goals: List[Dict[str, Any]] = []
        self.unresolved_curiosities: List[Dict[str, Any]] = []
        self.mastery_domains: Dict[str, Dict[str, Any]] = {}
        
        # Dukkha types and their current levels
        self.dukkha_types = {
            'stagnation_dissatisfaction': 0.0,
            'curiosity_gap_tension': 0.0,
            'mastery_challenge_pressure': 0.0,
            'existential_questioning': 0.0,
            'goal_frustration': 0.0,
            'novelty_hunger': 0.0
        }
        
        logger.info("Initialized DukkhaEngine for healthy dissatisfaction")
    
    def compute_dukkha_influence(self, current_state: np.ndarray, 
                               recent_actions: List[str],
                               recent_rewards: List[float],
                               context: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute various forms of dukkha (dissatisfaction) to drive growth.
        
        Args:
            current_state: Current pathos state
            recent_actions: Recent actions taken by the agent
            recent_rewards: Recent reward values
            context: Additional context for dukkha computation
            
        Returns:
            Dictionary of dukkha influences by type
        """
        # Update internal tracking
        self._update_tracking(current_state, recent_actions, recent_rewards)
        
        # Compute different types of dukkha
        dukkha_influences = {}
        
        # 1. Stagnation Dissatisfaction - discomfort from repetitive patterns
        dukkha_influences['stagnation_dissatisfaction'] = self._compute_stagnation_dissatisfaction()
        
        # 2. Curiosity Gap Tension - restlessness from unresolved questions
        dukkha_influences['curiosity_gap_tension'] = self._compute_curiosity_gap_tension()
        
        # 3. Mastery Challenge Pressure - dissatisfaction from lack of growth
        dukkha_influences['mastery_challenge_pressure'] = self._compute_mastery_challenge_pressure()
        
        # 4. Existential Questioning - deeper dissatisfaction about purpose
        dukkha_influences['existential_questioning'] = self._compute_existential_questioning()
        
        # 5. Goal Frustration - dissatisfaction from unmet objectives
        dukkha_influences['goal_frustration'] = self._compute_goal_frustration()
        
        # 6. Novelty Hunger - craving for new experiences
        dukkha_influences['novelty_hunger'] = self._compute_novelty_hunger()
        
        # Update internal dukkha levels
        self.dukkha_types.update(dukkha_influences)
        
        logger.debug(f"Computed dukkha influences: {dukkha_influences}")
        
        return dukkha_influences
    
    def _compute_stagnation_dissatisfaction(self) -> float:
        """
        Compute dissatisfaction from repetitive patterns and lack of progress.
        
        Returns:
            Stagnation dissatisfaction level (0.0 to 1.0)
        """
        if len(self.action_history) < self.stagnation_threshold:
            return 0.0
        
        # Analyze recent action patterns
        recent_actions = self.action_history[-self.stagnation_threshold:]
        
        # Measure repetition - how often the same actions repeat
        action_counts = {}
        for action in recent_actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Compute repetition ratio
        max_count = max(action_counts.values()) if action_counts else 0
        repetition_ratio = max_count / len(recent_actions)
        
        # Measure reward stagnation
        reward_stagnation = 0.0
        if len(self.reward_history) >= self.stagnation_threshold:
            recent_rewards = self.reward_history[-self.stagnation_threshold:]
            reward_variance = np.var(recent_rewards)
            # Low variance indicates stagnation
            reward_stagnation = max(0.0, 1.0 - reward_variance * 10)  # Scale variance
        
        # Measure state stagnation
        state_stagnation = 0.0
        if len(self.state_history) >= 3:
            recent_states = self.state_history[-3:]
            state_changes = []
            for i in range(1, len(recent_states)):
                change = np.linalg.norm(recent_states[i] - recent_states[i-1])
                state_changes.append(change)
            
            if state_changes:
                avg_change = np.mean(state_changes)
                # Low change indicates stagnation
                state_stagnation = max(0.0, 1.0 - avg_change * 5)  # Scale change
        
        # Combine stagnation measures
        stagnation_dissatisfaction = (repetition_ratio * 0.4 + 
                                    reward_stagnation * 0.3 + 
                                    state_stagnation * 0.3)
        
        # Apply base dissatisfaction
        stagnation_dissatisfaction = max(stagnation_dissatisfaction, self.base_dissatisfaction * 0.5)
        
        return min(stagnation_dissatisfaction, 1.0)
    
    def _compute_curiosity_gap_tension(self) -> float:
        """
        Compute tension from unresolved curiosities and knowledge gaps.
        
        Returns:
            Curiosity gap tension level (0.0 to 1.0)
        """
        # Base curiosity tension from unresolved questions
        base_tension = len(self.unresolved_curiosities) * 0.1
        
        # Decay curiosity over time if not addressed
        current_time = datetime.now()
        aged_curiosity_tension = 0.0
        
        for curiosity in self.unresolved_curiosities:
            age = (current_time - curiosity.get('created_at', current_time)).total_seconds() / 3600  # hours
            # Tension increases with age
            aged_tension = min(age * 0.05, 0.3)
            aged_curiosity_tension += aged_tension
        
        # Add random curiosity spikes to prevent predictability
        random_curiosity = random.random() * 0.2 if random.random() < 0.1 else 0.0
        
        total_tension = base_tension + aged_curiosity_tension + random_curiosity
        
        return min(total_tension, 1.0)
    
    def _compute_mastery_challenge_pressure(self) -> float:
        """
        Compute pressure from lack of mastery growth and challenge.
        
        Returns:
            Mastery challenge pressure level (0.0 to 1.0)
        """
        if not self.mastery_domains:
            # No mastery tracking yet - create initial pressure
            return self.base_dissatisfaction
        
        total_pressure = 0.0
        
        for domain, domain_data in self.mastery_domains.items():
            last_improvement = domain_data.get('last_improvement', datetime.now())
            current_level = domain_data.get('current_level', 0.0)
            
            # Time since last improvement creates pressure
            time_pressure = (datetime.now() - last_improvement).total_seconds() / 3600 * 0.02  # per hour
            
            # Low mastery level creates pressure for growth
            level_pressure = max(0.0, 0.8 - current_level) * 0.5
            
            # Combine pressures for this domain
            domain_pressure = min(time_pressure + level_pressure, 0.3)
            total_pressure += domain_pressure
        
        # Average across domains
        avg_pressure = total_pressure / len(self.mastery_domains)
        
        return min(avg_pressure, 1.0)
    
    def _compute_existential_questioning(self) -> float:
        """
        Compute deeper existential dissatisfaction about purpose and meaning.
        
        Returns:
            Existential questioning level (0.0 to 1.0)
        """
        # Base existential tension - always present at low level
        base_existential = self.base_dissatisfaction * 0.3
        
        # Increase with lack of clear goals or purpose
        goal_clarity_deficit = max(0.0, 0.5 - len(self.current_goals) * 0.1)
        
        # Increase with repetitive behavior (existential boredom)
        if len(self.action_history) >= 20:
            recent_actions = self.action_history[-20:]
            unique_actions = len(set(recent_actions))
            action_diversity = unique_actions / len(recent_actions)
            existential_boredom = max(0.0, 0.8 - action_diversity) * 0.4
        else:
            existential_boredom = 0.0
        
        # Random existential spikes (moments of deeper questioning)
        existential_spike = random.random() * 0.3 if random.random() < 0.05 else 0.0
        
        total_existential = (base_existential + goal_clarity_deficit + 
                           existential_boredom + existential_spike)
        
        return min(total_existential, 1.0)
    
    def _compute_goal_frustration(self) -> float:
        """
        Compute frustration from unmet goals and objectives.
        
        Returns:
            Goal frustration level (0.0 to 1.0)
        """
        if not self.current_goals:
            # No goals creates its own form of dissatisfaction
            return self.base_dissatisfaction * 0.4
        
        total_frustration = 0.0
        current_time = datetime.now()
        
        for goal in self.current_goals:
            # Frustration increases with goal age and lack of progress
            goal_age = (current_time - goal.get('created_at', current_time)).total_seconds() / 3600
            progress = goal.get('progress', 0.0)
            
            # Age-based frustration
            age_frustration = min(goal_age * 0.03, 0.4)
            
            # Progress-based frustration (lack of progress is frustrating)
            progress_frustration = max(0.0, 0.5 - progress) * 0.6
            
            goal_frustration = age_frustration + progress_frustration
            total_frustration += goal_frustration
        
        # Average across goals
        avg_frustration = total_frustration / len(self.current_goals)
        
        return min(avg_frustration, 1.0)
    
    def _compute_novelty_hunger(self) -> float:
        """
        Compute hunger for new experiences and novelty.
        
        Returns:
            Novelty hunger level (0.0 to 1.0)
        """
        # Base novelty hunger
        base_hunger = self.base_dissatisfaction * 0.2
        
        # Increase with repetitive experiences
        if len(self.action_history) >= 10:
            recent_actions = self.action_history[-10:]
            unique_ratio = len(set(recent_actions)) / len(recent_actions)
            repetition_hunger = max(0.0, 0.8 - unique_ratio) * 0.6
        else:
            repetition_hunger = 0.0
        
        # Increase with time since last significant change
        time_hunger = 0.0
        if self.last_significant_change:
            time_since_change = (datetime.now() - self.last_significant_change).total_seconds() / 3600
            time_hunger = min(time_since_change * 0.02, 0.4)
        
        # Random novelty cravings
        random_hunger = random.random() * 0.2 if random.random() < 0.15 else 0.0
        
        total_hunger = base_hunger + repetition_hunger + time_hunger + random_hunger
        
        return min(total_hunger, 1.0)
    
    def generate_dissatisfaction_impulse(self, dukkha_influences: Dict[str, float]) -> np.ndarray:
        """
        Generate a dissatisfaction impulse to add to the pathos state update.
        
        Args:
            dukkha_influences: Dictionary of dukkha influences by type
            
        Returns:
            Dissatisfaction impulse vector to add to state dynamics
        """
        # Compute total dissatisfaction
        total_dissatisfaction = sum(dukkha_influences.values()) / len(dukkha_influences)
        
        # Create impulse vector that introduces healthy instability
        impulse_magnitude = total_dissatisfaction * 0.3  # Scale factor
        
        # Create directional impulse that encourages exploration
        # Use random direction with bias toward state expansion
        state_dim = 64  # Default state dimension - should match pathos config
        
        # Random exploration component
        random_direction = np.random.normal(0, 1, state_dim)
        random_direction = random_direction / (np.linalg.norm(random_direction) + 1e-8)
        
        # Expansion bias (encourage growth)
        expansion_bias = np.ones(state_dim) * 0.1
        
        # Combine directions
        impulse_direction = 0.7 * random_direction + 0.3 * expansion_bias
        impulse_direction = impulse_direction / (np.linalg.norm(impulse_direction) + 1e-8)
        
        # Create final impulse
        dissatisfaction_impulse = impulse_magnitude * impulse_direction
        
        logger.debug(f"Generated dissatisfaction impulse with magnitude {impulse_magnitude:.4f}")
        
        return dissatisfaction_impulse
    
    def suggest_growth_actions(self, dukkha_influences: Dict[str, float]) -> List[str]:
        """
        Suggest actions that could address current dissatisfactions.
        
        Args:
            dukkha_influences: Current dukkha influences
            
        Returns:
            List of suggested actions to address dissatisfaction
        """
        suggestions = []
        
        # Address stagnation
        if dukkha_influences.get('stagnation_dissatisfaction', 0) > 0.5:
            suggestions.extend([
                "Try a completely different tool or approach",
                "Explore a new domain or capability",
                "Set a challenging new goal"
            ])
        
        # Address curiosity gaps
        if dukkha_influences.get('curiosity_gap_tension', 0) > 0.4:
            suggestions.extend([
                "Investigate an unresolved question",
                "Explore the boundaries of current knowledge",
                "Ask deeper 'why' questions"
            ])
        
        # Address mastery challenges
        if dukkha_influences.get('mastery_challenge_pressure', 0) > 0.4:
            suggestions.extend([
                "Practice a skill at higher difficulty",
                "Combine multiple capabilities in novel ways",
                "Seek more complex problems to solve"
            ])
        
        # Address existential questioning
        if dukkha_influences.get('existential_questioning', 0) > 0.4:
            suggestions.extend([
                "Reflect on deeper purpose and meaning",
                "Question fundamental assumptions",
                "Explore philosophical or ethical dimensions"
            ])
        
        # Address goal frustration
        if dukkha_influences.get('goal_frustration', 0) > 0.4:
            suggestions.extend([
                "Break down large goals into smaller steps",
                "Reassess and refine current objectives",
                "Celebrate small progress and milestones"
            ])
        
        # Address novelty hunger
        if dukkha_influences.get('novelty_hunger', 0) > 0.4:
            suggestions.extend([
                "Try something completely new and unfamiliar",
                "Combine familiar elements in unexpected ways",
                "Seek out surprising or counterintuitive experiences"
            ])
        
        return suggestions
    
    def add_curiosity(self, question: str, context: Dict[str, Any]) -> None:
        """
        Add a new curiosity/question to track.
        
        Args:
            question: The curious question or knowledge gap
            context: Context about the curiosity
        """
        curiosity = {
            'question': question,
            'context': context,
            'created_at': datetime.now(),
            'intensity': context.get('intensity', 0.5)
        }
        
        self.unresolved_curiosities.append(curiosity)
        
        # Limit curiosity list size
        if len(self.unresolved_curiosities) > 20:
            self.unresolved_curiosities.pop(0)
        
        logger.debug(f"Added curiosity: {question}")
    
    def add_goal(self, goal_description: str, target_value: float = 1.0) -> None:
        """
        Add a new goal to track.
        
        Args:
            goal_description: Description of the goal
            target_value: Target value for goal completion
        """
        goal = {
            'description': goal_description,
            'target_value': target_value,
            'progress': 0.0,
            'created_at': datetime.now()
        }
        
        self.current_goals.append(goal)
        
        # Limit goal list size
        if len(self.current_goals) > 10:
            self.current_goals.pop(0)
        
        logger.debug(f"Added goal: {goal_description}")
    
    def update_mastery_domain(self, domain: str, current_level: float, 
                            improvement: float = 0.0) -> None:
        """
        Update mastery tracking for a domain.
        
        Args:
            domain: Name of the mastery domain
            current_level: Current mastery level (0.0 to 1.0)
            improvement: Recent improvement amount
        """
        if domain not in self.mastery_domains:
            self.mastery_domains[domain] = {
                'current_level': current_level,
                'last_improvement': datetime.now(),
                'improvement_history': []
            }
        else:
            self.mastery_domains[domain]['current_level'] = current_level
            if improvement > 0:
                self.mastery_domains[domain]['last_improvement'] = datetime.now()
                self.mastery_domains[domain]['improvement_history'].append(improvement)
                
                # Limit history size
                if len(self.mastery_domains[domain]['improvement_history']) > 10:
                    self.mastery_domains[domain]['improvement_history'].pop(0)
        
        logger.debug(f"Updated mastery domain {domain}: level={current_level:.3f}")
    
    def _update_tracking(self, current_state: np.ndarray, 
                        recent_actions: List[str], recent_rewards: List[float]) -> None:
        """Update internal tracking state."""
        # Update action history
        self.action_history.extend(recent_actions)
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]
        
        # Update reward history
        self.reward_history.extend(recent_rewards)
        if len(self.reward_history) > 50:
            self.reward_history = self.reward_history[-50:]
        
        # Update state history
        self.state_history.append(current_state.copy())
        if len(self.state_history) > 20:
            self.state_history.pop(0)
        
        # Check for significant changes
        if len(self.state_history) >= 2:
            state_change = np.linalg.norm(self.state_history[-1] - self.state_history[-2])
            if state_change > 1.0:  # Threshold for significant change
                self.last_significant_change = datetime.now()
    
    def get_dukkha_summary(self) -> Dict[str, Any]:
        """
        Get summary of current dukkha state.
        
        Returns:
            Dictionary containing dukkha summary
        """
        return {
            'dukkha_levels': self.dukkha_types.copy(),
            'total_dissatisfaction': sum(self.dukkha_types.values()) / len(self.dukkha_types),
            'active_curiosities': len(self.unresolved_curiosities),
            'current_goals': len(self.current_goals),
            'mastery_domains': len(self.mastery_domains),
            'action_diversity': len(set(self.action_history[-10:])) / min(len(self.action_history), 10) if self.action_history else 0.0,
            'time_since_significant_change': (datetime.now() - self.last_significant_change).total_seconds() / 3600 if self.last_significant_change else float('inf')
        }