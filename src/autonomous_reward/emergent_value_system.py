"""
Emergent Value System for the Autonomous State-Derived Reward System.

This module implements the EmergentValueSystem that develops authentic value systems
through experience and state dynamics, enabling autonomous goal generation and
value transfer across contexts.
"""

import numpy as np
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import logging

from .interfaces import EmergentValueSystemInterface
from .models import ValuePattern, Experience, AutonomousGoal


logger = logging.getLogger(__name__)


class EmergentValueSystem(EmergentValueSystemInterface):
    """
    Develops authentic value system through experience and state dynamics.
    
    This system discovers value patterns from successful experiences, integrates
    new values into existing frameworks, transfers values across contexts, and
    generates autonomous goals based on developed values.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the emergent value system.
        
        Args:
            config: Configuration parameters for value system
        """
        self.config = config or {}
        
        # Core value storage
        self.value_patterns: List[ValuePattern] = []
        self.context_value_map: Dict[str, List[str]] = defaultdict(list)  # context -> value_pattern_ids
        self.value_strength_history: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Experience tracking for value discovery
        self.experience_buffer: List[Experience] = []
        self.reward_threshold = self.config.get('reward_threshold', 0.5)
        self.max_experience_buffer = self.config.get('max_experience_buffer', 1000)
        
        # Value integration parameters
        self.min_pattern_strength = self.config.get('min_pattern_strength', 0.1)
        self.max_value_patterns = self.config.get('max_value_patterns', 100)
        self.value_decay_rate = self.config.get('value_decay_rate', 0.95)
        
        # Goal generation parameters
        self.goal_generation_threshold = self.config.get('goal_generation_threshold', 0.7)
        self.max_autonomous_goals = self.config.get('max_autonomous_goals', 10)
        
        # Context transfer parameters
        self.context_similarity_threshold = self.config.get('context_similarity_threshold', 0.6)
        self.transfer_strength_multiplier = self.config.get('transfer_strength_multiplier', 0.8)
        
        logger.info("EmergentValueSystem initialized with %d value patterns", len(self.value_patterns))
    
    def discover_value_patterns(self, experiences: List[Experience], 
                              rewards: List[float]) -> List[ValuePattern]:
        """
        Discover emerging value patterns from experiences.
        
        Args:
            experiences: List of experiences to analyze
            rewards: Associated reward values for each experience
            
        Returns:
            List of discovered value patterns
        """
        if len(experiences) != len(rewards):
            raise ValueError("Number of experiences must match number of rewards")
        
        if not experiences:
            return []
        
        logger.debug("Discovering value patterns from %d experiences", len(experiences))
        
        # Add experiences to buffer
        self.experience_buffer.extend(experiences)
        
        # Maintain buffer size
        if len(self.experience_buffer) > self.max_experience_buffer:
            self.experience_buffer = self.experience_buffer[-self.max_experience_buffer:]
        
        discovered_patterns = []
        
        # Analyze high-reward experiences for value patterns
        high_reward_experiences = [
            (exp, reward) for exp, reward in zip(experiences, rewards)
            if reward >= self.reward_threshold
        ]
        
        if not high_reward_experiences:
            logger.debug("No high-reward experiences found for value discovery")
            return discovered_patterns
        
        # Group experiences by action type to find patterns
        action_groups = defaultdict(list)
        for exp, reward in high_reward_experiences:
            action_groups[exp.action_taken].append((exp, reward))
        
        # Discover patterns within each action group
        for action_type, exp_reward_pairs in action_groups.items():
            if len(exp_reward_pairs) >= 2:  # Need multiple instances to establish pattern
                pattern = self._analyze_action_pattern(action_type, exp_reward_pairs)
                if pattern:
                    discovered_patterns.append(pattern)
        
        # Discover context-based patterns
        context_patterns = self._discover_context_patterns(high_reward_experiences)
        discovered_patterns.extend(context_patterns)
        
        # Discover state transition patterns
        state_patterns = self._discover_state_transition_patterns(high_reward_experiences)
        discovered_patterns.extend(state_patterns)
        
        logger.info("Discovered %d new value patterns", len(discovered_patterns))
        return discovered_patterns
    
    def integrate_new_values(self, new_values: List[ValuePattern], 
                           existing_values: List[ValuePattern]) -> List[ValuePattern]:
        """
        Integrate new values into existing value system.
        
        Args:
            new_values: Newly discovered values
            existing_values: Existing value system
            
        Returns:
            Updated value system
        """
        logger.debug("Integrating %d new values with %d existing values", 
                    len(new_values), len(existing_values))
        
        # Start with existing values
        integrated_values = existing_values.copy()
        
        for new_value in new_values:
            # Check if this value pattern already exists
            existing_match = self._find_similar_value_pattern(new_value, integrated_values)
            
            if existing_match:
                # Strengthen existing pattern
                self._strengthen_value_pattern(existing_match, new_value)
                logger.debug("Strengthened existing value pattern: %s", existing_match.pattern_id)
            else:
                # Add new pattern if we haven't reached the limit
                if len(integrated_values) < self.max_value_patterns:
                    integrated_values.append(new_value)
                    self._track_value_strength(new_value)
                    logger.debug("Added new value pattern: %s", new_value.pattern_id)
                else:
                    # Replace weakest pattern if new one is stronger
                    weakest_pattern = min(integrated_values, key=lambda v: v.strength)
                    if new_value.strength > weakest_pattern.strength:
                        integrated_values.remove(weakest_pattern)
                        integrated_values.append(new_value)
                        self._track_value_strength(new_value)
                        logger.debug("Replaced weak value pattern with stronger one: %s -> %s",
                                   weakest_pattern.pattern_id, new_value.pattern_id)
        
        # Apply value decay to maintain relevance
        integrated_values = self._apply_value_decay(integrated_values)
        
        # Update internal state
        self.value_patterns = integrated_values
        self._update_context_mappings()
        
        logger.info("Integration complete: %d total value patterns", len(integrated_values))
        return integrated_values
    
    def transfer_values_across_contexts(self, source_context: str, 
                                      target_context: str) -> List[ValuePattern]:
        """
        Transfer learned values across different contexts.
        
        Args:
            source_context: Source context for value transfer
            target_context: Target context for value application
            
        Returns:
            Transferred value patterns
        """
        logger.debug("Transferring values from context '%s' to '%s'", source_context, target_context)
        
        # Get values from source context
        source_values = self._get_values_for_context(source_context)
        
        if not source_values:
            logger.debug("No values found for source context: %s", source_context)
            return []
        
        transferred_patterns = []
        
        for value_pattern in source_values:
            # Check if this value can be transferred to target context
            if self._can_transfer_value(value_pattern, source_context, target_context):
                # Create transferred pattern
                transferred_pattern = self._create_transferred_pattern(
                    value_pattern, source_context, target_context
                )
                transferred_patterns.append(transferred_pattern)
                
                # Update context mappings
                if target_context not in transferred_pattern.contexts:
                    transferred_pattern.contexts.append(target_context)
                
                logger.debug("Transferred value pattern %s to context %s", 
                           transferred_pattern.pattern_id, target_context)
        
        logger.info("Transferred %d value patterns to context '%s'", 
                   len(transferred_patterns), target_context)
        return transferred_patterns
    
    def generate_autonomous_goals(self, current_values: List[ValuePattern], 
                                current_state: np.ndarray) -> List[AutonomousGoal]:
        """
        Generate autonomous goals based on developed values.
        
        Args:
            current_values: Current value system
            current_state: Current pathos state
            
        Returns:
            List of generated autonomous goals
        """
        logger.debug("Generating autonomous goals from %d value patterns", len(current_values))
        
        if not current_values:
            logger.debug("No value patterns available for goal generation")
            return []
        
        generated_goals = []
        
        # Filter values strong enough to generate goals
        strong_values = [
            value for value in current_values 
            if value.strength >= self.goal_generation_threshold
        ]
        
        if not strong_values:
            logger.debug("No values strong enough for goal generation (threshold: %.2f)", 
                        self.goal_generation_threshold)
            return []
        
        # Generate goals based on different value types
        for value_pattern in strong_values[:self.max_autonomous_goals]:
            goal = self._generate_goal_from_value(value_pattern, current_state)
            if goal:
                generated_goals.append(goal)
                logger.debug("Generated goal: %s", goal.description)
        
        # Sort goals by expected reward (highest first)
        generated_goals.sort(key=lambda g: g.expected_reward, reverse=True)
        
        logger.info("Generated %d autonomous goals", len(generated_goals))
        return generated_goals
    
    def _analyze_action_pattern(self, action_type: str, 
                              exp_reward_pairs: List[Tuple[Experience, float]]) -> Optional[ValuePattern]:
        """Analyze action patterns to discover values."""
        if len(exp_reward_pairs) < 2:
            return None
        
        # Calculate average reward for this action type
        avg_reward = np.mean([reward for _, reward in exp_reward_pairs])
        
        # Extract contexts where this action was successful
        contexts = []
        associated_rewards = []
        
        for exp, reward in exp_reward_pairs:
            if 'context' in exp.context:
                contexts.append(exp.context['context'])
            associated_rewards.append(reward)
        
        # Create value pattern if action shows consistent positive outcomes
        if avg_reward > self.reward_threshold:
            pattern_id = str(uuid.uuid4())
            return ValuePattern(
                pattern_id=pattern_id,
                value_type=f"action_preference_{action_type}",
                strength=min(avg_reward, 1.0),  # Normalize strength
                contexts=list(set(contexts)) if contexts else ["general"],
                associated_rewards=associated_rewards,
                emergence_date=datetime.now()
            )
        
        return None
    
    def _discover_context_patterns(self, high_reward_experiences: List[Tuple[Experience, float]]) -> List[ValuePattern]:
        """Discover patterns based on contexts."""
        context_groups = defaultdict(list)
        
        for exp, reward in high_reward_experiences:
            context_key = exp.context.get('context', 'unknown')
            context_groups[context_key].append((exp, reward))
        
        patterns = []
        for context, exp_reward_pairs in context_groups.items():
            if len(exp_reward_pairs) >= 2:  # Need multiple instances
                avg_reward = np.mean([reward for _, reward in exp_reward_pairs])
                
                if avg_reward > self.reward_threshold:
                    pattern_id = str(uuid.uuid4())
                    pattern = ValuePattern(
                        pattern_id=pattern_id,
                        value_type=f"context_preference_{context}",
                        strength=min(avg_reward, 1.0),
                        contexts=[context],
                        associated_rewards=[reward for _, reward in exp_reward_pairs],
                        emergence_date=datetime.now()
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _discover_state_transition_patterns(self, high_reward_experiences: List[Tuple[Experience, float]]) -> List[ValuePattern]:
        """Discover patterns based on state transitions."""
        patterns = []
        
        # Group by similar state transition magnitudes
        transition_groups = defaultdict(list)
        
        for exp, reward in high_reward_experiences:
            transition_magnitude = np.linalg.norm(exp.state_after - exp.state_before)
            magnitude_key = round(transition_magnitude, 1)  # Group by 0.1 intervals
            transition_groups[magnitude_key].append((exp, reward))
        
        for magnitude, exp_reward_pairs in transition_groups.items():
            if len(exp_reward_pairs) >= 2:
                avg_reward = np.mean([reward for _, reward in exp_reward_pairs])
                
                if avg_reward > self.reward_threshold:
                    pattern_id = str(uuid.uuid4())
                    
                    # Determine value type based on transition characteristics
                    if magnitude < 0.5:
                        value_type = "stability_preference"
                    elif magnitude > 2.0:
                        value_type = "growth_preference"
                    else:
                        value_type = "balanced_change_preference"
                    
                    contexts = []
                    for exp, _ in exp_reward_pairs:
                        if 'context' in exp.context:
                            contexts.append(exp.context['context'])
                    
                    pattern = ValuePattern(
                        pattern_id=pattern_id,
                        value_type=value_type,
                        strength=min(avg_reward, 1.0),
                        contexts=list(set(contexts)) if contexts else ["general"],
                        associated_rewards=[reward for _, reward in exp_reward_pairs],
                        emergence_date=datetime.now()
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _find_similar_value_pattern(self, new_pattern: ValuePattern, 
                                  existing_patterns: List[ValuePattern]) -> Optional[ValuePattern]:
        """Find similar existing value pattern."""
        for existing in existing_patterns:
            # Check if patterns are of the same type
            if existing.value_type == new_pattern.value_type:
                # Check context overlap
                context_overlap = set(existing.contexts) & set(new_pattern.contexts)
                if context_overlap or (not existing.contexts and not new_pattern.contexts):
                    return existing
        
        return None
    
    def _strengthen_value_pattern(self, existing_pattern: ValuePattern, new_pattern: ValuePattern):
        """Strengthen existing value pattern with new evidence."""
        # Update strength (weighted average)
        total_evidence = len(existing_pattern.associated_rewards) + len(new_pattern.associated_rewards)
        existing_weight = len(existing_pattern.associated_rewards) / total_evidence
        new_weight = len(new_pattern.associated_rewards) / total_evidence
        
        existing_pattern.strength = (
            existing_pattern.strength * existing_weight + 
            new_pattern.strength * new_weight
        )
        
        # Merge contexts
        existing_pattern.contexts = list(set(existing_pattern.contexts + new_pattern.contexts))
        
        # Merge rewards (keep recent ones)
        all_rewards = existing_pattern.associated_rewards + new_pattern.associated_rewards
        existing_pattern.associated_rewards = all_rewards[-50:]  # Keep last 50 rewards
        
        # Track strength change
        self._track_value_strength(existing_pattern)
    
    def _track_value_strength(self, value_pattern: ValuePattern):
        """Track value pattern strength over time."""
        self.value_strength_history[value_pattern.pattern_id].append(
            (datetime.now(), value_pattern.strength)
        )
        
        # Keep only recent history
        history = self.value_strength_history[value_pattern.pattern_id]
        cutoff_date = datetime.now() - timedelta(days=30)
        self.value_strength_history[value_pattern.pattern_id] = [
            (date, strength) for date, strength in history if date > cutoff_date
        ]
    
    def _apply_value_decay(self, value_patterns: List[ValuePattern]) -> List[ValuePattern]:
        """Apply decay to value patterns to maintain relevance."""
        current_time = datetime.now()
        
        for pattern in value_patterns:
            # Calculate age in days
            age_days = (current_time - pattern.emergence_date).days
            
            # Apply decay based on age (older patterns decay more)
            if age_days > 0:
                decay_factor = self.value_decay_rate ** (age_days / 7.0)  # Weekly decay
                pattern.strength *= decay_factor
        
        # Remove patterns that have become too weak
        return [pattern for pattern in value_patterns if pattern.strength >= self.min_pattern_strength]
    
    def _update_context_mappings(self):
        """Update context to value pattern mappings."""
        self.context_value_map.clear()
        
        for pattern in self.value_patterns:
            for context in pattern.contexts:
                self.context_value_map[context].append(pattern.pattern_id)
    
    def _get_values_for_context(self, context: str) -> List[ValuePattern]:
        """Get value patterns associated with a specific context."""
        pattern_ids = self.context_value_map.get(context, [])
        return [
            pattern for pattern in self.value_patterns 
            if pattern.pattern_id in pattern_ids
        ]
    
    def _can_transfer_value(self, value_pattern: ValuePattern, 
                          source_context: str, target_context: str) -> bool:
        """Check if a value can be transferred between contexts."""
        # Always allow transfer for general values
        if "general" in value_pattern.contexts:
            return True
        
        # Check if contexts are similar enough for transfer
        similarity = self._compute_context_similarity(source_context, target_context)
        return similarity >= self.context_similarity_threshold
    
    def _compute_context_similarity(self, context1: str, context2: str) -> float:
        """Compute similarity between two contexts."""
        if context1 == context2:
            return 1.0
        
        # Simple string similarity (can be enhanced with semantic similarity)
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union) if union else 0.0
    
    def _create_transferred_pattern(self, original_pattern: ValuePattern, 
                                  source_context: str, target_context: str) -> ValuePattern:
        """Create a transferred value pattern for the target context."""
        # Reduce strength for transferred patterns
        transferred_strength = original_pattern.strength * self.transfer_strength_multiplier
        
        # Create new pattern ID for the transfer
        pattern_id = str(uuid.uuid4())
        
        return ValuePattern(
            pattern_id=pattern_id,
            value_type=original_pattern.value_type,
            strength=transferred_strength,
            contexts=[target_context],
            associated_rewards=original_pattern.associated_rewards.copy(),
            emergence_date=datetime.now()
        )
    
    def _generate_goal_from_value(self, value_pattern: ValuePattern, 
                                current_state: np.ndarray) -> Optional[AutonomousGoal]:
        """Generate an autonomous goal from a value pattern."""
        # Generate goal based on value type
        if "action_preference" in value_pattern.value_type:
            return self._generate_action_goal(value_pattern, current_state)
        elif "context_preference" in value_pattern.value_type:
            return self._generate_context_goal(value_pattern, current_state)
        elif "growth_preference" in value_pattern.value_type:
            return self._generate_growth_goal(value_pattern, current_state)
        elif "stability_preference" in value_pattern.value_type:
            return self._generate_stability_goal(value_pattern, current_state)
        else:
            return self._generate_general_goal(value_pattern, current_state)
    
    def _generate_action_goal(self, value_pattern: ValuePattern, 
                            current_state: np.ndarray) -> AutonomousGoal:
        """Generate goal based on action preference."""
        action_type = value_pattern.value_type.replace("action_preference_", "")
        
        goal_id = str(uuid.uuid4())
        description = f"Pursue {action_type} actions in {', '.join(value_pattern.contexts)} contexts"
        
        return AutonomousGoal(
            goal_id=goal_id,
            description=description,
            value_alignment=value_pattern.strength,
            complexity_level=0.5,  # Medium complexity for action goals
            expected_reward=np.mean(value_pattern.associated_rewards) if value_pattern.associated_rewards else 0.5,
            creation_state=current_state.copy()
        )
    
    def _generate_context_goal(self, value_pattern: ValuePattern, 
                             current_state: np.ndarray) -> AutonomousGoal:
        """Generate goal based on context preference."""
        context = value_pattern.value_type.replace("context_preference_", "")
        
        goal_id = str(uuid.uuid4())
        description = f"Engage more in {context} context to maximize value alignment"
        
        return AutonomousGoal(
            goal_id=goal_id,
            description=description,
            value_alignment=value_pattern.strength,
            complexity_level=0.4,  # Lower complexity for context goals
            expected_reward=np.mean(value_pattern.associated_rewards) if value_pattern.associated_rewards else 0.6,
            creation_state=current_state.copy()
        )
    
    def _generate_growth_goal(self, value_pattern: ValuePattern, 
                            current_state: np.ndarray) -> AutonomousGoal:
        """Generate goal based on growth preference."""
        goal_id = str(uuid.uuid4())
        description = "Pursue opportunities for significant state growth and capability expansion"
        
        return AutonomousGoal(
            goal_id=goal_id,
            description=description,
            value_alignment=value_pattern.strength,
            complexity_level=0.8,  # High complexity for growth goals
            expected_reward=np.mean(value_pattern.associated_rewards) if value_pattern.associated_rewards else 0.7,
            creation_state=current_state.copy()
        )
    
    def _generate_stability_goal(self, value_pattern: ValuePattern, 
                               current_state: np.ndarray) -> AutonomousGoal:
        """Generate goal based on stability preference."""
        goal_id = str(uuid.uuid4())
        description = "Maintain current state stability while making incremental improvements"
        
        return AutonomousGoal(
            goal_id=goal_id,
            description=description,
            value_alignment=value_pattern.strength,
            complexity_level=0.3,  # Low complexity for stability goals
            expected_reward=np.mean(value_pattern.associated_rewards) if value_pattern.associated_rewards else 0.4,
            creation_state=current_state.copy()
        )
    
    def _generate_general_goal(self, value_pattern: ValuePattern, 
                             current_state: np.ndarray) -> AutonomousGoal:
        """Generate general goal from value pattern."""
        goal_id = str(uuid.uuid4())
        description = f"Pursue activities aligned with {value_pattern.value_type} values"
        
        return AutonomousGoal(
            goal_id=goal_id,
            description=description,
            value_alignment=value_pattern.strength,
            complexity_level=0.5,  # Medium complexity for general goals
            expected_reward=np.mean(value_pattern.associated_rewards) if value_pattern.associated_rewards else 0.5,
            creation_state=current_state.copy()
        )
    
    def update_values_from_experience(self, experience: Experience, reward: float):
        """
        Update value system based on a single experience.
        
        Args:
            experience: The experience to learn from
            reward: The reward received from this experience
        """
        logger.debug("Updating values from experience with reward: %.3f", reward)
        
        # Discover patterns from this single experience
        discovered_patterns = self.discover_value_patterns([experience], [reward])
        
        # Integrate new patterns into existing value system
        if discovered_patterns:
            self.integrate_new_values(discovered_patterns, self.value_patterns)
            logger.debug("Integrated %d new patterns from experience", len(discovered_patterns))
    
    def get_discovered_patterns(self) -> List[ValuePattern]:
        """
        Get all discovered value patterns.
        
        Returns:
            List of all value patterns discovered by the system
        """
        return self.value_patterns.copy()
    
    def get_value_system_summary(self) -> Dict[str, Any]:
        """Get summary of current value system."""
        if not self.value_patterns:
            return {
                'total_patterns': 0,
                'strongest_values': [],
                'contexts': [],
                'average_strength': 0.0
            }
        
        # Sort by strength
        sorted_patterns = sorted(self.value_patterns, key=lambda v: v.strength, reverse=True)
        
        # Get unique contexts
        all_contexts = set()
        for pattern in self.value_patterns:
            all_contexts.update(pattern.contexts)
        
        return {
            'total_patterns': len(self.value_patterns),
            'strongest_values': [
                {
                    'type': pattern.value_type,
                    'strength': pattern.strength,
                    'contexts': pattern.contexts
                }
                for pattern in sorted_patterns[:5]  # Top 5
            ],
            'contexts': list(all_contexts),
            'average_strength': np.mean([pattern.strength for pattern in self.value_patterns])
        }