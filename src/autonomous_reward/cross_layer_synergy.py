"""
Cross-Layer Synergy System for the Autonomous State-Derived Reward System.

This module implements the CrossLayerSynergy class that creates synergy between
Logos reasoning, Pathos affective dynamics, and Memory integration, ensuring
all aspects of cognition work together harmoniously.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

from .interfaces import CrossLayerSynergyInterface
from .models import Pattern
from ..core.models import MemoryTrace

logger = logging.getLogger(__name__)


class CrossLayerSynergy(CrossLayerSynergyInterface):
    """
    Creates synergy between Logos reasoning, Pathos affective dynamics, and Memory integration.
    
    This system coordinates reward signals across all layers, provides synergy bonuses
    for harmonious operation, detects emergent behaviors from cross-layer patterns,
    and creates feedback loops that strengthen inter-layer cooperation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cross-layer synergy system.
        
        Args:
            config: Configuration parameters for synergy computation
        """
        self.config = config or {}
        
        # Synergy computation parameters
        self.harmony_threshold = self.config.get('harmony_threshold', 0.7)
        self.synergy_bonus_multiplier = self.config.get('synergy_bonus_multiplier', 1.5)
        self.emergence_detection_window = self.config.get('emergence_detection_window', 10)
        self.feedback_strength = self.config.get('feedback_strength', 0.3)
        
        # Internal state tracking
        self.layer_harmony_history: List[Dict[str, float]] = []
        self.synergy_bonus_history: List[float] = []
        self.emergent_patterns: List[Pattern] = []
        self.cross_layer_patterns: List[Pattern] = []
        self.feedback_adjustments: Dict[str, List[float]] = {
            'logos': [],
            'pathos': [],
            'memory': []
        }
        
        # Pattern detection state
        self.pattern_detection_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = 100
        
        logger.info("Initialized CrossLayerSynergy system")
    
    def coordinate_reward_signals(self, logos_rewards: Dict[str, float], 
                                pathos_rewards: Dict[str, float], 
                                memory_rewards: Dict[str, float]) -> Dict[str, float]:
        """
        Coordinate reward signals across all layers.
        
        This method ensures that reward signals from different layers are properly
        balanced and integrated to create coherent motivation signals.
        
        Args:
            logos_rewards: Rewards from logos layer (reasoning, planning, tool usage)
            pathos_rewards: Rewards from pathos layer (coherence, growth, balance)
            memory_rewards: Rewards from memory layer (integration, recall, novelty)
            
        Returns:
            Coordinated reward signals with balanced contributions from all layers
        """
        # Normalize reward signals from each layer
        normalized_logos = self._normalize_reward_dict(logos_rewards)
        normalized_pathos = self._normalize_reward_dict(pathos_rewards)
        normalized_memory = self._normalize_reward_dict(memory_rewards)
        
        # Compute layer weights based on current harmony levels
        layer_weights = self._compute_layer_weights(
            normalized_logos, normalized_pathos, normalized_memory
        )
        
        # Coordinate rewards with weighted integration
        coordinated_rewards = {}
        
        # Combine all unique reward types
        all_reward_types = set()
        all_reward_types.update(logos_rewards.keys())
        all_reward_types.update(pathos_rewards.keys())
        all_reward_types.update(memory_rewards.keys())
        
        for reward_type in all_reward_types:
            # Get rewards from each layer (default to 0 if not present)
            logos_value = normalized_logos.get(reward_type, 0.0)
            pathos_value = normalized_pathos.get(reward_type, 0.0)
            memory_value = normalized_memory.get(reward_type, 0.0)
            
            # Weighted coordination
            coordinated_value = (
                layer_weights['logos'] * logos_value +
                layer_weights['pathos'] * pathos_value +
                layer_weights['memory'] * memory_value
            )
            
            coordinated_rewards[reward_type] = coordinated_value
        
        # Add cross-layer synergy rewards
        coordinated_rewards['cross_layer_coherence'] = self._compute_cross_layer_coherence(
            normalized_logos, normalized_pathos, normalized_memory
        )
        
        coordinated_rewards['inter_layer_resonance'] = self._compute_inter_layer_resonance(
            logos_rewards, pathos_rewards, memory_rewards
        )
        
        logger.debug(f"Coordinated rewards across layers: {len(coordinated_rewards)} reward types")
        
        return coordinated_rewards
    
    def compute_synergy_bonus(self, layer_harmony: Dict[str, float]) -> float:
        """
        Compute synergy bonus for harmonious layer operation.
        
        When all layers work in harmony, the system provides additional reward
        bonuses that encourage continued cooperation and integration.
        
        Args:
            layer_harmony: Harmony metrics for each layer (logos, pathos, memory)
            
        Returns:
            Synergy bonus value (0.0 to 2.0, with higher values for better harmony)
        """
        # Store harmony history
        self.layer_harmony_history.append(layer_harmony.copy())
        if len(self.layer_harmony_history) > self.max_buffer_size:
            self.layer_harmony_history.pop(0)
        
        # Compute overall harmony score
        harmony_values = list(layer_harmony.values())
        if not harmony_values:
            return 0.0
        
        # Use geometric mean for harmony (all layers must be reasonably harmonious)
        geometric_mean = np.prod(harmony_values) ** (1.0 / len(harmony_values))
        
        # Compute harmony stability (consistency over time)
        stability_bonus = self._compute_harmony_stability()
        
        # Base synergy bonus
        base_bonus = geometric_mean * self.synergy_bonus_multiplier
        
        # Apply stability bonus
        synergy_bonus = base_bonus * (1.0 + stability_bonus)
        
        # Apply harmony threshold bonus (extra reward for exceeding threshold)
        if geometric_mean > self.harmony_threshold:
            threshold_bonus = (geometric_mean - self.harmony_threshold) * 2.0
            synergy_bonus += threshold_bonus
        
        # Ensure bonus is within reasonable bounds
        synergy_bonus = np.clip(synergy_bonus, 0.0, 2.0)
        
        # Store bonus history
        self.synergy_bonus_history.append(synergy_bonus)
        if len(self.synergy_bonus_history) > self.max_buffer_size:
            self.synergy_bonus_history.pop(0)
        
        logger.debug(f"Computed synergy bonus: {synergy_bonus:.4f} "
                    f"(harmony={geometric_mean:.4f}, stability={stability_bonus:.4f})")
        
        return synergy_bonus
    
    def detect_emergent_behaviors(self, cross_layer_patterns: List[Pattern]) -> List[Pattern]:
        """
        Detect emergent behaviors from cross-layer patterns.
        
        Emergent behaviors arise when patterns from different layers combine
        in novel ways, creating new capabilities or behaviors that weren't
        explicitly programmed.
        
        Args:
            cross_layer_patterns: Patterns detected across different layers
            
        Returns:
            List of emergent behavior patterns
        """
        # Store cross-layer patterns
        self.cross_layer_patterns.extend(cross_layer_patterns)
        if len(self.cross_layer_patterns) > self.max_buffer_size:
            self.cross_layer_patterns = self.cross_layer_patterns[-self.max_buffer_size:]
        
        emergent_behaviors = []
        
        # Detect pattern combinations that create emergent behaviors
        emergent_behaviors.extend(self._detect_resonance_patterns())
        emergent_behaviors.extend(self._detect_amplification_patterns())
        emergent_behaviors.extend(self._detect_synthesis_patterns())
        emergent_behaviors.extend(self._detect_meta_cognitive_patterns())
        
        # Store detected emergent patterns
        self.emergent_patterns.extend(emergent_behaviors)
        if len(self.emergent_patterns) > self.max_buffer_size:
            self.emergent_patterns = self.emergent_patterns[-self.max_buffer_size:]
        
        logger.debug(f"Detected {len(emergent_behaviors)} emergent behavior patterns")
        
        return emergent_behaviors
    
    def create_feedback_loops(self, layer_states: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create feedback loops that strengthen inter-layer cooperation.
        
        Feedback loops allow successful patterns in one layer to influence
        and strengthen corresponding patterns in other layers, creating
        self-reinforcing cycles of improvement.
        
        Args:
            layer_states: Current states of all layers (logos, pathos, memory)
            
        Returns:
            Feedback loop adjustments for each layer
        """
        feedback_adjustments = {
            'logos': {},
            'pathos': {},
            'memory': {}
        }
        
        # Extract layer states
        logos_state = layer_states.get('logos', {})
        pathos_state = layer_states.get('pathos', {})
        memory_state = layer_states.get('memory', {})
        
        # Create logos -> pathos feedback loops
        logos_to_pathos = self._create_logos_to_pathos_feedback(logos_state, pathos_state)
        feedback_adjustments['pathos'].update(logos_to_pathos)
        
        # Create pathos -> logos feedback loops
        pathos_to_logos = self._create_pathos_to_logos_feedback(pathos_state, logos_state)
        feedback_adjustments['logos'].update(pathos_to_logos)
        
        # Create memory -> logos feedback loops
        memory_to_logos = self._create_memory_to_logos_feedback(memory_state, logos_state)
        feedback_adjustments['logos'].update(memory_to_logos)
        
        # Create memory -> pathos feedback loops
        memory_to_pathos = self._create_memory_to_pathos_feedback(memory_state, pathos_state)
        feedback_adjustments['pathos'].update(memory_to_pathos)
        
        # Create logos -> memory feedback loops
        logos_to_memory = self._create_logos_to_memory_feedback(logos_state, memory_state)
        feedback_adjustments['memory'].update(logos_to_memory)
        
        # Create pathos -> memory feedback loops
        pathos_to_memory = self._create_pathos_to_memory_feedback(pathos_state, memory_state)
        feedback_adjustments['memory'].update(pathos_to_memory)
        
        # Store feedback adjustments in history
        for layer, adjustments in feedback_adjustments.items():
            if adjustments:
                adjustment_strength = np.mean(list(adjustments.values()))
                self.feedback_adjustments[layer].append(adjustment_strength)
                if len(self.feedback_adjustments[layer]) > self.max_buffer_size:
                    self.feedback_adjustments[layer].pop(0)
        
        logger.debug(f"Created feedback loops: "
                    f"logos={len(feedback_adjustments['logos'])}, "
                    f"pathos={len(feedback_adjustments['pathos'])}, "
                    f"memory={len(feedback_adjustments['memory'])}")
        
        return feedback_adjustments
    
    def get_synergy_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive synergy metrics for monitoring and analysis.
        
        Returns:
            Dictionary containing synergy performance metrics
        """
        recent_harmony = self.layer_harmony_history[-10:] if self.layer_harmony_history else []
        recent_bonuses = self.synergy_bonus_history[-10:] if self.synergy_bonus_history else []
        
        return {
            'current_harmony': self.layer_harmony_history[-1] if self.layer_harmony_history else {},
            'average_harmony': self._compute_average_harmony(recent_harmony),
            'harmony_stability': self._compute_harmony_stability(),
            'current_synergy_bonus': self.synergy_bonus_history[-1] if self.synergy_bonus_history else 0.0,
            'average_synergy_bonus': np.mean(recent_bonuses) if recent_bonuses else 0.0,
            'emergent_pattern_count': len(self.emergent_patterns),
            'cross_layer_pattern_count': len(self.cross_layer_patterns),
            'feedback_loop_strength': {
                layer: np.mean(adjustments[-5:]) if adjustments else 0.0
                for layer, adjustments in self.feedback_adjustments.items()
            }
        }
    
    # Private helper methods
    
    def _normalize_reward_dict(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Normalize reward values to [0, 1] range."""
        if not rewards:
            return {}
        
        values = list(rewards.values())
        if not values:
            return rewards
        
        max_val = max(values)
        min_val = min(values)
        
        if max_val == min_val:
            return {k: 0.5 for k in rewards.keys()}
        
        normalized = {}
        for k, v in rewards.items():
            normalized[k] = (v - min_val) / (max_val - min_val)
        
        return normalized
    
    def _compute_layer_weights(self, logos_rewards: Dict[str, float], 
                             pathos_rewards: Dict[str, float], 
                             memory_rewards: Dict[str, float]) -> Dict[str, float]:
        """Compute dynamic weights for each layer based on current performance."""
        # Compute average reward for each layer
        logos_avg = np.mean(list(logos_rewards.values())) if logos_rewards else 0.0
        pathos_avg = np.mean(list(pathos_rewards.values())) if pathos_rewards else 0.0
        memory_avg = np.mean(list(memory_rewards.values())) if memory_rewards else 0.0
        
        # Base weights (equal by default)
        base_weight = 1.0 / 3.0
        
        # Adjust weights based on performance (higher performing layers get slightly more weight)
        total_performance = logos_avg + pathos_avg + memory_avg
        
        if total_performance > 0:
            logos_weight = base_weight + (logos_avg / total_performance - base_weight) * 0.2
            pathos_weight = base_weight + (pathos_avg / total_performance - base_weight) * 0.2
            memory_weight = base_weight + (memory_avg / total_performance - base_weight) * 0.2
        else:
            logos_weight = pathos_weight = memory_weight = base_weight
        
        # Ensure weights sum to 1.0
        total_weight = logos_weight + pathos_weight + memory_weight
        if total_weight > 0:
            logos_weight /= total_weight
            pathos_weight /= total_weight
            memory_weight /= total_weight
        
        return {
            'logos': logos_weight,
            'pathos': pathos_weight,
            'memory': memory_weight
        }
    
    def _compute_cross_layer_coherence(self, logos_rewards: Dict[str, float], 
                                     pathos_rewards: Dict[str, float], 
                                     memory_rewards: Dict[str, float]) -> float:
        """Compute coherence between layers based on reward alignment."""
        # Find common reward types
        common_types = set(logos_rewards.keys()) & set(pathos_rewards.keys()) & set(memory_rewards.keys())
        
        if not common_types:
            return 0.0
        
        coherence_scores = []
        for reward_type in common_types:
            logos_val = logos_rewards[reward_type]
            pathos_val = pathos_rewards[reward_type]
            memory_val = memory_rewards[reward_type]
            
            # Compute variance (lower variance = higher coherence)
            values = [logos_val, pathos_val, memory_val]
            variance = np.var(values)
            coherence = 1.0 / (1.0 + variance)  # Convert to coherence score
            coherence_scores.append(coherence)
        
        return np.mean(coherence_scores) if coherence_scores else 0.0
    
    def _compute_inter_layer_resonance(self, logos_rewards: Dict[str, float], 
                                     pathos_rewards: Dict[str, float], 
                                     memory_rewards: Dict[str, float]) -> float:
        """Compute resonance between layers based on reward patterns."""
        # Compute correlation between layer reward patterns
        logos_values = list(logos_rewards.values())
        pathos_values = list(pathos_rewards.values())
        memory_values = list(memory_rewards.values())
        
        if not (logos_values and pathos_values and memory_values):
            return 0.0
        
        # Pad shorter lists with zeros for correlation computation
        max_len = max(len(logos_values), len(pathos_values), len(memory_values))
        logos_padded = logos_values + [0.0] * (max_len - len(logos_values))
        pathos_padded = pathos_values + [0.0] * (max_len - len(pathos_values))
        memory_padded = memory_values + [0.0] * (max_len - len(memory_values))
        
        # Compute pairwise correlations
        correlations = []
        
        if len(set(logos_padded)) > 1 and len(set(pathos_padded)) > 1:
            corr = np.corrcoef(logos_padded, pathos_padded)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        if len(set(logos_padded)) > 1 and len(set(memory_padded)) > 1:
            corr = np.corrcoef(logos_padded, memory_padded)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        if len(set(pathos_padded)) > 1 and len(set(memory_padded)) > 1:
            corr = np.corrcoef(pathos_padded, memory_padded)[0, 1]
            if not np.isnan(corr):
                correlations.append(abs(corr))
        
        return np.mean(correlations) if correlations else 0.0
    
    def _compute_harmony_stability(self) -> float:
        """Compute stability of harmony over time."""
        if len(self.layer_harmony_history) < 2:
            return 0.0
        
        # Compute variance in harmony over recent history
        recent_harmony = self.layer_harmony_history[-10:]
        
        # Extract harmony values for each layer
        layer_stabilities = []
        for layer in ['logos', 'pathos', 'memory']:
            layer_values = [h.get(layer, 0.0) for h in recent_harmony]
            if len(set(layer_values)) > 1:
                stability = 1.0 / (1.0 + np.var(layer_values))
                layer_stabilities.append(stability)
        
        return np.mean(layer_stabilities) if layer_stabilities else 0.0
    
    def _compute_average_harmony(self, harmony_history: List[Dict[str, float]]) -> Dict[str, float]:
        """Compute average harmony for each layer."""
        if not harmony_history:
            return {}
        
        average_harmony = {}
        for layer in ['logos', 'pathos', 'memory']:
            layer_values = [h.get(layer, 0.0) for h in harmony_history]
            average_harmony[layer] = np.mean(layer_values)
        
        return average_harmony
    
    def _detect_resonance_patterns(self) -> List[Pattern]:
        """Detect resonance patterns between layers."""
        patterns = []
        
        # Look for synchronized oscillations between layers
        if len(self.layer_harmony_history) >= self.emergence_detection_window:
            recent_harmony = self.layer_harmony_history[-self.emergence_detection_window:]
            
            # Check for synchronized patterns
            for layer1 in ['logos', 'pathos', 'memory']:
                for layer2 in ['logos', 'pathos', 'memory']:
                    if layer1 != layer2:
                        values1 = [h.get(layer1, 0.0) for h in recent_harmony]
                        values2 = [h.get(layer2, 0.0) for h in recent_harmony]
                        
                        if len(set(values1)) > 1 and len(set(values2)) > 1:
                            correlation = np.corrcoef(values1, values2)[0, 1]
                            if not np.isnan(correlation) and abs(correlation) > 0.8:
                                pattern = Pattern(
                                    pattern_type=f"resonance_{layer1}_{layer2}",
                                    strength=abs(correlation),
                                    frequency=1,
                                    last_occurrence=datetime.now(),
                                    characteristics={
                                        'layer1': layer1,
                                        'layer2': layer2,
                                        'correlation': correlation,
                                        'type': 'resonance'
                                    }
                                )
                                patterns.append(pattern)
        
        return patterns
    
    def _detect_amplification_patterns(self) -> List[Pattern]:
        """Detect amplification patterns where layers boost each other."""
        patterns = []
        
        # Look for patterns where high performance in one layer leads to high performance in others
        if len(self.synergy_bonus_history) >= self.emergence_detection_window:
            recent_bonuses = self.synergy_bonus_history[-self.emergence_detection_window:]
            recent_harmony = self.layer_harmony_history[-self.emergence_detection_window:]
            
            if len(recent_harmony) == len(recent_bonuses):
                # Check if high synergy bonuses correlate with high individual layer performance
                for layer in ['logos', 'pathos', 'memory']:
                    layer_values = [h.get(layer, 0.0) for h in recent_harmony]
                    
                    if len(set(layer_values)) > 1 and len(set(recent_bonuses)) > 1:
                        correlation = np.corrcoef(layer_values, recent_bonuses)[0, 1]
                        if not np.isnan(correlation) and correlation > 0.7:
                            pattern = Pattern(
                                pattern_type=f"amplification_{layer}",
                                strength=correlation,
                                frequency=1,
                                last_occurrence=datetime.now(),
                                characteristics={
                                    'layer': layer,
                                    'correlation': correlation,
                                    'type': 'amplification'
                                }
                            )
                            patterns.append(pattern)
        
        return patterns
    
    def _detect_synthesis_patterns(self) -> List[Pattern]:
        """Detect synthesis patterns where layers create novel combinations."""
        patterns = []
        
        # Look for patterns where multiple layers contribute to emergent behaviors
        if len(self.cross_layer_patterns) >= 3:
            # Group patterns by time windows
            recent_patterns = self.cross_layer_patterns[-20:]  # Last 20 patterns
            
            # Look for patterns that appear across multiple layers simultaneously
            pattern_groups = {}
            for pattern in recent_patterns:
                time_key = pattern.last_occurrence.strftime("%Y-%m-%d-%H-%M")  # Group by minute
                if time_key not in pattern_groups:
                    pattern_groups[time_key] = []
                pattern_groups[time_key].append(pattern)
            
            # Find time windows with patterns from multiple layers
            for time_key, group_patterns in pattern_groups.items():
                if len(group_patterns) >= 2:  # Multiple patterns in same time window
                    pattern_types = set([p.pattern_type for p in group_patterns])
                    if len(pattern_types) >= 2:  # Different pattern types
                        synthesis_pattern = Pattern(
                            pattern_type="synthesis_multi_layer",
                            strength=np.mean([p.strength for p in group_patterns]),
                            frequency=len(group_patterns),
                            last_occurrence=datetime.now(),
                            characteristics={
                                'pattern_types': list(pattern_types),
                                'pattern_count': len(group_patterns),
                                'type': 'synthesis'
                            }
                        )
                        patterns.append(synthesis_pattern)
        
        return patterns
    
    def _detect_meta_cognitive_patterns(self) -> List[Pattern]:
        """Detect meta-cognitive patterns where the system reflects on its own processes."""
        patterns = []
        
        # Look for patterns in feedback loop adjustments
        for layer, adjustments in self.feedback_adjustments.items():
            if len(adjustments) >= self.emergence_detection_window:
                recent_adjustments = adjustments[-self.emergence_detection_window:]
                
                # Check for increasing feedback strength (self-reinforcement)
                if len(set(recent_adjustments)) > 1:
                    # Compute trend
                    x = np.arange(len(recent_adjustments))
                    slope, _ = np.polyfit(x, recent_adjustments, 1)
                    
                    if abs(slope) > 0.01:  # Significant trend
                        pattern = Pattern(
                            pattern_type=f"meta_cognitive_{layer}",
                            strength=abs(slope),
                            frequency=1,
                            last_occurrence=datetime.now(),
                            characteristics={
                                'layer': layer,
                                'trend_slope': slope,
                                'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                                'type': 'meta_cognitive'
                            }
                        )
                        patterns.append(pattern)
        
        return patterns
    
    def _create_logos_to_pathos_feedback(self, logos_state: Dict[str, Any], 
                                       pathos_state: Dict[str, Any]) -> Dict[str, float]:
        """Create feedback from logos to pathos layer."""
        feedback = {}
        
        # If logos shows high reasoning performance, boost pathos coherence
        reasoning_performance = logos_state.get('reasoning_performance', 0.0)
        if reasoning_performance > 0.7:
            feedback['coherence_boost'] = reasoning_performance * self.feedback_strength
        
        # If logos shows high interest, boost pathos growth
        interest_level = logos_state.get('interest_level', 0.0)
        if interest_level > 0.6:
            feedback['growth_boost'] = interest_level * self.feedback_strength
        
        return feedback
    
    def _create_pathos_to_logos_feedback(self, pathos_state: Dict[str, Any], 
                                       logos_state: Dict[str, Any]) -> Dict[str, float]:
        """Create feedback from pathos to logos layer."""
        feedback = {}
        
        # If pathos shows high coherence, boost logos reasoning
        coherence_level = pathos_state.get('coherence_level', 0.0)
        if coherence_level > 0.7:
            feedback['reasoning_boost'] = coherence_level * self.feedback_strength
        
        # If pathos shows high energy, boost logos exploration
        energy_level = pathos_state.get('energy_level', 0.0)
        if energy_level > 0.6:
            feedback['exploration_boost'] = energy_level * self.feedback_strength
        
        return feedback
    
    def _create_memory_to_logos_feedback(self, memory_state: Dict[str, Any], 
                                       logos_state: Dict[str, Any]) -> Dict[str, float]:
        """Create feedback from memory to logos layer."""
        feedback = {}
        
        # If memory shows high integration, boost logos planning
        integration_level = memory_state.get('integration_level', 0.0)
        if integration_level > 0.7:
            feedback['planning_boost'] = integration_level * self.feedback_strength
        
        # If memory shows high novelty detection, boost logos curiosity
        novelty_level = memory_state.get('novelty_level', 0.0)
        if novelty_level > 0.6:
            feedback['curiosity_boost'] = novelty_level * self.feedback_strength
        
        return feedback
    
    def _create_memory_to_pathos_feedback(self, memory_state: Dict[str, Any], 
                                        pathos_state: Dict[str, Any]) -> Dict[str, float]:
        """Create feedback from memory to pathos layer."""
        feedback = {}
        
        # If memory shows high recall success, boost pathos stability
        recall_success = memory_state.get('recall_success', 0.0)
        if recall_success > 0.7:
            feedback['stability_boost'] = recall_success * self.feedback_strength
        
        # If memory shows rich associations, boost pathos integration
        association_richness = memory_state.get('association_richness', 0.0)
        if association_richness > 0.6:
            feedback['integration_boost'] = association_richness * self.feedback_strength
        
        return feedback
    
    def _create_logos_to_memory_feedback(self, logos_state: Dict[str, Any], 
                                       memory_state: Dict[str, Any]) -> Dict[str, float]:
        """Create feedback from logos to memory layer."""
        feedback = {}
        
        # If logos shows high semantic processing, boost memory encoding
        semantic_processing = logos_state.get('semantic_processing', 0.0)
        if semantic_processing > 0.7:
            feedback['encoding_boost'] = semantic_processing * self.feedback_strength
        
        # If logos shows high tool usage success, boost memory consolidation
        tool_success = logos_state.get('tool_success', 0.0)
        if tool_success > 0.6:
            feedback['consolidation_boost'] = tool_success * self.feedback_strength
        
        return feedback
    
    def _create_pathos_to_memory_feedback(self, pathos_state: Dict[str, Any], 
                                        memory_state: Dict[str, Any]) -> Dict[str, float]:
        """Create feedback from pathos to memory layer."""
        feedback = {}
        
        # If pathos shows high salience, boost memory storage priority
        salience_level = pathos_state.get('salience_level', 0.0)
        if salience_level > 0.7:
            feedback['storage_priority_boost'] = salience_level * self.feedback_strength
        
        # If pathos shows high emotional resonance, boost memory retrieval
        emotional_resonance = pathos_state.get('emotional_resonance', 0.0)
        if emotional_resonance > 0.6:
            feedback['retrieval_boost'] = emotional_resonance * self.feedback_strength
        
        return feedback