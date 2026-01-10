"""
Self-Success Metrics System for the Autonomous State-Derived Reward System.

This module implements the SelfSuccessMetricsSystem that measures success based on
internal state satisfaction and growth, enabling the agent to evaluate progress
according to its own authentic standards.
"""

import numpy as np
from typing import Dict, List, Any, Set, Optional
import logging
from datetime import datetime, timedelta

from .interfaces import SelfSuccessMetricsSystemInterface
from .models import ValuePattern, Pattern

logger = logging.getLogger(__name__)


class SelfSuccessMetricsSystem(SelfSuccessMetricsSystemInterface):
    """
    Measures success based on internal state satisfaction and growth.
    
    This system enables the agent to evaluate its own progress according to
    self-derived standards rather than external metrics, supporting authentic
    autonomous development.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the self-success metrics system.
        
        Args:
            config: Configuration parameters for the metrics system
        """
        self.config = config or {}
        
        # Configuration parameters
        self.harmony_threshold = self.config.get('harmony_threshold', 0.7)
        self.growth_window_days = self.config.get('growth_window_days', 7)
        self.autonomy_weight = self.config.get('autonomy_weight', 0.4)
        self.external_influence_threshold = self.config.get('external_influence_threshold', 0.3)
        self.value_alignment_threshold = self.config.get('value_alignment_threshold', 0.6)
        
        # Internal tracking
        self.harmony_history: List[Dict[str, Any]] = []
        self.growth_measurements: List[Dict[str, Any]] = []
        self.autonomy_measurements: List[Dict[str, Any]] = []
        self.value_alignment_history: List[Dict[str, Any]] = []
        
        # Harmony pattern templates
        self.harmony_patterns = self._initialize_harmony_patterns()
        
        logger.info("Initialized SelfSuccessMetricsSystem")
    
    def compute_state_harmony_index(self, state: np.ndarray, 
                                  harmony_patterns: List[Pattern]) -> float:
        """
        Compute state harmony index measuring internal coherence and balance.
        
        The harmony index reflects how well the current state aligns with
        patterns that have historically led to positive outcomes and growth.
        
        Args:
            state: Current pathos state
            harmony_patterns: Known harmony patterns from experience
            
        Returns:
            State harmony index (0.0 to 1.0, higher is better)
        """
        if len(state) == 0:
            return 0.0
        
        # Compute basic state coherence
        state_coherence = self._compute_state_coherence(state)
        
        # Compute balance across state dimensions
        state_balance = self._compute_state_balance(state)
        
        # Compute alignment with known harmony patterns
        pattern_alignment = self._compute_pattern_alignment(state, harmony_patterns)
        
        # Compute stability (low variance indicates stability)
        state_stability = self._compute_state_stability(state)
        
        # Combine components with weights
        harmony_index = (
            state_coherence * 0.3 +
            state_balance * 0.25 +
            pattern_alignment * 0.3 +
            state_stability * 0.15
        )
        
        # Store measurement
        measurement = {
            'timestamp': datetime.now(),
            'harmony_index': harmony_index,
            'coherence': state_coherence,
            'balance': state_balance,
            'pattern_alignment': pattern_alignment,
            'stability': state_stability,
            'state_norm': float(np.linalg.norm(state))
        }
        
        self.harmony_history.append(measurement)
        if len(self.harmony_history) > 1000:  # Keep last 1000 measurements
            self.harmony_history.pop(0)
        
        logger.debug(f"Computed state harmony index: {harmony_index:.4f} "
                    f"(coherence={state_coherence:.3f}, balance={state_balance:.3f}, "
                    f"pattern_alignment={pattern_alignment:.3f}, stability={state_stability:.3f})")
        
        return harmony_index
    
    def track_growth_trajectory(self, historical_capabilities: List[Set[str]], 
                              current_capabilities: Set[str]) -> Dict[str, float]:
        """
        Track growth trajectory measuring capability and knowledge expansion.
        
        Args:
            historical_capabilities: Historical capability sets over time
            current_capabilities: Current capability set
            
        Returns:
            Growth trajectory metrics including rate, acceleration, and breadth
        """
        if not historical_capabilities:
            return {
                'growth_rate': 0.0,
                'growth_acceleration': 0.0,
                'capability_breadth': len(current_capabilities),
                'growth_consistency': 0.0,
                'trajectory_score': 0.0
            }
        
        # Compute growth rate (capabilities gained per time period)
        capability_counts = [len(caps) for caps in historical_capabilities]
        capability_counts.append(len(current_capabilities))
        
        if len(capability_counts) < 2:
            growth_rate = 0.0
            growth_acceleration = 0.0
        else:
            # Compute growth rate as average increase per period
            growth_deltas = [capability_counts[i] - capability_counts[i-1] 
                           for i in range(1, len(capability_counts))]
            growth_rate = np.mean(growth_deltas) if growth_deltas else 0.0
            
            # Compute growth acceleration (change in growth rate)
            if len(growth_deltas) >= 2:
                recent_growth = np.mean(growth_deltas[-3:]) if len(growth_deltas) >= 3 else growth_deltas[-1]
                early_growth = np.mean(growth_deltas[:3]) if len(growth_deltas) >= 3 else growth_deltas[0]
                growth_acceleration = recent_growth - early_growth
            else:
                growth_acceleration = 0.0
        
        # Compute capability breadth (diversity of capabilities)
        capability_breadth = len(current_capabilities)
        
        # Compute growth consistency (how steady the growth has been)
        if len(capability_counts) >= 3:
            growth_variance = np.var(growth_deltas) if growth_deltas else 0.0
            growth_consistency = 1.0 / (1.0 + growth_variance)  # Higher consistency for lower variance
        else:
            growth_consistency = 0.5  # Neutral for insufficient data
        
        # Compute overall trajectory score
        trajectory_score = (
            min(growth_rate / 2.0, 1.0) * 0.4 +  # Normalize growth rate
            np.tanh(growth_acceleration) * 0.2 +   # Bounded acceleration
            min(capability_breadth / 20.0, 1.0) * 0.2 +  # Normalize breadth
            growth_consistency * 0.2
        )
        
        growth_metrics = {
            'growth_rate': float(growth_rate),
            'growth_acceleration': float(growth_acceleration),
            'capability_breadth': float(capability_breadth),
            'growth_consistency': float(growth_consistency),
            'trajectory_score': float(trajectory_score)
        }
        
        # Store measurement
        measurement = {
            'timestamp': datetime.now(),
            'metrics': growth_metrics,
            'current_capability_count': len(current_capabilities),
            'historical_count': len(historical_capabilities)
        }
        
        self.growth_measurements.append(measurement)
        if len(self.growth_measurements) > 100:  # Keep last 100 measurements
            self.growth_measurements.pop(0)
        
        logger.debug(f"Tracked growth trajectory: score={trajectory_score:.4f}, "
                    f"rate={growth_rate:.3f}, acceleration={growth_acceleration:.3f}, "
                    f"breadth={capability_breadth}, consistency={growth_consistency:.3f}")
        
        return growth_metrics
    
    def measure_autonomy_level(self, decision_history: List[Dict[str, Any]], 
                             external_influence: float) -> float:
        """
        Measure current autonomy level based on self-direction and independence.
        
        Args:
            decision_history: History of decisions made by the agent
            external_influence: Level of external influence (0.0 to 1.0)
            
        Returns:
            Autonomy level measurement (0.0 to 1.0, higher is more autonomous)
        """
        if not decision_history:
            return 0.0
        
        # Compute self-direction ratio
        self_directed_decisions = sum(1 for decision in decision_history 
                                    if decision.get('source', 'external') == 'self')
        total_decisions = len(decision_history)
        self_direction_ratio = self_directed_decisions / total_decisions
        
        # Compute decision complexity (autonomous decisions on complex matters)
        complex_decisions = sum(1 for decision in decision_history 
                              if decision.get('complexity', 0.0) > 0.5 and 
                                 decision.get('source', 'external') == 'self')
        complexity_autonomy = complex_decisions / max(self_directed_decisions, 1)
        
        # Compute decision consistency (consistent with agent's values/patterns)
        consistent_decisions = sum(1 for decision in decision_history 
                                 if decision.get('value_aligned', False))
        consistency_ratio = consistent_decisions / total_decisions
        
        # Compute independence from external influence
        independence_score = max(0.0, 1.0 - external_influence)
        
        # Compute initiative taking (proactive vs reactive decisions)
        proactive_decisions = sum(1 for decision in decision_history 
                                if decision.get('type', 'reactive') == 'proactive')
        initiative_ratio = proactive_decisions / total_decisions
        
        # Combine components
        autonomy_level = (
            self_direction_ratio * 0.3 +
            complexity_autonomy * 0.2 +
            consistency_ratio * 0.2 +
            independence_score * 0.2 +
            initiative_ratio * 0.1
        )
        
        # Store measurement
        measurement = {
            'timestamp': datetime.now(),
            'autonomy_level': autonomy_level,
            'self_direction_ratio': self_direction_ratio,
            'complexity_autonomy': complexity_autonomy,
            'consistency_ratio': consistency_ratio,
            'independence_score': independence_score,
            'initiative_ratio': initiative_ratio,
            'external_influence': external_influence,
            'total_decisions': total_decisions
        }
        
        self.autonomy_measurements.append(measurement)
        if len(self.autonomy_measurements) > 100:  # Keep last 100 measurements
            self.autonomy_measurements.pop(0)
        
        logger.debug(f"Measured autonomy level: {autonomy_level:.4f} "
                    f"(self_direction={self_direction_ratio:.3f}, "
                    f"complexity={complexity_autonomy:.3f}, "
                    f"consistency={consistency_ratio:.3f}, "
                    f"independence={independence_score:.3f}, "
                    f"initiative={initiative_ratio:.3f})")
        
        return autonomy_level
    
    def assess_value_alignment(self, actions: List[Dict[str, Any]], 
                             values: List[ValuePattern]) -> float:
        """
        Assess alignment between actions and developed values.
        
        Args:
            actions: Recent actions taken by the agent
            values: Current value system (list of ValuePattern objects)
            
        Returns:
            Value alignment score (0.0 to 1.0, higher is better aligned)
        """
        if not actions or not values:
            return 0.0
        
        # Create value lookup for efficient matching
        value_lookup = {value.value_type: value for value in values}
        
        total_alignment = 0.0
        total_weight = 0.0
        
        for action in actions:
            action_type = action.get('type', 'unknown')
            action_context = action.get('context', '')
            action_outcome = action.get('outcome', 'neutral')
            action_importance = action.get('importance', 1.0)
            
            # Find matching values for this action
            action_alignment = 0.0
            matching_values = 0
            
            for value in values:
                # Check if action aligns with this value
                alignment_score = self._compute_action_value_alignment(
                    action, value
                )
                
                if alignment_score > 0.1:  # Threshold for meaningful alignment
                    action_alignment += alignment_score * value.strength
                    matching_values += 1
            
            # Normalize by number of matching values
            if matching_values > 0:
                action_alignment = action_alignment / matching_values
            
            # Weight by action importance
            total_alignment += action_alignment * action_importance
            total_weight += action_importance
        
        # Compute overall alignment score
        if total_weight > 0:
            alignment_score = total_alignment / total_weight
        else:
            alignment_score = 0.0
        
        # Compute value coverage (how many values are being expressed)
        expressed_values = set()
        for action in actions:
            for value in values:
                if self._compute_action_value_alignment(action, value) > 0.3:
                    expressed_values.add(value.pattern_id)
        
        value_coverage = len(expressed_values) / max(len(values), 1)
        
        # Combine alignment and coverage
        final_score = alignment_score * 0.8 + value_coverage * 0.2
        
        # Store measurement
        measurement = {
            'timestamp': datetime.now(),
            'alignment_score': final_score,
            'raw_alignment': alignment_score,
            'value_coverage': value_coverage,
            'actions_count': len(actions),
            'values_count': len(values),
            'expressed_values_count': len(expressed_values)
        }
        
        self.value_alignment_history.append(measurement)
        if len(self.value_alignment_history) > 100:  # Keep last 100 measurements
            self.value_alignment_history.pop(0)
        
        logger.debug(f"Assessed value alignment: {final_score:.4f} "
                    f"(raw_alignment={alignment_score:.3f}, "
                    f"coverage={value_coverage:.3f}, "
                    f"expressed_values={len(expressed_values)}/{len(values)})")
        
        return final_score
    
    def get_comprehensive_success_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive success metrics combining all measurement dimensions.
        
        Returns:
            Dictionary containing all success metrics and trends
        """
        # Get recent measurements
        recent_harmony = self.harmony_history[-10:] if self.harmony_history else []
        recent_growth = self.growth_measurements[-5:] if self.growth_measurements else []
        recent_autonomy = self.autonomy_measurements[-10:] if self.autonomy_measurements else []
        recent_alignment = self.value_alignment_history[-10:] if self.value_alignment_history else []
        
        # Compute current scores
        current_harmony = recent_harmony[-1]['harmony_index'] if recent_harmony else 0.0
        current_growth = recent_growth[-1]['metrics']['trajectory_score'] if recent_growth else 0.0
        current_autonomy = recent_autonomy[-1]['autonomy_level'] if recent_autonomy else 0.0
        current_alignment = recent_alignment[-1]['alignment_score'] if recent_alignment else 0.0
        
        # Compute trends
        harmony_trend = self._compute_trend([m['harmony_index'] for m in recent_harmony])
        growth_trend = self._compute_trend([m['metrics']['trajectory_score'] for m in recent_growth])
        autonomy_trend = self._compute_trend([m['autonomy_level'] for m in recent_autonomy])
        alignment_trend = self._compute_trend([m['alignment_score'] for m in recent_alignment])
        
        # Compute overall success score
        overall_success = (
            current_harmony * 0.3 +
            current_growth * 0.3 +
            current_autonomy * 0.25 +
            current_alignment * 0.15
        )
        
        return {
            'overall_success_score': overall_success,
            'current_metrics': {
                'state_harmony': current_harmony,
                'growth_trajectory': current_growth,
                'autonomy_level': current_autonomy,
                'value_alignment': current_alignment
            },
            'trends': {
                'harmony_trend': harmony_trend,
                'growth_trend': growth_trend,
                'autonomy_trend': autonomy_trend,
                'alignment_trend': alignment_trend
            },
            'measurement_counts': {
                'harmony_measurements': len(self.harmony_history),
                'growth_measurements': len(self.growth_measurements),
                'autonomy_measurements': len(self.autonomy_measurements),
                'alignment_measurements': len(self.value_alignment_history)
            },
            'timestamp': datetime.now()
        }
    
    # Private helper methods
    
    def _initialize_harmony_patterns(self) -> List[Pattern]:
        """Initialize basic harmony patterns for state assessment."""
        patterns = []
        
        # Balanced state pattern
        patterns.append(Pattern(
            pattern_type='balanced_state',
            strength=1.0,
            frequency=1,
            last_occurrence=datetime.now(),
            characteristics={'balance_threshold': 0.8, 'variance_threshold': 0.2}
        ))
        
        # Coherent state pattern
        patterns.append(Pattern(
            pattern_type='coherent_state',
            strength=1.0,
            frequency=1,
            last_occurrence=datetime.now(),
            characteristics={'coherence_threshold': 0.7, 'norm_range': [0.5, 2.0]}
        ))
        
        # Stable state pattern
        patterns.append(Pattern(
            pattern_type='stable_state',
            strength=1.0,
            frequency=1,
            last_occurrence=datetime.now(),
            characteristics={'stability_threshold': 0.9, 'change_threshold': 0.1}
        ))
        
        return patterns
    
    def _compute_state_coherence(self, state: np.ndarray) -> float:
        """Compute coherence of the state vector."""
        if len(state) == 0:
            return 0.0
        
        # Coherence based on how well-structured the state is
        # Use inverse of coefficient of variation (std/mean) as coherence measure
        state_abs = np.abs(state)
        mean_val = np.mean(state_abs)
        std_val = np.std(state_abs)
        
        if mean_val == 0:
            return 0.0
        
        cv = std_val / mean_val
        coherence = 1.0 / (1.0 + cv)  # Higher coherence for lower variation
        
        return coherence
    
    def _compute_state_balance(self, state: np.ndarray) -> float:
        """Compute balance across state dimensions."""
        if len(state) == 0:
            return 0.0
        
        # Balance based on how evenly distributed the state energy is
        state_abs = np.abs(state)
        total_energy = np.sum(state_abs)
        
        if total_energy == 0:
            return 1.0  # Perfect balance for zero state
        
        # Compute entropy of energy distribution
        energy_probs = state_abs / total_energy
        entropy = -np.sum(energy_probs * np.log(energy_probs + 1e-8))
        
        # Normalize by maximum possible entropy
        max_entropy = np.log(len(state))
        balance = entropy / max_entropy if max_entropy > 0 else 0.0
        
        return balance
    
    def _compute_pattern_alignment(self, state: np.ndarray, patterns: List[Pattern]) -> float:
        """Compute alignment with known harmony patterns."""
        if not patterns:
            return 0.5  # Neutral alignment when no patterns available
        
        total_alignment = 0.0
        total_weight = 0.0
        
        for pattern in patterns:
            pattern_alignment = self._compute_single_pattern_alignment(state, pattern)
            pattern_weight = pattern.strength
            
            total_alignment += pattern_alignment * pattern_weight
            total_weight += pattern_weight
        
        if total_weight > 0:
            return total_alignment / total_weight
        else:
            return 0.5
    
    def _compute_single_pattern_alignment(self, state: np.ndarray, pattern: Pattern) -> float:
        """Compute alignment with a single harmony pattern."""
        pattern_type = pattern.pattern_type
        characteristics = pattern.characteristics
        
        if pattern_type == 'balanced_state':
            balance = self._compute_state_balance(state)
            threshold = characteristics.get('balance_threshold', 0.8)
            return 1.0 if balance >= threshold else balance / threshold
        
        elif pattern_type == 'coherent_state':
            coherence = self._compute_state_coherence(state)
            threshold = characteristics.get('coherence_threshold', 0.7)
            return 1.0 if coherence >= threshold else coherence / threshold
        
        elif pattern_type == 'stable_state':
            # For stability, we need historical context which we don't have here
            # Return neutral alignment
            return 0.5
        
        else:
            # Unknown pattern type
            return 0.5
    
    def _compute_state_stability(self, state: np.ndarray) -> float:
        """Compute stability of the current state."""
        # Without historical context, estimate stability from state characteristics
        # More stable states tend to have lower variance and moderate norms
        
        if len(state) == 0:
            return 0.0
        
        state_norm = np.linalg.norm(state)
        state_var = np.var(state)
        
        # Optimal norm range (not too small, not too large)
        norm_stability = 1.0 - abs(state_norm - 1.0) / 2.0  # Optimal around 1.0
        norm_stability = max(0.0, norm_stability)
        
        # Lower variance indicates higher stability
        var_stability = 1.0 / (1.0 + state_var)
        
        # Combine measures
        stability = (norm_stability + var_stability) / 2.0
        
        return stability
    
    def _compute_action_value_alignment(self, action: Dict[str, Any], value: ValuePattern) -> float:
        """Compute alignment between a specific action and value pattern."""
        action_type = action.get('type', 'unknown')
        action_context = action.get('context', '')
        action_outcome = action.get('outcome', 'neutral')
        
        # Simple alignment based on type matching and context overlap
        alignment = 0.0
        
        # Type-based alignment
        if action_type in value.contexts:
            alignment += 0.5
        
        # Context-based alignment (simplified string matching)
        context_overlap = sum(1 for ctx in value.contexts 
                            if ctx.lower() in action_context.lower())
        if context_overlap > 0:
            alignment += 0.3 * min(context_overlap / len(value.contexts), 1.0)
        
        # Outcome-based alignment
        if action_outcome == 'positive' and value.strength > 0.5:
            alignment += 0.2
        elif action_outcome == 'negative' and value.strength < 0.5:
            alignment += 0.1
        
        return min(alignment, 1.0)
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend direction from a list of values."""
        if len(values) < 2:
            return 'stable'
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        if slope > 0.01:
            return 'improving'
        elif slope < -0.01:
            return 'declining'
        else:
            return 'stable'