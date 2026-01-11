"""
State-Derived Reward Generator for the Autonomous Reward System.

This module implements the StateDerivedRewardGenerator class that generates reward signals
directly from pathos state patterns and transitions, ensuring all rewards derive from
internal state dynamics.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from scipy.stats import entropy
from scipy.spatial.distance import cosine
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

from .interfaces import StateDerivedRewardGeneratorInterface
from .models import Pattern

logger = logging.getLogger(__name__)


class StateDerivedRewardGenerator(StateDerivedRewardGeneratorInterface):
    """
    Generates reward signals directly from pathos state patterns and transitions.
    
    This class implements the core state-derived reward computation that forms the
    foundation of the autonomous reward system. All rewards are computed based on
    internal state dynamics including coherence, growth, integration, elegance,
    and emergent patterns.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the state-derived reward generator.
        
        Args:
            config: Optional configuration parameters for reward computation
        """
        self.config = config or {}
        
        # Reward computation parameters
        self.coherence_weight = self.config.get('coherence_weight', 1.0)
        self.growth_weight = self.config.get('growth_weight', 1.0)
        self.integration_weight = self.config.get('integration_weight', 0.8)
        self.elegance_weight = self.config.get('elegance_weight', 0.6)
        self.emergence_weight = self.config.get('emergence_weight', 1.2)
        
        # State analysis parameters
        self.coherence_threshold = self.config.get('coherence_threshold', 0.5)
        self.growth_smoothing = self.config.get('growth_smoothing', 0.1)
        self.integration_window = self.config.get('integration_window', 5)
        self.elegance_balance_point = self.config.get('elegance_balance_point', 0.5)
        
        # Pattern detection parameters
        self.pattern_detection_eps = self.config.get('pattern_detection_eps', 0.3)
        self.pattern_min_samples = self.config.get('pattern_min_samples', 2)
        self.pattern_history_size = self.config.get('pattern_history_size', 20)
        
        # Internal state for pattern tracking
        self.detected_patterns: List[Pattern] = []
        self.state_history: List[np.ndarray] = []
        self.coherence_history: List[float] = []
        self.growth_history: List[float] = []
        
        logger.info("Initialized StateDerivedRewardGenerator")
    
    def compute_coherence_reward(self, state: np.ndarray) -> float:
        """
        Compute reward for achieving internal state harmony.
        
        Coherence reward measures how well-integrated and harmonious the current
        state is. Higher coherence indicates better internal organization and
        balance across state dimensions.
        
        Args:
            state: Current pathos state
            
        Returns:
            Coherence reward value (0.0 to 2.0)
        """
        if len(state) == 0:
            return 0.0
        
        # Normalize state to prevent numerical issues
        state_norm = np.linalg.norm(state)
        if state_norm < 1e-8:
            return 0.0
        
        normalized_state = state / state_norm
        
        # Measure 1: State balance (how evenly distributed the state energy is)
        state_abs = np.abs(normalized_state)
        state_balance = 1.0 - np.std(state_abs)  # Lower std = better balance
        state_balance = max(0.0, state_balance)
        
        # Measure 2: State smoothness (lack of abrupt changes between dimensions)
        if len(state) > 1:
            state_diffs = np.diff(normalized_state)
            smoothness = 1.0 / (1.0 + np.mean(np.abs(state_diffs)))
        else:
            smoothness = 1.0
        
        # Measure 3: State entropy (optimal complexity - not too ordered, not too chaotic)
        # Convert to probability distribution
        state_probs = state_abs / (np.sum(state_abs) + 1e-8)
        state_entropy = entropy(state_probs + 1e-8)
        max_entropy = np.log(len(state))
        normalized_entropy = state_entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Optimal entropy is around 0.7 (not too ordered, not too chaotic)
        entropy_reward = 1.0 - abs(normalized_entropy - 0.7)
        entropy_reward = max(0.0, entropy_reward)
        
        # Combine coherence measures
        coherence_score = (state_balance * 0.4 + smoothness * 0.3 + entropy_reward * 0.3)
        
        # Apply coherence threshold and scaling
        if coherence_score > self.coherence_threshold:
            coherence_reward = coherence_score * self.coherence_weight
        else:
            # Reduced reward below threshold
            coherence_reward = coherence_score * self.coherence_weight * 0.5
        
        # Reduced positive bias - only add minimal bias to prevent complete stagnation
        # The main positive signal should come from successful actions, not state bias
        coherence_reward += 0.1  # Reduced from 0.3 to 0.1
        
        # Store in history for pattern analysis
        self.coherence_history.append(coherence_score)
        if len(self.coherence_history) > self.pattern_history_size:
            self.coherence_history.pop(0)
        
        logger.debug(f"Computed coherence reward: {coherence_reward:.4f} "
                    f"(balance={state_balance:.3f}, smoothness={smoothness:.3f}, "
                    f"entropy={entropy_reward:.3f})")
        
        return coherence_reward
    
    def compute_growth_reward(self, current_state: np.ndarray, 
                            historical_states: List[np.ndarray]) -> float:
        """
        Compute reward for healthy state evolution and expansion.
        
        Growth reward measures positive development in the state space,
        including expansion of capabilities, learning, and adaptive changes
        that indicate healthy development.
        
        Args:
            current_state: Current pathos state
            historical_states: Recent historical states for trend analysis
            
        Returns:
            Growth reward value (0.0 to 2.0)
        """
        if len(historical_states) < 2:
            return 0.0
        
        # Ensure we have valid states
        valid_states = [s for s in historical_states if len(s) == len(current_state)]
        if len(valid_states) < 2:
            return 0.0
        
        # Measure 1: State expansion (growth in state magnitude over time)
        state_norms = [np.linalg.norm(s) for s in valid_states]
        current_norm = np.linalg.norm(current_state)
        
        if len(state_norms) >= 2:
            # Compute trend in state magnitude
            recent_norms = state_norms[-3:] + [current_norm]  # Last 3 + current
            if len(recent_norms) >= 2:
                norm_trend = np.polyfit(range(len(recent_norms)), recent_norms, 1)[0]
                expansion_reward = max(0.0, norm_trend * 2.0)  # Positive trend = growth
                expansion_reward = min(expansion_reward, 1.0)  # Cap at 1.0
            else:
                expansion_reward = 0.0
        else:
            expansion_reward = 0.0
        
        # Measure 2: State diversification (exploration of new state regions)
        if len(valid_states) >= 1:
            # Compute average distance to historical states
            distances = []
            for hist_state in valid_states[-5:]:  # Last 5 states
                if len(hist_state) == len(current_state):
                    dist = np.linalg.norm(current_state - hist_state)
                    distances.append(dist)
            
            if distances:
                avg_distance = np.mean(distances)
                # Moderate distance indicates healthy exploration
                diversification_reward = min(avg_distance * 0.5, 1.0)
            else:
                diversification_reward = 0.0
        else:
            diversification_reward = 0.0
        
        # Measure 3: State complexity growth (increasing sophistication)
        if len(valid_states) >= 2:
            # Compute complexity trend
            complexities = []
            for state in valid_states[-3:]:  # Last 3 states
                state_abs = np.abs(state)
                state_probs = state_abs / (np.sum(state_abs) + 1e-8)
                complexity = entropy(state_probs + 1e-8)
                complexities.append(complexity)
            
            current_complexity = self._compute_state_complexity(current_state)
            complexities.append(current_complexity)
            
            if len(complexities) >= 2:
                complexity_trend = np.polyfit(range(len(complexities)), complexities, 1)[0]
                complexity_growth = max(0.0, complexity_trend * 1.0)
                complexity_growth = min(complexity_growth, 1.0)
            else:
                complexity_growth = 0.0
        else:
            complexity_growth = 0.0
        
        # Combine growth measures with smoothing
        raw_growth = (expansion_reward * 0.4 + diversification_reward * 0.3 + 
                     complexity_growth * 0.3)
        
        # Apply smoothing to prevent reward spikes
        if self.growth_history:
            smoothed_growth = (self.growth_smoothing * raw_growth + 
                             (1 - self.growth_smoothing) * self.growth_history[-1])
        else:
            smoothed_growth = raw_growth
        
        growth_reward = smoothed_growth * self.growth_weight
        
        # Reduced positive bias - only add minimal bias to prevent complete stagnation
        # The main positive signal should come from successful actions, not state bias
        growth_reward += 0.1  # Reduced from 0.2 to 0.1
        
        # Store in history
        self.growth_history.append(raw_growth)
        if len(self.growth_history) > self.pattern_history_size:
            self.growth_history.pop(0)
        
        logger.debug(f"Computed growth reward: {growth_reward:.4f} "
                    f"(expansion={expansion_reward:.3f}, diversification={diversification_reward:.3f}, "
                    f"complexity={complexity_growth:.3f})")
        
        return growth_reward
    
    def compute_integration_reward(self, state: np.ndarray, 
                                 memory_patterns: Dict[str, Any]) -> float:
        """
        Compute reward for connecting disparate state patterns.
        
        Integration reward measures how well the current state connects and
        synthesizes information from different sources, creating coherent
        wholes from disparate parts.
        
        Args:
            state: Current pathos state
            memory_patterns: Relevant memory patterns for integration assessment
            
        Returns:
            Integration reward value (0.0 to 1.5)
        """
        if len(state) == 0:
            return 0.0
        
        # Measure 1: Cross-dimensional correlation (how well different state dimensions work together)
        if len(state) > 1:
            # Compute correlation matrix
            state_reshaped = state.reshape(-1, 1)
            if len(state) >= 4:  # Need minimum dimensions for meaningful correlation
                # Split state into chunks and compute cross-correlations
                chunk_size = len(state) // 4
                chunks = [state[i:i+chunk_size] for i in range(0, len(state), chunk_size)]
                chunks = [chunk for chunk in chunks if len(chunk) > 1]
                
                if len(chunks) >= 2:
                    correlations = []
                    for i in range(len(chunks)):
                        for j in range(i+1, len(chunks)):
                            if len(chunks[i]) == len(chunks[j]):
                                corr = np.corrcoef(chunks[i], chunks[j])[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(abs(corr))
                    
                    if correlations:
                        cross_correlation = np.mean(correlations)
                    else:
                        cross_correlation = 0.0
                else:
                    cross_correlation = 0.0
            else:
                cross_correlation = 0.0
        else:
            cross_correlation = 0.0
        
        # Measure 2: Memory pattern integration (how well current state integrates with memory)
        memory_integration = 0.0
        if memory_patterns:
            experience_count = memory_patterns.get('experience_count', 0)
            action_types = memory_patterns.get('action_types', [])
            
            # Integration increases with experience diversity and count
            if experience_count > 0:
                experience_factor = min(experience_count / 10.0, 1.0)  # Normalize to [0,1]
                diversity_factor = min(len(action_types) / 5.0, 1.0)   # Normalize to [0,1]
                memory_integration = (experience_factor + diversity_factor) / 2.0
        
        # Measure 3: State coherence with historical patterns
        historical_integration = 0.0
        if len(self.state_history) >= self.integration_window:
            recent_states = self.state_history[-self.integration_window:]
            
            # Compute similarity with recent state patterns
            similarities = []
            for hist_state in recent_states:
                if len(hist_state) == len(state):
                    # Use cosine similarity for pattern matching
                    similarity = 1.0 - cosine(state, hist_state)
                    if not np.isnan(similarity):
                        similarities.append(max(0.0, similarity))
            
            if similarities:
                # Moderate similarity indicates good integration (not too similar, not too different)
                avg_similarity = np.mean(similarities)
                # Optimal similarity around 0.6 (some connection but not repetition)
                historical_integration = 1.0 - abs(avg_similarity - 0.6)
                historical_integration = max(0.0, historical_integration)
        
        # Combine integration measures
        integration_score = (cross_correlation * 0.4 + memory_integration * 0.3 + 
                           historical_integration * 0.3)
        
        integration_reward = integration_score * self.integration_weight
        
        # Update state history
        self.state_history.append(state.copy())
        if len(self.state_history) > self.pattern_history_size:
            self.state_history.pop(0)
        
        logger.debug(f"Computed integration reward: {integration_reward:.4f} "
                    f"(cross_corr={cross_correlation:.3f}, memory={memory_integration:.3f}, "
                    f"historical={historical_integration:.3f})")
        
        return integration_reward
    
    def compute_elegance_reward(self, state_complexity: float, 
                              solution_efficiency: float) -> float:
        """
        Compute reward for optimal complexity-simplicity balance.
        
        Elegance reward measures the achievement of optimal balance between
        complexity and simplicity, rewarding solutions that are sophisticated
        yet efficient and comprehensible.
        
        Args:
            state_complexity: Measured complexity of current state (0.0 to 1.0)
            solution_efficiency: Efficiency of current solution/approach (0.0 to 1.0)
            
        Returns:
            Elegance reward value (0.0 to 1.0)
        """
        # Ensure inputs are in valid range
        state_complexity = max(0.0, min(1.0, state_complexity))
        solution_efficiency = max(0.0, min(1.0, solution_efficiency))
        
        # Measure 1: Complexity-simplicity balance
        # Optimal complexity is around the balance point (default 0.5)
        complexity_balance = 1.0 - abs(state_complexity - self.elegance_balance_point)
        complexity_balance = max(0.0, complexity_balance)
        
        # Measure 2: Efficiency reward (higher efficiency is better)
        efficiency_reward = solution_efficiency
        
        # Measure 3: Elegance bonus for high efficiency with moderate complexity
        elegance_bonus = 0.0
        if (0.3 <= state_complexity <= 0.7) and (solution_efficiency > 0.7):
            # Bonus for achieving high efficiency with balanced complexity
            elegance_bonus = 0.2
        
        # Combine elegance measures
        elegance_score = (complexity_balance * 0.5 + efficiency_reward * 0.4 + elegance_bonus)
        
        elegance_reward = elegance_score * self.elegance_weight
        
        logger.debug(f"Computed elegance reward: {elegance_reward:.4f} "
                    f"(complexity_balance={complexity_balance:.3f}, "
                    f"efficiency={efficiency_reward:.3f}, bonus={elegance_bonus:.3f})")
        
        return elegance_reward
    
    def detect_emergent_patterns(self, state_sequence: List[np.ndarray]) -> List[Pattern]:
        """
        Detect novel state configurations and emergent behaviors.
        
        This method identifies patterns in state sequences that indicate
        emergent behaviors, novel configurations, or significant state
        transitions that represent new capabilities or insights.
        
        Args:
            state_sequence: Sequence of recent states for pattern detection
            
        Returns:
            List of detected emergent patterns
        """
        if len(state_sequence) < 3:
            return []
        
        # Ensure all states have the same dimension
        state_dim = len(state_sequence[0])
        valid_states = [s for s in state_sequence if len(s) == state_dim]
        
        if len(valid_states) < 3:
            return []
        
        detected_patterns = []
        
        # Pattern 1: Oscillatory patterns (rhythmic state changes)
        oscillatory_pattern = self._detect_oscillatory_pattern(valid_states)
        if oscillatory_pattern:
            detected_patterns.append(oscillatory_pattern)
        
        # Pattern 2: Convergence patterns (states converging to attractor)
        convergence_pattern = self._detect_convergence_pattern(valid_states)
        if convergence_pattern:
            detected_patterns.append(convergence_pattern)
        
        # Pattern 3: Divergence patterns (exploration of new state regions)
        divergence_pattern = self._detect_divergence_pattern(valid_states)
        if divergence_pattern:
            detected_patterns.append(divergence_pattern)
        
        # Pattern 4: Clustering patterns (states forming distinct clusters)
        clustering_patterns = self._detect_clustering_patterns(valid_states)
        detected_patterns.extend(clustering_patterns)
        
        # Pattern 5: Transition patterns (significant state transitions)
        transition_patterns = self._detect_transition_patterns(valid_states)
        detected_patterns.extend(transition_patterns)
        
        # Update internal pattern history
        self.detected_patterns.extend(detected_patterns)
        if len(self.detected_patterns) > self.pattern_history_size * 2:
            self.detected_patterns = self.detected_patterns[-self.pattern_history_size:]
        
        logger.debug(f"Detected {len(detected_patterns)} emergent patterns")
        
        return detected_patterns
    
    def _compute_state_complexity(self, state: np.ndarray) -> float:
        """Compute complexity measure of a state using entropy."""
        if len(state) == 0:
            return 0.0
        
        state_abs = np.abs(state)
        state_probs = state_abs / (np.sum(state_abs) + 1e-8)
        state_entropy = entropy(state_probs + 1e-8)
        max_entropy = np.log(len(state))
        
        return state_entropy / max_entropy if max_entropy > 0 else 0.0
    
    def _detect_oscillatory_pattern(self, states: List[np.ndarray]) -> Optional[Pattern]:
        """Detect oscillatory patterns in state sequence."""
        if len(states) < 4:
            return None
        
        # Compute state norms to detect oscillations
        norms = [np.linalg.norm(s) for s in states]
        
        # Look for oscillatory behavior in norms
        diffs = np.diff(norms)
        sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
        
        # If we have multiple sign changes, it might be oscillatory
        if sign_changes >= 2:
            oscillation_strength = sign_changes / len(diffs)
            
            if oscillation_strength > 0.5:  # At least 50% sign changes
                return Pattern(
                    pattern_type="oscillatory",
                    strength=oscillation_strength,
                    frequency=sign_changes,
                    last_occurrence=datetime.now(),
                    characteristics={
                        'amplitude': np.std(norms),
                        'period_estimate': len(states) / (sign_changes / 2)
                    }
                )
        
        return None
    
    def _detect_convergence_pattern(self, states: List[np.ndarray]) -> Optional[Pattern]:
        """Detect convergence patterns in state sequence."""
        if len(states) < 3:
            return None
        
        # Compute distances between consecutive states
        distances = []
        for i in range(1, len(states)):
            dist = np.linalg.norm(states[i] - states[i-1])
            distances.append(dist)
        
        # Check if distances are decreasing (convergence)
        if len(distances) >= 2:
            trend = np.polyfit(range(len(distances)), distances, 1)[0]
            
            if trend < -0.01:  # Decreasing trend
                convergence_strength = abs(trend)
                
                return Pattern(
                    pattern_type="convergence",
                    strength=min(convergence_strength * 10, 1.0),
                    frequency=1,
                    last_occurrence=datetime.now(),
                    characteristics={
                        'trend': trend,
                        'final_distance': distances[-1],
                        'convergence_rate': convergence_strength
                    }
                )
        
        return None
    
    def _detect_divergence_pattern(self, states: List[np.ndarray]) -> Optional[Pattern]:
        """Detect divergence patterns in state sequence."""
        if len(states) < 3:
            return None
        
        # Compute distances from first state
        distances = []
        first_state = states[0]
        for state in states[1:]:
            dist = np.linalg.norm(state - first_state)
            distances.append(dist)
        
        # Check if distances are increasing (divergence)
        if len(distances) >= 2:
            trend = np.polyfit(range(len(distances)), distances, 1)[0]
            
            if trend > 0.01:  # Increasing trend
                divergence_strength = trend
                
                return Pattern(
                    pattern_type="divergence",
                    strength=min(divergence_strength * 5, 1.0),
                    frequency=1,
                    last_occurrence=datetime.now(),
                    characteristics={
                        'trend': trend,
                        'final_distance': distances[-1],
                        'divergence_rate': divergence_strength
                    }
                )
        
        return None
    
    def _detect_clustering_patterns(self, states: List[np.ndarray]) -> List[Pattern]:
        """Detect clustering patterns in state sequence."""
        if len(states) < 4:
            return []
        
        try:
            # Convert states to matrix for clustering
            state_matrix = np.array(states)
            
            # Standardize the data
            scaler = StandardScaler()
            state_matrix_scaled = scaler.fit_transform(state_matrix)
            
            # Apply DBSCAN clustering
            clustering = DBSCAN(
                eps=self.pattern_detection_eps,
                min_samples=self.pattern_min_samples
            ).fit(state_matrix_scaled)
            
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters >= 2:  # Found meaningful clusters
                cluster_pattern = Pattern(
                    pattern_type="clustering",
                    strength=n_clusters / len(states),
                    frequency=n_clusters,
                    last_occurrence=datetime.now(),
                    characteristics={
                        'n_clusters': n_clusters,
                        'noise_points': np.sum(labels == -1),
                        'cluster_labels': labels.tolist()
                    }
                )
                return [cluster_pattern]
        
        except Exception as e:
            logger.debug(f"Clustering pattern detection failed: {e}")
        
        return []
    
    def _detect_transition_patterns(self, states: List[np.ndarray]) -> List[Pattern]:
        """Detect significant state transitions."""
        if len(states) < 3:
            return []
        
        patterns = []
        
        # Compute transition magnitudes
        transitions = []
        for i in range(1, len(states)):
            transition_magnitude = np.linalg.norm(states[i] - states[i-1])
            transitions.append(transition_magnitude)
        
        # Find significant transitions (outliers)
        if len(transitions) >= 3:
            mean_transition = np.mean(transitions)
            std_transition = np.std(transitions)
            
            for i, transition in enumerate(transitions):
                if transition > mean_transition + 2 * std_transition:  # 2-sigma outlier
                    transition_pattern = Pattern(
                        pattern_type="significant_transition",
                        strength=min((transition - mean_transition) / std_transition / 2, 1.0),
                        frequency=1,
                        last_occurrence=datetime.now(),
                        characteristics={
                            'transition_index': i,
                            'magnitude': transition,
                            'z_score': (transition - mean_transition) / std_transition
                        }
                    )
                    patterns.append(transition_pattern)
        
        return patterns
    
    def get_pattern_history(self) -> List[Pattern]:
        """Get the history of detected patterns."""
        return self.detected_patterns.copy()
    
    def get_coherence_history(self) -> List[float]:
        """Get the history of coherence scores."""
        return self.coherence_history.copy()
    
    def get_growth_history(self) -> List[float]:
        """Get the history of growth scores."""
        return self.growth_history.copy()
    
    def reset_history(self) -> None:
        """Reset all internal history for fresh start."""
        self.detected_patterns.clear()
        self.state_history.clear()
        self.coherence_history.clear()
        self.growth_history.clear()
        logger.info("Reset StateDerivedRewardGenerator history")