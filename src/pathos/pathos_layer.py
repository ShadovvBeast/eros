"""
Concrete implementation of the Pathos Layer affective dynamics engine.

This module implements the mathematical dynamics specified in the design document:
F(t+1) = g(α·F(t) + h(S_t, F(t)) + β·Σ(w_i·F_i))
"""

import numpy as np
from typing import List, Optional, Dict, Tuple, Any
import time
import logging

from .interfaces import PathosLayer as PathosLayerInterface
from ..core.models import SemanticVector, MemoryTrace, PathosState
from ..core.config import PathosConfig
from ..core.math_utils import (
    tanh_squash, compute_homeostatic_balance, compute_state_change_penalty,
    cosine_similarity, compute_similarity_weights, sigmoid_squash
)

logger = logging.getLogger(__name__)


class PathosLayer(PathosLayerInterface):
    """
    Concrete implementation of the Pathos affective dynamics engine.
    
    Implements the mathematical model:
    F(t+1) = g(α·F(t) + h(S_t, F(t)) + β·Σ(w_i·F_i))
    
    Where:
    - F(t): Current affective state vector
    - α: Decay factor (config.decay_factor)
    - h(S_t, F(t)): Affective impulse function
    - β: Echo strength parameter (config.echo_strength)
    - w_i: Similarity weights for memory echoes
    - g(·): Nonlinear squashing function (tanh)
    """
    
    def __init__(self, config: PathosConfig):
        super().__init__(config)
        self.homeostatic_targets = self._initialize_homeostatic_targets()
        
        # Attractor dynamics components
        self.attractor_states: List[np.ndarray] = []  # Previously rewarding states
        self.attractor_rewards: List[float] = []  # Associated rewards
        self.attractor_strengths: List[float] = []  # Attractor strength values
        self.pattern_recognition_threshold = 0.8  # Similarity threshold for pattern recognition
        self.max_attractors = 20  # Maximum number of attractors to maintain
        
        logger.info(f"Initialized PathosLayer with state dimension {config.state_dimension}")
    
    def _initialize_state(self) -> np.ndarray:
        """
        Initialize the high-dimensional affective state vector F(t).
        
        Returns:
            Initial state vector with small random values around zero
        """
        # Initialize with small random values to break symmetry
        # Use normal distribution with small variance
        state = np.random.normal(0.0, 0.1, size=self.config.state_dimension)
        
        # Ensure the initial state is within reasonable bounds
        state = np.clip(state, -1.0, 1.0)
        
        logger.debug(f"Initialized affective state with shape {state.shape}, "
                    f"mean={np.mean(state):.4f}, std={np.std(state):.4f}")
        
        return state
    
    def _initialize_homeostatic_targets(self) -> Dict[str, Tuple[float, float]]:
        """
        Initialize target ranges for homeostatic balance computation.
        
        Returns:
            Dictionary mapping metric names to (min, max) target ranges
        """
        return {
            'stability': (0.01, 0.1),    # Variance should be low but not zero
            'complexity': (2.0, 4.0),    # Entropy should be moderate
            'coherence': (0.1, 0.5),     # Norm per dimension should be moderate
            'balance': (0.05, 0.3)       # Mean absolute activation should be moderate
        }
    
    def update_state(self, semantic_input: SemanticVector, external_reward: float,
                    interest: float, memory_echoes: List[MemoryTrace] = None) -> np.ndarray:
        """
        Update affective state using the mathematical dynamics:
        F(t+1) = g(α·F(t) + h(S_t, F(t)) + β·Σ(w_i·F_i))
        
        Args:
            semantic_input: Semantic vector from Logos
            external_reward: Reward from tool execution
            interest: Interest signal from Logos
            memory_echoes: Retrieved memory traces for echo computation
            
        Returns:
            Updated affective state vector F(t+1)
        """
        if memory_echoes is None:
            memory_echoes = []
        
        # Store previous state for change computation
        self.previous_state = self.current_state.copy()
        
        # Term 1: α·F(t) - Decay of current state
        decay_term = self.config.decay_factor * self.current_state
        
        # Term 2: h(S_t, F(t)) - Affective impulse from semantic input
        impulse_term = self._compute_affective_impulse(semantic_input, external_reward, interest)
        
        # Term 3: β·Σ(w_i·F_i) - Memory echo influence with attractor dynamics
        echo_term = self.implement_memory_driven_echo_dynamics(memory_echoes)
        
        # Term 4: Attractor influence for emergent behavior
        attractor_influence = self.compute_attractor_influence(self.current_state)
        
        # Combine all terms
        raw_update = decay_term + impulse_term + echo_term + attractor_influence
        
        # Apply nonlinear squashing function g(·)
        self.current_state = self._apply_squashing_function(raw_update)
        
        logger.debug(f"State update - decay_norm: {np.linalg.norm(decay_term):.4f}, "
                    f"impulse_norm: {np.linalg.norm(impulse_term):.4f}, "
                    f"echo_norm: {np.linalg.norm(echo_term):.4f}")
        
        return self.current_state
    
    def _compute_affective_impulse(self, semantic_input: SemanticVector, 
                                 external_reward: float, interest: float) -> np.ndarray:
        """
        Compute affective impulse h(S_t, F(t)) from semantic input and current state.
        
        This function combines:
        1. Semantic embedding influence
        2. External reward influence
        3. Interest signal influence
        4. Current state modulation
        
        Args:
            semantic_input: Semantic vector from Logos
            external_reward: External reward signal
            interest: Interest signal from Logos
            
        Returns:
            Affective impulse vector
        """
        # Ensure semantic embedding matches state dimension
        semantic_embedding = semantic_input.embedding
        if len(semantic_embedding) != self.config.state_dimension:
            # Project or pad semantic embedding to match state dimension
            if len(semantic_embedding) > self.config.state_dimension:
                # Truncate
                semantic_embedding = semantic_embedding[:self.config.state_dimension]
            else:
                # Pad with zeros
                padding = np.zeros(self.config.state_dimension - len(semantic_embedding))
                semantic_embedding = np.concatenate([semantic_embedding, padding])
        
        # Base impulse from semantic input
        semantic_impulse = 0.1 * semantic_embedding
        
        # Modulate by current state (creates nonlinear dynamics)
        # Use element-wise product to create state-dependent response
        state_modulation = 0.05 * semantic_embedding * np.tanh(self.current_state)
        
        # External reward influence (broadcast across all dimensions)
        reward_impulse = 0.02 * external_reward * np.ones(self.config.state_dimension)
        
        # Interest signal influence (amplifies semantic impulse)
        interest_modulation = interest * 0.03 * semantic_embedding
        
        # Combine all impulse components
        total_impulse = (semantic_impulse + state_modulation + 
                        reward_impulse + interest_modulation)
        
        return total_impulse
    
    def _compute_memory_echo(self, memory_echoes: List[MemoryTrace]) -> np.ndarray:
        """
        Compute memory echo influence β·Σ(w_i·F_i).
        
        Args:
            memory_echoes: List of retrieved memory traces
            
        Returns:
            Weighted sum of memory affect states
        """
        if not memory_echoes:
            return np.zeros(self.config.state_dimension)
        
        # Compute similarity weights based on affective similarity to current state
        similarities = []
        for trace in memory_echoes:
            similarity = cosine_similarity(self.current_state, trace.affect_state)
            similarities.append(max(0.0, similarity))  # Only positive similarities
        
        # Convert similarities to weights
        weights = compute_similarity_weights(similarities, temperature=1.0, method='exponential')
        
        # Compute weighted sum of memory affect states
        echo_sum = np.zeros(self.config.state_dimension)
        for i, trace in enumerate(memory_echoes):
            echo_sum += weights[i] * trace.affect_state
        
        # Scale by echo strength parameter β
        return self.config.echo_strength * echo_sum
    
    def _apply_squashing_function(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Apply nonlinear squashing function g(·) to keep state bounded.
        
        Args:
            raw_state: Raw state update before squashing
            
        Returns:
            Squashed state vector
        """
        # Use tanh squashing to keep values in [-1, 1]
        return tanh_squash(raw_state, scale=1.0)
    
    def compute_internal_reward(self, current_state: np.ndarray, previous_state: np.ndarray) -> float:
        """
        Compute internal reward: r_t^int = -λ₁·D_t - λ₂·||F(t+1) - F(t)||²
        
        Args:
            current_state: Current affective state F(t+1)
            previous_state: Previous affective state F(t)
            
        Returns:
            Internal reward value
        """
        # Compute homeostatic discomfort D_t
        balance_metrics, discomfort = self.compute_homeostatic_balance(current_state)
        
        # Compute state change penalty ||F(t+1) - F(t)||²
        change_penalty = compute_state_change_penalty(current_state, previous_state, penalty_type='l2')
        
        # Compute internal reward
        internal_reward = (-self.config.lambda_1 * discomfort - 
                          self.config.lambda_2 * change_penalty)
        
        logger.debug(f"Internal reward computation - discomfort: {discomfort:.4f}, "
                    f"change_penalty: {change_penalty:.4f}, reward: {internal_reward:.4f}")
        
        return internal_reward
    
    def compute_salience(self, state_change: float, reward: float,
                        novelty_affect: float, novelty_semantic: float, interest: float) -> float:
        """
        Compute salience score for memory storage decision.
        salience_t = σ(c₁·||ΔF|| + c₂·|r_t| + c₃·novelty_F + c₄·novelty_S + c₅·I_t)
        
        Args:
            state_change: Magnitude of affective state change ||ΔF||
            reward: Total reward (internal + external)
            novelty_affect: Novelty in affective space
            novelty_semantic: Novelty in semantic space
            interest: Interest signal from Logos
            
        Returns:
            Salience score (0.0 to 1.0)
        """
        # Compute weighted sum of salience factors
        raw_salience = (self.config.c1 * state_change +
                       self.config.c2 * abs(reward) +
                       self.config.c3 * novelty_affect +
                       self.config.c4 * novelty_semantic +
                       self.config.c5 * interest)
        
        # Apply sigmoid to get value in [0, 1]
        salience = float(sigmoid_squash(np.array([raw_salience]), scale=1.0)[0])
        
        logger.debug(f"Salience computation - state_change: {state_change:.4f}, "
                    f"reward: {reward:.4f}, novelty_affect: {novelty_affect:.4f}, "
                    f"novelty_semantic: {novelty_semantic:.4f}, interest: {interest:.4f}, "
                    f"salience: {salience:.4f}")
        
        return salience
    
    def should_write_memory(self, salience: float) -> bool:
        """
        Determine if experience should be stored as memory based on salience.
        
        Args:
            salience: Computed salience score
            
        Returns:
            True if memory should be written, False otherwise
        """
        return salience >= self.config.salience_threshold
    
    def compute_homeostatic_balance(self, state: np.ndarray) -> Tuple[Dict[str, float], float]:
        """
        Compute homeostatic balance metrics and discomfort.
        
        Args:
            state: Current affective state
            
        Returns:
            Tuple of (balance_metrics, discomfort_D_t)
        """
        balance_metrics = compute_homeostatic_balance(
            state, 
            self.homeostatic_targets, 
            self.config.homeostatic_weights
        )
        
        # Extract total discomfort
        discomfort = balance_metrics.get('total_discomfort', 0.0)
        
        return balance_metrics, discomfort
    
    def get_current_state(self) -> PathosState:
        """
        Get current Pathos state as a structured object.
        
        Returns:
            Current PathosState with all relevant information
        """
        # Compute current internal reward if we have a previous state
        internal_reward = 0.0
        if self.previous_state is not None:
            internal_reward = self.compute_internal_reward(self.current_state, self.previous_state)
        
        # Compute homeostatic balance
        balance_metrics, _ = self.compute_homeostatic_balance(self.current_state)
        
        return PathosState(
            vector=self.current_state.copy(),
            timestamp=int(time.time()),
            internal_reward=internal_reward,
            homeostatic_balance=balance_metrics
        )
    
    def set_state(self, state: np.ndarray) -> None:
        """
        Set the current affective state (for testing or initialization).
        
        Args:
            state: New affective state vector
        """
        if len(state) != self.config.state_dimension:
            raise ValueError(f"State dimension mismatch: expected {self.config.state_dimension}, "
                           f"got {len(state)}")
        
        self.previous_state = self.current_state.copy() if self.current_state is not None else None
        self.current_state = state.copy()
    
    def recognize_familiar_patterns(self, current_state: np.ndarray) -> Tuple[bool, Optional[int]]:
        """
        Recognize if current state is similar to previously rewarding states.
        
        Args:
            current_state: Current affective state to check
            
        Returns:
            Tuple of (is_familiar, attractor_index)
        """
        if not self.attractor_states:
            return False, None
        
        # Find most similar attractor state
        max_similarity = -1.0
        best_attractor_idx = None
        
        for i, attractor_state in enumerate(self.attractor_states):
            similarity = cosine_similarity(current_state, attractor_state)
            if similarity > max_similarity:
                max_similarity = similarity
                best_attractor_idx = i
        
        # Check if similarity exceeds threshold
        is_familiar = max_similarity >= self.pattern_recognition_threshold
        
        if is_familiar:
            logger.debug(f"Recognized familiar pattern with similarity {max_similarity:.3f}")
        
        return is_familiar, best_attractor_idx if is_familiar else None
    
    def add_attractor_state(self, state: np.ndarray, reward: float) -> None:
        """
        Add a new attractor state based on positive reward experience.
        
        Args:
            state: Affective state to potentially become an attractor
            reward: Associated reward value
        """
        # Only add states with positive rewards as attractors
        if reward <= 0.0:
            return
        
        # Check if this state is already similar to an existing attractor
        is_familiar, existing_idx = self.recognize_familiar_patterns(state)
        
        if is_familiar and existing_idx is not None:
            # Update existing attractor with new reward information
            old_reward = self.attractor_rewards[existing_idx]
            old_strength = self.attractor_strengths[existing_idx]
            
            # Weighted average of rewards
            new_reward = 0.7 * old_reward + 0.3 * reward
            new_strength = min(old_strength + 0.1, 2.0)  # Cap strength at 2.0
            
            self.attractor_rewards[existing_idx] = new_reward
            self.attractor_strengths[existing_idx] = new_strength
            
            logger.debug(f"Updated existing attractor {existing_idx} with reward {reward:.3f}")
        else:
            # Add new attractor state
            self.attractor_states.append(state.copy())
            self.attractor_rewards.append(reward)
            self.attractor_strengths.append(1.0)  # Initial strength
            
            # Maintain maximum number of attractors
            if len(self.attractor_states) > self.max_attractors:
                # Remove weakest attractor
                weakest_idx = np.argmin(self.attractor_strengths)
                self.attractor_states.pop(weakest_idx)
                self.attractor_rewards.pop(weakest_idx)
                self.attractor_strengths.pop(weakest_idx)
            
            logger.debug(f"Added new attractor state with reward {reward:.3f}")
    
    def compute_attractor_influence(self, current_state: np.ndarray) -> np.ndarray:
        """
        Compute attractor influence on current state for emergent behavior.
        
        Args:
            current_state: Current affective state
            
        Returns:
            Attractor influence vector to be added to state update
        """
        if not self.attractor_states:
            return np.zeros_like(current_state)
        
        total_influence = np.zeros_like(current_state)
        
        for i, attractor_state in enumerate(self.attractor_states):
            # Compute similarity-based influence
            similarity = cosine_similarity(current_state, attractor_state)
            
            if similarity > 0.5:  # Only apply influence for somewhat similar states
                # Influence strength based on similarity, reward, and attractor strength
                influence_magnitude = (
                    similarity * 
                    np.tanh(self.attractor_rewards[i]) * 
                    self.attractor_strengths[i] * 
                    0.05  # Scale factor to prevent overwhelming the dynamics
                )
                
                # Direction of influence: toward the attractor state
                direction = attractor_state - current_state
                direction_normalized = direction / (np.linalg.norm(direction) + 1e-8)
                
                influence = influence_magnitude * direction_normalized
                total_influence += influence
        
        return total_influence
    
    def exhibit_attractor_behavior(self, current_state: np.ndarray, 
                                 semantic_input: SemanticVector, 
                                 reward: float) -> np.ndarray:
        """
        Exhibit attractor behavior toward previously rewarding states.
        
        This method implements the attractor dynamics by:
        1. Recognizing familiar patterns
        2. Computing attractor influence
        3. Adding new attractors for positive experiences
        
        Args:
            current_state: Current affective state
            semantic_input: Current semantic input
            reward: Current reward signal
            
        Returns:
            Modified state update with attractor influence
        """
        # Recognize familiar patterns
        is_familiar, attractor_idx = self.recognize_familiar_patterns(current_state)
        
        # Compute attractor influence
        attractor_influence = self.compute_attractor_influence(current_state)
        
        # Add new attractor if this is a positive experience
        if reward > 0.5:  # Threshold for positive experience
            self.add_attractor_state(current_state, reward)
        
        # Log attractor behavior
        if is_familiar and attractor_idx is not None:
            logger.debug(f"Exhibiting attractor behavior toward state {attractor_idx}")
        
        return attractor_influence
    
    def implement_memory_driven_echo_dynamics(self, memory_echoes: List[MemoryTrace]) -> np.ndarray:
        """
        Implement memory-driven echo dynamics for pattern reinforcement.
        
        This enhances the basic memory echo computation with attractor-aware dynamics.
        
        Args:
            memory_echoes: Retrieved memory traces
            
        Returns:
            Enhanced echo influence with attractor dynamics
        """
        if not memory_echoes:
            return np.zeros(self.config.state_dimension)
        
        # Start with basic echo computation
        basic_echo = self._compute_memory_echo(memory_echoes)
        
        # Enhance with attractor dynamics
        attractor_enhanced_echo = np.zeros_like(basic_echo)
        
        for trace in memory_echoes:
            # Check if this memory corresponds to an attractor state
            is_attractor_memory = False
            attractor_strength = 1.0
            
            for i, attractor_state in enumerate(self.attractor_states):
                similarity = cosine_similarity(trace.affect_state, attractor_state)
                if similarity > 0.8:  # High similarity threshold
                    is_attractor_memory = True
                    attractor_strength = self.attractor_strengths[i]
                    break
            
            # Weight memory echo by attractor strength if applicable
            if is_attractor_memory:
                echo_weight = attractor_strength * 0.1  # Scale factor
                attractor_enhanced_echo += echo_weight * trace.affect_state
        
        # Combine basic echo with attractor-enhanced echo
        total_echo = 0.7 * basic_echo + 0.3 * attractor_enhanced_echo
        
        return total_echo
    
    def get_attractor_summary(self) -> Dict[str, Any]:
        """
        Get summary of current attractor dynamics state.
        
        Returns:
            Dictionary containing attractor information
        """
        return {
            'num_attractors': len(self.attractor_states),
            'average_reward': np.mean(self.attractor_rewards) if self.attractor_rewards else 0.0,
            'average_strength': np.mean(self.attractor_strengths) if self.attractor_strengths else 0.0,
            'strongest_attractor_reward': max(self.attractor_rewards) if self.attractor_rewards else 0.0,
            'pattern_recognition_threshold': self.pattern_recognition_threshold
        }
    
    def decay_attractor_strengths(self, decay_rate: float = 0.01) -> None:
        """
        Gradually decay attractor strengths over time to allow adaptation.
        
        Args:
            decay_rate: Rate at which to decay attractor strengths
        """
        for i in range(len(self.attractor_strengths)):
            self.attractor_strengths[i] = max(0.1, self.attractor_strengths[i] - decay_rate)
        
        # Remove very weak attractors
        indices_to_remove = []
        for i, strength in enumerate(self.attractor_strengths):
            if strength < 0.2:
                indices_to_remove.append(i)
        
        # Remove in reverse order to maintain indices
        for i in reversed(indices_to_remove):
            self.attractor_states.pop(i)
            self.attractor_rewards.pop(i)
            self.attractor_strengths.pop(i)