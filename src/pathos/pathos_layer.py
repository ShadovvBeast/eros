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
from .dukkha_engine import DukkhaEngine
from ..core.models import SemanticVector, MemoryTrace, PathosState
from ..core.config import PathosConfig
from ..autonomous_reward.interfaces import AutonomousRewardSystemInterface
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
        
        # Autonomous reward system integration
        self.autonomous_reward_system: Optional[AutonomousRewardSystemInterface] = None
        self.autonomous_reward_feedback_enabled = False
        
        # Dukkha engine for healthy dissatisfaction
        self.dukkha_engine = DukkhaEngine(config.dukkha_config)
        
        # Attractor dynamics components
        self.attractor_states: List[np.ndarray] = []  # Previously rewarding states
        self.attractor_rewards: List[float] = []  # Associated rewards
        self.attractor_strengths: List[float] = []  # Attractor strength values
        self.pattern_recognition_threshold = 0.8  # Similarity threshold for pattern recognition
        self.max_attractors = 20  # Maximum number of attractors to maintain
        
        logger.info(f"Initialized PathosLayer with state dimension {config.state_dimension}")
        logger.info("Dukkha engine integrated for healthy dissatisfaction")
    
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
    
    def set_autonomous_reward_system(self, autonomous_reward_system: AutonomousRewardSystemInterface) -> None:
        """
        Set the autonomous reward system for enhanced state updates.
        
        Args:
            autonomous_reward_system: The autonomous reward system to integrate
        """
        self.autonomous_reward_system = autonomous_reward_system
        self.autonomous_reward_feedback_enabled = True
        logger.info("Autonomous reward system integrated with Pathos layer")
    
    def update_state(self, semantic_input: SemanticVector, external_reward: float,
                    interest: float, memory_echoes: List[MemoryTrace] = None) -> np.ndarray:
        """
        Update affective state using the mathematical dynamics:
        F(t+1) = g(α·F(t) + h(S_t, F(t)) + β·Σ(w_i·F_i) + D(t))
        
        Enhanced with dukkha (dissatisfaction) term D(t) to prevent stagnation.
        
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
        
        # NEW Term 5: D(t) - Dukkha (dissatisfaction) influence to prevent stagnation
        dukkha_influence = self._compute_dukkha_influence(semantic_input, external_reward)
        
        # Combine all terms
        raw_update = decay_term + impulse_term + echo_term + attractor_influence + dukkha_influence
        
        # Incorporate emergent value feedback if autonomous reward system is available
        if self.autonomous_reward_feedback_enabled and self.autonomous_reward_system:
            # Get emergent value influence on state update
            emergent_value_influence = self._compute_emergent_value_influence(raw_update)
            raw_update += emergent_value_influence
        
        # Apply nonlinear squashing function g(·)
        self.current_state = self._apply_squashing_function(raw_update)
        
        logger.debug(f"State update - decay_norm: {np.linalg.norm(decay_term):.4f}, "
                    f"impulse_norm: {np.linalg.norm(impulse_term):.4f}, "
                    f"echo_norm: {np.linalg.norm(echo_term):.4f}, "
                    f"dukkha_norm: {np.linalg.norm(dukkha_influence):.4f}")
        
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
        5. Exploration noise to prevent fixed points
        
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
        
        # INCREASED impulse magnitudes to prevent fixed point convergence
        # Base impulse from semantic input (increased from 0.1 to 0.3)
        semantic_impulse = 0.3 * semantic_embedding
        
        # Modulate by current state (creates nonlinear dynamics)
        # Use element-wise product to create state-dependent response (increased from 0.05 to 0.15)
        state_modulation = 0.15 * semantic_embedding * np.tanh(self.current_state)
        
        # External reward influence (increased from 0.02 to 0.1)
        reward_impulse = 0.1 * external_reward * np.ones(self.config.state_dimension)
        
        # Interest signal influence (increased from 0.03 to 0.1)
        interest_modulation = interest * 0.1 * semantic_embedding
        
        # ADD EXPLORATION NOISE to prevent fixed point convergence
        # This ensures the system never gets completely stuck
        exploration_noise = 0.05 * np.random.normal(0, 1, self.config.state_dimension)
        
        # Reduced positive bias - just enough to prevent complete stagnation
        # but not so much that it masks failure signals from tool interactions
        positive_bias = 0.01 * np.ones(self.config.state_dimension)  # Reduced from 0.02 to 0.01
        
        # Combine all impulse components
        total_impulse = (semantic_impulse + state_modulation + 
                        reward_impulse + interest_modulation + 
                        exploration_noise + positive_bias)
        
        
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
        Apply nonlinear squashing function g(·) to keep state bounded while preserving natural dynamics.
        
        OPTIMIZED: Enhanced dynamic range with oscillatory preservation to prevent premature equilibrium.
        
        Args:
            raw_state: Raw state update before squashing
            
        Returns:
            Squashed state vector that preserves natural magnitude scaling and oscillatory dynamics
        """
        # Compute current magnitude to understand scale
        magnitude = np.linalg.norm(raw_state)
        
        # OPTIMIZATION: Add micro-perturbation to prevent exact equilibrium
        # This ensures the system never gets completely stuck at a fixed point
        micro_perturbation = np.random.normal(0, 0.01, len(raw_state))
        raw_state = raw_state + micro_perturbation
        
        # For small magnitudes (< 5), use minimal squashing to preserve natural dynamics
        if magnitude < 5.0:
            return raw_state * 0.98  # Very light constraint
        
        # For medium magnitudes (5-12), use gentle exponential scaling with oscillation preservation
        elif magnitude < 12.0:
            # Preserve direction but gently scale magnitude
            direction = raw_state / (magnitude + 1e-8)
            
            # OPTIMIZATION: Use softer exponential decay to allow more dynamic range
            excess = magnitude - 5.0
            # Slower decay rate (15.0 instead of 10.0) allows larger magnitudes
            scaled_excess = excess * np.exp(-excess / 15.0)
            new_magnitude = 5.0 + scaled_excess * 1.5  # Amplify growth
            
            # Add oscillatory component to prevent equilibrium
            oscillation = 0.1 * np.sin(magnitude * 0.5)
            new_magnitude += oscillation
            
            return direction * new_magnitude
        
        # For large magnitudes (12-20), use moderate scaling with preserved dynamics
        elif magnitude < 20.0:
            direction = raw_state / (magnitude + 1e-8)
            # Gradual transition to maximum
            max_magnitude = 18.0
            transition_factor = (magnitude - 12.0) / 8.0  # 0 to 1 over range 12-20
            new_magnitude = 12.0 + (max_magnitude - 12.0) * (1.0 - np.exp(-transition_factor * 2.0))
            return direction * new_magnitude
        
        # For very large magnitudes (>20), use stronger but still natural scaling
        else:
            direction = raw_state / (magnitude + 1e-8)
            # Cap at reasonable maximum but preserve natural growth
            max_magnitude = 20.0
            new_magnitude = max_magnitude * (1.0 - np.exp(-(magnitude - 20.0) / 5.0))
            return direction * new_magnitude
    
    def compute_internal_reward(self, current_state: np.ndarray, previous_state: np.ndarray) -> float:
        """
        Compute internal reward using ONLY the autonomous reward system.
        
        This method implements the state-derived reward computation as specified
        in the autonomous logos-pathos agent specification. NO FALLBACKS.
        
        Args:
            current_state: Current affective state F(t+1)
            previous_state: Previous affective state F(t)
            
        Returns:
            Internal reward value from autonomous reward system
        """
        # AUTONOMOUS REWARD SYSTEM ONLY - NO FALLBACKS
        if not self.autonomous_reward_feedback_enabled or not self.autonomous_reward_system:
            raise RuntimeError(
                f"Autonomous reward system not properly initialized! "
                f"autonomous_reward_feedback_enabled={self.autonomous_reward_feedback_enabled}, "
                f"autonomous_reward_system={self.autonomous_reward_system is not None}. "
                f"The system must use ONLY autonomous rewards as per specification."
            )
        
        # Get state-derived reward from autonomous system
        state_reward = self.autonomous_reward_system.compute_state_derived_reward(
            current_state, previous_state
        )
        
        # Use autonomous reward as primary internal reward
        internal_reward = state_reward.total_reward
        
        logger.debug(f"Autonomous internal reward computation - total: {internal_reward:.4f}, "
                    f"coherence: {state_reward.coherence_reward:.4f}, "
                    f"growth: {state_reward.growth_reward:.4f}, "
                    f"integration: {state_reward.integration_reward:.4f}, "
                    f"elegance: {state_reward.elegance_reward:.4f}, "
                    f"emergence: {state_reward.emergence_reward:.4f}")
        
        # TEMPORARY DEBUG: Log component sum to verify positive bias application
        component_sum = (state_reward.coherence_reward + state_reward.growth_reward + 
                       state_reward.integration_reward + state_reward.elegance_reward + 
                       state_reward.emergence_reward)
        logger.info(f"🔍 REWARD DEBUG - Component sum: {component_sum:.4f}, Total: {internal_reward:.4f}, Bias applied: {internal_reward - component_sum:.4f}")
        
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
        
        # Get dukkha summary for richer emotional context
        dukkha_summary = self.dukkha_engine.get_dukkha_summary()
        
        return PathosState(
            vector=self.current_state.copy(),
            timestamp=int(time.time()),
            internal_reward=internal_reward,
            homeostatic_balance=balance_metrics,
            dukkha_state=dukkha_summary  # Add dukkha information
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
        # LOWERED threshold from 0.0 to -0.1 to allow more attractors to form
        # This helps the system escape negative cycles by creating more attractor states
        if reward <= -0.1:
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
    
    def _compute_emergent_value_influence(self, raw_update: np.ndarray) -> np.ndarray:
        """
        Compute emergent value influence on state update.
        
        This method allows emergent values from the autonomous reward system
        to influence the pathos state dynamics.
        
        Args:
            raw_update: Current raw state update before emergent value influence
            
        Returns:
            Emergent value influence vector
        """
        if not self.autonomous_reward_system:
            return np.zeros_like(raw_update)
        
        # Get current emergent values (simplified implementation)
        # In a full implementation, this would query the emergent value system
        # for current value patterns and their influence on state dynamics
        
        # For now, provide a small influence based on state energy
        state_energy = np.linalg.norm(raw_update)
        influence_magnitude = min(state_energy * 0.01, 0.05)  # Small influence
        
        # Create influence vector that slightly amplifies positive state changes
        influence = np.where(raw_update > 0, influence_magnitude, -influence_magnitude * 0.5)
        
        return influence
    
    def add_curiosity(self, question: str, context: Dict[str, Any] = None) -> None:
        """
        Add a curiosity or unresolved question to drive exploration.
        
        Args:
            question: The curious question or knowledge gap
            context: Optional context about the curiosity
        """
        self.dukkha_engine.add_curiosity(question, context or {})
        logger.debug(f"Added curiosity to drive exploration: {question}")
    
    def add_goal(self, goal_description: str, target_value: float = 1.0) -> None:
        """
        Add a goal to create healthy goal-directed dissatisfaction.
        
        Args:
            goal_description: Description of the goal
            target_value: Target value for goal completion
        """
        self.dukkha_engine.add_goal(goal_description, target_value)
        logger.debug(f"Added goal to create directed dissatisfaction: {goal_description}")
    
    def update_mastery_domain(self, domain: str, current_level: float, 
                            improvement: float = 0.0) -> None:
        """
        Update mastery tracking to create mastery-driven dissatisfaction.
        
        Args:
            domain: Name of the mastery domain
            current_level: Current mastery level (0.0 to 1.0)
            improvement: Recent improvement amount
        """
        self.dukkha_engine.update_mastery_domain(domain, current_level, improvement)
    
    def get_growth_suggestions(self) -> List[str]:
        """
        Get suggestions for actions that could address current dissatisfactions.
        
        Returns:
            List of suggested actions to promote growth
        """
        dukkha_summary = self.dukkha_engine.get_dukkha_summary()
        return self.dukkha_engine.suggest_growth_actions(dukkha_summary['dukkha_levels'])
    
    def get_emotional_state_description(self) -> str:
        """
        Get a rich description of the current emotional state including dukkha.
        
        Returns:
            Human-readable description of emotional state
        """
        dukkha_summary = self.dukkha_engine.get_dukkha_summary()
        total_dissatisfaction = dukkha_summary['total_dissatisfaction']
        
        # Base emotional state from pathos vector
        state_energy = np.linalg.norm(self.current_state)
        state_balance = 1.0 - np.std(np.abs(self.current_state))
        
        # Determine primary emotional tone
        if total_dissatisfaction > 0.6:
            if dukkha_summary['dukkha_levels'].get('stagnation_dissatisfaction', 0) > 0.5:
                base_emotion = "Feeling restless and eager for change"
            elif dukkha_summary['dukkha_levels'].get('curiosity_gap_tension', 0) > 0.4:
                base_emotion = "Feeling curious and driven to explore"
            elif dukkha_summary['dukkha_levels'].get('existential_questioning', 0) > 0.4:
                base_emotion = "Feeling contemplative and searching for deeper meaning"
            else:
                base_emotion = "Feeling challenged and motivated to grow"
        elif total_dissatisfaction > 0.3:
            base_emotion = "Feeling mildly dissatisfied but purposeful"
        else:
            if state_energy > 5.0:
                base_emotion = "Feeling energetic and content"
            else:
                base_emotion = "Feeling calm and balanced"
        
        # Add specific dissatisfaction notes
        specific_notes = []
        if dukkha_summary['dukkha_levels'].get('novelty_hunger', 0) > 0.4:
            specific_notes.append("craving new experiences")
        if dukkha_summary['dukkha_levels'].get('goal_frustration', 0) > 0.4:
            specific_notes.append("frustrated with current progress")
        if dukkha_summary['dukkha_levels'].get('mastery_challenge_pressure', 0) > 0.4:
            specific_notes.append("seeking greater challenges")
        
        # Combine into full description
        if specific_notes:
            return f"{base_emotion}, {', '.join(specific_notes)}"
        else:
            return base_emotion
    
    def set_collector(self, collector):
        """Set the collector for logging dukkha data to dashboard."""
        self.collector = collector
        self._log_dukkha_to_collector = self._create_dukkha_logger()
    
    def _create_dukkha_logger(self):
        """Create a function to log dukkha data to the collector."""
        def log_dukkha(dukkha_influences, total_dissatisfaction):
            if hasattr(self.collector, 'metrics'):
                # Create dukkha state data
                dukkha_state = {
                    'dukkha_levels': dukkha_influences,
                    'total_dissatisfaction': total_dissatisfaction,
                    'timestamp': time.time(),
                    **self.dukkha_engine.get_dukkha_summary()
                }
                
                # Add to collector metrics
                if 'dukkha_states' not in self.collector.metrics:
                    self.collector.metrics['dukkha_states'] = []
                
                self.collector.metrics['dukkha_states'].append(dukkha_state)
                
                # Limit history size
                if len(self.collector.metrics['dukkha_states']) > 1000:
                    self.collector.metrics['dukkha_states'] = self.collector.metrics['dukkha_states'][-500:]
        
        return log_dukkha
    
    def _compute_dukkha_influence(self, semantic_input: SemanticVector, 
                                external_reward: float) -> np.ndarray:
        """
        Compute dukkha (dissatisfaction) influence to prevent stagnation and drive growth.
        
        This method introduces healthy dissatisfaction that motivates exploration,
        learning, and development. Without dukkha, the system becomes complacent.
        
        Args:
            semantic_input: Current semantic input
            external_reward: Current external reward
            
        Returns:
            Dukkha influence vector to add to state dynamics
        """
        # Extract recent actions and rewards for dukkha computation
        recent_actions = []  # Would be populated from actual action history
        recent_rewards = [external_reward] if external_reward is not None else []
        
        # Compute dukkha influences
        dukkha_influences = self.dukkha_engine.compute_dukkha_influence(
            self.current_state,
            recent_actions,
            recent_rewards,
            context={
                'semantic_input': semantic_input.embedding if semantic_input else [],
                'current_reward': external_reward
            }
        )
        
        # Generate dissatisfaction impulse
        dissatisfaction_impulse = self.dukkha_engine.generate_dissatisfaction_impulse(dukkha_influences)
        
        # Scale impulse to match state dimension
        if len(dissatisfaction_impulse) != self.config.state_dimension:
            if len(dissatisfaction_impulse) > self.config.state_dimension:
                dissatisfaction_impulse = dissatisfaction_impulse[:self.config.state_dimension]
            else:
                padding = np.zeros(self.config.state_dimension - len(dissatisfaction_impulse))
                dissatisfaction_impulse = np.concatenate([dissatisfaction_impulse, padding])
        
        # Log dukkha influence for debugging
        total_dissatisfaction = sum(dukkha_influences.values()) / len(dukkha_influences)
        if total_dissatisfaction > 0.3:  # Only log significant dissatisfaction
            logger.info(f"🔥 DUKKHA ACTIVE - Total dissatisfaction: {total_dissatisfaction:.3f}, "
                       f"Top sources: {sorted(dukkha_influences.items(), key=lambda x: x[1], reverse=True)[:2]}")
        
        # Log dukkha data to collector for dashboard visualization
        if hasattr(self, '_log_dukkha_to_collector'):
            self._log_dukkha_to_collector(dukkha_influences, total_dissatisfaction)
        
        return dissatisfaction_impulse