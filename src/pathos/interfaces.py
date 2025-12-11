"""
Pathos Layer interfaces and base classes.
"""

from abc import ABC, abstractmethod
from typing import Tuple, List, Optional, Dict
import numpy as np

from ..models import SemanticVector, MemoryTrace
from ..config import PathosConfig


class PathosLayer(ABC):
    """Abstract base class for the Pathos affective dynamics engine"""
    
    def __init__(self, config: PathosConfig):
        self.config = config
        self.current_state: np.ndarray = self._initialize_state()
        self.previous_state: Optional[np.ndarray] = None
    
    @abstractmethod
    def _initialize_state(self) -> np.ndarray:
        """Initialize the high-dimensional affective state vector F(t)"""
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def compute_internal_reward(self, current_state: np.ndarray, previous_state: np.ndarray) -> float:
        """
        Compute internal reward: r_t^int = -λ₁·D_t - λ₂·||F(t+1) - F(t)||²
        
        Args:
            current_state: Current affective state F(t+1)
            previous_state: Previous affective state F(t)
            
        Returns:
            Internal reward value
        """
        pass
    
    @abstractmethod
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
        pass
    
    @abstractmethod
    def should_write_memory(self, salience: float) -> bool:
        """
        Determine if experience should be stored as memory based on salience.
        
        Args:
            salience: Computed salience score
            
        Returns:
            True if memory should be written, False otherwise
        """
        pass
    
    @abstractmethod
    def compute_homeostatic_balance(self, state: np.ndarray) -> Tuple[Dict[str, float], float]:
        """
        Compute homeostatic balance metrics and discomfort.
        
        Args:
            state: Current affective state
            
        Returns:
            Tuple of (balance_metrics, discomfort_D_t)
        """
        pass