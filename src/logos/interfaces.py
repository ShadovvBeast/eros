"""
Logos Layer interfaces and base classes.
"""

from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np

from ..models import SemanticVector, Intention, MemoryTrace, ToolCall
from ..config import LogosConfig


class LogosLayer(ABC):
    """Abstract base class for the Logos cognitive reasoning engine"""
    
    def __init__(self, config: LogosConfig):
        self.config = config
    
    @abstractmethod
    def interpret_state(self, pathos_state: np.ndarray, memories: List[MemoryTrace]) -> SemanticVector:
        """
        Interpret current pathos state and memories to create semantic representation.
        
        Args:
            pathos_state: Current high-dimensional affective state F(t)
            memories: Recently retrieved memory traces
            
        Returns:
            SemanticVector representing the interpreted state
        """
        pass
    
    @abstractmethod
    def generate_intention(self, semantic_vector: SemanticVector, pathos_state: np.ndarray) -> Intention:
        """
        Generate intention based on semantic interpretation and current state.
        
        Args:
            semantic_vector: Interpreted semantic representation
            pathos_state: Current affective state
            
        Returns:
            Generated intention with priority and tool candidates
        """
        pass
    
    @abstractmethod
    def evaluate_tool_usage(self, intention: Intention, available_tools: List[str]) -> Optional[ToolCall]:
        """
        Evaluate whether tool usage is needed for the current intention.
        
        Args:
            intention: Current intention to evaluate
            available_tools: List of available tool names
            
        Returns:
            ToolCall if tool usage is needed, None otherwise
        """
        pass
    
    @abstractmethod
    def compute_interest_signal(self, semantic_input: SemanticVector) -> float:
        """
        Compute interest signal for Pathos integration.
        
        Args:
            semantic_input: Semantic vector to evaluate
            
        Returns:
            Interest signal value (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def update_preferences(self, reward: float, semantic_vector: SemanticVector) -> None:
        """
        Update preference weights based on reward feedback.
        
        Args:
            reward: Reward signal from experience
            semantic_vector: Associated semantic representation
        """
        pass