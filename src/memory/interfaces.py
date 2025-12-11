"""
Memory System interfaces and base classes.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np

from ..models import MemoryTrace, SemanticVector
from ..config import MemoryConfig


class MemorySystem(ABC):
    """Abstract base class for the Memory associative storage system"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
    
    @abstractmethod
    def store_trace(self, trace: MemoryTrace) -> None:
        """
        Store a memory trace in the associative storage system.
        
        Args:
            trace: MemoryTrace to store
        """
        pass
    
    @abstractmethod
    def query_by_affect(self, affect_state: np.ndarray, k: int = 5) -> List[MemoryTrace]:
        """
        Retrieve memories primarily by affective similarity to current state.
        
        Args:
            affect_state: Current affective state for similarity matching
            k: Number of memories to retrieve
            
        Returns:
            List of most similar memory traces
        """
        pass
    
    @abstractmethod
    def query_by_semantic(self, semantic_vector: SemanticVector, k: int = 5) -> List[MemoryTrace]:
        """
        Retrieve memories by semantic similarity.
        
        Args:
            semantic_vector: Semantic vector for similarity matching
            k: Number of memories to retrieve
            
        Returns:
            List of most similar memory traces
        """
        pass
    
    @abstractmethod
    def estimate_novelty(self, affect_state: np.ndarray, semantic_vector: SemanticVector) -> Tuple[float, float]:
        """
        Estimate novelty by computing distance to nearest existing entries.
        
        Args:
            affect_state: Affective state to evaluate
            semantic_vector: Semantic vector to evaluate
            
        Returns:
            Tuple of (affect_novelty, semantic_novelty)
        """
        pass
    
    @abstractmethod
    def get_recent_traces(self, n: int = 10) -> List[MemoryTrace]:
        """
        Get the most recent memory traces.
        
        Args:
            n: Number of recent traces to retrieve
            
        Returns:
            List of recent memory traces
        """
        pass
    
    @abstractmethod
    def get_trace_count(self) -> int:
        """
        Get the total number of stored memory traces.
        
        Returns:
            Number of stored traces
        """
        pass
    
    @abstractmethod
    def clear_memories(self) -> None:
        """Clear all stored memories (for testing/reset)"""
        pass