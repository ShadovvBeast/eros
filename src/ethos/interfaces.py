"""
Ethos Framework interfaces and base classes.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
from ..core.models import Intention, ToolCall
from ..core.config import EthosConfig


class EthosFramework(ABC):
    """Abstract base class for the Ethos identity and constraint system"""
    
    def __init__(self, config: EthosConfig, pathos_layer=None):
        self.config = config
        self._pathos_layer = pathos_layer
    
    @abstractmethod
    def validate_intention(self, intention: Intention) -> Tuple[bool, Optional[str]]:
        """
        Validate intention against identity and safety constraints.
        
        Args:
            intention: Intention to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def validate_tool_call(self, tool_call: ToolCall) -> Tuple[bool, Optional[str]]:
        """
        Validate tool call against safety restrictions.
        
        Args:
            tool_call: Tool call to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def get_personality_bias(self, semantic_category: str) -> float:
        """
        Get personality bias for a semantic category.
        
        Args:
            semantic_category: Category to evaluate
            
        Returns:
            Bias value (-1.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def check_value_alignment(self, action_description: str) -> float:
        """
        Check how well an action aligns with core values.
        
        Args:
            action_description: Description of the action
            
        Returns:
            Alignment score (0.0 to 1.0)
        """
        pass
    
    @abstractmethod
    def get_identity_description(self) -> str:
        """
        Get the current identity description.
        
        Returns:
            Identity description string
        """
        pass
    
    @abstractmethod
    def update_personality_traits(self, trait_updates: Dict[str, float]) -> None:
        """
        Update personality traits based on experience.
        
        Args:
            trait_updates: Dictionary of trait changes
        """
        pass
    
    @abstractmethod
    def ensure_intention_reflects_personality(self, intention: Intention) -> Intention:
        """
        Ensure intentions reflect personality traits by adjusting priority and description.
        
        Args:
            intention: Original intention
            
        Returns:
            Modified intention that reflects personality traits
        """
        pass
    
    @abstractmethod
    def validate_decision_against_identity(self, decision_description: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that a decision aligns with identity principles.
        
        Args:
            decision_description: Description of the decision
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        pass
    
    @abstractmethod
    def demonstrate_consistent_personality(self, context: str) -> Dict[str, Any]:
        """
        Generate personality demonstration markers for a given context.
        
        Args:
            context: Context in which to demonstrate personality
            
        Returns:
            Dictionary containing personality demonstration elements
        """
        pass