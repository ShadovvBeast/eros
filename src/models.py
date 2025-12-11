"""
Core data models for the Autonomous Logos-Pathos-Memory Agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
import json
from datetime import datetime


@dataclass
class PathosState:
    """High-dimensional affective state vector F(t)"""
    vector: np.ndarray  # F(t) - high-dimensional affective state
    timestamp: int
    internal_reward: float
    homeostatic_balance: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'vector': self.vector.tolist(),
            'timestamp': self.timestamp,
            'internal_reward': self.internal_reward,
            'homeostatic_balance': self.homeostatic_balance
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PathosState':
        """Deserialize from dictionary"""
        return cls(
            vector=np.array(data['vector']),
            timestamp=data['timestamp'],
            internal_reward=data['internal_reward'],
            homeostatic_balance=data['homeostatic_balance']
        )


@dataclass
class SemanticVector:
    """Vector representation of thoughts and concepts from Logos"""
    embedding: np.ndarray
    intention_text: str
    semantic_category: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'embedding': self.embedding.tolist(),
            'intention_text': self.intention_text,
            'semantic_category': self.semantic_category
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SemanticVector':
        """Deserialize from dictionary"""
        return cls(
            embedding=np.array(data['embedding']),
            intention_text=data['intention_text'],
            semantic_category=data['semantic_category']
        )


@dataclass
class MemoryTrace:
    """Stored experience containing affective state, semantic representation, reward, and timestamp"""
    affect_state: np.ndarray
    semantic_vector: SemanticVector
    reward: float
    timestamp: int
    salience: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'affect_state': self.affect_state.tolist(),
            'semantic_vector': self.semantic_vector.to_dict(),
            'reward': self.reward,
            'timestamp': self.timestamp,
            'salience': self.salience,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryTrace':
        """Deserialize from dictionary"""
        return cls(
            affect_state=np.array(data['affect_state']),
            semantic_vector=SemanticVector.from_dict(data['semantic_vector']),
            reward=data['reward'],
            timestamp=data['timestamp'],
            salience=data['salience'],
            metadata=data.get('metadata', {})
        )


@dataclass
class Intention:
    """Generated intention with semantic representation and tool candidates"""
    description: str
    semantic_vector: SemanticVector
    priority: float
    tool_candidates: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'description': self.description,
            'semantic_vector': self.semantic_vector.to_dict(),
            'priority': self.priority,
            'tool_candidates': self.tool_candidates
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Intention':
        """Deserialize from dictionary"""
        return cls(
            description=data['description'],
            semantic_vector=SemanticVector.from_dict(data['semantic_vector']),
            priority=data['priority'],
            tool_candidates=data['tool_candidates']
        )


@dataclass
class ToolCall:
    """Tool execution request with arguments"""
    tool_name: str
    arguments: Dict[str, Any]
    timestamp: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'tool_name': self.tool_name,
            'arguments': self.arguments,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """Deserialize from dictionary"""
        return cls(
            tool_name=data['tool_name'],
            arguments=data['arguments'],
            timestamp=data['timestamp']
        )


@dataclass
class ToolResult:
    """Result from tool execution"""
    tool_name: str
    result: Any
    success: bool
    error_message: Optional[str] = None
    timestamp: int = field(default_factory=lambda: int(datetime.now().timestamp()))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'tool_name': self.tool_name,
            'result': self.result,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResult':
        """Deserialize from dictionary"""
        return cls(
            tool_name=data['tool_name'],
            result=data['result'],
            success=data['success'],
            error_message=data.get('error_message'),
            timestamp=data['timestamp']
        )