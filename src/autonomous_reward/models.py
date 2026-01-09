"""
Data models for the Autonomous State-Derived Reward System.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import numpy as np
from datetime import datetime


@dataclass
class StateReward:
    """Reward derived from state dynamics"""
    coherence_reward: float
    growth_reward: float
    integration_reward: float
    elegance_reward: float
    emergence_reward: float
    total_reward: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'coherence_reward': self.coherence_reward,
            'growth_reward': self.growth_reward,
            'integration_reward': self.integration_reward,
            'elegance_reward': self.elegance_reward,
            'emergence_reward': self.emergence_reward,
            'total_reward': self.total_reward,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateReward':
        """Deserialize from dictionary"""
        return cls(
            coherence_reward=data['coherence_reward'],
            growth_reward=data['growth_reward'],
            integration_reward=data['integration_reward'],
            elegance_reward=data['elegance_reward'],
            emergence_reward=data['emergence_reward'],
            total_reward=data['total_reward'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class IntrinsicMotivation:
    """Self-generated motivation signals"""
    curiosity_drive: float
    mastery_drive: float
    autonomy_drive: float
    growth_drive: float
    combined_motivation: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'curiosity_drive': self.curiosity_drive,
            'mastery_drive': self.mastery_drive,
            'autonomy_drive': self.autonomy_drive,
            'growth_drive': self.growth_drive,
            'combined_motivation': self.combined_motivation
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'IntrinsicMotivation':
        """Deserialize from dictionary"""
        return cls(
            curiosity_drive=data['curiosity_drive'],
            mastery_drive=data['mastery_drive'],
            autonomy_drive=data['autonomy_drive'],
            growth_drive=data['growth_drive'],
            combined_motivation=data['combined_motivation']
        )


@dataclass
class ValuePattern:
    """Emergent value pattern"""
    pattern_id: str
    value_type: str
    strength: float
    contexts: List[str]
    associated_rewards: List[float]
    emergence_date: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'pattern_id': self.pattern_id,
            'value_type': self.value_type,
            'strength': self.strength,
            'contexts': self.contexts,
            'associated_rewards': self.associated_rewards,
            'emergence_date': self.emergence_date.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValuePattern':
        """Deserialize from dictionary"""
        return cls(
            pattern_id=data['pattern_id'],
            value_type=data['value_type'],
            strength=data['strength'],
            contexts=data['contexts'],
            associated_rewards=data['associated_rewards'],
            emergence_date=datetime.fromisoformat(data['emergence_date'])
        )


@dataclass
class LearningState:
    """Current learning configuration"""
    learning_rate: float
    complexity_level: float
    readiness_score: float
    consolidation_needed: bool
    meta_learning_params: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'learning_rate': self.learning_rate,
            'complexity_level': self.complexity_level,
            'readiness_score': self.readiness_score,
            'consolidation_needed': self.consolidation_needed,
            'meta_learning_params': self.meta_learning_params
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningState':
        """Deserialize from dictionary"""
        return cls(
            learning_rate=data['learning_rate'],
            complexity_level=data['complexity_level'],
            readiness_score=data['readiness_score'],
            consolidation_needed=data['consolidation_needed'],
            meta_learning_params=data.get('meta_learning_params', {})
        )


@dataclass
class WorldInteractionResult:
    """Result of world interaction"""
    action_type: str
    success_level: float
    discovery_value: float
    connection_quality: float
    creativity_score: float
    total_reward: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'action_type': self.action_type,
            'success_level': self.success_level,
            'discovery_value': self.discovery_value,
            'connection_quality': self.connection_quality,
            'creativity_score': self.creativity_score,
            'total_reward': self.total_reward
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorldInteractionResult':
        """Deserialize from dictionary"""
        return cls(
            action_type=data['action_type'],
            success_level=data['success_level'],
            discovery_value=data['discovery_value'],
            connection_quality=data['connection_quality'],
            creativity_score=data['creativity_score'],
            total_reward=data['total_reward']
        )


@dataclass
class AutonomousGoal:
    """Self-generated goal"""
    goal_id: str
    description: str
    value_alignment: float
    complexity_level: float
    expected_reward: float
    creation_state: np.ndarray
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'goal_id': self.goal_id,
            'description': self.description,
            'value_alignment': self.value_alignment,
            'complexity_level': self.complexity_level,
            'expected_reward': self.expected_reward,
            'creation_state': self.creation_state.tolist()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AutonomousGoal':
        """Deserialize from dictionary"""
        return cls(
            goal_id=data['goal_id'],
            description=data['description'],
            value_alignment=data['value_alignment'],
            complexity_level=data['complexity_level'],
            expected_reward=data['expected_reward'],
            creation_state=np.array(data['creation_state'])
        )


@dataclass
class Experience:
    """Experience data for learning and value development"""
    state_before: np.ndarray
    state_after: np.ndarray
    action_taken: str
    reward_received: float
    context: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'state_before': self.state_before.tolist(),
            'state_after': self.state_after.tolist(),
            'action_taken': self.action_taken,
            'reward_received': self.reward_received,
            'context': self.context,
            'timestamp': self.timestamp.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Experience':
        """Deserialize from dictionary"""
        return cls(
            state_before=np.array(data['state_before']),
            state_after=np.array(data['state_after']),
            action_taken=data['action_taken'],
            reward_received=data['reward_received'],
            context=data['context'],
            timestamp=datetime.fromisoformat(data['timestamp'])
        )


@dataclass
class Pattern:
    """Detected pattern in state dynamics"""
    pattern_type: str
    strength: float
    frequency: int
    last_occurrence: datetime
    characteristics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for persistence"""
        return {
            'pattern_type': self.pattern_type,
            'strength': self.strength,
            'frequency': self.frequency,
            'last_occurrence': self.last_occurrence.isoformat(),
            'characteristics': self.characteristics
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Pattern':
        """Deserialize from dictionary"""
        return cls(
            pattern_type=data['pattern_type'],
            strength=data['strength'],
            frequency=data['frequency'],
            last_occurrence=datetime.fromisoformat(data['last_occurrence']),
            characteristics=data.get('characteristics', {})
        )