"""
Pytest configuration and fixtures for the Autonomous Logos-Pathos-Memory Agent tests.
"""

import pytest
import numpy as np
from hypothesis import settings, Verbosity
from typing import Dict, Any

from src.config import AgentConfig, PathosConfig, EthosConfig, LogosConfig, MemoryConfig, ToolConfig
from src.models import PathosState, SemanticVector, MemoryTrace, Intention


# Configure Hypothesis for property-based testing
settings.register_profile("default", max_examples=100, verbosity=Verbosity.normal)
settings.load_profile("default")


@pytest.fixture
def pathos_config():
    """Default Pathos configuration for testing"""
    return PathosConfig(
        state_dimension=64,  # Smaller for testing
        decay_factor=0.95,
        echo_strength=0.1,
        salience_threshold=0.7,
        memory_retrieval_k=5
    )


@pytest.fixture
def ethos_config():
    """Default Ethos configuration for testing"""
    return EthosConfig(
        identity_description="Test agent for validation",
        core_values=["safety", "learning", "coherence"],
        safety_constraints=["No harmful actions", "Respect boundaries"],
        allowed_tool_categories=["test", "analysis"],
        personality_traits={
            'openness': 0.7,
            'conscientiousness': 0.8,
            'extraversion': 0.5,
            'agreeableness': 0.7,
            'neuroticism': 0.3
        }
    )


@pytest.fixture
def logos_config():
    """Default Logos configuration for testing"""
    return LogosConfig(
        semantic_dimension=256,  # Smaller for testing
        max_intentions_per_cycle=2,
        planning_horizon=3,
        interest_threshold=0.5,
        gemini_api_key="test_key"
    )


@pytest.fixture
def memory_config():
    """Default Memory configuration for testing"""
    return MemoryConfig(
        max_memory_traces=1000,
        similarity_threshold=0.8,
        novelty_k_neighbors=5,
        storage_backend='memory',  # Use in-memory for testing
        storage_path='test_memories.json'
    )


@pytest.fixture
def tool_config():
    """Default Tool configuration for testing"""
    return ToolConfig(
        max_concurrent_tools=2,
        tool_timeout_seconds=10,
        allowed_tool_types=['test', 'mock']
    )


@pytest.fixture
def agent_config(pathos_config, ethos_config, logos_config, memory_config, tool_config):
    """Complete agent configuration for testing"""
    return AgentConfig(
        pathos=pathos_config,
        ethos=ethos_config,
        logos=logos_config,
        memory=memory_config,
        tools=tool_config,
        cycle_interval_seconds=0.1,  # Fast cycles for testing
        log_level='DEBUG'
    )


@pytest.fixture
def sample_pathos_state():
    """Sample PathosState for testing"""
    return PathosState(
        vector=np.random.randn(64),
        timestamp=1000,
        internal_reward=0.5,
        homeostatic_balance={'stability': 0.8, 'complexity': 0.6, 'coherence': 0.7}
    )


@pytest.fixture
def sample_semantic_vector():
    """Sample SemanticVector for testing"""
    return SemanticVector(
        embedding=np.random.randn(256),
        intention_text="Test intention for exploration",
        semantic_category="exploration"
    )


@pytest.fixture
def sample_memory_trace(sample_semantic_vector):
    """Sample MemoryTrace for testing"""
    return MemoryTrace(
        affect_state=np.random.randn(64),
        semantic_vector=sample_semantic_vector,
        reward=0.3,
        timestamp=1000,
        salience=0.8,
        metadata={'test': True}
    )


@pytest.fixture
def sample_intention(sample_semantic_vector):
    """Sample Intention for testing"""
    return Intention(
        description="Test intention for validation",
        semantic_vector=sample_semantic_vector,
        priority=0.7,
        tool_candidates=['test_tool', 'analysis_tool']
    )