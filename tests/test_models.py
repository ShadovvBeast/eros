"""
Tests for core data models and serialization.
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, HealthCheck
from hypothesis.extra.numpy import arrays

from src.models import PathosState, SemanticVector, MemoryTrace, Intention, ToolCall, ToolResult
from src.math_utils import (
    cosine_similarity, euclidean_distance, manhattan_distance, normalize_vector,
    tanh_squash, sigmoid_squash, soft_clamp, compute_homeostatic_balance,
    compute_state_change_penalty, find_k_nearest_neighbors, compute_similarity_weights
)


class TestDataModelSerialization:
    """Test serialization/deserialization of core data models"""
    
    @given(
        vector=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64, 128]), 
                     elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)),
        timestamp=st.integers(min_value=0, max_value=2**30),
        internal_reward=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        homeostatic_balance=st.dictionaries(
            st.sampled_from(['stability', 'coherence', 'complexity', 'balance']),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=1, max_size=4
        )
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_pathos_state_round_trip_property(self, vector, timestamp, internal_reward, homeostatic_balance):
        """**Feature: autonomous-logos-pathos-agent, Property 1: Data model round trip consistency**
        **Validates: Requirements 2.1, 3.2**
        
        For any valid PathosState, serializing then deserializing should produce an equivalent object.
        """
        original = PathosState(
            vector=vector,
            timestamp=timestamp,
            internal_reward=internal_reward,
            homeostatic_balance=homeostatic_balance
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = PathosState.from_dict(serialized)
        
        # Verify equality
        np.testing.assert_array_equal(original.vector, restored.vector)
        assert original.timestamp == restored.timestamp
        assert original.internal_reward == restored.internal_reward
        assert original.homeostatic_balance == restored.homeostatic_balance
    
    @given(
        embedding=arrays(np.float64, shape=st.sampled_from([16, 32, 64, 128, 256]),
                        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        intention_text=st.text(min_size=1, max_size=200),
        semantic_category=st.sampled_from(['exploration', 'analysis', 'communication', 'learning', 'planning'])
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_semantic_vector_round_trip_property(self, embedding, intention_text, semantic_category):
        """**Feature: autonomous-logos-pathos-agent, Property 1: Data model round trip consistency**
        **Validates: Requirements 2.1, 3.2**
        
        For any valid SemanticVector, serializing then deserializing should produce an equivalent object.
        """
        original = SemanticVector(
            embedding=embedding,
            intention_text=intention_text,
            semantic_category=semantic_category
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = SemanticVector.from_dict(serialized)
        
        # Verify equality
        np.testing.assert_array_equal(original.embedding, restored.embedding)
        assert original.intention_text == restored.intention_text
        assert original.semantic_category == restored.semantic_category
    
    @given(
        affect_state=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64, 128]),
                           elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)),
        embedding=arrays(np.float64, shape=st.sampled_from([16, 32, 64, 128, 256]),
                        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        intention_text=st.text(min_size=1, max_size=200),
        semantic_category=st.sampled_from(['exploration', 'analysis', 'communication', 'learning', 'planning']),
        reward=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        timestamp=st.integers(min_value=0, max_value=2**30),
        salience=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        metadata=st.dictionaries(
            st.sampled_from(['source', 'type', 'confidence', 'context']),
            st.one_of(st.text(max_size=50), st.integers(min_value=0, max_value=1000), 
                     st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), 
                     st.booleans()),
            max_size=3
        )
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_memory_trace_round_trip_property(self, affect_state, embedding, intention_text, semantic_category, 
                                            reward, timestamp, salience, metadata):
        """**Feature: autonomous-logos-pathos-agent, Property 1: Data model round trip consistency**
        **Validates: Requirements 2.1, 3.2**
        
        For any valid MemoryTrace, serializing then deserializing should produce an equivalent object.
        """
        semantic_vector = SemanticVector(
            embedding=embedding,
            intention_text=intention_text,
            semantic_category=semantic_category
        )
        
        original = MemoryTrace(
            affect_state=affect_state,
            semantic_vector=semantic_vector,
            reward=reward,
            timestamp=timestamp,
            salience=salience,
            metadata=metadata
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = MemoryTrace.from_dict(serialized)
        
        # Verify equality
        np.testing.assert_array_equal(original.affect_state, restored.affect_state)
        assert original.reward == restored.reward
        assert original.timestamp == restored.timestamp
        assert original.salience == restored.salience
        assert original.metadata == restored.metadata
        
        # Verify nested semantic vector
        np.testing.assert_array_equal(
            original.semantic_vector.embedding, 
            restored.semantic_vector.embedding
        )
        assert original.semantic_vector.intention_text == restored.semantic_vector.intention_text
        assert original.semantic_vector.semantic_category == restored.semantic_vector.semantic_category
    
    @given(
        description=st.text(min_size=1, max_size=200),
        embedding=arrays(np.float64, shape=st.sampled_from([16, 32, 64, 128, 256]),
                        elements=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False)),
        intention_text=st.text(min_size=1, max_size=200),
        semantic_category=st.sampled_from(['exploration', 'analysis', 'communication', 'learning', 'planning']),
        priority=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        tool_candidates=st.lists(st.sampled_from(['web_search', 'file_read', 'analysis', 'communication']), 
                                min_size=0, max_size=5)
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_intention_round_trip_property(self, description, embedding, intention_text, semantic_category, 
                                         priority, tool_candidates):
        """**Feature: autonomous-logos-pathos-agent, Property 1: Data model round trip consistency**
        **Validates: Requirements 2.1, 3.2**
        
        For any valid Intention, serializing then deserializing should produce an equivalent object.
        """
        semantic_vector = SemanticVector(
            embedding=embedding,
            intention_text=intention_text,
            semantic_category=semantic_category
        )
        
        original = Intention(
            description=description,
            semantic_vector=semantic_vector,
            priority=priority,
            tool_candidates=tool_candidates
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = Intention.from_dict(serialized)
        
        # Verify equality
        assert original.description == restored.description
        assert original.priority == restored.priority
        assert original.tool_candidates == restored.tool_candidates
        
        # Verify nested semantic vector
        np.testing.assert_array_equal(
            original.semantic_vector.embedding, 
            restored.semantic_vector.embedding
        )
        assert original.semantic_vector.intention_text == restored.semantic_vector.intention_text
        assert original.semantic_vector.semantic_category == restored.semantic_vector.semantic_category
    
    def test_pathos_state_round_trip(self):
        """Test PathosState serialization round trip"""
        original = PathosState(
            vector=np.array([1.0, 2.0, 3.0]),
            timestamp=1000,
            internal_reward=0.5,
            homeostatic_balance={'stability': 0.8, 'coherence': 0.7}
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = PathosState.from_dict(serialized)
        
        # Verify equality
        np.testing.assert_array_equal(original.vector, restored.vector)
        assert original.timestamp == restored.timestamp
        assert original.internal_reward == restored.internal_reward
        assert original.homeostatic_balance == restored.homeostatic_balance
    
    def test_semantic_vector_round_trip(self):
        """Test SemanticVector serialization round trip"""
        original = SemanticVector(
            embedding=np.array([0.1, 0.2, 0.3, 0.4]),
            intention_text="Test intention",
            semantic_category="exploration"
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = SemanticVector.from_dict(serialized)
        
        # Verify equality
        np.testing.assert_array_equal(original.embedding, restored.embedding)
        assert original.intention_text == restored.intention_text
        assert original.semantic_category == restored.semantic_category
    
    def test_memory_trace_round_trip(self):
        """Test MemoryTrace serialization round trip"""
        semantic_vector = SemanticVector(
            embedding=np.array([0.1, 0.2]),
            intention_text="Test",
            semantic_category="test"
        )
        
        original = MemoryTrace(
            affect_state=np.array([1.0, 2.0]),
            semantic_vector=semantic_vector,
            reward=0.3,
            timestamp=1000,
            salience=0.8,
            metadata={'test': True}
        )
        
        # Serialize and deserialize
        serialized = original.to_dict()
        restored = MemoryTrace.from_dict(serialized)
        
        # Verify equality
        np.testing.assert_array_equal(original.affect_state, restored.affect_state)
        assert original.reward == restored.reward
        assert original.timestamp == restored.timestamp
        assert original.salience == restored.salience
        assert original.metadata == restored.metadata
        
        # Verify nested semantic vector
        np.testing.assert_array_equal(
            original.semantic_vector.embedding, 
            restored.semantic_vector.embedding
        )
        assert original.semantic_vector.intention_text == restored.semantic_vector.intention_text


class TestConfigurationValidation:
    """Test configuration classes"""
    
    def test_agent_config_from_env(self, monkeypatch):
        """Test AgentConfig creation from environment variables"""
        from src.config import AgentConfig
        
        # Set environment variables
        monkeypatch.setenv('PATHOS_STATE_DIMENSION', '256')
        monkeypatch.setenv('PATHOS_DECAY_FACTOR', '0.9')
        monkeypatch.setenv('GEMINI_API_KEY', 'test_key_123')
        monkeypatch.setenv('LOG_LEVEL', 'DEBUG')
        
        config = AgentConfig.from_env()
        
        assert config.pathos.state_dimension == 256
        assert config.pathos.decay_factor == 0.9
        assert config.logos.gemini_api_key == 'test_key_123'
        assert config.log_level == 'DEBUG'


class TestLoggingAndInstrumentation:
    """Test logging and instrumentation systems"""
    
    def test_logger_initialization(self):
        """Test logger can be initialized without errors"""
        from src.logging_config import AgentLogger
        
        logger = AgentLogger(log_level='DEBUG')
        
        # Test logging methods don't raise exceptions
        logger.info("Test info message", test_param="value")
        logger.debug("Test debug message")
        logger.warning("Test warning")
        logger.error("Test error")
    
    def test_instrumentation_collector(self):
        """Test instrumentation data collection"""
        from src.logging_config import InstrumentationCollector
        
        collector = InstrumentationCollector(enable_monitoring=True)
        
        # Record some test data
        collector.record_pathos_state(1, np.array([1.0, 2.0]), 0.5, {'stability': 0.8})
        collector.record_memory_event('store', 0.7, 10)
        collector.record_tool_usage('test_tool', True, 0.1)
        collector.record_cycle_timing(1, 0.5, {'logos': 0.1, 'pathos': 0.2})
        
        # Verify data was recorded
        summary = collector.get_metrics_summary()
        assert summary['total_cycles'] == 1
        assert summary['total_memory_events'] == 1
        assert summary['total_tool_calls'] == 1
        
        # Test disabled monitoring
        disabled_collector = InstrumentationCollector(enable_monitoring=False)
        disabled_collector.record_pathos_state(1, np.array([1.0]), 0.5, {})
        
        disabled_summary = disabled_collector.get_metrics_summary()
        assert disabled_summary == {}


class TestMathematicalOperations:
    """Property-based tests for mathematical utility functions"""
    
    @given(
        a=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64]),
                elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        b=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64]),
                elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_cosine_similarity_properties(self, a, b):
        """**Feature: autonomous-logos-pathos-agent, Property 4: Mathematical operation consistency**
        **Validates: Requirements 2.2, 2.3**
        
        For any two vectors, cosine similarity should be symmetric and bounded.
        """
        # Ensure vectors have same shape for this test
        min_len = min(len(a), len(b))
        a_trimmed = a[:min_len]
        b_trimmed = b[:min_len]
        
        # Skip if either vector is zero (undefined cosine similarity)
        if np.linalg.norm(a_trimmed) == 0 or np.linalg.norm(b_trimmed) == 0:
            return
        
        sim_ab = cosine_similarity(a_trimmed, b_trimmed)
        sim_ba = cosine_similarity(b_trimmed, a_trimmed)
        
        # Symmetry property
        assert abs(sim_ab - sim_ba) < 1e-10
        
        # Bounded property (with floating point tolerance)
        assert -1.0 - 1e-10 <= sim_ab <= 1.0 + 1e-10
        
        # Self-similarity should be 1 (for non-zero vectors)
        sim_aa = cosine_similarity(a_trimmed, a_trimmed)
        assert abs(sim_aa - 1.0) < 1e-10
    
    @given(
        a=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64]),
                elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        b=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64]),
                elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_distance_properties(self, a, b):
        """**Feature: autonomous-logos-pathos-agent, Property 4: Mathematical operation consistency**
        **Validates: Requirements 2.2, 2.3**
        
        For any two vectors, distance metrics should satisfy basic properties.
        """
        # Ensure vectors have same shape for this test
        min_len = min(len(a), len(b))
        a_trimmed = a[:min_len]
        b_trimmed = b[:min_len]
        
        euclidean_ab = euclidean_distance(a_trimmed, b_trimmed)
        euclidean_ba = euclidean_distance(b_trimmed, a_trimmed)
        manhattan_ab = manhattan_distance(a_trimmed, b_trimmed)
        manhattan_ba = manhattan_distance(b_trimmed, a_trimmed)
        
        # Symmetry property
        assert abs(euclidean_ab - euclidean_ba) < 1e-10
        assert abs(manhattan_ab - manhattan_ba) < 1e-10
        
        # Non-negativity
        assert euclidean_ab >= 0
        assert manhattan_ab >= 0
        
        # Self-distance should be 0
        assert euclidean_distance(a_trimmed, a_trimmed) < 1e-10
        assert manhattan_distance(a_trimmed, a_trimmed) < 1e-10
    
    @given(
        vector=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64]),
                     elements=st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False)),
        method=st.sampled_from(['l2', 'l1', 'max', 'unit_range'])
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_normalization_properties(self, vector, method):
        """**Feature: autonomous-logos-pathos-agent, Property 4: Mathematical operation consistency**
        **Validates: Requirements 2.2, 2.3**
        
        For any vector and normalization method, normalized vector should have expected properties.
        """
        # Skip zero vectors for l1 and l2 normalization
        if method in ['l1', 'l2'] and np.linalg.norm(vector) == 0:
            return
        
        normalized = normalize_vector(vector, method)
        
        # Should have same shape
        assert normalized.shape == vector.shape
        
        # Check specific properties based on method
        if method == 'l2' and np.linalg.norm(vector) > 0:
            # L2 norm should be 1
            assert abs(np.linalg.norm(normalized) - 1.0) < 1e-10
        
        elif method == 'l1' and np.sum(np.abs(vector)) > 0:
            # L1 norm should be 1
            assert abs(np.sum(np.abs(normalized)) - 1.0) < 1e-10
        
        elif method == 'max':
            # Max absolute value should be 1 (unless zero vector)
            max_val = np.max(np.abs(normalized))
            if np.max(np.abs(vector)) > 0:
                assert abs(max_val - 1.0) < 1e-10
        
        elif method == 'unit_range':
            # Should be in [0, 1] range
            assert np.all(normalized >= 0.0)
            assert np.all(normalized <= 1.0)
            # If not constant vector, max should be 1 and min should be 0
            if np.max(vector) != np.min(vector):
                assert abs(np.max(normalized) - 1.0) < 1e-10
                assert abs(np.min(normalized) - 0.0) < 1e-10
    
    def test_zero_vector_handling_strategies(self):
        """Test different strategies for handling zero vectors in normalization"""
        zero_vector = np.zeros(5)
        
        # Test return_zero strategy (default)
        result = normalize_vector(zero_vector, method='l2', zero_vector_strategy='return_zero')
        assert np.allclose(result, np.zeros(5))
        
        # Test return_original strategy
        result = normalize_vector(zero_vector, method='l2', zero_vector_strategy='return_original')
        assert np.allclose(result, zero_vector)
        
        # Test random_unit strategy
        result = normalize_vector(zero_vector, method='l2', zero_vector_strategy='random_unit')
        assert abs(np.linalg.norm(result) - 1.0) < 1e-10  # Should be unit vector
        
        # Test raise_error strategy
        with pytest.raises(ValueError, match="Cannot L2 normalize zero vector"):
            normalize_vector(zero_vector, method='l2', zero_vector_strategy='raise_error')
        
        # Test warn_and_return strategy (should issue warning)
        with pytest.warns(UserWarning, match="Cannot L2 normalize zero vector"):
            result = normalize_vector(zero_vector, method='l2', zero_vector_strategy='warn_and_return')
            assert np.allclose(result, zero_vector)
        
        # Test invalid strategy
        with pytest.raises(ValueError, match="Unknown zero vector strategy"):
            normalize_vector(zero_vector, method='l2', zero_vector_strategy='invalid_strategy')
    
    @given(
        vector=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64]),
                     elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)),
        scale=st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_squashing_functions_properties(self, vector, scale):
        """**Feature: autonomous-logos-pathos-agent, Property 4: Mathematical operation consistency**
        **Validates: Requirements 2.2, 2.3**
        
        For any vector, squashing functions should produce bounded outputs.
        """
        tanh_result = tanh_squash(vector, scale)
        sigmoid_result = sigmoid_squash(vector, scale)
        
        # Tanh should be in [-1, 1]
        assert np.all(tanh_result >= -1.0)
        assert np.all(tanh_result <= 1.0)
        
        # Sigmoid should be in [0, 1]
        assert np.all(sigmoid_result >= 0.0)
        assert np.all(sigmoid_result <= 1.0)
        
        # Should preserve shape
        assert tanh_result.shape == vector.shape
        assert sigmoid_result.shape == vector.shape
    
    @given(
        current_state=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64]),
                           elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False)),
        previous_state=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64]),
                            elements=st.floats(min_value=-5.0, max_value=5.0, allow_nan=False, allow_infinity=False))
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_state_change_penalty_properties(self, current_state, previous_state):
        """**Feature: autonomous-logos-pathos-agent, Property 4: Mathematical operation consistency**
        **Validates: Requirements 2.2, 2.3**
        
        For any two states, change penalty should be non-negative and symmetric for identical states.
        """
        # Ensure same shape
        min_len = min(len(current_state), len(previous_state))
        current_trimmed = current_state[:min_len]
        previous_trimmed = previous_state[:min_len]
        
        penalty_l2 = compute_state_change_penalty(current_trimmed, previous_trimmed, 'l2')
        penalty_l1 = compute_state_change_penalty(current_trimmed, previous_trimmed, 'l1')
        
        # Non-negativity
        assert penalty_l2 >= 0
        assert penalty_l1 >= 0
        
        # Self-penalty should be 0
        self_penalty_l2 = compute_state_change_penalty(current_trimmed, current_trimmed, 'l2')
        self_penalty_l1 = compute_state_change_penalty(current_trimmed, current_trimmed, 'l1')
        
        assert abs(self_penalty_l2) < 1e-10
        assert abs(self_penalty_l1) < 1e-10
    
    @given(
        state=arrays(np.float64, shape=st.sampled_from([8, 16, 32, 64]),
                    elements=st.floats(min_value=-3.0, max_value=3.0, allow_nan=False, allow_infinity=False))
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
    def test_homeostatic_balance_properties(self, state):
        """**Feature: autonomous-logos-pathos-agent, Property 4: Mathematical operation consistency**
        **Validates: Requirements 2.2, 2.3**
        
        For any state, homeostatic balance computation should produce valid metrics.
        """
        target_ranges = {
            'stability': (0.1, 2.0),
            'complexity': (1.0, 5.0),
            'coherence': (0.5, 2.0),
            'balance': (0.1, 1.0)
        }
        weights = {
            'stability': 1.0,
            'complexity': 0.5,
            'coherence': 0.8,
            'balance': 0.6
        }
        
        balance_metrics = compute_homeostatic_balance(state, target_ranges, weights)
        
        # Should contain all expected metrics
        expected_keys = {'stability', 'complexity', 'coherence', 'balance', 'total_discomfort'}
        assert set(balance_metrics.keys()) == expected_keys
        
        # All metrics should be non-negative
        for key, value in balance_metrics.items():
            assert value >= 0, f"Metric {key} should be non-negative, got {value}"
        
        # Total discomfort should be sum of weighted individual discomforts
        expected_total = (
            weights['stability'] * balance_metrics['stability'] +
            weights['complexity'] * balance_metrics['complexity'] +
            weights['coherence'] * balance_metrics['coherence'] +
            weights['balance'] * balance_metrics['balance']
        )
        assert abs(balance_metrics['total_discomfort'] - expected_total) < 1e-10


if __name__ == "__main__":
    pytest.main([__file__])