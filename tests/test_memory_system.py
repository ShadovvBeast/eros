"""
Property-based tests for the Memory System.

This module tests the correctness properties of the Memory System including:
- Affective memory retrieval
- Novelty distance computation
- Associative memory coherence
"""

import pytest
import numpy as np
from hypothesis import given, strategies as st, assume
from typing import List

from src.memory.memory_system import ConcreteMemorySystem
from src.models import MemoryTrace, SemanticVector
from src.config import MemoryConfig


class TestMemorySystemProperties:
    """Property-based tests for Memory System correctness properties"""
    
    @given(
        state_dim=st.integers(min_value=32, max_value=128),
        semantic_dim=st.integers(min_value=64, max_value=256),
        num_traces=st.integers(min_value=1, max_value=20),
        k=st.integers(min_value=1, max_value=10)
    )
    def test_affective_memory_retrieval_property(self, state_dim, semantic_dim, num_traces, k):
        """
        **Feature: autonomous-logos-pathos-agent, Property 7: Affective Memory Retrieval**
        
        For any memory query, retrieval should prioritize affective similarity to current state 
        and return complete memory traces with all required components.
        
        **Validates: Requirements 3.3, 3.4**
        """
        assume(k <= num_traces)  # Can't retrieve more than we have
        
        # Create fresh memory system for this test
        memory_config = MemoryConfig(storage_backend='memory')
        memory_system = ConcreteMemorySystem(memory_config)
        
        # Create and store multiple memory traces
        traces = []
        for i in range(num_traces):
            affect_state = np.random.randn(state_dim)
            semantic_vector = SemanticVector(
                embedding=np.random.randn(semantic_dim),
                intention_text=f"Test intention {i}",
                semantic_category=f"category_{i % 3}"
            )
            
            trace = MemoryTrace(
                affect_state=affect_state,
                semantic_vector=semantic_vector,
                reward=np.random.uniform(-1, 1),
                timestamp=1000 + i,
                salience=np.random.uniform(0, 1),
                metadata={'trace_id': i}
            )
            
            memory_system.store_trace(trace)
            traces.append(trace)
        
        # Query with a random affective state
        query_state = np.random.randn(state_dim)
        retrieved_traces = memory_system.query_by_affect(query_state, k=k)
        
        # Property: Should return exactly k traces (or fewer if less available)
        assert len(retrieved_traces) == min(k, num_traces)
        
        # Property: All returned traces should be complete MemoryTrace objects
        for trace in retrieved_traces:
            assert isinstance(trace, MemoryTrace)
            assert trace.affect_state is not None
            assert trace.semantic_vector is not None
            assert trace.reward is not None
            assert trace.timestamp is not None
            assert trace.salience is not None
            assert hasattr(trace, 'metadata')
        
        # Property: All returned traces should be from the stored traces
        retrieved_ids = {trace.metadata.get('trace_id') for trace in retrieved_traces}
        stored_ids = {trace.metadata.get('trace_id') for trace in traces}
        assert retrieved_ids.issubset(stored_ids)
        
        # Property: If k < num_traces, returned traces should be the most similar
        if k < num_traces and len(retrieved_traces) == k:
            # Compute distances for all traces
            distances = []
            for trace in traces:
                distance = np.linalg.norm(query_state - trace.affect_state)
                distances.append((distance, trace.metadata.get('trace_id')))
            
            # Sort by distance and get k nearest
            distances.sort(key=lambda x: x[0])
            expected_ids = {trace_id for _, trace_id in distances[:k]}
            
            assert retrieved_ids == expected_ids
    
    @given(
        state_dim=st.integers(min_value=32, max_value=128),
        semantic_dim=st.integers(min_value=64, max_value=256),
        num_existing=st.integers(min_value=0, max_value=15)
    )
    def test_novelty_distance_computation_property(self, state_dim, semantic_dim, num_existing):
        """
        **Feature: autonomous-logos-pathos-agent, Property 8: Novelty Distance Computation**
        
        For any novelty estimation, the computation should be based on distance to nearest 
        existing entries in both affective and semantic spaces.
        
        **Validates: Requirements 3.5**
        """
        # Create fresh memory system for this test
        memory_config = MemoryConfig(storage_backend='memory')
        memory_system = ConcreteMemorySystem(memory_config)
        
        # Store existing traces
        existing_traces = []
        for i in range(num_existing):
            affect_state = np.random.randn(state_dim)
            semantic_vector = SemanticVector(
                embedding=np.random.randn(semantic_dim),
                intention_text=f"Existing intention {i}",
                semantic_category=f"existing_{i % 2}"
            )
            
            trace = MemoryTrace(
                affect_state=affect_state,
                semantic_vector=semantic_vector,
                reward=np.random.uniform(-1, 1),
                timestamp=1000 + i,
                salience=np.random.uniform(0, 1)
            )
            
            memory_system.store_trace(trace)
            existing_traces.append(trace)
        
        # Create new state and semantic vector for novelty estimation
        new_affect_state = np.random.randn(state_dim)
        new_semantic_vector = SemanticVector(
            embedding=np.random.randn(semantic_dim),
            intention_text="New test intention",
            semantic_category="new_category"
        )
        
        affect_novelty, semantic_novelty = memory_system.estimate_novelty(
            new_affect_state, new_semantic_vector
        )
        
        # Property: Novelty values should be non-negative
        assert affect_novelty >= 0.0
        assert semantic_novelty >= 0.0
        
        if num_existing == 0:
            # Property: If no existing memories, novelty should be maximum (1.0)
            assert affect_novelty == 1.0
            assert semantic_novelty == 1.0
        else:
            # Property: Novelty should correspond to minimum distance to existing entries
            
            # Compute expected affect novelty (minimum distance to existing affect states)
            affect_distances = []
            for trace in existing_traces:
                distance = np.linalg.norm(new_affect_state - trace.affect_state)
                affect_distances.append(distance)
            
            expected_affect_novelty = min(affect_distances)
            
            # Compute expected semantic novelty (minimum cosine distance)
            semantic_distances = []
            for trace in existing_traces:
                # Cosine distance = 1 - cosine_similarity
                dot_product = np.dot(new_semantic_vector.embedding, trace.semantic_vector.embedding)
                norm_new = np.linalg.norm(new_semantic_vector.embedding)
                norm_existing = np.linalg.norm(trace.semantic_vector.embedding)
                
                if norm_new > 0 and norm_existing > 0:
                    cosine_sim = dot_product / (norm_new * norm_existing)
                    cosine_distance = 1.0 - cosine_sim
                else:
                    cosine_distance = 1.0
                
                semantic_distances.append(cosine_distance)
            
            expected_semantic_novelty = min(semantic_distances)
            
            # Property: Computed novelty should match expected minimum distances
            assert abs(affect_novelty - expected_affect_novelty) < 1e-10
            assert abs(semantic_novelty - expected_semantic_novelty) < 1e-10
    
    @given(
        state_dim=st.integers(min_value=32, max_value=128),
        semantic_dim=st.integers(min_value=64, max_value=256),
        num_traces=st.integers(min_value=2, max_value=10)
    )
    def test_memory_storage_completeness_property(self, state_dim, semantic_dim, num_traces):
        """
        Property: All stored memory traces should be retrievable and maintain data integrity.
        """
        # Create fresh memory system for this test
        memory_config = MemoryConfig(storage_backend='memory')
        memory_system = ConcreteMemorySystem(memory_config)
        
        # Store traces with known data
        stored_traces = []
        for i in range(num_traces):
            affect_state = np.random.randn(state_dim)
            semantic_vector = SemanticVector(
                embedding=np.random.randn(semantic_dim),
                intention_text=f"Stored intention {i}",
                semantic_category=f"stored_{i}"
            )
            
            trace = MemoryTrace(
                affect_state=affect_state,
                semantic_vector=semantic_vector,
                reward=0.5 + i * 0.1,  # Distinct rewards
                timestamp=2000 + i,    # Distinct timestamps
                salience=0.8,
                metadata={'stored_id': i}
            )
            
            memory_system.store_trace(trace)
            stored_traces.append(trace)
        
        # Property: Total count should match stored traces
        assert memory_system.get_trace_count() == num_traces
        
        # Property: Recent traces should include all stored traces (since we stored few)
        recent_traces = memory_system.get_recent_traces(n=num_traces)
        assert len(recent_traces) == num_traces
        
        # Property: All stored data should be preserved
        recent_ids = {trace.metadata.get('stored_id') for trace in recent_traces}
        expected_ids = {i for i in range(num_traces)}
        assert recent_ids == expected_ids
        
        # Property: Data integrity should be maintained
        for trace in recent_traces:
            stored_id = trace.metadata.get('stored_id')
            original_trace = stored_traces[stored_id]
            
            # Check that affect states match
            np.testing.assert_array_equal(trace.affect_state, original_trace.affect_state)
            
            # Check that semantic vectors match
            np.testing.assert_array_equal(
                trace.semantic_vector.embedding, 
                original_trace.semantic_vector.embedding
            )
            assert trace.semantic_vector.intention_text == original_trace.semantic_vector.intention_text
            assert trace.semantic_vector.semantic_category == original_trace.semantic_vector.semantic_category
            
            # Check other fields
            assert trace.reward == original_trace.reward
            assert trace.timestamp == original_trace.timestamp
            assert trace.salience == original_trace.salience
    
    def test_memory_echo_integration_basic(self):
        """
        Basic test for memory echo functionality used by Pathos layer.
        """
        # Create fresh memory system for this test
        memory_config = MemoryConfig(storage_backend='memory')
        memory_system = ConcreteMemorySystem(memory_config)
        
        state_dim = 64
        
        # Store a few traces
        for i in range(3):
            affect_state = np.random.randn(state_dim)
            semantic_vector = SemanticVector(
                embedding=np.random.randn(128),
                intention_text=f"Echo test {i}",
                semantic_category="echo_test"
            )
            
            trace = MemoryTrace(
                affect_state=affect_state,
                semantic_vector=semantic_vector,
                reward=0.5,
                timestamp=3000 + i,
                salience=0.7
            )
            
            memory_system.store_trace(trace)
        
        # Test echo computation
        query_state = np.random.randn(state_dim)
        echo_vector, weights = memory_system.get_memory_echoes(query_state, k=2)
        
        # Should return echo vector of same dimension
        assert echo_vector.shape == query_state.shape
        
        # Should return weights for retrieved memories
        assert len(weights) <= 2  # At most k weights
        
        # Weights should be non-negative and sum to approximately 1
        if weights:
            assert all(w >= 0 for w in weights)
            assert abs(sum(weights) - 1.0) < 1e-10
    
    def test_echo_integration_methods(self):
        """
        Test the enhanced echo integration methods for Pathos layer.
        """
        # Create fresh memory system
        memory_config = MemoryConfig(storage_backend='memory')
        memory_system = ConcreteMemorySystem(memory_config)
        
        state_dim = 64
        
        # Store several traces with known affect states
        stored_traces = []
        for i in range(5):
            # Create affect states with some structure
            affect_state = np.random.randn(state_dim)
            affect_state[0] = i * 0.5  # Make first dimension predictable
            
            semantic_vector = SemanticVector(
                embedding=np.random.randn(128),
                intention_text=f"Echo integration test {i}",
                semantic_category="echo_test"
            )
            
            trace = MemoryTrace(
                affect_state=affect_state,
                semantic_vector=semantic_vector,
                reward=0.1 * i,
                timestamp=4000 + i,
                salience=0.6 + 0.1 * i,
                metadata={'echo_test_id': i}
            )
            
            memory_system.store_trace(trace)
            stored_traces.append(trace)
        
        # Test get_echo_traces_for_pathos
        query_state = np.random.randn(state_dim)
        echo_traces = memory_system.get_echo_traces_for_pathos(query_state, k=3)
        
        assert len(echo_traces) == 3
        assert all(isinstance(trace, MemoryTrace) for trace in echo_traces)
        
        # Test compute_echo_strength_weights
        echo_influence, weights = memory_system.compute_echo_strength_weights(
            query_state, echo_traces, echo_strength=0.2
        )
        
        assert echo_influence.shape == query_state.shape
        assert len(weights) == len(echo_traces)
        assert all(w >= 0 for w in weights)
        assert abs(sum(weights) - 1.0) < 1e-10
        
        # Test that echo strength parameter affects the result
        echo_influence_weak, _ = memory_system.compute_echo_strength_weights(
            query_state, echo_traces, echo_strength=0.1
        )
        echo_influence_strong, _ = memory_system.compute_echo_strength_weights(
            query_state, echo_traces, echo_strength=0.3
        )
        
        # Stronger echo should have larger magnitude
        weak_norm = np.linalg.norm(echo_influence_weak)
        strong_norm = np.linalg.norm(echo_influence_strong)
        assert strong_norm > weak_norm
    
    @given(
        state_dim=st.integers(min_value=32, max_value=128),
        semantic_dim=st.integers(min_value=64, max_value=256),
        num_related_traces=st.integers(min_value=3, max_value=8),
        similarity_factor=st.floats(min_value=0.1, max_value=0.9)
    )
    def test_associative_memory_coherence_property(self, state_dim, semantic_dim, num_related_traces, similarity_factor):
        """
        **Feature: autonomous-logos-pathos-agent, Property 13: Associative Memory Coherence**
        
        For any memory recall operation, the system should maintain associative relationships 
        between related experiences and preserve core attractor patterns.
        
        **Validates: Requirements 6.3, 6.4**
        """
        # Create fresh memory system
        memory_config = MemoryConfig(storage_backend='memory')
        memory_system = ConcreteMemorySystem(memory_config)
        
        # Create a base pattern for related memories
        base_affect_pattern = np.random.randn(state_dim)
        base_semantic_pattern = np.random.randn(semantic_dim)
        
        # Store related traces that share similar patterns (attractor)
        related_traces = []
        for i in range(num_related_traces):
            # Create variations of the base pattern
            affect_variation = base_affect_pattern + similarity_factor * np.random.randn(state_dim)
            semantic_variation = base_semantic_pattern + similarity_factor * np.random.randn(semantic_dim)
            
            semantic_vector = SemanticVector(
                embedding=semantic_variation,
                intention_text=f"Related pattern {i}",
                semantic_category="coherent_group"
            )
            
            trace = MemoryTrace(
                affect_state=affect_variation,
                semantic_vector=semantic_vector,
                reward=0.5 + 0.1 * i,  # Similar rewards for related experiences
                timestamp=5000 + i * 10,
                salience=0.7 + 0.05 * i,
                metadata={'group': 'coherent', 'pattern_id': i}
            )
            
            memory_system.store_trace(trace)
            related_traces.append(trace)
        
        # Store some unrelated traces as noise
        for i in range(3):
            unrelated_affect = np.random.randn(state_dim) * 2.0  # Different scale
            unrelated_semantic = np.random.randn(semantic_dim) * 2.0
            
            semantic_vector = SemanticVector(
                embedding=unrelated_semantic,
                intention_text=f"Unrelated pattern {i}",
                semantic_category="noise_group"
            )
            
            trace = MemoryTrace(
                affect_state=unrelated_affect,
                semantic_vector=semantic_vector,
                reward=-0.2 + 0.1 * i,  # Different reward range
                timestamp=6000 + i * 10,
                salience=0.3 + 0.1 * i,
                metadata={'group': 'noise', 'pattern_id': i}
            )
            
            memory_system.store_trace(trace)
        
        # Property 1: Querying with a pattern similar to the base should retrieve related memories
        query_affect = base_affect_pattern + 0.5 * similarity_factor * np.random.randn(state_dim)
        retrieved_by_affect = memory_system.query_by_affect(query_affect, k=num_related_traces)
        
        # Most retrieved memories should be from the coherent group
        coherent_retrieved = sum(1 for trace in retrieved_by_affect 
                               if trace.metadata.get('group') == 'coherent')
        
        # At least half should be from the coherent group (associative coherence)
        assert coherent_retrieved >= num_related_traces // 2
        
        # Property 2: Semantic similarity should also maintain coherence
        query_semantic = SemanticVector(
            embedding=base_semantic_pattern + 0.5 * similarity_factor * np.random.randn(semantic_dim),
            intention_text="Query for coherent group",
            semantic_category="test_query"
        )
        
        retrieved_by_semantic = memory_system.query_by_semantic(query_semantic, k=num_related_traces)
        coherent_semantic = sum(1 for trace in retrieved_by_semantic 
                              if trace.metadata.get('group') == 'coherent')
        
        # Semantic retrieval should also favor coherent group
        assert coherent_semantic >= num_related_traces // 2
        
        # Property 3: Core attractor patterns should be preserved in echoes
        echo_traces = memory_system.get_echo_traces_for_pathos(query_affect, k=num_related_traces)
        echo_influence, weights = memory_system.compute_echo_strength_weights(
            query_affect, echo_traces, echo_strength=0.1
        )
        
        # Echo influence should be non-zero (pattern preservation)
        assert np.linalg.norm(echo_influence) > 0.0
        
        # Property 4: Related memories should have higher similarity weights
        if len(echo_traces) >= 2:
            coherent_weights = []
            noise_weights = []
            
            for i, trace in enumerate(echo_traces):
                if trace.metadata.get('group') == 'coherent':
                    coherent_weights.append(weights[i])
                elif trace.metadata.get('group') == 'noise':
                    noise_weights.append(weights[i])
            
            # If we have both types, coherent memories should generally have higher weights
            if coherent_weights and noise_weights:
                avg_coherent_weight = np.mean(coherent_weights)
                avg_noise_weight = np.mean(noise_weights)
                
                # This is a statistical tendency, not absolute
                # Allow some tolerance for randomness
                assert avg_coherent_weight >= avg_noise_weight * 0.5
        
        # Property 5: Novelty estimation should reflect pattern familiarity
        # A pattern similar to stored ones should have lower novelty
        similar_affect = base_affect_pattern + 0.2 * similarity_factor * np.random.randn(state_dim)
        similar_semantic = SemanticVector(
            embedding=base_semantic_pattern + 0.2 * similarity_factor * np.random.randn(semantic_dim),
            intention_text="Similar to stored patterns",
            semantic_category="coherent_group"
        )
        
        # A very different pattern should have higher novelty
        different_affect = np.random.randn(state_dim) * 3.0
        different_semantic = SemanticVector(
            embedding=np.random.randn(semantic_dim) * 3.0,
            intention_text="Very different pattern",
            semantic_category="novel_group"
        )
        
        similar_novelty_affect, similar_novelty_semantic = memory_system.estimate_novelty(
            similar_affect, similar_semantic
        )
        different_novelty_affect, different_novelty_semantic = memory_system.estimate_novelty(
            different_affect, different_semantic
        )
        
        # Different patterns should generally be more novel
        # (This is statistical, so we allow some tolerance)
        assert different_novelty_affect >= similar_novelty_affect * 0.8
        assert different_novelty_semantic >= similar_novelty_semantic * 0.8