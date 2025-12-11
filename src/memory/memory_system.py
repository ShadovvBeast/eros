"""
Concrete implementation of the Memory System for associative storage.

This module implements the MemorySystem interface with support for:
- Memory trace storage with timestamp indexing
- Affective similarity-based retrieval methods
- Novelty estimation using distance to nearest neighbors
"""

import json
import os
import sqlite3
from typing import List, Tuple, Dict, Any
import numpy as np
from datetime import datetime
import logging

from .interfaces import MemorySystem
from ..models import MemoryTrace, SemanticVector
from ..config import MemoryConfig
from ..math_utils import (
    cosine_similarity, euclidean_distance, find_k_nearest_neighbors,
    compute_similarity_weights
)

logger = logging.getLogger(__name__)


class ConcreteMemorySystem(MemorySystem):
    """
    Concrete implementation of the Memory System with multiple storage backends.
    
    Supports JSON file storage, SQLite database, and in-memory storage.
    Provides affective similarity-based retrieval and novelty estimation.
    """
    
    def __init__(self, config: MemoryConfig):
        super().__init__(config)
        self.traces: List[MemoryTrace] = []
        self._initialize_storage()
        
    def _initialize_storage(self) -> None:
        """Initialize the storage backend based on configuration."""
        if self.config.storage_backend == 'json':
            self._initialize_json_storage()
        elif self.config.storage_backend == 'sqlite':
            self._initialize_sqlite_storage()
        elif self.config.storage_backend == 'memory':
            # In-memory storage is already initialized
            logger.info("Using in-memory storage for memory traces")
        else:
            raise ValueError(f"Unknown storage backend: {self.config.storage_backend}")
    
    def _initialize_json_storage(self) -> None:
        """Initialize JSON file storage."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.storage_path), exist_ok=True)
        
        # Load existing traces if file exists
        if os.path.exists(self.config.storage_path):
            try:
                with open(self.config.storage_path, 'r') as f:
                    data = json.load(f)
                    self.traces = [MemoryTrace.from_dict(trace_data) for trace_data in data]
                logger.info(f"Loaded {len(self.traces)} memory traces from {self.config.storage_path}")
            except Exception as e:
                logger.error(f"Failed to load memory traces: {e}")
                self.traces = []
        else:
            logger.info(f"No existing memory file found at {self.config.storage_path}")
    
    def _initialize_sqlite_storage(self) -> None:
        """Initialize SQLite database storage."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.storage_path), exist_ok=True)
        
        # Initialize database
        db_path = self.config.storage_path.replace('.json', '.db')
        self.db_connection = sqlite3.connect(db_path, check_same_thread=False)
        
        # Create table if it doesn't exist
        self.db_connection.execute('''
            CREATE TABLE IF NOT EXISTS memory_traces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                affect_state TEXT NOT NULL,
                semantic_vector TEXT NOT NULL,
                reward REAL NOT NULL,
                timestamp INTEGER NOT NULL,
                salience REAL NOT NULL,
                metadata TEXT
            )
        ''')
        
        # Create index on timestamp for efficient retrieval
        self.db_connection.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON memory_traces(timestamp)
        ''')
        
        self.db_connection.commit()
        
        # Load existing traces
        cursor = self.db_connection.execute('SELECT * FROM memory_traces ORDER BY timestamp')
        for row in cursor:
            trace_data = {
                'affect_state': json.loads(row[1]),
                'semantic_vector': json.loads(row[2]),
                'reward': row[3],
                'timestamp': row[4],
                'salience': row[5],
                'metadata': json.loads(row[6]) if row[6] else {}
            }
            self.traces.append(MemoryTrace.from_dict(trace_data))
        
        logger.info(f"Loaded {len(self.traces)} memory traces from SQLite database")
    
    def _save_to_storage(self) -> None:
        """Save current traces to persistent storage."""
        if self.config.storage_backend == 'json':
            self._save_to_json()
        elif self.config.storage_backend == 'sqlite':
            # SQLite saves immediately on each store_trace call
            pass
        # In-memory storage doesn't need persistence
    
    def _save_to_json(self) -> None:
        """Save traces to JSON file."""
        try:
            trace_dicts = [trace.to_dict() for trace in self.traces]
            with open(self.config.storage_path, 'w') as f:
                json.dump(trace_dicts, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save memory traces: {e}")
    
    def store_trace(self, trace: MemoryTrace) -> None:
        """
        Store a memory trace in the associative storage system.
        
        Args:
            trace: MemoryTrace to store
        """
        # Add to in-memory list
        self.traces.append(trace)
        
        # Enforce memory limit
        if len(self.traces) > self.config.max_memory_traces:
            # Remove oldest traces (FIFO)
            self.traces = self.traces[-self.config.max_memory_traces:]
        
        # Save to persistent storage
        if self.config.storage_backend == 'sqlite':
            self._store_trace_sqlite(trace)
        elif self.config.storage_backend == 'json':
            self._save_to_json()
        
        logger.debug(f"Stored memory trace with salience {trace.salience:.3f} at timestamp {trace.timestamp}")
    
    def _store_trace_sqlite(self, trace: MemoryTrace) -> None:
        """Store a single trace to SQLite database."""
        self.db_connection.execute('''
            INSERT INTO memory_traces (affect_state, semantic_vector, reward, timestamp, salience, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            json.dumps(trace.affect_state.tolist()),
            json.dumps(trace.semantic_vector.to_dict()),
            trace.reward,
            trace.timestamp,
            trace.salience,
            json.dumps(trace.metadata)
        ))
        self.db_connection.commit()
    
    def query_by_affect(self, affect_state: np.ndarray, k: int = 5) -> List[MemoryTrace]:
        """
        Retrieve memories primarily by affective similarity to current state.
        
        Args:
            affect_state: Current affective state for similarity matching
            k: Number of memories to retrieve
            
        Returns:
            List of most similar memory traces
        """
        if not self.traces:
            return []
        
        # Extract affect vectors from all traces
        affect_vectors = [trace.affect_state for trace in self.traces]
        
        # Find k nearest neighbors by affective similarity
        indices, distances = find_k_nearest_neighbors(
            affect_state, affect_vectors, k, distance_metric='euclidean'
        )
        
        # Return corresponding memory traces
        similar_traces = [self.traces[i] for i in indices]
        
        logger.debug(f"Retrieved {len(similar_traces)} memory traces by affective similarity")
        return similar_traces
    
    def query_by_semantic(self, semantic_vector: SemanticVector, k: int = 5) -> List[MemoryTrace]:
        """
        Retrieve memories by semantic similarity.
        
        Args:
            semantic_vector: Semantic vector for similarity matching
            k: Number of memories to retrieve
            
        Returns:
            List of most similar memory traces
        """
        if not self.traces:
            return []
        
        # Extract semantic vectors from all traces
        semantic_vectors = [trace.semantic_vector.embedding for trace in self.traces]
        
        # Find k nearest neighbors by semantic similarity
        indices, distances = find_k_nearest_neighbors(
            semantic_vector.embedding, semantic_vectors, k, distance_metric='cosine'
        )
        
        # Return corresponding memory traces
        similar_traces = [self.traces[i] for i in indices]
        
        logger.debug(f"Retrieved {len(similar_traces)} memory traces by semantic similarity")
        return similar_traces
    
    def estimate_novelty(self, affect_state: np.ndarray, semantic_vector: SemanticVector) -> Tuple[float, float]:
        """
        Estimate novelty by computing distance to nearest existing entries.
        
        Args:
            affect_state: Affective state to evaluate
            semantic_vector: Semantic vector to evaluate
            
        Returns:
            Tuple of (affect_novelty, semantic_novelty)
        """
        if not self.traces:
            # If no memories exist, everything is novel
            return 1.0, 1.0
        
        # Compute affective novelty
        affect_vectors = [trace.affect_state for trace in self.traces]
        affect_indices, affect_distances = find_k_nearest_neighbors(
            affect_state, affect_vectors, self.config.novelty_k_neighbors, 
            distance_metric='euclidean'
        )
        
        # Use minimum distance as novelty measure (higher distance = more novel)
        affect_novelty = min(affect_distances) if affect_distances else 1.0
        
        # Compute semantic novelty
        semantic_vectors = [trace.semantic_vector.embedding for trace in self.traces]
        semantic_indices, semantic_distances = find_k_nearest_neighbors(
            semantic_vector.embedding, semantic_vectors, self.config.novelty_k_neighbors,
            distance_metric='cosine'
        )
        
        # Use minimum distance as novelty measure
        semantic_novelty = min(semantic_distances) if semantic_distances else 1.0
        
        logger.debug(f"Estimated novelty - affect: {affect_novelty:.3f}, semantic: {semantic_novelty:.3f}")
        return affect_novelty, semantic_novelty
    
    def get_recent_traces(self, n: int = 10) -> List[MemoryTrace]:
        """
        Get the most recent memory traces.
        
        Args:
            n: Number of recent traces to retrieve
            
        Returns:
            List of recent memory traces
        """
        # Sort by timestamp (most recent first) and take n
        sorted_traces = sorted(self.traces, key=lambda t: t.timestamp, reverse=True)
        return sorted_traces[:n]
    
    def get_trace_count(self) -> int:
        """
        Get the total number of stored memory traces.
        
        Returns:
            Number of stored traces
        """
        return len(self.traces)
    
    def clear_memories(self) -> None:
        """Clear all stored memories (for testing/reset)"""
        self.traces.clear()
        
        if self.config.storage_backend == 'sqlite':
            self.db_connection.execute('DELETE FROM memory_traces')
            self.db_connection.commit()
        elif self.config.storage_backend == 'json':
            if os.path.exists(self.config.storage_path):
                os.remove(self.config.storage_path)
        
        logger.info("Cleared all memory traces")
    
    def get_memory_echoes(self, affect_state: np.ndarray, k: int = 5) -> Tuple[np.ndarray, List[float]]:
        """
        Get memory echoes for Pathos layer integration.
        
        Args:
            affect_state: Current affective state
            k: Number of similar memories to retrieve
            
        Returns:
            Tuple of (weighted_echo_vector, similarity_weights)
        """
        similar_traces = self.query_by_affect(affect_state, k)
        
        if not similar_traces:
            # Return zero echo if no memories
            return np.zeros_like(affect_state), []
        
        # Compute distances for similarity weighting
        distances = []
        echo_vectors = []
        
        for trace in similar_traces:
            distance = euclidean_distance(affect_state, trace.affect_state)
            distances.append(distance)
            echo_vectors.append(trace.affect_state)
        
        # Convert distances to similarity weights
        weights = compute_similarity_weights(distances, temperature=1.0, method='exponential')
        
        # Compute weighted sum of echo vectors
        weighted_echo = np.zeros_like(affect_state)
        for i, echo_vector in enumerate(echo_vectors):
            weighted_echo += weights[i] * echo_vector
        
        return weighted_echo, weights.tolist()
    
    def get_echo_traces_for_pathos(self, affect_state: np.ndarray, k: int = 5) -> List[MemoryTrace]:
        """
        Get memory traces for echo computation in Pathos layer.
        
        This method is specifically designed for Pathos layer integration,
        returning the actual MemoryTrace objects that Pathos can use for
        its own similarity weight computation.
        
        Args:
            affect_state: Current affective state
            k: Number of similar memories to retrieve
            
        Returns:
            List of MemoryTrace objects for echo computation
        """
        return self.query_by_affect(affect_state, k)
    
    def compute_echo_strength_weights(self, affect_state: np.ndarray, 
                                    memory_traces: List[MemoryTrace],
                                    echo_strength: float = 0.1) -> Tuple[np.ndarray, List[float]]:
        """
        Compute echo influence with configurable echo strength parameter.
        
        This method implements the β·Σ(w_i·F_i) computation from the Pathos
        mathematical model, allowing configurable echo strength.
        
        Args:
            affect_state: Current affective state
            memory_traces: Retrieved memory traces
            echo_strength: Echo strength parameter β
            
        Returns:
            Tuple of (echo_influence_vector, similarity_weights)
        """
        if not memory_traces:
            return np.zeros_like(affect_state), []
        
        # Compute similarity weights based on cosine similarity
        similarities = []
        for trace in memory_traces:
            similarity = cosine_similarity(affect_state, trace.affect_state)
            # Only use positive similarities for echo influence
            similarities.append(max(0.0, similarity))
        
        # Convert similarities to normalized weights
        weights = compute_similarity_weights(similarities, temperature=1.0, method='exponential')
        
        # Compute weighted echo influence: β·Σ(w_i·F_i)
        echo_influence = np.zeros_like(affect_state)
        for i, trace in enumerate(memory_traces):
            echo_influence += weights[i] * trace.affect_state
        
        # Apply echo strength parameter
        echo_influence *= echo_strength
        
        return echo_influence, weights.tolist()
    
    def __del__(self):
        """Cleanup resources on destruction."""
        if hasattr(self, 'db_connection'):
            self.db_connection.close()