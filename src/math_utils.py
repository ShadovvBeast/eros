"""
Mathematical utility functions for the Autonomous Logos-Pathos-Memory Agent.

This module provides vector operations, nonlinear functions, and homeostatic
balance computations required by the Pathos layer dynamics.

## Zero Vector Handling Best Practices

When normalizing vectors, zero vectors (vectors with norm = 0) require special handling
since division by zero is undefined. This module provides several strategies:

1. **return_zero** (default): Return a zero vector of the same shape
   - Best for: Most mathematical contexts where zero input should yield zero output
   - Use case: Semantic embeddings, state directions

2. **return_original**: Return the original zero vector unchanged
   - Best for: When you want to preserve the original input
   - Use case: Legacy compatibility, debugging

3. **random_unit**: Return a random unit vector
   - Best for: When you need a valid direction but input is zero
   - Use case: Initialization, avoiding singularities in optimization

4. **raise_error**: Raise a ValueError
   - Best for: When zero vectors indicate a programming error
   - Use case: Strict mathematical contexts, debugging

5. **warn_and_return**: Issue warning and return original (legacy)
   - Best for: Backward compatibility during migration
   - Use case: Gradual migration from warning-based to explicit handling

Choose the strategy based on your specific use case and mathematical requirements.
"""

import numpy as np
from typing import Dict, Tuple, List
import warnings
import logging

logger = logging.getLogger(__name__)


def _handle_zero_vector(vector: np.ndarray, strategy: str, norm_type: str) -> np.ndarray:
    """
    Handle zero vector cases in normalization.
    
    Args:
        vector: The zero vector
        strategy: Strategy for handling zero vectors
        norm_type: Type of normalization (for logging/error messages)
        
    Returns:
        Handled vector according to strategy
        
    Raises:
        ValueError: If strategy is 'raise_error'
    """
    if strategy == 'return_zero':
        return np.zeros_like(vector)
    elif strategy == 'return_original':
        return vector.copy()
    elif strategy == 'random_unit':
        # Generate random unit vector
        random_vec = np.random.randn(len(vector))
        return random_vec / np.linalg.norm(random_vec)
    elif strategy == 'raise_error':
        raise ValueError(f"Cannot {norm_type} normalize zero vector")
    elif strategy == 'warn_and_return':
        warnings.warn(f"Cannot {norm_type} normalize zero vector, returning original")
        return vector.copy()
    else:
        raise ValueError(f"Unknown zero vector strategy: {strategy}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity value between -1 and 1
        
    Raises:
        ValueError: If vectors have different dimensions or are zero vectors
    """
    if a.shape != b.shape:
        raise ValueError(f"Vector dimensions must match: {a.shape} vs {b.shape}")
    
    # Handle zero vectors
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a, b) / (norm_a * norm_b)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Euclidean distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Euclidean distance (non-negative)
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if a.shape != b.shape:
        raise ValueError(f"Vector dimensions must match: {a.shape} vs {b.shape}")
    
    return np.linalg.norm(a - b)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute Manhattan (L1) distance between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Manhattan distance (non-negative)
        
    Raises:
        ValueError: If vectors have different dimensions
    """
    if a.shape != b.shape:
        raise ValueError(f"Vector dimensions must match: {a.shape} vs {b.shape}")
    
    return np.sum(np.abs(a - b))


def normalize_vector(vector: np.ndarray, method: str = 'l2', 
                   zero_vector_strategy: str = 'return_zero') -> np.ndarray:
    """
    Normalize a vector using specified method.
    
    Args:
        vector: Input vector to normalize
        method: Normalization method ('l2', 'l1', 'max', 'unit_range')
        zero_vector_strategy: How to handle zero vectors:
            - 'return_zero': Return zero vector (default)
            - 'return_original': Return original zero vector
            - 'random_unit': Return random unit vector
            - 'raise_error': Raise ValueError
            - 'warn_and_return': Warn and return original (legacy behavior)
        
    Returns:
        Normalized vector
        
    Raises:
        ValueError: If method is unknown or zero_vector_strategy is 'raise_error'
    """
    if method == 'l2':
        norm = np.linalg.norm(vector)
        if norm == 0:
            return _handle_zero_vector(vector, zero_vector_strategy, 'L2')
        return vector / norm
    
    elif method == 'l1':
        norm = np.sum(np.abs(vector))
        if norm == 0:
            return _handle_zero_vector(vector, zero_vector_strategy, 'L1')
        return vector / norm
    
    elif method == 'max':
        max_val = np.max(np.abs(vector))
        if max_val == 0:
            return vector.copy()
        return vector / max_val
    
    elif method == 'unit_range':
        min_val = np.min(vector)
        max_val = np.max(vector)
        if max_val == min_val:
            return np.zeros_like(vector)
        return (vector - min_val) / (max_val - min_val)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def tanh_squash(x: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Apply tanh squashing function to vector elements.
    
    Args:
        x: Input vector
        scale: Scaling factor applied before tanh
        
    Returns:
        Squashed vector with elements in [-1, 1]
    """
    return np.tanh(scale * x)


def sigmoid_squash(x: np.ndarray, scale: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """
    Apply sigmoid squashing function to vector elements.
    
    Args:
        x: Input vector
        scale: Scaling factor applied before sigmoid
        offset: Offset applied before sigmoid
        
    Returns:
        Squashed vector with elements in [0, 1]
    """
    # Clip to prevent overflow
    clipped = np.clip(scale * (x - offset), -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))


def soft_clamp(x: np.ndarray, min_val: float = -5.0, max_val: float = 5.0, 
               softness: float = 1.0) -> np.ndarray:
    """
    Apply soft clamping using tanh to keep values within bounds.
    
    Args:
        x: Input vector
        min_val: Minimum bound
        max_val: Maximum bound
        softness: Controls how soft the clamping is (higher = softer)
        
    Returns:
        Soft-clamped vector
    """
    center = (max_val + min_val) / 2
    range_half = (max_val - min_val) / 2
    
    normalized = (x - center) / range_half
    clamped = np.tanh(normalized / softness) * softness
    
    return center + clamped * range_half


def compute_homeostatic_balance(state: np.ndarray, 
                              target_ranges: Dict[str, Tuple[float, float]],
                              weights: Dict[str, float]) -> Dict[str, float]:
    """
    Compute homeostatic balance metrics for different aspects of the state.
    
    Args:
        state: Current affective state vector
        target_ranges: Dictionary mapping metric names to (min, max) target ranges
        weights: Dictionary mapping metric names to importance weights
        
    Returns:
        Dictionary of balance metrics and overall discomfort
    """
    balance_metrics = {}
    
    # Stability: variance of state vector (lower is more stable)
    if 'stability' in target_ranges:
        variance = np.var(state)
        target_min, target_max = target_ranges['stability']
        if variance < target_min:
            stability_discomfort = (target_min - variance) ** 2
        elif variance > target_max:
            stability_discomfort = (variance - target_max) ** 2
        else:
            stability_discomfort = 0.0
        balance_metrics['stability'] = stability_discomfort
    
    # Complexity: entropy-like measure (diversity of activation)
    if 'complexity' in target_ranges:
        # Normalize state to probabilities for entropy calculation
        normalized_state = np.abs(state)
        if np.sum(normalized_state) > 0:
            normalized_state = normalized_state / np.sum(normalized_state)
            # Add small epsilon to avoid log(0)
            epsilon = 1e-10
            entropy = -np.sum(normalized_state * np.log(normalized_state + epsilon))
        else:
            entropy = 0.0
        
        target_min, target_max = target_ranges['complexity']
        if entropy < target_min:
            complexity_discomfort = (target_min - entropy) ** 2
        elif entropy > target_max:
            complexity_discomfort = (entropy - target_max) ** 2
        else:
            complexity_discomfort = 0.0
        balance_metrics['complexity'] = complexity_discomfort
    
    # Coherence: how well-structured the state is (inverse of randomness)
    if 'coherence' in target_ranges:
        # Measure coherence as correlation with a structured pattern
        # For simplicity, use the norm of the state as a coherence measure
        coherence = np.linalg.norm(state) / len(state)
        
        target_min, target_max = target_ranges['coherence']
        if coherence < target_min:
            coherence_discomfort = (target_min - coherence) ** 2
        elif coherence > target_max:
            coherence_discomfort = (coherence - target_max) ** 2
        else:
            coherence_discomfort = 0.0
        balance_metrics['coherence'] = coherence_discomfort
    
    # Balance: how centered the state is around zero
    if 'balance' in target_ranges:
        mean_activation = np.mean(np.abs(state))
        target_min, target_max = target_ranges['balance']
        if mean_activation < target_min:
            balance_discomfort = (target_min - mean_activation) ** 2
        elif mean_activation > target_max:
            balance_discomfort = (mean_activation - target_max) ** 2
        else:
            balance_discomfort = 0.0
        balance_metrics['balance'] = balance_discomfort
    
    # Compute weighted overall discomfort
    total_discomfort = 0.0
    for metric, discomfort in balance_metrics.items():
        weight = weights.get(metric, 1.0)
        total_discomfort += weight * discomfort
    
    balance_metrics['total_discomfort'] = total_discomfort
    
    return balance_metrics


def compute_state_change_penalty(current_state: np.ndarray, 
                               previous_state: np.ndarray,
                               penalty_type: str = 'l2') -> float:
    """
    Compute penalty for rapid state changes to encourage smoothness.
    
    Args:
        current_state: Current affective state
        previous_state: Previous affective state
        penalty_type: Type of penalty ('l2', 'l1', 'max')
        
    Returns:
        State change penalty (non-negative)
    """
    if current_state.shape != previous_state.shape:
        raise ValueError("State vectors must have same dimensions")
    
    change = current_state - previous_state
    
    if penalty_type == 'l2':
        return np.sum(change ** 2)
    elif penalty_type == 'l1':
        return np.sum(np.abs(change))
    elif penalty_type == 'max':
        return np.max(np.abs(change))
    else:
        raise ValueError(f"Unknown penalty type: {penalty_type}")


def find_k_nearest_neighbors(query_vector: np.ndarray, 
                           candidate_vectors: List[np.ndarray],
                           k: int = 5,
                           distance_metric: str = 'euclidean') -> Tuple[List[int], List[float]]:
    """
    Find k nearest neighbors to a query vector.
    
    Args:
        query_vector: Vector to find neighbors for
        candidate_vectors: List of candidate vectors
        k: Number of neighbors to return
        distance_metric: Distance metric to use ('euclidean', 'cosine', 'manhattan')
        
    Returns:
        Tuple of (indices, distances) for k nearest neighbors
    """
    if not candidate_vectors:
        return [], []
    
    if k <= 0:
        return [], []
    
    distances = []
    
    for i, candidate in enumerate(candidate_vectors):
        if distance_metric == 'euclidean':
            dist = euclidean_distance(query_vector, candidate)
        elif distance_metric == 'cosine':
            # Convert cosine similarity to distance
            sim = cosine_similarity(query_vector, candidate)
            dist = 1.0 - sim  # Distance in [0, 2]
        elif distance_metric == 'manhattan':
            dist = manhattan_distance(query_vector, candidate)
        else:
            raise ValueError(f"Unknown distance metric: {distance_metric}")
        
        distances.append((i, dist))
    
    # Sort by distance and take k nearest
    distances.sort(key=lambda x: x[1])
    k_nearest = distances[:min(k, len(distances))]
    
    indices = [idx for idx, _ in k_nearest]
    dists = [dist for _, dist in k_nearest]
    
    return indices, dists


def compute_similarity_weights(distances: List[float], 
                             temperature: float = 1.0,
                             method: str = 'exponential') -> np.ndarray:
    """
    Convert distances to similarity weights.
    
    Args:
        distances: List of distances
        temperature: Temperature parameter for softmax-like weighting
        method: Weighting method ('exponential', 'inverse', 'gaussian')
        
    Returns:
        Normalized similarity weights
    """
    if not distances:
        return np.array([])
    
    distances = np.array(distances)
    
    if method == 'exponential':
        # Exponential decay with distance
        weights = np.exp(-distances / temperature)
    elif method == 'inverse':
        # Inverse distance weighting
        weights = 1.0 / (distances + 1e-8)  # Add small epsilon to avoid division by zero
    elif method == 'gaussian':
        # Gaussian weighting
        weights = np.exp(-(distances ** 2) / (2 * temperature ** 2))
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    # Normalize weights to sum to 1
    total_weight = np.sum(weights)
    if total_weight > 0:
        weights = weights / total_weight
    
    return weights