"""
Pathos Layer - Affective dynamics engine for the autonomous agent.
"""

from .interfaces import PathosLayer as PathosLayerInterface
from .pathos_layer import PathosLayer

__all__ = ['PathosLayerInterface', 'PathosLayer']