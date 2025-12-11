"""
Example tool implementations for analysis and research capabilities.

These tools demonstrate how to extend the agent's capabilities with
domain-specific analysis functions that integrate with the Tool Layer.
"""

import json
import time
import random
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime


class TextAnalysisTool:
    """Tool for analyzing text content and extracting insights"""
    
    def __init__(self):
        self.name = "text_analyzer"
        self.description = "Analyze text content for sentiment, complexity, and key themes"
        self.parameters = {
            "text": {"type": "string", "description": "Text content to analyze"},
            "analysis_type": {
                "type": "string", 
                "enum": ["sentiment", "complexity", "themes", "all"],
                "description": "Type of analysis to perform"
            }
        }
    
    def execute(self, text: str, analysis_type: str = "all") -> Dict[str, Any]:
        """Execute text analysis"""
        try:
            results = {}
            
            if analysis_type in ["sentiment", "all"]:
                results["sentiment"] = self._analyze_sentiment(text)
            
            if analysis_type in ["complexity", "all"]:
                results["complexity"] = self._analyze_complexity(text)
            
            if analysis_type in ["themes", "all"]:
                results["themes"] = self._extract_themes(text)
            
            return {
                "success": True,
                "results": results,
                "text_length": len(text),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Simple sentiment analysis (mock implementation)"""
        # In a real implementation, this would use NLP libraries
        positive_words = ["good", "great", "excellent", "positive", "happy", "success"]
        negative_words = ["bad", "terrible", "negative", "sad", "failure", "problem"]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total_words = len(text.split())
        
        return {
            "positive_score": positive_count / max(total_words, 1),
            "negative_score": negative_count / max(total_words, 1),
            "neutrality": 1.0 - (positive_count + negative_count) / max(total_words, 1)
        }
    
    def _analyze_complexity(self, text: str) -> Dict[str, float]:
        """Analyze text complexity metrics"""
        words = text.split()
        sentences = text.split('.')
        
        avg_word_length = np.mean([len(word) for word in words]) if words else 0
        avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences]) if sentences else 0
        
        return {
            "average_word_length": float(avg_word_length),
            "average_sentence_length": float(avg_sentence_length),
            "total_words": len(words),
            "total_sentences": len(sentences),
            "complexity_score": float(avg_word_length * avg_sentence_length / 100)
        }
    
    def _extract_themes(self, text: str) -> List[str]:
        """Extract key themes from text (mock implementation)"""
        # In a real implementation, this would use topic modeling
        common_themes = [
            "technology", "science", "business", "education", "health",
            "environment", "politics", "culture", "art", "sports"
        ]
        
        text_lower = text.lower()
        detected_themes = [theme for theme in common_themes if theme in text_lower]
        
        # Add some randomness for demonstration
        if not detected_themes and len(text) > 50:
            detected_themes = random.sample(common_themes, min(3, len(common_themes)))
        
        return detected_themes


class DataPatternTool:
    """Tool for analyzing patterns in numerical data"""
    
    def __init__(self):
        self.name = "pattern_analyzer"
        self.description = "Analyze patterns, trends, and anomalies in numerical data"
        self.parameters = {
            "data": {"type": "array", "description": "Numerical data array to analyze"},
            "pattern_type": {
                "type": "string",
                "enum": ["trend", "anomaly", "periodicity", "all"],
                "description": "Type of pattern analysis to perform"
            }
        }
    
    def execute(self, data: List[float], pattern_type: str = "all") -> Dict[str, Any]:
        """Execute pattern analysis"""
        try:
            if not data or not all(isinstance(x, (int, float)) for x in data):
                return {
                    "success": False,
                    "error": "Invalid data format - expected list of numbers"
                }
            
            data_array = np.array(data)
            results = {}
            
            if pattern_type in ["trend", "all"]:
                results["trend"] = self._analyze_trend(data_array)
            
            if pattern_type in ["anomaly", "all"]:
                results["anomalies"] = self._detect_anomalies(data_array)
            
            if pattern_type in ["periodicity", "all"]:
                results["periodicity"] = self._analyze_periodicity(data_array)
            
            return {
                "success": True,
                "results": results,
                "data_points": len(data),
                "analysis_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "analysis_timestamp": datetime.now().isoformat()
            }
    
    def _analyze_trend(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze trend in the data"""
        if len(data) < 2:
            return {"trend": "insufficient_data"}
        
        # Simple linear trend analysis
        x = np.arange(len(data))
        slope = np.polyfit(x, data, 1)[0]
        
        trend_strength = abs(slope) / (np.std(data) + 1e-8)
        
        if slope > 0.01:
            trend_direction = "increasing"
        elif slope < -0.01:
            trend_direction = "decreasing"
        else:
            trend_direction = "stable"
        
        return {
            "direction": trend_direction,
            "slope": float(slope),
            "strength": float(trend_strength),
            "r_squared": float(np.corrcoef(x, data)[0, 1] ** 2) if len(data) > 1 else 0.0
        }
    
    def _detect_anomalies(self, data: np.ndarray) -> Dict[str, Any]:
        """Detect anomalies using statistical methods"""
        if len(data) < 3:
            return {"anomalies": [], "method": "insufficient_data"}
        
        # Simple z-score based anomaly detection
        mean = np.mean(data)
        std = np.std(data)
        z_scores = np.abs((data - mean) / (std + 1e-8))
        
        anomaly_threshold = 2.0
        anomaly_indices = np.where(z_scores > anomaly_threshold)[0].tolist()
        anomaly_values = data[anomaly_indices].tolist()
        
        return {
            "anomaly_indices": anomaly_indices,
            "anomaly_values": anomaly_values,
            "anomaly_count": len(anomaly_indices),
            "threshold": anomaly_threshold,
            "method": "z_score"
        }
    
    def _analyze_periodicity(self, data: np.ndarray) -> Dict[str, Any]:
        """Analyze periodicity in the data"""
        if len(data) < 4:
            return {"periodic": False, "reason": "insufficient_data"}
        
        # Simple autocorrelation-based periodicity detection
        autocorr = np.correlate(data, data, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        
        # Look for peaks in autocorrelation (excluding lag 0)
        if len(autocorr) > 1:
            max_corr_idx = np.argmax(autocorr[1:]) + 1
            max_corr_value = autocorr[max_corr_idx] / autocorr[0]
            
            is_periodic = max_corr_value > 0.5
            
            return {
                "periodic": bool(is_periodic),
                "period": int(max_corr_idx) if is_periodic else None,
                "correlation_strength": float(max_corr_value),
                "method": "autocorrelation"
            }
        
        return {"periodic": False, "reason": "analysis_failed"}


class KnowledgeGraphTool:
    """Tool for building and querying knowledge graphs from information"""
    
    def __init__(self):
        self.name = "knowledge_graph"
        self.description = "Build and query knowledge graphs from structured information"
        self.parameters = {
            "action": {
                "type": "string",
                "enum": ["add_entity", "add_relationship", "query", "get_neighbors"],
                "description": "Action to perform on the knowledge graph"
            },
            "entity": {"type": "string", "description": "Entity name (for add_entity, query, get_neighbors)"},
            "entity_type": {"type": "string", "description": "Type of entity (for add_entity)"},
            "source": {"type": "string", "description": "Source entity (for add_relationship)"},
            "target": {"type": "string", "description": "Target entity (for add_relationship)"},
            "relationship": {"type": "string", "description": "Relationship type (for add_relationship)"}
        }
        
        # Simple in-memory knowledge graph
        self.entities = {}  # entity_name -> {type, properties}
        self.relationships = []  # [(source, relationship, target)]
    
    def execute(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute knowledge graph operation"""
        try:
            if action == "add_entity":
                return self._add_entity(kwargs.get("entity"), kwargs.get("entity_type"))
            
            elif action == "add_relationship":
                return self._add_relationship(
                    kwargs.get("source"), 
                    kwargs.get("relationship"), 
                    kwargs.get("target")
                )
            
            elif action == "query":
                return self._query_entity(kwargs.get("entity"))
            
            elif action == "get_neighbors":
                return self._get_neighbors(kwargs.get("entity"))
            
            else:
                return {
                    "success": False,
                    "error": f"Unknown action: {action}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _add_entity(self, entity: str, entity_type: str) -> Dict[str, Any]:
        """Add an entity to the knowledge graph"""
        if not entity or not entity_type:
            return {"success": False, "error": "Entity and entity_type are required"}
        
        self.entities[entity] = {
            "type": entity_type,
            "created": datetime.now().isoformat(),
            "properties": {}
        }
        
        return {
            "success": True,
            "message": f"Added entity '{entity}' of type '{entity_type}'",
            "total_entities": len(self.entities)
        }
    
    def _add_relationship(self, source: str, relationship: str, target: str) -> Dict[str, Any]:
        """Add a relationship between entities"""
        if not all([source, relationship, target]):
            return {"success": False, "error": "Source, relationship, and target are required"}
        
        # Auto-create entities if they don't exist
        if source not in self.entities:
            self.entities[source] = {"type": "unknown", "created": datetime.now().isoformat()}
        if target not in self.entities:
            self.entities[target] = {"type": "unknown", "created": datetime.now().isoformat()}
        
        self.relationships.append((source, relationship, target))
        
        return {
            "success": True,
            "message": f"Added relationship: {source} --{relationship}--> {target}",
            "total_relationships": len(self.relationships)
        }
    
    def _query_entity(self, entity: str) -> Dict[str, Any]:
        """Query information about an entity"""
        if not entity:
            return {"success": False, "error": "Entity name is required"}
        
        if entity not in self.entities:
            return {
                "success": False,
                "error": f"Entity '{entity}' not found in knowledge graph"
            }
        
        # Find all relationships involving this entity
        outgoing = [(rel, target) for source, rel, target in self.relationships if source == entity]
        incoming = [(rel, source) for source, rel, target in self.relationships if target == entity]
        
        return {
            "success": True,
            "entity": entity,
            "entity_info": self.entities[entity],
            "outgoing_relationships": outgoing,
            "incoming_relationships": incoming,
            "total_connections": len(outgoing) + len(incoming)
        }
    
    def _get_neighbors(self, entity: str) -> Dict[str, Any]:
        """Get all neighboring entities"""
        if not entity:
            return {"success": False, "error": "Entity name is required"}
        
        if entity not in self.entities:
            return {
                "success": False,
                "error": f"Entity '{entity}' not found in knowledge graph"
            }
        
        neighbors = set()
        for source, rel, target in self.relationships:
            if source == entity:
                neighbors.add(target)
            elif target == entity:
                neighbors.add(source)
        
        return {
            "success": True,
            "entity": entity,
            "neighbors": list(neighbors),
            "neighbor_count": len(neighbors)
        }


# Tool registration helper
def get_analysis_tools() -> List[Dict[str, Any]]:
    """Get all analysis tools for registration with the Tool Layer"""
    
    text_tool = TextAnalysisTool()
    pattern_tool = DataPatternTool()
    kg_tool = KnowledgeGraphTool()
    
    return [
        {
            "name": text_tool.name,
            "function": text_tool.execute,
            "metadata": {
                "description": text_tool.description,
                "parameters": text_tool.parameters,
                "category": "analysis",
                "complexity": "medium"
            }
        },
        {
            "name": pattern_tool.name,
            "function": pattern_tool.execute,
            "metadata": {
                "description": pattern_tool.description,
                "parameters": pattern_tool.parameters,
                "category": "analysis",
                "complexity": "medium"
            }
        },
        {
            "name": kg_tool.name,
            "function": kg_tool.execute,
            "metadata": {
                "description": kg_tool.description,
                "parameters": kg_tool.parameters,
                "category": "knowledge_management",
                "complexity": "high"
            }
        }
    ]


if __name__ == "__main__":
    # Example usage
    tools = get_analysis_tools()
    
    print("Available Analysis Tools:")
    for tool in tools:
        print(f"- {tool['name']}: {tool['metadata']['description']}")
    
    # Test text analysis tool
    text_tool = TextAnalysisTool()
    result = text_tool.execute("This is a great example of positive text analysis!", "all")
    print(f"\nText Analysis Result: {json.dumps(result, indent=2)}")
    
    # Test pattern analysis tool
    pattern_tool = DataPatternTool()
    test_data = [1, 2, 3, 5, 8, 13, 21, 34, 55]  # Fibonacci sequence
    result = pattern_tool.execute(test_data, "trend")
    print(f"\nPattern Analysis Result: {json.dumps(result, indent=2)}")