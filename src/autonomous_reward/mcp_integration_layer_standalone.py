"""
Standalone MCP Integration Layer for Autonomous Reward System.

This module implements the MCPIntegrationLayer class that provides seamless
plug-and-play integration with external capabilities through the Model Context Protocol,
while providing rich reward feedback based on tool usage effectiveness.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Set
import numpy as np

logger = logging.getLogger(__name__)


# Minimal required classes for standalone operation
class AutonomousGoal:
    def __init__(self, goal_id: str, goal_type: str, description: str, priority: float, created_at: int):
        self.goal_id = goal_id
        self.goal_type = goal_type
        self.description = description
        self.priority = priority
        self.created_at = created_at


class ToolResult:
    def __init__(self, success: bool = True, content: str = "", metadata: Optional[Dict[str, Any]] = None):
        self.success = success
        self.content = content
        self.metadata = metadata or {}


class ToolCall:
    def __init__(self, tool_name: str, arguments: Dict[str, Any], timestamp: int):
        self.tool_name = tool_name
        self.arguments = arguments
        self.timestamp = timestamp


class AgentConfig:
    def __init__(self):
        pass


class ToolLayer:
    def __init__(self):
        self.available_tools = ['search_web', 'analyze_data', 'generate_text', 'process_image']
    
    def get_available_tools(self) -> List[str]:
        return self.available_tools.copy()
    
    def execute_tool(self, tool_call: ToolCall) -> ToolResult:
        return ToolResult(success=True, content=f"Mock result for {tool_call.tool_name}")


class MCPIntegrationLayer:
    """
    Provides seamless plug-and-play integration with external capabilities.
    
    This class integrates existing ToolLayer MCP capabilities with autonomous rewards,
    implementing reward feedback from tool usage effectiveness, tool selection based
    on autonomous state and emergent goals, and tool effectiveness learning that
    feeds back to reward systems.
    """
    
    def __init__(self, config: AgentConfig, tool_layer: ToolLayer):
        """
        Initialize the MCP integration layer.
        
        Args:
            config: Agent configuration
            tool_layer: Existing ToolLayer instance with MCP capabilities
        """
        self.config = config
        self.tool_layer = tool_layer
        
        # Tool effectiveness learning
        self.tool_effectiveness_history: Dict[str, List[Dict[str, Any]]] = {}
        self.tool_usage_patterns: Dict[str, Dict[str, Any]] = {}
        self.goal_tool_associations: Dict[str, List[str]] = {}
        
        # State-based tool selection
        self.state_tool_preferences: Dict[str, float] = {}
        self.emergent_goal_tools: Dict[str, Set[str]] = {}
        
        # Reward feedback system
        self.tool_reward_multipliers: Dict[str, float] = {}
        self.context_reward_modifiers: Dict[str, float] = {}
        
        # Auto-discovery tracking
        self.discovered_servers: List[Dict[str, Any]] = []
        self.server_discovery_timestamp = 0
        self.discovery_interval = 300  # 5 minutes
        
        logger.info("Initialized MCPIntegrationLayer with autonomous reward integration")
    
    def discover_mcp_servers(self) -> List[Dict[str, Any]]:
        """
        Discover available MCP servers.
        
        Returns:
            List of discovered MCP server information
        """
        current_time = time.time()
        
        # Check if we need to refresh discovery
        if current_time - self.server_discovery_timestamp > self.discovery_interval:
            self._refresh_server_discovery()
            self.server_discovery_timestamp = current_time
        
        return self.discovered_servers.copy()
    
    def select_optimal_tool(self, goal: AutonomousGoal, available_tools: List[Dict[str, Any]], 
                          current_state: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Select optimal tool based on current state and emergent goals.
        
        Args:
            goal: Current autonomous goal
            available_tools: List of available tools
            current_state: Current pathos state
            
        Returns:
            Selected tool information or None
        """
        if not available_tools:
            logger.warning("No available tools for goal selection")
            return None
        
        # Get available tool names from ToolLayer
        tool_names = self.tool_layer.get_available_tools()
        
        # Filter available_tools to only include registered tools
        valid_tools = []
        for tool_info in available_tools:
            tool_name = tool_info.get('name', '')
            if tool_name in tool_names:
                valid_tools.append(tool_info)
        
        if not valid_tools:
            logger.warning("No valid tools available for selection")
            return None
        
        # Score tools based on multiple factors
        tool_scores = {}
        
        for tool_info in valid_tools:
            tool_name = tool_info.get('name', '')
            score = self._compute_tool_selection_score(
                tool_name, goal, current_state, tool_info
            )
            tool_scores[tool_name] = score
        
        # Select tool with highest score
        if tool_scores:
            best_tool_name = max(tool_scores, key=tool_scores.get)
            best_tool_info = next(
                (tool for tool in valid_tools if tool.get('name') == best_tool_name),
                None
            )
            
            logger.debug(f"Selected optimal tool: {best_tool_name} with score {tool_scores[best_tool_name]:.4f}")
            return best_tool_info
        
        return None
    
    def execute_tool_with_feedback(self, tool: Dict[str, Any], 
                                 parameters: Dict[str, Any]) -> ToolResult:
        """
        Execute tool and provide rich reward feedback.
        
        Args:
            tool: Tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result with enhanced feedback
        """
        tool_name = tool.get('name', 'unknown_tool')
        
        # Create ToolCall for execution
        tool_call = ToolCall(
            tool_name=tool_name,
            arguments=parameters,
            timestamp=int(time.time())
        )
        
        # Execute through ToolLayer
        start_time = time.time()
        result = self.tool_layer.execute_tool(tool_call)
        execution_time = time.time() - start_time
        
        # Enhance result with autonomous reward feedback
        enhanced_result = self._enhance_result_with_feedback(
            result, tool, parameters, execution_time
        )
        
        # Record usage for learning
        self._record_tool_usage(tool, parameters, enhanced_result, execution_time)
        
        logger.debug(f"Executed tool {tool_name} with enhanced feedback")
        return enhanced_result
    
    def learn_tool_effectiveness(self, tool: Dict[str, Any], context: Dict[str, Any], 
                               outcome: Dict[str, Any]) -> float:
        """
        Learn from tool usage effectiveness and update reward feedback.
        
        Args:
            tool: Tool that was used
            context: Context in which tool was used
            outcome: Outcome of tool usage
            
        Returns:
            Updated effectiveness score
        """
        tool_name = tool.get('name', 'unknown_tool')
        
        # Calculate effectiveness score
        effectiveness_score = self._calculate_effectiveness_score(outcome, context)
        
        # Update tool effectiveness history
        if tool_name not in self.tool_effectiveness_history:
            self.tool_effectiveness_history[tool_name] = []
        
        effectiveness_record = {
            'timestamp': time.time(),
            'context': context.copy(),
            'outcome': outcome.copy(),
            'effectiveness_score': effectiveness_score
        }
        
        self.tool_effectiveness_history[tool_name].append(effectiveness_record)
        
        # Keep only recent history (last 100 uses)
        if len(self.tool_effectiveness_history[tool_name]) > 100:
            self.tool_effectiveness_history[tool_name] = \
                self.tool_effectiveness_history[tool_name][-100:]
        
        # Update reward multipliers based on learning
        self._update_reward_multipliers(tool_name, effectiveness_score)
        
        # Update usage patterns
        self._update_usage_patterns(tool_name, context, effectiveness_score)
        
        logger.debug(f"Updated effectiveness for {tool_name}: {effectiveness_score:.4f}")
        return effectiveness_score
    
    def get_tool_reward_feedback(self, tool: Dict[str, Any], 
                               result: ToolResult) -> Dict[str, float]:
        """
        Generate reward feedback based on tool usage effectiveness.
        
        Args:
            tool: Tool that was used
            result: Tool execution result
            
        Returns:
            Dictionary of reward feedback values
        """
        tool_name = tool.get('name', 'unknown_tool')
        
        # Base reward from result success
        base_reward = 1.0 if result.success else 0.0
        
        # Apply effectiveness multiplier
        effectiveness_multiplier = self.tool_reward_multipliers.get(tool_name, 1.0)
        
        # Calculate context-specific modifiers
        context_modifier = self._calculate_context_modifier(tool, result)
        
        # Calculate novelty bonus
        novelty_bonus = self._calculate_novelty_bonus(tool_name)
        
        # Calculate efficiency bonus
        efficiency_bonus = self._calculate_efficiency_bonus(tool, result)
        
        reward_feedback = {
            'base_reward': base_reward,
            'effectiveness_multiplier': effectiveness_multiplier,
            'context_modifier': context_modifier,
            'novelty_bonus': novelty_bonus,
            'efficiency_bonus': efficiency_bonus,
            'total_reward': (base_reward * effectiveness_multiplier + 
                           context_modifier + novelty_bonus + efficiency_bonus)
        }
        
        logger.debug(f"Generated reward feedback for {tool_name}: {reward_feedback['total_reward']:.4f}")
        return reward_feedback
    
    def integrate_with_autonomous_goals(self, goals: List[AutonomousGoal]) -> Dict[str, Any]:
        """
        Integrate tool capabilities with autonomous goals.
        
        Args:
            goals: List of current autonomous goals
            
        Returns:
            Integration mapping and recommendations
        """
        integration_map = {}
        tool_recommendations = {}
        
        # Get available tools
        available_tools = self.tool_layer.get_available_tools()
        
        for goal in goals:
            goal_id = goal.goal_id
            goal_type = goal.goal_type
            goal_description = goal.description
            
            # Find relevant tools for this goal
            relevant_tools = self._find_relevant_tools_for_goal(
                goal, available_tools
            )
            
            # Score tools for this goal
            tool_scores = {}
            for tool_name in relevant_tools:
                score = self._score_tool_for_goal(tool_name, goal)
                tool_scores[tool_name] = score
            
            # Store integration mapping
            integration_map[goal_id] = {
                'goal': goal,
                'relevant_tools': relevant_tools,
                'tool_scores': tool_scores,
                'best_tool': max(tool_scores, key=tool_scores.get) if tool_scores else None
            }
            
            # Generate recommendations
            if tool_scores:
                sorted_tools = sorted(tool_scores.items(), key=lambda x: x[1], reverse=True)
                tool_recommendations[goal_id] = sorted_tools[:3]  # Top 3 tools
        
        # Update goal-tool associations
        self._update_goal_tool_associations(integration_map)
        
        logger.info(f"Integrated {len(goals)} goals with available tools")
        return {
            'integration_map': integration_map,
            'tool_recommendations': tool_recommendations,
            'total_goals': len(goals),
            'total_tools': len(available_tools)
        }
    
    # Private helper methods
    
    def _refresh_server_discovery(self):
        """Refresh the list of discovered MCP servers."""
        try:
            # Use ToolLayer's server discovery if available
            if hasattr(self.tool_layer, 'discover_servers'):
                servers = self.tool_layer.discover_servers()
                self.discovered_servers = servers
            else:
                # Fallback: get available tools as proxy for servers
                tools = self.tool_layer.get_available_tools()
                self.discovered_servers = [
                    {
                        'name': f'server_for_{tool}',
                        'tools': [tool],
                        'status': 'active'
                    }
                    for tool in tools
                ]
            
            logger.debug(f"Discovered {len(self.discovered_servers)} MCP servers")
        except Exception as e:
            logger.error(f"Error refreshing server discovery: {e}")
            self.discovered_servers = []
    
    def _compute_tool_selection_score(self, tool_name: str, goal: AutonomousGoal, 
                                    current_state: np.ndarray, tool_info: Dict[str, Any]) -> float:
        """Compute selection score for a tool."""
        score = 0.0
        
        # Base effectiveness score
        if tool_name in self.tool_effectiveness_history:
            recent_scores = [
                record['effectiveness_score'] 
                for record in self.tool_effectiveness_history[tool_name][-10:]
            ]
            if recent_scores:
                score += np.mean(recent_scores) * 0.4
        else:
            score += 0.5  # Neutral score for unknown tools
        
        # Goal relevance score
        goal_relevance = self._calculate_goal_relevance(tool_name, goal)
        score += goal_relevance * 0.3
        
        # State preference score
        state_preference = self.state_tool_preferences.get(tool_name, 0.5)
        score += state_preference * 0.2
        
        # Novelty bonus (encourage exploration)
        usage_count = len(self.tool_effectiveness_history.get(tool_name, []))
        novelty_bonus = max(0, (10 - usage_count) / 10) * 0.1
        score += novelty_bonus
        
        return max(0.0, min(1.0, score))  # Clamp to [0, 1]
    
    def _enhance_result_with_feedback(self, result: ToolResult, tool: Dict[str, Any], 
                                    parameters: Dict[str, Any], execution_time: float) -> ToolResult:
        """Enhance tool result with autonomous reward feedback."""
        # Generate reward feedback
        reward_feedback = self.get_tool_reward_feedback(tool, result)
        
        # Add autonomous reward metadata
        if not hasattr(result, 'metadata'):
            result.metadata = {}
        
        result.metadata.update({
            'autonomous_reward_feedback': reward_feedback,
            'execution_time': execution_time,
            'tool_effectiveness_score': reward_feedback.get('effectiveness_multiplier', 1.0),
            'autonomous_integration': True
        })
        
        return result
    
    def _record_tool_usage(self, tool: Dict[str, Any], parameters: Dict[str, Any], 
                         result: ToolResult, execution_time: float):
        """Record tool usage for learning."""
        tool_name = tool.get('name', 'unknown_tool')
        
        # Create usage record
        usage_record = {
            'timestamp': time.time(),
            'tool': tool.copy(),
            'parameters': parameters.copy(),
            'success': result.success,
            'execution_time': execution_time,
            'result_quality': self._assess_result_quality(result)
        }
        
        # Update usage patterns
        if tool_name not in self.tool_usage_patterns:
            self.tool_usage_patterns[tool_name] = {
                'total_uses': 0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'parameter_patterns': {}
            }
        
        patterns = self.tool_usage_patterns[tool_name]
        patterns['total_uses'] += 1
        
        # Update success rate
        current_successes = patterns['success_rate'] * (patterns['total_uses'] - 1)
        new_successes = current_successes + (1 if result.success else 0)
        patterns['success_rate'] = new_successes / patterns['total_uses']
        
        # Update average execution time
        current_avg_time = patterns['avg_execution_time'] * (patterns['total_uses'] - 1)
        patterns['avg_execution_time'] = (current_avg_time + execution_time) / patterns['total_uses']
    
    def _calculate_effectiveness_score(self, outcome: Dict[str, Any], 
                                     context: Dict[str, Any]) -> float:
        """Calculate effectiveness score from outcome and context."""
        score = 0.0
        
        # Success component
        if outcome.get('success', False):
            score += 0.5
        
        # Quality component
        quality = outcome.get('quality', 0.5)
        score += quality * 0.3
        
        # Efficiency component
        execution_time = outcome.get('execution_time', 1.0)
        efficiency = max(0, 1.0 - min(execution_time / 10.0, 1.0))  # Normalize to 10s max
        score += efficiency * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _update_reward_multipliers(self, tool_name: str, effectiveness_score: float):
        """Update reward multipliers based on effectiveness learning."""
        current_multiplier = self.tool_reward_multipliers.get(tool_name, 1.0)
        
        # Exponential moving average update
        alpha = 0.1  # Learning rate
        new_multiplier = (1 - alpha) * current_multiplier + alpha * effectiveness_score
        
        self.tool_reward_multipliers[tool_name] = max(0.1, min(2.0, new_multiplier))
    
    def _update_usage_patterns(self, tool_name: str, context: Dict[str, Any], 
                             effectiveness_score: float):
        """Update usage patterns based on context and effectiveness."""
        if tool_name not in self.tool_usage_patterns:
            self.tool_usage_patterns[tool_name] = {
                'total_uses': 0,
                'success_rate': 0.0,
                'avg_execution_time': 0.0,
                'parameter_patterns': {},
                'context_effectiveness': {},
                'best_contexts': []
            }
        
        # Ensure context_effectiveness exists
        if 'context_effectiveness' not in self.tool_usage_patterns[tool_name]:
            self.tool_usage_patterns[tool_name]['context_effectiveness'] = {}
        
        # Update context effectiveness
        context_key = str(sorted(context.items()))
        if context_key not in self.tool_usage_patterns[tool_name]['context_effectiveness']:
            self.tool_usage_patterns[tool_name]['context_effectiveness'][context_key] = []
        
        self.tool_usage_patterns[tool_name]['context_effectiveness'][context_key].append(
            effectiveness_score
        )
        
        # Keep only recent scores
        if len(self.tool_usage_patterns[tool_name]['context_effectiveness'][context_key]) > 10:
            self.tool_usage_patterns[tool_name]['context_effectiveness'][context_key] = \
                self.tool_usage_patterns[tool_name]['context_effectiveness'][context_key][-10:]
    
    def _calculate_context_modifier(self, tool: Dict[str, Any], result: ToolResult) -> float:
        """Calculate context-specific reward modifier."""
        tool_name = tool.get('name', 'unknown_tool')
        
        # Base modifier
        modifier = 0.0
        
        # Check if this context has been successful before
        if tool_name in self.context_reward_modifiers:
            modifier += self.context_reward_modifiers[tool_name] * 0.1
        
        # Add result-specific bonuses
        if result.success and hasattr(result, 'metadata'):
            metadata = result.metadata or {}
            if metadata.get('high_quality', False):
                modifier += 0.2
            if metadata.get('innovative', False):
                modifier += 0.1
        
        return modifier
    
    def _calculate_novelty_bonus(self, tool_name: str) -> float:
        """Calculate novelty bonus for tool usage."""
        usage_count = len(self.tool_effectiveness_history.get(tool_name, []))
        
        # Higher bonus for less used tools
        if usage_count == 0:
            return 0.3  # High bonus for first use
        elif usage_count < 5:
            return 0.1  # Medium bonus for early uses
        else:
            return 0.0  # No bonus for frequently used tools
    
    def _calculate_efficiency_bonus(self, tool: Dict[str, Any], result: ToolResult) -> float:
        """Calculate efficiency bonus based on execution performance."""
        if not hasattr(result, 'metadata') or not result.metadata:
            return 0.0
        
        execution_time = result.metadata.get('execution_time', 1.0)
        
        # Bonus for fast execution (under 1 second)
        if execution_time < 1.0:
            return 0.1
        elif execution_time < 5.0:
            return 0.05
        else:
            return 0.0
    
    def _find_relevant_tools_for_goal(self, goal: AutonomousGoal, 
                                    available_tools: List[str]) -> List[str]:
        """Find tools relevant to a specific goal."""
        relevant_tools = []
        
        goal_keywords = self._extract_goal_keywords(goal)
        
        for tool_name in available_tools:
            # Check if tool name contains goal keywords
            tool_relevance = self._calculate_tool_goal_relevance(tool_name, goal_keywords)
            
            if tool_relevance > 0.3:  # Threshold for relevance
                relevant_tools.append(tool_name)
        
        # Also check historical associations
        goal_type = goal.goal_type
        if goal_type in self.goal_tool_associations:
            for tool_name in self.goal_tool_associations[goal_type]:
                if tool_name in available_tools and tool_name not in relevant_tools:
                    relevant_tools.append(tool_name)
        
        return relevant_tools
    
    def _score_tool_for_goal(self, tool_name: str, goal: AutonomousGoal) -> float:
        """Score how well a tool matches a goal."""
        score = 0.0
        
        # Historical effectiveness
        if tool_name in self.tool_effectiveness_history:
            recent_scores = [
                record['effectiveness_score'] 
                for record in self.tool_effectiveness_history[tool_name][-5:]
            ]
            if recent_scores:
                score += np.mean(recent_scores) * 0.5
        
        # Goal-tool association strength
        goal_type = goal.goal_type
        if goal_type in self.goal_tool_associations:
            if tool_name in self.goal_tool_associations[goal_type]:
                score += 0.3
        
        # Keyword relevance
        goal_keywords = self._extract_goal_keywords(goal)
        keyword_relevance = self._calculate_tool_goal_relevance(tool_name, goal_keywords)
        score += keyword_relevance * 0.2
        
        return max(0.0, min(1.0, score))
    
    def _update_goal_tool_associations(self, integration_map: Dict[str, Any]):
        """Update goal-tool associations based on integration results."""
        for goal_id, mapping in integration_map.items():
            goal = mapping['goal']
            goal_type = goal.goal_type
            relevant_tools = mapping['relevant_tools']
            
            if goal_type not in self.goal_tool_associations:
                self.goal_tool_associations[goal_type] = []
            
            # Add new associations
            for tool_name in relevant_tools:
                if tool_name not in self.goal_tool_associations[goal_type]:
                    self.goal_tool_associations[goal_type].append(tool_name)
            
            # Keep only top 10 associations per goal type
            if len(self.goal_tool_associations[goal_type]) > 10:
                # Sort by effectiveness and keep top 10
                tool_scores = [
                    (tool, self._get_tool_effectiveness(tool))
                    for tool in self.goal_tool_associations[goal_type]
                ]
                tool_scores.sort(key=lambda x: x[1], reverse=True)
                self.goal_tool_associations[goal_type] = [
                    tool for tool, _ in tool_scores[:10]
                ]
    
    def _calculate_goal_relevance(self, tool_name: str, goal: AutonomousGoal) -> float:
        """Calculate how relevant a tool is to a goal."""
        goal_keywords = self._extract_goal_keywords(goal)
        return self._calculate_tool_goal_relevance(tool_name, goal_keywords)
    
    def _extract_goal_keywords(self, goal: AutonomousGoal) -> List[str]:
        """Extract keywords from goal description."""
        description = goal.description.lower()
        
        # Simple keyword extraction (could be enhanced with NLP)
        keywords = []
        
        # Common action words
        action_words = ['create', 'build', 'analyze', 'process', 'generate', 'search', 
                       'find', 'calculate', 'transform', 'optimize', 'learn', 'explore']
        
        for word in action_words:
            if word in description:
                keywords.append(word)
        
        # Extract nouns (simple heuristic)
        words = description.split()
        for word in words:
            if len(word) > 3 and word.isalpha():
                keywords.append(word)
        
        return list(set(keywords))  # Remove duplicates
    
    def _calculate_tool_goal_relevance(self, tool_name: str, goal_keywords: List[str]) -> float:
        """Calculate relevance between tool name and goal keywords."""
        if not goal_keywords:
            return 0.0
        
        tool_name_lower = tool_name.lower()
        matches = 0
        
        for keyword in goal_keywords:
            if keyword in tool_name_lower:
                matches += 1
        
        return matches / len(goal_keywords)
    
    def _assess_result_quality(self, result: ToolResult) -> float:
        """Assess the quality of a tool result."""
        if not result.success:
            return 0.0
        
        quality = 0.5  # Base quality for successful results
        
        # Check for additional quality indicators
        if hasattr(result, 'metadata') and result.metadata:
            metadata = result.metadata
            
            if metadata.get('comprehensive', False):
                quality += 0.2
            if metadata.get('accurate', False):
                quality += 0.2
            if metadata.get('efficient', False):
                quality += 0.1
        
        # Check result content quality (simple heuristics)
        if hasattr(result, 'content') and result.content:
            content_length = len(str(result.content))
            if content_length > 100:  # Substantial content
                quality += 0.1
        
        return max(0.0, min(1.0, quality))
    
    def _get_tool_effectiveness(self, tool_name: str) -> float:
        """Get current effectiveness score for a tool."""
        if tool_name not in self.tool_effectiveness_history:
            return 0.5  # Default for unknown tools
        
        recent_scores = [
            record['effectiveness_score'] 
            for record in self.tool_effectiveness_history[tool_name][-10:]
        ]
        
        if recent_scores:
            return np.mean(recent_scores)
        else:
            return 0.5