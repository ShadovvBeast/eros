"""
Main Agent class that coordinates all layers of the Autonomous Logos-Pathos-Memory Agent.
"""

from typing import Optional, Dict, Any
import time
from datetime import datetime
import numpy as np

from .config import AgentConfig
from .models import PathosState, SemanticVector, MemoryTrace, Intention, ToolCall, ToolResult
from ..logos.interfaces import LogosLayer
from ..pathos.interfaces import PathosLayer
from ..memory.interfaces import MemorySystem
from ..ethos.interfaces import EthosFramework
from ..tools.interfaces import ToolLayer
from .logging_config import logger, instrumentation as default_instrumentation
from .math_utils import cosine_similarity


class AutonomousAgent:
    """
    Main agent class implementing the continuous operation loop:
    Logos → Pathos → Memory → Tool execution
    """
    
    def __init__(self, config: AgentConfig, instrumentation=None):
        self.config = config
        self.cycle_count = 0
        self.running = False
        self.instrumentation = instrumentation or default_instrumentation
        
        # Initialize layers (will be implemented in subsequent tasks)
        self.logos: Optional[LogosLayer] = None
        self.pathos: Optional[PathosLayer] = None
        self.memory: Optional[MemorySystem] = None
        self.ethos: Optional[EthosFramework] = None
        self.tools: Optional[ToolLayer] = None
        
        logger.info("Agent initialized")
    
    def initialize_layers(self, logos: LogosLayer, pathos: PathosLayer, 
                         memory: MemorySystem, ethos: EthosFramework, tools: ToolLayer):
        """Initialize all agent layers"""
        self.logos = logos
        self.pathos = pathos
        self.memory = memory
        self.ethos = ethos
        self.tools = tools
        
        # Register tool result callbacks for layer integration
        self.tools.register_result_callback(self._handle_tool_result)
        
        logger.info("All agent layers initialized")
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        Execute one complete agent cycle.
        
        Returns:
            Dictionary containing cycle results and metrics
        """
        # Check if all layers are initialized
        if not all([self.logos, self.pathos, self.memory, self.ethos, self.tools]):
            raise RuntimeError("Agent layers not fully initialized. Call initialize_layers() first.")
        
        cycle_start = time.time()
        phase_timings = {}
        
        try:
            # Phase 1: Logos interpretation and intention generation
            phase_start = time.time()
            recent_memories = self.memory.get_recent_traces(self.config.pathos.memory_retrieval_k)
            semantic_vector = self.logos.interpret_state(self.pathos.current_state, recent_memories)
            intention = self.logos.generate_intention(semantic_vector, self.pathos.current_state)
            interest_signal = self.logos.compute_interest_signal(semantic_vector)
            phase_timings['logos'] = time.time() - phase_start
            
            # Create a "thinking" prompt-like log entry
            thinking_prompt = self._generate_thinking_prompt(semantic_vector, self.pathos.current_state, recent_memories)
            logger.debug(f"Cycle {self.cycle_count} - Agent Reasoning", 
                        thinking_prompt=thinking_prompt,
                        semantic_category=semantic_vector.semantic_category,
                        interest_signal=f"{interest_signal:.3f}",
                        recent_memories_count=len(recent_memories),
                        intention_preview=intention.description[:80])
            
            # Phase 2: Ethos validation
            phase_start = time.time()
            intention_valid, intention_error = self.ethos.validate_intention(intention)
            if not intention_valid:
                logger.warning("Intention rejected by Ethos", error=intention_error)
                # Generate fallback intention
                intention = self._generate_fallback_intention()
            phase_timings['ethos'] = time.time() - phase_start
            
            # Phase 3: Tool execution (if needed)
            phase_start = time.time()
            tool_result = None
            external_reward = 0.0
            
            tool_call = self.logos.evaluate_tool_usage(intention, self.tools.get_available_tools())
            if tool_call:
                # Log tool decision reasoning
                tool_reasoning = self._generate_tool_reasoning(intention, tool_call)
                logger.debug(f"Cycle {self.cycle_count} - Tool Decision", 
                            tool_reasoning=tool_reasoning,
                            selected_tool=tool_call.tool_name,
                            tool_args=str(tool_call.arguments)[:100])
                
                tool_valid, tool_error = self.ethos.validate_tool_call(tool_call)
                if tool_valid:
                    logger.debug(f"Cycle {self.cycle_count} - Executing tool", 
                                tool_name=tool_call.tool_name,
                                validation_status="APPROVED",
                                execution_context=f"For {intention.description[:50]}")
                    tool_result = self.tools.execute_tool(tool_call)
                    external_reward = 1.0 if tool_result.success else -0.5
                    
                    result_analysis = "SUCCESS - Tool achieved intended outcome" if tool_result.success else "FAILURE - Tool did not achieve intended outcome"
                    logger.debug(f"Cycle {self.cycle_count} - Tool result", 
                                tool_name=tool_call.tool_name,
                                result_analysis=result_analysis,
                                external_reward=f"{external_reward:+.3f}",
                                impact="Positive reinforcement" if external_reward > 0 else "Negative feedback")
                    self.instrumentation.record_tool_usage(
                        tool_call.tool_name, tool_result.success, 
                        time.time() - phase_start
                    )
                else:
                    logger.warning(f"Cycle {self.cycle_count} - Tool rejected by Ethos", 
                                  tool_name=tool_call.tool_name,
                                  rejection_reason=tool_error,
                                  safety_impact="Prevented potentially unsafe action")
            else:
                reasoning = f"No tools needed for '{intention.description[:50]}' - can be accomplished through internal processing"
                logger.debug(f"Cycle {self.cycle_count} - No tool usage", 
                            reasoning=reasoning,
                            decision="INTERNAL_PROCESSING_SUFFICIENT")
            phase_timings['tools'] = time.time() - phase_start
            
            # Phase 4: Pathos state update
            phase_start = time.time()
            retrieved_memories = self.memory.query_by_affect(
                self.pathos.current_state, self.config.pathos.memory_retrieval_k
            )
            
            new_state = self.pathos.update_state(
                semantic_vector, external_reward, interest_signal, retrieved_memories
            )
            
            internal_reward = self.pathos.compute_internal_reward(
                new_state, self.pathos.current_state
            )
            
            # Update preference learning
            total_reward = internal_reward + external_reward
            self.logos.update_preferences(total_reward, semantic_vector)
            
            logger.debug(f"Cycle {self.cycle_count} - Pathos update", 
                        internal_reward=internal_reward,
                        external_reward=external_reward,
                        total_reward=total_reward,
                        state_change=float(np.linalg.norm(new_state - self.pathos.current_state)),
                        retrieved_memories=len(retrieved_memories))
            
            phase_timings['pathos'] = time.time() - phase_start
            
            # Phase 5: Memory storage decision
            phase_start = time.time()
            novelty_affect, novelty_semantic = self.memory.estimate_novelty(
                new_state, semantic_vector
            )
            
            state_change = float(np.linalg.norm(new_state - self.pathos.current_state))
            salience = self.pathos.compute_salience(
                state_change, total_reward, novelty_affect, novelty_semantic, interest_signal
            )
            
            if self.pathos.should_write_memory(salience):
                memory_trace = MemoryTrace(
                    affect_state=new_state.copy(),
                    semantic_vector=semantic_vector,
                    reward=total_reward,
                    timestamp=int(datetime.now().timestamp()),
                    salience=salience,
                    metadata={
                        'cycle': self.cycle_count,
                        'intention': intention.description,
                        'tool_used': tool_call.tool_name if tool_call else None
                    }
                )
                self.memory.store_trace(memory_trace)
                logger.debug(f"Cycle {self.cycle_count} - Memory stored", 
                            salience=salience,
                            total_traces=self.memory.get_trace_count(),
                            total_reward=total_reward)
                self.instrumentation.record_memory_event('store', salience, self.memory.get_trace_count())
            else:
                logger.debug(f"Cycle {self.cycle_count} - Memory not stored", 
                            salience=salience,
                            threshold="below_threshold")
            phase_timings['memory'] = time.time() - phase_start
            
            # Update Pathos state
            self.pathos.previous_state = self.pathos.current_state.copy()
            self.pathos.current_state = new_state
            
            # Update attractor dynamics based on experience
            if hasattr(self.pathos, 'exhibit_attractor_behavior'):
                self.pathos.exhibit_attractor_behavior(new_state, semantic_vector, total_reward)
            
            # Periodically decay attractor strengths (every 10 cycles)
            if hasattr(self.pathos, 'decay_attractor_strengths') and self.cycle_count % 10 == 0:
                self.pathos.decay_attractor_strengths()
            
            # Record instrumentation
            homeostatic_balance, _ = self.pathos.compute_homeostatic_balance(new_state)
            self.instrumentation.record_pathos_state(
                self.cycle_count, new_state, internal_reward, homeostatic_balance
            )
            
            # Record behavior patterns
            state_similarity = float(cosine_similarity(new_state, self.pathos.previous_state)) if hasattr(self.pathos, 'previous_state') else 0.0
            self.instrumentation.record_behavior_pattern(
                self.cycle_count, intention.description, 
                tool_call.tool_name if tool_call else None,
                total_reward, state_similarity
            )
            
            # Record attractor patterns if available
            if hasattr(self.pathos, 'attractor_states'):
                strongest_strength = max(self.pathos.attractor_strengths) if self.pathos.attractor_strengths else 0.0
                pattern_recognized = hasattr(self.pathos, '_last_pattern_recognized') and self.pathos._last_pattern_recognized
                self.instrumentation.record_attractor_pattern(
                    self.cycle_count, len(self.pathos.attractor_states),
                    strongest_strength, pattern_recognized
                )
            
            cycle_duration = time.time() - cycle_start
            self.instrumentation.record_cycle_timing(self.cycle_count, cycle_duration, phase_timings)
            
            self.cycle_count += 1
            
            return {
                'cycle': self.cycle_count - 1,
                'intention': intention.description,
                'semantic_category': semantic_vector.semantic_category,
                'internal_reward': internal_reward,
                'external_reward': external_reward,
                'salience': salience,
                'memory_stored': self.pathos.should_write_memory(salience),
                'tool_used': tool_call.tool_name if tool_call else None,
                'duration': cycle_duration
            }
            
        except Exception as e:
            self.instrumentation.record_error(type(e).__name__, str(e), 'agent_cycle')
            logger.error("Error in agent cycle", error=str(e), cycle=self.cycle_count)
            raise
    
    def run_autonomous(self, max_cycles: Optional[int] = None) -> None:
        """
        Run the agent autonomously for specified number of cycles.
        
        Args:
            max_cycles: Maximum number of cycles to run (None for infinite)
        """
        self.running = True
        max_cycles = max_cycles or self.config.max_cycles
        
        logger.info("Starting autonomous operation", max_cycles=max_cycles)
        
        try:
            while self.running and (max_cycles < 0 or self.cycle_count < max_cycles):
                cycle_result = self.run_cycle()
                
                logger.debug("Cycle completed", **cycle_result)
                
                # Sleep between cycles
                time.sleep(self.config.cycle_interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Autonomous operation interrupted by user")
        except Exception as e:
            logger.error("Fatal error in autonomous operation", error=str(e))
            raise
        finally:
            self.running = False
            logger.info("Autonomous operation stopped", total_cycles=self.cycle_count)
    
    def stop(self):
        """Stop autonomous operation"""
        self.running = False
        logger.info("Stop signal sent to agent")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        return {
            'running': self.running,
            'cycle_count': self.cycle_count,
            'current_state_norm': float(np.linalg.norm(self.pathos.current_state)) if self.pathos else 0.0,
            'memory_trace_count': self.memory.get_trace_count() if self.memory else 0,
            'metrics_summary': self.instrumentation.get_metrics_summary()
        }
    
    def _generate_fallback_intention(self) -> Intention:
        """Generate a safe fallback intention when Ethos rejects the primary intention"""
        fallback_semantic = SemanticVector(
            embedding=np.zeros(self.config.logos.semantic_dimension),
            intention_text="Reflect on current state and maintain stability",
            semantic_category="reflection"
        )
        
        return Intention(
            description="Maintain current state and reflect on internal dynamics",
            semantic_vector=fallback_semantic,
            priority=0.3,
            tool_candidates=[]
        )
    
    def _generate_thinking_prompt(self, semantic_vector, pathos_state, recent_memories):
        """Generate a human-readable thinking prompt showing agent's reasoning process."""
        state_magnitude = float(np.linalg.norm(pathos_state))
        
        # Analyze current state
        if state_magnitude > 0.8:
            energy_desc = "high energy and focused attention"
        elif state_magnitude > 0.4:
            energy_desc = "balanced and contemplative state"
        else:
            energy_desc = "calm and reflective mood"
        
        # Memory context
        memory_context = f"drawing from {len(recent_memories)} recent experiences" if recent_memories else "starting fresh"
        
        # Create thinking prompt
        thinking_prompt = (
            f"I am currently in a {energy_desc}. "
            f"Based on my {semantic_vector.semantic_category} mindset and {memory_context}, "
            f"I am considering: '{semantic_vector.intention_text}'. "
            f"This aligns with my current preference for {semantic_vector.semantic_category} activities."
        )
        
        return thinking_prompt
    
    def _generate_tool_reasoning(self, intention, tool_call):
        """Generate reasoning for why a specific tool was selected."""
        reasoning = (
            f"To accomplish '{intention.description[:60]}', I have determined that "
            f"the '{tool_call.tool_name}' tool is most appropriate. "
            f"This tool will help me {self._get_tool_purpose(tool_call.tool_name)} "
            f"which directly supports my current {intention.semantic_vector.semantic_category} objective."
        )
        return reasoning
    
    def _get_tool_purpose(self, tool_name):
        """Get a human-readable purpose for a tool."""
        tool_purposes = {
            'observe_environment': 'gather information about my current context',
            'analyze_data': 'process and understand available information',
            'create_content': 'generate new ideas or solutions',
            'communicate': 'share insights or interact with others',
            'file_operations': 'manage and organize information',
            'web_search': 'research and gather external knowledge',
            'memory_query': 'recall relevant past experiences'
        }
        
        # Extract base tool name (remove prefixes/suffixes)
        base_name = tool_name.lower()
        for key in tool_purposes:
            if key in base_name:
                return tool_purposes[key]
        
        return f"utilize the capabilities of {tool_name}"
    
    def _handle_tool_result(self, result: ToolResult) -> None:
        """
        Handle tool execution results for layer integration.
        
        Args:
            result: ToolResult from tool execution
        """
        # Update tool effectiveness in Logos layer
        if hasattr(self.logos, 'update_tool_effectiveness'):
            # Determine semantic category from current context
            # This is a simplified approach - in practice we'd track the context
            category = "general"  # Default category
            reward = 1.0 if result.success else -0.5
            self.logos.update_tool_effectiveness(result.tool_name, reward, category)
        
        logger.debug(f"Processed tool result for {result.tool_name}: success={result.success}")