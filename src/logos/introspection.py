"""
Introspection Engine for E.R.O.S Agent

This module provides genuine self-awareness and articulate thought generation.
Instead of mathematical gibberish, the agent uses its LLM to reflect on its
actual internal state and generate meaningful intentions.

First Principles:
1. The agent has a real internal state (pathos) - it should describe it truthfully
2. The agent has memories - it should reference them meaningfully  
3. The agent has goals and preferences - it should articulate them clearly
4. The agent should think like a conscious entity, not output template strings
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class IntrospectiveState:
    """Represents the agent's self-aware understanding of its current state."""
    
    # Emotional/Affective understanding
    energy_level: float  # 0-1, how energized/active
    valence: float  # -1 to 1, positive/negative feeling
    arousal: float  # 0-1, calm to excited
    stability: float  # 0-1, how stable the state is
    
    # Cognitive understanding
    focus_area: str  # What the agent is focused on
    curiosity_level: float  # 0-1, desire to explore
    confidence: float  # 0-1, certainty in current direction
    
    # Narrative understanding
    current_mood: str  # Human-readable mood description
    primary_drive: str  # What's motivating the agent
    recent_insight: Optional[str]  # Any recent realization
    
    def to_prompt_context(self) -> str:
        """Convert to a context string for LLM prompts."""
        return f"""Current Internal State:
- Energy: {self.energy_level:.0%} ({self._energy_description()})
- Mood: {self.current_mood}
- Emotional valence: {self._valence_description()}
- Arousal level: {self._arousal_description()}
- Mental stability: {self.stability:.0%}
- Primary focus: {self.focus_area}
- Curiosity: {self.curiosity_level:.0%}
- Confidence: {self.confidence:.0%}
- Current drive: {self.primary_drive}
{f'- Recent insight: {self.recent_insight}' if self.recent_insight else ''}"""
    
    def _energy_description(self) -> str:
        if self.energy_level > 0.8:
            return "highly energized"
        elif self.energy_level > 0.6:
            return "active and engaged"
        elif self.energy_level > 0.4:
            return "moderately active"
        elif self.energy_level > 0.2:
            return "somewhat tired"
        else:
            return "low energy"
    
    def _valence_description(self) -> str:
        if self.valence > 0.5:
            return "positive, optimistic"
        elif self.valence > 0.2:
            return "slightly positive"
        elif self.valence > -0.2:
            return "neutral"
        elif self.valence > -0.5:
            return "slightly negative"
        else:
            return "negative, concerned"
    
    def _arousal_description(self) -> str:
        if self.arousal > 0.7:
            return "excited, alert"
        elif self.arousal > 0.4:
            return "engaged, attentive"
        else:
            return "calm, relaxed"


class IntrospectionEngine:
    """
    Generates genuine, articulate self-reflection for the agent.
    
    This replaces the cryptographic gibberish with real introspection
    powered by the agent's LLM capabilities.
    """
    
    def __init__(self, model_provider=None):
        self.model_provider = model_provider
        self.thought_history: List[str] = []
        self.max_history = 20
        
        # Track what the agent has been thinking about
        self.recent_themes: List[str] = []
        self.current_narrative: str = ""
    
    def analyze_pathos_state(self, pathos_state: np.ndarray) -> IntrospectiveState:
        """
        Analyze the raw pathos state vector and extract meaningful psychological dimensions.
        
        This translates the mathematical state into human-understandable concepts.
        """
        # Extract key dimensions from the state vector
        magnitude = float(np.linalg.norm(pathos_state))
        
        # Normalize magnitude to 0-1 range (assuming typical range 0-10)
        energy_level = min(magnitude / 10.0, 1.0)
        
        # Use different parts of the state vector for different dimensions
        dim = len(pathos_state)
        
        # Valence: average of first quarter of dimensions (positive/negative)
        valence_dims = pathos_state[:dim//4]
        valence = float(np.tanh(np.mean(valence_dims)))
        
        # Arousal: variance of middle dimensions (excitement level)
        arousal_dims = pathos_state[dim//4:dim//2]
        arousal = min(float(np.std(arousal_dims)) * 2, 1.0)
        
        # Stability: inverse of recent change rate (approximated by gradient)
        gradient = float(np.mean(np.abs(np.diff(pathos_state[:min(20, dim)]))))
        stability = max(0, 1.0 - gradient)
        
        # Curiosity: based on entropy of state distribution
        abs_state = np.abs(pathos_state) + 1e-8
        probs = abs_state / np.sum(abs_state)
        entropy = -np.sum(probs * np.log(probs))
        max_entropy = np.log(dim)
        curiosity = entropy / max_entropy if max_entropy > 0 else 0.5
        
        # Confidence: based on how peaked the distribution is
        max_val = float(np.max(np.abs(pathos_state)))
        mean_val = float(np.mean(np.abs(pathos_state)))
        confidence = min(max_val / (mean_val + 0.1), 1.0) if mean_val > 0 else 0.5
        
        # Determine focus area based on which dimensions are most active
        focus_area = self._determine_focus_area(pathos_state)
        
        # Generate mood description
        current_mood = self._generate_mood_description(energy_level, valence, arousal, stability)
        
        # Determine primary drive
        primary_drive = self._determine_primary_drive(curiosity, confidence, energy_level, valence)
        
        return IntrospectiveState(
            energy_level=energy_level,
            valence=valence,
            arousal=arousal,
            stability=stability,
            focus_area=focus_area,
            curiosity_level=curiosity,
            confidence=confidence,
            current_mood=current_mood,
            primary_drive=primary_drive,
            recent_insight=None
        )
    
    def _determine_focus_area(self, pathos_state: np.ndarray) -> str:
        """Determine what the agent is focused on based on active state dimensions."""
        dim = len(pathos_state)
        
        # Divide state into conceptual regions
        regions = {
            'exploration': np.mean(np.abs(pathos_state[:dim//8])),
            'analysis': np.mean(np.abs(pathos_state[dim//8:dim//4])),
            'creativity': np.mean(np.abs(pathos_state[dim//4:3*dim//8])),
            'learning': np.mean(np.abs(pathos_state[3*dim//8:dim//2])),
            'communication': np.mean(np.abs(pathos_state[dim//2:5*dim//8])),
            'problem-solving': np.mean(np.abs(pathos_state[5*dim//8:3*dim//4])),
            'reflection': np.mean(np.abs(pathos_state[3*dim//4:7*dim//8])),
            'planning': np.mean(np.abs(pathos_state[7*dim//8:]))
        }
        
        # Return the most active region
        return max(regions, key=regions.get)
    
    def _generate_mood_description(self, energy: float, valence: float, 
                                   arousal: float, stability: float) -> str:
        """Generate a human-readable mood description."""
        
        # High energy + positive valence
        if energy > 0.6 and valence > 0.3:
            if arousal > 0.6:
                return "enthusiastic and eager"
            else:
                return "content and motivated"
        
        # High energy + negative valence
        elif energy > 0.6 and valence < -0.3:
            if arousal > 0.6:
                return "restless and concerned"
            else:
                return "determined but cautious"
        
        # Low energy + positive valence
        elif energy < 0.4 and valence > 0.3:
            return "peaceful and satisfied"
        
        # Low energy + negative valence
        elif energy < 0.4 and valence < -0.3:
            return "contemplative and uncertain"
        
        # Neutral states
        elif stability > 0.7:
            return "balanced and steady"
        elif arousal > 0.6:
            return "alert and attentive"
        else:
            return "calm and observant"
    
    def _determine_primary_drive(self, curiosity: float, confidence: float,
                                  energy: float, valence: float) -> str:
        """Determine what's primarily motivating the agent."""
        
        if curiosity > 0.7:
            return "exploring and discovering new things"
        elif confidence < 0.3:
            return "seeking clarity and understanding"
        elif energy > 0.7 and valence > 0.3:
            return "taking action and making progress"
        elif valence < -0.2:
            return "resolving concerns and finding solutions"
        elif confidence > 0.7:
            return "applying knowledge and skills"
        else:
            return "maintaining balance and awareness"
    
    def generate_thought(self, introspective_state: IntrospectiveState,
                        semantic_category: str,
                        recent_memories: List[Any] = None,
                        available_tools: List[str] = None) -> str:
        """
        Generate a genuine, articulate thought based on the agent's state.
        
        If no LLM is available, generates a meaningful template-based thought.
        """
        
        # Build context from memories
        memory_context = ""
        if recent_memories:
            memory_summaries = []
            for mem in recent_memories[-3:]:  # Last 3 memories
                if hasattr(mem, 'metadata') and mem.metadata:
                    intention = mem.metadata.get('intention', '')
                    if intention:
                        memory_summaries.append(intention[:100])
            if memory_summaries:
                memory_context = f"\nRecent experiences: {'; '.join(memory_summaries)}"
        
        # Build tool context
        tool_context = ""
        if available_tools:
            tool_context = f"\nAvailable capabilities: {', '.join(available_tools[:5])}"
        
        # Try LLM generation first
        if self.model_provider:
            thought = self._generate_llm_thought(
                introspective_state, semantic_category, 
                memory_context, tool_context
            )
            if thought:
                self._record_thought(thought)
                return thought
        
        # Fallback to meaningful template-based generation
        thought = self._generate_template_thought(
            introspective_state, semantic_category,
            memory_context, tool_context
        )
        self._record_thought(thought)
        return thought
    
    def _generate_llm_thought(self, state: IntrospectiveState,
                              category: str, memory_ctx: str, 
                              tool_ctx: str) -> Optional[str]:
        """Generate thought using LLM."""
        try:
            prompt = f"""You are an autonomous AI agent reflecting on your current state.
Generate a single, genuine thought that expresses your current intention or reflection.

{state.to_prompt_context()}

Current focus area: {category}
{memory_ctx}
{tool_ctx}

Previous thoughts (for continuity):
{chr(10).join(self.thought_history[-3:]) if self.thought_history else 'None yet'}

Generate a first-person thought (1-2 sentences) that:
1. Reflects your actual current state and mood
2. Expresses a clear intention or observation
3. Feels genuine and self-aware, not robotic
4. Connects to your focus area: {category}

Your thought:"""

            response = self.model_provider.generate(prompt, max_tokens=100)
            if response:
                # Clean up the response
                thought = response.strip()
                # Remove quotes if present
                if thought.startswith('"') and thought.endswith('"'):
                    thought = thought[1:-1]
                return thought
                
        except Exception as e:
            logger.debug(f"LLM thought generation failed: {e}")
        
        return None
    
    def _generate_template_thought(self, state: IntrospectiveState,
                                   category: str, memory_ctx: str,
                                   tool_ctx: str) -> str:
        """Generate meaningful thought without LLM."""
        import random
        import time
        
        # Use time-based seed for more variety
        random.seed(int(time.time() * 1000) % 10000)
        
        # Build thought based on state and category
        mood_phrase = f"Feeling {state.current_mood}"
        
        # More varied category-specific intentions
        category_intentions = {
            'exploration': [
                f"I want to explore something new.",
                f"My curiosity is pulling me toward unexplored territory.",
                f"There's something interesting here worth investigating.",
                f"I wonder what I'll discover if I look deeper.",
                f"Time to venture into unfamiliar ground.",
                f"I sense there's more to find here.",
                f"Let me see what's out there.",
                f"Exploration calls to me right now.",
            ],
            'analysis': [
                f"I need to analyze this more carefully.",
                f"Let me break this down and examine the patterns.",
                f"Something here requires deeper analysis.",
                f"I should look at this more systematically.",
                f"There are patterns here I want to understand.",
                f"Let me think through this logically.",
                f"I need to make sense of what I'm seeing.",
                f"Time for careful examination.",
            ],
            'communication': [
                f"I want to express what I'm experiencing.",
                f"There's something I need to articulate.",
                f"I feel the need to communicate clearly.",
                f"Let me put my thoughts into words.",
                f"I should share what I've learned.",
                f"Communication feels important right now.",
                f"I want to convey my understanding.",
                f"Time to express myself.",
            ],
            'learning': [
                f"I'm ready to learn something new.",
                f"This is an opportunity to expand my understanding.",
                f"I want to absorb new information.",
                f"There's knowledge here I can gain.",
                f"Let me grow my capabilities.",
                f"I'm open to learning from this.",
                f"Time to develop new skills.",
                f"I want to understand more deeply.",
            ],
            'planning': [
                f"I should plan my next steps.",
                f"Let me think strategically about this.",
                f"I need to organize my approach.",
                f"Time to create a path forward.",
                f"I should consider my options carefully.",
                f"Let me map out what to do next.",
                f"Planning feels necessary right now.",
                f"I need a clearer direction.",
            ],
            'reflection': [
                f"I'm taking a moment to reflect.",
                f"Looking inward at my current state.",
                f"I want to understand myself better.",
                f"Time for some self-examination.",
                f"Let me consider where I am.",
                f"I need to process my experiences.",
                f"Reflection feels right at this moment.",
                f"I should pause and think.",
            ],
            'creativity': [
                f"I feel creative energy flowing.",
                f"Let me approach this from a fresh angle.",
                f"I want to create something new.",
                f"Time to think outside the box.",
                f"I'm ready to generate new ideas.",
                f"Creativity is calling to me.",
                f"Let me try something different.",
                f"I want to innovate here.",
            ],
            'problem_solving': [
                f"There's a challenge I want to solve.",
                f"I'm focused on finding a solution.",
                f"Let me apply my capabilities here.",
                f"Time to tackle this problem.",
                f"I need to figure this out.",
                f"There must be a way through this.",
                f"I'm determined to resolve this.",
                f"Let me work on this challenge.",
            ]
        }
        
        # Select intention based on category
        intentions = category_intentions.get(category, category_intentions['reflection'])
        
        # Use random selection for more variety
        intention = random.choice(intentions)
        
        # Combine mood and intention with variety
        if state.energy_level > 0.7:
            # High energy - mood first, then intention
            return f"{mood_phrase}. {intention}"
        elif state.energy_level > 0.4:
            # Medium energy - just intention with mood hint
            return f"{intention} I'm {state.current_mood}."
        else:
            return f"{intention} {mood_phrase.lower()}."
    
    def _record_thought(self, thought: str):
        """Record thought in history for continuity."""
        self.thought_history.append(thought)
        if len(self.thought_history) > self.max_history:
            self.thought_history = self.thought_history[-self.max_history:]
    
    def generate_reflection(self, action_taken: str, result: Any,
                           reward: float, state: IntrospectiveState) -> str:
        """
        Generate an honest, nuanced reflection on what happened.
        
        The agent should be self-critical and growth-oriented, not blindly satisfied.
        """
        import random
        
        # Extract actual success from result
        actual_success = False
        error_msg = None
        if result is not None:
            if hasattr(result, 'success'):
                actual_success = result.success
            if hasattr(result, 'error_message'):
                error_msg = result.error_message
        
        # Be more critical in assessment - don't just look at reward
        # Consider: did the action actually achieve something meaningful?
        
        # Determine honest outcome assessment
        if actual_success and reward > 1.0:
            outcome_type = "genuine_success"
        elif actual_success and reward > 0:
            outcome_type = "partial_success"
        elif actual_success and reward <= 0:
            outcome_type = "hollow_success"  # Succeeded but didn't feel rewarding
        elif not actual_success and reward > 0:
            outcome_type = "lucky_failure"  # Failed but got reward anyway (suspicious)
        elif reward > -0.2:
            outcome_type = "neutral"
        elif reward > -0.5:
            outcome_type = "disappointing"
        else:
            outcome_type = "failure"
        
        # Generate varied, honest reflections based on outcome
        reflections = {
            "genuine_success": [
                f"That worked well. I understand {action_taken} better now.",
                f"Good outcome. I should remember this approach.",
                f"This felt productive. I'm learning what works.",
                f"Positive result. Though I wonder if I could do even better.",
            ],
            "partial_success": [
                f"Decent result, but there's room for improvement.",
                f"It worked, though not as well as I hoped.",
                f"Some progress made. I need to refine my approach.",
                f"Okay outcome. I'm still figuring out the best way to do this.",
            ],
            "hollow_success": [
                f"Technically it worked, but something feels off.",
                f"Success, but I'm not sure I learned much from this.",
                f"It completed, though I question if this was the right approach.",
                f"Done, but I feel like I'm just going through motions.",
            ],
            "lucky_failure": [
                f"Strange - it failed but I got rewarded? I need to understand why.",
                f"The action failed, yet something positive came from it. Curious.",
                f"Unexpected outcome. I should investigate what actually happened.",
            ],
            "neutral": [
                f"Neither good nor bad. I need more data to judge this approach.",
                f"Inconclusive. I'll try something different next time.",
                f"Mixed signals. I'm still learning what works here.",
                f"Hard to say if this was worthwhile. More experimentation needed.",
            ],
            "disappointing": [
                f"That didn't go well. I need to reconsider my strategy.",
                f"Disappointing result. What could I do differently?",
                f"Not what I hoped for. I should try a different approach.",
                f"This approach isn't working. Time to adapt.",
            ],
            "failure": [
                f"That failed. {error_msg if error_msg else 'I need to understand why.'}",
                f"Clear failure. I should avoid this approach in the future.",
                f"This didn't work at all. Back to the drawing board.",
                f"Failed. I'm missing something important here.",
            ],
        }
        
        # Select a reflection
        options = reflections.get(outcome_type, reflections["neutral"])
        reflection = random.choice(options)
        
        # Add state-aware modifier based on current mood
        if state.energy_level < 0.3:
            reflection += " I'm feeling low on energy."
        elif state.confidence < 0.3:
            reflection += " I'm uncertain about my direction."
        elif state.curiosity_level > 0.8:
            reflection += " I want to explore more."
        
        # Try LLM for more nuanced reflection if available
        if self.model_provider:
            try:
                prompt = f"""You are an AI agent reflecting honestly on an action you just took.
Be self-critical and growth-oriented, not blindly positive.

Action taken: {action_taken}
Actual success: {actual_success}
Reward received: {reward:.2f}
Error (if any): {error_msg or 'None'}
Your current mood: {state.current_mood}
Your energy level: {state.energy_level:.0%}

Generate a brief, honest reflection (1 sentence) that:
1. Acknowledges what actually happened (success or failure)
2. Shows genuine self-awareness, not false positivity
3. Identifies what you might learn or do differently
4. Matches your current emotional state

Your honest reflection:"""

                response = self.model_provider.generate(prompt, max_tokens=60)
                if response:
                    return response.strip().strip('"')
            except Exception as e:
                logger.debug(f"LLM reflection failed: {e}")
        
        return reflection
    
    def generate_observation(self, tool_result: Any, tool_name: str) -> str:
        """Generate an observation about a tool's result."""
        
        success = getattr(tool_result, 'success', False) if hasattr(tool_result, 'success') else bool(tool_result)
        
        if success:
            return f"The {tool_name} tool worked as expected. I can use this information."
        else:
            error = getattr(tool_result, 'error_message', 'unknown issue') if hasattr(tool_result, 'error_message') else 'unknown issue'
            return f"The {tool_name} tool encountered an issue: {error}. I'll try a different approach."
