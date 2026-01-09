"""
Prompt Debugging System

Provides functionality to intercept and review prompts before they are sent to external LLM services.
"""

import os
import threading
from typing import Dict, Any, Optional, Callable
from datetime import datetime


class PromptDebugger:
    """
    Global prompt debugging system that can intercept and review prompts.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.initialized = True
            self.review_callback: Optional[Callable[[Dict[str, Any]], bool]] = None
            self.is_debugging_enabled = False
            self.intercepted_prompts = []
    
    def set_review_callback(self, callback: Callable[[Dict[str, Any]], bool]):
        """
        Set the callback function for prompt review.
        
        Args:
            callback: Function that takes prompt data and returns True if approved
        """
        self.review_callback = callback
    
    def is_enabled(self) -> bool:
        """Check if prompt debugging is currently enabled."""
        # Check environment variable first
        env_enabled = os.getenv('CYCLE_LOGOS_DEBUGGING', 'false').lower() == 'true'
        return env_enabled or self.is_debugging_enabled
    
    def enable_debugging(self):
        """Enable prompt debugging."""
        self.is_debugging_enabled = True
        os.environ['CYCLE_LOGOS_DEBUGGING'] = 'true'
    
    def disable_debugging(self):
        """Disable prompt debugging."""
        self.is_debugging_enabled = False
        if 'CYCLE_LOGOS_DEBUGGING' in os.environ:
            del os.environ['CYCLE_LOGOS_DEBUGGING']
    
    def intercept_prompt(self, prompt_data: Dict[str, Any]) -> bool:
        """
        Intercept a prompt and request user review if debugging is enabled.
        
        Args:
            prompt_data: Dictionary containing prompt information including:
                - prompt_text: The actual prompt to be sent
                - cycle: Current cycle number
                - semantic_category: Category of the semantic vector
                - intention: Current intention description
                - state_magnitude: Magnitude of the pathos state
                - memory_context: Context from memories
                - tool_candidates: List of tool candidates
                - priority: Intention priority
                - additional_context: Any additional context
        
        Returns:
            True if prompt is approved (or debugging disabled), False if rejected
        """
        if not self.is_enabled():
            return True  # Allow prompt if debugging is disabled
        
        # Add timestamp to prompt data
        prompt_data['timestamp'] = datetime.now().isoformat()
        
        # Store the intercepted prompt
        self.intercepted_prompts.append(prompt_data.copy())
        
        # Keep only recent prompts (last 100)
        if len(self.intercepted_prompts) > 100:
            self.intercepted_prompts = self.intercepted_prompts[-100:]
        
        # If no review callback is set, default to approval with warning
        if self.review_callback is None:
            print(f"WARNING: Prompt debugging enabled but no review callback set. Auto-approving prompt.")
            return True
        
        # Call the review callback
        try:
            return self.review_callback(prompt_data)
        except Exception as e:
            print(f"ERROR in prompt review callback: {e}")
            return False  # Default to rejection on error
    
    def get_intercepted_prompts(self) -> list:
        """Get list of all intercepted prompts."""
        return self.intercepted_prompts.copy()
    
    def clear_intercepted_prompts(self):
        """Clear the list of intercepted prompts."""
        self.intercepted_prompts.clear()


# Global instance
prompt_debugger = PromptDebugger()


def intercept_gemini_prompt(prompt_text: str, context: Dict[str, Any] = None) -> bool:
    """
    Convenience function to intercept a Gemini prompt.
    
    Args:
        prompt_text: The prompt text to be sent to Gemini
        context: Additional context information
    
    Returns:
        True if prompt is approved, False if rejected
    """
    prompt_data = {
        'prompt_text': prompt_text,
        'service': 'gemini',
        **(context or {})
    }
    
    return prompt_debugger.intercept_prompt(prompt_data)


def is_prompt_debugging_enabled() -> bool:
    """Check if prompt debugging is currently enabled."""
    return prompt_debugger.is_enabled()


def set_prompt_review_callback(callback: Callable[[Dict[str, Any]], bool]):
    """Set the callback function for prompt review."""
    prompt_debugger.set_review_callback(callback)