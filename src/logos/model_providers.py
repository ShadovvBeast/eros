"""
Model provider abstraction for the Logos layer.

Supports multiple model providers including Gemini, local models via Ollama,
llama.cpp, vLLM, and other OpenAI-compatible APIs.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import requests
import json
import time

logger = logging.getLogger(__name__)


class ModelProvider(ABC):
    """Abstract base class for model providers"""
    
    def __init__(self, logos_config):
        """Initialize with LogosConfig directly"""
        self.logos_config = logos_config
        self.name = self.__class__.__name__
    
    @abstractmethod
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Generate response from the model"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the model provider is available"""
        pass
    
    @property
    @abstractmethod
    def provider_type(self) -> str:
        """Return the provider type identifier"""
        pass


class GeminiProvider(ModelProvider):
    """Google Gemini model provider"""
    
    @property
    def provider_type(self) -> str:
        return "gemini"
    
    def is_available(self) -> bool:
        """Check if Gemini API is available"""
        try:
            api_key = self.logos_config.gemini_api_key
            return bool(api_key and api_key.strip())
        except Exception:
            return False
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Generate response using Gemini API"""
        try:
            from google import genai
            from google.genai import types
            
            # Configure API key
            api_key = self.logos_config.gemini_api_key
            if not api_key:
                logger.error("Gemini API key not configured")
                return None
            
            client = genai.Client(api_key=api_key)
            model_name = self.logos_config.gemini_model
            
            # Create system instruction if context provided
            system_instruction = None
            if context and context.get('system_instruction'):
                system_instruction = context['system_instruction']
            
            # Generate response
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=self.logos_config.gemini_temperature,
                    top_p=self.logos_config.gemini_top_p,
                    max_output_tokens=self.logos_config.gemini_max_tokens
                )
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            return None


class OllamaProvider(ModelProvider):
    """Ollama local model provider"""
    
    @property
    def provider_type(self) -> str:
        return "ollama"
    
    def is_available(self) -> bool:
        """Check if Ollama is available"""
        try:
            base_url = self.logos_config.ollama_base_url
            response = requests.get(f"{base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Generate response using Ollama API"""
        try:
            base_url = self.logos_config.ollama_base_url
            model_name = self.logos_config.ollama_model
            
            # Prepare request
            data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.logos_config.ollama_temperature,
                    "top_p": self.logos_config.ollama_top_p,
                    "num_predict": self.logos_config.ollama_max_tokens
                }
            }
            
            # Add system message if provided
            if context and context.get('system_instruction'):
                data["system"] = context['system_instruction']
            
            response = requests.post(
                f"{base_url}/api/generate",
                json=data,
                timeout=self.logos_config.ollama_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Ollama API error: {e}")
            return None


class OpenAICompatibleProvider(ModelProvider):
    """OpenAI-compatible API provider (vLLM, llama.cpp server, etc.)"""
    
    @property
    def provider_type(self) -> str:
        return "openai_compatible"
    
    def is_available(self) -> bool:
        """Check if the OpenAI-compatible API is available"""
        try:
            base_url = self.logos_config.openai_compatible_base_url
            response = requests.get(f"{base_url}/v1/models", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> Optional[str]:
        """Generate response using OpenAI-compatible API"""
        try:
            base_url = self.logos_config.openai_compatible_base_url
            model_name = self.logos_config.openai_compatible_model
            
            # Prepare messages
            messages = []
            if context and context.get('system_instruction'):
                messages.append({"role": "system", "content": context['system_instruction']})
            messages.append({"role": "user", "content": prompt})
            
            # Prepare request
            data = {
                "model": model_name,
                "messages": messages,
                "temperature": self.logos_config.openai_compatible_temperature,
                "top_p": self.logos_config.openai_compatible_top_p,
                "max_tokens": self.logos_config.openai_compatible_max_tokens
            }
            
            headers = {
                "Content-Type": "application/json"
            }
            
            # Add API key if provided
            api_key = self.logos_config.openai_compatible_api_key
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
            
            response = requests.post(
                f"{base_url}/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=self.logos_config.openai_compatible_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"OpenAI-compatible API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"OpenAI-compatible API error: {e}")
            return None


class ModelProviderFactory:
    """Factory for creating model providers"""
    
    _providers = {
        'gemini': GeminiProvider,
        'ollama': OllamaProvider,
        'openai_compatible': OpenAICompatibleProvider,
    }
    
    @classmethod
    def create_provider(cls, provider_type: str, logos_config) -> Optional[ModelProvider]:
        """Create a model provider instance using LogosConfig directly"""
        provider_class = cls._providers.get(provider_type)
        if provider_class:
            return provider_class(logos_config)
        else:
            logger.error(f"Unknown provider type: {provider_type}")
            return None
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider types"""
        return list(cls._providers.keys())