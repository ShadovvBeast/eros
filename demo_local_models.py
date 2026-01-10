#!/usr/bin/env python3
"""
Demo: Using Local Models with E.R.O.S

This demo shows how to configure and use local models (Ollama, vLLM, llama.cpp)
as alternatives to Gemini for the Logos reasoning layer.
"""

import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def demo_ollama_setup():
    """Demo setting up Ollama for local model inference"""
    print("🦙 Ollama Setup Demo")
    print("=" * 50)
    
    from src.core.config import LogosConfig
    from src.logos.model_providers import ModelProviderFactory
    
    # Configure for Ollama
    config = LogosConfig()
    config.model_provider = 'ollama'
    config.ollama_base_url = 'http://localhost:11434'
    config.ollama_model = 'qwen2.5:7b'
    config.ollama_temperature = 0.7
    config.ollama_timeout = 60
    
    print(f"✅ Configured for Ollama:")
    print(f"   - Base URL: {config.ollama_base_url}")
    print(f"   - Model: {config.ollama_model}")
    print(f"   - Temperature: {config.ollama_temperature}")
    
    # Create provider
    provider = ModelProviderFactory.create_provider('ollama', config)
    
    # Check availability
    if provider.is_available():
        print("🟢 Ollama is available and ready!")
        
        # Test generation
        try:
            response = provider.generate_response("Hello, how are you?")
            if response:
                print(f"🤖 Model response: {response[:100]}...")
            else:
                print("❌ No response generated")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    else:
        print("🔴 Ollama is not available")
        print("💡 To set up Ollama:")
        print("   1. Install: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. Start server: ollama serve")
        print("   3. Pull model: ollama pull qwen2.5:7b")


def demo_vllm_setup():
    """Demo setting up vLLM for local model inference"""
    print("\n🚀 vLLM Setup Demo")
    print("=" * 50)
    
    from src.core.config import LogosConfig
    from src.logos.model_providers import ModelProviderFactory
    
    # Configure for vLLM (OpenAI-compatible)
    config = LogosConfig()
    config.model_provider = 'openai_compatible'
    config.openai_compatible_base_url = 'http://localhost:8000'
    config.openai_compatible_model = 'Qwen/Qwen2.5-7B-Instruct'
    config.openai_compatible_temperature = 0.7
    config.openai_compatible_timeout = 60
    
    print(f"✅ Configured for vLLM:")
    print(f"   - Base URL: {config.openai_compatible_base_url}")
    print(f"   - Model: {config.openai_compatible_model}")
    print(f"   - Temperature: {config.openai_compatible_temperature}")
    
    # Create provider
    provider = ModelProviderFactory.create_provider('openai_compatible', config)
    
    # Check availability
    if provider.is_available():
        print("🟢 vLLM is available and ready!")
        
        # Test generation
        try:
            response = provider.generate_response("Explain quantum computing in simple terms.")
            if response:
                print(f"🤖 Model response: {response[:100]}...")
            else:
                print("❌ No response generated")
        except Exception as e:
            print(f"❌ Generation failed: {e}")
    else:
        print("🔴 vLLM is not available")
        print("💡 To set up vLLM:")
        print("   1. Install: pip install vllm")
        print("   2. Start server: python -m vllm.entrypoints.openai.api_server \\")
        print("      --model Qwen/Qwen2.5-7B-Instruct --port 8000")


def demo_logos_layer_with_local_model():
    """Demo using LogosLayer with a local model"""
    print("\n🧠 LogosLayer with Local Model Demo")
    print("=" * 50)
    
    from src.core.config import LogosConfig
    from src.logos.logos_layer import LogosLayer
    import numpy as np
    
    # Configure for local model
    config = LogosConfig()
    config.model_provider = 'ollama'  # Switch to local model
    config.ollama_base_url = 'http://localhost:11434'
    config.ollama_model = 'qwen2.5:7b'
    config.semantic_dimension = 256  # Smaller for demo
    
    print(f"✅ Creating LogosLayer with {config.model_provider} provider")
    
    # Create LogosLayer
    logos = LogosLayer(config)
    
    print(f"✅ LogosLayer initialized")
    print(f"   - Provider: {logos.config.model_provider}")
    print(f"   - Model: {logos.config.ollama_model}")
    print(f"   - Semantic dimension: {logos.config.semantic_dimension}")
    
    # Test semantic interpretation (without actual model call)
    print("\n🔍 Testing semantic interpretation...")
    
    # Create mock pathos state
    pathos_state = np.random.rand(64) * 0.5  # Small random state
    
    try:
        # This will create semantic vector without calling the model
        semantic_vector = logos.interpret_state(pathos_state, [])
        print(f"✅ Semantic interpretation successful")
        print(f"   - Category: {semantic_vector.semantic_category}")
        print(f"   - Intention: {semantic_vector.intention_text[:60]}...")
    except Exception as e:
        print(f"❌ Semantic interpretation failed: {e}")


def demo_config_presets():
    """Demo configuration presets for local models"""
    print("\n⚙️ Configuration Presets Demo")
    print("=" * 50)
    
    from src.core.config_presets import (
        create_development_config,
        create_local_development_config, 
        create_offline_config
    )
    
    # Standard development config (Gemini)
    dev_config = create_development_config()
    print(f"🔧 Development Config:")
    print(f"   - Provider: {dev_config.logos.model_provider}")
    print(f"   - Model: {dev_config.logos.gemini_model}")
    
    # Local development config (Ollama)
    local_config = create_local_development_config()
    print(f"\n🏠 Local Development Config:")
    print(f"   - Provider: {local_config.logos.model_provider}")
    print(f"   - Model: {local_config.logos.ollama_model}")
    print(f"   - Base URL: {local_config.logos.ollama_base_url}")
    print(f"   - Timeout: {local_config.logos.ollama_timeout}s")
    
    # Offline config (Ollama with longer timeouts)
    offline_config = create_offline_config()
    print(f"\n📴 Offline Config:")
    print(f"   - Provider: {offline_config.logos.model_provider}")
    print(f"   - Model: {offline_config.logos.ollama_model}")
    print(f"   - Timeout: {offline_config.logos.ollama_timeout}s")
    print(f"   - Storage: {offline_config.memory.storage_backend}")


def demo_gui_integration():
    """Demo GUI integration for local models"""
    print("\n🖥️ GUI Integration Demo")
    print("=" * 50)
    
    print("✅ The GUI ConfigTab now includes:")
    print("   - Model Provider dropdown (gemini, ollama, openai_compatible)")
    print("   - Provider-specific configuration sections")
    print("   - Real-time provider testing")
    print("   - Dynamic show/hide of configuration options")
    
    print("\n💡 To use the GUI:")
    print("   1. Run: python main.py gui")
    print("   2. Go to Configuration tab")
    print("   3. Select 'ollama' or 'openai_compatible' from dropdown")
    print("   4. Configure the provider settings")
    print("   5. Click 'Test Provider' to verify connection")
    print("   6. Save configuration")


def main():
    """Run all demos"""
    print("🎯 E.R.O.S Local Model Integration Demo")
    print("=" * 60)
    
    demo_ollama_setup()
    demo_vllm_setup()
    demo_logos_layer_with_local_model()
    demo_config_presets()
    demo_gui_integration()
    
    print("\n" + "=" * 60)
    print("🎉 Demo Complete!")
    print("\n📚 Key Benefits of Local Model Integration:")
    print("   ✅ Privacy: All inference happens locally")
    print("   ✅ Cost: No API costs for model usage")
    print("   ✅ Speed: No network latency for inference")
    print("   ✅ Offline: Works without internet connection")
    print("   ✅ Control: Full control over model and parameters")
    
    print("\n🚀 Recommended Local Models:")
    print("   - Qwen2.5:7B - Excellent reasoning, moderate size")
    print("   - Qwen2.5:14B - Better performance, larger size")
    print("   - Llama3.2:8B - Good alternative, Meta model")
    print("   - Mistral:7B - Fast inference, good quality")
    
    print("\n⚡ Quick Start:")
    print("   1. Install Ollama: curl -fsSL https://ollama.ai/install.sh | sh")
    print("   2. Pull model: ollama pull qwen2.5:7b")
    print("   3. Start server: ollama serve")
    print("   4. Run E.R.O.S: python main.py gui")
    print("   5. Select 'ollama' provider in Configuration tab")


if __name__ == "__main__":
    main()