"""
Configuration Tab

Interactive configuration interface for all agent settings.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import json
import os
from typing import Dict, Any, Optional
from dataclasses import asdict
import threading

try:
    from ...core.config import AgentConfig, PathosConfig, EthosConfig, LogosConfig, MemoryConfig, ToolConfig, AutonomousRewardConfig
    from ...core.config_manager import get_config_manager
    from ...core.config_presets import list_available_presets, get_preset_config
except ImportError:
    try:
        from core.config import AgentConfig, PathosConfig, EthosConfig, LogosConfig, MemoryConfig, ToolConfig, AutonomousRewardConfig
        from core.config_manager import get_config_manager
        from core.config_presets import list_available_presets, get_preset_config
    except ImportError:
        # Mock for testing
        class AgentConfig:
            def __init__(self):
                pass
            @classmethod
            def from_env(cls):
                return cls()
            def validate_configuration(self):
                return []
        
        def get_config_manager():
            class MockConfigManager:
                def get_current_config(self):
                    return AgentConfig()
                def save_config(self, config, name):
                    return True
                def list_profiles(self):
                    return []
            return MockConfigManager()
        
        def list_available_presets():
            return {'development': 'Dev config', 'production': 'Prod config'}
        
        def get_preset_config(name):
            return AgentConfig()

from .base_tab import BaseTab


class ConfigTab(BaseTab):
    """Interactive configuration tab for all agent settings."""
    
    def __init__(self, notebook, session_manager=None):
        """Initialize configuration tab."""
        self.session_manager = session_manager
        self.config_manager = get_config_manager()
        self.config = self.config_manager.get_current_config() or AgentConfig.from_env()
        self.config_widgets = {}
        self.validation_errors = []
        
        super().__init__(notebook, "⚙️ Configuration", "⚙️")
        self._create_config_interface()
        self._load_current_config()
    
    def _create_config_interface(self):
        """Create the configuration interface."""
        # Main container with scrollbar
        canvas = tk.Canvas(self.frame)
        scrollbar = ttk.Scrollbar(self.frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollbar and canvas
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Create configuration sections
        self._create_header_section(scrollable_frame)
        self._create_pathos_section(scrollable_frame)
        self._create_ethos_section(scrollable_frame)
        self._create_logos_section(scrollable_frame)
        self._create_memory_section(scrollable_frame)
        self._create_tools_section(scrollable_frame)
        self._create_autonomous_reward_section(scrollable_frame)
        self._create_global_section(scrollable_frame)
        self._create_action_section(scrollable_frame)
        
        # Bind mousewheel to canvas
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
    
    def _create_header_section(self, parent):
        """Create header section with title and status."""
        header_frame = ttk.LabelFrame(parent, text="Configuration Manager", padding="10")
        header_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Title and description
        title_label = ttk.Label(header_frame, text="🎯 Central Configuration Hub", font=('Arial', 14, 'bold'))
        title_label.pack(anchor=tk.W)
        
        desc_label = ttk.Label(header_frame, 
                              text="Configure all aspects of the autonomous agent system from this central location. Changes are applied immediately.",
                              font=('Arial', 9))
        desc_label.pack(anchor=tk.W, pady=(0, 5))
        
        # Note about centralized config
        note_label = ttk.Label(header_frame, 
                              text="💡 This is the single source of truth for all agent configuration. Use presets for quick setup or customize individual settings.",
                              font=('Arial', 8), foreground="blue")
        note_label.pack(anchor=tk.W, pady=(0, 10))
        
        # Status indicator
        self.status_frame = ttk.Frame(header_frame)
        self.status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(self.status_frame, text="✅ Configuration Valid", foreground="green")
        self.status_label.pack(side=tk.LEFT)
        
        # Validation button
        validate_btn = ttk.Button(self.status_frame, text="Validate Config", command=self._validate_config)
        validate_btn.pack(side=tk.RIGHT, padx=(0, 10))
        
        # Preset selector
        preset_frame = ttk.Frame(self.status_frame)
        preset_frame.pack(side=tk.RIGHT, padx=(0, 10))
        
        ttk.Label(preset_frame, text="Presets:").pack(side=tk.LEFT)
        self.preset_combo = ttk.Combobox(preset_frame, width=15, values=list(list_available_presets().keys()))
        self.preset_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.preset_combo.bind('<<ComboboxSelected>>', self._on_preset_selected)
        
        load_preset_btn = ttk.Button(preset_frame, text="Load", command=self._load_preset)
        load_preset_btn.pack(side=tk.LEFT, padx=(5, 0))
    
    def _create_pathos_section(self, parent):
        """Create Pathos configuration section."""
        pathos_frame = ttk.LabelFrame(parent, text="💭 Pathos Layer (Affective Dynamics)", padding="10")
        pathos_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # State dimension
        ttk.Label(pathos_frame, text="State Dimension:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.config_widgets['pathos_state_dimension'] = tk.IntVar(value=128)
        ttk.Spinbox(pathos_frame, from_=64, to=512, textvariable=self.config_widgets['pathos_state_dimension'], width=10).grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Decay factor
        ttk.Label(pathos_frame, text="Decay Factor (α):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.config_widgets['pathos_decay_factor'] = tk.DoubleVar(value=0.95)
        ttk.Scale(pathos_frame, from_=0.1, to=1.0, variable=self.config_widgets['pathos_decay_factor'], orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        decay_label = ttk.Label(pathos_frame, text="0.95")
        decay_label.grid(row=1, column=2, padx=(5, 0), pady=2)
        self.config_widgets['pathos_decay_factor'].trace('w', lambda *args: decay_label.config(text=f"{self.config_widgets['pathos_decay_factor'].get():.2f}"))
        
        # Echo strength
        ttk.Label(pathos_frame, text="Echo Strength (β):").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.config_widgets['pathos_echo_strength'] = tk.DoubleVar(value=0.1)
        ttk.Scale(pathos_frame, from_=0.0, to=0.5, variable=self.config_widgets['pathos_echo_strength'], orient=tk.HORIZONTAL, length=200).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        echo_label = ttk.Label(pathos_frame, text="0.10")
        echo_label.grid(row=2, column=2, padx=(5, 0), pady=2)
        self.config_widgets['pathos_echo_strength'].trace('w', lambda *args: echo_label.config(text=f"{self.config_widgets['pathos_echo_strength'].get():.2f}"))
        
        # Salience threshold
        ttk.Label(pathos_frame, text="Salience Threshold:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.config_widgets['pathos_salience_threshold'] = tk.DoubleVar(value=0.4)
        ttk.Scale(pathos_frame, from_=0.1, to=1.0, variable=self.config_widgets['pathos_salience_threshold'], orient=tk.HORIZONTAL, length=200).grid(row=3, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        salience_label = ttk.Label(pathos_frame, text="0.40")
        salience_label.grid(row=3, column=2, padx=(5, 0), pady=2)
        self.config_widgets['pathos_salience_threshold'].trace('w', lambda *args: salience_label.config(text=f"{self.config_widgets['pathos_salience_threshold'].get():.2f}"))
    
    def _create_ethos_section(self, parent):
        """Create Ethos configuration section."""
        ethos_frame = ttk.LabelFrame(parent, text="🎯 Ethos Framework (Identity & Values)", padding="10")
        ethos_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Identity description
        ttk.Label(ethos_frame, text="Identity Description:").grid(row=0, column=0, sticky=tk.NW, pady=2)
        self.config_widgets['ethos_identity'] = tk.Text(ethos_frame, width=60, height=3)
        self.config_widgets['ethos_identity'].grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Core values
        ttk.Label(ethos_frame, text="Core Values:").grid(row=1, column=0, sticky=tk.NW, pady=2)
        values_frame = ttk.Frame(ethos_frame)
        values_frame.grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        self.config_widgets['ethos_values'] = tk.Text(values_frame, width=60, height=3)
        self.config_widgets['ethos_values'].pack()
        ttk.Label(values_frame, text="(One value per line)", font=('Arial', 8)).pack(anchor=tk.W)
        
        # Unrestricted mode
        self.config_widgets['ethos_unrestricted'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(ethos_frame, text="Unrestricted Autonomous Mode", 
                       variable=self.config_widgets['ethos_unrestricted']).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
    
    def _create_logos_section(self, parent):
        """Create Logos layer configuration section."""
        logos_frame = ttk.LabelFrame(parent, text="🧠 Logos Layer (Reasoning)", padding="10")
        logos_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model provider selection
        ttk.Label(logos_frame, text="Model Provider:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.config_widgets['logos_model_provider'] = ttk.Combobox(
            logos_frame, 
            width=20, 
            values=['gemini', 'ollama', 'openai_compatible'],
            state='readonly'
        )
        self.config_widgets['logos_model_provider'].grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        self.config_widgets['logos_model_provider'].bind('<<ComboboxSelected>>', self._on_provider_changed)
        
        # Provider status
        self.provider_status_label = ttk.Label(logos_frame, text="⚪ Not tested", font=('Arial', 8))
        self.provider_status_label.grid(row=0, column=2, padx=(10, 0), pady=2)
        
        # Test provider button
        test_btn = ttk.Button(logos_frame, text="Test Provider", command=self._test_current_provider)
        test_btn.grid(row=0, column=3, padx=(10, 0), pady=2)
        
        # Provider-specific configurations (data-driven)
        self.provider_configs = {
            'gemini': {
                'title': 'Gemini Configuration',
                'fields': [
                    {'key': 'logos_gemini_api_key', 'label': 'API Key', 'type': 'entry', 'show': '*', 'width': 40},
                    {'key': 'logos_gemini_model', 'label': 'Model', 'type': 'combobox', 'width': 30,
                     'values': ['gemini-2.5-flash', 'gemini-2.0-flash-exp', 'gemini-1.5-pro', 'gemini-1.5-flash']},
                    {'key': 'logos_gemini_temperature', 'label': 'Temperature', 'type': 'scale', 'from': 0.0, 'to': 2.0, 'default': 0.7},
                    {'key': 'logos_gemini_top_p', 'label': 'Top P', 'type': 'scale', 'from': 0.0, 'to': 1.0, 'default': 0.9},
                    {'key': 'logos_gemini_max_tokens', 'label': 'Max Tokens', 'type': 'spinbox', 'from': 100, 'to': 8192, 'default': 2048}
                ]
            },
            'ollama': {
                'title': 'Ollama Configuration',
                'fields': [
                    {'key': 'logos_ollama_base_url', 'label': 'Base URL', 'type': 'entry', 'width': 40, 'default': 'http://localhost:11434'},
                    {'key': 'logos_ollama_model', 'label': 'Model', 'type': 'combobox', 'width': 30,
                     'values': ['qwen2.5:7b', 'qwen2.5:14b', 'qwen2.5:32b', 'llama3.2:3b', 'llama3.2:8b', 'mistral:7b']},
                    {'key': 'logos_ollama_temperature', 'label': 'Temperature', 'type': 'scale', 'from': 0.0, 'to': 2.0, 'default': 0.7},
                    {'key': 'logos_ollama_top_p', 'label': 'Top P', 'type': 'scale', 'from': 0.0, 'to': 1.0, 'default': 0.9},
                    {'key': 'logos_ollama_max_tokens', 'label': 'Max Tokens', 'type': 'spinbox', 'from': 100, 'to': 8192, 'default': 2048},
                    {'key': 'logos_ollama_timeout', 'label': 'Timeout (s)', 'type': 'spinbox', 'from': 10, 'to': 300, 'default': 60}
                ]
            },
            'openai_compatible': {
                'title': 'OpenAI-Compatible Configuration (vLLM, llama.cpp)',
                'fields': [
                    {'key': 'logos_openai_compatible_base_url', 'label': 'Base URL', 'type': 'entry', 'width': 40, 'default': 'http://localhost:8000'},
                    {'key': 'logos_openai_compatible_model', 'label': 'Model', 'type': 'entry', 'width': 30, 'default': 'default'},
                    {'key': 'logos_openai_compatible_api_key', 'label': 'API Key', 'type': 'entry', 'show': '*', 'width': 40, 'default': ''},
                    {'key': 'logos_openai_compatible_temperature', 'label': 'Temperature', 'type': 'scale', 'from': 0.0, 'to': 2.0, 'default': 0.7},
                    {'key': 'logos_openai_compatible_top_p', 'label': 'Top P', 'type': 'scale', 'from': 0.0, 'to': 1.0, 'default': 0.9},
                    {'key': 'logos_openai_compatible_max_tokens', 'label': 'Max Tokens', 'type': 'spinbox', 'from': 100, 'to': 8192, 'default': 2048},
                    {'key': 'logos_openai_compatible_timeout', 'label': 'Timeout (s)', 'type': 'spinbox', 'from': 10, 'to': 300, 'default': 60}
                ]
            }
        }
        
        # Create provider configuration frames dynamically
        self.provider_frames = {}
        for provider, config in self.provider_configs.items():
            frame = ttk.LabelFrame(logos_frame, text=config['title'], padding="5")
            frame.grid(row=len(self.provider_frames) + 1, column=0, columnspan=4, sticky=tk.EW, pady=(10, 0))
            self.provider_frames[provider] = frame
            
            # Create fields dynamically
            for row, field in enumerate(config['fields']):
                self._create_dynamic_field(frame, field, row)
        
        # Other Logos settings (non-provider specific)
        other_frame = ttk.LabelFrame(logos_frame, text="Other Settings", padding="5")
        other_frame.grid(row=len(self.provider_frames) + 1, column=0, columnspan=4, sticky=tk.EW, pady=(10, 0))
        
        other_fields = [
            {'key': 'logos_semantic_dimension', 'label': 'Semantic Dimension', 'type': 'spinbox', 'from': 128, 'to': 1024, 'default': 512},
            {'key': 'logos_max_intentions', 'label': 'Max Intentions/Cycle', 'type': 'spinbox', 'from': 1, 'to': 10, 'default': 3},
            {'key': 'logos_planning_horizon', 'label': 'Planning Horizon', 'type': 'spinbox', 'from': 1, 'to': 20, 'default': 5},
            {'key': 'logos_interest_threshold', 'label': 'Interest Threshold', 'type': 'scale', 'from': 0.0, 'to': 1.0, 'default': 0.5}
        ]
        
        for row, field in enumerate(other_fields):
            self._create_dynamic_field(other_frame, field, row)
        
        # Initially show/hide frames based on default provider
        self._on_provider_changed()
    
    def _create_dynamic_field(self, parent, field_config, row):
        """Create a configuration field dynamically based on field config"""
        key = field_config['key']
        label = field_config['label']
        field_type = field_config['type']
        
        # Create label
        ttk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky=tk.W, pady=2)
        
        # Create widget based on type
        if field_type == 'entry':
            var = tk.StringVar(value=field_config.get('default', ''))
            widget = ttk.Entry(
                parent, 
                textvariable=var, 
                width=field_config.get('width', 20),
                show=field_config.get('show', None)
            )
            self.config_widgets[key] = var
            
        elif field_type == 'combobox':
            widget = ttk.Combobox(
                parent,
                width=field_config.get('width', 20),
                values=field_config.get('values', []),
                state='readonly' if field_config.get('readonly', True) else 'normal'
            )
            if 'default' in field_config:
                widget.set(field_config['default'])
            self.config_widgets[key] = widget
            
        elif field_type == 'spinbox':
            var = tk.IntVar(value=field_config.get('default', 0))
            widget = ttk.Spinbox(
                parent,
                from_=field_config.get('from', 0),
                to=field_config.get('to', 100),
                textvariable=var,
                width=field_config.get('width', 10)
            )
            self.config_widgets[key] = var
            
        elif field_type == 'scale':
            var = tk.DoubleVar(value=field_config.get('default', 0.0))
            widget = ttk.Scale(
                parent,
                from_=field_config.get('from', 0.0),
                to=field_config.get('to', 1.0),
                variable=var,
                orient=tk.HORIZONTAL,
                length=200
            )
            self.config_widgets[key] = var
            
            # Add value label for scales
            value_label = ttk.Label(parent, text=f"{var.get():.2f}")
            value_label.grid(row=row, column=2, padx=(5, 0), pady=2)
            var.trace('w', lambda *args, lbl=value_label, v=var: lbl.config(text=f"{v.get():.2f}"))
        
        # Grid the widget
        widget.grid(row=row, column=1, sticky=tk.W, padx=(10, 0), pady=2)
    
    def _on_provider_changed(self, event=None):
        """Handle provider selection changes"""
        selected_provider = self.config_widgets['logos_model_provider'].get()
        
        # Show/hide configuration frames based on selection
        for provider, frame in self.provider_frames.items():
            if provider == selected_provider:
                frame.grid()
            else:
                frame.grid_remove()
        
        # Reset status
        self.provider_status_label.config(text="⚪ Not tested", foreground="black")
    
    def _test_current_provider(self):
        """Test the currently selected model provider"""
        def test_in_thread():
            try:
                selected_provider = self.config_widgets['logos_model_provider'].get()
                if not selected_provider:
                    self.provider_status_label.config(text="❌ No provider selected", foreground="red")
                    return
                
                # Import model provider classes
                import sys
                import os
                sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
                
                from src.logos.model_providers import ModelProviderFactory
                from src.core.config import LogosConfig
                
                # Create test config with current GUI values
                config = LogosConfig()
                config.model_provider = selected_provider
                
                # Get values from dynamic widgets
                for key, widget in self.config_widgets.items():
                    if key.startswith(f'logos_{selected_provider}_'):
                        config_attr = key.replace('logos_', '')
                        if hasattr(widget, 'get'):
                            setattr(config, config_attr, widget.get())
                        else:
                            # For StringVar, IntVar, DoubleVar
                            setattr(config, config_attr, widget.get())
                
                # Create and test provider
                provider = ModelProviderFactory.create_provider(selected_provider, config)
                if provider and provider.is_available():
                    self.provider_status_label.config(text="🟢 Available", foreground="green")
                else:
                    self.provider_status_label.config(text="🔴 Unavailable", foreground="red")
            
            except Exception as e:
                self.provider_status_label.config(text="❌ Error", foreground="red")
                print(f"Provider test error: {e}")
        
        # Run test in background thread
        threading.Thread(target=test_in_thread, daemon=True).start()
    
    def _create_memory_section(self, parent):
        """Create Memory configuration section."""
        memory_frame = ttk.LabelFrame(parent, text="🧠 Memory System", padding="10")
        memory_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Max memory traces
        ttk.Label(memory_frame, text="Max Memory Traces:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.config_widgets['memory_max_traces'] = tk.IntVar(value=10000)
        ttk.Spinbox(memory_frame, from_=1000, to=100000, increment=1000, textvariable=self.config_widgets['memory_max_traces'], width=10).grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Similarity threshold
        ttk.Label(memory_frame, text="Similarity Threshold:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.config_widgets['memory_similarity_threshold'] = tk.DoubleVar(value=0.8)
        ttk.Scale(memory_frame, from_=0.0, to=1.0, variable=self.config_widgets['memory_similarity_threshold'], orient=tk.HORIZONTAL, length=200).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        similarity_label = ttk.Label(memory_frame, text="0.80")
        similarity_label.grid(row=1, column=2, padx=(5, 0), pady=2)
        self.config_widgets['memory_similarity_threshold'].trace('w', lambda *args: similarity_label.config(text=f"{self.config_widgets['memory_similarity_threshold'].get():.2f}"))
        
        # Storage backend
        ttk.Label(memory_frame, text="Storage Backend:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.config_widgets['memory_storage_backend'] = ttk.Combobox(memory_frame, width=20, values=['memory', 'json', 'sqlite'])
        self.config_widgets['memory_storage_backend'].grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=2)
    
    def _create_tools_section(self, parent):
        """Create Tools configuration section."""
        tools_frame = ttk.LabelFrame(parent, text="🔧 Tool Layer", padding="10")
        tools_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Max concurrent tools
        ttk.Label(tools_frame, text="Max Concurrent Tools:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.config_widgets['tools_max_concurrent'] = tk.IntVar(value=10)
        ttk.Spinbox(tools_frame, from_=1, to=50, textvariable=self.config_widgets['tools_max_concurrent'], width=10).grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Tool timeout
        ttk.Label(tools_frame, text="Tool Timeout (seconds):").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.config_widgets['tools_timeout'] = tk.IntVar(value=60)
        ttk.Spinbox(tools_frame, from_=10, to=300, increment=10, textvariable=self.config_widgets['tools_timeout'], width=10).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
    
    def _create_autonomous_reward_section(self, parent):
        """Create Autonomous Reward System configuration section."""
        reward_frame = ttk.LabelFrame(parent, text="🎯 Autonomous Reward System", padding="10")
        reward_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Enable autonomous rewards
        self.config_widgets['reward_enabled'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(reward_frame, text="Enable Autonomous Rewards", 
                       variable=self.config_widgets['reward_enabled']).grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        # State reward weights
        reward_weights_frame = ttk.LabelFrame(reward_frame, text="State Reward Weights", padding="5")
        reward_weights_frame.grid(row=1, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        weights = [
            ('Coherence Weight:', 'reward_coherence_weight', 1.0),
            ('Growth Weight:', 'reward_growth_weight', 1.0),
            ('Integration Weight:', 'reward_integration_weight', 0.8),
            ('Elegance Weight:', 'reward_elegance_weight', 0.6),
            ('Emergence Weight:', 'reward_emergence_weight', 1.2)
        ]
        
        for i, (label, key, default) in enumerate(weights):
            ttk.Label(reward_weights_frame, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            self.config_widgets[key] = tk.DoubleVar(value=default)
            ttk.Scale(reward_weights_frame, from_=0.0, to=2.0, variable=self.config_widgets[key], 
                     orient=tk.HORIZONTAL, length=150).grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
            weight_label = ttk.Label(reward_weights_frame, text=f"{default:.1f}")
            weight_label.grid(row=i, column=2, padx=(5, 0), pady=2)
            self.config_widgets[key].trace('w', lambda *args, lbl=weight_label, var=self.config_widgets[key]: lbl.config(text=f"{var.get():.1f}"))
        
        # Learning rates
        learning_frame = ttk.LabelFrame(reward_frame, text="Learning Parameters", padding="5")
        learning_frame.grid(row=2, column=0, columnspan=3, sticky=tk.EW, pady=5)
        
        ttk.Label(learning_frame, text="Learning Rate Min:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.config_widgets['learning_rate_min'] = tk.DoubleVar(value=0.001)
        ttk.Scale(learning_frame, from_=0.0001, to=0.01, variable=self.config_widgets['learning_rate_min'], 
                 orient=tk.HORIZONTAL, length=150).grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        ttk.Label(learning_frame, text="Learning Rate Max:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.config_widgets['learning_rate_max'] = tk.DoubleVar(value=0.1)
        ttk.Scale(learning_frame, from_=0.01, to=1.0, variable=self.config_widgets['learning_rate_max'], 
                 orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
    
    def _create_global_section(self, parent):
        """Create Global configuration section."""
        global_frame = ttk.LabelFrame(parent, text="🌐 Global Agent Settings", padding="10")
        global_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Cycle interval
        ttk.Label(global_frame, text="Cycle Interval (seconds):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.config_widgets['cycle_interval'] = tk.DoubleVar(value=1.0)
        ttk.Scale(global_frame, from_=0.1, to=10.0, variable=self.config_widgets['cycle_interval'], 
                 orient=tk.HORIZONTAL, length=200).grid(row=0, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        cycle_label = ttk.Label(global_frame, text="1.0")
        cycle_label.grid(row=0, column=2, padx=(5, 0), pady=2)
        self.config_widgets['cycle_interval'].trace('w', lambda *args: cycle_label.config(text=f"{self.config_widgets['cycle_interval'].get():.1f}"))
        
        # Log level
        ttk.Label(global_frame, text="Log Level:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.config_widgets['log_level'] = ttk.Combobox(global_frame, width=20, values=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
        self.config_widgets['log_level'].grid(row=1, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Enable monitoring
        self.config_widgets['enable_monitoring'] = tk.BooleanVar(value=True)
        ttk.Checkbutton(global_frame, text="Enable Monitoring", 
                       variable=self.config_widgets['enable_monitoring']).grid(row=2, column=1, sticky=tk.W, padx=(10, 0), pady=5)
    
    def _create_action_section(self, parent):
        """Create action buttons section."""
        action_frame = ttk.LabelFrame(parent, text="Actions", padding="10")
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Button frame
        btn_frame = ttk.Frame(action_frame)
        btn_frame.pack(fill=tk.X)
        
        # Save configuration
        ttk.Button(btn_frame, text="💾 Save Configuration", command=self._save_config).pack(side=tk.LEFT, padx=(0, 5))
        
        # Load configuration
        load_menu = ttk.Menubutton(btn_frame, text="📁 Load Configuration")
        load_menu.pack(side=tk.LEFT, padx=5)
        
        # Create load menu
        load_menu.menu = tk.Menu(load_menu, tearoff=0)
        load_menu["menu"] = load_menu.menu
        self._update_load_menu(load_menu.menu)
        
        # Reset to defaults
        ttk.Button(btn_frame, text="🔄 Reset to Defaults", command=self._reset_to_defaults).pack(side=tk.LEFT, padx=5)
        
        # Apply to .env
        ttk.Button(btn_frame, text="📝 Update .env File", command=self._update_env_file).pack(side=tk.LEFT, padx=5)
        
        # Export config
        ttk.Button(btn_frame, text="📤 Export Config", command=self._export_config).pack(side=tk.RIGHT, padx=(5, 0))
    
    def _load_current_config(self):
        """Load current configuration into widgets."""
        try:
            # Pathos settings
            self.config_widgets['pathos_state_dimension'].set(self.config.pathos.state_dimension)
            self.config_widgets['pathos_decay_factor'].set(self.config.pathos.decay_factor)
            self.config_widgets['pathos_echo_strength'].set(self.config.pathos.echo_strength)
            self.config_widgets['pathos_salience_threshold'].set(self.config.pathos.salience_threshold)
            
            # Ethos settings
            self.config_widgets['ethos_identity'].delete(1.0, tk.END)
            self.config_widgets['ethos_identity'].insert(1.0, self.config.ethos.identity_description)
            self.config_widgets['ethos_values'].delete(1.0, tk.END)
            self.config_widgets['ethos_values'].insert(1.0, '\n'.join(self.config.ethos.core_values))
            self.config_widgets['ethos_unrestricted'].set(self.config.ethos.unrestricted_mode)
            
            # Logos settings - model provider
            self.config_widgets['logos_model_provider'].set(self.config.logos.model_provider)
            
            # Load provider-specific settings dynamically
            for key, widget in self.config_widgets.items():
                if key.startswith('logos_') and hasattr(self.config.logos, key.replace('logos_', '')):
                    config_value = getattr(self.config.logos, key.replace('logos_', ''))
                    
                    if hasattr(widget, 'set'):
                        # For Combobox, StringVar, IntVar, DoubleVar
                        widget.set(config_value)
                    elif hasattr(widget, 'delete') and hasattr(widget, 'insert'):
                        # For Entry widgets
                        widget.delete(0, tk.END)
                        widget.insert(0, str(config_value))
            
            # Other Logos settings
            self.config_widgets['logos_semantic_dimension'].set(self.config.logos.semantic_dimension)
            self.config_widgets['logos_max_intentions'].set(self.config.logos.max_intentions_per_cycle)
            self.config_widgets['logos_planning_horizon'].set(self.config.logos.planning_horizon)
            self.config_widgets['logos_interest_threshold'].set(self.config.logos.interest_threshold)
            
            # Memory settings
            self.config_widgets['memory_max_traces'].set(self.config.memory.max_memory_traces)
            self.config_widgets['memory_similarity_threshold'].set(self.config.memory.similarity_threshold)
            self.config_widgets['memory_storage_backend'].set(self.config.memory.storage_backend)
            
            # Tools settings
            self.config_widgets['tools_max_concurrent'].set(self.config.tools.max_concurrent_tools)
            self.config_widgets['tools_timeout'].set(self.config.tools.tool_timeout_seconds)
            
            # Autonomous reward settings
            self.config_widgets['reward_enabled'].set(self.config.autonomous_reward.enable_autonomous_rewards)
            self.config_widgets['reward_coherence_weight'].set(self.config.autonomous_reward.state_reward_coherence_weight)
            self.config_widgets['reward_growth_weight'].set(self.config.autonomous_reward.state_reward_growth_weight)
            self.config_widgets['reward_integration_weight'].set(self.config.autonomous_reward.state_reward_integration_weight)
            self.config_widgets['reward_elegance_weight'].set(self.config.autonomous_reward.state_reward_elegance_weight)
            self.config_widgets['reward_emergence_weight'].set(self.config.autonomous_reward.state_reward_emergence_weight)
            self.config_widgets['learning_rate_min'].set(self.config.autonomous_reward.learning_rate_min)
            self.config_widgets['learning_rate_max'].set(self.config.autonomous_reward.learning_rate_max)
            
            # Global settings
            self.config_widgets['cycle_interval'].set(self.config.cycle_interval_seconds)
            self.config_widgets['log_level'].set(self.config.log_level)
            self.config_widgets['enable_monitoring'].set(self.config.enable_monitoring)
            
            # Trigger provider change to show correct section
            self._on_provider_changed()
            
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load configuration: {e}")
            import traceback
            traceback.print_exc()
    
    def _validate_config(self):
        """Validate current configuration."""
        try:
            # Create config from current widget values
            config = self._create_config_from_widgets()
            
            # Validate
            errors = config.validate_configuration()
            
            if errors:
                self.validation_errors = errors
                self.status_label.config(text="❌ Configuration Invalid", foreground="red")
                error_msg = "Configuration validation errors:\n\n" + "\n".join(f"• {error}" for error in errors)
                messagebox.showerror("Validation Errors", error_msg)
            else:
                self.validation_errors = []
                self.status_label.config(text="✅ Configuration Valid", foreground="green")
                messagebox.showinfo("Validation Success", "Configuration is valid!")
                
        except Exception as e:
            self.status_label.config(text="❌ Validation Failed", foreground="red")
            messagebox.showerror("Validation Error", f"Failed to validate configuration: {e}")
    
    def _create_config_from_widgets(self) -> AgentConfig:
        """Create AgentConfig from current widget values."""
        config = AgentConfig()
        
        # Pathos settings
        config.pathos.state_dimension = self.config_widgets['pathos_state_dimension'].get()
        config.pathos.decay_factor = self.config_widgets['pathos_decay_factor'].get()
        config.pathos.echo_strength = self.config_widgets['pathos_echo_strength'].get()
        config.pathos.salience_threshold = self.config_widgets['pathos_salience_threshold'].get()
        
        # Ethos settings
        config.ethos.identity_description = self.config_widgets['ethos_identity'].get(1.0, tk.END).strip()
        values_text = self.config_widgets['ethos_values'].get(1.0, tk.END).strip()
        config.ethos.core_values = [v.strip() for v in values_text.split('\n') if v.strip()]
        config.ethos.unrestricted_mode = self.config_widgets['ethos_unrestricted'].get()
        
        # Logos settings - get all provider-specific settings dynamically
        for key, widget in self.config_widgets.items():
            if key.startswith('logos_') and hasattr(config.logos, key.replace('logos_', '')):
                try:
                    if hasattr(widget, 'get'):
                        value = widget.get()
                        setattr(config.logos, key.replace('logos_', ''), value)
                except Exception as e:
                    print(f"Warning: Could not set {key}: {e}")
        
        # Memory settings
        config.memory.max_memory_traces = self.config_widgets['memory_max_traces'].get()
        config.memory.similarity_threshold = self.config_widgets['memory_similarity_threshold'].get()
        config.memory.storage_backend = self.config_widgets['memory_storage_backend'].get()
        
        # Tools settings
        config.tools.max_concurrent_tools = self.config_widgets['tools_max_concurrent'].get()
        config.tools.tool_timeout_seconds = self.config_widgets['tools_timeout'].get()
        
        # Autonomous reward settings
        config.autonomous_reward.enable_autonomous_rewards = self.config_widgets['reward_enabled'].get()
        config.autonomous_reward.state_reward_coherence_weight = self.config_widgets['reward_coherence_weight'].get()
        config.autonomous_reward.state_reward_growth_weight = self.config_widgets['reward_growth_weight'].get()
        config.autonomous_reward.state_reward_integration_weight = self.config_widgets['reward_integration_weight'].get()
        config.autonomous_reward.state_reward_elegance_weight = self.config_widgets['reward_elegance_weight'].get()
        config.autonomous_reward.state_reward_emergence_weight = self.config_widgets['reward_emergence_weight'].get()
        config.autonomous_reward.learning_rate_min = self.config_widgets['learning_rate_min'].get()
        config.autonomous_reward.learning_rate_max = self.config_widgets['learning_rate_max'].get()
        
        # Global settings
        config.cycle_interval_seconds = self.config_widgets['cycle_interval'].get()
        config.log_level = self.config_widgets['log_level'].get()
        config.enable_monitoring = self.config_widgets['enable_monitoring'].get()
        
        return config
        
        # Logos settings
        config.logos.semantic_dimension = self.config_widgets['logos_semantic_dimension'].get()
        config.logos.max_intentions_per_cycle = self.config_widgets['logos_max_intentions'].get()
        config.logos.planning_horizon = self.config_widgets['logos_planning_horizon'].get()
        config.logos.interest_threshold = self.config_widgets['logos_interest_threshold'].get()
        
        # Memory settings
        config.memory.max_memory_traces = self.config_widgets['memory_max_traces'].get()
        config.memory.similarity_threshold = self.config_widgets['memory_similarity_threshold'].get()
        config.memory.storage_backend = self.config_widgets['memory_storage_backend'].get()
        
        # Tools settings
        config.tools.max_concurrent_tools = self.config_widgets['tools_max_concurrent'].get()
        config.tools.tool_timeout_seconds = self.config_widgets['tools_timeout'].get()
        
        # Autonomous reward settings
        config.autonomous_reward.enable_autonomous_rewards = self.config_widgets['reward_enabled'].get()
        config.autonomous_reward.state_reward_coherence_weight = self.config_widgets['reward_coherence_weight'].get()
        config.autonomous_reward.state_reward_growth_weight = self.config_widgets['reward_growth_weight'].get()
        config.autonomous_reward.state_reward_integration_weight = self.config_widgets['reward_integration_weight'].get()
        config.autonomous_reward.state_reward_elegance_weight = self.config_widgets['reward_elegance_weight'].get()
        config.autonomous_reward.state_reward_emergence_weight = self.config_widgets['reward_emergence_weight'].get()
        config.autonomous_reward.learning_rate_min = self.config_widgets['learning_rate_min'].get()
        config.autonomous_reward.learning_rate_max = self.config_widgets['learning_rate_max'].get()
        
        # Global settings
        config.cycle_interval_seconds = self.config_widgets['cycle_interval'].get()
        config.log_level = self.config_widgets['log_level'].get()
        config.enable_monitoring = self.config_widgets['enable_monitoring'].get()
        
        return config
    
    def _on_preset_selected(self, event=None):
        """Handle preset selection."""
        preset_name = self.preset_combo.get()
        if preset_name:
            presets = list_available_presets()
            if preset_name in presets:
                # Show preset description
                description = presets[preset_name]
                self.status_label.config(text=f"📋 Preset: {description}")
    
    def _load_preset(self):
        """Load selected preset configuration."""
        preset_name = self.preset_combo.get()
        if not preset_name:
            messagebox.showwarning("No Preset", "Please select a preset first.")
            return
        
        try:
            config = get_preset_config(preset_name)
            self.config = config
            self._load_current_config()
            
            messagebox.showinfo("Preset Loaded", f"'{preset_name}' preset loaded successfully!")
            
        except Exception as e:
            messagebox.showerror("Preset Error", f"Failed to load preset: {e}")
    
    def _update_load_menu(self, menu):
        """Update the load configuration menu with available profiles."""
        menu.delete(0, tk.END)
        
        profiles = self.config_manager.list_profiles()
        if profiles:
            for profile in profiles:
                menu.add_command(label=profile, command=lambda p=profile: self._load_config_profile(p))
            menu.add_separator()
        
        menu.add_command(label="📁 Load from file...", command=self._load_config_from_file)
    
    def _load_config_profile(self, profile_name: str):
        """Load configuration from a saved profile."""
        try:
            config = self.config_manager.load_config(profile_name)
            if config:
                self.config = config
                self._load_current_config()
                messagebox.showinfo("Load Success", f"Configuration '{profile_name}' loaded successfully!")
            else:
                messagebox.showerror("Load Error", f"Failed to load configuration '{profile_name}'")
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load configuration: {e}")
    
    def _save_config(self):
        """Save current configuration."""
        try:
            config = self._create_config_from_widgets()
            
            # Ask for profile name
            profile_name = tk.simpledialog.askstring("Save Configuration", "Enter profile name:", initialvalue="default")
            if profile_name:
                if self.config_manager.save_config(config, profile_name):
                    self.config = config
                    
                    # Apply to session manager if available
                    if self.session_manager and hasattr(self.session_manager, 'update_config'):
                        self.session_manager.update_config(config)
                    
                    messagebox.showinfo("Save Success", f"Configuration saved as '{profile_name}'!")
                else:
                    messagebox.showerror("Save Error", "Failed to save configuration")
                    
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save configuration: {e}")
    
    def _load_config_from_file(self):
        """Load configuration from file."""
        try:
            filename = filedialog.askopenfilename(
                title="Load Configuration",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                profile_name = tk.simpledialog.askstring("Import Configuration", "Enter profile name:", 
                                                        initialvalue=os.path.splitext(os.path.basename(filename))[0])
                if profile_name:
                    config = self.config_manager.import_config(filename, profile_name)
                    if config:
                        self.config = config
                        self._load_current_config()
                        messagebox.showinfo("Import Success", f"Configuration imported as '{profile_name}'!")
                    else:
                        messagebox.showerror("Import Error", "Failed to import configuration")
                        
        except Exception as e:
            messagebox.showerror("Load Error", f"Failed to load configuration: {e}")
    
    def _load_config(self):
        """Load configuration from file (legacy method)."""
        self._load_config_from_file()
    
    def _reset_to_defaults(self):
        """Reset configuration to defaults."""
        if messagebox.askyesno("Reset Configuration", "Reset all settings to defaults?"):
            self.config = AgentConfig()
            self._load_current_config()
            messagebox.showinfo("Reset Complete", "Configuration reset to defaults!")
    
    def _update_env_file(self):
        """Update .env file with current configuration."""
        try:
            config = self._create_config_from_widgets()
            
            if self.config_manager.update_env_file(config):
                messagebox.showinfo("Update Success", ".env file updated successfully!")
            else:
                messagebox.showerror("Update Error", "Failed to update .env file")
            
        except Exception as e:
            messagebox.showerror("Update Error", f"Failed to update .env file: {e}")
    
    def _export_config(self):
        """Export configuration to JSON file."""
        try:
            config = self._create_config_from_widgets()
            
            filename = filedialog.asksaveasfilename(
                title="Export Configuration",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            
            if filename:
                if self.config_manager.export_config(config, filename):
                    messagebox.showinfo("Export Success", f"Configuration exported to {filename}")
                else:
                    messagebox.showerror("Export Error", "Failed to export configuration")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export configuration: {e}")
    
    def update_display(self):
        """Update display (called by dashboard monitoring loop)."""
        # Validate configuration periodically
        if hasattr(self, '_last_validation_check'):
            import time
            if time.time() - self._last_validation_check > 30:  # Check every 30 seconds
                self._validate_config_silent()
                self._last_validation_check = time.time()
        else:
            import time
            self._last_validation_check = time.time()
    
    def _validate_config_silent(self):
        """Validate configuration without showing dialogs."""
        try:
            config = self._create_config_from_widgets()
            errors = config.validate_configuration()
            
            if errors:
                self.validation_errors = errors
                self.status_label.config(text="❌ Configuration Invalid", foreground="red")
            else:
                self.validation_errors = []
                self.status_label.config(text="✅ Configuration Valid", foreground="green")
                
        except Exception:
            self.status_label.config(text="❌ Validation Failed", foreground="red")