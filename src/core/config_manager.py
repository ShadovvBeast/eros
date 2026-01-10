"""
Configuration Manager

Centralized configuration management with GUI integration.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import asdict
from pathlib import Path

from .config import AgentConfig


class ConfigManager:
    """
    Centralized configuration manager for the autonomous agent system.
    
    Provides functionality to:
    - Load/save configurations from multiple sources
    - Validate configurations
    - Apply configurations to running systems
    - Export/import configuration profiles
    """
    
    def __init__(self, config_dir: str = "configs"):
        """Initialize configuration manager."""
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.current_config: Optional[AgentConfig] = None
        self.config_profiles: Dict[str, AgentConfig] = {}
        
        self.logger = logging.getLogger(__name__)
        
        # Load default configuration
        self.load_default_config()
    
    def load_default_config(self) -> AgentConfig:
        """Load default configuration from environment."""
        try:
            self.current_config = AgentConfig.from_env()
            self.logger.info("✅ Default configuration loaded from environment")
            return self.current_config
        except Exception as e:
            self.logger.error(f"❌ Failed to load default configuration: {e}")
            # Fallback to basic config
            self.current_config = AgentConfig()
            return self.current_config
    
    def save_config(self, config: AgentConfig, profile_name: str = "default") -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration to save
            profile_name: Name of the configuration profile
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            config_path = self.config_dir / f"{profile_name}.json"
            config_dict = asdict(config)
            
            with open(config_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.config_profiles[profile_name] = config
            self.logger.info(f"✅ Configuration saved to {config_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {e}")
            return False
    
    def load_config(self, profile_name: str = "default") -> Optional[AgentConfig]:
        """
        Load configuration from file.
        
        Args:
            profile_name: Name of the configuration profile
            
        Returns:
            Loaded configuration or None if failed
        """
        try:
            config_path = self.config_dir / f"{profile_name}.json"
            
            if not config_path.exists():
                self.logger.warning(f"⚠️ Configuration profile '{profile_name}' not found")
                return None
            
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            # Reconstruct config object (simplified approach)
            config = AgentConfig.from_env()  # Start with defaults
            self._apply_config_dict(config, config_dict)
            
            self.config_profiles[profile_name] = config
            self.current_config = config
            
            self.logger.info(f"✅ Configuration loaded from {config_path}")
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load configuration: {e}")
            return None
    
    def _apply_config_dict(self, config: AgentConfig, config_dict: Dict[str, Any]):
        """Apply configuration dictionary to config object."""
        # This is a simplified implementation
        # In a full implementation, you'd recursively apply all nested values
        
        if 'pathos' in config_dict:
            pathos_dict = config_dict['pathos']
            if 'state_dimension' in pathos_dict:
                config.pathos.state_dimension = pathos_dict['state_dimension']
            if 'decay_factor' in pathos_dict:
                config.pathos.decay_factor = pathos_dict['decay_factor']
            if 'echo_strength' in pathos_dict:
                config.pathos.echo_strength = pathos_dict['echo_strength']
        
        if 'logos' in config_dict:
            logos_dict = config_dict['logos']
            if 'gemini_api_key' in logos_dict:
                config.logos.gemini_api_key = logos_dict['gemini_api_key']
            if 'gemini_model' in logos_dict:
                config.logos.gemini_model = logos_dict['gemini_model']
        
        # Add more sections as needed...
    
    def validate_config(self, config: Optional[AgentConfig] = None) -> List[str]:
        """
        Validate configuration.
        
        Args:
            config: Configuration to validate (uses current if None)
            
        Returns:
            List of validation errors (empty if valid)
        """
        if config is None:
            config = self.current_config
        
        if config is None:
            return ["No configuration loaded"]
        
        try:
            return config.validate_configuration()
        except Exception as e:
            return [f"Validation failed: {e}"]
    
    def list_profiles(self) -> List[str]:
        """List available configuration profiles."""
        try:
            profiles = []
            for config_file in self.config_dir.glob("*.json"):
                profiles.append(config_file.stem)
            return sorted(profiles)
        except Exception as e:
            self.logger.error(f"❌ Failed to list profiles: {e}")
            return []
    
    def delete_profile(self, profile_name: str) -> bool:
        """
        Delete configuration profile.
        
        Args:
            profile_name: Name of profile to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        try:
            config_path = self.config_dir / f"{profile_name}.json"
            
            if config_path.exists():
                config_path.unlink()
                if profile_name in self.config_profiles:
                    del self.config_profiles[profile_name]
                self.logger.info(f"✅ Profile '{profile_name}' deleted")
                return True
            else:
                self.logger.warning(f"⚠️ Profile '{profile_name}' not found")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to delete profile: {e}")
            return False
    
    def export_config(self, config: AgentConfig, export_path: str) -> bool:
        """
        Export configuration to specified path.
        
        Args:
            config: Configuration to export
            export_path: Path to export to
            
        Returns:
            True if exported successfully, False otherwise
        """
        try:
            config_dict = asdict(config)
            
            with open(export_path, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"✅ Configuration exported to {export_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to export configuration: {e}")
            return False
    
    def import_config(self, import_path: str, profile_name: str) -> Optional[AgentConfig]:
        """
        Import configuration from specified path.
        
        Args:
            import_path: Path to import from
            profile_name: Name for the imported profile
            
        Returns:
            Imported configuration or None if failed
        """
        try:
            with open(import_path, 'r') as f:
                config_dict = json.load(f)
            
            # Create config from imported data
            config = AgentConfig.from_env()  # Start with defaults
            self._apply_config_dict(config, config_dict)
            
            # Save as new profile
            if self.save_config(config, profile_name):
                self.logger.info(f"✅ Configuration imported as '{profile_name}'")
                return config
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to import configuration: {e}")
            return None
    
    def update_env_file(self, config: Optional[AgentConfig] = None) -> bool:
        """
        Update .env file with configuration values.
        
        Args:
            config: Configuration to use (uses current if None)
            
        Returns:
            True if updated successfully, False otherwise
        """
        if config is None:
            config = self.current_config
        
        if config is None:
            self.logger.error("❌ No configuration to update .env file")
            return False
        
        try:
            env_path = '.env'
            env_vars = {}
            
            # Read existing .env file
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key] = value
            
            # Update with configuration values
            env_vars['GEMINI_API_KEY'] = config.logos.gemini_api_key
            env_vars['PATHOS_STATE_DIMENSION'] = str(config.pathos.state_dimension)
            env_vars['PATHOS_DECAY_FACTOR'] = str(config.pathos.decay_factor)
            env_vars['PATHOS_ECHO_STRENGTH'] = str(config.pathos.echo_strength)
            env_vars['LOG_LEVEL'] = config.log_level
            env_vars['MEMORY_MAX_TRACES'] = str(config.memory.max_memory_traces)
            env_vars['CYCLE_INTERVAL_SECONDS'] = str(config.cycle_interval_seconds)
            env_vars['ENABLE_MONITORING'] = str(config.enable_monitoring).lower()
            
            # Write updated .env file
            with open(env_path, 'w') as f:
                f.write("# Agent Configuration (Auto-generated)\n")
                f.write(f"GEMINI_API_KEY={env_vars['GEMINI_API_KEY']}\n\n")
                f.write("# Pathos Layer Configuration\n")
                f.write(f"PATHOS_STATE_DIMENSION={env_vars['PATHOS_STATE_DIMENSION']}\n")
                f.write(f"PATHOS_DECAY_FACTOR={env_vars['PATHOS_DECAY_FACTOR']}\n")
                f.write(f"PATHOS_ECHO_STRENGTH={env_vars['PATHOS_ECHO_STRENGTH']}\n\n")
                f.write("# Logging Configuration\n")
                f.write(f"LOG_LEVEL={env_vars['LOG_LEVEL']}\n\n")
                f.write("# Memory Configuration\n")
                f.write(f"MEMORY_MAX_TRACES={env_vars['MEMORY_MAX_TRACES']}\n\n")
                f.write("# Agent Configuration\n")
                f.write(f"CYCLE_INTERVAL_SECONDS={env_vars['CYCLE_INTERVAL_SECONDS']}\n")
                f.write(f"ENABLE_MONITORING={env_vars['ENABLE_MONITORING']}\n")
            
            self.logger.info("✅ .env file updated successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to update .env file: {e}")
            return False
    
    def get_current_config(self) -> Optional[AgentConfig]:
        """Get current configuration."""
        return self.current_config
    
    def set_current_config(self, config: AgentConfig) -> bool:
        """
        Set current configuration.
        
        Args:
            config: Configuration to set as current
            
        Returns:
            True if set successfully, False otherwise
        """
        try:
            # Validate first
            errors = self.validate_config(config)
            if errors:
                self.logger.warning(f"⚠️ Configuration has validation errors: {errors}")
            
            self.current_config = config
            self.logger.info("✅ Current configuration updated")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set current configuration: {e}")
            return False


# Global configuration manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager