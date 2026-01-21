"""
Configuration Loader Module
Handles loading and validation of YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml


class ConfigLoader:
    """
    Loads and manages YAML configuration files.
    
    Provides a unified interface for accessing configuration values
    across the entire application.
    
    Attributes:
        config_dir: Path to the configuration directory
        _configs: Dictionary holding all loaded configurations
    """
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the configuration loader.
        
        Args:
            config_dir: Path to configuration directory. Defaults to ./configs
        """
        if config_dir is None:
            # Default to configs directory relative to project root
            self.config_dir = Path(__file__).parent.parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)
        
        self._configs: Dict[str, Dict[str, Any]] = {}
        
    def load(self, config_name: str) -> Dict[str, Any]:
        """
        Load a configuration file.
        
        Args:
            config_name: Name of the config file (with or without .yaml extension)
            
        Returns:
            Dictionary containing the configuration values
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            yaml.YAMLError: If configuration file is invalid YAML
        """
        # Add .yaml extension if not present
        if not config_name.endswith('.yaml') and not config_name.endswith('.yml'):
            config_name = f"{config_name}.yaml"
        
        config_path = self.config_dir / config_name
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Cache the loaded config
        self._configs[config_name] = config
        
        return config
    
    def load_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all configuration files in the config directory.
        
        Returns:
            Dictionary mapping config names to their contents
        """
        for config_file in self.config_dir.glob("*.yaml"):
            self.load(config_file.name)
        
        for config_file in self.config_dir.glob("*.yml"):
            self.load(config_file.name)
        
        return self._configs
    
    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        """
        Get a specific value from a configuration.
        
        Args:
            config_name: Name of the configuration file
            key: Dot-separated key path (e.g., 'training.learning_rate')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        # Load config if not already loaded
        if config_name not in self._configs:
            try:
                self.load(config_name)
            except FileNotFoundError:
                return default
        
        config = self._configs.get(config_name, {})
        
        # Navigate nested keys
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_env_config(self) -> Dict[str, Any]:
        """Load and return environment configuration."""
        return self.load('env_config')
    
    def get_ppo_config(self) -> Dict[str, Any]:
        """Load and return PPO configuration."""
        return self.load('ppo_config')
    
    def get_device_config(self) -> Dict[str, Any]:
        """Load and return device configuration."""
        return self.load('device_config')
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configurations into one.
        
        Later configurations override earlier ones for duplicate keys.
        
        Args:
            *config_names: Variable number of config names to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for name in config_names:
            config = self.load(name)
            merged = self._deep_merge(merged, config)
        
        return merged
    
    @staticmethod
    def _deep_merge(base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary
            override: Dictionary to merge on top
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigLoader._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save(self, config_name: str, config: Dict[str, Any]) -> None:
        """
        Save a configuration to file.
        
        Args:
            config_name: Name of the configuration file
            config: Configuration dictionary to save
        """
        if not config_name.endswith('.yaml') and not config_name.endswith('.yml'):
            config_name = f"{config_name}.yaml"
        
        config_path = self.config_dir / config_name
        
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        # Update cache
        self._configs[config_name] = config


# Convenience function for quick access
def load_config(config_name: str, config_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Quick function to load a configuration file.
    
    Args:
        config_name: Name of the configuration file
        config_dir: Optional path to config directory
        
    Returns:
        Configuration dictionary
    """
    loader = ConfigLoader(config_dir)
    return loader.load(config_name)
