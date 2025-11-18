"""
Configuration management for training.
"""

import yaml
from typing import Dict, Any, Optional
from pathlib import Path


class Config:
    """Configuration class for managing training parameters."""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Args:
            config_dict: Configuration dictionary
        """
        if config_dict is None:
            config_dict = self.get_default_config()
        
        self._config = config_dict
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default configuration."""
        return {
            # Data settings
            'data': {
                'batch_size': 32,
                'num_workers': 4,
                'train_ratio': 0.8,
                'random_state': 42,
                'normalize': True,
                'augmentation': False,
            },
            
            # Model settings
            'model': {
                'type': 'mlp',
                'input_dim': 784,
                'output_dim': 10,
                'hidden_dims': [256, 128],
                'dropout_rate': 0.5,
            },
            
            # Training settings
            'training': {
                'epochs': 10,
                'learning_rate': 0.001,
                'weight_decay': 1e-5,
                'gradient_clip': 1.0,
                'optimizer': 'adam',
                'use_scheduler': True,
                'scheduler_patience': 5,
                'scheduler_factor': 0.5,
                'early_stopping': True,
                'patience': 10,
            },
            
            # Logging and checkpointing
            'logging': {
                'log_dir': 'logs',
                'checkpoint_dir': 'checkpoints',
                'save_freq': 1,
            },
            
            # Device
            'device': 'cuda',
        }
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
        
        Returns:
            Config object
        """
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(config_dict)
    
    def to_yaml(self, yaml_path: str):
        """
        Save configuration to YAML file.
        
        Args:
            yaml_path: Path to save YAML configuration
        """
        Path(yaml_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False)
        
        print(f"Configuration saved to {yaml_path}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def update(self, updates: Dict[str, Any]):
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates
        """
        self._deep_update(self._config, updates)
    
    def _deep_update(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionaries."""
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        return f"Config({self._config})"
    
    def __str__(self) -> str:
        return yaml.dump(self._config, default_flow_style=False)