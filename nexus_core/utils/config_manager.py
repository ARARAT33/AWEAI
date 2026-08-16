"""
Configuration management for Nexus Core
"""

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """
    Configuration manager with support for multiple formats
    """
    
    DEFAULT_CONFIG = {
        'system': {
            'name': 'Nexus Core',
            'version': '1.0.0',
            'debug_mode': False,
            'max_concurrent_tasks': 10,
            'log_level': 'INFO'
        },
        'learning': {
            'auto_learn': True,
            'model_save_path': './models',
            'knowledge_ttl_days': 30,
            'max_knowledge_entries': 10000
        },
        'automation': {
            'safe_mode': True,
            'allowed_directories': [],
            'timeout_seconds': 300
        },
        'storage': {
            'path': './nexus_memory',
            'backup_enabled': True,
            'compression': False
        },
        'api': {
            'enabled': False,
            'host': 'localhost',
            'port': 8080
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = self.DEFAULT_CONFIG.copy()
        
        # Load from file if exists
        if self.config_path and self.config_path.exists():
            self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            with open(self.config_path, 'r') as f:
                if self.config_path.suffix in ['.yaml', '.yml']:
                    loaded = yaml.safe_load(f)
                else:
                    loaded = json.load(f)
                
                # Merge with defaults
                self._deep_merge(self.config, loaded)
                
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
    
    def _deep_merge(self, base: Dict, override: Dict):
        """Deep merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, path: Optional[str] = None):
        """Save configuration to file"""
        save_path = Path(path) if path else self.config_path
        
        if not save_path:
            raise ValueError("No path specified for saving config")
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            if save_path.suffix in ['.yaml', '.yml']:
                yaml.dump(self.config, f, default_flow_style=False)
            else:
                json.dump(self.config, f, indent=2)
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config = self.DEFAULT_CONFIG.copy()
    
    def to_dict(self) -> Dict:
        """Get configuration as dictionary"""
        return self.config.copy()
