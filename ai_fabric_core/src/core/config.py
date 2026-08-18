"""
System Configuration Manager
Handles all configuration settings for AI Fabric Core
"""

import os
import yaml
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class HardwareConfig:
    """Hardware configuration settings"""
    gpu_type: str = "NVIDIA H100"
    num_gpus: int = 8
    memory_gb: int = 80
    interconnect: str = "NVLink"
    cpu_cores: int = 64
    ram_gb: int = 512


@dataclass
class DistributedConfig:
    """Distributed training configuration"""
    num_nodes: int = 1
    workers_per_node: int = 8
    gradient_sync_strategy: str = "all_reduce"
    communication_backend: str = "nccl"
    checkpoint_interval: int = 1000
    mixed_precision: bool = True
    precision_format: str = "fp16"


@dataclass
class DataConfig:
    """Data pipeline configuration"""
    batch_size: int = 32
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class InferenceConfig:
    """Inference serving configuration"""
    max_batch_size: int = 128
    max_concurrent_requests: int = 1000
    timeout_seconds: int = 30
    quantization_enabled: bool = False
    quantization_bits: int = 8


@dataclass
class SecurityConfig:
    """Security and governance configuration"""
    encryption_enabled: bool = True
    authentication_required: bool = True
    audit_logging: bool = True
    privacy_budget: float = 1.0
    differential_privacy: bool = False


@dataclass
class SystemConfig:
    """Main system configuration"""
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # General settings
    log_level: str = "INFO"
    environment: str = "production"
    project_name: str = "ai_fabric_core"
    version: str = "1.0.0"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'SystemConfig':
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        return cls._from_dict(config_data)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'SystemConfig':
        """Create configuration from dictionary"""
        return cls._from_dict(config_dict)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'SystemConfig':
        """Internal method to create config from dict"""
        hardware = HardwareConfig(**data.get('hardware', {}))
        distributed = DistributedConfig(**data.get('distributed', {}))
        data_config = DataConfig(**data.get('data', {}))
        inference = InferenceConfig(**data.get('inference', {}))
        security = SecurityConfig(**data.get('security', {}))
        
        return cls(
            hardware=hardware,
            distributed=distributed,
            data=data_config,
            inference=inference,
            security=security,
            log_level=data.get('log_level', 'INFO'),
            environment=data.get('environment', 'production'),
            project_name=data.get('project_name', 'ai_fabric_core'),
            version=data.get('version', '1.0.0')
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'hardware': {
                'gpu_type': self.hardware.gpu_type,
                'num_gpus': self.hardware.num_gpus,
                'memory_gb': self.hardware.memory_gb,
                'interconnect': self.hardware.interconnect,
                'cpu_cores': self.hardware.cpu_cores,
                'ram_gb': self.hardware.ram_gb,
            },
            'distributed': {
                'num_nodes': self.distributed.num_nodes,
                'workers_per_node': self.distributed.workers_per_node,
                'gradient_sync_strategy': self.distributed.gradient_sync_strategy,
                'communication_backend': self.distributed.communication_backend,
                'checkpoint_interval': self.distributed.checkpoint_interval,
                'mixed_precision': self.distributed.mixed_precision,
                'precision_format': self.distributed.precision_format,
            },
            'data': {
                'batch_size': self.data.batch_size,
                'num_workers': self.data.num_workers,
                'prefetch_factor': self.data.prefetch_factor,
                'pin_memory': self.data.pin_memory,
                'persistent_workers': self.data.persistent_workers,
            },
            'inference': {
                'max_batch_size': self.inference.max_batch_size,
                'max_concurrent_requests': self.inference.max_concurrent_requests,
                'timeout_seconds': self.inference.timeout_seconds,
                'quantization_enabled': self.inference.quantization_enabled,
                'quantization_bits': self.inference.quantization_bits,
            },
            'security': {
                'encryption_enabled': self.security.encryption_enabled,
                'authentication_required': self.security.authentication_required,
                'audit_logging': self.security.audit_logging,
                'privacy_budget': self.security.privacy_budget,
                'differential_privacy': self.security.differential_privacy,
            },
            'log_level': self.log_level,
            'environment': self.environment,
            'project_name': self.project_name,
            'version': self.version,
        }
    
    def save_to_yaml(self, config_path: str) -> None:
        """Save configuration to YAML file"""
        with open(config_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def validate(self) -> bool:
        """Validate configuration settings"""
        if self.hardware.num_gpus < 1:
            raise ValueError("Number of GPUs must be at least 1")
        
        if self.distributed.num_nodes < 1:
            raise ValueError("Number of nodes must be at least 1")
        
        if self.data.batch_size < 1:
            raise ValueError("Batch size must be at least 1")
        
        if self.inference.max_batch_size < 1:
            raise ValueError("Max batch size must be at least 1")
        
        if self.security.privacy_budget <= 0:
            raise ValueError("Privacy budget must be positive")
        
        return True
    
    def __repr__(self) -> str:
        return (f"SystemConfig(project={self.project_name}, "
                f"environment={self.environment}, "
                f"gpus={self.hardware.num_gpus}, "
                f"nodes={self.distributed.num_nodes})")
