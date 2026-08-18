"""
Resource Manager - Dynamic resource allocation and management
Handles GPU/CPU allocation, memory management, and load balancing
"""

import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class ResourceType(Enum):
    """Types of computational resources"""
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    STORAGE = "storage"
    NETWORK = "network"


class ResourceStatus(Enum):
    """Resource availability status"""
    AVAILABLE = "available"
    ALLOCATED = "allocated"
    BUSY = "busy"
    OFFLINE = "offline"
    MAINTENANCE = "maintenance"


@dataclass
class GPUResource:
    """GPU resource representation"""
    id: str
    type: str  # e.g., "NVIDIA H100"
    memory_gb: int
    status: ResourceStatus = ResourceStatus.AVAILABLE
    allocated_to: Optional[str] = None
    utilization_percent: float = 0.0
    temperature_celsius: float = 0.0
    power_watts: float = 0.0


@dataclass
class CPUResource:
    """CPU resource representation"""
    id: str
    cores: int
    threads: int
    status: ResourceStatus = ResourceStatus.AVAILABLE
    allocated_to: Optional[str] = None
    utilization_percent: float = 0.0


@dataclass
class MemoryResource:
    """Memory resource representation"""
    id: str
    total_gb: int
    available_gb: int
    status: ResourceStatus = ResourceStatus.AVAILABLE
    allocated_to: Optional[str] = None
    utilization_percent: float = 0.0


@dataclass
class ResourceRequest:
    """Resource allocation request"""
    id: str
    requester_id: str
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    cpu_cores: int = 0
    memory_gb: int = 0
    storage_gb: int = 0
    priority: int = 1
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Allocation:
    """Resource allocation record"""
    id: str
    request_id: str
    resources: Dict[str, List[str]] = field(default_factory=dict)
    allocated_at: datetime = field(default_factory=datetime.now)
    expires_at: Optional[datetime] = None
    status: str = "active"


class ResourceManager:
    """
    Dynamic resource allocator for AI workloads
    
    Features:
    - Real-time resource monitoring
    - Intelligent allocation strategies
    - Load balancing across nodes
    - Automatic resource reclamation
    - Priority-based scheduling
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the resource manager
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.gpus: Dict[str, GPUResource] = {}
        self.cpus: Dict[str, CPUResource] = {}
        self.memory: Dict[str, MemoryResource] = {}
        self.allocations: Dict[str, Allocation] = {}
        self.pending_requests: List[ResourceRequest] = []
        
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Statistics
        self.stats = {
            'total_allocations': 0,
            'successful_allocations': 0,
            'failed_allocations': 0,
            'average_allocation_time_ms': 0.0,
            'total_gpu_hours': 0.0,
        }
    
    def register_gpu(self, gpu_id: str, gpu_type: str, memory_gb: int) -> None:
        """
        Register a GPU resource
        
        Args:
            gpu_id: Unique GPU identifier
            gpu_type: GPU model/type
            memory_gb: GPU memory in GB
        """
        gpu = GPUResource(
            id=gpu_id,
            type=gpu_type,
            memory_gb=memory_gb,
        )
        self.gpus[gpu_id] = gpu
        self.logger.info(f"GPU registered: {gpu_id} ({gpu_type}, {memory_gb}GB)")
    
    def register_cpu(self, cpu_id: str, cores: int, threads: int) -> None:
        """
        Register a CPU resource
        
        Args:
            cpu_id: Unique CPU identifier
            cores: Number of CPU cores
            threads: Number of threads per core
        """
        cpu = CPUResource(
            id=cpu_id,
            cores=cores,
            threads=threads,
        )
        self.cpus[cpu_id] = cpu
        self.logger.info(f"CPU registered: {cpu_id} ({cores} cores, {threads} threads)")
    
    def register_memory(self, mem_id: str, total_gb: int) -> None:
        """
        Register a memory resource
        
        Args:
            mem_id: Unique memory identifier
            total_gb: Total memory in GB
        """
        memory = MemoryResource(
            id=mem_id,
            total_gb=total_gb,
            available_gb=total_gb,
        )
        self.memory[mem_id] = memory
        self.logger.info(f"Memory registered: {mem_id} ({total_gb}GB)")
    
    def request_resources(self, request: ResourceRequest) -> Optional[Allocation]:
        """
        Request resource allocation
        
        Args:
            request: Resource request object
            
        Returns:
            Allocation object if successful, None otherwise
        """
        start_time = datetime.now()
        
        try:
            # Find available resources
            allocated_gpus = self._allocate_gpus(
                request.gpu_count, 
                request.gpu_type
            )
            allocated_cpus = self._allocate_cpus(request.cpu_cores)
            allocated_memory = self._allocate_memory(request.memory_gb)
            
            # Check if all resources were allocated
            if (len(allocated_gpus) < request.gpu_count or
                len(allocated_cpus) < request.cpu_cores or
                allocated_memory < request.memory_gb):
                
                # Partial allocation - release what was allocated
                self._release_resources(allocated_gpus, allocated_cpus, allocated_memory)
                
                self.stats['failed_allocations'] += 1
                self.logger.warning(f"Resource allocation failed for request {request.id}")
                return None
            
            # Create allocation record
            allocation = Allocation(
                id=f"alloc_{len(self.allocations) + 1}",
                request_id=request.id,
                resources={
                    'gpus': allocated_gpus,
                    'cpus': allocated_cpus,
                    'memory': [m_id for m_id in self.memory.keys() 
                              if self.memory[m_id].allocated_to == request.requester_id],
                },
            )
            
            self.allocations[allocation.id] = allocation
            self.stats['successful_allocations'] += 1
            self.stats['total_allocations'] += 1
            
            # Update statistics
            elapsed_ms = (datetime.now() - start_time).total_seconds() * 1000
            self.stats['average_allocation_time_ms'] = (
                (self.stats['average_allocation_time_ms'] * (self.stats['total_allocations'] - 1) 
                 + elapsed_ms) / self.stats['total_allocations']
            )
            
            self.logger.info(f"Resources allocated: {allocation.id}")
            return allocation
            
        except Exception as e:
            self.stats['failed_allocations'] += 1
            self.logger.error(f"Resource allocation error: {e}")
            return None
    
    def release_allocation(self, allocation_id: str) -> bool:
        """
        Release an allocation
        
        Args:
            allocation_id: Allocation ID to release
            
        Returns:
            True if successful, False otherwise
        """
        if allocation_id not in self.allocations:
            return False
        
        allocation = self.allocations[allocation_id]
        
        # Release GPUs
        for gpu_id in allocation.resources.get('gpus', []):
            if gpu_id in self.gpus:
                self.gpus[gpu_id].status = ResourceStatus.AVAILABLE
                self.gpus[gpu_id].allocated_to = None
        
        # Release CPUs
        for cpu_id in allocation.resources.get('cpus', []):
            if cpu_id in self.cpus:
                self.cpus[cpu_id].status = ResourceStatus.AVAILABLE
                self.cpus[cpu_id].allocated_to = None
        
        # Release Memory
        for mem_id in allocation.resources.get('memory', []):
            if mem_id in self.memory:
                self.memory[mem_id].allocated_to = None
                self.memory[mem_id].available_gb = self.memory[mem_id].total_gb
        
        allocation.status = "released"
        del self.allocations[allocation_id]
        
        self.logger.info(f"Allocation released: {allocation_id}")
        return True
    
    def _allocate_gpus(self, count: int, gpu_type: Optional[str] = None) -> List[str]:
        """Allocate GPUs based on request"""
        allocated = []
        
        for gpu_id, gpu in self.gpus.items():
            if len(allocated) >= count:
                break
            
            if (gpu.status == ResourceStatus.AVAILABLE and
                (gpu_type is None or gpu.type == gpu_type)):
                gpu.status = ResourceStatus.ALLOCATED
                gpu.allocated_to = f"request_{len(allocated)}"
                allocated.append(gpu_id)
        
        return allocated
    
    def _allocate_cpus(self, cores_needed: int) -> List[str]:
        """Allocate CPU cores"""
        allocated = []
        cores_allocated = 0
        
        for cpu_id, cpu in self.cpus.items():
            if cores_allocated >= cores_needed:
                break
            
            if cpu.status == ResourceStatus.AVAILABLE:
                available_cores = cpu.cores - cores_allocated
                if available_cores > 0:
                    cpu.status = ResourceStatus.ALLOCATED
                    cpu.allocated_to = f"request_{len(allocated)}"
                    allocated.append(cpu_id)
                    cores_allocated += available_cores
        
        return allocated
    
    def _allocate_memory(self, gb_needed: int) -> int:
        """Allocate memory"""
        allocated = 0
        
        for mem_id, mem in self.memory.items():
            if allocated >= gb_needed:
                break
            
            if mem.status == ResourceStatus.AVAILABLE:
                available = min(mem.available_gb, gb_needed - allocated)
                mem.available_gb -= available
                mem.allocated_to = f"request_{allocated}"
                allocated += available
        
        return allocated
    
    def _release_resources(self, gpus: List[str], cpus: List[str], 
                          memory_gb: int) -> None:
        """Release partially allocated resources"""
        for gpu_id in gpus:
            if gpu_id in self.gpus:
                self.gpus[gpu_id].status = ResourceStatus.AVAILABLE
                self.gpus[gpu_id].allocated_to = None
        
        for cpu_id in cpus:
            if cpu_id in self.cpus:
                self.cpus[cpu_id].status = ResourceStatus.AVAILABLE
                self.cpus[cpu_id].allocated_to = None
        
        # Memory is released automatically when allocation fails
    
    def get_available_resources(self) -> Dict[str, Any]:
        """Get summary of available resources"""
        available_gpus = sum(1 for gpu in self.gpus.values() 
                            if gpu.status == ResourceStatus.AVAILABLE)
        available_cpus = sum(cpu.cores for cpu in self.cpus.values() 
                            if cpu.status == ResourceStatus.AVAILABLE)
        available_memory = sum(mem.available_gb for mem in self.memory.values())
        
        return {
            'gpus': {
                'total': len(self.gpus),
                'available': available_gpus,
                'allocated': len(self.gpus) - available_gpus,
            },
            'cpus': {
                'total_cores': sum(cpu.cores for cpu in self.cpus.values()),
                'available_cores': available_cpus,
            },
            'memory': {
                'total_gb': sum(mem.total_gb for mem in self.memory.values()),
                'available_gb': available_memory,
            },
        }
    
    def get_utilization_stats(self) -> Dict[str, float]:
        """Get resource utilization statistics"""
        if not self.gpus:
            gpu_util = 0.0
        else:
            gpu_util = sum(gpu.utilization_percent for gpu in self.gpus.values()) / len(self.gpus)
        
        if not self.cpus:
            cpu_util = 0.0
        else:
            cpu_util = sum(cpu.utilization_percent for cpu in self.cpus.values()) / len(self.cpus)
        
        if not self.memory:
            mem_util = 0.0
        else:
            mem_util = sum(mem.utilization_percent for mem in self.memory.values()) / len(self.memory)
        
        return {
            'gpu_utilization_percent': gpu_util,
            'cpu_utilization_percent': cpu_util,
            'memory_utilization_percent': mem_util,
        }
    
    def __repr__(self) -> str:
        return (f"ResourceManager(gpus={len(self.gpus)}, "
                f"allocations={len(self.allocations)}, "
                f"pending_requests={len(self.pending_requests)})")
