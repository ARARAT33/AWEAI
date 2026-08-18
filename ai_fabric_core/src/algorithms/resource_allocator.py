"""
Dynamic Resource Allocator (DRA)
Դինամիկ ռեսուրսների բաշխում

Intelligent resource allocation for AI workloads.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Workload:
    """AI workload definition"""
    id: str
    type: str  # training, inference, preprocessing
    priority: int
    gpu_required: int
    memory_gb: int
    estimated_duration_sec: int
    submitted_at: datetime = None
    
    def __post_init__(self):
        if self.submitted_at is None:
            self.submitted_at = datetime.now()


@dataclass
class ResourceAllocation:
    """Resource allocation record"""
    workload_id: str
    gpu_ids: List[str]
    memory_allocated_gb: int
    start_time: datetime
    end_time: Optional[datetime] = None


class DynamicResourceAllocator:
    """
    Dynamic Resource Allocator
    
    Features:
    - Priority-based scheduling
    - Resource prediction
    - Load balancing
    - Cost optimization
    """
    
    def __init__(
        self,
        total_gpus: int = 8,
        total_memory_gb: int = 640,
        allocation_strategy: str = 'priority',
    ):
        self.total_gpus = total_gpus
        self.total_memory_gb = total_memory_gb
        self.allocation_strategy = allocation_strategy
        
        self.logger = logging.getLogger(__name__)
        
        self.available_gpus = list(range(total_gpus))
        self.available_memory_gb = total_memory_gb
        self.allocations: Dict[str, ResourceAllocation] = {}
        self.pending_workloads: List[Workload] = []
        self.completed_workloads: List[Workload] = []
        
        # Statistics
        self.stats = {
            'total_allocations': 0,
            'total_deallocations': 0,
            'average_wait_time_sec': 0.0,
            'gpu_utilization_percent': 0.0,
        }
    
    def submit_workload(self, workload: Workload) -> str:
        """Submit workload for execution"""
        self.pending_workloads.append(workload)
        self.logger.info(f"Submitted workload: {workload.id} ({workload.type})")
        
        # Try to allocate immediately
        self._try_allocate()
        
        return workload.id
    
    def _try_allocate(self) -> None:
        """Try to allocate resources for pending workloads"""
        if not self.pending_workloads:
            return
        
        # Sort by strategy
        if self.allocation_strategy == 'priority':
            self.pending_workloads.sort(key=lambda w: -w.priority)
        elif self.allocation_strategy == 'fifo':
            self.pending_workloads.sort(key=lambda w: w.submitted_at)
        elif self.allocation_strategy == 'shortest':
            self.pending_workloads.sort(key=lambda w: w.estimated_duration_sec)
        
        allocated = []
        
        for workload in self.pending_workloads:
            if self._can_allocate(workload):
                self._allocate(workload)
                allocated.append(workload)
        
        # Remove allocated from pending
        for w in allocated:
            self.pending_workloads.remove(w)
    
    def _can_allocate(self, workload: Workload) -> bool:
        """Check if resources are available"""
        return (len(self.available_gpus) >= workload.gpu_required and
                self.available_memory_gb >= workload.memory_gb)
    
    def _allocate(self, workload: Workload) -> ResourceAllocation:
        """Allocate resources for workload"""
        # Allocate GPUs
        gpu_ids = self.available_gpus[:workload.gpu_required]
        self.available_gpus = self.available_gpus[workload.gpu_required:]
        
        # Allocate memory
        self.available_memory_gb -= workload.memory_gb
        
        # Create allocation
        allocation = ResourceAllocation(
            workload_id=workload.id,
            gpu_ids=[f"gpu_{i}" for i in gpu_ids],
            memory_allocated_gb=workload.memory_gb,
            start_time=datetime.now(),
        )
        
        self.allocations[workload.id] = allocation
        self.stats['total_allocations'] += 1
        
        self.logger.info(
            f"Allocated {workload.gpu_required} GPUs "
            f"and {workload.memory_gb}GB for {workload.id}"
        )
        
        return allocation
    
    def deallocate(self, workload_id: str) -> bool:
        """Release resources after workload completion"""
        if workload_id not in self.allocations:
            return False
        
        allocation = self.allocations[workload_id]
        
        # Find original workload
        workload = None
        for w in self.pending_workloads + self.completed_workloads:
            if w.id == workload_id:
                workload = w
                break
        
        if workload:
            # Release GPUs
            for gpu_str in allocation.gpu_ids:
                gpu_id = int(gpu_str.split('_')[1])
                if gpu_id not in self.available_gpus:
                    self.available_gpus.append(gpu_id)
            
            # Release memory
            self.available_memory_gb += allocation.memory_allocated_gb
            
            # Move to completed
            self.completed_workloads.append(workload)
            del self.allocations[workload_id]
            
            self.stats['total_deallocations'] += 1
            
            self.logger.info(f"Deallocated resources for {workload_id}")
            
            # Try to allocate pending workloads
            self._try_allocate()
            
            return True
        
        return False
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get current resource status"""
        allocated_gpus = self.total_gpus - len(self.available_gpus)
        allocated_memory = self.total_memory_gb - self.available_memory_gb
        
        return {
            'gpus': {
                'total': self.total_gpus,
                'available': len(self.available_gpus),
                'allocated': allocated_gpus,
                'utilization_percent': (allocated_gpus / self.total_gpus * 100) 
                                       if self.total_gpus > 0 else 0,
            },
            'memory_gb': {
                'total': self.total_memory_gb,
                'available': self.available_memory_gb,
                'allocated': allocated_memory,
                'utilization_percent': (allocated_memory / self.total_memory_gb * 100)
                                       if self.total_memory_gb > 0 else 0,
            },
            'pending_workloads': len(self.pending_workloads),
            'active_allocations': len(self.allocations),
        }
    
    def predict_wait_time(self, workload: Workload) -> float:
        """Predict wait time for workload"""
        if self._can_allocate(workload):
            return 0.0
        
        # Estimate based on pending workloads
        total_pending_compute = sum(
            w.estimated_duration_sec for w in self.pending_workloads
        )
        
        avg_duration = total_pending_compute / len(self.pending_workloads) \
                       if self.pending_workloads else 0
        
        return avg_duration
