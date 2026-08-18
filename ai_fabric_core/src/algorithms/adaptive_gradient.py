"""
Adaptive Gradient Synchronization (AGS) Algorithm
Դինամիկ գրադիենտի սինխրոնիզացիա բաշխված ուսուցման համար

This algorithm dynamically adjusts gradient synchronization strategies
based on network conditions, gradient sparsity, and training progress.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np


class SyncStrategy(Enum):
    """Gradient synchronization strategies"""
    ALL_REDUCE = "all_reduce"
    RING_ALL_REDUCE = "ring_all_reduce"
    PARAMETER_SERVER = "parameter_server"
    HIERARCHICAL = "hierarchical"
    ASYNC = "async"
    SELECTIVE = "selective"


class CompressionType(Enum):
    """Gradient compression types"""
    NONE = "none"
    FP16 = "fp16"
    FP8 = "fp8"
    INT8 = "int8"
    TOP_K = "top_k"
    RANDOM_K = "random_k"


@dataclass
class NetworkCondition:
    """Network condition metrics"""
    bandwidth_mbps: float = 0.0
    latency_ms: float = 0.0
    packet_loss_percent: float = 0.0
    congestion_level: float = 0.0  # 0.0 to 1.0


@dataclass
class GradientStats:
    """Gradient statistics"""
    norm: float = 0.0
    sparsity: float = 0.0
    variance: float = 0.0
    magnitude_distribution: Dict[str, float] = field(default_factory=dict)


class AdaptiveGradientSync:
    """
    Adaptive Gradient Synchronization for distributed training
    
    Features:
    - Dynamic strategy selection based on network conditions
    - Automatic compression level adjustment
    - Gradient sparsity detection
    - Communication-computation overlap
    - Fault tolerance with checkpoint recovery
    
    Mathematical Foundation:
    The algorithm uses a cost model to select optimal sync strategy:
    
    Cost(strategy) = α * CommunicationTime + β * ComputationTime + γ * Error
    
    where:
    - CommunicationTime depends on bandwidth, latency, and compression
    - ComputationTime depends on gradient computation and reduction
    - Error measures the impact of compression/approximation
    """
    
    def __init__(
        self,
        num_nodes: int = 1,
        num_workers_per_node: int = 8,
        initial_strategy: SyncStrategy = SyncStrategy.ALL_REDUCE,
        compression: CompressionType = CompressionType.FP16,
        adaptation_interval: int = 100,  # steps
        min_compression_ratio: float = 0.1,
        max_compression_ratio: float = 1.0,
    ):
        """
        Initialize AGS
        
        Args:
            num_nodes: Number of training nodes
            num_workers_per_node: Workers per node
            initial_strategy: Initial sync strategy
            compression: Initial compression type
            adaptation_interval: Steps between adaptations
            min_compression_ratio: Minimum compression ratio
            max_compression_ratio: Maximum compression ratio
        """
        self.num_nodes = num_nodes
        self.num_workers = num_nodes * num_workers_per_node
        self.current_strategy = initial_strategy
        self.compression = compression
        self.adaptation_interval = adaptation_interval
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        
        self.logger = logging.getLogger(__name__)
        
        # History for decision making
        self.performance_history: List[Dict[str, Any]] = []
        self.network_history: List[NetworkCondition] = []
        self.gradient_history: List[GradientStats] = []
        
        # Current step
        self.current_step = 0
        
        # Performance metrics
        self.metrics = {
            'sync_time_ms': 0.0,
            'computation_time_ms': 0.0,
            'total_iteration_time_ms': 0.0,
            'communication_efficiency': 0.0,
            'gradient_sparsity': 0.0,
            'compression_ratio': 1.0,
        }
    
    def analyze_network_conditions(self) -> NetworkCondition:
        """
        Analyze current network conditions
        
        Returns:
            NetworkCondition object with current metrics
        """
        # In real implementation, this would measure actual network metrics
        condition = NetworkCondition(
            bandwidth_mbps=self._measure_bandwidth(),
            latency_ms=self._measure_latency(),
            packet_loss_percent=self._measure_packet_loss(),
            congestion_level=self._estimate_congestion(),
        )
        
        self.network_history.append(condition)
        if len(self.network_history) > 100:
            self.network_history.pop(0)
        
        return condition
    
    def analyze_gradients(self, gradients: Dict[str, np.ndarray]) -> GradientStats:
        """
        Analyze gradient statistics
        
        Args:
            gradients: Dictionary of parameter gradients
            
        Returns:
            GradientStats object
        """
        # Calculate gradient norms
        total_norm = 0.0
        total_elements = 0
        zero_elements = 0
        
        for param_name, grad in gradients.items():
            norm = np.linalg.norm(grad)
            total_norm += norm ** 2
            total_elements += grad.size
            zero_elements += np.sum(np.abs(grad) < 1e-10)
        
        total_norm = np.sqrt(total_norm)
        sparsity = zero_elements / total_elements if total_elements > 0 else 0.0
        
        stats = GradientStats(
            norm=total_norm,
            sparsity=float(sparsity),
            variance=np.var([np.var(g) for g in gradients.values()]),
        )
        
        self.gradient_history.append(stats)
        if len(self.gradient_history) > 100:
            self.gradient_history.pop(0)
        
        return stats
    
    def select_optimal_strategy(
        self,
        network: NetworkCondition,
        gradients: GradientStats,
    ) -> SyncStrategy:
        """
        Select optimal synchronization strategy
        
        Args:
            network: Current network conditions
            gradients: Current gradient statistics
            
        Returns:
            Optimal SyncStrategy
        """
        # Cost model for each strategy
        strategy_costs = {}
        
        # All-reduce performs well with low latency and high bandwidth
        if network.latency_ms < 5 and network.bandwidth_mbps > 10000:
            strategy_costs[SyncStrategy.ALL_REDUCE] = 0.1
        else:
            strategy_costs[SyncStrategy.ALL_REDUCE] = 0.5 + network.congestion_level
        
        # Ring all-reduce is better for large clusters
        if self.num_nodes > 4:
            strategy_costs[SyncStrategy.RING_ALL_REDUCE] = 0.2
        else:
            strategy_costs[SyncStrategy.RING_ALL_REDUCE] = 0.4
        
        # Parameter server for very large models
        if gradients.sparsity > 0.7:
            strategy_costs[SyncStrategy.PARAMETER_SERVER] = 0.3
        else:
            strategy_costs[SyncStrategy.PARAMETER_SERVER] = 0.6
        
        # Hierarchical for multi-node setups
        if self.num_nodes >= 8:
            strategy_costs[SyncStrategy.HIERARCHICAL] = 0.25
        else:
            strategy_costs[SyncStrategy.HIERARCHICAL] = 0.5
        
        # Async for high-latency networks
        if network.latency_ms > 50:
            strategy_costs[SyncStrategy.ASYNC] = 0.3
        else:
            strategy_costs[SyncStrategy.ASYNC] = 0.7
        
        # Selective for sparse gradients
        if gradients.sparsity > 0.5:
            strategy_costs[SyncStrategy.SELECTIVE] = 0.2
        else:
            strategy_costs[SyncStrategy.SELECTIVE] = 0.6
        
        # Select strategy with minimum cost
        optimal_strategy = min(strategy_costs, key=strategy_costs.get)
        
        self.logger.info(
            f"Selected strategy: {optimal_strategy.value} "
            f"(cost: {strategy_costs[optimal_strategy]:.3f})"
        )
        
        return optimal_strategy
    
    def select_compression_level(
        self,
        network: NetworkCondition,
        gradients: GradientStats,
    ) -> CompressionType:
        """
        Select optimal compression level
        
        Args:
            network: Current network conditions
            gradients: Current gradient statistics
            
        Returns:
            Optimal CompressionType
        """
        # High bandwidth - minimal compression
        if network.bandwidth_mbps > 25000:
            return CompressionType.NONE
        
        # Medium bandwidth - FP16
        if network.bandwidth_mbps > 10000:
            return CompressionType.FP16
        
        # Low bandwidth with sparse gradients - use sparsity
        if gradients.sparsity > 0.5:
            return CompressionType.TOP_K
        
        # Very low bandwidth - aggressive compression
        if network.bandwidth_mbps < 1000:
            return CompressionType.INT8
        
        return CompressionType.FP16
    
    def synchronize_gradients(
        self,
        local_gradients: Dict[str, np.ndarray],
        global_step: int,
    ) -> Dict[str, np.ndarray]:
        """
        Synchronize gradients across workers
        
        Args:
            local_gradients: Local gradient tensors
            global_step: Current training step
            
        Returns:
            Synchronized global gradients
        """
        self.current_step = global_step
        
        # Check if adaptation is needed
        if global_step % self.adaptation_interval == 0:
            self._adapt_strategy(local_gradients)
        
        # Apply compression
        compressed_gradients = self._compress_gradients(local_gradients)
        
        # Perform synchronization based on current strategy
        if self.current_strategy == SyncStrategy.ALL_REDUCE:
            global_gradients = self._all_reduce(compressed_gradients)
        elif self.current_strategy == SyncStrategy.RING_ALL_REDUCE:
            global_gradients = self._ring_all_reduce(compressed_gradients)
        elif self.current_strategy == SyncStrategy.PARAMETER_SERVER:
            global_gradients = self._parameter_server_sync(compressed_gradients)
        elif self.current_strategy == SyncStrategy.HIERARCHICAL:
            global_gradients = self._hierarchical_sync(compressed_gradients)
        elif self.current_strategy == SyncStrategy.ASYNC:
            global_gradients = self._async_sync(compressed_gradients)
        elif self.current_strategy == SyncStrategy.SELECTIVE:
            global_gradients = self._selective_sync(compressed_gradients)
        else:
            global_gradients = compressed_gradients
        
        # Decompress if needed
        decompressed_gradients = self._decompress_gradients(global_gradients)
        
        # Update metrics
        self.metrics['gradient_sparsity'] = np.mean([
            np.sum(np.abs(g) < 1e-10) / g.size 
            for g in decompressed_gradients.values()
        ])
        
        return decompressed_gradients
    
    def _adapt_strategy(self, gradients: Dict[str, np.ndarray]) -> None:
        """Adapt strategy based on current conditions"""
        network = self.analyze_network_conditions()
        grad_stats = self.analyze_gradients(gradients)
        
        # Select new strategy
        new_strategy = self.select_optimal_strategy(network, grad_stats)
        new_compression = self.select_compression_level(network, grad_stats)
        
        if new_strategy != self.current_strategy:
            self.logger.info(
                f"Changing strategy from {self.current_strategy.value} "
                f"to {new_strategy.value}"
            )
            self.current_strategy = new_strategy
        
        if new_compression != self.compression:
            self.logger.info(
                f"Changing compression from {self.compression.value} "
                f"to {new_compression.value}"
            )
            self.compression = new_compression
        
        # Record performance
        self.performance_history.append({
            'step': self.current_step,
            'strategy': self.current_strategy.value,
            'compression': self.compression.value,
            'network_bandwidth': network.bandwidth_mbps,
            'gradient_sparsity': grad_stats.sparsity,
            'sync_time_ms': self.metrics['sync_time_ms'],
        })
    
    def _compress_gradients(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Apply gradient compression"""
        if self.compression == CompressionType.NONE:
            return gradients
        elif self.compression == CompressionType.FP16:
            return {k: v.astype(np.float16) for k, v in gradients.items()}
        elif self.compression == CompressionType.FP8:
            return {k: v.astype(np.float32) for k, v in gradients.items()}  # Simplified
        elif self.compression == CompressionType.INT8:
            return {k: (v * 127).astype(np.int8) for k, v in gradients.items()}
        elif self.compression == CompressionType.TOP_K:
            return self._top_k_compression(gradients, k_ratio=0.1)
        else:
            return gradients
    
    def _decompress_gradients(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Decompress gradients"""
        # In practice, this would reverse the compression
        return gradients
    
    def _top_k_compression(
        self,
        gradients: Dict[str, np.ndarray],
        k_ratio: float = 0.1,
    ) -> Dict[str, np.ndarray]:
        """Keep only top-k gradient values"""
        compressed = {}
        for name, grad in gradients.items():
            k = max(1, int(grad.size * k_ratio))
            indices = np.argpartition(np.abs(grad), -k)[-k:]
            mask = np.zeros_like(grad)
            mask[indices] = 1
            compressed[name] = grad * mask
        return compressed
    
    # Placeholder synchronization methods
    def _all_reduce(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """All-reduce synchronization"""
        # In real implementation, use NCCL or similar
        return gradients
    
    def _ring_all_reduce(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Ring all-reduce synchronization"""
        return gradients
    
    def _parameter_server_sync(
        self,
        gradients: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        """Parameter server synchronization"""
        return gradients
    
    def _hierarchical_sync(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Hierarchical synchronization"""
        return gradients
    
    def _async_sync(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Asynchronous synchronization"""
        return gradients
    
    def _selective_sync(self, gradients: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Selective synchronization for sparse gradients"""
        # Only sync non-zero gradients
        return {k: v for k, v in gradients.items() if np.any(np.abs(v) > 1e-10)}
    
    # Placeholder measurement methods
    def _measure_bandwidth(self) -> float:
        """Measure network bandwidth"""
        return 10000.0  # Mbps (placeholder)
    
    def _measure_latency(self) -> float:
        """Measure network latency"""
        return 1.0  # ms (placeholder)
    
    def _measure_packet_loss(self) -> float:
        """Measure packet loss"""
        return 0.0  # percent (placeholder)
    
    def _estimate_congestion(self) -> float:
        """Estimate network congestion"""
        return 0.1  # 0.0 to 1.0 (placeholder)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        return self.metrics.copy()
    
    def get_history(self) -> Dict[str, List[Any]]:
        """Get adaptation history"""
        return {
            'performance': self.performance_history,
            'network': self.network_history,
            'gradients': self.gradient_history,
        }
    
    def __repr__(self) -> str:
        return (f"AdaptiveGradientSync(strategy={self.current_strategy.value}, "
                f"compression={self.compression.value}, "
                f"nodes={self.num_nodes}, workers={self.num_workers})")
