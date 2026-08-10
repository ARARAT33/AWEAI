"""Cluster node manager with health monitoring and resource scheduling."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set


class NodeType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    NPU = "npu"
    IPU = "ipu"
    CUSTOM = "custom"


@dataclass
class ResourceSpec:
    cpu_cores: float = 0.0
    memory_gb: float = 0.0
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    gpu_vram_gb: float = 0.0
    tpu_count: int = 0
    tpu_type: Optional[str] = None
    npu_count: int = 0
    ipu_count: int = 0
    custom_resources: Dict[str, float] = field(default_factory=dict)
    disk_gb: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "gpu_vram_gb": self.gpu_vram_gb,
            "tpu_count": self.tpu_count,
            "tpu_type": self.tpu_type,
            "npu_count": self.npu_count,
            "ipu_count": self.ipu_count,
            "custom_resources": self.custom_resources,
            "disk_gb": self.disk_gb,
        }


@dataclass
class NodeInfo:
    node_id: str
    name: str
    node_type: NodeType
    host: str
    port: int = 22
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    allocated_resources: ResourceSpec = field(default_factory=ResourceSpec)
    labels: Dict[str, str] = field(default_factory=dict)
    taints: List[str] = field(default_factory=list)
    status: str = "ready"
    health_status: str = "healthy"
    last_heartbeat: float = field(default_factory=time.time)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3
    created_at: float = field(default_factory=time.time)

    def available_resources(self) -> ResourceSpec:
        cpu = self.resources.cpu_cores - self.allocated_resources.cpu_cores
        mem = self.resources.memory_gb - self.allocated_resources.memory_gb
        gpus = max(0, self.resources.gpu_count - self.allocated_resources.gpu_count)
        vram = max(0.0, self.resources.gpu_vram_gb - self.allocated_resources.gpu_vram_gb)
        tpus = max(0, self.resources.tpu_count - self.allocated_resources.tpu_count)
        npus = max(0, self.resources.npu_count - self.allocated_resources.npu_count)
        ipus = max(0, self.resources.ipu_count - self.allocated_resources.ipu_count)
        custom: Dict[str, float] = {}
        for k in self.resources.custom_resources:
            allocated = self.allocated_resources.custom_resources.get(k, 0.0)
            custom[k] = max(0.0, self.resources.custom_resources[k] - allocated)
        disk = max(0.0, self.resources.disk_gb - self.allocated_resources.disk_gb)
        return ResourceSpec(
            cpu_cores=max(0.0, cpu),
            memory_gb=max(0.0, mem),
            gpu_count=gpus,
            gpu_type=self.resources.gpu_type,
            gpu_vram_gb=vram,
            tpu_count=tpus,
            tpu_type=self.resources.tpu_type,
            npu_count=npus,
            ipu_count=ipus,
            custom_resources=custom,
            disk_gb=disk,
        )

    def can_schedule(self, spec: ResourceSpec) -> bool:
        avail = self.available_resources()
        if spec.cpu_cores > avail.cpu_cores:
            return False
        if spec.memory_gb > avail.memory_gb:
            return False
        if spec.gpu_count > avail.gpu_count:
            return False
        if spec.gpu_type and avail.gpu_type and spec.gpu_type != avail.gpu_type:
            return False
        if spec.gpu_vram_gb > avail.gpu_vram_gb:
            return False
        if spec.tpu_count > avail.tpu_count:
            return False
        if spec.tpu_type and avail.tpu_type and spec.tpu_type != avail.tpu_type:
            return False
        if spec.npu_count > avail.npu_count:
            return False
        if spec.ipu_count > avail.ipu_count:
            return False
        if spec.disk_gb > avail.disk_gb:
            return False
        for k, v in spec.custom_resources.items():
            if v > avail.custom_resources.get(k, 0.0):
                return False
        return True

    def allocate(self, spec: ResourceSpec) -> None:
        self.allocated_resources.cpu_cores += spec.cpu_cores
        self.allocated_resources.memory_gb += spec.memory_gb
        self.allocated_resources.gpu_count += spec.gpu_count
        self.allocated_resources.gpu_vram_gb += spec.gpu_vram_gb
        self.allocated_resources.tpu_count += spec.tpu_count
        self.allocated_resources.npu_count += spec.npu_count
        self.allocated_resources.ipu_count += spec.ipu_count
        self.allocated_resources.disk_gb += spec.disk_gb
        for k, v in spec.custom_resources.items():
            self.allocated_resources.custom_resources[k] = (
                self.allocated_resources.custom_resources.get(k, 0.0) + v
            )

    def release(self, spec: ResourceSpec) -> None:
        self.allocated_resources.cpu_cores = max(0.0, self.allocated_resources.cpu_cores - spec.cpu_cores)
        self.allocated_resources.memory_gb = max(0.0, self.allocated_resources.memory_gb - spec.memory_gb)
        self.allocated_resources.gpu_count = max(0, self.allocated_resources.gpu_count - spec.gpu_count)
        self.allocated_resources.gpu_vram_gb = max(0.0, self.allocated_resources.gpu_vram_gb - spec.gpu_vram_gb)
        self.allocated_resources.tpu_count = max(0, self.allocated_resources.tpu_count - spec.tpu_count)
        self.allocated_resources.npu_count = max(0, self.allocated_resources.npu_count - spec.npu_count)
        self.allocated_resources.ipu_count = max(0, self.allocated_resources.ipu_count - spec.ipu_count)
        self.allocated_resources.disk_gb = max(0.0, self.allocated_resources.disk_gb - spec.disk_gb)
        for k, v in spec.custom_resources.items():
            current = self.allocated_resources.custom_resources.get(k, 0.0)
            self.allocated_resources.custom_resources[k] = max(0.0, current - v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "node_type": self.node_type.value,
            "host": self.host,
            "port": self.port,
            "resources": self.resources.to_dict(),
            "allocated_resources": self.allocated_resources.to_dict(),
            "labels": self.labels,
            "taints": self.taints,
            "status": self.status,
            "health_status": self.health_status,
            "last_heartbeat": self.last_heartbeat,
            "recovery_attempts": self.recovery_attempts,
            "created_at": self.created_at,
        }


@dataclass
class ResourcePool:
    total: ResourceSpec = field(default_factory=ResourceSpec)
    allocated: ResourceSpec = field(default_factory=ResourceSpec)
    node_count: int = 0
    healthy_node_count: int = 0
    gpu_node_count: int = 0
    cpu_node_count: int = 0
    tpu_node_count: int = 0

    def utilization(self) -> Dict[str, float]:
        def util(t: float, a: float) -> float:
            return round(a / t, 4) if t > 0 else 0.0
        return {
            "cpu": util(self.total.cpu_cores, self.allocated.cpu_cores),
            "memory": util(self.total.memory_gb, self.allocated.memory_gb),
            "gpu": util(float(self.total.gpu_count), float(self.allocated.gpu_count)),
            "gpu_vram": util(self.total.gpu_vram_gb, self.allocated.gpu_vram_gb),
            "tpu": util(float(self.total.tpu_count), float(self.allocated.tpu_count)),
            "disk": util(self.total.disk_gb, self.allocated.disk_gb),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total.to_dict(),
            "allocated": self.allocated.to_dict(),
            "node_count": self.node_count,
            "healthy_node_count": self.healthy_node_count,
            "gpu_node_count": self.gpu_node_count,
            "cpu_node_count": self.cpu_node_count,
            "tpu_node_count": self.tpu_node_count,
            "utilization": self.utilization(),
        }


class ClusterManager:
    def __init__(self, heartbeat_interval: float = 10.0, heartbeat_timeout: float = 30.0) -> None:
        self._nodes: Dict[str, NodeInfo] = {}
        self._heartbeat_interval = heartbeat_interval
        self._heartbeat_timeout = heartbeat_timeout
        self._running = False
        self._health_callbacks: List[Callable[[str, str], None]] = []
        self._recovery_callbacks: List[Callable[[str], None]] = []
        self._scheduled_jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = type("_Lock", (), {"acquire": lambda: None, "release": lambda: None})()

    def add_node(self, node: NodeInfo) -> None:
        self._nodes[node.node_id] = node
        self._refresh_pool()

    def remove_node(self, node_id: str) -> None:
        if node_id in self._nodes:
            self._release_all_for_node(node_id)
            del self._nodes[node_id]
            self._refresh_pool()

    def _release_all_for_node(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if not node:
            return
        released: List[ResourceSpec] = []
        for job_id, job in list(self._scheduled_jobs.items()):
            if job.get("node_id") == node_id:
                spec = job.get("resource_spec")
                if isinstance(spec, ResourceSpec):
                    released.append(spec)
                del self._scheduled_jobs[job_id]
        for spec in released:
            node.release(spec)

    def get_node(self, node_id: str) -> Optional[NodeInfo]:
        return self._nodes.get(node_id)

    def get_nodes_by_type(self, node_type: NodeType) -> List[NodeInfo]:
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_nodes_by_label(self, label_key: str, label_value: str) -> List[NodeInfo]:
        return [n for n in self._nodes.values() if n.labels.get(label_key) == label_value]

    def get_healthy_nodes(self) -> List[NodeInfo]:
        now = time.time()
        return [
            n for n in self._nodes.values()
            if n.health_status == "healthy" and (now - n.last_heartbeat) < self._heartbeat_timeout
        ]

    def get_cluster_status(self) -> Dict[str, Any]:
        pool = self._pool()
        now = time.time()
        nodes = []
        for n in self._nodes.values():
            nd = n.to_dict()
            nd["age_seconds"] = round(now - n.created_at, 2)
            nodes.append(nd)
        return {
            "running": self._running,
            "total_nodes": len(self._nodes),
            "healthy_nodes": pool.healthy_node_count,
            "resource_pool": pool.to_dict(),
            "scheduled_jobs": len(self._scheduled_jobs),
            "nodes": nodes,
        }

    def _pool(self) -> ResourcePool:
        pool = ResourcePool()
        now = time.time()
        for node in self._nodes.values():
            pool.node_count += 1
            is_healthy = (
                node.health_status == "healthy"
                and (now - node.last_heartbeat) < self._heartbeat_timeout
            )
            if is_healthy:
                pool.healthy_node_count += 1
                pool.total.cpu_cores += node.resources.cpu_cores
                pool.total.memory_gb += node.resources.memory_gb
                pool.total.gpu_count += node.resources.gpu_count
                pool.total.gpu_vram_gb += node.resources.gpu_vram_gb
                pool.total.tpu_count += node.resources.tpu_count
                pool.total.npu_count += node.resources.npu_count
                pool.total.ipu_count += node.resources.ipu_count
                pool.total.disk_gb += node.resources.disk_gb
                for k, v in node.resources.custom_resources.items():
                    pool.total.custom_resources[k] = pool.total.custom_resources.get(k, 0.0) + v
                pool.allocated.cpu_cores += node.allocated_resources.cpu_cores
                pool.allocated.memory_gb += node.allocated_resources.memory_gb
                pool.allocated.gpu_count += node.allocated_resources.gpu_count
                pool.allocated.gpu_vram_gb += node.allocated_resources.gpu_vram_gb
                pool.allocated.tpu_count += node.allocated_resources.tpu_count
                pool.allocated.npu_count += node.allocated_resources.npu_count
                pool.allocated.ipu_count += node.allocated_resources.ipu_count
                pool.allocated.disk_gb += node.allocated_resources.disk_gb
                for k, v in node.allocated_resources.custom_resources.items():
                    pool.allocated.custom_resources[k] = pool.allocated.custom_resources.get(k, 0.0) + v
            if node.node_type == NodeType.GPU:
                pool.gpu_node_count += 1
            if node.node_type == NodeType.CPU:
                pool.cpu_node_count += 1
            if node.node_type == NodeType.TPU:
                pool.tpu_node_count += 1
        return pool

    def _refresh_pool(self) -> None:
        pass

    def register_health_callback(self, callback: Callable[[str, str], None]) -> None:
        self._health_callbacks.append(callback)

    def register_recovery_callback(self, callback: Callable[[str], None]) -> None:
        self._recovery_callbacks.append(callback)

    def start(self) -> None:
        self._running = True

    def stop(self) -> None:
        self._running = False

    def heartbeat(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if node:
            node.last_heartbeat = time.time()
            if node.health_status != "healthy":
                node.health_status = "healthy"
                node.recovery_attempts = 0
                for cb in self._health_callbacks:
                    cb(node_id, "healthy")

    def check_health(self) -> Dict[str, str]:
        now = time.time()
        result: Dict[str, str] = {}
        for node_id, node in list(self._nodes.items()):
            elapsed = now - node.last_heartbeat
            if elapsed > self._heartbeat_timeout and node.health_status == "healthy":
                node.health_status = "unhealthy"
                for cb in self._health_callbacks:
                    cb(node_id, "unhealthy")
            if node.health_status == "unhealthy":
                result[node_id] = "unhealthy"
                if node.recovery_attempts < node.max_recovery_attempts:
                    node.recovery_attempts += 1
                    for cb in self._recovery_callbacks:
                        cb(node_id)
        return result

    def schedule(self, job_id: str, spec: ResourceSpec, constraints: Optional[Dict[str, Any]] = None) -> Optional[str]:
        constraints = constraints or {}
        candidates = self._healthy_candidates(spec, constraints)
        if not candidates:
            return None
        selected = self._best_candidate(candidates, spec, constraints)
        if selected is None:
            return None
        selected.allocate(spec)
        self._scheduled_jobs[job_id] = {
            "node_id": selected.node_id,
            "resource_spec": spec,
            "constraints": constraints,
            "scheduled_at": time.time(),
        }
        return selected.node_id

    def unschedule(self, job_id: str) -> bool:
        job = self._scheduled_jobs.pop(job_id, None)
        if not job:
            return False
        node_id = job.get("node_id")
        spec = job.get("resource_spec")
        node = self._nodes.get(node_id) if isinstance(node_id, str) else None
        if node and isinstance(spec, ResourceSpec):
            node.release(spec)
        return True

    def _healthy_candidates(self, spec: ResourceSpec, constraints: Dict[str, Any]) -> List[NodeInfo]:
        candidates = []
        required_type = constraints.get("node_type")
        required_gpu_type = constraints.get("gpu_type")
        required_tpu_type = constraints.get("tpu_type")
        affinity_labels = constraints.get("affinity_labels", {})
        not_taints = constraints.get("not_taints", [])
        for node in self.get_healthy_nodes():
            if required_type and node.node_type.value != required_type:
                continue
            if required_gpu_type and node.resources.gpu_type != required_gpu_type:
                continue
            if required_tpu_type and node.resources.tpu_type != required_tpu_type:
                continue
            if any(t in node.taints for t in not_taints):
                continue
            matched = True
            for k, v in affinity_labels.items():
                if node.labels.get(k) != v:
                    matched = False
                    break
            if not matched:
                continue
            if node.can_schedule(spec):
                candidates.append(node)
        return candidates

    def _best_candidate(self, candidates: List[NodeInfo], spec: ResourceSpec, constraints: Dict[str, Any]) -> Optional[NodeInfo]:
        strategy = constraints.get("strategy", "binpack")
        if strategy == "spread":
            return min(candidates, key=lambda n: sum(n.available_resources().to_dict().values() if isinstance(n.available_resources().to_dict(), dict) else []))
        return min(candidates, key=lambda n: (
            n.allocated_resources.cpu_cores / max(n.resources.cpu_cores, 1.0),
            n.allocated_resources.memory_gb / max(n.resources.memory_gb, 1.0),
        ))

    def migrate_job(self, job_id: str, target_node_id: str) -> bool:
        job = self._scheduled_jobs.get(job_id)
        if not job:
            return False
        source_id = job.get("node_id")
        spec = job.get("resource_spec")
        if not isinstance(source_id, str) or not isinstance(spec, ResourceSpec):
            return False
        source = self._nodes.get(source_id)
        target = self._nodes.get(target_node_id)
        if not source or not target:
            return False
        if not target.can_schedule(spec):
            return False
        source.release(spec)
        target.allocate(spec)
        job["node_id"] = target_node_id
        return True
