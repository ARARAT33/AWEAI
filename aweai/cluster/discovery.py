"""Service and node discovery for optimal placement."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class GPUInfo:
    index: int
    name: str
    uuid: str
    vram_gb: float
    driver_version: str
    compute_capability: str
    nvlink_peers: List[int] = field(default_factory=list)
    pcie_bus_id: str = ""
    power_limit_w: float = 0.0
    temperature_c: float = 0.0
    utilization_percent: float = 0.0
    memory_util_percent: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "name": self.name,
            "uuid": self.uuid,
            "vram_gb": self.vram_gb,
            "driver_version": self.driver_version,
            "compute_capability": self.compute_capability,
            "nvlink_peers": self.nvlink_peers,
            "pcie_bus_id": self.pcie_bus_id,
            "power_limit_w": self.power_limit_w,
            "temperature_c": self.temperature_c,
            "utilization_percent": self.utilization_percent,
            "memory_util_percent": self.memory_util_percent,
        }


@dataclass
class TPUInfo:
    device_id: str
    type: str
    cores: int
    hbm_gb: float
    version: str
    location: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "type": self.type,
            "cores": self.cores,
            "hbm_gb": self.hbm_gb,
            "version": self.version,
            "location": self.location,
        }


@dataclass
class NPUInfo:
    device_id: str
    name: str
    memory_gb: float
    cores: int
    driver_version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "name": self.name,
            "memory_gb": self.memory_gb,
            "cores": self.cores,
            "driver_version": self.driver_version,
        }


@dataclass
class NetworkLink:
    src_node: str
    dst_node: str
    src_iface: str
    dst_iface: str
    bandwidth_gbps: float
    latency_us: float
    link_type: str = "unknown"
    switch_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "src_node": self.src_node,
            "dst_node": self.dst_node,
            "src_iface": self.src_iface,
            "dst_iface": self.dst_iface,
            "bandwidth_gbps": self.bandwidth_gbps,
            "latency_us": self.latency_us,
            "link_type": self.link_type,
            "switch_name": self.switch_name,
        }


@dataclass
class TopologyInfo:
    node_count: int = 0
    gpu_count: int = 0
    tpu_count: int = 0
    npu_count: int = 0
    ipu_count: int = 0
    gpus: List[GPUInfo] = field(default_factory=list)
    tpus: List[TPUInfo] = field(default_factory=list)
    npus: List[NPUInfo] = field(default_factory=list)
    nvlink_topology: Dict[int, List[int]] = field(default_factory=dict)
    network_links: List[NetworkLink] = field(default_factory=list)
    network_type: str = "unknown"
    fabric_manager: str = ""
    discovered_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "gpu_count": self.gpu_count,
            "tpu_count": self.tpu_count,
            "npu_count": self.npu_count,
            "ipu_count": self.ipu_count,
            "gpus": [g.to_dict() for g in self.gpus],
            "tpus": [t.to_dict() for t in self.tpus],
            "npus": [n.to_dict() for n in self.npus],
            "nvlink_topology": self.nvlink_topology,
            "network_links": [l.to_dict() for l in self.network_links],
            "network_type": self.network_type,
            "fabric_manager": self.fabric_manager,
            "discovered_at": self.discovered_at,
        }


class NodeDiscovery:
    def __init__(self) -> None:
        self._topology: Optional[TopologyInfo] = None
        self._callbacks: List[Callable[[TopologyInfo], None]] = []

    def discover(self) -> TopologyInfo:
        self._topology = TopologyInfo()
        self._discover_gpus()
        self._discover_tpus()
        self._discover_npus()
        self._discover_network()
        self._discover_nvlink()
        self._topology.node_count = self._detect_node_count()
        self._topology.gpu_count = len(self._topology.gpus)
        self._topology.tpu_count = len(self._topology.tpus)
        self._topology.npu_count = len(self._topology.npus)
        for cb in self._callbacks:
            cb(self._topology)
        return self._topology

    def get_topology(self) -> TopologyInfo:
        if self._topology is None:
            return self.discover()
        return self._topology

    def register_callback(self, callback: Callable[[TopologyInfo], None]) -> None:
        self._callbacks.append(callback)

    def _run(self, cmd: List[str]) -> str:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            return result.stdout.strip()
        except Exception:
            return ""

    def _discover_gpus(self) -> None:
        if not self._topology:
            return
        nvidia_smi = self._run(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,driver_version,compute_cap", "--format=csv,noheader,nounits"])
        if not nvidia_smi:
            return
        for line in nvidia_smi.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 6:
                continue
            try:
                idx = int(parts[0])
                mem = float(parts[3]) / 1024.0
                gpu = GPUInfo(
                    index=idx,
                    name=parts[1],
                    uuid=parts[2],
                    vram_gb=round(mem, 2),
                    driver_version=parts[4],
                    compute_capability=parts[5],
                )
                self._topology.gpus.append(gpu)
            except Exception:
                continue

    def _discover_tpus(self) -> None:
        if not self._topology:
            return
        result = self._run(["cat", "/proc/device-tree/opal/compatible"])
        if "google,tpu" in result:
            self._topology.tpus.append(TPUInfo(device_id="tpu0", type="v4", cores=4, hbm_gb=32.0, version="v4"))
        else:
            tpuid = self._run(["cat", "/sys/class/tpu/tpu0/device_type"])
            if tpuid:
                parts = tpuid.split(":")
                tpu_type = parts[0] if parts else tpuid
                self._topology.tpus.append(TPUInfo(device_id="tpu0", type=tpu_type, cores=4, hbm_gb=16.0, version=tpu_type))

    def _discover_npus(self) -> None:
        if not self._topology:
            return
        npu_devices = self._run(["ls", "/dev/", "davinci*"])
        if not npu_devices:
            return
        for dev in npu_devices.splitlines():
            dev_id = dev.strip()
            if not dev_id:
                continue
            self._topology.npus.append(NPUInfo(device_id=dev_id, name="Ascend", memory_gb=32.0, cores=1))

    def _discover_network(self) -> None:
        if not self._topology:
            return
        ib_info = self._run(["ibstat", "-p"])
        if ib_info:
            self._topology.network_type = "InfiniBand"
        else:
            roce = self._run(["cat", "/sys/class/net/ib0/speed"])
            if roce:
                self._topology.network_type = "RoCE"
            else:
                eth_info = self._run(["ethtool", "eth0"])
                if eth_info and "100000mbps" in eth_info.lower():
                    self._topology.network_type = "100GbE"
                else:
                    self._topology.network_type = "Ethernet"

    def _discover_nvlink(self) -> None:
        if not self._topology:
            return
        nvidia_smi_topology = self._run(["nvidia-smi", "topo", "-m", "-c", "-i", "0"])
        if not nvidia_smi_topology:
            return
        lines = nvidia_smi_topology.splitlines()
        current_idx = 0
        for line in lines[3:]:
            parts = line.strip().split()
            if not parts or not parts[0].isdigit():
                continue
            current_idx = int(parts[0])
            peers = []
            for i, p in enumerate(parts[1:], 0):
                if p in ("NV", "NV1", "NV2", "NV3", "NV4", "NV5", "NV6", "NV7", "NV8", "NV9", "NV10", "NV11", "NV12"):
                    peers.append(i)
            self._topology.nvlink_topology[current_idx] = peers
        for gpu in self._topology.gpus:
            if gpu.index in self._topology.nvlink_topology:
                gpu.nvlink_peers = self._topology.nvlink_topology[gpu.index]

    def _detect_node_count(self) -> int:
        hostname = os.environ.get("HOSTNAME", "")
        return 1 if hostname else 1

    def compute_optimal_placement(self, gpu_count: int, prefer_nvlink: bool = True, prefer_infiniband: bool = True) -> Dict[str, Any]:
        topology = self.get_topology()
        clusters: Dict[str, List[int]] = {}
        visited = set()
        for gpu in topology.gpus:
            if gpu.index in visited:
                continue
            cluster = [gpu.index]
            visited.add(gpu.index)
            if prefer_nvlink and gpu.nvlink_peers:
                for peer in gpu.nvlink_peers:
                    if peer not in visited and peer < len(topology.gpus):
                        cluster.append(peer)
                        visited.add(peer)
            key = "nvlink" if prefer_nvlink and gpu.nvlink_peers else "none"
            clusters.setdefault(key, []).extend(cluster)
        if prefer_nvlink and clusters.get("nvlink"):
            placements = clusters["nvlink"][:gpu_count]
        elif clusters.get("none"):
            placements = clusters["none"][:gpu_count]
        else:
            placements = list(range(min(gpu_count, len(topology.gpus))))
        if len(placements) < gpu_count:
            placements.extend(list(range(len(topology.gpus))))
        placements = placements[:gpu_count]
        links = []
        for i in placements:
            gpu = topology.gpus[i]
            for peer_idx in gpu.nvlink_peers:
                if peer_idx in placements:
                    links.append({"src": i, "dst": peer_idx, "bandwidth_gbps": 900.0})
        return {
            "gpu_indices": placements,
            "cluster_type": "nvlink" if prefer_nvlink else "pcie",
            "network_type": topology.network_type,
            "link_count": len(links),
            "links": links,
        }

    def detect_roce_config(self) -> Dict[str, Any]:
        ib_info = self._run(["ibv_devinfo", "-l"])
        roce_config: Dict[str, Any] = {"active": False, "interfaces": [], "mtu": 0, "speed_gbps": 0}
        if ib_info:
            roce_config["active"] = True
            roce_config["interfaces"] = [i.strip() for i in ib_info.splitlines() if i.strip()]
        roce_config["mtu"] = int(self._run(["cat", "/sys/class/net/ib0/mtu"]) or "0")
        roce_config["speed_gbps"] = int(self._run(["cat", "/sys/class/net/ib0/speed"]) or "0")
        return roce_config

    def discover_services(self, service_types: Optional[List[str]] = None) -> Dict[str, Any]:
        service_types = service_types or ["etcd", "redis", "nats", "minio", "mlflow"]
        discovered: Dict[str, Any] = {}
        for svc in service_types:
            try:
                result = subprocess.run(["getent", "hosts", svc], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    discovered[svc] = {"resolved": True, "addresses": result.stdout.strip().splitlines()}
                else:
                    discovered[svc] = {"resolved": False}
            except Exception:
                discovered[svc] = {"resolved": False}
        return discovered

    def get_nvlink_bandwidth(self, src_idx: int, dst_idx: int) -> float:
        topology = self.get_topology()
        if src_idx in topology.nvlink_topology and dst_idx in topology.nvlink_topology.get(src_idx, []):
            return 900.0
        return 50.0
