"""Kubernetes integration for job submission and auto-scaling."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class K8sResourceSpec:
    cpu_cores: str = "1"
    memory_gb: str = "4Gi"
    gpu_count: int = 0
    gpu_type: Optional[str] = None
    gpu_vram_gb: Optional[str] = None
    tpu_count: int = 0
    tpu_type: Optional[str] = None
    storage_gb: str = "20Gi"
    custom_resources: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "gpu_count": self.gpu_count,
            "gpu_type": self.gpu_type,
            "gpu_vram_gb": self.gpu_vram_gb,
            "tpu_count": self.tpu_count,
            "tpu_type": self.tpu_type,
            "storage_gb": self.storage_gb,
            "custom_resources": self.custom_resources,
        }


@dataclass
class K8sNodeAffinity:
    required: Dict[str, Any] = field(default_factory=dict)
    preferred: List[Dict[str, Any]] = field(default_factory=list)
    node_selector: Dict[str, str] = field(default_factory=dict)
    tolerations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "required": self.required,
            "preferred": self.preferred,
            "node_selector": self.node_selector,
            "tolerations": self.tolerations,
        }


@dataclass
class K8sJobSpec:
    job_id: str
    name: str
    image: str
    command: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    resources: K8sResourceSpec = field(default_factory=K8sResourceSpec)
    affinity: K8sNodeAffinity = field(default_factory=K8sNodeAffinity)
    replicas: int = 1
    restart_policy: str = "OnFailure"
    backoff_limit: int = 3
    ttl_seconds_after_finished: int = 86400
    priority_class: str = "default"
    service_account: str = "default"
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    volume_mounts: List[Dict[str, Any]] = field(default_factory=list)
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    node_selector_map: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "image": self.image,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "resources": self.resources.to_dict(),
            "affinity": self.affinity.to_dict(),
            "replicas": self.replicas,
            "restart_policy": self.restart_policy,
            "backoff_limit": self.backoff_limit,
            "ttl_seconds_after_finished": self.ttl_seconds_after_finished,
            "priority_class": self.priority_class,
            "service_account": self.service_account,
            "volumes": self.volumes,
            "volume_mounts": self.volume_mounts,
            "labels": self.labels,
            "annotations": self.annotations,
            "node_selector_map": self.node_selector_map,
        }


class K8sOrchestrator:
    def __init__(self, namespace: str = "default", kubeconfig: Optional[str] = None) -> None:
        self._namespace = namespace
        self._kubeconfig = kubeconfig or os.environ.get("KUBECONFIG", "~/.kube/config")
        self._jobs: Dict[str, K8sJobSpec] = {}
        self._running: Dict[str, Dict[str, Any]] = {}
        self._services: Dict[str, Dict[str, Any]] = {}
        self._scaling_policies: Dict[str, Dict[str, Any]] = {}

    def _kubectl(self, args: List[str], stdin: Optional[str] = None) -> Dict[str, Any]:
        cmd = ["kubectl", f"--kubeconfig={self._kubeconfig}", f"--namespace={self._namespace}"] + args
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30, input=stdin)
            return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"returncode": -1, "stdout": "", "stderr": str(e)}

    def _job_name(self, job_id: str) -> str:
        return f"aweai-{job_id}"

    def submit_job(self, spec: K8sJobSpec) -> Dict[str, Any]:
        manifest = self._job_manifest(spec)
        res = self._kubectl(["apply", "-f", "-"], stdin=json.dumps(manifest))
        if res["returncode"] == 0:
            self._jobs[spec.job_id] = spec
            self._running[spec.job_id] = {
                "status": "submitted",
                "submitted_at": time.time(),
                "phase": "Pending",
            }
            return {"job_id": spec.job_id, "status": "submitted", "manifest": manifest}
        return {"job_id": spec.job_id, "status": "error", "error": res["stderr"]}

    def delete_job(self, job_id: str, grace_period: int = 30) -> Dict[str, Any]:
        name = self._job_name(job_id)
        res = self._kubectl(["delete", "job", name, f"--grace-period={grace_period}", "--force", "--ignore-not-found=true"])
        self._jobs.pop(job_id, None)
        self._running.pop(job_id, None)
        return {"job_id": job_id, "status": "deleted" if res["returncode"] == 0 else "error", "error": res.get("stderr", "")}

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        name = self._job_name(job_id)
        res = self._kubectl(["get", "job", name, "-o", "json"])
        if res["returncode"] != 0:
            return {"job_id": job_id, "status": "not_found"}
        try:
            data = json.loads(res["stdout"])
            status = data.get("status", {})
            conditions = status.get("conditions", [])
            phase = "Unknown"
            for cond in conditions:
                if cond.get("type") == "Complete":
                    phase = "Succeeded"
                elif cond.get("type") == "Failed":
                    phase = "Failed"
            phase = phase if phase != "Unknown" else status.get("phase", "Unknown")
            return {
                "job_id": job_id,
                "status": phase,
                "succeeded": status.get("succeeded", 0),
                "failed": status.get("failed", 0),
                "active": status.get("active", 0),
                "start_time": status.get("startTime"),
                "completion_time": status.get("completionTime"),
            }
        except Exception as e:
            return {"job_id": job_id, "status": "error", "error": str(e)}

    def list_pods(self, label_selector: Optional[str] = None) -> Dict[str, Any]:
        args = ["get", "pods", "-o", "json"]
        if label_selector:
            args.extend(["-l", label_selector])
        res = self._kubectl(args)
        if res["returncode"] != 0:
            return {"pods": [], "error": res["stderr"]}
        try:
            data = json.loads(res["stdout"])
            pods = []
            for item in data.get("items", []):
                meta = item.get("metadata", {})
                spec = item.get("spec", {})
                containers = spec.get("containers", [])
                gpu_count = 0
                for c in containers:
                    limits = c.get("resources", {}).get("limits", {})
                    gpu = limits.get("nvidia.com/gpu", "0")
                    try:
                        gpu_count += int(gpu)
                    except Exception:
                        pass
                pods.append({
                    "name": meta.get("name", ""),
                    "namespace": meta.get("namespace", ""),
                    "node": spec.get("nodeName", ""),
                    "phase": item.get("status", {}).get("phase", "Unknown"),
                    "gpu_count": gpu_count,
                    "labels": meta.get("labels", {}),
                })
            return {"pods": pods, "count": len(pods)}
        except Exception as e:
            return {"pods": [], "error": str(e)}

    def get_pod_logs(self, pod_name: str, tail_lines: int = 100, container: Optional[str] = None) -> Dict[str, Any]:
        args = ["logs", pod_name, f"--tail={tail_lines}"]
        if container:
            args.extend(["-c", container])
        res = self._kubectl(args)
        return {"pod": pod_name, "logs": res["stdout"], "error": res.get("stderr", "")}

    def create_service(self, name: str, selector: Dict[str, str], ports: List[Dict[str, Any]], service_type: str = "ClusterIP") -> Dict[str, Any]:
        manifest = {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": name, "namespace": self._namespace},
            "spec": {"selector": selector, "ports": ports, "type": service_type},
        }
        res = self._kubectl(["apply", "-f", "-"], stdin=json.dumps(manifest))
        self._services[name] = {"selector": selector, "ports": ports, "type": service_type}
        return {"service": name, "status": "created" if res["returncode"] == 0 else "error", "error": res.get("stderr", "")}

    def discover_service(self, name: str) -> Dict[str, Any]:
        res = self._kubectl(["get", "service", name, "-o", "json"])
        if res["returncode"] != 0:
            return {"service": name, "status": "not_found"}
        try:
            data = json.loads(res["stdout"])
            spec = data.get("spec", {})
            return {
                "service": name,
                "cluster_ip": spec.get("clusterIP", ""),
                "ports": spec.get("ports", []),
                "type": spec.get("type", "ClusterIP"),
                "selector": spec.get("selector", {}),
            }
        except Exception as e:
            return {"service": name, "status": "error", "error": str(e)}

    def set_autoscaling(self, deployment: str, min_replicas: int, max_replicas: int, metrics: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        metrics = metrics or [
            {"type": "Resource", "resource": {"name": "nvidia.com/gpu", "target": {"type": "Utilization", "averageUtilization": 80}}},
            {"type": "Resource", "resource": {"name": "cpu", "target": {"type": "Utilization", "averageUtilization": 70}}},
        ]
        hpa = {
            "apiVersion": "autoscaling/v2",
            "kind": "HorizontalPodAutoscaler",
            "metadata": {"name": f"aweai-hpa-{deployment}", "namespace": self._namespace},
            "spec": {
                "scaleTargetRef": {"apiVersion": "apps/v1", "kind": "Deployment", "name": deployment},
                "minReplicas": min_replicas,
                "maxReplicas": max_replicas,
                "metrics": metrics,
                "behavior": {
                    "scaleUp": {"stabilizationWindowSeconds": 30, "policies": [{"type": "Percent", "value": 100, "periodSeconds": 15}]},
                    "scaleDown": {"stabilizationWindowSeconds": 300, "policies": [{"type": "Percent", "value": 50, "periodSeconds": 60}]},
                },
            },
        }
        res = self._kubectl(["apply", "-f", "-"], stdin=json.dumps(hpa))
        self._scaling_policies[deployment] = {"min_replicas": min_replicas, "max_replicas": max_replicas, "metrics": metrics}
        return {"deployment": deployment, "status": "configured" if res["returncode"] == 0 else "error", "error": res.get("stderr", "")}

    def scale_deployment(self, deployment: str, replicas: int) -> Dict[str, Any]:
        res = self._kubectl(["scale", "deployment", deployment, f"--replicas={replicas}"])
        return {"deployment": deployment, "replicas": replicas, "status": "scaled" if res["returncode"] == 0 else "error", "error": res.get("stderr", "")}

    def get_node_metrics(self) -> Dict[str, Any]:
        res = self._kubectl(["top", "nodes", "-o", "json"])
        if res["returncode"] != 0:
            return {"nodes": [], "error": res.get("stderr", "")}
        try:
            data = json.loads(res["stdout"])
            metrics = []
            for item in data.get("items", []):
                usage = item.get("usage", {})
                metrics.append({
                    "name": item.get("metadata", {}).get("name", ""),
                    "cpu": usage.get("cpu", "0"),
                    "memory": usage.get("memory", "0"),
                })
            return {"nodes": metrics, "count": len(metrics)}
        except Exception as e:
            return {"nodes": [], "error": str(e)}

    def _job_manifest(self, spec: K8sJobSpec) -> Dict[str, Any]:
        resources = {"requests": {}, "limits": {}}
        if spec.resources.cpu_cores != "0":
            resources["requests"]["cpu"] = spec.resources.cpu_cores
            resources["limits"]["cpu"] = spec.resources.cpu_cores
        if spec.resources.memory_gb != "0":
            resources["requests"]["memory"] = spec.resources.memory_gb
            resources["limits"]["memory"] = spec.resources.memory_gb
        if spec.resources.gpu_count > 0:
            resources["limits"]["nvidia.com/gpu"] = str(spec.resources.gpu_count)
        if spec.resources.tpu_count > 0:
            resources["limits"]["google.com/tpu"] = str(spec.resources.tpu_count)
        for k, v in spec.resources.custom_resources.items():
            resources["requests"][k] = v
            resources["limits"][k] = v
        pod_spec: Dict[str, Any] = {
            "restartPolicy": spec.restart_policy,
            "containers": [{
                "name": "worker",
                "image": spec.image,
                "command": spec.command,
                "args": spec.args,
                "env": [{"name": k, "value": v} for k, v in spec.env.items()],
                "resources": resources,
                "volumeMounts": spec.volume_mounts,
            }],
            "volumes": spec.volumes,
            "nodeSelector": spec.node_selector_map or spec.affinity.node_selector,
        }
        if spec.affinity.tolerations:
            pod_spec["tolerations"] = spec.affinity.tolerations
        affinity = {}
        if spec.affinity.required or spec.affinity.preferred:
            affinity = {
                "nodeAffinity": {},
            }
            if spec.affinity.required:
                affinity["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"] = spec.affinity.required
            if spec.affinity.preferred:
                affinity["nodeAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"] = spec.affinity.preferred
            pod_spec["affinity"] = affinity
        manifest = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {"name": self._job_name(spec.job_id), "namespace": self._namespace, "labels": spec.labels, "annotations": spec.annotations},
            "spec": {
                "ttlSecondsAfterFinished": spec.ttl_seconds_after_finished,
                "backoffLimit": spec.backoff_limit,
                "template": {
                    "metadata": {"labels": spec.labels},
                    "spec": pod_spec,
                },
            },
        }
        return manifest
