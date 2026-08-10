"""Deployment engine for multi-target AI infrastructure."""

from __future__ import annotations

import json
import os
import subprocess
import textwrap
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class DeploymentTarget(str, Enum):
    DOCKER = "docker"
    PODMAN = "podman"
    KUBERNETES = "kubernetes"
    SYSTEMD = "systemd"
    DOCKER_COMPOSE = "docker_compose"
    KUSTOMIZE = "kustomize"


@dataclass
class ContainerConfig:
    image: str
    command: List[str] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    ports: List[Dict[str, int]] = field(default_factory=list)
    devices: List[str] = field(default_factory=list)
    gpu_all: bool = False
    privileged: bool = False
    network: str = "bridge"
    restart_policy: str = "unless-stopped"
    ulimits: Dict[str, int] = field(default_factory=dict)
    labels: Dict[str, str] = field(default_factory=dict)
    healthcheck: Optional[Dict[str, Any]] = None
    working_dir: str = "/workspace"
    user: str = "root"
    shm_size: str = "4g"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image": self.image,
            "command": self.command,
            "args": self.args,
            "env": self.env,
            "volumes": self.volumes,
            "ports": self.ports,
            "devices": self.devices,
            "gpu_all": self.gpu_all,
            "privileged": self.privileged,
            "network": self.network,
            "restart_policy": self.restart_policy,
            "ulimits": self.ulimits,
            "labels": self.labels,
            "healthcheck": self.healthcheck,
            "working_dir": self.working_dir,
            "user": self.user,
            "shm_size": self.shm_size,
        }


@dataclass
class DeploymentSpec:
    name: str
    target: DeploymentTarget
    containers: List[ContainerConfig] = field(default_factory=list)
    services: List[Dict[str, Any]] = field(default_factory=list)
    volumes: List[Dict[str, Any]] = field(default_factory=list)
    networks: List[str] = field(default_factory=list)
    namespace: str = "default"
    replicas: int = 1
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "target": self.target.value,
            "containers": [c.to_dict() for c in self.containers],
            "services": self.services,
            "volumes": self.volumes,
            "networks": self.networks,
            "namespace": self.namespace,
            "replicas": self.replicas,
            "labels": self.labels,
            "annotations": self.annotations,
        }


class DeploymentEngine:
    def __init__(self) -> None:
        self._deployed: Dict[str, Dict[str, Any]] = {}

    def deploy(self, spec: DeploymentSpec) -> Dict[str, Any]:
        if spec.target == DeploymentTarget.DOCKER:
            return self._deploy_docker(spec)
        if spec.target == DeploymentTarget.PODMAN:
            return self._deploy_podman(spec)
        if spec.target == DeploymentTarget.KUBERNETES:
            return self._deploy_kubernetes(spec)
        if spec.target == DeploymentTarget.SYSTEMD:
            return self._deploy_systemd(spec)
        if spec.target == DeploymentTarget.DOCKER_COMPOSE:
            return self._deploy_docker_compose(spec)
        if spec.target == DeploymentTarget.KUSTOMIZE:
            return self._deploy_kustomize(spec)
        return {"status": "error", "error": f"unsupported target {spec.target}"}

    def undeploy(self, name: str, target: DeploymentTarget) -> Dict[str, Any]:
        if target == DeploymentTarget.DOCKER:
            return self._remove_container(name)
        if target == DeploymentTarget.PODMAN:
            return self._remove_container(name, runtime="podman")
        if target == DeploymentTarget.KUBERNETES:
            return self._delete_k8s_resources(name)
        if target == DeploymentTarget.SYSTEMD:
            return self._stop_systemd_service(name)
        if target == DeploymentTarget.DOCKER_COMPOSE:
            return self._compose_down(name)
        return {"status": "error", "error": f"unsupported target {target}"}

    def generate_manifest(self, spec: DeploymentSpec) -> str:
        if spec.target == DeploymentTarget.DOCKER:
            return self._dockerfile(spec)
        if spec.target == DeploymentTarget.KUBERNETES:
            return self._k8s_manifest(spec)
        if spec.target == DeploymentTarget.SYSTEMD:
            return self._systemd_unit(spec)
        if spec.target == DeploymentTarget.DOCKER_COMPOSE:
            return self._compose_file(spec)
        return ""

    def _deploy_docker(self, spec: DeploymentSpec) -> Dict[str, Any]:
        container = spec.containers[0]
        run_args = ["docker", "run", "-d"]
        run_args.extend(["--name", spec.name])
        run_args.extend(["--restart", container.restart_policy])
        for p in container.ports:
            for k, v in p.items():
                run_args.extend(["-p", f"{k}:{v}"])
        for vol in container.volumes:
            for k, v in vol.items():
                run_args.extend(["-v", f"{k}:{v}"])
        for dev in container.devices:
            run_args.extend(["--device", dev])
        if container.gpu_all:
            run_args.extend(["--gpus", "all"])
        if container.privileged:
            run_args.append("--privileged")
        if container.network:
            run_args.extend(["--network", container.network])
        if container.shm_size:
            run_args.extend(["--shm-size", container.shm_size])
        for k, v in container.env.items():
            run_args.extend(["-e", f"{k}={v}"])
        for k, v in container.ulimits.items():
            run_args.extend(["--ulimit", f"{k}={v}"])
        run_args.append(container.image)
        run_args.extend(container.command)
        run_args.extend(container.args)
        try:
            result = subprocess.run(run_args, capture_output=True, text=True, timeout=30)
            cid = result.stdout.strip()
            if result.returncode == 0:
                self._deployed[spec.name] = {"target": "docker", "container_id": cid}
                return {"status": "deployed", "container_id": cid}
            return {"status": "error", "error": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_podman(self, spec: DeploymentSpec) -> Dict[str, Any]:
        container = spec.containers[0]
        run_args = ["podman", "run", "-d"]
        run_args.extend(["--name", spec.name])
        run_args.extend(["--restart", container.restart_policy])
        for p in container.ports:
            for k, v in p.items():
                run_args.extend(["-p", f"{k}:{v}"])
        for vol in container.volumes:
            for k, v in vol.items():
                run_args.extend(["-v", f"{k}:{v}"])
        if container.gpu_all:
            run_args.extend(["--gpus", "all"])
        if container.privileged:
            run_args.append("--privileged")
        for k, v in container.env.items():
            run_args.extend(["-e", f"{k}={v}"])
        run_args.append(container.image)
        run_args.extend(container.command)
        run_args.extend(container.args)
        try:
            result = subprocess.run(run_args, capture_output=True, text=True, timeout=30)
            cid = result.stdout.strip()
            if result.returncode == 0:
                self._deployed[spec.name] = {"target": "podman", "container_id": cid}
                return {"status": "deployed", "container_id": cid}
            return {"status": "error", "error": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _remove_container(self, name: str, runtime: str = "docker") -> Dict[str, Any]:
        try:
            result = subprocess.run([runtime, "rm", "-f", name], capture_output=True, text=True, timeout=15)
            self._deployed.pop(name, None)
            return {"status": "removed" if result.returncode == 0 else "error", "error": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_kubernetes(self, spec: DeploymentSpec) -> Dict[str, Any]:
        manifest = self._k8s_manifest(spec)
        tmp = f"/tmp/aweai-k8s-{spec.name}-{int(time.time())}.yaml"
        with open(tmp, "w") as f:
            f.write(manifest)
        try:
            result = subprocess.run(["kubectl", "apply", "-f", tmp], capture_output=True, text=True, timeout=30)
            self._deployed[spec.name] = {"target": "kubernetes", "manifest": tmp}
            return {"status": "deployed" if result.returncode == 0 else "error", "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

    def _delete_k8s_resources(self, name: str) -> Dict[str, Any]:
        tmp = self._deployed.get(name, {}).get("manifest", "")
        if not tmp or not os.path.exists(tmp):
            return {"status": "error", "error": "manifest not found"}
        try:
            result = subprocess.run(["kubectl", "delete", "-f", tmp], capture_output=True, text=True, timeout=30)
            self._deployed.pop(name, None)
            return {"status": "deleted" if result.returncode == 0 else "error", "error": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_systemd(self, spec: DeploymentSpec) -> Dict[str, Any]:
        unit = self._systemd_unit(spec)
        unit_path = f"/etc/systemd/system/{spec.name}.service"
        try:
            with open(unit_path, "w") as f:
                f.write(unit)
            subprocess.run(["systemctl", "daemon-reload"], capture_output=True, text=True, timeout=10)
            subprocess.run(["systemctl", "enable", f"{spec.name}.service"], capture_output=True, text=True, timeout=10)
            subprocess.run(["systemctl", "start", f"{spec.name}.service"], capture_output=True, text=True, timeout=10)
            self._deployed[spec.name] = {"target": "systemd", "unit_path": unit_path}
            return {"status": "deployed", "unit_path": unit_path}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _stop_systemd_service(self, name: str) -> Dict[str, Any]:
        try:
            subprocess.run(["systemctl", "stop", f"{name}.service"], capture_output=True, text=True, timeout=10)
            subprocess.run(["systemctl", "disable", f"{name}.service"], capture_output=True, text=True, timeout=10)
            unit_path = f"/etc/systemd/system/{name}.service"
            if os.path.exists(unit_path):
                os.remove(unit_path)
            subprocess.run(["systemctl", "daemon-reload"], capture_output=True, text=True, timeout=10)
            self._deployed.pop(name, None)
            return {"status": "stopped"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_docker_compose(self, spec: DeploymentSpec) -> Dict[str, Any]:
        compose = self._compose_file(spec)
        tmp = f"/tmp/aweai-compose-{spec.name}-{int(time.time())}.yaml"
        with open(tmp, "w") as f:
            f.write(compose)
        try:
            result = subprocess.run(["docker", "compose", "-f", tmp, "up", "-d"], capture_output=True, text=True, timeout=60)
            self._deployed[spec.name] = {"target": "docker_compose", "file": tmp}
            return {"status": "deployed" if result.returncode == 0 else "error", "stdout": result.stdout, "stderr": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}
        finally:
            try:
                os.remove(tmp)
            except Exception:
                pass

    def _compose_down(self, name: str) -> Dict[str, Any]:
        tmp = self._deployed.get(name, {}).get("file", "")
        if not tmp or not os.path.exists(tmp):
            return {"status": "error", "error": "compose file not found"}
        try:
            result = subprocess.run(["docker", "compose", "-f", tmp, "down"], capture_output=True, text=True, timeout=30)
            self._deployed.pop(name, None)
            return {"status": "down" if result.returncode == 0 else "error", "error": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _deploy_kustomize(self, spec: DeploymentSpec) -> Dict[str, Any]:
        tmp = f"/tmp/aweai-kustomize-{spec.name}-{int(time.time())}"
        os.makedirs(tmp, exist_ok=True)
        kustomization = {"apiVersion": "kustomize.config.k8s.io/v1beta1", "kind": "Kustomization", "namespace": spec.namespace}
        manifest = json.loads(self._k8s_manifest(spec))
        kustomization["resources"] = [f"{spec.name}.yaml"]
        with open(os.path.join(tmp, f"{spec.name}.yaml"), "w") as f:
            json.dump(manifest, f)
        with open(os.path.join(tmp, "kustomization.yaml"), "w") as f:
            json.dump(kustomization, f)
        try:
            result = subprocess.run(["kubectl", "apply", "-k", tmp], capture_output=True, text=True, timeout=30)
            self._deployed[spec.name] = {"target": "kustomize", "dir": tmp}
            return {"status": "deployed" if result.returncode == 0 else "error", "error": result.stderr}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def _dockerfile(self, spec: DeploymentSpec) -> str:
        lines = ["FROM python:3.11-slim"]
        lines.extend(["WORKDIR /workspace"])
        lines.extend(["COPY requirements.txt ."])
        lines.extend(["RUN pip install --no-cache-dir -r requirements.txt"])
        for c in spec.containers:
            for k, v in c.env.items():
                lines.extend([f"ENV {k}={v}"])
        lines.extend(["COPY . ."])
        for c in spec.containers:
            lines.extend(["EXPOSE " + str(p.get("port", p.get("target", 80))) for p in c.ports])
        for c in spec.containers:
            lines.extend(["CMD " + " ".join(c.command or ["sleep", "infinity"])])
        return "\n".join(lines)

    def _k8s_manifest(self, spec: DeploymentSpec) -> str:
        container = spec.containers[0]
        resources = {"requests": {}, "limits": {}}
        if container.gpu_all:
            resources["limits"]["nvidia.com/gpu"] = "1"
        pod_spec: Dict[str, Any] = {
            "restartPolicy": "Always",
            "containers": [{
                "name": spec.name,
                "image": container.image,
                "command": container.command,
                "args": container.args,
                "env": [{"name": k, "value": v} for k, v in container.env.items()],
                "resources": resources,
                "volumeMounts": container.volumes,
            }],
        }
        manifest = {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": spec.name, "namespace": spec.namespace, "labels": spec.labels},
            "spec": {"replicas": spec.replicas, "selector": {"matchLabels": {"app": spec.name}}, "template": {"metadata": {"labels": {"app": spec.name}}, "spec": pod_spec}},
        }
        return json.dumps(manifest, indent=2)

    def _systemd_unit(self, spec: DeploymentSpec) -> str:
        container = spec.containers[0]
        exec_start = ["docker", "run", "--rm", "--name", spec.name]
        for k, v in container.env.items():
            exec_start.extend(["-e", f"{k}={v}"])
        for vol in container.volumes:
            for k, v in vol.items():
                exec_start.extend(["-v", f"{k}:{v}"])
        if container.gpu_all:
            exec_start.extend(["--gpus", "all"])
        exec_start.append(container.image)
        exec_start.extend(container.command)
        exec_start.extend(container.args)
        return textwrap.dedent(f"""\
            [Unit]
            Description={spec.name}
            After=docker.service
            Requires=docker.service

            [Service]
            Type=simple
            Restart=always
            RestartSec=5
            ExecStart={' '.join(exec_start)}
            Environment=DOCKER_BUILDKIT=1

            [Install]
            WantedBy=multi-user.target
        """)

    def _compose_file(self, spec: DeploymentSpec) -> str:
        lines = [f"version: '3.8'\n\nservices:\n  {spec.name}:"]
        container = spec.containers[0]
        lines.extend([f"    image: {container.image}"])
        if container.command:
            lines.extend([f"    command: {' '.join(container.command)}"])
        if container.env:
            lines.extend(["    environment:"])
            for k, v in container.env.items():
                lines.extend([f"      {k}: {v}"])
        if container.ports:
            lines.extend(["    ports:"])
            for p in container.ports:
                for k, v in p.items():
                    lines.extend([f"      - \"{k}:{v}\""])
        if container.volumes:
            lines.extend(["    volumes:"])
            for vol in container.volumes:
                for k, v in vol.items():
                    lines.extend([f"      - {k}:{v}"])
        if container.gpu_all:
            lines.extend(["    deploy:"])
            lines.extend(["      resources:"])
            lines.extend(["        reservations:"])
            lines.extend(["          devices:"])
            lines.extend(["            - driver: nvidia"])
            lines.extend(["              count: all"])
            lines.extend(["              capabilities: [gpu]"])
        lines.extend([f"    restart: {container.restart_policy}"])
        return "\n".join(lines)
