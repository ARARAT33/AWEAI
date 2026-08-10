"""Slurm HPC integration for job arrays and gang scheduling."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SlurmResources:
    nodes: int = 1
    ntasks_per_node: int = 1
    cpus_per_task: int = 1
    gpus_per_node: int = 0
    gpu_type: Optional[str] = None
    time_limit: str = "01:00:00"
    memory_per_cpu_gb: Optional[float] = None
    partition: Optional[str] = None
    qos: Optional[str] = None
    account: Optional[str] = None
    constraint: Optional[str] = None
    gres: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "ntasks_per_node": self.ntasks_per_node,
            "cpus_per_task": self.cpus_per_task,
            "gpus_per_node": self.gpus_per_node,
            "gpu_type": self.gpu_type,
            "time_limit": self.time_limit,
            "memory_per_cpu_gb": self.memory_per_cpu_gb,
            "partition": self.partition,
            "qos": self.qos,
            "account": self.account,
            "constraint": self.constraint,
            "gres": self.gres,
        }

    def to_sbatch_args(self) -> List[str]:
        args = ["sbatch"]
        args.extend(["-N", str(self.nodes)])
        args.extend(["-n", str(self.nodes * self.ntasks_per_node)])
        args.extend(["--ntasks-per-node", str(self.ntasks_per_node)])
        args.extend(["--cpus-per-task", str(self.cpus_per_task)])
        args.extend(["--time", self.time_limit])
        if self.gpus_per_node > 0:
            args.extend(["--gpus-per-node", str(self.gpus_per_node)])
            if self.gpu_type:
                args.extend(["--gpu-type", self.gpu_type])
        if self.memory_per_cpu_gb:
            args.extend(["--mem-per-cpu", f"{int(self.memory_per_cpu_gb * 1024)}M"])
        if self.partition:
            args.extend(["-p", self.partition])
        if self.qos:
            args.extend(["--qos", self.qos])
        if self.account:
            args.extend(["-A", self.account])
        if self.constraint:
            args.extend(["--constraint", self.constraint])
        if self.gres:
            args.extend(["--gres", self.gres])
        return args


@dataclass
class SlurmJob:
    job_id: str
    script: str
    resources: SlurmResources = field(default_factory=SlurmResources)
    dependencies: List[str] = field(default_factory=list)
    array_spec: Optional[str] = None
    array_length: Optional[int] = None
    array_max_parallel: Optional[int] = None
    output_file: Optional[str] = None
    error_file: Optional[str] = None
    export_env: Dict[str, str] = field(default_factory=dict)
    extra_args: List[str] = field(default_factory=list)
    cluster_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "script": self.script,
            "resources": self.resources.to_dict(),
            "dependencies": self.dependencies,
            "array_spec": self.array_spec,
            "array_length": self.array_length,
            "array_max_parallel": self.array_max_parallel,
            "output_file": self.output_file,
            "error_file": self.error_file,
            "export_env": self.export_env,
            "extra_args": self.extra_args,
            "cluster_name": self.cluster_name,
        }

    def build_script(self) -> str:
        lines = ["#!/bin/bash"]
        lines.extend([f"#SBATCH -J {self.job_id}"])
        lines.extend([f"#SBATCH -N {self.resources.nodes}"])
        lines.extend([f"#SBATCH -n {self.resources.nodes * self.resources.ntasks_per_node}"])
        lines.extend([f"#SBATCH --ntasks-per-node={self.resources.ntasks_per_node}"])
        lines.extend([f"#SBATCH --cpus-per-task={self.resources.cpus_per_task}"])
        lines.extend([f"#SBATCH --time={self.resources.time_limit}"])
        if self.resources.gpus_per_node > 0:
            lines.extend([f"#SBATCH --gpus-per-node={self.resources.gpus_per_node}"])
            if self.resources.gpu_type:
                lines.extend([f"#SBATCH --gpu-type={self.resources.gpu_type}"])
        if self.resources.memory_per_cpu_gb:
            lines.extend([f"#SBATCH --mem-per-cpu={int(self.resources.memory_per_cpu_gb * 1024)}M"])
        if self.resources.partition:
            lines.extend([f"#SBATCH -p {self.resources.partition}"])
        if self.resources.qos:
            lines.extend([f"#SBATCH --qos={self.resources.qos}"])
        if self.resources.account:
            lines.extend([f"#SBATCH -A {self.resources.account}"])
        if self.resources.constraint:
            lines.extend([f"#SBATCH --constraint={self.resources.constraint}"])
        if self.resources.gres:
            lines.extend([f"#SBATCH --gres={self.resources.gres}"])
        if self.output_file:
            lines.extend([f"#SBATCH -o {self.output_file}"])
        if self.error_file:
            lines.extend([f"#SBATCH -e {self.error_file}"])
        if self.dependencies:
            dep_str = ":".join(self.dependencies)
            lines.extend([f"#SBATCH --dependency=afterok:{dep_str}"])
        if self.array_spec:
            arr = self.array_spec
            if self.array_max_parallel:
                arr += f"%{self.array_max_parallel}"
            lines.extend([f"#SBATCH --array={arr}"])
        for k, v in self.export_env.items():
            lines.extend([f"export {k}={v}"])
        lines.extend(["set -e"])
        lines.extend(["set -x"])
        lines.extend([""])
        lines.extend([self.script])
        return "\n".join(lines)


class SlurmManager:
    def __init__(self, cluster_name: Optional[str] = None) -> None:
        self._cluster_name = cluster_name
        self._jobs: Dict[str, SlurmJob] = {}
        self._job_status: Dict[str, Dict[str, Any]] = {}
        self._gang_groups: Dict[str, List[str]] = {}

    def submit(self, job: SlurmJob) -> Dict[str, Any]:
        script = job.build_script()
        args = job.resources.to_sbatch_args()
        if job.array_spec:
            arr = job.array_spec
            if job.array_max_parallel:
                arr += f"%{job.array_max_parallel}"
            args.extend(["--array", arr])
        if job.dependencies:
            dep_str = ":".join(job.dependencies)
            args.extend(["--dependency", f"afterok:{dep_str}"])
        if job.output_file:
            args.extend(["-o", job.output_file])
        if job.error_file:
            args.extend(["-e", job.error_file])
        for k, v in job.export_env.items():
            args.extend(["--export", f"{k}={v}"])
        args.extend(["--parsable"])
        tmp_path = f"/tmp/aweai-slurm-{job.job_id}-{int(time.time())}.sh"
        with open(tmp_path, "w") as f:
            f.write(script)
        args.append(tmp_path)
        try:
            result = subprocess.run(args, capture_output=True, text=True, timeout=30)
            raw_id = result.stdout.strip()
            slurm_id = raw_id.split(";")[0].strip() if raw_id else ""
            self._jobs[job.job_id] = job
            status = {
                "job_id": job.job_id,
                "slurm_id": slurm_id,
                "status": "submitted" if slurm_id else "error",
                "submitted_at": time.time(),
                "state": "PENDING",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }
            self._job_status[job.job_id] = status
            return status
        except Exception as e:
            status = {"job_id": job.job_id, "status": "error", "error": str(e), "state": "FAILED"}
            self._job_status[job.job_id] = status
            return status

    def cancel(self, job_id: str) -> Dict[str, Any]:
        job = self._jobs.get(job_id)
        if not job:
            return {"job_id": job_id, "status": "error", "error": "job not found"}
        status = self._job_status.get(job_id, {})
        slurm_id = status.get("slurm_id", "")
        if not slurm_id:
            return {"job_id": job_id, "status": "error", "error": "no slurm id"}
        try:
            result = subprocess.run(["scancel", slurm_id], capture_output=True, text=True, timeout=15)
            return {"job_id": job_id, "status": "cancelled" if result.returncode == 0 else "error", "error": result.stderr}
        except Exception as e:
            return {"job_id": job_id, "status": "error", "error": str(e)}

    def job_status(self, job_id: str) -> Dict[str, Any]:
        stored = self._job_status.get(job_id)
        if not stored:
            return {"job_id": job_id, "status": "unknown"}
        slurm_id = stored.get("slurm_id", "")
        if not slurm_id:
            return stored
        try:
            result = subprocess.run(["squeue", "-j", slurm_id, "--json"], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                jobs = data.get("jobs", [])
                if jobs:
                    state = jobs[0].get("job_state", "UNKNOWN")
                    stored["state"] = state
                    stored["node"] = jobs[0].get("node_list", "")
                    stored["start_time"] = jobs[0].get("start_time", "")
        except Exception:
            pass
        return stored

    def job_info(self, job_id: str) -> Dict[str, Any]:
        stored = self._job_status.get(job_id, {})
        slurm_id = stored.get("slurm_id", "")
        if not slurm_id:
            return {"job_id": job_id, "status": "unknown"}
        try:
            result = subprocess.run(["sacct", "-j", slurm_id, "--json"], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                jobs = data.get("jobs", [])
                if jobs:
                    j = jobs[0]
                    return {
                        "job_id": job_id,
                        "slurm_id": slurm_id,
                        "state": j.get("state", "UNKNOWN"),
                        "exit_code": j.get("exit_code", ""),
                        "start": j.get("start", ""),
                        "end": j.get("end", ""),
                        "elapsed": j.get("elapsed", ""),
                        "allocation_nodes": j.get("allocation_nodes", 0),
                        "partition": j.get("partition", ""),
                        "qos": j.get("qos", ""),
                        "account": j.get("account", ""),
                    }
        except Exception:
            pass
        return stored

    def list_partitions(self) -> Dict[str, Any]:
        try:
            result = subprocess.run(["sinfo", "--json"], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                partitions = []
                for p in data.get("partitions", []):
                    partitions.append({
                        "name": p.get("name", ""),
                        "state": p.get("state", ""),
                        "nodes": p.get("nodes", ""),
                        "default": p.get("default", False),
                        "priority": p.get("priority", 0),
                    })
                return {"partitions": partitions, "count": len(partitions)}
        except Exception:
            pass
        return {"partitions": [], "count": 0}

    def gang_schedule(self, job_ids: List[str]) -> Dict[str, Any]:
        if not job_ids:
            return {"status": "error", "error": "no jobs provided"}
        first = self._jobs.get(job_ids[0])
        if not first:
            return {"status": "error", "error": "first job not found"}
        gang_id = f"gang-{int(time.time())}"
        for jid in job_ids:
            job = self._jobs.get(jid)
            if not job:
                continue
            job.dependencies = job_ids[: job_ids.index(jid)]
            job.export_env["AWEAI_GANG_ID"] = gang_id
        results = {}
        for jid in job_ids:
            results[jid] = self.submit(self._jobs[jid])
        self._gang_groups[gang_id] = job_ids
        return {"gang_id": gang_id, "jobs": results}

    def array_submit(self, base_job: SlurmJob, array_length: int, max_parallel: Optional[int] = None) -> Dict[str, Any]:
        base_job.array_spec = f"1-{array_length}"
        base_job.array_length = array_length
        base_job.array_max_parallel = max_parallel
        return self.submit(base_job)

    def get_queue_status(self) -> Dict[str, Any]:
        try:
            result = subprocess.run(["squeue", "--json"], capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                data = json.loads(result.stdout)
                jobs = data.get("jobs", [])
                states: Dict[str, int] = {}
                for j in jobs:
                    state = j.get("job_state", "UNKNOWN")
                    states[state] = states.get(state, 0) + 1
                return {"queue_jobs": len(jobs), "states": states}
        except Exception:
            pass
        return {"queue_jobs": 0, "states": {}}
