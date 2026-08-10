from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class BackupEngine:
    def __init__(self, source_dir: str, backup_dir: str) -> None:
        self.source_dir = Path(source_dir)
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_file = self.backup_dir / "manifest.json"
        self._load_manifest()

    def _load_manifest(self) -> None:
        if self._manifest_file.exists():
            try:
                self._manifest = json.loads(self._manifest_file.read_text(encoding="utf-8"))
            except Exception:
                self._manifest = {"backups": [], "replicas": []}
        else:
            self._manifest = {"backups": [], "replicas": []}

    def _save_manifest(self) -> None:
        self._manifest_file.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _checksum(self, path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def backup(self, name: str, incremental: bool = True) -> Dict[str, Any]:
        ts = datetime.utcnow().isoformat()
        backup_path = self.backup_dir / name
        backup_path.mkdir(parents=True, exist_ok=True)
        manifest = {
            "name": name,
            "timestamp": ts,
            "source": str(self.source_dir),
            "files": [],
            "incremental": incremental,
        }
        if incremental and self._manifest.get("last_full_backup"):
            last_checksums = self._manifest["last_full_backup"].get("checksums", {})
        else:
            last_checksums = {}
        current_checksums = {}
        for f in sorted(self.source_dir.rglob("*")):
            if f.is_file():
                rel = f.relative_to(self.source_dir)
                checksum = self._checksum(f)
                current_checksums[str(rel)] = checksum
                if checksum == last_checksums.get(str(rel)):
                    continue
                dst = backup_path / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                manifest["files"].append({"path": str(rel), "checksum": checksum, "size": f.stat().st_size})
        self._manifest["backups"].append(manifest)
        self._manifest["last_full_backup"] = manifest
        self._save_manifest()
        return {"name": name, "timestamp": ts, "files_copied": len(manifest["files"])}

    def restore(self, name: str, target_dir: str, point_in_time: Optional[str] = None) -> Dict[str, Any]:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        backup_path = self.backup_dir / name
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup {name} not found")
        restored = 0
        for src in sorted(backup_path.rglob("*")):
            if src.is_file():
                rel = src.relative_to(backup_path)
                dst = target / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                restored += 1
        return {"name": name, "target": str(target), "files_restored": restored, "point_in_time": point_in_time}

    def list_backups(self) -> List[Dict[str, Any]]:
        return self._manifest.get("backups", [])

    def delete_backup(self, name: str) -> None:
        backup_path = self.backup_dir / name
        if backup_path.exists():
            shutil.rmtree(backup_path)
        self._manifest["backups"] = [b for b in self._manifest["backups"] if b["name"] != name]
        self._save_manifest()

    def replicate(self, replica_dir: str) -> Dict[str, Any]:
        replica_path = Path(replica_dir)
        replica_path.mkdir(parents=True, exist_ok=True)
        for src in sorted(self.source_dir.rglob("*")):
            if src.is_file():
                rel = src.relative_to(self.source_dir)
                dst = replica_path / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        record = {"path": str(replica_path), "timestamp": datetime.utcnow().isoformat()}
        self._manifest.setdefault("replicas", []).append(record)
        self._save_manifest()
        return record

    def consistency_check(self) -> Dict[str, Any]:
        source_checksums = {}
        for f in sorted(self.source_dir.rglob("*")):
            if f.is_file():
                source_checksums[str(f.relative_to(self.source_dir))] = self._checksum(f)
        issues = []
        for backup in self._manifest.get("backups", []):
            backup_path = self.backup_dir / backup["name"]
            if not backup_path.exists():
                issues.append({"backup": backup["name"], "issue": "missing_backup"})
                continue
            for file_info in backup.get("files", []):
                fpath = backup_path / file_info["path"]
                if fpath.exists():
                    actual = self._checksum(fpath)
                    if actual != file_info["checksum"]:
                        issues.append({"backup": backup["name"], "file": file_info["path"], "issue": "checksum_mismatch"})
        return {"checked": len(source_checksums), "issues": issues, "healthy": len(issues) == 0}

    def close(self) -> None:
        pass
