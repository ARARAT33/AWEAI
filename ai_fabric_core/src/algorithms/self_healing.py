"""
Self-Healing Model System (SHMS)
Ինքնավերականգնվող մոդելային համակարգ

Automatic model health monitoring and recovery system.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class HealthStatus(Enum):
    """Model health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


@dataclass
class HealthMetrics:
    """Model health metrics"""
    latency_p50_ms: float = 0.0
    latency_p99_ms: float = 0.0
    error_rate: float = 0.0
    throughput_qps: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_utilization: float = 0.0


class SelfHealingModelSystem:
    """
    Self-Healing Model System
    
    Features:
    - Continuous health monitoring
    - Automatic anomaly detection
    - Recovery strategies
    - Rollback capabilities
    """
    
    def __init__(
        self,
        health_check_interval: int = 60,
        recovery_strategy: str = 'rollback',
        max_error_rate: float = 0.05,
        max_latency_ms: float = 1000,
    ):
        self.health_check_interval = health_check_interval
        self.recovery_strategy = recovery_strategy
        self.max_error_rate = max_error_rate
        self.max_latency_ms = max_latency_ms
        
        self.logger = logging.getLogger(__name__)
        
        self.model_versions: List[Dict[str, Any]] = []
        self.current_version: Optional[str] = None
        self.health_history: List[Dict[str, Any]] = []
        self.incidents: List[Dict[str, Any]] = []
        
        # Thresholds
        self.thresholds = {
            'error_rate': self.max_error_rate,
            'latency_p99': self.max_latency_ms,
            'throughput_min': 100,
            'memory_max_mb': 8000,
        }
    
    def register_model_version(self, version: str, model_path: str) -> None:
        """Register a new model version"""
        self.model_versions.append({
            'version': version,
            'path': model_path,
            'registered_at': datetime.now(),
            'status': 'inactive',
        })
        
        if self.current_version is None:
            self.current_version = version
            self.model_versions[-1]['status'] = 'active'
        
        self.logger.info(f"Registered model version: {version}")
    
    def check_health(self, metrics: HealthMetrics) -> HealthStatus:
        """Check model health based on metrics"""
        issues = []
        
        if metrics.error_rate > self.thresholds['error_rate']:
            issues.append(f"High error rate: {metrics.error_rate:.4f}")
        
        if metrics.latency_p99_ms > self.thresholds['latency_p99']:
            issues.append(f"High latency: {metrics.latency_p99_ms:.2f}ms")
        
        if metrics.throughput_qps < self.thresholds['throughput_min']:
            issues.append(f"Low throughput: {metrics.throughput_qps:.2f} qps")
        
        if metrics.memory_usage_mb > self.thresholds['memory_max_mb']:
            issues.append(f"High memory usage: {metrics.memory_usage_mb:.2f}MB")
        
        # Determine health status
        if len(issues) == 0:
            status = HealthStatus.HEALTHY
        elif len(issues) == 1:
            status = HealthStatus.DEGRADED
        elif len(issues) == 2:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.CRITICAL
        
        # Record health check
        self.health_history.append({
            'timestamp': datetime.now(),
            'status': status.value,
            'issues': issues,
            'metrics': {
                'latency_p50_ms': metrics.latency_p50_ms,
                'latency_p99_ms': metrics.latency_p99_ms,
                'error_rate': metrics.error_rate,
                'throughput_qps': metrics.throughput_qps,
            },
        })
        
        if status != HealthStatus.HEALTHY:
            self.logger.warning(f"Health check failed: {', '.join(issues)}")
        
        return status
    
    def trigger_recovery(self, reason: str) -> bool:
        """Trigger recovery process"""
        self.logger.warning(f"Triggering recovery: {reason}")
        
        incident = {
            'timestamp': datetime.now(),
            'reason': reason,
            'recovery_strategy': self.recovery_strategy,
            'success': False,
        }
        
        try:
            if self.recovery_strategy == 'rollback':
                success = self._rollback()
            elif self.recovery_strategy == 'restart':
                success = self._restart()
            elif self.recovery_strategy == 'scale':
                success = self._scale()
            else:
                success = False
            
            incident['success'] = success
            
            if success:
                self.logger.info("Recovery successful")
            else:
                self.logger.error("Recovery failed")
            
        except Exception as e:
            self.logger.error(f"Recovery error: {e}")
            success = False
        
        self.incidents.append(incident)
        return success
    
    def _rollback(self) -> bool:
        """Rollback to previous model version"""
        if len(self.model_versions) < 2:
            self.logger.error("No previous version to rollback to")
            return False
        
        # Find previous version
        for i, version in enumerate(self.model_versions):
            if version['version'] == self.current_version and i > 0:
                # Deactivate current
                self.model_versions[i]['status'] = 'inactive'
                
                # Activate previous
                prev_version = self.model_versions[i - 1]
                prev_version['status'] = 'active'
                self.current_version = prev_version['version']
                
                self.logger.info(f"Rolled back to version: {self.current_version}")
                return True
        
        return False
    
    def _restart(self) -> bool:
        """Restart model service"""
        self.logger.info("Restarting model service")
        # Simulate restart
        return True
    
    def _scale(self) -> bool:
        """Scale model resources"""
        self.logger.info("Scaling model resources")
        # Simulate scaling
        return True
    
    def monitor(self, endpoint: str) -> None:
        """Start continuous monitoring"""
        self.logger.info(f"Starting monitoring for endpoint: {endpoint}")
        
        # In real implementation, this would run continuously
        # and automatically trigger recovery when needed
    
    def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        recent_checks = self.health_history[-100:] if self.health_history else []
        
        healthy_checks = sum(1 for c in recent_checks 
                            if c['status'] == HealthStatus.HEALTHY.value)
        
        total_checks = len(recent_checks) if recent_checks else 1
        
        return {
            'current_version': self.current_version,
            'current_status': recent_checks[-1]['status'] if recent_checks else 'unknown',
            'health_percentage': (healthy_checks / total_checks * 100) if total_checks > 0 else 0,
            'total_incidents': len(self.incidents),
            'successful_recoveries': sum(1 for i in self.incidents if i['success']),
            'model_versions': len(self.model_versions),
        }
