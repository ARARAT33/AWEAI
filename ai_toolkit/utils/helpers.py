#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Helper Utilities - Օժանդակ գործիքներ
Logging, configuration, task scheduling and progress tracking
"""

import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, Future


class Logger:
    """Հզոր լոգավորման համակարգ"""
    
    def __init__(self, name: str = "ai_toolkit", 
                 log_file: Optional[str] = None,
                 level: int = logging.INFO):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.handlers = []
        
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        self.logger.addHandler(console_handler)
        self.handlers.append(console_handler)
        
        if log_file:
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_format = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            file_handler.setFormatter(file_format)
            self.logger.addHandler(file_handler)
            self.handlers.append(file_handler)
        
        self.log_history = []
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)
        self._save_to_history('DEBUG', message, kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
        self._save_to_history('INFO', message, kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
        self._save_to_history('WARNING', message, kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
        self._save_to_history('ERROR', message, kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra=kwargs)
        self._save_to_history('CRITICAL', message, kwargs)
    
    def _save_to_history(self, level: str, message: str, kwargs: Dict):
        self.log_history.append({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message,
            'extra': kwargs
        })
    
    def get_history(self, level: Optional[str] = None, 
                   limit: int = 100) -> List[Dict]:
        history = self.log_history
        if level:
            history = [h for h in history if h['level'] == level]
        return history[-limit:]
    
    def export_logs(self, filename: str = "logs_export.json") -> str:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({
                'export_timestamp': datetime.now().isoformat(),
                'logger_name': self.name,
                'total_logs': len(self.log_history),
                'logs': self.log_history
            }, f, indent=2, ensure_ascii=False)
        return filename


class ConfigManager:
    """Կոնֆիգուրացիայի կառավարում"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config = {}
        self.config_file = Path(config_file) if config_file else None
        self.config_history = []
        
        if self.config_file and self.config_file.exists():
            self.load_config()
    
    def set(self, key: str, value: Any, section: Optional[str] = None):
        if section:
            if section not in self.config:
                self.config[section] = {}
            self.config[section][key] = value
        else:
            self.config[key] = value
        
        self._record_change('set', key, value, section)
    
    def get(self, key: str, section: Optional[str] = None, 
           default: Any = None) -> Any:
        try:
            if section:
                return self.config.get(section, {}).get(key, default)
            return self.config.get(key, default)
        except:
            return default
    
    def delete(self, key: str, section: Optional[str] = None):
        if section:
            if section in self.config and key in self.config[section]:
                del self.config[section][key]
        else:
            if key in self.config:
                del self.config[key]
        
        self._record_change('delete', key, None, section)
    
    def load_config(self, filename: Optional[str] = None) -> bool:
        file_path = Path(filename) if filename else self.config_file
        if not file_path or not file_path.exists():
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    self.config = json.load(f)
                elif file_path.suffix in ['.yaml', '.yml']:
                    import yaml
                    self.config = yaml.safe_load(f)
            
            self._record_change('load', str(file_path), None, None)
            return True
        except Exception as e:
            return False
    
    def save_config(self, filename: Optional[str] = None) -> bool:
        file_path = Path(filename) if filename else self.config_file
        if not file_path:
            return False
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.suffix == '.json':
                    json.dump(self.config, f, indent=2, ensure_ascii=False)
                elif file_path.suffix in ['.yaml', '.yml']:
                    import yaml
                    yaml.dump(self.config, f, default_flow_style=False)
            
            self._record_change('save', str(file_path), None, None)
            return True
        except Exception as e:
            return False
    
    def merge_config(self, override_config: Dict):
        self._deep_merge(self.config, override_config)
        self._record_change('merge', 'config', override_config, None)
    
    def _deep_merge(self, base: Dict, override: Dict):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _record_change(self, action: str, key: str, value: Any, section: Optional[str]):
        self.config_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': action,
            'key': key,
            'value': value,
            'section': section
        })
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        return self.config_history[-limit:]
    
    def reset(self):
        self.config = {}
        self._record_change('reset', 'all', None, None)


class TaskScheduler:
    """Խնդիրների պլանավորում և կատարում"""
    
    def __init__(self, max_workers: int = 5):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks = {}
        self.task_results = {}
        self.scheduled_tasks = []
        
    def submit_task(self, func: Callable, *args, 
                   task_id: Optional[str] = None, **kwargs) -> str:
        if task_id is None:
            task_id = f"task_{len(self.tasks) + 1}_{int(time.time())}"
        
        future = self.executor.submit(func, *args, **kwargs)
        
        self.tasks[task_id] = {
            'function': func.__name__,
            'args': args,
            'kwargs': kwargs,
            'submitted_at': datetime.now().isoformat(),
            'future': future,
            'status': 'pending'
        }
        
        future.add_done_callback(
            lambda f: self._on_task_complete(task_id, f)
        )
        
        return task_id
    
    def _on_task_complete(self, task_id: str, future: Future):
        if task_id in self.tasks:
            self.tasks[task_id]['status'] = 'completed'
            self.tasks[task_id]['completed_at'] = datetime.now().isoformat()
            
            try:
                result = future.result()
                self.task_results[task_id] = {
                    'success': True,
                    'result': result
                }
            except Exception as e:
                self.task_results[task_id] = {
                    'success': False,
                    'error': str(e)
                }
    
    def schedule_task(self, func: Callable, delay: float, 
                     *args, task_id: Optional[str] = None, **kwargs) -> str:
        if task_id is None:
            task_id = f"scheduled_{len(self.scheduled_tasks) + 1}"
        
        scheduled_time = datetime.now().timestamp() + delay
        
        self.scheduled_tasks.append({
            'task_id': task_id,
            'function': func,
            'args': args,
            'kwargs': kwargs,
            'scheduled_at': scheduled_time,
            'delay': delay
        })
        
        def delayed_execution():
            time.sleep(delay)
            return self.submit_task(func, *args, task_id=task_id, **kwargs)
        
        thread = threading.Thread(target=delayed_execution)
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def schedule_recurring(self, func: Callable, interval: float,
                          task_id: Optional[str] = None, 
                          max_iterations: Optional[int] = None,
                          *args, **kwargs) -> str:
        if task_id is None:
            task_id = f"recurring_{len(self.scheduled_tasks) + 1}"
        
        iteration_count = 0
        
        def recurring_execution():
            nonlocal iteration_count
            while max_iterations is None or iteration_count < max_iterations:
                try:
                    func(*args, **kwargs)
                    iteration_count += 1
                except Exception as e:
                    pass
                time.sleep(interval)
        
        thread = threading.Thread(target=recurring_execution)
        thread.daemon = True
        thread.start()
        
        self.tasks[task_id] = {
            'function': func.__name__,
            'type': 'recurring',
            'interval': interval,
            'max_iterations': max_iterations,
            'iterations_completed': iteration_count,
            'submitted_at': datetime.now().isoformat(),
            'status': 'running'
        }
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Dict:
        if task_id not in self.tasks:
            return {'error': 'Task not found'}
        
        task = self.tasks[task_id]
        future = task.get('future')
        
        status = {
            'task_id': task_id,
            'function': task['function'],
            'status': task['status'],
            'submitted_at': task['submitted_at']
        }
        
        if future and not future.done():
            status['status'] = 'running'
        
        if task_id in self.task_results:
            status['result'] = self.task_results[task_id]
        
        return status
    
    def cancel_task(self, task_id: str) -> bool:
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        future = task.get('future')
        
        if future and not future.done():
            cancelled = future.cancel()
            if cancelled:
                task['status'] = 'cancelled'
                task['cancelled_at'] = datetime.now().isoformat()
            return cancelled
        
        return False
    
    def wait_all(self, timeout: Optional[float] = None) -> bool:
        futures = [t['future'] for t in self.tasks.values() if t.get('future')]
        
        if not futures:
            return True
        
        try:
            for future in futures:
                future.result(timeout=timeout)
            return True
        except:
            return False
    
    def shutdown(self, wait: bool = True):
        self.executor.shutdown(wait=wait)


class ProgressTracker:
    """Առաջընթացի հետևում"""
    
    def __init__(self, total: int = 100, description: str = "Progress"):
        self.total = total
        self.current = 0
        self.description = description
        self.start_time = None
        self.end_time = None
        self.checkpoints = []
        
    def start(self):
        self.start_time = datetime.now()
        self.current = 0
        self.checkpoints = []
        return self
    
    def update(self, amount: int = 1):
        self.current += amount
        if self.current > self.total:
            self.current = self.total
        
        self.checkpoints.append({
            'current': self.current,
            'timestamp': datetime.now().isoformat(),
            'percentage': self.get_percentage()
        })
        
        return self
    
    def set_current(self, value: int):
        self.current = min(value, self.total)
        self.checkpoints.append({
            'current': self.current,
            'timestamp': datetime.now().isoformat(),
            'percentage': self.get_percentage()
        })
        return self
    
    def complete(self):
        self.current = self.total
        self.end_time = datetime.now()
        return self
    
    def get_percentage(self) -> float:
        if self.total == 0:
            return 100.0
        return (self.current / self.total) * 100
    
    def get_eta(self) -> Optional[float]:
        if not self.start_time or self.current == 0:
            return None
        
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        
        if rate == 0:
            return None
        
        remaining = self.total - self.current
        eta_seconds = remaining / rate
        
        return eta_seconds
    
    def get_elapsed(self) -> float:
        if not self.start_time:
            return 0.0
        
        end = self.end_time if self.end_time else datetime.now()
        return (end - self.start_time).total_seconds()
    
    def get_speed(self) -> float:
        elapsed = self.get_elapsed()
        if elapsed == 0:
            return 0.0
        return self.current / elapsed
    
    def get_status(self) -> Dict:
        return {
            'description': self.description,
            'current': self.current,
            'total': self.total,
            'percentage': self.get_percentage(),
            'elapsed_seconds': self.get_elapsed(),
            'eta_seconds': self.get_eta(),
            'speed_per_second': self.get_speed(),
            'is_complete': self.current >= self.total
        }
    
    def print_progress(self, show_eta: bool = True, show_speed: bool = True):
        status = self.get_status()
        bar_length = 40
        filled_length = int(bar_length * status['current'] // self.total)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        output = f"\r{self.description}: [{bar}] {status['percentage']:.1f}%"
        
        if show_speed:
            output += f" | {status['speed_per_second']:.2f} it/s"
        
        if show_eta and status['eta_seconds'] is not None:
            eta_min = int(status['eta_seconds'] // 60)
            eta_sec = int(status['eta_seconds'] % 60)
            output += f" | ETA: {eta_min}:{eta_sec:02d}"
        
        if status['is_complete']:
            output += " ✓"
        
        print(output, end='', flush=True)
        
        if status['is_complete']:
            print()
    
    def reset(self):
        self.current = 0
        self.start_time = None
        self.end_time = None
        self.checkpoints = []
        return self


__all__ = ['Logger', 'ConfigManager', 'TaskScheduler', 'ProgressTracker']
