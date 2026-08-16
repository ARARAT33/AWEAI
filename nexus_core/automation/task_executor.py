"""
TaskExecutor - Autonomous computer control and automation engine
Executes tasks on the computer system with full autonomy
"""

import asyncio
import logging
import os
import subprocess
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from ..utils.logger import setup_logger


class TaskExecutor:
    """
    Autonomous task execution engine
    
    Capabilities:
    - File system operations
    - Application control
    - Web automation
    - System commands
    - Scheduled tasks
    - Error recovery
    """
    
    def __init__(self, config):
        self.logger = setup_logger("TaskExecutor")
        self.config = config
        
        # Dependencies (set by engine)
        self.knowledge_base = None
        self.cognitive_processor = None
        
        # Execution state
        self.active_executions: Dict[str, Any] = {}
        self.execution_history: List[Dict] = []
        self.max_history_length = 1000
        
        # Safety settings
        self.safe_mode = True
        self.allowed_directories = self._get_allowed_directories()
        self.blocked_commands = ['rm -rf /', 'format', 'del /s', 'shutdown']
        
        # Performance settings
        self.max_concurrent_executions = 5
        self.timeout_seconds = 300
        
        self.logger.info("Task Executor initialized")
    
    def _get_allowed_directories(self) -> List[Path]:
        """Get list of directories where operations are allowed"""
        allowed = [
            Path.home() / 'Documents',
            Path.home() / 'Downloads',
            Path.home() / 'Desktop',
            Path.cwd()
        ]
        
        # Create if they don't exist
        for directory in allowed:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        
        return allowed
    
    async def execute_automation_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an automation task
        
        Args:
            task: Task dictionary with action and parameters
            
        Returns:
            Execution result with output and status
        """
        task_id = task.get('id', 'unknown')
        action = task.get('action', '')
        parameters = task.get('parameters', {})
        
        self.logger.info(f"Executing automation task {task_id}: {action}")
        
        self.active_executions[task_id] = {
            'start_time': datetime.now(),
            'action': action,
            'status': 'running'
        }
        
        try:
            # Route to appropriate executor
            if action == 'file_operation':
                result = await self._execute_file_operation(parameters)
            elif action == 'system_command':
                result = await self._execute_system_command(parameters)
            elif action == 'web_automation':
                result = await self._execute_web_automation(parameters)
            elif action == 'application_control':
                result = await self._execute_application_control(parameters)
            elif action == 'scheduled_task':
                result = await self._execute_scheduled_task(parameters)
            else:
                result = await self._execute_generic_action(action, parameters)
            
            # Record success
            self._record_execution(task_id, action, result, success=True)
            
        except Exception as e:
            self.logger.error(f"Task {task_id} failed: {e}")
            result = {
                'success': False,
                'error': str(e),
                'output': None
            }
            self._record_execution(task_id, action, result, success=False)
        
        finally:
            if task_id in self.active_executions:
                del self.active_executions[task_id]
        
        return result
    
    async def _execute_file_operation(self, params: Dict) -> Dict[str, Any]:
        """Execute file system operations"""
        operation = params.get('operation', '')
        source = params.get('source', '')
        destination = params.get('destination', '')
        content = params.get('content', '')
        
        # Security check
        if not self._is_path_safe(source) or not self._is_path_safe(destination):
            raise ValueError("Operation outside allowed directories")
        
        result = {'success': True, 'operation': operation}
        
        if operation == 'create':
            path = Path(source)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            result['path'] = str(path)
            
        elif operation == 'read':
            path = Path(source)
            result['content'] = path.read_text()
            
        elif operation == 'copy':
            shutil.copy2(source, destination)
            result['destination'] = destination
            
        elif operation == 'move':
            shutil.move(source, destination)
            result['destination'] = destination
            
        elif operation == 'delete':
            path = Path(source)
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            result['deleted'] = source
            
        elif operation == 'list':
            path = Path(source)
            result['files'] = [str(f) for f in path.iterdir()]
            
        elif operation == 'search':
            pattern = params.get('pattern', '*')
            path = Path(source)
            matches = list(path.rglob(pattern))
            result['matches'] = [str(m) for m in matches[:100]]
        
        return result
    
    async def _execute_system_command(self, params: Dict) -> Dict[str, Any]:
        """Execute system commands safely"""
        command = params.get('command', '')
        shell = params.get('shell', False)
        timeout = params.get('timeout', self.timeout_seconds)
        
        # Security check
        if any(blocked in command for blocked in self.blocked_commands):
            raise ValueError("Command blocked for safety")
        
        self.logger.info(f"Executing command: {command}")
        
        try:
            process = await asyncio.create_subprocess_shell(
                command if shell else command.split(),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout
            )
            
            return {
                'success': process.returncode == 0,
                'return_code': process.returncode,
                'stdout': stdout.decode() if stdout else '',
                'stderr': stderr.decode() if stderr else '',
                'command': command
            }
            
        except asyncio.TimeoutError:
            process.kill()
            return {
                'success': False,
                'error': 'Command timed out',
                'command': command
            }
    
    async def _execute_web_automation(self, params: Dict) -> Dict[str, Any]:
        """Execute web automation tasks"""
        try:
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support.ui import WebDriverWait
            from selenium.webdriver.support import expected_conditions as EC
        except ImportError:
            return {
                'success': False,
                'error': 'Selenium not installed',
                'message': 'Install selenium: pip install selenium'
            }
        
        url = params.get('url', '')
        action = params.get('action', 'navigate')
        selector = params.get('selector', '')
        value = params.get('value', '')
        
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        
        driver = None
        try:
            driver = webdriver.Chrome(options=options)
            driver.set_page_load_timeout(30)
            
            if action == 'navigate':
                driver.get(url)
                result = {
                    'success': True,
                    'title': driver.title,
                    'url': driver.current_url
                }
            
            elif action == 'click':
                driver.get(url)
                element = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                )
                element.click()
                result = {'success': True, 'action': 'clicked'}
            
            elif action == 'input':
                driver.get(url)
                element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                element.clear()
                element.send_keys(value)
                result = {'success': True, 'action': 'input_entered'}
            
            elif action == 'extract':
                driver.get(url)
                element = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                )
                result = {
                    'success': True,
                    'content': element.text,
                    'html': element.get_attribute('innerHTML')
                }
            
            else:
                result = {'success': False, 'error': f'Unknown action: {action}'}
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        finally:
            if driver:
                driver.quit()
    
    async def _execute_application_control(self, params: Dict) -> Dict[str, Any]:
        """Control desktop applications"""
        try:
            import pyautogui
        except ImportError:
            return {
                'success': False,
                'error': 'PyAutoGUI not installed',
                'message': 'Install pyautogui: pip install pyautogui'
            }
        
        action = params.get('action', '')
        x = params.get('x', 0)
        y = params.get('y', 0)
        keys = params.get('keys', '')
        text = params.get('text', '')
        duration = params.get('duration', 0.5)
        
        pyautogui.FAILSAFE = True
        
        try:
            if action == 'click':
                pyautogui.click(x=x, y=y)
                result = {'success': True, 'action': 'clicked', 'position': (x, y)}
            
            elif action == 'type':
                pyautogui.write(text, interval=0.05)
                result = {'success': True, 'action': 'typed', 'length': len(text)}
            
            elif action == 'press':
                pyautogui.press(keys)
                result = {'success': True, 'action': 'pressed', 'keys': keys}
            
            elif action == 'hotkey':
                pyautogui.hotkey(*keys.split('+'))
                result = {'success': True, 'action': 'hotkey', 'combination': keys}
            
            elif action == 'move':
                pyautogui.moveTo(x, y, duration=duration)
                result = {'success': True, 'action': 'moved', 'position': (x, y)}
            
            elif action == 'screenshot':
                screenshot = pyautogui.screenshot()
                save_path = params.get('save_path', None)
                if save_path:
                    screenshot.save(save_path)
                result = {
                    'success': True,
                    'action': 'screenshot',
                    'size': screenshot.size,
                    'saved': save_path
                }
            
            else:
                result = {'success': False, 'error': f'Unknown action: {action}'}
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def _execute_scheduled_task(self, params: Dict) -> Dict[str, Any]:
        """Execute or manage scheduled tasks"""
        import schedule
        
        action = params.get('action', '')
        task_func = params.get('task_func')
        interval = params.get('interval', 60)
        time_str = params.get('time', '')
        
        if action == 'schedule_daily':
            schedule.every().day.at(time_str).do(task_func)
            result = {'success': True, 'scheduled': 'daily', 'time': time_str}
        
        elif action == 'schedule_hourly':
            schedule.every().hour.do(task_func)
            result = {'success': True, 'scheduled': 'hourly'}
        
        elif action == 'schedule_interval':
            schedule.every(interval).seconds.do(task_func)
            result = {'success': True, 'scheduled': 'interval', 'seconds': interval}
        
        else:
            result = {'success': False, 'error': f'Unknown scheduling action: {action}'}
        
        return result
    
    async def _execute_generic_action(self, action: str, params: Dict) -> Dict[str, Any]:
        """Execute generic actions not covered by specific methods"""
        # Try to interpret and execute based on action name
        self.logger.info(f"Executing generic action: {action}")
        
        # Common generic actions
        if action.startswith('wait'):
            delay = params.get('delay', 1)
            await asyncio.sleep(delay)
            return {'success': True, 'action': 'wait', 'duration': delay}
        
        elif action.startswith('log'):
            message = params.get('message', '')
            self.logger.info(f"Log: {message}")
            return {'success': True, 'action': 'log', 'message': message}
        
        elif action.startswith('notify'):
            message = params.get('message', '')
            # Send notification (platform-specific)
            self.logger.info(f"Notification: {message}")
            return {'success': True, 'action': 'notify', 'message': message}
        
        else:
            return {
                'success': False,
                'error': f'Unknown action: {action}',
                'suggestion': 'Specify action type: file_operation, system_command, etc.'
            }
    
    def _is_path_safe(self, path: str) -> bool:
        """Check if path is within allowed directories"""
        if not path:
            return True
        
        try:
            path_obj = Path(path).resolve()
            
            # Check against allowed directories
            for allowed in self.allowed_directories:
                try:
                    path_obj.relative_to(allowed.resolve())
                    return True
                except ValueError:
                    continue
            
            # If safe mode is off, allow other paths
            if not self.safe_mode:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _record_execution(self, task_id: str, action: str, result: Dict, 
                          success: bool):
        """Record execution in history"""
        record = {
            'task_id': task_id,
            'action': action,
            'timestamp': datetime.now(),
            'success': success,
            'result': result
        }
        
        self.execution_history.append(record)
        
        # Trim history if too long
        if len(self.execution_history) > self.max_history_length:
            self.execution_history = self.execution_history[-self.max_history_length:]
        
        # Learn from execution
        if self.knowledge_base:
            asyncio.create_task(
                self.knowledge_base.store_execution_experience(record)
            )
    
    def get_execution_status(self, task_id: str) -> Optional[Dict]:
        """Get status of a running execution"""
        return self.active_executions.get(task_id)
    
    def get_history(self, limit: int = 50) -> List[Dict]:
        """Get recent execution history"""
        return self.execution_history[-limit:]
    
    def clear_history(self):
        """Clear execution history"""
        self.execution_history.clear()
        self.logger.info("Execution history cleared")
    
    def set_safe_mode(self, enabled: bool):
        """Enable or disable safe mode"""
        self.safe_mode = enabled
        self.logger.info(f"Safe mode {'enabled' if enabled else 'disabled'}")
    
    def add_allowed_directory(self, directory: str):
        """Add an allowed directory"""
        path = Path(directory).resolve()
        if path not in self.allowed_directories:
            self.allowed_directories.append(path)
            self.logger.info(f"Added allowed directory: {path}")
