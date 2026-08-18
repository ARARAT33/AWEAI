"""
System Orchestrator - Central coordination system for AI Fabric Core
Manages workflows, task scheduling, and system-wide operations
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Priority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a computational task"""
    id: str
    name: str
    func: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: dict = field(default_factory=dict)
    priority: Priority = Priority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None
    retries: int = 0
    max_retries: int = 3


@dataclass
class Workflow:
    """Represents a workflow of connected tasks"""
    id: str
    name: str
    tasks: List[Task] = field(default_factory=list)
    dependencies: Dict[str, List[str]] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)


class SystemOrchestrator:
    """
    Central orchestration system for managing AI workloads
    
    Features:
    - Task scheduling with priority queues
    - Workflow management with dependency resolution
    - Resource allocation and load balancing
    - Fault tolerance with automatic retries
    - Real-time monitoring and logging
    """
    
    def __init__(self, config: Optional[Any] = None):
        """
        Initialize the system orchestrator
        
        Args:
            config: System configuration object
        """
        self.config = config
        self.tasks: Dict[str, Task] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.task_queue: List[Task] = []
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        # Callbacks
        self.on_task_start: Optional[Callable] = None
        self.on_task_complete: Optional[Callable] = None
        self.on_task_fail: Optional[Callable] = None
        self.on_workflow_complete: Optional[Callable] = None
        
        # Logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Statistics
        self.stats = {
            'total_tasks_submitted': 0,
            'total_tasks_completed': 0,
            'total_tasks_failed': 0,
            'total_workflows_submitted': 0,
            'total_workflows_completed': 0,
            'average_task_duration': 0.0,
        }
    
    def submit_task(
        self,
        name: str,
        func: Callable,
        *args,
        priority: Priority = Priority.NORMAL,
        max_retries: int = 3,
        **kwargs
    ) -> str:
        """
        Submit a task for execution
        
        Args:
            name: Task name
            func: Function to execute
            *args: Positional arguments for the function
            priority: Task priority level
            max_retries: Maximum number of retry attempts
            **kwargs: Keyword arguments for the function
            
        Returns:
            Task ID
        """
        task_id = f"task_{len(self.tasks) + 1}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        task = Task(
            id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
        )
        
        self.tasks[task_id] = task
        self._insert_into_queue(task)
        self.stats['total_tasks_submitted'] += 1
        
        self.logger.info(f"Task submitted: {task_id} - {name}")
        return task_id
    
    def submit_workflow(self, name: str, tasks: List[Task], 
                       dependencies: Optional[Dict[str, List[str]]] = None) -> str:
        """
        Submit a workflow for execution
        
        Args:
            name: Workflow name
            tasks: List of tasks in the workflow
            dependencies: Task dependencies (task_id -> [dependency_task_ids])
            
        Returns:
            Workflow ID
        """
        workflow_id = f"workflow_{len(self.workflows) + 1}"
        
        workflow = Workflow(
            id=workflow_id,
            name=name,
            tasks=tasks,
            dependencies=dependencies or {},
        )
        
        self.workflows[workflow_id] = workflow
        self.stats['total_workflows_submitted'] += 1
        
        self.logger.info(f"Workflow submitted: {workflow_id} - {name}")
        return workflow_id
    
    async def execute_task(self, task: Task) -> Any:
        """
        Execute a single task
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_tasks[task.id] = task
        
        if self.on_task_start:
            self.on_task_start(task)
        
        try:
            # Execute task
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)
            
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            
            # Move to completed
            del self.running_tasks[task.id]
            self.completed_tasks[task.id] = task
            self.stats['total_tasks_completed'] += 1
            
            if self.on_task_complete:
                self.on_task_complete(task)
            
            self.logger.info(f"Task completed: {task.id}")
            return result
            
        except Exception as e:
            task.error = str(e)
            task.retries += 1
            
            if task.retries < task.max_retries:
                self.logger.warning(f"Task {task.id} failed, retrying ({task.retries}/{task.max_retries})")
                self._insert_into_queue(task)
                return None
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                
                del self.running_tasks[task.id]
                self.failed_tasks[task.id] = task
                self.stats['total_tasks_failed'] += 1
                
                if self.on_task_fail:
                    self.on_task_fail(task)
                
                self.logger.error(f"Task failed permanently: {task.id} - {e}")
                raise
    
    async def execute_workflow(self, workflow: Workflow) -> Dict[str, Any]:
        """
        Execute a workflow with dependency resolution
        
        Args:
            workflow: Workflow to execute
            
        Returns:
            Dictionary of task results
        """
        workflow.status = TaskStatus.RUNNING
        results = {}
        completed_task_ids = set()
        
        # Topological sort based on dependencies
        pending_tasks = set(task.id for task in workflow.tasks)
        
        while pending_tasks:
            # Find tasks with all dependencies satisfied
            ready_tasks = []
            for task_id in pending_tasks:
                deps = workflow.dependencies.get(task_id, [])
                if all(dep in completed_task_ids for dep in deps):
                    task = next(t for t in workflow.tasks if t.id == task_id)
                    ready_tasks.append(task)
            
            if not ready_tasks:
                if pending_tasks:
                    raise ValueError("Circular dependency detected in workflow")
                break
            
            # Execute ready tasks in parallel
            tasks_to_execute = [self.execute_task(task) for task in ready_tasks]
            task_results = await asyncio.gather(*tasks_to_execute, return_exceptions=True)
            
            # Process results
            for task, result in zip(ready_tasks, task_results):
                if isinstance(result, Exception):
                    workflow.status = TaskStatus.FAILED
                    self.logger.error(f"Workflow {workflow.id} failed at task {task.id}")
                    break
                
                results[task.id] = result
                completed_task_ids.add(task.id)
                pending_tasks.remove(task.id)
        
        if workflow.status != TaskStatus.FAILED:
            workflow.status = TaskStatus.COMPLETED
            self.stats['total_workflows_completed'] += 1
            
            if self.on_workflow_complete:
                self.on_workflow_complete(workflow)
        
        return results
    
    def _insert_into_queue(self, task: Task) -> None:
        """Insert task into priority queue"""
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: (-t.priority.value, t.created_at))
    
    async def run_scheduler(self) -> None:
        """Run the task scheduler"""
        while True:
            if self.task_queue:
                task = self.task_queue.pop(0)
                await self.execute_task(task)
            else:
                await asyncio.sleep(0.1)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return None
    
    def get_workflow_status(self, workflow_id: str) -> Optional[TaskStatus]:
        """Get status of a workflow"""
        if workflow_id in self.workflows:
            return self.workflows[workflow_id].status
        return None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return self.stats.copy()
    
    def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending task"""
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        if task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            self.task_queue = [t for t in self.task_queue if t.id != task_id]
            return True
        
        return False
    
    def __repr__(self) -> str:
        return (f"SystemOrchestrator(tasks={len(self.tasks)}, "
                f"running={len(self.running_tasks)}, "
                f"completed={len(self.completed_tasks)}, "
                f"failed={len(self.failed_tasks)})")
