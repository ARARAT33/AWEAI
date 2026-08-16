"""
NexusEngine - Central orchestrator for all AI operations
Manages cognitive processing, task execution, learning, and system coordination
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..utils.logger import setup_logger
from ..utils.config_manager import ConfigManager


class SystemState(Enum):
    """System operational states"""
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    EXECUTING = "executing"
    OPTIMIZING = "optimizing"
    ERROR = "error"


@dataclass
class TaskResult:
    """Result of a completed task"""
    task_id: str
    success: bool
    result: Any
    execution_time: float
    confidence_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """Real-time system performance metrics"""
    cpu_usage: float
    memory_usage: float
    active_tasks: int
    completed_tasks: int
    learning_rate: float
    accuracy: float
    response_time: float
    timestamp: datetime


class NexusEngine:
    """
    Central AI Engine that coordinates all subsystems
    
    Features:
    - Autonomous decision making
    - Multi-task orchestration
    - Self-optimization
    - Real-time adaptation
    - Resource management
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = setup_logger("NexusEngine")
        self.config = ConfigManager(config_path)
        
        # System state
        self.state = SystemState.IDLE
        self.is_running = False
        self.start_time = None
        
        # Core components (to be initialized)
        self.cognitive_processor = None
        self.task_executor = None
        self.adaptive_learner = None
        self.multi_modal_analyzer = None
        self.knowledge_base = None
        
        # Task management
        self.task_queue = asyncio.Queue()
        self.active_tasks: Dict[str, Any] = {}
        self.completed_tasks: List[TaskResult] = []
        self.task_history: List[Dict] = []
        
        # Performance metrics
        self.metrics = SystemMetrics(
            cpu_usage=0.0,
            memory_usage=0.0,
            active_tasks=0,
            completed_tasks=0,
            learning_rate=0.01,
            accuracy=0.0,
            response_time=0.0,
            timestamp=datetime.now()
        )
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = {}
        
        # Optimization parameters
        self.optimization_level = 0.8
        self.auto_learn = True
        self.max_concurrent_tasks = 10
        
        self.logger.info("Nexus Engine initialized successfully")
    
    async def initialize_components(self):
        """Initialize all AI subsystems"""
        self.logger.info("Initializing AI subsystems...")
        
        try:
            # Import components dynamically to avoid circular imports
            from ..brain.cognitive_processor import CognitiveProcessor
            from ..automation.task_executor import TaskExecutor
            from ..learning.adaptive_learner import AdaptiveLearner
            from ..perception.multi_modal_analyzer import MultiModalAnalyzer
            from ..memory.knowledge_base import KnowledgeBase
            
            # Initialize components
            self.cognitive_processor = CognitiveProcessor(self.config)
            self.task_executor = TaskExecutor(self.config)
            self.adaptive_learner = AdaptiveLearner(self.config)
            self.multi_modal_analyzer = MultiModalAnalyzer(self.config)
            self.knowledge_base = KnowledgeBase(self.config)
            
            # Link components to engine
            await self._link_components()
            
            self.logger.info("All subsystems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {e}")
            self.state = SystemState.ERROR
            raise
    
    async def _link_components(self):
        """Create connections between subsystems"""
        # Share knowledge base across components
        if self.cognitive_processor:
            self.cognitive_processor.knowledge_base = self.knowledge_base
        
        if self.task_executor:
            self.task_executor.knowledge_base = self.knowledge_base
            self.task_executor.cognitive_processor = self.cognitive_processor
        
        if self.adaptive_learner:
            self.adaptive_learner.knowledge_base = self.knowledge_base
            self.adaptive_learner.cognitive_processor = self.cognitive_processor
        
        if self.multi_modal_analyzer:
            self.multi_modal_analyzer.knowledge_base = self.knowledge_base
    
    async def start(self):
        """Start the Nexus Engine"""
        if self.is_running:
            self.logger.warning("Engine already running")
            return
        
        self.logger.info("Starting Nexus Engine...")
        self.is_running = True
        self.start_time = datetime.now()
        self.state = SystemState.PROCESSING
        
        # Start background tasks
        asyncio.create_task(self._process_task_queue())
        asyncio.create_task(self._monitor_system())
        asyncio.create_task(self._auto_optimize())
        
        self.logger.info("Nexus Engine started successfully")
    
    async def stop(self):
        """Gracefully shutdown the engine"""
        self.logger.info("Shutting down Nexus Engine...")
        self.is_running = False
        self.state = SystemState.IDLE
        
        # Save state
        if self.knowledge_base:
            await self.knowledge_base.save_state()
        
        if self.adaptive_learner:
            await self.adaptive_learner.save_model()
        
        self.logger.info("Nexus Engine stopped")
    
    async def submit_task(self, task: Dict[str, Any]) -> str:
        """
        Submit a new task for execution
        
        Args:
            task: Task dictionary with type, parameters, and metadata
            
        Returns:
            task_id: Unique identifier for the task
        """
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000)}"
        task['id'] = task_id
        task['submitted_at'] = datetime.now()
        task['status'] = 'pending'
        
        await self.task_queue.put(task)
        self.active_tasks[task_id] = task
        
        self.logger.info(f"Task submitted: {task_id}")
        return task_id
    
    async def _process_task_queue(self):
        """Continuously process tasks from the queue"""
        while self.is_running:
            try:
                if not self.task_queue.empty() and len(self.active_tasks) < self.max_concurrent_tasks:
                    task = await self.task_queue.get()
                    asyncio.create_task(self._execute_task(task))
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.error(f"Error processing task queue: {e}")
                await asyncio.sleep(1)
    
    async def _execute_task(self, task: Dict[str, Any]):
        """Execute a single task"""
        start_time = datetime.now()
        task_id = task.get('id')
        
        try:
            self.state = SystemState.EXECUTING
            task['status'] = 'running'
            
            # Analyze task requirements
            analysis = await self.cognitive_processor.analyze_task(task)
            
            # Execute based on task type
            task_type = task.get('type', 'generic')
            
            if task_type == 'automation':
                result = await self.task_executor.execute_automation_task(task)
            elif task_type == 'analysis':
                result = await self.multi_modal_analyzer.analyze(task)
            elif task_type == 'learning':
                result = await self.adaptive_learner.learn(task)
            else:
                result = await self.cognitive_processor.process(task)
            
            # Record success
            execution_time = (datetime.now() - start_time).total_seconds()
            task_result = TaskResult(
                task_id=task_id,
                success=True,
                result=result,
                execution_time=execution_time,
                confidence_score=analysis.get('confidence', 0.9),
                metadata=task.get('metadata', {})
            )
            
            self.completed_tasks.append(task_result)
            self.metrics.completed_tasks += 1
            
            # Learn from successful execution
            if self.auto_learn:
                await self.adaptive_learner.record_experience(task, result, success=True)
            
            task['status'] = 'completed'
            task['completed_at'] = datetime.now()
            
            self.logger.info(f"Task completed: {task_id} in {execution_time:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Task failed: {task_id} - {e}")
            task['status'] = 'failed'
            task['error'] = str(e)
            
            # Learn from failure
            if self.auto_learn:
                await self.adaptive_learner.record_experience(task, None, success=False)
            
            # Emit error event
            await self._emit_event('task_error', {'task_id': task_id, 'error': str(e)})
        
        finally:
            if task_id in self.active_tasks:
                del self.active_tasks[task_id]
            
            self.state = SystemState.PROCESSING
    
    async def _monitor_system(self):
        """Monitor system health and performance"""
        while self.is_running:
            try:
                import psutil
                
                # Update metrics
                self.metrics.cpu_usage = psutil.cpu_percent()
                self.metrics.memory_usage = psutil.virtual_memory().percent
                self.metrics.active_tasks = len(self.active_tasks)
                self.metrics.timestamp = datetime.now()
                
                # Check for resource constraints
                if self.metrics.cpu_usage > 90 or self.metrics.memory_usage > 90:
                    self.logger.warning("High resource usage detected")
                    await self._optimize_resource_usage()
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _auto_optimize(self):
        """Automatically optimize system performance"""
        while self.is_running:
            try:
                if self.optimization_level > 0.5:
                    # Optimize task scheduling
                    await self._optimize_task_scheduling()
                    
                    # Optimize memory usage
                    await self._optimize_memory()
                    
                    # Update learning rate based on performance
                    if self.adaptive_learner:
                        await self.adaptive_learner.adjust_learning_rate()
                
                await asyncio.sleep(60)  # Optimize every minute
                
            except Exception as e:
                self.logger.error(f"Optimization error: {e}")
                await asyncio.sleep(120)
    
    async def _optimize_resource_usage(self):
        """Reduce resource consumption when under pressure"""
        self.logger.info("Optimizing resource usage...")
        
        # Reduce concurrent tasks
        self.max_concurrent_tasks = max(3, self.max_concurrent_tasks - 2)
        
        # Trigger garbage collection
        import gc
        gc.collect()
        
        self.logger.info(f"Reduced max concurrent tasks to {self.max_concurrent_tasks}")
    
    async def _optimize_task_scheduling(self):
        """Optimize task scheduling algorithm"""
        # Implement advanced scheduling logic here
        pass
    
    async def _optimize_memory(self):
        """Optimize memory usage"""
        if self.knowledge_base:
            await self.knowledge_base.cleanup_old_entries()
    
    async def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a natural language query
        
        Args:
            query_text: User's question or command
            
        Returns:
            Response dictionary with answer and metadata
        """
        if not self.cognitive_processor:
            return {"error": "Engine not initialized"}
        
        response = await self.cognitive_processor.process_query(query_text)
        return response
    
    async def learn_from_feedback(self, task_id: str, feedback: Dict[str, Any]):
        """Learn from user feedback on completed tasks"""
        if self.adaptive_learner:
            await self.adaptive_learner.process_feedback(task_id, feedback)
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register a callback for system events"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    async def _emit_event(self, event_type: str, data: Dict[str, Any]):
        """Emit an event to all registered handlers"""
        if event_type in self.event_handlers:
            for handler in self.event_handlers[event_type]:
                try:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(data)
                    else:
                        handler(data)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "state": self.state.value,
            "is_running": self.is_running,
            "start_time": str(self.start_time) if self.start_time else None,
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "metrics": {
                "cpu_usage": self.metrics.cpu_usage,
                "memory_usage": self.metrics.memory_usage,
                "accuracy": self.metrics.accuracy,
                "response_time": self.metrics.response_time
            },
            "components": {
                "cognitive_processor": self.cognitive_processor is not None,
                "task_executor": self.task_executor is not None,
                "adaptive_learner": self.adaptive_learner is not None,
                "multi_modal_analyzer": self.multi_modal_analyzer is not None,
                "knowledge_base": self.knowledge_base is not None
            }
        }
    
    def __repr__(self):
        return f"NexusEngine(state={self.state.value}, tasks={len(self.active_tasks)})"
