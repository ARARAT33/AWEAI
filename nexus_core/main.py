#!/usr/bin/env python3
"""
Nexus Core - Advanced Autonomous AI System
==========================================

Main entry point for the Nexus AI system.
This application can autonomously perform computer tasks, learn from experience,
and continuously improve its capabilities.

Usage:
    python main.py [--config CONFIG_PATH]
"""

import asyncio
import argparse
import signal
import sys
from datetime import datetime

from nexus_core.core.engine import NexusEngine
from nexus_core.utils.logger import setup_logger


class NexusApplication:
    """Main application wrapper for Nexus Core"""
    
    def __init__(self, config_path: str = None):
        self.logger = setup_logger("NexusApp")
        self.config_path = config_path
        self.engine = None
        self.running = False
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def initialize(self):
        """Initialize the Nexus system"""
        self.logger.info("=" * 60)
        self.logger.info("NEXUS CORE - Advanced Autonomous AI System")
        self.logger.info("=" * 60)
        
        try:
            # Create engine
            self.engine = NexusEngine(self.config_path)
            
            # Initialize all components
            await self.engine.initialize_components()
            
            self.logger.info("System initialization complete!")
            self.logger.info(f"Version: 1.0.0")
            self.logger.info(f"Components: Cognitive Processor, Task Executor, ")
            self.logger.info(f"            Adaptive Learner, Multi-Modal Analyzer, Knowledge Base")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    async def run_interactive(self):
        """Run in interactive mode"""
        self.logger.info("\nEntering interactive mode. Type 'help' for commands.")
        self.logger.info("Type 'exit' to quit.\n")
        
        await self.engine.start()
        self.running = True
        
        while self.running:
            try:
                # Get user input
                user_input = await asyncio.get_event_loop().run_in_executor(
                    None, input, "Nexus > "
                )
                
                if not user_input.strip():
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'q']:
                    break
                
                elif user_input.lower() == 'help':
                    self._show_help()
                
                elif user_input.lower() == 'status':
                    status = self.engine.get_status()
                    self._print_status(status)
                
                elif user_input.lower() == 'learn':
                    await self.engine.adaptive_learner.optimize_hyperparameters()
                
                else:
                    # Process as query/task
                    response = await self.engine.query(user_input)
                    self._print_response(response)
                    
            except EOFError:
                break
            except Exception as e:
                self.logger.error(f"Error: {e}")
        
        await self.engine.stop()
    
    async def run_task(self, task_description: str):
        """Run a single task"""
        await self.engine.start()
        
        task = {
            'type': 'generic',
            'description': task_description,
            'submitted_at': datetime.now()
        }
        
        task_id = await self.engine.submit_task(task)
        self.logger.info(f"Task submitted: {task_id}")
        
        # Wait for completion (simplified)
        await asyncio.sleep(5)
        
        await self.engine.stop()
    
    def _show_help(self):
        """Show help message"""
        help_text = """
Nexus Core Commands:
--------------------
  help              Show this help message
  status            Show system status
  learn             Run learning optimization
  exit/quit/q       Exit the application
  
Natural Language:
  Just type any command or question in natural language.
  Examples:
    - "Analyze the files in my documents folder"
    - "Create a Python script that does X"
    - "What's the weather like?"
    - "Automate my daily backup task"
"""
        self.logger.info(help_text)
    
    def _print_status(self, status: dict):
        """Print system status"""
        status_text = f"""
System Status:
--------------
  State: {status.get('state', 'unknown')}
  Running: {status.get('is_running', False)}
  Active Tasks: {status.get('active_tasks', 0)}
  Completed Tasks: {status.get('completed_tasks', 0)}
  
Performance:
  CPU Usage: {status.get('metrics', {}).get('cpu_usage', 0):.1f}%
  Memory Usage: {status.get('metrics', {}).get('memory_usage', 0):.1f}%
  Accuracy: {status.get('metrics', {}).get('accuracy', 0):.2f}
"""
        self.logger.info(status_text)
    
    def _print_response(self, response: dict):
        """Print query response"""
        if 'error' in response:
            self.logger.error(f"Error: {response['error']}")
        else:
            self.logger.info(f"\nAnswer: {response.get('answer', 'No answer')}")
            self.logger.info(f"Confidence: {response.get('confidence', 0):.2f}")
            if response.get('reasoning_steps'):
                self.logger.info(f"Reasoning steps: {response['reasoning_steps']}")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Nexus Core AI System')
    parser.add_argument('--config', '-c', help='Path to configuration file')
    parser.add_argument('--task', '-t', help='Run a single task')
    parser.add_argument('--interactive', '-i', action='store_true', 
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    app = NexusApplication(config_path=args.config)
    
    # Initialize
    if not await app.initialize():
        sys.exit(1)
    
    # Run
    if args.task:
        await app.run_task(args.task)
    elif args.interactive or not args.task:
        await app.run_interactive()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nNexus Core stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
