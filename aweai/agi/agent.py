from __future__ import annotations

import ast
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np


@dataclass
class Perception:
    modality: str
    data: Any
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    source: str = "environment"


@dataclass
class Thought:
    content: str
    reasoning_type: str
    confidence: float
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Action:
    action_type: str
    parameters: Dict[str, Any]
    expected_outcome: str
    timestamp: float = field(default_factory=time.time)
    priority: float = 1.0


@dataclass
class Goal:
    description: str
    priority: float = 1.0
    status: str = "pending"
    subgoals: List[Goal] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    deadline: Optional[float] = None
    success_criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_subgoal(self, goal: Goal) -> None:
        self.subgoals.append(goal)

    def is_achieved(self, context: Dict[str, Any]) -> bool:
        if self.status == "achieved":
            return True
        for criterion in self.success_criteria:
            if criterion not in context:
                return False
        return True


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: str = "{}"
    safety_level: str = "safe"
    rate_limit: Optional[int] = None


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}
        self._call_counts: Dict[str, int] = {}
        self._last_called: Dict[str, float] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool
        self._call_counts.setdefault(tool.name, 0)
        self._last_called.setdefault(tool.name, 0.0)

    def call(self, name: str, **kwargs: Any) -> Any:
        if name not in self._tools:
            raise KeyError(f"Unknown tool: {name}")
        tool = self._tools[name]
        if tool.rate_limit is not None:
            elapsed = time.time() - self._last_called[name]
            if elapsed < 1.0 / tool.rate_limit and self._call_counts[name] > 0:
                raise RuntimeError(f"Rate limit exceeded for tool: {name}")
        self._call_counts[name] += 1
        self._last_called[name] = time.time()
        try:
            parsed: Dict[str, Any] = {}
            raw = tool.parameters
            if isinstance(raw, str) and raw.strip():
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = {}
            merged = {**parsed, **kwargs}
            return tool.func(**merged)
        except TypeError:
            return tool.func(kwargs)
        except Exception as e:
            return {"error": str(e)}

    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description, "safety": t.safety_level} for t in self._tools.values()]

    def get_tool(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)


class APIClient:
    def __init__(self, base_url: str = "", headers: Optional[Dict[str, str]] = None, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = headers or {"Content-Type": "application/json"}
        self.timeout = timeout
        self._history: List[Dict[str, Any]] = []

    def call(self, endpoint: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        record = {
            "url": url,
            "method": method,
            "payload": payload,
            "timestamp": time.time(),
            "status": "simulated",
        }
        self._history.append(record)
        return {"url": url, "method": method, "simulated": True, "payload": payload}

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


class CodeExecutor:
    def __init__(self, sandbox: bool = True, max_execution_time: float = 5.0) -> None:
        self.sandbox = sandbox
        self.max_execution_time = max_execution_time
        self._allowed_imports: List[str] = ["math", "json", "random", "statistics", "collections", "itertools", "functools"]
        self._execution_log: List[Dict[str, Any]] = []

    def execute(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.sandbox:
            try:
                tree = ast.parse(code)
            except SyntaxError as e:
                return {"error": f"Syntax error: {e}", "result": None}
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] not in self._allowed_imports:
                            return {"error": f"Import not allowed: {alias.name}", "result": None}
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module.split(".")[0] not in self._allowed_imports:
                        return {"error": f"Import not allowed: {node.module}", "result": None}
        local_context: Dict[str, Any] = {"__builtins__": {}}
        if context:
            local_context.update(context)
        start = time.time()
        try:
            exec_globals: Dict[str, Any] = {}
            exec(code, exec_globals, local_context)
            result = local_context.get("result", local_context.get("output", None))
            elapsed = time.time() - start
            record = {"code": code[:200], "elapsed": elapsed, "success": True}
            self._execution_log.append(record)
            return {"result": result, "elapsed": elapsed, "success": True}
        except Exception as e:
            elapsed = time.time() - start
            record = {"code": code[:200], "elapsed": elapsed, "success": False, "error": str(e)}
            self._execution_log.append(record)
            return {"error": str(e), "elapsed": elapsed, "success": False}

    def execution_log(self) -> List[Dict[str, Any]]:
        return list(self._execution_log)


class Agent:
    def __init__(self, name: str = "aweai-agent", goal: Optional[Goal] = None) -> None:
        self.name = name
        self.goal = goal
        self.tools = ToolRegistry()
        self.api = APIClient()
        self.code_executor = CodeExecutor()
        self.memory: List[Dict[str, Any]] = []
        self.state: Dict[str, Any] = {"status": "idle", "step_count": 0, "errors": 0}
        self._perception_buffer: List[Perception] = []
        self._reflection_log: List[Thought] = []
        self._action_history: List[Action] = []
        self._plan_cache: List[Action] = []

    def perceive(self, modality: str, data: Any, confidence: float = 1.0, source: str = "environment") -> Perception:
        perception = Perception(modality=modality, data=data, confidence=confidence, source=source)
        self._perception_buffer.append(perception)
        if len(self._perception_buffer) > 1000:
            self._perception_buffer = self._perception_buffer[-1000:]
        return perception

    def process_multimodal(self, inputs: Sequence[Perception]) -> Dict[str, Any]:
        fused: Dict[str, Any] = {}
        for p in inputs:
            if p.modality not in fused:
                fused[p.modality] = {"data": [], "confidence": []}
            fused[p.modality]["data"].append(p.data)
            fused[p.modality]["confidence"].append(p.confidence)
        for modality, info in fused.items():
            confidences = info["confidence"]
            info["weighted_confidence"] = float(np.mean(confidences)) if confidences else 0.0
        return fused

    def decompose_goal(self, goal: Goal) -> List[Goal]:
        if not goal.subgoals:
            return [goal]
        all_subgoals: List[Goal] = []
        for sub in goal.subgoals:
            all_subgoals.extend(self.decompose_goal(sub))
        return all_subgoals

    def plan(self, goal: Goal) -> List[Action]:
        subgoals = self.decompose_goal(goal)
        actions: List[Action] = []
        for sg in subgoals:
            action = Action(
                action_type="pursue",
                parameters={"description": sg.description, "priority": sg.priority},
                expected_outcome=f"Complete {sg.description}",
                priority=sg.priority,
            )
            actions.append(action)
        actions.sort(key=lambda a: a.priority, reverse=True)
        self._plan_cache = actions
        return actions

    def execute_action(self, action: Action) -> Any:
        self.state["step_count"] += 1
        self._action_history.append(action)
        if action.action_type == "pursue":
            return {"status": "completed", "detail": action.parameters}
        if action.action_type == "tool":
            tool_name = action.parameters.get("tool", "")
            return self.tools.call(tool_name, **{k: v for k, v in action.parameters.items() if k != "tool"})
        if action.action_type == "api_call":
            return self.api.call(
                action.parameters.get("endpoint", ""),
                method=action.parameters.get("method", "GET"),
                payload=action.parameters.get("payload"),
            )
        if action.action_type == "code_exec":
            return self.code_executor.execute(action.parameters.get("code", ""), action.parameters.get("context"))
        return {"error": f"Unknown action type: {action.action_type}"}

    def reflect(self, observation: Any, expected: str) -> Thought:
        success = False
        if isinstance(observation, dict):
            success = observation.get("status") == "completed" or observation.get("success", False)
        confidence = 1.0 if success else 0.3
        if not success and isinstance(observation, dict) and "error" in observation:
            confidence = 0.1
        thought = Thought(
            content=f"Reflection on '{expected}': {'Success' if success else 'Failure'}",
            reasoning_type="self_reflection",
            confidence=confidence,
            metadata={"observation": str(observation)[:500], "expected": expected},
        )
        self._reflection_log.append(thought)
        return thought

    def monitor_execution(self, trace: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(trace)
        successes = sum(1 for t in trace if t.get("thought", {}).get("confidence", 0) >= 0.5)
        failures = total - successes
        avg_confidence = float(np.mean([t.get("thought", {}).get("confidence", 0) for t in trace])) if trace else 0.0
        return {
            "total_steps": total,
            "successes": successes,
            "failures": failures,
            "success_rate": successes / total if total > 0 else 0.0,
            "average_confidence": avg_confidence,
        }

    def _recover(self, action: Action, result: Any) -> Dict[str, Any]:
        if isinstance(result, dict) and "error" in result:
            error_msg = str(result["error"])
            if "Rate limit" in error_msg:
                return {"strategy": "backoff", "delay": 1.0, "original_action": action.action_type}
            if "Syntax error" in error_msg:
                return {"strategy": "fix_code", "original_action": action.action_type}
            if "not allowed" in error_msg:
                return {"strategy": "use_alternative_tool", "original_action": action.action_type}
        return {
            "strategy": "retry_with_parameters",
            "original_action": action.action_type,
            "suggestion": "Adjust parameters and retry",
        }

    def run(self, goal: Optional[Goal] = None, max_iterations: int = 20, verbose: bool = False) -> Dict[str, Any]:
        goal = goal or self.goal
        if goal is None:
            return {"error": "No goal provided"}
        self.state["status"] = "running"
        trace: List[Dict[str, Any]] = []
        actions = self.plan(goal)
        for i, action in enumerate(actions):
            if i >= max_iterations:
                break
            if verbose:
                print(f"[{self.name}] Step {i + 1}: {action.action_type} -> {action.expected_outcome}")
            result = self.execute_action(action)
            thought = self.reflect(result, action.expected_outcome)
            trace.append({"action": action, "result": result, "thought": thought})
            if thought.confidence < 0.5:
                recovery = self._recover(action, result)
                trace.append({"recovery": recovery})
                if recovery["strategy"] == "backoff":
                    time.sleep(recovery.get("delay", 1.0))
            if goal.is_achieved({"last_result": result}):
                goal.status = "achieved"
                break
        monitoring = self.monitor_execution(trace)
        self.state["status"] = "completed"
        return {"agent": self.name, "goal": goal.description, "trace": trace, "monitoring": monitoring}

    def add_tool(self, name: str, func: Callable, description: str, parameters: str = "{}", safety_level: str = "safe") -> None:
        tool = Tool(name=name, description=description, func=func, parameters=parameters, safety_level=safety_level)
        self.tools.register(tool)

    def call_api(self, endpoint: str, method: str = "GET", payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.api.call(endpoint, method=method, payload=payload)

    def execute_code(self, code: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self.code_executor.execute(code, context)

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": dict(self.state),
            "perception_count": len(self._perception_buffer),
            "reflection_count": len(self._reflection_log),
            "action_count": len(self._action_history),
            "tool_count": len(self.tools.list_tools()),
        }

    def reset(self) -> None:
        self.state = {"status": "idle", "step_count": 0, "errors": 0}
        self._perception_buffer.clear()
        self._reflection_log.clear()
        self._action_history.clear()
        self._plan_cache.clear()
        self.memory.clear()
