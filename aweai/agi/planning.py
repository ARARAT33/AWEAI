from __future__ import annotations

import heapq
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PlanStep:
    id: str
    action: str
    parameters: Dict[str, Any]
    preconditions: List[str] = field(default_factory=list)
    effects: List[str] = field(default_factory=list)
    cost: float = 1.0
    priority: float = 1.0
    status: str = "pending"
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Plan:
    goal_id: str
    steps: List[PlanStep]
    estimated_cost: float = 0.0
    estimated_duration: float = 0.0
    status: str = "draft"
    created_at: float = field(default_factory=time.time)


class Planner:
    def __init__(self) -> None:
        self._plans: Dict[str, Plan] = {}
        self._actions: Dict[str, PlanStep] = {}

    def register_action(self, action: PlanStep) -> None:
        self._actions[action.id] = action

    def create_plan(self, goal_id: str, steps: List[PlanStep]) -> Plan:
        plan = Plan(goal_id=goal_id, steps=steps)
        plan.estimated_cost = sum(s.cost for s in steps)
        plan.estimated_duration = len(steps)
        self._plans[goal_id] = plan
        return plan

    def execute_plan(self, plan: Plan) -> Dict[str, Any]:
        results = []
        for step in plan.steps:
            step.status = "running"
            results.append({"step_id": step.id, "status": "completed", "output": f"executed_{step.action}"})
            step.status = "completed"
        plan.status = "completed"
        return {"plan_id": plan.goal_id, "results": results, "status": plan.status}

    def get_plan(self, goal_id: str) -> Optional[Plan]:
        return self._plans.get(goal_id)

    def plans(self) -> Dict[str, Plan]:
        return dict(self._plans)


class HTNPlanner:
    def __init__(self) -> None:
        self._methods: Dict[str, List[List[PlanStep]]] = {}
        self._operators: Dict[str, PlanStep] = {}

    def add_method(self, task: str, subtasks: List[PlanStep]) -> None:
        self._methods.setdefault(task, []).append(subtasks)

    def add_operator(self, name: str, step: PlanStep) -> None:
        self._operators[name] = step

    def decompose(self, task: str) -> List[PlanStep]:
        methods = self._methods.get(task, [])
        if not methods:
            return []
        return methods[0]

    def refine(self, plan: List[PlanStep]) -> List[PlanStep]:
        refined: List[PlanStep] = []
        for step in plan:
            subtasks = self.decompose(step.action)
            if subtasks:
                refined.extend(subtasks)
            else:
                refined.append(step)
        return refined

    def methods_count(self) -> int:
        return sum(len(v) for v in self._methods.values())

    def operators_count(self) -> int:
        return len(self._operators)


class ConstraintSatisfier:
    def __init__(self) -> None:
        self._constraints: List[Callable[[Dict[str, Any]], bool]] = []

    def add_constraint(self, constraint: Callable[[Dict[str, Any]], bool]) -> None:
        self._constraints.append(constraint)

    def satisfy(self, variables: Dict[str, Any]) -> bool:
        return all(c(variables) for c in self._constraints)

    def find_satisfying_assignment(self, domain: Dict[str, List[Any]], max_attempts: int = 1000) -> Optional[Dict[str, Any]]:
        import itertools
        keys = list(domain.keys())
        values = [domain[k] for k in keys]
        for i, combo in enumerate(itertools.product(*values)):
            if i >= max_attempts:
                return None
            assignment = dict(zip(keys, combo))
            if self.satisfy(assignment):
                return assignment
        return None

    def constraints_count(self) -> int:
        return len(self._constraints)


class DynamicReplanner:
    def __init__(self, planner: Optional[Planner] = None) -> None:
        self.planner = planner or Planner()
        self._replan_history: List[Dict[str, Any]] = []

    def replan(self, current_plan: Plan, failure: Dict[str, Any]) -> Optional[Plan]:
        remaining_steps = [s for s in current_plan.steps if s.status == "pending"]
        new_plan = Plan(goal_id=current_plan.goal_id, steps=remaining_steps)
        new_plan.estimated_cost = sum(s.cost for s in remaining_steps)
        record = {"original_plan": current_plan.goal_id, "failure": failure, "new_plan": new_plan.goal_id}
        self._replan_history.append(record)
        self.planner._plans[new_plan.goal_id] = new_plan
        return new_plan

    def replan_history(self) -> List[Dict[str, Any]]:
        return list(self._replan_history)


class ContingencyPlanner:
    def __init__(self) -> None:
        self._contingencies: Dict[str, List[PlanStep]] = {}

    def add_contingency(self, trigger: str, fallback_steps: List[PlanStep]) -> None:
        self._contingencies[trigger] = fallback_steps

    def get_fallback(self, trigger: str) -> List[PlanStep]:
        return self._contingencies.get(trigger, [])

    def contingencies_count(self) -> int:
        return len(self._contingencies)


class MultiAgentCoordinator:
    def __init__(self) -> None:
        self._agents: Dict[str, Any] = {}
        self._assignments: Dict[str, str] = {}

    def register_agent(self, agent_id: str, agent: Any) -> None:
        self._agents[agent_id] = agent

    def assign_task(self, agent_id: str, task: PlanStep) -> Dict[str, Any]:
        self._assignments[task.id] = agent_id
        return {"agent": agent_id, "task": task.id, "status": "assigned"}

    def coordinate(self, plan: Plan) -> Dict[str, Any]:
        results = []
        for step in plan.steps:
            agent_id = self._assignments.get(step.id, "default")
            results.append({"step": step.id, "agent": agent_id, "status": "coordinated"})
        return {"plan_id": plan.goal_id, "assignments": results}

    def agents_count(self) -> int:
        return len(self._agents)
