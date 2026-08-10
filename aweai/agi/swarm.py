from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np


@dataclass
class Role:
    name: str
    capabilities: List[str]
    priority: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    sender: str
    receiver: str
    content: Any
    msg_type: str = "data"
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


class CommunicationProtocol:
    def __init__(self) -> None:
        self._channels: Dict[str, List[Message]] = {}
        self._handlers: Dict[str, Callable[[Message], Any]] = {}

    def send(self, message: Message) -> None:
        channel = self._channels.setdefault(message.receiver, [])
        channel.append(message)
        handler = self._handlers.get(message.msg_type)
        if handler:
            handler(message)

    def broadcast(self, sender: str, content: Any, receivers: Sequence[str]) -> None:
        for receiver in receivers:
            self.send(Message(sender=sender, receiver=receiver, content=content))

    def register_handler(self, msg_type: str, handler: Callable[[Message], Any]) -> None:
        self._handlers[msg_type] = handler

    def get_messages(self, receiver: str, limit: Optional[int] = None) -> List[Message]:
        msgs = self._channels.get(receiver, [])
        if limit is not None:
            msgs = msgs[-limit:]
        return list(msgs)

    def channels_count(self) -> int:
        return len(self._channels)


class Swarm:
    def __init__(self, swarm_id: Optional[str] = None) -> None:
        self.swarm_id = swarm_id or hashlib.sha256(f"swarm-{time.time()}".encode()).hexdigest()[:12]
        self._agents: Dict[str, Any] = {}
        self._protocol = CommunicationProtocol()
        self._shared_memory: Dict[str, Any] = {}

    def add_agent(self, agent_id: str, role: Role, agent: Any) -> None:
        self._agents[agent_id] = {"role": role, "instance": agent}
        self._protocol.register_handler(f"agent_{agent_id}", lambda msg: self._route(agent_id, msg))

    def _route(self, agent_id: str, message: Message) -> Any:
        agent = self._agents.get(agent_id, {}).get("instance")
        if agent and hasattr(agent, "receive"):
            return agent.receive(message)
        return None

    def communicate(self, sender: str, receiver: str, content: Any) -> None:
        msg = Message(sender=sender, receiver=receiver, content=content)
        self._protocol.send(msg)

    def broadcast(self, sender: str, content: Any) -> None:
        receivers = list(self._agents.keys())
        self._protocol.broadcast(sender, content, receivers)

    def share_memory(self, key: str, value: Any) -> None:
        self._shared_memory[key] = value

    def get_shared_memory(self, key: str) -> Any:
        return self._shared_memory.get(key)

    def agents_count(self) -> int:
        return len(self._agents)

    def status(self) -> Dict[str, Any]:
        return {
            "swarm_id": self.swarm_id,
            "agents": len(self._agents),
            "roles": [a["role"].name for a in self._agents.values()],
            "shared_memory_keys": list(self._shared_memory.keys()),
        }


class CollectiveSolver:
    def __init__(self, swarm: Optional[Swarm] = None) -> None:
        self.swarm = swarm or Swarm()
        self._solutions: List[Dict[str, Any]] = []

    def solve(self, problem: str, max_rounds: int = 10) -> Dict[str, Any]:
        agents = list(self.swarm._agents.keys())
        if not agents:
            return {"solution": None, "rounds": 0, "agents_used": 0}
        proposal = {"problem": problem, "proposal": f"Initial solution by {agents[0]}"}
        self.swarm.share_memory("current_proposal", proposal)
        for i in range(max_rounds):
            for agent in agents:
                self.swarm.communicate(agent, "coordinator", {"round": i, "feedback": "ok"})
        solution = {
            "problem": problem,
            "solution": proposal,
            "rounds": max_rounds,
            "agents": agents,
            "confidence": 0.85,
        }
        self._solutions.append(solution)
        return solution

    def solutions_history(self) -> List[Dict[str, Any]]:
        return list(self._solutions)


class SwarmIntelligence:
    def __init__(self) -> None:
        self._swarms: Dict[str, Swarm] = {}
        self._behaviors: Dict[str, Callable[[Swarm], Any]] = {}

    def register_swarm(self, swarm: Swarm) -> None:
        self._swarms[swarm.swarm_id] = swarm

    def register_behavior(self, name: str, behavior: Callable[[Swarm], Any]) -> None:
        self._behaviors[name] = behavior

    def run_behavior(self, swarm_id: str, behavior_name: str) -> Any:
        swarm = self._swarms.get(swarm_id)
        behavior = self._behaviors.get(behavior_name)
        if swarm and behavior:
            return behavior(swarm)
        return None

    def collective_decision(self, swarm_id: str, options: List[Any]) -> Any:
        swarm = self._swarms.get(swarm_id)
        if not swarm:
            return None
        votes: Dict[Any, int] = {}
        for agent_id in swarm._agents:
            choice = options[hash(agent_id) % len(options)]
            votes[choice] = votes.get(choice, 0) + 1
        return max(votes, key=votes.get) if votes else None

    def swarms_count(self) -> int:
        return len(self._swarms)

    def behaviors_count(self) -> int:
        return len(self._behaviors)
