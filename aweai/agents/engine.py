"""ReAct agent engine with pluggable tools.

The agent alternates Thought -> Action -> Observation -> ... -> Final Answer,
powered by any LLM (local, API, or fallback). Tools are plain callables
registered with a name and description; the LLM picks tools by name.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from aweai.models.inference import LLM
from aweai.utils import truncate


@dataclass
class Tool:
    name: str
    description: str
    func: Callable
    parameters: str = "{}"

    def call(self, args: str) -> str:
        try:
            parsed = json.loads(args) if args.strip() else {}
            if not isinstance(parsed, dict):
                parsed = {}
        except json.JSONDecodeError:
            parsed = {"query": args.strip()}
        try:
            result = self.func(**parsed)
            return str(result)
        except TypeError:
            try:
                result = self.func(parsed)
                return str(result)
            except Exception as e:  # pragma: no cover
                return f"Error: {e}"
        except Exception as e:  # pragma: no cover
            return f"Error: {e}"


@dataclass
class Agent:
    name: str = "aweai-agent"
    system_prompt: str = (
        "You are AWEAI Agent, a helpful AI assistant that can use tools. "
        "You answer in the language of the user's question."
    )
    max_steps: int = 6
    tools: List[Tool] = field(default_factory=list)
    llm: Optional[LLM] = None
    history: List[Dict] = field(default_factory=list)

    def add_tool(self, name: str, description: str, func: Callable, parameters: str = "{}") -> None:
        self.tools.append(Tool(name=name, description=description, func=func, parameters=parameters))

    def _tool_prompt(self) -> str:
        if not self.tools:
            return ""
        lines = ["Available tools (call them with JSON args in Action):"]
        for t in self.tools:
            lines.append(f"- {t.name}: {t.description} | args schema: {t.parameters}")
        return "\n".join(lines)

    def run(self, task: str, max_steps: Optional[int] = None, verbose: bool = True) -> Dict:
        max_steps = max_steps or self.max_steps
        llm = self.llm or LLM()
        steps: List[Dict] = []
        final = ""
        current_task = task

        for step in range(1, max_steps + 1):
            prompt = self._build_prompt(current_task, steps)
            output = llm.generate(prompt, max_tokens=512)
            steps.append({"step": step, "thought": output, "action": None, "observation": None})

            if "Final Answer:" in output or "FINAL:" in output:
                final = re.split(r"Final Answer:|FINAL:", output)[-1].strip()
                break

            action_match = re.search(r"Action:\s*(\w+)\s*(?:\[(.*?)\])?\s*$", output, re.MULTILINE | re.DOTALL)
            if action_match:
                tool_name = action_match.group(1).strip()
                args = (action_match.group(2) or "").strip()
                tool = next((t for t in self.tools if t.name == tool_name), None)
                if tool:
                    obs = tool.call(args)
                else:
                    obs = f"Error: unknown tool {tool_name}"
                steps[-1]["action"] = {"tool": tool_name, "args": args}
                steps[-1]["observation"] = truncate(obs, 500)
                if verbose:
                    print(f"[step {step}] {tool_name}({args}) -> {truncate(obs, 120)}")
                current_task = task + f"\nPrevious observation: {obs}"
            else:
                # no action: treat the whole output as final if short, else continue
                final = output.strip()
                if len(final) > 2000:
                    current_task = task + f"\n(Partial output: {truncate(output, 500)})"
                    continue
                break

        if not final:
            final = steps[-1]["thought"] if steps else "No output produced."

        self.history.append({"task": task, "steps": steps, "final": final})
        return {"task": task, "steps": steps, "final": final, "tool_calls": len([s for s in steps if s["action"]])}

    def _build_prompt(self, task: str, steps: List[Dict]) -> str:
        parts = [self.system_prompt, self._tool_prompt()]
        parts.append("\nUse this format:\nThought: ...\nAction: tool_name {\"arg\": \"value\"}\nObservation: ...\n... (repeat)\nFinal Answer: ...")
        if steps:
            parts.append("\nPrevious steps:")
            for s in steps:
                parts.append(f"Step {s['step']}: {s['thought']}")
                if s.get("action"):
                    parts.append(f"Action: {s['action']['tool']} {s['action']['args']}")
                if s.get("observation"):
                    parts.append(f"Observation: {s['observation']}")
        parts.append(f"\nTask: {task}\nAssistant:")
        return "\n".join(parts)


# ---------- built-in tools ----------
def _tool_read_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"Error: file not found: {path}"
    return p.read_text(encoding="utf-8", errors="ignore")[:4000]


def _tool_list_dir(path: str = ".") -> str:
    try:
        entries = list(Path(path).iterdir())
        return "\n".join(
            f"{'[D]' if e.is_dir() else '[F]'} {e.name}" for e in sorted(entries)[:100]
        )
    except Exception as e:
        return f"Error: {e}"


def _tool_calc(expr: str) -> str:
    try:
        import ast

        tree = ast.parse(expr, mode="eval")
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
                                     ast.Name, ast.Load, ast.Add, ast.Sub, ast.Mult,
                                     ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd)):
                return "Error: expression contains unsupported operations"
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"


def _tool_now() -> str:
    import datetime

    return datetime.datetime.now().isoformat()


def _tool_search(query: str) -> str:
    """Safe local search: grep-like over ~/.aweai/data."""
    from aweai.config import app_dir

    hits = []
    base = app_dir() / "data"
    if base.exists():
        for f in list(base.rglob("*.txt")) + list(base.rglob("*.md")) + list(base.rglob("*.json")):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            if query.lower() in content.lower():
                hits.append(f"{f}: {content[max(0, content.lower().find(query.lower()) - 60): content.lower().find(query.lower()) + 120]}")
    return "\n".join(hits[:10]) or "No local results found."


def default_tools() -> List[Tool]:
    return [
        Tool("read_file", "Read a text file from disk", _tool_read_file, '{"path": "..."}'),
        Tool("list_dir", "List directory contents", _tool_list_dir, '{"path": "."}'),
        Tool("calculate", "Evaluate a math expression safely", _tool_calc, '{"expr": "2+2"}'),
        Tool("now", "Get the current date and time", _tool_now, "{}"),
        Tool("search_local", "Search local documents for a query", _tool_search, '{"query": "..."}'),
    ]


class AgentEngine:
    """Simple factory for creating agents with default tools."""

    @staticmethod
    def create(name: str = "aweai-agent", with_default_tools: bool = True, llm: Optional[LLM] = None) -> Agent:
        agent = Agent(name=name, llm=llm)
        if with_default_tools:
            for t in default_tools():
                agent.tools.append(t)
        return agent
