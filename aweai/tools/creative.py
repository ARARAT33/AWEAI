"""AWEAI creative/innovation tools — ideas, design, content, naming.

Each tool has a unique purpose. These power the "never-seen-before"
features: idea generators, design sprints, content scaffolding and the
100k-menu combinatorics.
"""

from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

from aweai.tools.registry import tool

IDEA_SEEDS = [
    "AI copilot", "voice interface", "offline model", "smart dashboard",
    "auto-optimizer", "privacy-first", "edge device", "collaborative canvas",
    "self-healing system", "digital twin", "no-code builder", "live translator",
    "personal trainer", "knowledge graph", "smart scheduler", "anomaly radar",
]

@tool("idea_generate", "creative", "Generate creative product/feature ideas")
def idea_generate(n: int = 5, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    seeds = rng.sample(IDEA_SEEDS, min(n, len(IDEA_SEEDS)))
    ideas = [{"id": i + 1, "idea": f"{s} + {rng.choice(['AI', 'ML', 'automation', 'insight'])} powered tool"} for i, s in enumerate(seeds)]
    return {"ideas": ideas, "count": len(ideas)}


@tool("name_generate", "creative", "Generate product/project name suggestions")
def name_generate(base: str = "aweai", n: int = 8, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    suffixes = ["hub", "kit", "lab", "forge", "core", "flow", "mind", "stack", "mesh", "grid"]
    prefixes = ["smart", "meta", "ultra", "hyper", "nano", "omni", "auto", "deep"]
    names = set()
    while len(names) < n:
        choice = rng.random()
        if choice < 0.4:
            names.add(f"{base}-{rng.choice(suffixes)}")
        elif choice < 0.7:
            names.add(f"{rng.choice(prefixes)}-{base}")
        else:
            names.add(f"{base}{rng.randint(1, 99)}")
    return {"names": sorted(names)}


@tool("content_summary_template", "creative", "Build a content summary template from a topic")
def content_summary_template(topic: str) -> Dict[str, Any]:
    return {
        "topic": topic,
        "template": {
            "title": f"Everything about {topic}",
            "sections": ["What is it", "Why it matters", "How to get started", "Common pitfalls", "Resources"],
            "tone": "friendly-expert",
        },
    }


@tool("content_outline", "creative", "Generate a blog/article outline for a topic")
def content_outline(topic: str, sections: int = 5) -> Dict[str, Any]:
    return {
        "topic": topic,
        "outline": [{"section": i + 1, "title": f"{topic} — part {i + 1}", "points": [f"key point {j + 1}" for j in range(3)]} for i in range(sections)],
    }


@tool("design_palette", "creative", "Generate a color palette (hex values)")
def design_palette(seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)

    def hexc():
        return f"#{rng.randint(0, 255):02x}{rng.randint(0, 255):02x}{rng.randint(0, 255):02x}"

    return {"palette": [hexc() for _ in range(5)], "role": ["primary", "secondary", "accent", "background", "text"]}


@tool("design_font_pair", "creative", "Suggest a font pairing (system-safe choices)")
def design_font_pair() -> Dict[str, Any]:
    pairs = [
        {"heading": "Georgia", "body": "Verdana"},
        {"heading": "Arial Black", "body": "Arial"},
        {"heading": "Courier New", "body": "Consolas"},
        {"heading": "Trebuchet MS", "body": "Tahoma"},
        {"heading": "Palatino", "body": "Helvetica"},
    ]
    return {"pair": random.choice(pairs)}


@tool("design_ui_sketch", "creative", "Describe a UI layout sketch from a feature")
def design_ui_sketch(feature: str) -> Dict[str, Any]:
    return {
        "feature": feature,
        "sketch": {
            "layout": ["header", "sidebar", "content", "footer"],
            "components": [f"{feature} panel", "search bar", "action buttons", "status indicator", "settings drawer"],
            "responsive": ["mobile: stacked", "tablet: two-column", "desktop: three-column"],
        },
    }


@tool("menu_combine", "creative", "Generate menu combinations for the 100k menu structure")
def menu_combine(groups: str = '["tools", "models", "data"]', actions: str = '["list", "create", "edit"]', variants: int = 3) -> Dict[str, Any]:
    g = json.loads(groups) if isinstance(groups, str) else groups
    a = json.loads(actions) if isinstance(actions, str) else actions
    combos = []
    for group in g:
        for action in a:
            for v in range(variants):
                combos.append({"menu": f"{group} / {action}", "variant": v + 1, "path": f"/{group}/{action}/{v + 1}"})
    return {"combinations": combos, "count": len(combos)}


@tool("tagline_generate", "creative", "Generate taglines for a project name")
def tagline_generate(name: str = "AWEAI") -> Dict[str, Any]:
    return {
        "name": name,
        "taglines": [
            f"{name} — build anything, everywhere.",
            f"{name} — the AI model factory for everyone.",
            f"{name} — powerful tools, one command.",
            f"{name} — from idea to model in minutes.",
            f"{name} — your universal AI workbench.",
        ],
    }


@tool("roadmap_draft", "creative", "Draft a product roadmap from a vision")
def roadmap_draft(vision: str) -> Dict[str, Any]:
    return {
        "vision": vision,
        "roadmap": [
            {"phase": "v1", "focus": "core", "items": ["basics", "CLI", "docs"]},
            {"phase": "v2", "focus": "power", "items": ["models", "automation", "UI"]},
            {"phase": "v3", "focus": "scale", "items": ["tools", "compatibility", "cloud"]},
            {"phase": "v4", "focus": "ecosystem", "items": ["marketplace", "agents", "synergy"]},
        ],
    }


@tool("brainstorm_session", "creative", "Run a structured brainstorm session for a goal")
def brainstorm_session(goal: str, rounds: int = 3) -> Dict[str, Any]:
    rng = random.Random(0)
    return {
        "goal": goal,
        "session": [
            {"round": r + 1, "ideas": [f"{goal} — idea {i + 1} ({rng.choice(['fast', 'deep', 'wild', 'safe'])} approach)" for i in range(3)]}
            for r in range(rounds)
        ],
    }


@tool("story_generate", "creative", "Generate a short story scaffold from a theme")
def story_generate(theme: str = "adventure") -> Dict[str, Any]:
    return {
        "theme": theme,
        "story": {
            "hero": f"A curious {theme} explorer",
            "conflict": f"an unexpected {theme} challenge",
            "journey": [f"step {i + 1}" for i in range(4)],
            "ending": f"a {theme} resolution",
        },
    }


@tool("quiz_generate", "creative", "Generate a simple quiz from a topic")
def quiz_generate(topic: str, questions: int = 5) -> Dict[str, Any]:
    return {
        "topic": topic,
        "questions": [
            {"q": i + 1, "text": f"What is the most important thing about {topic} (question {i + 1})?", "options": ["A", "B", "C", "D"], "answer": "A"}
            for i in range(questions)
        ],
    }


__all__ = []
