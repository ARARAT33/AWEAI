"""Tests for the RAG engine and Actions runner (no ML deps)."""

import json
import tempfile
from pathlib import Path

from aweai.rag.engine import RAGEngine
from aweai.actions.runner import parse_action, ActionsRunner


def test_rag_index_and_search(tmp_path):
    engine = RAGEngine(data_dir=str(tmp_path / "rag"))
    engine.clear()
    doc = tmp_path / "doc.txt"
    doc.write_text(
        "AWEAI is a universal AI toolbox. It supports RAG and agents.\n"
        "Yerevan is the capital of Armenia.\n",
        encoding="utf-8",
    )
    added = engine.index_file(str(doc))
    assert added >= 1
    hits = engine.search("Yerevan capital")
    assert len(hits) >= 1
    assert "Yerevan" in hits[0]["text"]


def test_rag_ask_no_docs(tmp_path):
    engine = RAGEngine(data_dir=str(tmp_path / "rag2"))
    engine.clear()
    result = engine.ask("nothing indexed here")
    assert result["sources"] == []


def test_parse_action_train():
    parsed = parse_action("train a new model with this data /tmp/data.jsonl")
    assert parsed["intent"] == "train"
    assert parsed["params"].get("path") == "/tmp/data.jsonl"


def test_parse_action_rag_armenian():
    parsed = parse_action("ինդեքսավորել փաստաթղթերը docs/")
    assert parsed["intent"] == "rag"


def test_actions_runner_hardware():
    runner = ActionsRunner(verbose=False)
    result = runner.run("hardware")
    assert result["status"] == "ok"
    assert result["best_model"]


def test_actions_runner_train(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(json.dumps({"text": f"Sentence number {i} about AWEAI."}) for i in range(6)),
        encoding="utf-8",
    )
    runner = ActionsRunner(verbose=False)
    result = runner.run(f"train a new model named demo_actions with data {data}")
    assert result["status"] == "ok"
    assert result["model"] == "demo_actions"
