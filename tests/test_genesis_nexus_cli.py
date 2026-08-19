# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Unit tests for Genesis & Nexus tools and CLI commands."""

from typer.testing import CliRunner
from aweai.cli import app
from aweai.tools.genesis_nexus import (
    genesis_chaos_encode,
    genesis_hypervector_query,
    genesis_neuro_graph,
    genesis_quantum_state,
    nexus_pipeline,
    nexus_sign_payload,
    nexus_svd_compress,
)

runner = CliRunner()


def test_genesis_nexus_tools():
    q = genesis_quantum_state(value=50.0, entropy=0.1)
    assert "collapsed_value" in q

    g = genesis_neuro_graph(nodes=5, steps=5)
    assert "adaptive_path" in g

    c = genesis_chaos_encode(message="test", system="lorenz")
    assert "points_count" in c

    v = genesis_hypervector_query(concepts="a,b", query_concept="a")
    assert v["best_match"] == "a"

    p = nexus_pipeline(items_count=5)
    assert p["results_count"] == 5

    s = nexus_svd_compress(rows=10, cols=10, rank=2)
    assert "svd_result" in s

    sig = nexus_sign_payload(payload_str='{"a":1}')
    assert sig["verified"] is True


def test_genesis_cli_commands():
    res1 = runner.invoke(app, ["genesis", "quantum", "--value", "100"])
    assert res1.exit_code == 0
    assert "collapsed_value" in res1.stdout

    res2 = runner.invoke(app, ["genesis", "graph", "--nodes", "5"])
    assert res2.exit_code == 0
    assert "adaptive_path" in res2.stdout

    res3 = runner.invoke(app, ["genesis", "chaos", "--message", "hello"])
    assert res3.exit_code == 0
    assert "points_count" in res3.stdout

    res4 = runner.invoke(app, ["genesis", "hypervector", "--concepts", "a,b", "--query", "a"])
    assert res4.exit_code == 0
    assert "best_match" in res4.stdout


def test_nexus_cli_commands():
    res1 = runner.invoke(app, ["nexus", "pipeline", "--items", "5"])
    assert res1.exit_code == 0
    assert "results_count" in res1.stdout

    res2 = runner.invoke(app, ["nexus", "compress", "--rows", "10", "--cols", "10", "--rank", "3"])
    assert res2.exit_code == 0
    assert "svd_result" in res2.stdout

    res3 = runner.invoke(app, ["nexus", "sign", "--payload", '{"foo":"bar"}'])
    assert res3.exit_code == 0
    assert "verified" in res3.stdout
