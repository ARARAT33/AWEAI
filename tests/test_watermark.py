# Copyright (c) 2026 ARARAT33. Based on AWEAI. All rights reserved.
"""Unit tests for AWEAI Watermark Engine, product integrations, and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from typer.testing import CliRunner

from aweai.cli import app
from aweai.export.edge import export_edge
from aweai.export.exporter import export_model
from aweai.management.manager import save_model, load_model
from aweai.models.registry import create_model
from aweai.product import (
    AWEAIArtifactLedger,
    AWEAISecureVault,
    ModelGovernanceAudit,
    WatermarkedArtifactRegistry,
)
from aweai.tools.security import (
    watermark_embed_tool,
    watermark_extract_tool,
    watermark_inspect_tool,
    watermark_verify_tool,
)
from aweai.watermark import (
    AWEAIWatermarkEngine,
    embed_watermark,
    extract_watermark,
    get_watermark_status,
    inspect_watermark,
    text_to_zwc,
    verify_watermark,
    zwc_to_text,
)


def test_zero_width_unicode_steganography():
    payload = "ARARAT33 - Top Secret Watermark Signature 2026"
    zwc = text_to_zwc(payload)
    assert len(zwc) > len(payload)
    decoded = zwc_to_text(zwc)
    assert decoded == payload


def test_text_watermark_embed_and_extract():
    engine = AWEAIWatermarkEngine(owner="ARARAT33")
    original_text = "This is a machine learning dataset sentence for AWEAI training."
    wm_text = engine.embed_text(original_text)

    assert "ARARAT33" in wm_text
    extracted = engine.extract_text(wm_text)
    assert extracted["has_watermark"] is True
    assert extracted["signature_valid"] is True
    assert "ARARAT33" in extracted["extracted_payload"]


def test_dict_watermark_tamper_detection():
    engine = AWEAIWatermarkEngine(owner="ARARAT33")
    data = {"model_name": "mlp_v1", "accuracy": 0.985, "layers": [64, 32]}
    wm_dict = engine.embed_dict(data)

    assert wm_dict["_watermark_owner"] == "ARARAT33"
    assert "_stego_hash" in wm_dict

    # Verification before tampering
    res = engine.verify_dict(wm_dict)
    assert res["valid"] is True
    assert res["tampered"] is False

    # Tamper with payload
    wm_dict["accuracy"] = 0.500
    tampered_res = engine.verify_dict(wm_dict)
    assert tampered_res["tampered"] is True
    assert tampered_res["valid"] is False


def test_array_watermark():
    engine = AWEAIWatermarkEngine(owner="ARARAT33")
    weights = np.random.randn(10, 10)
    wm_weights, meta = engine.embed_array(weights)

    res = engine.verify_array(wm_weights, meta)
    assert res["valid"] is True
    assert res["tampered"] is False

    # Modify array to trigger tamper detection
    wm_weights[0, 0] += 5.0
    tampered_res = engine.verify_array(wm_weights, meta)
    assert tampered_res["tampered"] is True


def test_file_watermark():
    engine = AWEAIWatermarkEngine(owner="ARARAT33")
    with TemporaryDirectory() as td:
        p = Path(td) / "test_data.json"
        p.write_text(json.dumps({"key": "value"}), encoding="utf-8")

        out = engine.embed_file(p)
        assert Path(out).exists()

        verif = engine.verify_file(out)
        assert verif["valid"] is True


def test_product_ledger_and_vault():
    ledger = AWEAIArtifactLedger()
    rec = ledger.register("dataset", {"x": [1, 2, 3]}, metadata={"project": "asi"})
    assert rec.metadata.get("_owner") == "ARARAT33"
    assert ledger.verify(rec.artifact_id) is True

    registry = WatermarkedArtifactRegistry()
    pub = registry.publish_artifact("mod1", "model", {"weights": [0.1, 0.2]})
    v_art = registry.verify_artifact(pub.artifact_id)
    assert v_art["verified"] is True
    assert v_art["owner"] == "ARARAT33"

    vault = AWEAISecureVault(owner="ARARAT33")
    sealed = vault.seal({"api_key": "secret123"})
    unsealed, ok = vault.unseal(sealed)
    assert ok is True
    assert unsealed["api_key"] == "secret123"

    audit = ModelGovernanceAudit()
    report = audit.audit_model(sealed)
    assert report["compliant"] is True


def test_model_zoo_manager_watermarking():
    m = create_model("mlp", input_dim=4, hidden=[4], output_dim=2)
    X = np.random.randn(10, 4)
    y = np.random.randint(0, 2, 10)
    m.fit(X, y, epochs=1)

    save_info = save_model(m, "test_wm_mlp")
    assert save_info["watermarked"] is True

    loaded_m, meta = load_model("test_wm_mlp")
    assert meta["_watermark_info"]["valid"] is True

    # Test exporters
    json_exp = export_model("test_wm_mlp", fmt="json")
    assert json_exp["watermarked"] is True

    edge_exp = export_edge("test_wm_mlp", fmt="tflite")
    assert edge_exp["watermarked"] is True


def test_cli_watermark_commands():
    runner = CliRunner()

    rv_status = runner.invoke(app, ["watermark", "status"])
    assert rv_status.exit_code == 0
    assert "ARARAT33" in rv_status.output

    rv_embed = runner.invoke(app, ["watermark", "embed", "Sample test string"])
    assert rv_embed.exit_code == 0
    assert "watermarked" in rv_embed.output

    rv_inspect = runner.invoke(app, ["watermark", "inspect"])
    assert rv_inspect.exit_code == 0


def test_security_tools_watermark():
    embedded = watermark_embed_tool("Security tool test text")
    assert "result" in embedded

    verif = watermark_verify_tool(embedded["result"])
    assert verif["has_watermark"] is True

    status = watermark_inspect_tool()
    assert status["owner"] == "ARARAT33"
