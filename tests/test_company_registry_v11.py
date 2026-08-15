from aweai.company import CompanyToolRegistry


def test_registry_exact_lookup_and_search():
    registry = CompanyToolRegistry()
    capability = registry.get("engineering.code_generation")
    assert capability is not None
    assert capability.category == "engineering"
    assert registry.get("ENGINEERING.CODE_GENERATION") == capability
    assert any(row.name == "security.secret_scan" for row in registry.search("secret_scan"))


def test_registry_category_stats_match_manifest():
    registry = CompanyToolRegistry()
    stats = registry.category_stats()
    manifest = registry.manifest()
    assert sum(stats.values()) == manifest["capabilities"]
    assert stats == manifest["category_stats"]


def test_execution_plan_is_dry_run_and_aweai_controlled():
    registry = CompanyToolRegistry()
    plan = registry.execution_plan("inference.routing")
    assert plan["ok"] is True
    assert plan["control_plane"] == "AWEAI"
    assert plan["dry_run"] is True
    assert plan["execution_path"] == "aweai"

    adapter_plan = registry.execution_plan("model.finetune", adapter="approved-provider")
    assert adapter_plan["ok"] is True
    assert adapter_plan["execution_path"] == "aweai.adapter"
    assert adapter_plan["requires_external_adapter"] is True


def test_unknown_capability_returns_structured_error():
    result = CompanyToolRegistry().execution_plan("unknown.missing")
    assert result["ok"] is False
    assert "unknown capability" in result["error"]
