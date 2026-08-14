from aweai.tools import get_tool, run_tool


def test_ecosystem_tools_registered():
    assert get_tool("ecosystem_catalog") is not None
    assert get_tool("ecosystem_route") is not None
    assert get_tool("ecosystem_execute") is not None


def test_ecosystem_catalog_and_route():
    catalog = run_tool("ecosystem_catalog")["result"]
    assert catalog["provider_count"] >= 20
    routed = run_tool("ecosystem_route", capability="reasoning")["result"]
    assert routed["ok"] is True
    assert routed["gateway"] == "AWEAI"


def test_ecosystem_execute_dry_run():
    result = run_tool("ecosystem_execute", capability="chat", message="test", dry_run=True)["result"]
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["gateway"] == "AWEAI"
