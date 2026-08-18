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


def test_expanded_provider_chat_handling():
    from aweai.integrations.ai_tools import chat, PROVIDERS

    # Check that new providers are registered
    assert "deepseek" in PROVIDERS
    assert "qwen" in PROVIDERS
    assert "zhipu" in PROVIDERS
    assert "groq" in PROVIDERS
    assert "xai" in PROVIDERS
    assert "mistral" in PROVIDERS

    # Test execution without API key returns graceful error message
    res = chat("deepseek", "hello")
    assert res["ok"] is False
    assert "not configured" in res["error"] or "DeepSeek" in res["error"]

    res_qwen = chat("qwen", "hello")
    assert res_qwen["ok"] is False
    assert "not configured" in res_qwen["error"] or "Qwen" in res_qwen["error"]


def test_ecosystem_execute_routing_across_providers():
    res = run_tool("ecosystem_execute", capability="chat", message="hello", provider="deepseek")["result"]
    assert res["gateway"] == "AWEAI"
    # When missing credentials, returns ok=False with missing key details
    assert res["ok"] is False
    assert "DeepSeek" in res["error"]
