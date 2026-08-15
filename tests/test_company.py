from aweai.company import CAPABILITY_CATALOG, CompanyToolRegistry


def test_company_registry_is_nonempty_and_unique():
    registry = CompanyToolRegistry()
    report = registry.validate()
    assert report["ok"] is True
    assert report["capabilities"] > 250
    assert report["categories"] >= 15
    assert report["duplicates"] == []


def test_aweai_only_policy_blocks_direct_execution():
    registry = CompanyToolRegistry()
    registry.policy.assert_execution_path("aweai")
    try:
        registry.policy.assert_execution_path("direct_vendor")
    except PermissionError:
        return
    raise AssertionError("direct vendor execution must be blocked by AWEAI-only policy")


def test_manifest_is_deterministic():
    a = CompanyToolRegistry()
    b = CompanyToolRegistry()
    assert a.fingerprint() == b.fingerprint()
    assert len(CAPABILITY_CATALOG) >= 15
