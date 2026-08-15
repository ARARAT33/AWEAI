from aweai.serving import Autoscaler, Replica, ServingPool, serving_health

def test_serving_selects_healthy_replica():
    pool = ServingPool([Replica("b", 10, False), Replica("a", 10, True)])
    assert pool.choose().name == "a"

def test_serving_health_and_metrics():
    pool = ServingPool([Replica("a", 10)])
    pool.observe(50)
    pool.observe(1500, error=True)
    assert pool.requests == 2
    assert pool.error_rate == .5
    assert not serving_health(pool)

def test_autoscaler_is_bounded():
    scaler = Autoscaler(min_replicas=2, max_replicas=8)
    assert scaler.desired(2, 0, 10) == 2
    assert scaler.desired(2, 1000, 10) == 8
