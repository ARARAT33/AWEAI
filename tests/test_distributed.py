from aweai.distributed import DistributedPlanner, Task, Worker


def test_distributed_plan_is_deterministic_and_balanced():
    planner = DistributedPlanner([Worker("w1", 2), Worker("w2", 1)])
    tasks = [Task("train", replicas=2, priority=2), Task("eval", replicas=1)]
    a = planner.plan(tasks)
    b = planner.plan(tasks)
    assert a == b
    assert len(a) == 3
    assert {worker for _, worker in a} == {"w1", "w2"}


def test_unhealthy_workers_are_excluded():
    planner = DistributedPlanner([Worker("bad", 10, healthy=False), Worker("good", 1)])
    assert planner.plan([Task("x")]) == [("x", "good")]
