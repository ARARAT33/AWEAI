from aweai.checkpointing import CheckpointStore


def test_checkpoint_round_trip_and_latest():
    store = CheckpointStore()
    first = store.save("train", 1, {"loss": 1.0}, {"epoch": 1})
    second = store.save("train", 2, {"loss": 0.5}, {"epoch": 2})
    assert store.latest("train") == second
    assert store.verify(second, {"loss": 0.5}, {"epoch": 2})
    assert not store.verify(second, {"loss": 0.6}, {"epoch": 2})
    assert first.key != second.key


def test_unknown_workload_has_no_checkpoint():
    assert CheckpointStore().latest("missing") is None
