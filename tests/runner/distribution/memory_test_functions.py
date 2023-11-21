import time

import pytest

from srl.runner.distribution.interface import IMemoryServerParameters


def memory_connector_test(params: IMemoryServerParameters):
    receiver = params.create_memory_receiver()
    sender = params.create_memory_sender()

    assert receiver.ping()
    assert sender.ping()

    assert receiver.is_connected
    assert sender.is_connected

    receiver.memory_purge()
    n = sender.memory_size()
    if n != -1:
        assert n == 0

    sender.memory_add({"a": 1})
    time.sleep(0.1)
    n = sender.memory_size()
    if n != -1:
        assert n == 1

    d = receiver.memory_recv()
    print(d)
    assert d is not None
    assert d["a"] == 1
    assert receiver.memory_recv() is None

    sender.memory_add({"a": 1})
    time.sleep(0.1)
    n = sender.memory_size()
    if n != -1:
        assert n == 1
    receiver.memory_purge()
    n = sender.memory_size()
    if n != -1:
        assert n == 0


def memory_connector_error_test(params: IMemoryServerParameters):
    receiver = params.create_memory_receiver()
    sender = params.create_memory_sender()

    assert not receiver.ping()
    assert not sender.ping()
    assert not receiver.is_connected
    assert not sender.is_connected

    receiver.memory_purge()
    assert sender.memory_size() == -1

    with pytest.raises(Exception):
        sender.memory_add({"a": 1})

    with pytest.raises(Exception):
        receiver.memory_recv()
