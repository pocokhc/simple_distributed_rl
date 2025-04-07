import pytest

from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig


def test_step_scheduler():
    pytest.importorskip("torch")
    import torch
    from torch.optim import SGD

    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    config = LRSchedulerConfig().set_step(decay_steps=2, decay_rate=0.5)
    scheduler = config.apply_torch_scheduler(optimizer)

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs == [0.1, 0.1, 0.05, 0.05, 0.025]


def test_exp_scheduler():
    pytest.importorskip("torch")
    import torch
    from torch.optim import SGD

    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    config = LRSchedulerConfig().set_exp(decay_rate=0.9)
    scheduler = config.apply_torch_scheduler(optimizer)

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(3):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert lrs == pytest.approx([0.1, 0.09, 0.081, 0.0729], rel=1e-3)


def test_cosine_scheduler():
    pytest.importorskip("torch")
    import torch
    from torch.optim import SGD

    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    config = LRSchedulerConfig().set_cosine(decay_steps=3, min_lr=0.01)
    scheduler = config.apply_torch_scheduler(optimizer)

    lrs = [optimizer.param_groups[0]["lr"]]
    for _ in range(3):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])

    assert min(lrs) >= 0.01  # min_lr を下回らない
    assert max(lrs) <= 0.1  # 初期 lr を上回らない


def test_piecewise_scheduler():
    pytest.importorskip("torch")
    import torch
    from torch.optim import SGD

    model = torch.nn.Linear(10, 1)
    optimizer = SGD(model.parameters(), lr=0.1)
    config = LRSchedulerConfig().set_piecewise([2, 4], [0.1, 0.05, 0.01])
    scheduler = config.apply_torch_scheduler(optimizer)

    lrs = [optimizer.param_groups[0]["lr"]]
    for epoch in range(5):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    print(lrs)

    assert lrs == [0.1, 0.1, 0.05, 0.05, 0.01, 0.01]
