import math

import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.randint(0, 4, size=(data_num, 1))
    y = np.identity(5, dtype=np.float32)[x.reshape(-1)]
    return x.astype(np.float32), y.astype(np.float32)


@pytest.mark.parametrize("unimix", [0, 0.1])
def test_loss(unimix):
    pytest.importorskip("torch")

    import torch

    from srl.rl.torch_.distributions.categorical_dist_block import CategoricalDistBlock

    block = CategoricalDistBlock(1, 5, (32, 32), unimix=unimix)

    optimizer = torch.optim.Adam(block.parameters(), lr=0.05)
    for i in range(200):
        x_train, y_train = _create_dataset(64)
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)

        optimizer.zero_grad()
        loss = block.compute_train_loss(x_train, y_train)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"{i}: {loss.item()}")

    x_true, y_true = _create_dataset(10)
    x_true = torch.FloatTensor(x_true)
    dist = block(x_true)
    print(x_true.reshape(-1))
    print(y_true)
    print(dist.sample())

    x_true, y_true = _create_dataset(1000)
    x_true = torch.FloatTensor(x_true)
    dist = block(x_true)
    y_pred = dist.sample(onehot=True).detach().numpy()
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    assert rmse < 0.1 + unimix


@pytest.mark.parametrize("unimix", [0, 0.1])
def test_loss_grad(unimix):
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_.distributions.categorical_dist_block import CategoricalDistBlock

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.h1 = torch.nn.ModuleList(
                [
                    torch.nn.Linear(1, 128),
                    torch.nn.ReLU(inplace=True),
                ]
            )
            self.block = CategoricalDistBlock(128, 5, unimix=unimix)
            self.h2 = torch.nn.ModuleList(
                [
                    torch.nn.Linear(5, 128),
                    torch.nn.ReLU(inplace=True),
                    torch.nn.Linear(128, 1),
                ]
            )

        def forward(self, x):
            for h in self.h1:
                x = h(x)
            dist = self.block(x)
            x = dist.rsample()
            for h in self.h2:
                x = h(x)
            return x

    model = _Model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    mse_loss = torch.nn.MSELoss()
    for i in range(1000):
        x_train, _ = _create_dataset(128)
        x_train = torch.FloatTensor(x_train)

        optimizer.zero_grad()
        y = model(x_train)
        loss = mse_loss(x_train, y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"{i}: {loss.item()}")

    x_true, _ = _create_dataset(10)
    x_true = torch.FloatTensor(x_true)
    x_pred = model(x_true)
    print(x_true.reshape(-1))
    print(x_pred)

    x_true, _ = _create_dataset(1000)
    y_pred = model(torch.FloatTensor(x_true)).detach().numpy()
    rmse = np.sqrt(np.mean((x_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    assert rmse < 1 + unimix


def test_inf():
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_.distributions.categorical_dist_block import CategoricalDist

    m = CategoricalDist(
        torch.FloatTensor(
            [
                [53.855743, 5.476642, -35.126976, -31.001219],
                [37.039024, 3.6855803, -23.241371, -22.968405],
                [42.007465, 4.247389, -26.70271, -25.079786],
            ]
        )
    )
    log_prob = (
        m.log_prob(
            torch.FloatTensor(
                [
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                ]
            )
        )
        .detach()
        .numpy()
    )
    print(m.probs())
    print(log_prob)
    assert not np.isinf(log_prob).any()


if __name__ == "__main__":
    # test_loss(0)
    # test_loss_grad(0)
    test_inf()
