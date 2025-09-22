import math

import numpy as np
import pytest


def _create_dataset(data_num):
    x = np.random.uniform(0, 1, size=(data_num, 1))
    noise = np.random.normal(loc=0, scale=0.05, size=(data_num, 1))
    y = 5 + 0.5 * np.sin(2 * np.pi * x) + x + noise
    return np.asarray(x, np.float32), np.asarray(y, np.float32)


@pytest.mark.parametrize("fixed_scale", [-1, 0.1])
def test_train(fixed_scale, is_plot=False):
    pytest.importorskip("torch")

    import torch

    from srl.rl.torch_.distributions.normal_dist_block import NormalDistBlock

    block = NormalDistBlock(
        1,
        1,
        (64, 64, 64),
        (),
        (),
        fixed_scale=fixed_scale,
    )

    optimizer = torch.optim.Adam(block.parameters(), lr=0.001)
    for i in range(1000):
        x_train, y_train = _create_dataset(64)
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)

        optimizer.zero_grad()
        loss = block.compute_train_loss(x_train, y_train)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"{i}: {loss.item()}")

    x_true, y_true = _create_dataset(1000)
    x_true = torch.FloatTensor(x_true)
    dist = block(x_true)
    y_pred = dist.sample().detach().numpy()

    if is_plot:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(x_true, y_true, "ro", alpha=0.2, label="true")
        plt.plot(x_true, y_pred, "bo", alpha=0.2, label="pred")
        plt.legend()
        plt.show()
        plt.close()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    if fixed_scale < 0:
        assert rmse < 0.1
    else:
        assert rmse < 0.4


def _create_dataset2(data_num):
    x = np.random.randint(0, 5, size=(data_num, 1))
    return x.astype(np.float32), x.astype(np.float32)


@pytest.mark.parametrize("fixed_scale", [-1, 0.1])
def test_autoencoder(fixed_scale):
    pytest.importorskip("torch")
    import torch

    from srl.rl.torch_.distributions.normal_dist_block import NormalDistBlock

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.h1 = torch.nn.ModuleList(
                [
                    torch.nn.Linear(1, 32),
                    torch.nn.ReLU(),
                ]
            )
            self.block = NormalDistBlock(32, 32, fixed_scale=fixed_scale)
            self.h2 = torch.nn.ModuleList(
                [
                    torch.nn.Linear(32, 1),
                    torch.nn.ReLU(),
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
        x_train, y_train = _create_dataset2(128)
        x_train = torch.FloatTensor(x_train)
        y_train = torch.FloatTensor(y_train)

        optimizer.zero_grad()
        y = model(x_train)
        loss = mse_loss(y_train, y)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"{i}: {loss.item()}")

    x_true, y_true = _create_dataset2(1000)
    x_true = torch.FloatTensor(x_true)
    y_pred = model(x_true)
    y_pred = y_pred.detach().numpy()

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    print(f"rmse: {rmse}")
    if fixed_scale < 0:
        assert float(rmse) < 0.2
    else:
        assert float(rmse) < 0.4


def test_dist():
    pytest.importorskip("torch")

    import torch

    from srl.rl.torch_.distributions.normal_dist_block import NormalDist

    loc = torch.FloatTensor(np.array([[-1], [1]], dtype=np.float32))
    scale = torch.FloatTensor(np.array([[0.1], [5]], dtype=np.float32))

    dist_t = torch.distributions.Normal(loc, scale)
    dist = NormalDist(loc, torch.log(scale))

    n1 = dist_t.mean.numpy()
    n2 = dist.mean().detach().numpy()
    print(n1)
    print(n2)
    assert np.allclose(n1, n2)

    n1 = dist_t.stddev.numpy()
    n2 = dist.stddev().detach().numpy()
    print(n1)
    print(n2)
    assert np.allclose(n1, n2)

    n1 = dist_t.entropy().numpy()
    n2 = dist.entropy().detach().numpy()
    print(n1)
    print(n2)
    assert np.allclose(n1, n2)

    x = torch.FloatTensor([[-1.1, 1.1]])
    n1 = dist_t.log_prob(x).detach().numpy()
    n2 = dist.log_prob(x).detach().numpy()
    print(n1)
    print(n2)
    assert np.allclose(n1, n2)


def _normal(x, mean, stddev, epsilon=1e-10):
    x = np.array(x, dtype=np.float32)
    mean = np.array(mean, dtype=np.float32)
    stddev = np.array(stddev, dtype=np.float32)
    stddev = np.clip(stddev, epsilon, None)
    y = (1 / (np.sqrt(2 * np.pi * stddev * stddev))) * np.exp(-((x - mean) ** 2) / (2 * stddev * stddev))
    return np.array(y, dtype=np.float32)


@pytest.mark.parametrize(
    "action, mean, stddev",
    [
        (0, 0, 1),
        (5, 9, 1),
    ],
)
def test_compute_normal_logprob_sgp(action, mean, stddev):
    pytest.importorskip("torch")

    import torch

    from srl.rl.torch_.distributions.normal_dist_block import compute_normal_logprob_sgp

    np_mu = _normal(action, mean, stddev)
    np_logmu = np.log(np_mu)
    np_logpi = np_logmu - np.log(1 - np.tanh(action) ** 2)
    np_pi = np.exp(np_logpi)

    logpi = compute_normal_logprob_sgp(
        torch.FloatTensor([[action]]),
        torch.FloatTensor([[mean]]),
        torch.FloatTensor(np.log([[stddev]])),
    )
    pi = torch.exp(logpi)  # logpiが-130ぐらいだと-infになる
    pi = pi.numpy()[0][0]
    logpi = logpi.numpy()[0][0]

    print(f"np_mu={np_mu}, np_logmu={np_logmu}, np_logpi={np_logpi}, np_pi={np_pi}, logpi={logpi}, pi={pi}")
    assert math.isclose(np_pi, pi, rel_tol=0.1)
    assert math.isclose(np_logpi, logpi, rel_tol=0.01)


if __name__ == "__main__":
    # test_train(-1, is_plot=True)
    test_autoencoder(-1)
    # test_dist()
