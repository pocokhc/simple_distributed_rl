import srl
from srl.algorithms import ql
from srl.runner.callback import Callback, MPCallback, TrainerCallback
from srl.runner.runner import Runner


class StubCallback(Callback):
    def __init__(self) -> None:
        self.episodes_begin = 0
        self.episodes_end = 0
        self.episode_begin = 0
        self.episode_end = 0
        self.step_action_before = 0
        self.step_begin = 0
        self.step_end = 0
        self.skip_step = 0

        self._step = 0

    def on_episodes_begin(self, runner: Runner) -> None:
        self.episodes_begin += 1

    def on_episodes_end(self, runner: Runner) -> None:
        self.episodes_end += 1

    def on_episode_begin(self, runner: Runner) -> None:
        self.episode_begin += 1

    def on_episode_end(self, runner: Runner) -> None:
        self.episode_end += 1

    def on_step_action_before(self, runner: Runner) -> None:
        self.step_action_before += 1

    def on_step_begin(self, runner: Runner) -> None:
        self.step_begin += 1

    def on_step_end(self, runner: Runner) -> None:
        self.step_end += 1
        self._step += 1

    def on_skip_step(self, runner: Runner) -> None:
        self.skip_step += 1

    def intermediate_stop(self, runner: Runner) -> bool:
        if self._step >= 10:
            return True
        return False


class StubTrainerCallback(TrainerCallback):
    def __init__(self) -> None:
        self.trainer_start = 0
        self.trainer_train = 0
        self.trainer_end = 0

        self._step = 0

    def on_trainer_start(self, runner: Runner) -> None:
        self.trainer_start += 1

    def on_trainer_train(self, runner: Runner) -> None:
        self.trainer_train += 1
        self._step += 1

    def on_trainer_end(self, runner: Runner) -> None:
        self.trainer_end += 1

    # 外部から途中停止用
    def intermediate_stop(self, runner: Runner) -> bool:
        if self._step >= 10:
            return True
        return False


class StubMPCallback(MPCallback):
    def __init__(self) -> None:
        self.init = 0
        self.start = 0
        self.polling = 0
        self.end = 0

    def on_init(self, runner: "Runner") -> None:
        self.init += 1

    def on_start(self, runner: "Runner") -> None:
        self.start += 1

    def on_polling(self, runner: "Runner") -> None:
        self.polling += 1

    def on_end(self, runner: "Runner") -> None:
        self.end += 1


# ----------------------------------------------


def test_callback():
    env_config = srl.EnvConfig("Grid", frameskip=4)
    runner = Runner(env_config, ql.Config())

    c = StubCallback()

    runner.train(max_steps=100, callbacks=[c])

    assert runner.state.total_step == 10
    assert c.episodes_begin == 1
    assert c.episodes_end == 1
    assert c.episode_begin >= c.episode_end
    assert c.episode_end == runner.state.episode_count
    assert c.step_begin == 10
    assert c.step_begin == c.step_action_before
    assert c.step_begin == c.step_end
    assert c.skip_step > 30


def test_trainer_callback():
    env_config = srl.EnvConfig("Grid")
    runner = Runner(env_config, ql.Config())

    c = StubTrainerCallback()

    runner.train_only(max_train_count=100, callbacks=[c])

    assert c.trainer_start == 1
    assert c.trainer_end == 1
    assert c.trainer_train == 10


def test_mp_callback():
    env_config = srl.EnvConfig("Grid")
    runner = Runner(env_config, ql.Config())

    c = StubMPCallback()

    runner.train_mp(timeout=2, callbacks=[c])

    assert c.init == 1
    assert c.start == 1
    assert c.end == 1
    assert c.polling > 0
