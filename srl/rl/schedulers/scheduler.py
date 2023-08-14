from srl.rl.schedulers.schedulers.base import BaseScheduler


class SchedulerConfig:
    """
    数値のハイパーパラメータに対してスケジュールによる変更を提供します。
    主に学習率に使われることを想定していますが、それ以外でも使用できます。

    ・配列を使わない場合
    setから始まる関数を使ってください。
    例えば線形で減少する値を使用する場合は以下です。

    >>> sch = SchedulerConfig()
    >>> sch.set_linear(100, 1, 0)  # 100stepで1から0に減少
    >>> sch.plot()  # 画像で確認

    ・配列を使う場合
    複数のパターンを使うことも可能です。
    clear後、addから始まる関数を使ってください。

    >>> sch = SchedulerConfig()
    >>> sch.clear()
    >>> sch.add_linear(100, 0.5, 1)  # 0～100stepで0.5から1に増加
    >>> sch.add_linear(100, 1, 0.1)  # 100～200stepで1から0.1に減少
    >>> sch.plot()  # 画像で確認

    """

    def __init__(self, rate: float = 0):
        self._schedulers: list = []
        self.set_constant(rate)

    def clear(self):
        """配列を空にします"""
        self._schedulers = []

    # --- constant
    def set_rate(self, rate: float):  # other name
        """固定値を使用 これはset_constantと同じ動作です。

        y = rate

        Args:
            rate (float): val
        """
        self.clear()
        self.add_constant(0, rate)

    def set_constant(self, rate: float):
        """固定値を使用

        y = rate

        Args:
            rate (float): val
        """
        self.clear()
        self.add_constant(0, rate)

    def add_constant(self, phase_steps: int, rate: float):
        """固定値を追加

        y = rate

        Args:
            phase_steps (int): 継続するstep数
            rate (float): val
        """
        self._schedulers.append((phase_steps, "constant", dict(rate=rate)))

    # --- linear
    def set_linear(self, phase_steps: int, start_rate: float, end_rate: float):
        """線形に変化

        y = start_rate + (end_rate - start_rate) * step

        Args:
            phase_steps (int): 継続するstep数
            start_rate (float): 開始時のrate
            end_rate (float): 終了時のrate
        """
        self.clear()
        self.add_linear(phase_steps, start_rate, end_rate)

    def add_linear(self, phase_steps: int, start_rate: float, end_rate: float):
        """線形に変化を追加

        y = start_rate + (end_rate - start_rate) * step

        Args:
            phase_steps (int): 継続するstep数
            start_rate (float): 開始時のrate
            end_rate (float): 終了時のrate
        """
        self._schedulers.append(
            (
                phase_steps,
                "linear",
                dict(
                    start_rate=start_rate,
                    end_rate=end_rate,
                    phase_steps=phase_steps,
                ),
            )
        )

    # --- cos
    def set_cosine(self, phase_steps: int, start_rate: float):
        """cosに従って0へ変動

        Args:
            phase_steps (int): 継続するstep数
            start_rate (float): 開始時のrate
        """
        self.clear()
        self.add_cosine(phase_steps, start_rate)

    def add_cosine(self, phase_steps: int, start_rate: float):
        """cosに従って0へ変動

        Args:
            phase_steps (int): 継続するstep数
            start_rate (float): 開始時のrate
        """
        self._schedulers.append(
            (
                phase_steps,
                "cosine",
                dict(
                    start_rate=start_rate,
                    phase_steps=phase_steps,
                ),
            )
        )

    # --- cos restart
    def set_cosine_with_hard_restarts(self, phase_steps: int, start_rate: float, num_cycles: int):
        """cosに従って0へ変動、ただしnum_cycles数繰り返す

        Args:
            phase_steps (int): 継続するstep数
            start_rate (float): 開始時のrate
            num_cycles (int): 繰り返す回数
        """
        self.clear()
        self.add_cosine_with_hard_restarts(phase_steps, start_rate, num_cycles)

    def add_cosine_with_hard_restarts(self, phase_steps: int, start_rate: float, num_cycles: int):
        """cosに従って0へ変動、ただしnum_cycles数繰り返す

        Args:
            phase_steps (int): 継続するstep数
            start_rate (float): 開始時のrate
            num_cycles (int): 繰り返す回数
        """
        self._schedulers.append(
            (
                phase_steps,
                "cosine_with_hard_restarts",
                dict(
                    start_rate=start_rate,
                    phase_steps=phase_steps,
                    num_cycles=num_cycles,
                ),
            )
        )

    # --- polynomial
    def set_polynomial(self, phase_steps: int, start_rate: float, power: float = 2):
        """多項式に従って0へ減少

        y = start_rate * (1 - step/phase_steps)^power

        Args:
            phase_steps (int): 継続するstep数
            start_rate (float): 開始時のrate
            power (float, optional): 強さ、1で線形と同じ. Defaults to 2.
        """
        self.clear()
        self.add_polynomial(phase_steps, start_rate, power)

    def add_polynomial(self, phase_steps: int, start_rate: float, power: float = 2):
        """多項式に従って0へ減少

        y = start_rate * (1 - step/phase_steps)^power

        Args:
            phase_steps (int): 継続するstep数
            start_rate (float): 開始時のrate
            power (float, optional): 強さ、1で線形と同じ. Defaults to 2.
        """
        self._schedulers.append(
            (
                phase_steps,
                "polynomial",
                dict(
                    start_rate=start_rate,
                    phase_steps=phase_steps,
                    power=power,
                ),
            )
        )

    # --- custom
    def set_custom(self, phase_steps: int, entry_point: str, kwargs: dict):
        self.clear()
        self.add_custom(phase_steps, entry_point, kwargs)

    def add_custom(self, phase_steps: int, entry_point: str, kwargs: dict):
        self._schedulers.append((phase_steps, "custom", dict(entry_point=entry_point, kwargs=kwargs)))

    # ---------------------------

    def create_schedulers(self) -> BaseScheduler:
        assert len(self._schedulers) > 0, "Set at least one Scheduler."

        if len(self._schedulers) == 1:
            c = self._schedulers[0]
            return self.create_scheduler(c[1], c[2])
        else:
            return ListScheduler(self, self._schedulers)

    def create_scheduler(self, name: str, kwargs) -> BaseScheduler:
        if name == "constant":
            from srl.rl.schedulers.schedulers.constant import Constant

            return Constant(**kwargs)
        elif name == "linear":
            from srl.rl.schedulers.schedulers.linear import Linear

            return Linear(**kwargs)

        elif name == "cosine":
            from srl.rl.schedulers.schedulers.cosine import Cosine

            return Cosine(**kwargs)
        elif name == "cosine_with_hard_restarts":
            from srl.rl.schedulers.schedulers.cosine import CosineWithHardRestarts

            return CosineWithHardRestarts(**kwargs)
        elif name == "polynomial":
            from srl.rl.schedulers.schedulers.polynomial import Polynomial

            return Polynomial(**kwargs)
        elif name == "custom":
            from srl.utils.common import load_module

            c = load_module(kwargs["entry_point"])(**kwargs["kwargs"])
            return c
        else:
            raise ValueError(name)

    def plot(
        self,
        _no_plot: bool = False,  # for test
    ):
        from srl.utils.common import is_package_installed

        assert is_package_installed(
            "matplotlib"
        ), "To use plot you need to install the 'matplotlib'. (pip install matplotlib)"

        import matplotlib.pyplot as plt

        sch = self.create_schedulers()

        max_steps = 0
        for phase_steps, name, kwargs in self._schedulers:
            max_steps += phase_steps
        if max_steps < 100:
            max_steps = 100
        line_step = max_steps
        max_steps = int(max_steps * 1.1)

        y_arr = [sch.get_rate(i) for i in range(max_steps)]
        plt.plot(y_arr)
        plt.vlines(x=line_step, ymin=min(y_arr), ymax=max(y_arr), color="r", linestyles="dotted")
        plt.ylabel("rate")
        plt.xlabel("step")
        plt.grid()
        plt.tight_layout()
        if not _no_plot:
            plt.show()


class ListScheduler(BaseScheduler):
    def __init__(self, config: SchedulerConfig, schedulers):
        self.schedulers_idx = {}
        step = 0
        for phase_steps, name, kwargs in schedulers:
            sch = config.create_scheduler(name, kwargs)
            self.schedulers_idx[step] = sch
            step += phase_steps

        self.current_step = 0
        self.current_sch: BaseScheduler = self.schedulers_idx[0]

    def get_rate(self, step: int) -> float:
        rate = self.current_sch.get_rate(step - self.current_step)

        if step in self.schedulers_idx:
            self.current_step = step
            self.current_sch = self.schedulers_idx[step]

        return rate
