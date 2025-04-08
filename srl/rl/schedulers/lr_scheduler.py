from dataclasses import dataclass, field
from typing import List, Literal


@dataclass
class LRSchedulerConfig:
    """
    学習率スケジューラの設定を保持するデータクラス。

    Attributes:
        schedule_type (Literal["", "step", "exp", "cosine"]): "step" (ステップ減衰), "exp" (指数減衰), "cosine" (余弦アニーリング)
        decay_steps (int): 学習率を減衰させるステップ数 (正の整数)。
        decay_rate (float): 減衰率 (指数減衰やステップ減衰で使用、正の値)。
        min_lr (float): 最小学習率 (余弦アニーリングで使用、0 以上 base_lr 未満)。
        warmup_steps (int): ウォームアップ期間のステップ数 (0 以上)。
    """

    schedule_type: Literal["", "step", "exp", "cosine", "piecewise"] = ""
    decay_steps: int = 100_000
    decay_rate: float = 0.1
    min_lr: float = 1e-6
    warmup_steps: int = 0
    piecewise_boundaries: List[int] = field(default_factory=lambda: [100000, 110000])
    piecewise_values: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.1])

    def _validate_params(self, lr: float) -> None:
        if not (lr > 0):
            raise ValueError(f"assert {lr} > 0")
        if not (self.decay_steps > 0):
            raise ValueError(f"assert {self.decay_steps} > 0")
        if not (self.decay_rate > 0):
            raise ValueError(f"assert {self.decay_rate} > 0")
        if not (0 <= self.min_lr < lr):
            raise ValueError(f"assert 0 <= {self.min_lr} < {lr}")
        if not (self.warmup_steps >= 0):
            raise ValueError(f"assert {self.warmup_steps} >= 0")
        if self.schedule_type == "piecewise":
            if not (self.piecewise_boundaries and self.piecewise_values):
                raise ValueError("piecewise_boundaries と piecewise_values は必須です")
            if len(self.piecewise_boundaries) + 1 != len(self.piecewise_values):
                raise ValueError("values は boundaries より 1 つ多い必要があります")

    def clear(self):
        self.schedule_type = ""
        return self

    def set_constant(self):
        self.schedule_type = ""
        return self

    def set_step(self, decay_steps: int = 100_000, decay_rate: float = 0.1):
        self.schedule_type = "step"
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        return self

    def set_exp(self, decay_steps: int = 100_000, decay_rate: float = 0.1):
        self.schedule_type = "exp"
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        return self

    def set_cosine(self, decay_steps: int = 100_000, min_lr: float = 1e-6):
        self.schedule_type = "cosine"
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        return self

    def set_piecewise(self, piecewise_boundaries: List[int], piecewise_values: List[float]):
        self.schedule_type = "piecewise"
        self.piecewise_boundaries = piecewise_boundaries
        self.piecewise_values = piecewise_values
        return self

    # ---------------------------------------------------------------------------

    def apply_tf_scheduler(self, lr: float) -> "keras.optimizers.schedules.LearningRateSchedule":
        import tensorflow as tf
        from tensorflow import keras

        self._validate_params(lr)

        if self.schedule_type == "step":
            return keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                staircase=True,
            )
        elif self.schedule_type == "exp":
            return keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=lr,
                decay_steps=self.decay_steps,
                decay_rate=self.decay_rate,
                staircase=False,
            )
        elif self.schedule_type == "cosine":
            return keras.optimizers.schedules.CosineDecay(
                initial_learning_rate=lr,
                decay_steps=self.decay_steps,
                alpha=self.min_lr / lr,
            )
        elif self.schedule_type == "piecewise":
            return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                self.piecewise_boundaries,
                self.piecewise_values,
            )
        else:
            return lr

    def apply_torch_scheduler(self, optimizer: "torch.optim.Optimizer") -> "torch.optim.Optimizer":
        import torch.optim.lr_scheduler as torch_lr

        lr = optimizer.param_groups[0]["lr"]
        self._validate_params(lr)

        if self.schedule_type == "step":
            return torch_lr.StepLR(optimizer, step_size=self.decay_steps, gamma=self.decay_rate)
        elif self.schedule_type == "exp":
            return torch_lr.ExponentialLR(optimizer, gamma=self.decay_rate)
        elif self.schedule_type == "cosine":
            return torch_lr.CosineAnnealingLR(optimizer, T_max=self.decay_steps, eta_min=self.min_lr)
        elif self.schedule_type == "piecewise":

            def lr_lambda(epoch: int) -> float:
                for i, boundary in enumerate(self.piecewise_boundaries):
                    if epoch < boundary:
                        return self.piecewise_values[i] / lr
                return self.piecewise_values[-1] / lr

            return torch_lr.LambdaLR(optimizer, lr_lambda=lr_lambda)
        else:
            return optimizer
