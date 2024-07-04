import logging
import os
import tempfile
import time
from dataclasses import dataclass

import mlflow

import srl
from srl.base.context import RunContext
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.runner.callbacks.evaluate import Evaluate

logger = logging.getLogger(__name__)


@dataclass
class MLFlowCallback(RunCallback, Evaluate):
    auto_start: bool = True
    interval_episode: int = 1
    interval_eval: int = 10
    interval_checkpoint: int = 60 * 5
    enable_checkpoint: bool = True

    def on_start(self, context: RunContext, **kwargs) -> None:
        if self.auto_start:
            mlflow.set_experiment(context.env_config.name)
            mlflow.start_run()
        mlflow.set_tag("Version", srl.__version__)
        mlflow.set_tag("ENV", context.env_config.name)
        mlflow.set_tag("RL", context.rl_config.name)
        d = {"ENV_" + k: v for k, v in context.env_config.to_dict().items()}
        mlflow.log_params(d)
        d = {"RL_" + k: v for k, v in context.rl_config.to_dict().items()}
        mlflow.log_params(d)
        d = context.to_dict(include_env_config=False, include_rl_config=False)
        mlflow.log_params({"CONTEXT_" + k: v for k, v in d.items()})

        self.t0_episode = time.time()
        self.t0_eval = time.time()
        self.t0_checkpoint = time.time()

    def on_end(self, context: RunContext, **kwargs) -> None:
        if self.auto_start:
            mlflow.end_run()

    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        # error check
        self._log_episode(context, state)
        self._log_eval(context, state)
        if self.enable_checkpoint:
            self._log_checkpoint(context, state)

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs) -> bool:
        if time.time() - self.t0_episode > self.interval_episode:
            self.t0_episode = time.time()
            self._log_episode(context, state)

        if time.time() - self.t0_eval > self.interval_eval:
            self.t0_eval = time.time()
            self._log_eval(context, state)

        if self.enable_checkpoint:
            if time.time() - self.t0_checkpoint > self.interval_checkpoint:
                self.t0_checkpoint = time.time()
                self._log_checkpoint(context, state)

        return False

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        self._log_episode(context, state)
        self._log_eval(context, state)
        self._log_checkpoint(context, state)

    # ----------------
    def _log_episode(self, context: RunContext, state: RunStateActor):
        d = {}
        for i, r in enumerate(state.last_episode_rewards):
            d[f"reward{i}"] = r
        if state.trainer is not None:
            d.update(state.trainer.info.to_dict())
        d.update(state.worker.info.to_dict())
        mlflow.log_metrics(d, state.total_step)

    def _log_eval(self, context: RunContext, state: RunStateActor):
        eval_rewards = self.run_eval(context.env_config, context.rl_config, state.parameter)
        if eval_rewards is not None:
            d = {f"eval_reward{i}": r for i, r in enumerate(eval_rewards)}
            mlflow.log_metrics(d, state.total_step)

    def _log_checkpoint(self, context: RunContext, state: RunStateActor):
        with tempfile.TemporaryDirectory() as tmp_dir:
            name = context.rl_config.name
            path = os.path.join(tmp_dir, f"model_{name}_{state.total_step}.dat")
            state.parameter.save(path)
            mlflow.log_artifact(path)
