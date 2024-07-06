import datetime
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

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
    run_name: Optional[str] = None
    start_run_kwargs: dict = field(default_factory=dict)
    interval_episode: int = 1
    interval_eval: int = 10
    interval_checkpoint: int = 60 * 20
    enable_checkpoint: bool = True
    enable_html: bool = True
    tags: dict = field(default_factory=dict)

    def on_start(self, context: RunContext, **kwargs) -> None:
        if self.auto_start:
            mlflow.set_experiment(context.env_config.name)
            run_name = self.run_name
            if run_name is None:
                run_name = f"{context.rl_config.name}"
            mlflow.start_run(run_name=run_name, **self.start_run_kwargs)
        self.run_id = mlflow.active_run().info.run_id
        mlflow.set_tag("Version", srl.__version__)
        mlflow.set_tag("Env", context.env_config.name)
        mlflow.set_tag("RL", context.rl_config.name)
        mlflow.set_tag("Framework", context.rl_config.get_framework())
        mlflow.set_tag("Flow", context.flow_mode)
        mlflow.set_tags(self.tags)

        d = {"env/" + k: v for k, v in context.env_config.to_dict().items()}
        mlflow.log_params(d)
        d = {"rl/" + k: v for k, v in context.rl_config.to_dict().items()}
        mlflow.log_params(d)
        d = context.to_dict(include_env_config=False, include_rl_config=False)
        mlflow.log_params({"context/" + k: v for k, v in d.items()})

        self._render_runner = None

    def on_end(self, context: RunContext, **kwargs) -> None:
        if self.auto_start:
            mlflow.end_run()

    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        # error check
        self._log_episode(context, state)
        self._log_eval(context, state)
        self._log_checkpoint(context, state)
        self._log_html(context, state)
        self.t0_episode = time.time()
        self.t0_eval = time.time()
        self.t0_checkpoint = time.time()

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs) -> bool:
        if time.time() - self.t0_episode > self.interval_episode:
            self._log_episode(context, state)
            self.t0_episode = time.time()  # last

        if time.time() - self.t0_eval > self.interval_eval:
            self._log_eval(context, state)
            self.t0_eval = time.time()  # last

        if self.enable_checkpoint:
            if time.time() - self.t0_checkpoint > self.interval_checkpoint:
                self._log_checkpoint(context, state)
                self._log_html(context, state)
                self.t0_checkpoint = time.time()  # last

        return False

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        self._log_episode(context, state)
        self._log_eval(context, state)
        self._log_checkpoint(context, state)
        self._log_html(context, state)

    # ----------------
    def _log_episode(self, context: RunContext, state: RunStateActor):
        d = {}
        for i, r in enumerate(state.last_episode_rewards):
            d[f"reward{i}"] = r
        if state.trainer is not None:
            d2 = {"trainer/" + k: v for k, v in state.trainer.info.to_dict().items()}
            d.update(d2)
        d2 = {"worker/" + k: v for k, v in state.worker.info.to_dict().items()}
        d.update(d2)
        mlflow.log_metrics(d, state.total_step, run_id=self.run_id)

    def _log_eval(self, context: RunContext, state: RunStateActor):
        eval_rewards = self.run_eval(context.env_config, context.rl_config, state.parameter)
        if eval_rewards is not None:
            d = {f"eval_reward{i}": r for i, r in enumerate(eval_rewards)}
            mlflow.log_metrics(d, state.total_step, run_id=self.run_id)

    def _log_checkpoint(self, context: RunContext, state: RunStateActor):
        if not self.enable_checkpoint:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            name = context.rl_config.name.replace(":", "_")
            path = os.path.join(tmp_dir, f"{name}_{state.total_step}_model.dat")
            state.parameter.save(path)
            mlflow.log_artifact(path, run_id=self.run_id)

    def _log_html(self, context: RunContext, state: RunStateActor):
        if not self.enable_html:
            return
        if self._render_runner is None:
            self._render_runner = srl.Runner(context.env_config, context.rl_config)
            self._render_runner.make_memory(is_load=False)
        render = self._render_runner.run_render(parameter=state.parameter)
        html = render.to_jshtml()
        name = context.rl_config.name.replace(":", "_")
        fn = f"{name}_{state.total_step}_{self._render_runner.state.last_episode_rewards}.html"
        mlflow.log_text(html, fn, run_id=self.run_id)

    # ---------------

    @staticmethod
    def get_metrics(experiment_name: str, run_name: str, metric_name: str):
        experiment_id = MLFlowCallback.get_experiment_id(experiment_name)
        if experiment_id is None:
            return None, None
        run_id = MLFlowCallback.get_run_id(experiment_id, run_name)
        if run_id is None:
            return None, None

        client = mlflow.tracking.MlflowClient()
        metric_history = client.get_metric_history(run_id, metric_name)
        values = [m.value for m in metric_history]
        steps = [m.step for m in metric_history]
        return steps, values

    @staticmethod
    def get_run_id(experiment_id: str, run_name: str):
        client = mlflow.tracking.MlflowClient()
        for run in client.search_runs([experiment_id]):
            if run.data.tags.get("mlflow.runName") == run_name:
                return run.info.run_id
        return None

    @staticmethod
    def get_experiment_id(experiment_name: str):
        client = mlflow.tracking.MlflowClient()
        for experiment in client.search_experiments():
            if experiment.name == experiment_name:
                return experiment.experiment_id
        return None
