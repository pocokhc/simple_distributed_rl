import logging
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional, cast

import mlflow
import mlflow.entities

import srl
from srl.base.context import RunContext
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.run.callback import RunCallback, TrainCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callbacks.evaluate import Evaluate

logger = logging.getLogger(__name__)


@dataclass
class MLFlowCallback(RunCallback, TrainCallback, Evaluate):
    experiment_name: str = ""
    run_name: str = ""
    tags: dict = field(default_factory=dict)

    interval_episode: float = 1
    interval_eval: float = -1  # -1 is auto
    interval_checkpoint: float = 60 * 30
    enable_checkpoint: bool = True
    enable_html: bool = True

    def on_start(self, context: RunContext, **kwargs) -> None:
        self._auto_run = False
        framework = context.rl_config.get_framework()
        if mlflow.active_run() is None:
            if self.experiment_name == "":
                self.experiment_name = context.env_config.name
            mlflow.set_experiment(self.experiment_name)
            if self.run_name == "":
                self.run_name = context.rl_config.name
            if framework != "":
                self.run_name += f":{framework}"
            self.run_name += f"({context.flow_mode})"
            mlflow.start_run(run_name=self.run_name)
            self._auto_run = True
        active_run = mlflow.active_run()
        assert active_run is not None
        self.run_id = active_run.info.run_id
        logger.info(f"mlflow run_id: {self.run_id}")
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
        if self._auto_run:
            mlflow.end_run()

    # ---------------------------------------------------------------

    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        if context.actor_id != 0:
            return
        # --- error check
        self._log_episode(context, state)
        self._log_eval(context, state)
        if not context.distributed:
            self._log_checkpoint(context, state)
            self._log_html(context, state)
        self.t0_episode = time.time()
        self.t0_eval = time.time()
        self.t0_checkpoint = time.time()

    def on_step_end(self, context: RunContext, state: RunStateActor, **kwargs) -> bool:
        if context.actor_id != 0:
            return False
        _time = time.time()
        if _time - self.t0_episode > self.interval_episode:
            self._log_episode(context, state)
            self.t0_episode = time.time()  # last

        if _time - self.t0_eval > self.interval_eval:
            self._log_eval(context, state)
            self.t0_eval = time.time()  # last

        if _time - self.t0_checkpoint > self.interval_checkpoint:
            self._log_checkpoint(context, state)
            self._log_html(context, state)
            self.t0_checkpoint = time.time()  # last

        return False

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        if context.actor_id != 0:
            return
        self._log_episode(context, state)
        if not context.distributed:
            self._log_eval(context, state)
            self._log_checkpoint(context, state)
            self._log_html(context, state)

    # ---------------------------------------------------------------

    def on_trainer_start(self, context: RunContext, state: RunStateTrainer, **kwargs) -> None:
        self._log_episode(context, state)
        self._log_eval(context, state)
        self._log_checkpoint(context, state)
        self._log_html(context, state)
        self.t0_episode = time.time()
        self.t0_eval = time.time()
        self.t0_checkpoint = time.time()

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer, **kwargs) -> None:
        self._log_episode(context, state)
        self._log_eval(context, state)
        self._log_checkpoint(context, state)
        self._log_html(context, state)

    def on_train_after(self, context: RunContext, state: RunStateTrainer, **kwargs) -> bool:
        _time = time.time()
        if _time - self.t0_episode > self.interval_episode:
            self._log_episode(context, state)
            self.t0_episode = time.time()  # last

        if not context.distributed:
            if _time - self.t0_eval > self.interval_eval:
                self._log_eval(context, state)
                self.t0_eval = time.time()  # last

            if _time - self.t0_checkpoint > self.interval_checkpoint:
                self._log_checkpoint(context, state)
                self._log_html(context, state)
                self.t0_checkpoint = time.time()  # last

        return False

    # ---------------------------------------------------------------
    def _get_step(self, context: RunContext, state):
        if context.distributed:
            return state.train_count
        else:
            if state.trainer is None:
                return state.total_step
            else:
                return state.train_count

    def _log_episode(self, context: RunContext, state):
        if isinstance(state, RunStateActor):
            d = {}
            for i, r in enumerate(state.last_episode_rewards):
                d[f"reward{i}"] = r
            if state.trainer is not None:
                d2 = {"trainer/" + k: v for k, v in state.trainer.info.to_dict().items()}
                d.update(d2)
            d2 = {"worker/" + k: v for k, v in state.worker.info.to_dict().items()}
            d.update(d2)
            mlflow.log_metrics(d, self._get_step(context, state), run_id=self.run_id)
        elif isinstance(state, RunStateTrainer):
            d = {"trainer/" + k: v for k, v in state.trainer.info.to_dict().items()}
            mlflow.log_metrics(d, state.trainer.train_count, run_id=self.run_id)

    def _log_eval(self, context: RunContext, state):
        if not self.enable_eval:
            return

        t0 = time.time()
        eval_rewards = self.run_eval(context, state)
        if eval_rewards is not None:
            d = {f"eval_reward{i}": r for i, r in enumerate(eval_rewards)}
            mlflow.log_metrics(d, self._get_step(context, state), run_id=self.run_id)

        # check interval
        eval_time = time.time() - t0
        interval = eval_time * 10
        if interval < 1:
            interval = 1
        interval = math.ceil(interval)
        if self.interval_eval < interval:
            if interval > 60:
                self.enable_eval = False
                logger.info(f"Evaluation is done at the same time as the animation(eval time: {eval_time:.1f}s)")
            else:
                self.interval_eval = interval
                logger.info(f"set eval interval: {interval:.0f}s (eval time: {eval_time:.3f}s)")

    def _log_checkpoint(self, context: RunContext, state):
        if not self.enable_checkpoint:
            return
        with tempfile.TemporaryDirectory() as tmp_dir:
            name = context.rl_config.name.replace(":", "_")
            step = self._get_step(context, state)
            path = os.path.join(tmp_dir, f"{name}_{step}_model.dat")
            state.parameter.save(path)
            mlflow.log_artifact(path, run_id=self.run_id)

    def _log_html(self, context: RunContext, state):
        if not self.enable_html:
            return
        step = self._get_step(context, state)
        runner = self.create_eval_runner_if_not_exists(context, state)
        render = runner.run_render(parameter=state.parameter, enable_progress=False)
        html = render.to_jshtml()
        name = context.rl_config.name.replace(":", "_")
        rewards = runner.state.last_episode_rewards
        fn = f"{name}_{step}_{rewards}.html"
        mlflow.log_text(html, fn, run_id=self.run_id)
        d = {f"eval_reward{i}": r for i, r in enumerate(rewards)}
        mlflow.log_metrics(d, step, run_id=self.run_id)

    # ---------------------------------------------------------------

    @staticmethod
    def get_metric(env_name: str, rl_name: str, metric_name: str, idx: int = -1):
        run_id = MLFlowCallback.get_run_id(env_name, rl_name, idx)
        if run_id is None:
            return None

        # --- metric
        client = mlflow.tracking.MlflowClient()
        metric_history = client.get_metric_history(run_id, metric_name)
        return metric_history

    @staticmethod
    def load_parameter(
        experiment_name: str,
        parameter: RLParameter[RLConfig],
        run_idx: int = -1,
        parameter_idx: int = -1,
    ):
        run_id = MLFlowCallback.get_run_id(experiment_name, parameter.config.name, run_idx)
        if run_id is None:
            return

        path_list = []
        files = cast(list[mlflow.entities.FileInfo], mlflow.artifacts.list_artifacts(run_id=run_id))
        for file in files:
            m = re.search(r"(.+)_(\d+)_model.dat", str(file.path))
            if m:
                path_list.append((int(m.group(2)), file.path))
        path_list.sort()
        path = path_list[parameter_idx][1]

        logger.info(f"load artifact: run_id={run_id}, path={path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=temp_dir)
            parameter.load(os.path.join(temp_dir, path))

    @staticmethod
    def get_run_id(experiment_name: str, rl_name: str, idx: int = -1) -> Optional[str]:
        experiment_id = MLFlowCallback.get_experiment_id(experiment_name)
        if experiment_id is None:
            return None

        # --- tag search
        filter = f"tags.RL = '{rl_name}'"
        runs = mlflow.search_runs([experiment_id], filter_string=filter, order_by=["start_time ASC"])
        if runs.empty:
            return None
        run_id = runs.iloc[idx].run_id
        return run_id

    @staticmethod
    def get_experiment_id(experiment_name: str):
        client = mlflow.tracking.MlflowClient()
        for experiment in client.search_experiments(order_by=["creation_time DESC"]):
            if experiment.name == experiment_name:
                return experiment.experiment_id
        return None
