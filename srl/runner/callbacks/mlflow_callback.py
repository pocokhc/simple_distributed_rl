import logging
import math
import os
import re
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Literal, Optional, Union, cast

import mlflow
import mlflow.artifacts
import mlflow.entities
import mlflow.tracking

import srl
from srl.base.context import RunContext, RunState
from srl.base.env.config import EnvConfig
from srl.base.exception import UndefinedError
from srl.base.rl.config import RLConfig
from srl.base.rl.parameter import RLParameter
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callbacks.evaluate import Evaluate
from srl.runner.runner import Runner

logger = logging.getLogger(__name__)


@dataclass
class MLFlowCallback(RunCallback, Evaluate):
    experiment_name: str = ""
    run_name: str = ""
    tags: dict = field(default_factory=dict)

    interval: Union[float, int] = 60
    interval_mode: Literal["time", "step"] = "time"

    eval_interval: float = -1  # -1 is auto
    checkpoint_interval: float = 60 * 30
    enable_checkpoint: bool = True

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
                self.run_name += f"({context.play_mode})"
            mlflow.start_run(run_name=self.run_name)
            self._auto_run = True
        active_run = mlflow.active_run()
        assert active_run is not None
        self.run_id = active_run.info.run_id
        print(f"MLFlow experiment={self.experiment_name}, run={self.run_name}, run_id={self.run_id}")
        logger.info(f"MLFlow experiment={self.experiment_name}, run={self.run_name}, run_id={self.run_id}")
        tags = {
            "Version": srl.__version__,
            "Env": context.env_config.name,
            "RL": context.rl_config.name,
            "Framework": context.rl_config.get_framework(),
            "Play": context.play_mode,
        }
        tags.update(self.tags)
        mlflow.set_tags(tags)

        d = {"env/" + k: v for k, v in context.env_config.to_dict(to_print=True).items()}
        d.update({"rl/" + k: v for k, v in context.rl_config.to_dict(to_print=True).items()})
        d.update({"rl_base/" + k: v for k, v in context.rl_config.to_dict(to_print=True, include_rl_config=False, include_base_config=True).items()})
        con_dict = context.to_dict(to_print=True)
        d.update({"context/" + k: v for k, v in con_dict.items() if k != "callbacks"})
        for i, c in enumerate(context.callbacks):
            if isinstance(c, MLFlowCallback):
                continue
            d[f"context/callbacks_{i}"] = str(c)
        mlflow.log_params(d)

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "rl_config.yaml")
            context.rl_config.save(path)
            mlflow.log_artifact(path, run_id=self.run_id)

            path = os.path.join(tmp_dir, "env_config.yaml")
            context.env_config.save(path)
            mlflow.log_artifact(path, run_id=self.run_id)

            path = os.path.join(tmp_dir, "context.yaml")
            context.save(path, include_env_config=False, include_rl_config=False)
            mlflow.log_artifact(path, run_id=self.run_id)

        # if context.distributed:
        #    # メインプロセス以外でtkinterを使うと落ちるので使わない

    def on_end(self, context: RunContext, **kwargs) -> None:
        if self._auto_run:
            mlflow.end_run()

    # ---------------------------------------------------------------

    def on_episodes_begin(self, context: RunContext, state, **kwargs) -> None:
        if context.actor_id != 0:
            return

        # --- error check
        self._log_actor(context, state)
        self._log_eval(context, state, init=True)
        if not context.distributed:
            self._log_checkpoint(context, state)

        if self.interval_mode == "time":
            self.interval0 = time.time()
        elif self.interval_mode == "step":
            self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

        self.t0_eval = time.time()
        self.t0_checkpoint = time.time()

    def on_step_end(self, context: RunContext, state, **kwargs) -> bool:
        if context.actor_id != 0:
            return False
        _time = time.time()

        if self.interval_mode == "time":
            if _time - self.interval0 > self.interval:
                self._log_actor(context, state)
                self.interval0 = time.time()  # last
        elif self.interval_mode == "step":
            self.interval0 += 1
            if self.interval0 >= self.interval:
                self._log_actor(context, state)
                self.interval0 = 0  # last
        else:
            raise UndefinedError(self.interval_mode)

        if _time - self.t0_eval > self._eval_interval:
            self._log_eval(context, state)
            self.t0_eval = time.time()  # last

        if not context.distributed:
            if _time - self.t0_checkpoint > self.checkpoint_interval:
                self._log_checkpoint(context, state)
                self.t0_checkpoint = time.time()  # last

        return False

    def on_episodes_end(self, context: RunContext, state, **kwargs) -> None:
        if context.actor_id != 0:
            return
        self._log_actor(context, state)
        if not context.distributed:
            self._log_eval(context, state)  # 最後はevalしない
            self._log_checkpoint(context, state)

    # ---------------------------------------------------------------

    def on_trainer_start(self, context: RunContext, state, **kwargs) -> None:
        self._log_trainer(context, state)
        if not context.distributed:
            self._log_eval(context, state, init=True)
        self._log_checkpoint(context, state)

        if self.interval_mode == "time":
            self.interval0 = time.time()
        elif self.interval_mode == "step":
            self.interval0 = 0
        else:
            raise UndefinedError(self.interval_mode)

        self.t0_eval = time.time()
        self.t0_checkpoint = time.time()

    def on_trainer_end(self, context: RunContext, state, **kwargs) -> None:
        self._log_trainer(context, state)
        if not context.distributed:
            self._log_eval(context, state)
        self._log_checkpoint(context, state)

    def on_train_after(self, context: RunContext, state, **kwargs) -> bool:
        _time = time.time()

        if self.interval_mode == "time":
            if _time - self.interval0 > self.interval:
                self._log_trainer(context, state)
                self.interval0 = time.time()  # last
        elif self.interval_mode == "step":
            self.interval0 += 1
            if self.interval0 >= self.interval:
                self._log_trainer(context, state)
                self.interval0 = 0  # last
        else:
            raise UndefinedError(self.interval_mode)

        if not context.distributed:
            if _time - self.t0_eval > self._eval_interval:
                self._log_eval(context, state)
                self.t0_eval = time.time()  # last

        if _time - self.t0_checkpoint > self.checkpoint_interval:
            self._log_checkpoint(context, state)
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

    def _log_actor(self, context: RunContext, state: RunState):
        state = cast(RunStateActor, state)
        d = {}
        for i, r in enumerate(state.last_episode_rewards):
            d[f"worker/reward{i}"] = r
        d["worker/episode"] = state.episode_count
        d["worker/total_step"] = state.total_step
        if state.trainer is not None:
            for k, v in state.trainer.info.to_dict().items():
                d["trainer/" + k] = v
            d["trainer/train_count"] = state.trainer.train_count
        else:
            d["worker/train_count"] = state.train_count
        for k, v in state.worker.info.to_dict().items():
            d["worker/" + k] = v
        d.update(self._get_system_log(context, state))
        for k, v in state.env.info.to_dict().items():
            d["env/" + k] = v
        mlflow.log_metrics(d, self._get_step(context, state), run_id=self.run_id)

    def _log_trainer(self, context: RunContext, state: RunState):
        state = cast(RunStateTrainer, state)
        d = {"trainer/" + k: v for k, v in state.trainer.info.to_dict().items()}
        d["trainer/train_count"] = state.trainer.train_count
        d.update(self._get_system_log(context, state))
        mlflow.log_metrics(d, state.trainer.train_count, run_id=self.run_id)

    def _get_system_log(self, context: RunContext, state):
        if not context.enable_stats:
            return {}

        d = {}
        if context.actor_id == 0:
            try:
                from srl.base.system import psutil_

                d["system/memory"] = psutil_.read_memory()
                d["system/cpu"] = psutil_.read_cpu()
            except Exception:
                logger.debug(traceback.format_exc())

            try:
                from srl.base.system.pynvml_ import read_nvml

                gpus = read_nvml()
                # device_id, rate.gpu, rate.memory
                for device_id, gpu, gpu_memory in gpus:
                    d[f"system/gpu{device_id}"] = gpu
                    d[f"system/gpu{device_id}_memory"] = gpu_memory
            except Exception:
                logger.debug(traceback.format_exc())

        else:
            try:
                from srl.base.system import psutil_

                d[f"system/actor{context.actor_id}/cpu"] = psutil_.read_cpu()
            except Exception:
                logger.debug(traceback.format_exc())

        return d

    def _log_eval(self, context: RunContext, state: RunState, init: bool = False):
        if not self.enable_eval:
            return

        if init:
            self._eval_interval = 0 if self.eval_interval < 0 else self.eval_interval

        t0 = time.time()
        eval_rewards = self.run_eval_with_state(context, state)
        if eval_rewards is not None:
            d = {f"eval/reward{i}": r for i, r in enumerate(eval_rewards)}
            d["eval/train_count"] = state.train_count
            d["eval/total_step"] = state.total_step
            mlflow.log_metrics(d, self._get_step(context, state), run_id=self.run_id)

        # check interval
        if self.eval_interval < 0:
            eval_time = time.time() - t0
            interval = eval_time * 10
            if interval < 1:
                interval = 1
            interval = math.ceil(interval)
            if interval > 60 * 10:
                interval = 60 * 10
            if self._eval_interval < interval:
                self._eval_interval = interval
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

    # ---------------------------------------------------------------

    @staticmethod
    def get_experiment_id(experiment_name: str):
        client = mlflow.tracking.MlflowClient()
        for experiment in client.search_experiments(order_by=["creation_time DESC"]):
            if experiment.name == experiment_name:
                return experiment.experiment_id
        return None

    @classmethod
    def get_run_id(cls, experiment_name: str, run_name: str = "", rl_name: str = "", idx: int = -1) -> Optional[str]:
        experiment_id = cls.get_experiment_id(experiment_name)
        if experiment_id is None:
            return None

        if run_name != "":
            runs = mlflow.search_runs(
                [experiment_id],
                filter_string=f"tags.mlflow.runName = '{run_name}'",
                order_by=["start_time ASC"],
            )
            if not runs.empty:
                return runs.iloc[idx].run_id

        if rl_name != "":
            filter = f"tags.RL = '{rl_name}'"
            runs = mlflow.search_runs(
                [experiment_id],
                filter_string=filter,
                order_by=["start_time ASC"],
            )
            if not runs.empty:
                return runs.iloc[idx].run_id
        return None

    @classmethod
    def get_metric(cls, run_id: Optional[str], metric_name: str):
        if run_id is None:
            return None

        # --- metric
        client = mlflow.tracking.MlflowClient()
        metric_history = client.get_metric_history(run_id, metric_name)
        return metric_history

    @classmethod
    def get_parameter_files(cls, run_id: Optional[str]):
        if run_id is None:
            return []

        path_list = []
        files = cast(list[mlflow.entities.FileInfo], mlflow.artifacts.list_artifacts(run_id=run_id))
        for file in files:
            m = re.search(r"(.+)_(\d+)_model.dat", str(file.path))
            if m:
                path_list.append((int(m.group(2)), file.path))
        path_list.sort()
        return [p[1] for p in path_list]

    @staticmethod
    def load_parameter(run_id: Optional[str], path: str, parameter: RLParameter[RLConfig]):
        if run_id is None:
            return

        logger.info(f"load artifact: run_id={run_id}, path={path}")
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=temp_dir)
            parameter.load(os.path.join(temp_dir, path))

    @classmethod
    def make_html_all_parameters(cls, run_id: Optional[str], env_config: EnvConfig, rl_config: RLConfig, **render_kwargs):
        if run_id is None:
            return
        files = cls.get_parameter_files(run_id)

        runner = Runner(env_config, rl_config)
        for file in files:
            logger.info(f"{run_id}: {file=}")
            cls.load_parameter(run_id, file, runner.make_parameter())
            render = runner._run_render(enable_progress=False, **render_kwargs)
            html = render.to_jshtml()
            rewards = runner.state.last_episode_rewards
            fn = file[: -len("model.dat")] + f"{rewards}.html"
            mlflow.log_text(html, fn, run_id=run_id)

    @classmethod
    def load_env_config(cls, run_id: str, path: str = "env_config.yaml"):
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=temp_dir)
            cfg = EnvConfig.load(os.path.join(temp_dir, path))
        return cfg

    @classmethod
    def load_rl_config(cls, run_id: str, path: str = "rl_config.yaml"):
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=temp_dir)
            cfg = RLConfig.load(os.path.join(temp_dir, path))
        return cfg

    @classmethod
    def load_context(cls, run_id: str, path: str = "context.yaml"):
        with tempfile.TemporaryDirectory() as temp_dir:
            mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=path, dst_path=temp_dir)
            cfg = RunContext.load(os.path.join(temp_dir, path))
        return cfg
