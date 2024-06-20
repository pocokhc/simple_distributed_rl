import datetime
import glob
import logging
import os
import time
import traceback
from dataclasses import dataclass

from srl.base.context import RunContext
from srl.base.rl.parameter import RLParameter
from srl.base.rl.trainer import RLTrainer
from srl.base.run.callback import RunCallback, TrainCallback
from srl.base.run.core_play import RunStateActor
from srl.base.run.core_train_only import RunStateTrainer
from srl.runner.callbacks.evaluate import Evaluate

logger = logging.getLogger(__name__)


@dataclass
class Checkpoint(RunCallback, TrainCallback, Evaluate):
    save_dir: str = "checkpoints"
    interval: int = 60 * 20  # s

    @staticmethod
    def get_parameter_path(save_dir: str) -> str:
        # 最後のpathを取得
        trains = []
        for f in glob.glob(os.path.join(save_dir, "*.pickle")):
            try:
                date = os.path.basename(f).split("_")[0]
                date = datetime.datetime.strptime(date, "%Y%m%d-%H%M%S")
                trains.append([date, f])
            except Exception:
                logger.warning(traceback.format_exc())
        if len(trains) == 0:
            return ""
        trains.sort()
        return trains[-1][1]

    def on_start(self, context: RunContext, **kwargs) -> None:
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            logger.info(f"makedirs: {self.save_dir}")

    def _save_parameter(self, context: RunContext, trainer: RLTrainer, parameter: RLParameter, is_last: bool):
        train_count = trainer.get_train_count()

        eval_rewards = self.run_eval(context.env_config, context.rl_config, parameter)
        if eval_rewards is None:
            eval_rewards = "None"

        fn = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        fn += f"_{train_count}_{eval_rewards}"
        if is_last:
            fn += "_last"
        fn += ".pickle"

        parameter.save(os.path.join(self.save_dir, fn))

    # ---------------------------
    # actor
    # ---------------------------
    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs):
        # Trainerがいる場合のみ保存
        if state.trainer is None:
            return

        self.interval_t0 = time.time()
        self._save_parameter(context, state.trainer, state.parameter, is_last=False)

    def on_episode_end(self, context: RunContext, state: RunStateActor, **kwargs):
        if state.trainer is None:
            return
        if time.time() - self.interval_t0 > self.interval:
            self._save_parameter(context, state.trainer, state.parameter, is_last=False)
            self.interval_t0 = time.time()  # last

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        if state.trainer is None:
            return
        self._save_parameter(context, state.trainer, state.parameter, is_last=True)

    # ---------------------------
    # trainer
    # ---------------------------
    def on_trainer_start(self, context: RunContext, state: RunStateTrainer, **kwargs):
        self.interval_t0 = time.time()
        self._save_parameter(context, state.trainer, state.parameter, is_last=False)

    def on_train_before(self, context: RunContext, state: RunStateTrainer, **kwargs):
        if time.time() - self.interval_t0 > self.interval:
            self._save_parameter(context, state.trainer, state.parameter, is_last=False)
            self.interval_t0 = time.time()  # last

    def on_trainer_end(self, context: RunContext, state: RunStateTrainer, **kwargs):
        self._save_parameter(context, state.trainer, state.parameter, is_last=True)
