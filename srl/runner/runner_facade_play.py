import logging
from dataclasses import dataclass
from typing import Any, List, Union, cast

from srl.base.context import RunNameTypes
from srl.base.define import PlayerType, RenderModes
from srl.base.run.core_play import RunStateActor
from srl.runner.runner_base import CallbackType, RunnerBase

logger = logging.getLogger(__name__)


@dataclass()
class RunnerFacadePlay(RunnerBase):
    def evaluate(
        self,
        # --- stop config
        max_episodes: int = 10,
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
    ) -> Union[List[float], List[List[float]]]:  # single play , multi play
        """シミュレーションし、報酬を返します。

        Args:
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to 10.
            timeout (int, optional): 終了するまでの時間（秒）. Defaults to -1.
            max_steps (int, optional): 終了するまでの総ステップ. Defaults to -1.
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            progress_start_time (int, optional):  最初に進捗を表示する秒数. Defaults to 1.
            progress_interval_limit (int, optional): 進捗を表示する最大の間隔（秒）. Defaults to 60*10.
            progress_env_info (bool, optional): 進捗表示にenv infoを表示するか. Defaults to False.
            progress_worker_info (bool, optional): 進捗表示にworker infoを表示するか. Defaults to True.
            progress_worker (int, optional): 進捗表示に表示するworker index. Defaults to 0.
            callbacks (List[CallbackType], optional): callbacks. Defaults to [].

        Returns:
            Union[List[float], List[List[float]]]: プレイヤー数が1人なら Lost[float]、複数なら List[List[float]]] を返します。
        """
        callbacks = callbacks[:]

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.rendering = False
        self.context.render_mode = RenderModes.none
        # thread
        self.context.enable_train_thread = False

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )
        state = cast(
            RunStateActor,
            self._wrap_base_run(
                parameter=None,
                memory=None,
                trainer=None,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                logger_config=False,
            ),
        )
        self._base_run_after()

        if self.env_config.player_num == 1:
            return [r[0] for r in state.episode_rewards_list]
        else:
            return state.episode_rewards_list

    def render_terminal(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        callbacks = callbacks[:]
        mode = RenderModes.terminal

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        callbacks.append(
            Rendering(
                mode=mode,
                kwargs=render_kwargs,
                step_stop=step_stop,
                render_skip_step=render_skip_step,
            )
        )
        logger.info("enable Rendering")
        # -----------------

        self._base_run_before(
            enable_progress=False,
            enable_progress_eval=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )
        state = cast(
            RunStateActor,
            self._wrap_base_run(
                parameter=None,
                memory=None,
                trainer=None,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                logger_config=False,
            ),
        )
        self._base_run_after()

        return state.episode_rewards_list[0]

    def render_window(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        callbacks = callbacks[:]
        mode = RenderModes.window

        # --- context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        # --- rendering
        from srl.runner.callbacks.rendering import Rendering

        callbacks.append(
            Rendering(
                mode=mode,
                kwargs=render_kwargs,
                step_stop=step_stop,
                render_skip_step=render_skip_step,
                render_interval=render_interval,
                render_scale=render_scale,
                font_name=font_name,
                font_size=font_size,
            )
        )
        logger.info("add callback Rendering")

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )
        state = cast(
            RunStateActor,
            self._wrap_base_run(
                parameter=None,
                memory=None,
                trainer=None,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                logger_config=False,
            ),
        )
        self._base_run_after()

        return state.episode_rewards_list[0]

    def animation_save_gif(
        self,
        path: str,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
        #
        draw_info: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_interval=render_interval,
            render_scale=render_scale,
            font_name=font_name,
            font_size=font_size,
        )
        callbacks.append(rendering)
        logger.info("add callback Rendering")
        # -----------------

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )
        state = cast(
            RunStateActor,
            self._wrap_base_run(
                parameter=None,
                memory=None,
                trainer=None,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                logger_config=False,
            ),
        )
        self._base_run_after()

        print(f"animation_save_gif step: {state.total_step}, reward: {state.episode_rewards_list[0]}")
        rendering.save_gif(path, render_interval, draw_info)

        return state

    def animation_save_avi(
        self,
        path: str,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
        draw_info: bool = True,
        codec: str = "XVID",
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_interval=render_interval,
            render_scale=render_scale,
            font_name=font_name,
            font_size=font_size,
        )
        callbacks.append(rendering)
        logger.info("add callback Rendering")
        # -----------------

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )
        state = cast(
            RunStateActor,
            self._wrap_base_run(
                parameter=None,
                memory=None,
                trainer=None,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                logger_config=False,
            ),
        )
        self._base_run_after()

        print(f"animation_save_avi step: {state.total_step}, reward: {state.episode_rewards_list[0]}")
        rendering.save_avi(path, render_interval, draw_info, codec=codec)

        return state

    def animation_display(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        font_name: str = "",
        font_size: int = 18,
        #
        draw_info: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_interval=render_interval,
            render_scale=render_scale,
            font_name=font_name,
            font_size=font_size,
        )
        callbacks.append(rendering)
        logger.info("add callback Rendering")
        # -----------------

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )
        state = cast(
            RunStateActor,
            self._wrap_base_run(
                parameter=None,
                memory=None,
                trainer=None,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                logger_config=False,
            ),
        )
        self._base_run_after()

        print(f"animation_display step: {state.total_step}, reward: {state.episode_rewards_list[0]}")
        rendering.display(render_interval, render_scale, draw_info)

        return state

    def replay_window(
        self,
        view_state: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        _is_test: bool = False,  # for test
    ):
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        self._base_run_before(
            enable_progress=enable_progress,
            enable_progress_eval=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )

        from srl.runner.game_windows.replay_window import RePlayableGame

        window = RePlayableGame(self, view_state=view_state, callbacks=callbacks, _is_test=_is_test)
        window.play()

        self._base_run_after()

    def play_terminal(
        self,
        players: List[PlayerType] = ["human"],
        # Rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- other
        callbacks: List[CallbackType] = [],
    ):
        callbacks = callbacks[:]
        mode = RenderModes.terminal
        self.context.players = players

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
        )
        callbacks.append(rendering)
        logger.info("add callback Rendering")
        # -----------------

        self._base_run_before(
            enable_progress=False,
            enable_progress_eval=False,
            enable_checkpoint=False,
            enable_history_on_memory=False,
            enable_history_on_file=False,
            callbacks=callbacks,
        )

        state = cast(
            RunStateActor,
            self._wrap_base_run(
                parameter=None,
                memory=None,
                trainer=None,
                workers=None,
                main_worker_idx=0,
                callbacks=callbacks,
                logger_config=False,
            ),
        )
        self._base_run_after()

        return state.episode_rewards_list[0]

    def play_window(
        self,
        key_bind: Any = None,
        view_state: bool = True,
        action_division_num: int = 5,
        enable_memory: bool = False,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # other
        callbacks: List[CallbackType] = [],
        _is_test: bool = False,  # for test
    ):
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.run_name = RunNameTypes.main
        # stop config
        self.context.max_episodes = -1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = enable_memory
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        from srl.utils.common import is_packages_installed

        error_text = "This run requires installation of 'PIL', 'pygame'. "
        error_text += "(pip install pillow pygame)"
        assert is_packages_installed(["PIL", "pygame"]), error_text

        from srl.runner.game_windows.playable_game import PlayableGame

        workers, main_worker_idx = self.make_workers()
        game = PlayableGame(
            env=self.make_env(),
            context=self.context,
            workers=workers,
            view_state=view_state,
            action_division_num=action_division_num,
            key_bind=key_bind,
            callbacks=callbacks,
            _is_test=_is_test,
        )
        game.play()
