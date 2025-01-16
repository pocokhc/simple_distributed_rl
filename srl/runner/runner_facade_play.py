import logging
from dataclasses import dataclass
from typing import Any, List, Optional, Union, cast

from srl.base.context import RunNameTypes
from srl.base.define import PlayerType, RenderModes
from srl.base.rl.memory import RLMemory
from srl.base.rl.parameter import RLParameter
from srl.base.run.callback import RunCallback
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
        players: List[PlayerType] = [],
        shuffle_player: bool = True,
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
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
        self.context.flow_mode = "evaluate"
        # stop config
        self.context.max_episodes = max_episodes
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = shuffle_player
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.train_only = False
        self.context.rollout = False
        self.context.rendering = False
        self.context.render_mode = RenderModes.none
        # thread
        self.context.enable_train_thread = False

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        self.run_context(parameter=parameter, memory=memory, callbacks=callbacks)

        state = cast(RunStateActor, self.state)
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
        # --- play config
        players: List[PlayerType] = [],
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        callbacks = callbacks[:]
        mode = RenderModes.terminal

        # --- set context
        self.context.flow_mode = "render_terminal"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.train_only = False
        self.context.rollout = False
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
        # -----------------

        self.run_context(parameter=parameter, memory=memory, callbacks=callbacks)

        state = cast(RunStateActor, self.state)
        return state.episode_rewards_list[0]

    def render_window(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,
        render_worker: int = 0,
        render_add_rl_terminal: bool = True,
        render_add_rl_rgb: bool = True,
        render_add_rl_state: bool = True,
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        callbacks = callbacks[:]
        mode = RenderModes.window

        # --- context
        self.context.flow_mode = "render_window"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.train_only = False
        self.context.rollout = False
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
                render_interval=render_interval,
                render_skip_step=render_skip_step,
                render_worker=render_worker,
                render_add_rl_terminal=render_add_rl_terminal,
                render_add_rl_rgb=render_add_rl_rgb,
                render_add_rl_state=render_add_rl_state,
                render_add_info_text=render_add_info_text,
            )
        )

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        self.run_context(parameter=parameter, memory=memory, callbacks=callbacks)

        state = cast(RunStateActor, self.state)
        return state.episode_rewards_list[0]

    def run_render(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_worker: int = 0,
        render_add_rl_terminal: bool = True,
        render_add_rl_rgb: bool = True,
        render_add_rl_state: bool = True,
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.flow_mode = "run_render"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.train_only = False
        self.context.rollout = False
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        render = Rendering(
            mode=mode,
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_worker=render_worker,
            render_add_rl_terminal=render_add_rl_terminal,
            render_add_rl_rgb=render_add_rl_rgb,
            render_add_rl_state=render_add_rl_state,
            render_add_info_text=render_add_info_text,
        )
        callbacks.append(render)
        # -----------------

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        self.run_context(parameter=parameter, memory=memory, callbacks=callbacks)

        state = cast(RunStateActor, self.state)
        if self.context.run_name != RunNameTypes.eval:
            logger.info(f"render step: {state.total_step}, reward: {state.episode_rewards_list[0]}")
        return render

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
        render_worker: int = 0,
        render_add_rl_terminal: bool = True,
        render_add_rl_rgb: bool = True,
        render_add_rl_state: bool = True,
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        render = self.run_render(
            render_kwargs,
            step_stop,
            render_skip_step,
            render_worker,
            render_add_rl_terminal,
            render_add_rl_rgb,
            render_add_rl_state,
            render_add_info_text,
            timeout,
            max_steps,
            players,
            enable_progress,
            callbacks,
            parameter,
            memory,
        )
        render.save_gif(path, render_interval, render_scale)
        return render

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
        render_worker: int = 0,
        render_add_rl_terminal: bool = True,
        render_add_rl_rgb: bool = True,
        render_add_rl_state: bool = True,
        render_add_info_text: bool = True,
        codec: str = "XVID",
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        render = self.run_render(
            render_kwargs,
            step_stop,
            render_skip_step,
            render_worker,
            render_add_rl_terminal,
            render_add_rl_rgb,
            render_add_rl_state,
            render_add_info_text,
            timeout,
            max_steps,
            players,
            enable_progress,
            callbacks,
            parameter,
            memory,
        )
        render.save_avi(path, render_interval, render_scale, codec=codec)
        return render

    def animation_display(
        self,
        # rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # render option
        render_interval: float = -1,  # ms
        render_scale: float = 1.0,
        render_worker: int = 0,
        render_add_rl_terminal: bool = True,
        render_add_rl_rgb: bool = True,
        render_add_rl_state: bool = True,
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        render = self.run_render(
            render_kwargs,
            step_stop,
            render_skip_step,
            render_worker,
            render_add_rl_terminal,
            render_add_rl_rgb,
            render_add_rl_state,
            render_add_info_text,
            timeout,
            max_steps,
            players,
            enable_progress,
            callbacks,
            parameter,
            memory,
        )
        render.display(render_interval, render_scale)
        return render

    def replay_window(
        self,
        print_state: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[CallbackType] = [],
        _is_test: bool = False,  # for test
    ):
        callbacks = callbacks[:]
        mode = RenderModes.rgb_array

        # --- set context
        self.context.flow_mode = "replay_window"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.train_only = False
        self.context.rollout = False
        self.context.rendering = True
        self.context.render_mode = mode
        # thread
        self.context.enable_train_thread = False

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        from srl.runner.game_windows.replay_window import RePlayableGame

        window = RePlayableGame(self, print_state=print_state, callbacks=callbacks, _is_test=_is_test)
        window.play()

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
        parameter: Optional[RLParameter] = None,
        memory: Optional[RLMemory] = None,
    ):
        callbacks = callbacks[:]
        mode = RenderModes.terminal

        # --- set context
        self.context.flow_mode = "play_terminal"
        # stop config
        self.context.max_episodes = 1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = False
        self.context.train_only = False
        self.context.rollout = False
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
        # -----------------

        self.run_context(parameter=parameter, memory=memory, callbacks=callbacks)

        state = cast(RunStateActor, self.state)
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
        # --- play config
        players: List[PlayerType] = [],
        # other
        callbacks: List[CallbackType] = [],
        _is_test: bool = False,  # for test
    ):
        callbacks2 = cast(List[RunCallback], [c for c in callbacks if issubclass(c.__class__, RunCallback)])
        mode = RenderModes.rgb_array

        # --- set context
        self.context.flow_mode = "play_window"
        # stop config
        self.context.max_episodes = -1
        self.context.timeout = timeout
        self.context.max_steps = max_steps
        self.context.max_train_count = 0
        self.context.max_memory = 0
        # play config
        self.context.players = players
        self.context.shuffle_player = False
        self.context.disable_trainer = True
        # play info
        self.context.distributed = False
        self.context.training = enable_memory
        self.context.train_only = False
        self.context.rollout = enable_memory
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
            callbacks=callbacks2,
            _is_test=_is_test,
        )
        game.play()
