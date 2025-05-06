import logging
from dataclasses import dataclass
from typing import Any, List, Union, cast

from srl.base.define import PlayerType
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor, play
from srl.runner.runner_base import RunnerBase

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
        callbacks: List[RunCallback] = [],
    ) -> Union[List[float], List[List[float]]]:  # single play , multi play
        """シミュレーションし、報酬を返します。

        Args:
            max_episodes (int, optional): 終了するまでのエピソード数. Defaults to 10.
            timeout (int, optional): 終了するまでの時間（秒）. Defaults to -1.
            max_steps (int, optional): 終了するまでの総ステップ. Defaults to -1.
            shuffle_player (bool, optional): playersをシャッフルするかどうか. Defaults to True.
            enable_progress (bool, optional): 進捗を表示するか. Defaults to True.
            callbacks (List[RunCallback], optional): callbacks. Defaults to [].

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
        self.context.env_render_mode = ""
        self.context.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        self.setup_process()
        play(
            self.context,
            self.state,
            self._parameter_dat,
            self._memory_dat,
            callbacks,
        )
        self._parameter_dat = None
        self._memory_dat = None

        state = cast(RunStateActor, self.state)
        if self.env.player_num == 1:
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
        callbacks: List[RunCallback] = [],
    ):
        callbacks = callbacks[:]

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
        # render_modeはRendering側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        callbacks.append(
            Rendering(
                mode="terminal",
                kwargs=render_kwargs,
                step_stop=step_stop,
                render_skip_step=render_skip_step,
            )
        )
        # -----------------

        self.setup_process()
        play(
            self.context,
            self.state,
            self._parameter_dat,
            self._memory_dat,
            callbacks,
        )
        self._parameter_dat = None
        self._memory_dat = None

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
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[RunCallback] = [],
    ):
        callbacks = callbacks[:]

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
        # render_modeはRendering側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        # --- rendering
        from srl.runner.callbacks.rendering import Rendering

        callbacks.append(
            Rendering(
                mode="window",
                kwargs=render_kwargs,
                step_stop=step_stop,
                render_interval=render_interval,
                render_skip_step=render_skip_step,
                render_worker=render_worker,
                render_add_rl_terminal=render_add_rl_terminal,
                render_add_rl_rgb=render_add_rl_rgb,
                render_add_info_text=render_add_info_text,
            )
        )

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        self.setup_process()
        play(
            self.context,
            self.state,
            self._parameter_dat,
            self._memory_dat,
            callbacks,
        )
        self._parameter_dat = None
        self._memory_dat = None

        return self.state.episode_rewards_list[0]

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
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[RunCallback] = [],
    ):
        callbacks = callbacks[:]

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
        # render_modeはRendering側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        render = Rendering(
            mode="rgb_array",
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_worker=render_worker,
            render_add_rl_terminal=render_add_rl_terminal,
            render_add_rl_rgb=render_add_rl_rgb,
            render_add_info_text=render_add_info_text,
        )
        callbacks.append(render)
        # -----------------

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        self.setup_process()
        play(
            self.context,
            self.state,
            self._parameter_dat,
            self._memory_dat,
            callbacks,
        )
        self._parameter_dat = None
        self._memory_dat = None

        if self.context.run_name != "eval":
            logger.info(f"render step: {self.state.total_step}, reward: {self.state.episode_rewards_list[0]}")
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
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[RunCallback] = [],
    ):
        render = self.run_render(
            render_kwargs,
            step_stop,
            render_skip_step,
            render_worker,
            render_add_rl_terminal,
            render_add_rl_rgb,
            render_add_info_text,
            timeout,
            max_steps,
            players,
            enable_progress,
            callbacks,
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
        callbacks: List[RunCallback] = [],
    ):
        render = self.run_render(
            render_kwargs,
            step_stop,
            render_skip_step,
            render_worker,
            render_add_rl_terminal,
            render_add_rl_rgb,
            render_add_info_text,
            timeout,
            max_steps,
            players,
            enable_progress,
            callbacks,
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
        render_add_info_text: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- play config
        players: List[PlayerType] = [],
        # --- progress
        enable_progress: bool = True,
        # --- other
        callbacks: List[RunCallback] = [],
    ):
        render = self.run_render(
            render_kwargs,
            step_stop,
            render_skip_step,
            render_worker,
            render_add_rl_terminal,
            render_add_rl_rgb,
            render_add_info_text,
            timeout,
            max_steps,
            players,
            enable_progress,
            callbacks,
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
        callbacks: List[RunCallback] = [],
        _is_test: bool = False,  # for test
    ):
        callbacks = callbacks[:]

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
        # render_modeはRePlayableGame側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        if enable_progress:
            self.apply_progress(callbacks, enable_eval=False)

        from srl.runner.game_windows.replay_window import RePlayableGame

        self.setup_process()
        window = RePlayableGame(
            self.context,
            self.make_parameter(),
            print_state=print_state,
            callbacks=callbacks,
            _is_test=_is_test,
        )
        window.play()

    def play_terminal(
        self,
        action_division_num: int = 5,
        enable_memory: bool = False,
        players: List[PlayerType] = [],
        # Rendering
        render_kwargs: dict = {},
        step_stop: bool = False,
        render_skip_step: bool = True,
        # --- stop config
        timeout: float = -1,
        max_steps: int = -1,
        # --- other
        callbacks: List[RunCallback] = [],
    ):
        callbacks = callbacks[:]

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
        self.context.training = enable_memory
        self.context.train_only = False
        self.context.rollout = enable_memory
        # render_modeはRendering側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        # --- rendering ---
        from srl.runner.callbacks.rendering import Rendering

        rendering = Rendering(
            mode="terminal",
            kwargs=render_kwargs,
            step_stop=step_stop,
            render_skip_step=render_skip_step,
            render_env=False,
        )
        callbacks.append(rendering)

        # -----------------
        from srl.runner.callbacks.manual_play_callback import ManualPlayCallback

        callbacks.append(ManualPlayCallback(self.make_env(), action_division_num))
        # -----------------

        self.setup_process()
        play(
            self.context,
            self.state,
            self._parameter_dat,
            self._memory_dat,
            callbacks,
        )
        self._parameter_dat = None
        self._memory_dat = None

        return self.state.episode_rewards_list[0]

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
        callbacks: List[RunCallback] = [],
        _is_test: bool = False,  # for test
    ):
        callbacks2 = cast(List[RunCallback], [c for c in callbacks if issubclass(c.__class__, RunCallback)])

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
        # render_modeはPlayableGame側で設定
        # self.context.env_render_mode = ""
        # self.context.rl_render_mode = ""

        from srl.utils.common import is_packages_installed

        error_text = "This run requires installation of 'PIL', 'pygame'. "
        error_text += "(pip install pillow pygame)"
        assert is_packages_installed(["PIL", "pygame"]), error_text

        from srl.runner.game_windows.playable_game import PlayableGame

        self.setup_process()
        game = PlayableGame(
            env=self.make_env(),
            context=self.context,
            worker=self.make_worker(),
            view_state=view_state,
            action_division_num=action_division_num,
            key_bind=key_bind,
            callbacks=callbacks2,
            _is_test=_is_test,
        )
        game.play()
