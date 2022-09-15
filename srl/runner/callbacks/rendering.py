import logging
import os
import time
from dataclasses import dataclass, field
from typing import List

import numpy as np
from srl.base.define import PlayRenderMode
from srl.base.env.base import EnvRun
from srl.base.rl.worker import WorkerRun
from srl.runner.callback import Callback
from srl.utils.common import is_package_installed
from srl.utils.render_functions import text_to_rgb_array

try:
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.animation import ArtistAnimation
except ImportError:
    pass

logger = logging.getLogger(__name__)


@dataclass
class Rendering(Callback):

    render_terminal: bool = True
    render_window: bool = False
    render_kwargs: dict = field(default_factory=dict)
    step_stop: bool = False
    enable_animation: bool = False
    use_skip_step: bool = True

    def __post_init__(self):
        self.frames = []
        self.info_maxw = 0
        self.info_maxh = 0
        self.env_maxw = 0
        self.env_maxh = 0

        self.rl_text = ""
        self.rl_img = None

        self.render_interval = -1

        self.default_font_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "font", "PlemolJPConsoleHS-Regular.ttf")
        )
        self.font = None

        if self.enable_animation or self.render_window:
            if not (
                is_package_installed("cv2")
                and is_package_installed("matplotlib")
                and is_package_installed("PIL")
                and is_package_installed("pygame")
            ):
                assert (
                    False
                ), "To use animation you need to install 'cv2', 'matplotlib', 'PIL', 'pygame'. (pip install opencv-python matplotlib pillow pygame)"

    def on_episodes_begin(self, info) -> None:
        env: EnvRun = info["env"]
        worker: List[WorkerRun] = info["workers"]

        if self.render_window:
            env_mode = PlayRenderMode.window
            rl_mode = PlayRenderMode.terminal
        elif self.enable_animation:
            env_mode = PlayRenderMode.rgb_array
            rl_mode = PlayRenderMode.rgb_array
        elif self.render_terminal:
            env_mode = PlayRenderMode.terminal
            rl_mode = PlayRenderMode.terminal
        else:
            env_mode = PlayRenderMode.none
            rl_mode = PlayRenderMode.none
        env.set_render_mode(env_mode)
        [w.set_render_mode(rl_mode) for w in worker]
        self.render_interval = env.env.render_interval

    def on_step_begin(self, info) -> None:
        self._render_step(info)

    def on_episode_end(self, info) -> None:
        self._render_step(info)

    def on_skip_step(self, info):
        if not self.use_skip_step:
            return
        self._render_step(info, True)

    def _render_step(self, info, is_skip=False):
        env: EnvRun = info["env"]
        worker_idx: int = info["worker_idx"]
        worker: WorkerRun = info["workers"][worker_idx]
        action = info["action"]
        step_time = info["step_time"] if "step_time" in info else None

        # env text
        env_text = env.render_terminal(return_text=True, **self.render_kwargs)

        # rl text
        if not is_skip:
            self.rl_text = worker.render_terminal(env, return_text=True, **self.render_kwargs)

        # --- info text
        info_text = f"### {env.step_num}"
        if isinstance(action, float):
            info_text += f", action {action:.3f}"
        else:
            info_text += f", action {action}"
        info_text += ", rewards[" + ",".join([f"{r:.3f}," for r in env.step_rewards]) + "]"
        if env.done:
            info_text += f", done({env.done_reason})"
        info_text += f", next {env.next_player_index}"
        if is_skip:
            info_text += "(skip frame)"
        if step_time is not None:
            info_text += f" ({step_time:.1f}s)"
        info_text += f"\nenv   {env.info}"
        info_text += f"\nwork{worker_idx: <2d}{worker.info}"

        # --- render_terminal
        if self.render_terminal:
            print(info_text)
            if env_text != "":
                print(env_text)
            if self.rl_text != "" and not env.done:
                print(self.rl_text)

        # --- render window
        if self.render_window:
            env_img = env.render_window(**self.render_kwargs)
        else:
            env_img = None

        # --- animation
        if self.enable_animation:
            info_img = text_to_rgb_array(info_text)
            if env_img is None:
                env_img = env.render_rgb_array(**self.render_kwargs)
            if self.rl_img is None or not is_skip:
                self.rl_img = worker.render_rgb_array(env, **self.render_kwargs)

            self.info_maxw = max(max(self.info_maxw, info_img.shape[1]), self.rl_img.shape[1])
            self.info_maxh = max(self.info_maxh, info_img.shape[0] + self.rl_img.shape[0])
            self.env_maxw = max(self.env_maxw, env_img.shape[1])
            self.env_maxh = max(self.env_maxh, env_img.shape[0])

            self.frames.append(
                {
                    "info_image": info_img,
                    "env_image": env_img,
                    "rl_image": self.rl_img,
                }
            )

        if self.step_stop:
            input("Enter to continue:")

    # -----------------------------------------------
    def _create_image(self, info_image: np.ndarray, env_image: np.ndarray, rl_image: np.ndarray):

        # --- 余白を追加
        padding = 2
        info_image = cv2.copyMakeBorder(
            info_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        rl_image = cv2.copyMakeBorder(
            rl_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        env_image = cv2.copyMakeBorder(
            env_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )

        # --- info + rl_image、大きいほうに合わせる、余白は右に埋める
        maxw = max(self.info_maxw + padding * 2, self.info_maxh + padding * 4)
        info_w = maxw - info_image.shape[1]
        rl_w = maxw - rl_image.shape[1]
        info_image = cv2.copyMakeBorder(info_image, 0, 0, 0, info_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        rl_image = cv2.copyMakeBorder(rl_image, 0, 0, 0, rl_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img1 = cv2.vconcat([info_image, rl_image])  # 縦連結

        # --- env + info、下を埋める
        maxw = max(self.env_maxh + padding * 2, self.info_maxh + padding * 4)
        env_h = maxw - env_image.shape[0]
        img1_h = maxw - img1.shape[0]
        env_image = cv2.copyMakeBorder(env_image, 0, env_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        img1 = cv2.copyMakeBorder(img1, 0, img1_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img2 = cv2.hconcat([env_image, img1])  # 横連結

        return img2

    # -----------------------------------------------

    def create_anime(
        self,
        scale: float = 1.0,
        interval: float = -1,  # ms
        draw_info: bool = False,
    ):
        if len(self.frames) == 0:
            return None
        t0 = time.time()

        maxw = 0
        maxh = 0
        images = []
        for f in self.frames:
            env_img = f["env_image"]
            info_img = f["info_image"]
            rl_img = f["rl_image"]
            if rl_img is None:
                continue

            if draw_info:
                img = self._create_image(info_img, env_img, rl_img)
            else:
                img = env_img
            images.append(img)
            maxw = max(maxw, img.shape[1])
            maxh = max(maxh, img.shape[0])

        # --- interval
        if interval <= 0:
            interval = self.render_interval
        if interval <= 0:
            interval = 1000 / 60
        logger.info("interval: {:.1f}ms".format(interval))

        # --- size (inch = pixel / dpi)
        fig_dpi = 100
        fig = plt.figure(
            dpi=fig_dpi, figsize=(scale * maxw / fig_dpi, scale * maxh / fig_dpi), tight_layout=dict(pad=0)
        )

        # --- animation
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        images = [[ax.imshow(img, animated=True)] for img in images]
        anime = ArtistAnimation(fig, images, interval=interval, repeat=False)
        # plt.close(fig)  # notebook で画像が残るので出来ればcloseしたいけど、closeするとgym側でバグる

        logger.info("create animation({:.1f}s)".format(time.time() - t0))
        return anime

    def display(
        self,
        scale: float = 1.0,
        interval: float = -1,  # ms
        draw_info: bool = False,
    ) -> None:
        if len(self.frames) == 0:
            return

        from IPython import display

        t0 = time.time()
        anime = self.create_anime(scale, interval, draw_info)
        display.display(display.HTML(data=anime.to_jshtml()))
        logger.info("create display({:.1f}s)".format(time.time() - t0))
