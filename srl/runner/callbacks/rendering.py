import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
from matplotlib.animation import ArtistAnimation
from srl.base.env.base import EnvRun
from srl.base.rl.base import WorkerRun
from srl.runner.callback import Callback

logger = logging.getLogger(__name__)


@dataclass
class Rendering(Callback):

    render_terminal: bool = True
    render_window: bool = False
    step_stop: bool = False
    use_skip_step: bool = True
    enable_animation: bool = False

    # windows     : C:\Windows\Fonts
    # max         : /System/Library/Fonts
    # linux       : /usr/local/share/fonts/
    # google colab: /usr/share/fonts/

    font_name: str = ""
    font_size: int = 12

    def __post_init__(self):
        self.frames = []
        self.info_maxw = 0
        self.info_maxh = 0
        self.env_maxw = 0
        self.env_maxh = 0

        self.fig = None
        self.ax = None

        self.rl_text = ""
        self.rl_img = None

    def on_step_begin(
        self,
        env: EnvRun,
        action,
        worker_idx,
        workers,
        **kwargs,
    ) -> None:
        self._render_step(env, workers[worker_idx], worker_idx, action)

    def on_episode_end(
        self,
        env: EnvRun,
        action,
        worker_idx,
        workers,
        **kwargs,
    ) -> None:
        self._render_step(env, workers[worker_idx], worker_idx, action)

    def on_skip_step(
        self,
        env: EnvRun,
        action,
        worker_idx,
        workers,
        **kwargs,
    ):
        if not self.use_skip_step:
            return
        self._render_step(env, workers[worker_idx], worker_idx, action, True)

    def _render_step(
        self,
        env: EnvRun,
        worker: WorkerRun,
        worker_idx: int,
        action,
        is_skip=False,
    ):
        # env text
        env_text = env.render_terminal(return_text=True)

        # rl text
        if is_skip:
            rl_text = ""
        else:
            self.rl_text = worker.render_terminal(env, return_text=True)
            rl_text = self.rl_text

        # info text
        info_text = f"### {env.step_num}, action {action}, rewards {env.step_rewards}"
        if env.done:
            info_text += f", done({env.done_reason})"
        info_text += f", next {env.next_player_index}"
        if is_skip:
            info_text += "(skip frame)"
        info_text += f"\nenv   {env.info}"
        info_text += f"\nwork{worker_idx: <2d}{worker.info}"

        # --- render_terminal
        if self.render_terminal:
            print(info_text)
            if env_text != "":
                print(env_text)
            if rl_text != "" and not env.done:
                print(rl_text)

        # --- image
        if self.render_window or self.enable_animation:
            info_img = self._text_to_image(info_text)
            try:
                env_img = env.render_rgb_array()
            except NotImplementedError:
                env_img = self._text_to_image(env_text)
            if not is_skip:
                try:
                    self.rl_img = worker.render_rgb_array(env)
                except NotImplementedError:
                    self.rl_img = self._text_to_image(self.rl_text)

            self.info_maxw = max(max(self.info_maxw, info_img.shape[1]), self.rl_img.shape[1])
            self.info_maxh = max(self.info_maxh, info_img.shape[0] + self.rl_img.shape[0])
            self.env_maxw = max(self.env_maxw, env_img.shape[1])
            self.env_maxh = max(self.env_maxh, env_img.shape[0])

        if self.render_window:
            if self.step_stop:
                img = self._create_image(info_img, env_img, self.rl_img)
                self._draw_window_image(img)
            else:
                self._draw_window_image(env_img)

        if self.enable_animation:
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
    def _text_to_image(self, text: str) -> np.ndarray:
        text = text.encode("utf-8").decode("latin-1")
        if self.font_name == "":
            font = None
        else:
            font = PIL.ImageFont.truetype(self.font_name, size=self.font_size)
        canvas_size = (640, 480)
        img = PIL.Image.new("RGB", canvas_size)
        draw = PIL.ImageDraw.Draw(img)
        text_width, text_height = draw.multiline_textsize(text, font=font)

        canvas_size = (text_width, text_height)
        background_rgb = (0, 0, 0)
        text_rgb = (255, 255, 255)
        img = PIL.Image.new("RGB", canvas_size, background_rgb)
        draw = PIL.ImageDraw.Draw(img)
        draw.text((0, 0), text, fill=text_rgb, font=font)
        img = np.array(img, dtype=np.uint8)

        return img

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

    def _draw_window_image(self, image):
        """matplotlibを採用"""

        if self.fig is None:
            plt.ion()  # インタラクティブモードをオン
            self.fig, self.ax = plt.subplots()
            self.ax.axis("off")

        self.ax.imshow(image)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # -----------------------------------------------

    def create_anime(
        self,
        scale: float = 1.0,
        interval: float = 1000 / 60,
        gray: bool = False,
        resize: Optional[Tuple[int, int]] = None,
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

            if resize is not None:
                env_img = cv2.resize(env_img, resize)
            if gray:
                env_img = cv2.cvtColor(env_img, cv2.COLOR_RGB2GRAY)
                env_img = np.stack((env_img,) * 3, -1)

            if draw_info:
                img = self._create_image(info_img, env_img, rl_img)
            else:
                img = env_img
            images.append(img)
            maxw = max(maxw, img.shape[1])
            maxh = max(maxh, img.shape[0])

        fig_dpi = 100
        # inch = pixel / dpi
        fig = plt.figure(
            dpi=fig_dpi, figsize=(scale * maxw / fig_dpi, scale * maxh / fig_dpi), tight_layout=dict(pad=0)
        )
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        images = [[ax.imshow(img, animated=True)] for img in images]
        anime = ArtistAnimation(fig, images, interval=interval, repeat=False)
        # plt.close(fig)  # notebook で画像が残るので出来ればcloseしたいけど、closeするとgym側でバグる
        logger.debug("create animation({:.1f}s)".format(time.time() - t0))
        return anime

    def display(
        self,
        scale: float = 1.0,
        interval: int = 1000 / 60,
        gray: bool = False,
        resize: Optional[Tuple[int, int]] = None,
        draw_info: bool = False,
    ) -> None:
        if len(self.frames) == 0:
            return

        from IPython import display

        t0 = time.time()
        anime = self.create_anime(scale, interval, gray, resize, draw_info)
        display.display(display.HTML(data=anime.to_jshtml()))
        logger.info("create display({:.1f}s)".format(time.time() - t0))
