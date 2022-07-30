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

    def __post_init__(self):
        self.frames = []

        self.fig = None
        self.ax = None

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
        env_text = env.render_terminal(return_text=True)
        rl_text = worker.render_terminal(env, return_text=True)
        info_text = f"### {env.step_num}, action {action}, rewards {env.step_rewards}"
        if env.done:
            info_text += f", done({env.done_reason})"
        info_text += f", next {env.next_player_index}"
        if is_skip:
            info_text += "(skip frame)"
        info_text += f"\nenv   {env.info}"
        info_text += f"\nwork{worker_idx: <2d}{worker.info}"

        if self.render_terminal:
            print(info_text)
            if env_text != "":
                print(env_text)
            if rl_text != "" and not is_skip:
                print(rl_text)

        if self.render_window or self.enable_animation:
            info_image = self._text_to_image(info_text)
            rl_image = self._text_to_image(rl_text)
            try:
                env_image = env.render_rgb_array()
            except NotImplementedError:
                env_image = self._text_to_image(env_text)

        if self.render_window:
            image = self._create_image(info_image, env_image, rl_image)
            self._draw_window_image(image)

        if self.enable_animation:
            self.frames.append(
                {
                    "info_image": info_image,
                    "env_image": env_image,
                    "rl_image": rl_image,
                }
            )

        if self.step_stop:
            input("Enter to continue:")

    # -----------------------------------------------
    def _text_to_image(self, text: str) -> np.ndarray:
        text = text.encode("utf-8").decode("latin-1")
        # TODO: 文字の大きさ
        # font = PIL.ImageFont.truetype("meiryo.ttc", size=12)

        canvas_size = (640, 480)
        img = PIL.Image.new("RGB", canvas_size)
        draw = PIL.ImageDraw.Draw(img)
        text_width, text_height = draw.multiline_textsize(text)
        # text_width, text_height = draw.multiline_textsize(text, font=font)

        canvas_size = (text_width, text_height)
        background_rgb = (0, 0, 0)
        text_rgb = (255, 255, 255)
        img = PIL.Image.new("RGB", canvas_size, background_rgb)
        draw = PIL.ImageDraw.Draw(img)
        # draw.text((0, 0), text, fill=text_rgb, font=font)
        draw.text((0, 0), text, fill=text_rgb)
        img = np.array(img, dtype=np.uint8)

        return img

    def _create_image(self, info_image: np.ndarray, env_image: np.ndarray, rl_image: np.ndarray):
        # 大きいほうに合わせる(余白は埋める)
        maxw = max(info_image.shape[1], rl_image.shape[1])
        info_image = cv2.copyMakeBorder(
            info_image, 0, 0, 0, maxw - info_image.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        rl_image = cv2.copyMakeBorder(
            rl_image, 0, 0, 0, maxw - rl_image.shape[1], cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        img1 = cv2.vconcat([info_image, rl_image])  # 縦連結

        maxh = max(env_image.shape[0], img1.shape[0])
        env_image = cv2.copyMakeBorder(
            env_image, 0, maxh - env_image.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        img1 = cv2.copyMakeBorder(img1, 0, maxh - img1.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img2 = cv2.hconcat([env_image, img1])  # 横連結

        return img2

    def _draw_window_image(self, image):
        # 初回
        if self.fig is None:
            plt.ion()  # インタラクティブモードをオン
            self.fig, self.ax = plt.subplots()

        self.ax.imshow(image)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    # -----------------------------------------------

    def create_anime(
        self,
        scale: float = 1.0,
        fps: float = 60,
        gray: bool = False,
        resize: Optional[Tuple[int, int]] = None,
        draw_info: bool = False,
    ):
        if len(self.frames) == 0:
            return None
        t0 = time.time()
        interval = 1000 / fps
        fig = plt.figure(figsize=(6.4 * scale, 4.8 * scale), tight_layout=dict(pad=0))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        images = []
        for f in self.frames:
            env_img = f["env_image"]
            info_img = f["info_image"]
            rl_img = f["rl_image"]

            if resize is not None:
                env_img = cv2.resize(env_img, resize)
            if gray:
                env_img = cv2.cvtColor(env_img, cv2.COLOR_RGB2GRAY)

            if draw_info:
                img = self._create_image(info_img, env_img, rl_img)
            else:
                img = env_img
            images.append([ax.imshow(img, animated=True)])
        anime = ArtistAnimation(fig, images, interval=interval, repeat=False)
        # plt.close(fig)  # notebook で画像が残るので出来ればcloseしたいけど、closeするとgym側でバグる
        logger.debug("create animation({:.1f}s)".format(time.time() - t0))
        return anime

    def display(self, scale: float = 1.0, fps: float = 60) -> None:
        if len(self.frames) == 0:
            return

        from IPython import display

        t0 = time.time()
        anime = self.create_anime(scale, fps)
        display.display(display.HTML(data=anime.to_jshtml()))
        logger.info("create display({:.1f}s)".format(time.time() - t0))
