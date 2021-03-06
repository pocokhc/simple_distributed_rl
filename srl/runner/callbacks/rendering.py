import io
import logging
import sys
import time
from dataclasses import dataclass
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import PIL.ImageDraw
from matplotlib.animation import ArtistAnimation
from srl.base.define import RenderType
from srl.base.env.base import EnvRun
from srl.runner.callback import Callback

logger = logging.getLogger(__name__)


@dataclass
class Rendering(Callback):

    mode: Union[str, RenderType] = RenderType.Terminal
    step_stop: bool = False
    enable_animation: bool = False

    def __post_init__(self):
        self.frames = []

        if isinstance(self.mode, str):
            for t in RenderType:
                if t.value == self.mode:
                    self.mode = t
                    break
            else:
                self.mode = RenderType.NONE

    def on_episode_begin(self, env: EnvRun, **kwargs):
        if self.mode == RenderType.Terminal:
            print("### 0")
        env.render(self.mode)

        if self.enable_animation:
            self._add_image(env)

    def on_step_begin(
        self,
        env: EnvRun,
        worker_idx,
        workers,
        **kwargs,
    ) -> None:
        if self.mode == RenderType.Terminal:
            workers[worker_idx].render(env)

        if self.step_stop:
            input("Enter to continue:")

    def on_step_end(
        self,
        env: EnvRun,
        action,
        worker_idx,
        workers,
        train_info,
        **kwargs,
    ):

        if self.mode == RenderType.Terminal:
            print(
                "### {}, action {}, rewards {}, done {}({}), next {}".format(
                    env.step_num,
                    action,
                    env.step_rewards,
                    env.done,
                    env.done_reason,
                    env.next_player_index,
                )
            )
            env.render(self.mode)
            print(f"env_info  : {env.info}")
            print(f"work_info {worker_idx}: {workers[worker_idx].info}")
            print(f"train_info: {train_info}")
        else:
            env.render(self.mode)

        if self.enable_animation:
            self._add_image(env)

    def on_skip_step(self, env: EnvRun, **kwargs):
        env.render(self.mode)

        if self.enable_animation:
            self._add_image(env)

    # -----------------------------
    def _add_image(self, env: EnvRun):
        try:
            self.frames.append(env.render(RenderType.RGB_Array, is_except=True))
        except NotImplementedError:
            # --- print????????????
            text = ""
            _stdout = sys.stdout
            try:
                sys.stdout = io.StringIO()
                env.render(RenderType.Terminal)
                text = sys.stdout.getvalue()
            finally:
                try:
                    sys.stdout.close()
                except Exception:
                    pass
                sys.stdout = _stdout

            canvas_size = (300, 300)
            img = PIL.Image.new("RGB", canvas_size)
            draw = PIL.ImageDraw.Draw(img)
            text_width, text_height = draw.multiline_textsize(text)

            canvas_size = (text_width, text_height)
            background_rgb = (255, 255, 255)
            text_rgb = (0, 0, 0)
            img = PIL.Image.new("RGB", canvas_size, background_rgb)
            draw = PIL.ImageDraw.Draw(img)
            draw.text((0, 0), text, fill=text_rgb)
            self.frames.append(np.asarray(img))

    def create_anime(self, scale: float = 1.0, fps: float = 60):
        if len(self.frames) == 0:
            return None
        t0 = time.time()
        interval = 1000 / fps
        fig = plt.figure(figsize=(6.4 * scale, 4.8 * scale))
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        images = []
        for f in self.frames:
            images.append([ax.imshow(f, animated=True)])
        anime = ArtistAnimation(fig, images, interval=interval, repeat=False)
        # plt.close(fig)  # notebook ????????????????????????????????????close??????????????????close?????????gym???????????????
        logger.info("create animation({:.1f}s)".format(time.time() - t0))
        return anime

    def display(self, scale: float = 1.0, fps: float = 60) -> None:
        if len(self.frames) == 0:
            return

        from IPython import display

        t0 = time.time()
        anime = self.create_anime(scale, fps)
        display.display(display.HTML(data=anime.to_jshtml()))
        logger.info("create display({:.1f}s)".format(time.time() - t0))
