import logging
import time
from dataclasses import dataclass, field
from typing import List, Union

import numpy as np

from srl.base.context import RunContext
from srl.base.define import RenderModes
from srl.base.rl.worker_run import WorkerRun
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor
from srl.utils.render_functions import text_to_rgb_array

logger = logging.getLogger(__name__)


@dataclass
class Rendering(RunCallback):
    mode: Union[str, RenderModes] = RenderModes.none
    kwargs: dict = field(default_factory=lambda: {})
    step_stop: bool = False
    render_skip_step: bool = True

    # render option
    render_interval: float = -1  # ms
    render_scale: float = 1.0
    font_name: str = ""
    font_size: int = 18

    def __post_init__(self):
        self.frames = []
        self.info_maxw = 0
        self.info_maxh = 0
        self.env_maxw = 0
        self.env_maxh = 0
        self.rl_maxw = 0
        self.rl_maxh = 0
        self.rl_state_maxw = 0
        self.rl_state_maxh = 0

        self.info_text = ""
        self.env_img = None
        self.rl_text = ""
        self.rl_img = None
        self.rl_state_image = None
        self.font = None

        self.mode = RenderModes.from_str(self.mode)

    def on_episodes_begin(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        self.render_interval = state.env.set_render_options(
            self.render_interval,
            self.render_scale,
            self.font_name,
            self.font_size,
        )

    def on_step_action_before(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        self._render_env(context, state)

    def on_step_begin(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        self._render_worker(context, state)
        self._add_image()

        if self.step_stop:
            input("Enter to continue:")

    def on_skip_step(self, context: RunContext, state: RunStateActor, **kwargs):
        if not self.render_skip_step:
            return
        self._render_env(context, state, True)
        self._add_image()

    def on_episode_end(self, context: RunContext, state: RunStateActor) -> None:
        self._render_env(context, state)
        self._add_image()

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        if self.step_stop:
            input("Enter to continue:")

    # -----------------------------------------------

    def _render_env(self, context: RunContext, state: RunStateActor, skip_step=False):
        env = state.env

        # --- info text
        action = state.action
        worker_idx = state.worker_idx
        worker: WorkerRun = state.workers[worker_idx]
        info_text = f"### {env.step_num}"
        if isinstance(action, float):
            a1 = f"{action:.3f}"
        else:
            a1 = f"{action}"
        a2 = env.action_to_str(action)
        if a1 != a2:
            action = f"{a1}({a2})"
        info_text += f", action {action}"
        info_text += ", rewards[" + ",".join([f"{r:.3f}" for r in env.step_rewards]) + "]"
        if env.done:
            info_text += f", done({env.done_reason})"
        if env.player_num > 1:
            info_text += f", next {env.next_player_index}"
        if skip_step:
            info_text += "(skip frame)"
        info_text += f"\nenv   {env.info}"
        info_text += f"\nwork{worker_idx: <2d}{worker.info}"
        self.info_text = info_text

        # --- render_terminal
        if self.mode == RenderModes.terminal:
            print(info_text)

        if self.mode == RenderModes.rgb_array:
            self.env_img = env.render_rgb_array(**self.kwargs)
            self.env_maxw = max(self.env_maxw, self.env_img.shape[1])
            self.env_maxh = max(self.env_maxh, self.env_img.shape[0])

    def _add_image(self):
        # --- rgb
        if self.mode == RenderModes.rgb_array:
            info_img = text_to_rgb_array(self.info_text)
            self.info_maxw = max(self.info_maxw, info_img.shape[1])
            self.info_maxh = max(self.info_maxh, info_img.shape[0])

            self.frames.append(
                {
                    "info_image": info_img,
                    "env_image": self.env_img,
                    "rl_image": self.rl_img,
                    "rl_state_image": self.rl_state_image,
                }
            )

    def _render_worker(self, context: RunContext, state: RunStateActor):
        worker = state.workers[state.worker_idx]

        # --- rgb
        if self.mode == RenderModes.rgb_array:
            self.rl_img = worker.render_rgb_array(**self.kwargs)
            self.rl_maxw = max(self.rl_maxw, self.rl_img.shape[1])
            self.rl_maxh = max(self.rl_maxh, self.rl_img.shape[0])

            # rlへの入力
            rl_state_img = worker.render_rl_image()
            if rl_state_img is not None:
                self.rl_state_image = rl_state_img
                self.rl_state_maxw = max(self.rl_state_maxw, self.rl_state_image.shape[1])
                self.rl_state_maxh = max(self.rl_state_maxh, self.rl_state_image.shape[0])

    # -----------------------------------------------
    def _create_image(self, frame):
        import cv2

        info_image = frame["info_image"]
        env_image = frame["env_image"]
        rl_image = frame["rl_image"]
        rl_state_image = frame["rl_state_image"]

        # --- 余白を追加
        padding = 2
        info_image = cv2.copyMakeBorder(
            info_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        if rl_image is not None:
            rl_image = cv2.copyMakeBorder(
                rl_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
        env_image = cv2.copyMakeBorder(
            env_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        if rl_state_image is not None:
            rl_state_image = cv2.copyMakeBorder(
                rl_state_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )

        # --- info + rl_image: 余白は右を埋める
        if rl_image is None:
            right_img = info_image
            right_maxh = self.info_maxh + padding * 2
        else:
            maxw = max(self.info_maxw + padding * 2, self.rl_maxw + padding * 2)
            info_w = maxw - info_image.shape[1]
            rl_w = maxw - rl_image.shape[1]
            info_image = cv2.copyMakeBorder(info_image, 0, 0, 0, info_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            rl_image = cv2.copyMakeBorder(rl_image, 0, 0, 0, rl_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            right_img = cv2.vconcat([info_image, rl_image])  # 縦連結
            right_maxh = self.info_maxh + self.rl_maxh + padding * 4

        # --- env + rl_state:
        if rl_state_image is None:
            left_img = env_image
            left_maxh = self.env_maxh + padding * 2
        else:
            # worker input image
            maxw = max(self.env_maxw + padding * 2, self.rl_state_maxw + padding * 2)
            env_w = maxw - env_image.shape[1]
            rl_state_w = maxw - rl_state_image.shape[1]
            env_image = cv2.copyMakeBorder(env_image, 0, 0, 0, env_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            rl_state_image = cv2.copyMakeBorder(
                rl_state_image, 0, 0, 0, rl_state_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )
            text = "RL"
            org = (0, 12)
            fontFace = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 0.5
            rl_state_image = cv2.putText(
                rl_state_image,
                text,
                org,
                fontFace,
                fontScale,
                color=(0, 0, 0),
                thickness=3,
            )
            rl_state_image = cv2.putText(
                rl_state_image,
                text,
                org,
                fontFace,
                fontScale,
                color=(255, 255, 255),
                thickness=1,
            )
            left_img = cv2.vconcat([env_image, rl_state_image])  # 縦連結
            left_maxh = self.env_maxh + self.rl_state_maxh + padding * 4

        # --- left_img + right_img: 余白は下を埋める
        maxh = max(left_maxh, right_maxh)
        left_h = maxh - left_img.shape[0]
        right_h = maxh - right_img.shape[0]
        left_img = cv2.copyMakeBorder(left_img, 0, left_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        right_img = cv2.copyMakeBorder(right_img, 0, right_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img = cv2.hconcat([left_img, right_img])  # 横連結  # type: ignore , MAT

        return img

    # -----------------------------------------------

    def create_images(self, draw_info: bool = True) -> List[np.ndarray]:
        import cv2

        assert len(self.frames) > 0
        t0 = time.time()

        # 最大サイズを探す
        maxw = 0
        maxh = 0
        images: List[np.ndarray] = []
        for f in self.frames:
            if draw_info:
                img = self._create_image(f)
            else:
                img = f["env_image"]
            if img is None:
                continue
            images.append(img)
            maxw = max(maxw, img.shape[1])
            maxh = max(maxh, img.shape[0])

        # サイズを合わせる
        for i in range(len(images)):
            right = maxw - images[i].shape[1]
            bottom = maxh - images[i].shape[0]
            images[i] = cv2.copyMakeBorder(images[i], 0, bottom, 0, right, cv2.BORDER_REPLICATE)

        logger.info(f"image created(frames: {len(self.frames)}, create time {time.time() - t0:.1f}s)")
        return images

    def create_anime(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        draw_info: bool = True,
    ):
        assert len(self.frames) > 0

        import matplotlib.pyplot as plt
        from matplotlib.animation import ArtistAnimation

        images = self.create_images(draw_info)
        maxw = images[0].shape[1]
        maxh = images[0].shape[0]

        t0 = time.time()

        # --- interval
        if interval <= 0:
            interval = self.render_interval
        if interval <= 0:
            interval = 1000 / 60

        # --- size (inch = pixel / dpi)
        fig_dpi = 100
        fig = plt.figure(
            dpi=fig_dpi,
            figsize=(scale * maxw / fig_dpi, scale * maxh / fig_dpi),
            tight_layout=dict(pad=0),
        )

        # --- animation
        ax = fig.add_subplot(1, 1, 1)
        ax.axis("off")
        images = [[ax.imshow(img, animated=True)] for img in images]
        anime = ArtistAnimation(fig, images, interval=interval, repeat=False)
        # plt.close(fig)  # notebook で画像が残るので出来ればcloseしたいけど、closeするとgym側でバグる

        logger.info(f"animation created(interval: {interval:.1f}ms, create time {time.time() - t0:.1f}s)")
        return anime

    def save_gif(
        self,
        path: str,
        interval: float = -1,  # ms
        draw_info: bool = True,
    ):
        from PIL import Image

        if interval <= 0:
            interval = self.render_interval
        if interval <= 0:
            interval = 1000 / 60

        if path[-4:] != ".gif":
            path += ".gif"

        t0 = time.time()
        images = self.create_images(draw_info)
        image = [Image.fromarray(img_array) for img_array in images]
        image[0].save(path, save_all=True, append_images=image[1:], optimize=False, duration=interval, loop=0)
        logger.info(f"save gif: interval {interval:.1f}ms, save time {time.time() - t0:.1f}s, {path}")

    def save_avi(
        self,
        path: str,
        interval: float = -1,  # ms
        draw_info: bool = True,
        codec: str = "XVID",
    ):
        import cv2

        if interval <= 0:
            interval = self.render_interval
        if interval <= 0:
            interval = 1000 / 60

        exts = {
            "XVID": ".avi",
            "MJPG": ".avi",
            "MP4V": ".mov",
            "H264": ".mp4",
        }
        if codec in exts:
            ext = exts[codec]
            if path[-4:] != ext:
                path += ext

        images = self.create_images(draw_info)
        images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in images]
        capSize = (images[0].shape[1], images[0].shape[0])
        fps = 1000 / interval

        t0 = time.time()
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(path, fourcc, fps, capSize)
        try:
            for img in images:
                writer.write(img)
        finally:
            writer.release()
        logger.info(
            f"save animation: codec {codec}, interval {interval:.1f}ms, save time {time.time() - t0:.1f}s, {path}"
        )

    def display(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
        draw_info: bool = True,
    ) -> None:
        if len(self.frames) == 0:
            return

        from IPython import display  # type: ignore

        t0 = time.time()
        anime = self.create_anime(interval=interval, scale=scale, draw_info=draw_info)
        display.display(display.HTML(data=anime.to_jshtml()))  # type: ignore
        logger.info("display created({:.1f}s)".format(time.time() - t0))
