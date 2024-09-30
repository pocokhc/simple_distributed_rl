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

logger = logging.getLogger(__name__)


@dataclass
class Rendering(RunCallback):
    mode: Union[str, RenderModes] = RenderModes.none
    kwargs: dict = field(default_factory=lambda: {})
    step_stop: bool = False
    render_skip_step: bool = True
    render_worker: int = 0
    render_add_rl_terminal: bool = True
    render_add_rl_rgb: bool = True
    render_add_rl_state: bool = True
    render_add_info_text: bool = True

    def __post_init__(self):
        self.frames = []
        self.img_maxw = 0
        self.img_maxh = 0
        self.render_interval = -1
        self.mode = RenderModes.from_str(self.mode)

    def on_episodes_begin(self, context: "RunContext", state: "RunStateActor", **kwargs) -> None:
        self.render_interval = state.env.get_render_interval()

    def on_step_begin(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        self._render(context, state)

        if self.step_stop:
            input("Enter to continue:")

    def on_skip_step(self, context: RunContext, state: RunStateActor, **kwargs):
        if not self.render_skip_step:
            return
        self._render(context, state, skip_step=True)

    def on_episode_end(self, context: RunContext, state: RunStateActor) -> None:
        self._render(context, state)

    def on_episodes_end(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        if self.step_stop:
            input("Enter to continue:")

    def _render(self, context: RunContext, state: RunStateActor, skip_step=False):
        env = state.env

        # --- info text
        action = state.action
        worker_idx = state.worker_idx
        worker: WorkerRun = state.workers[worker_idx]
        info_text = f"### {env.step_num}"
        info_text += "(skip frame)" if skip_step else ""
        info_text += f", next {env.next_player}" if env.player_num > 1 else ""
        info_text += f", done({env.done_reason})" if env.done else ""
        _s = env.observation_space.to_str(env.state)
        _s = (_s[:40] + "...") if len(_s) > 40 else _s
        info_text += f"\nstate: {_s}"
        if isinstance(action, float):
            a1 = f"{action:.3f}"
        else:
            a1 = f"{action}"
        a2 = env.action_to_str(action)
        if a1 != a2:
            action = f"{a1}({a2})"
        info_text += f"\naction : {action}"
        info_text += "\nrewards:[" + ",".join([f"{r:.3f}" for r in env.rewards]) + "]"
        info_text += f"\nenv   {env.info}"
        info_text += f"\nwork{worker_idx: <2d}{worker.info}"

        # --- render info text
        if self.mode == RenderModes.terminal:
            print(info_text)

        if self.mode == RenderModes.rgb_array:
            worker = state.workers[self.render_worker]

            img = worker.create_render_image(
                add_terminal=self.render_add_rl_terminal,
                add_rgb_array=self.render_add_rl_rgb,
                add_rl_state=self.render_add_rl_state,
                info_text=info_text if self.render_add_info_text else "",
            )
            self.img_maxw = max(self.img_maxw, img.shape[1])
            self.img_maxh = max(self.img_maxh, img.shape[0])
            self.frames.append(img)

    # -----------------------------------------------

    def create_images(self, scale: float = 1.0) -> List[np.ndarray]:
        import cv2

        assert len(self.frames) > 0
        t0 = time.time()

        # サイズを合わせる
        imgs = []
        for img in self.frames:
            right = self.img_maxw - img.shape[1]
            bottom = self.img_maxh - img.shape[0]
            img = cv2.copyMakeBorder(img, 0, bottom, 0, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            if scale != 1.0:
                new_width = int(img.shape[1] * scale)
                new_height = int(img.shape[0] * scale)
                img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            imgs.append(img)

        logger.info(f"image created(frames: {len(self.frames)}, create time {time.time() - t0:.1f}s)")
        return imgs

    def save_gif(
        self,
        path: str,
        interval: float = -1,  # ms
        scale: float = 1.0,
    ):
        from PIL import Image

        if interval <= 0:
            interval = self.render_interval
        if interval <= 0:
            interval = 1000 / 60

        if path[-4:] != ".gif":
            path += ".gif"

        t0 = time.time()
        images = self.create_images(scale)
        image = [Image.fromarray(img_array) for img_array in images]
        image[0].save(
            path,
            save_all=True,
            append_images=image[1:],
            optimize=False,
            duration=interval,
            loop=0,
        )
        logger.info(f"save gif: interval {interval:.1f}ms, save time {time.time() - t0:.1f}s, {path}")

    def save_avi(
        self,
        path: str,
        interval: float = -1,  # ms
        scale: float = 1.0,
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

        images = self.create_images(scale)
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

    def to_jshtml(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
    ):
        assert len(self.frames) > 0

        import matplotlib.pyplot as plt
        from matplotlib.animation import ArtistAnimation

        images = self.create_images()
        maxw = images[0].shape[1]
        maxh = images[0].shape[0]

        t0 = time.time()

        # --- interval
        if interval <= 0:
            interval = self.render_interval
        if interval <= 0:
            interval = 1000 / 60

        # --- size (inch = pixel / dpi)
        plt.clf()
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
        html = anime.to_jshtml()
        logger.info(f"animation created(interval: {interval:.1f}ms, create time {time.time() - t0:.1f}s)")

        # なぜか2回やらないとウィンドウが残る
        plt.clf()
        plt.close()
        plt.clf()
        plt.close()

        return html

    def display(
        self,
        interval: float = -1,  # ms
        scale: float = 1.0,
    ) -> None:
        if len(self.frames) == 0:
            return

        from IPython import display  # type: ignore

        html = self.to_jshtml(interval=interval, scale=scale)
        display.display(display.HTML(data=html))  # type: ignore
