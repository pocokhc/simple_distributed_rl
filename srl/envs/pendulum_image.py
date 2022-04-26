import logging
import pickle
from dataclasses import dataclass
from typing import Any, Tuple

import gym
import gym.spaces
import numpy as np
from srl.base.define import EnvObservationType
from srl.base.env import registration
from srl.base.env.genre.singleplay import SingleActionContinuous

try:
    import PIL.Image
    import PIL.ImageDraw
except ModuleNotFoundError:
    pass


logger = logging.getLogger(__name__)

registration.register(
    id="PendulumImage",
    entry_point=__name__ + ":PendulumImage",
    kwargs={},
)


@dataclass
class PendulumImage(SingleActionContinuous):

    image_size = 84

    def __post_init__(self):
        self.env = gym.make("Pendulum-v1")

        # 正規化後の画像を返す(0～1)
        self._observation_space = gym.spaces.Box(low=0, high=1, shape=(self.image_size, self.image_size))

    @property
    def action_space(self) -> gym.spaces.Space:
        return self.env.action_space

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def observation_type(self) -> EnvObservationType:
        return EnvObservationType.GRAY_2ch

    @property
    def max_episode_steps(self) -> int:
        return 200

    def reset_single(self) -> Any:
        return self._get_rgb_state(self.env.reset())

    def step_single(self, action: Any) -> Tuple[Any, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        return self._get_rgb_state(state), float(reward), done, info

    # 状態（x,y座標）から対応画像を描画する関数
    def _get_rgb_state(self, state):

        h_size = self.image_size / 2.0

        img = PIL.Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        dr = PIL.ImageDraw.Draw(img)

        # 棒の長さ
        L = self.image_size / 4.0 * 3.0 / 2.0

        # 棒のラインの描写
        dr.line(((h_size - L * state[1], h_size - L * state[0]), (h_size, h_size)), (0, 0, 0), 1)

        # 棒の中心の円を描写（それっぽくしてみた）
        buff = self.image_size / 32.0
        dr.ellipse(
            ((h_size - buff, h_size - buff), (h_size + buff, h_size + buff)), outline=(0, 0, 0), fill=(255, 0, 0)
        )

        # 画像の一次元化（GrayScale化）とarrayへの変換
        pilImg = img.convert("L")
        img_arr = np.asarray(pilImg)

        return img_arr

    def render_terminal(self) -> None:
        print(self.env.render("ansi"))

    def render_gui(self) -> None:
        self.env.render("human")

    def render_rgb_array(self) -> np.ndarray:
        return np.asarray(self.env.render("rgb_array"))

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, state: Any) -> None:
        self.env = pickle.loads(state)
