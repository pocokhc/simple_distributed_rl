import logging
import pickle
from dataclasses import dataclass
from typing import Any

import gym
import gym.envs.registration
import gym.spaces
import numpy as np
from PIL import Image, ImageDraw
from srl.base.define import EnvObservationType
from srl.base.env import EnvBase

logger = logging.getLogger(__name__)

gym.envs.registration.register(
    id="PendulumImage-v0",
    entry_point=__name__ + ":PendulumImage",
    kwargs={},
)


@dataclass
class PendulumImage(EnvBase):

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

    def reset(self) -> Any:
        return self._get_rgb_state(self.env.reset())

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        state, reward, done, info = self.env.step(action)
        return self._get_rgb_state(state), reward, done, info

    # 状態（x,y座標）から対応画像を描画する関数
    def _get_rgb_state(self, state):

        h_size = self.image_size / 2.0

        img = Image.new("RGB", (self.image_size, self.image_size), (255, 255, 255))
        dr = ImageDraw.Draw(img)

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

        # 画像の規格化
        img_arr = img_arr / 255.0
        # img_arr = img_arr[..., np.newaxis]
        return img_arr

    def fetch_valid_actions(self):
        return None

    def render(self, mode: str = "human") -> Any:  # super
        return self.env.render(mode)

    def backup(self) -> Any:
        return pickle.dumps(self.env)

    def restore(self, state: Any) -> None:
        self.env = pickle.loads(state)
