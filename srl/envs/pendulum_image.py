import logging

import numpy as np
from srl.utils.common import is_package_installed

logger = logging.getLogger(__name__)


if is_package_installed("gym") and is_package_installed("pygame"):

    import gym
    import gym.envs.registration
    import gym.spaces
    from gym.envs.classic_control.pendulum import PendulumEnv

    try:
        import PIL.Image
        import PIL.ImageDraw
    except ImportError:
        pass

    gym.envs.registration.register(
        id="PendulumImage-v0",
        entry_point=__name__ + ":PendulumImage",
        max_episode_steps=200,
    )

    class PendulumImage(PendulumEnv):
        def __init__(self):
            super().__init__()

            self.image_size = 84

            # observation_space を変更
            # 正規化後の画像を返す(0～1)
            self.observation_space = gym.spaces.Box(
                low=0,
                high=1,
                shape=(self.image_size, self.image_size),
                dtype=np.float32,
            )

        def _get_obs(self):
            # 上書きして画像を返す
            state = super()._get_obs()
            return self._get_rgb_state(state)

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
            img_arr = np.asarray(pilImg, dtype=np.float32)
            img_arr /= 255.0

            return img_arr
