import random
from typing import Any, Optional

import numpy as np
import tensorflow as tf

from srl.base.rl.algorithms.base_dqn import RLWorker

from .config import Config
from .memory import Memory
from .parameter import Parameter


class Worker(RLWorker[Config, Parameter, Memory]):
    def on_setup(self, worker, context):
        self.screen = None

    def on_reset(self, worker):
        self.hc = self.parameter.actor_critic.get_initial_state(1)

        for _ in range(max(self.config.denoiser_cfg.num_steps_conditioning, self.config.burnin)):
            worker.add_tracking(
                {
                    "state": worker.state,
                    "action": random.randint(0, self.config.action_space.n - 1),
                    "reward": [0, 1, 0],
                    "terminated": [1, 0],
                }
            )
        if self.rendering:
            self.prev_hc = self.parameter.actor_critic.get_initial_state(1)
            self.hc_rewend = self.parameter.reward_end_model.get_initial_state(1)

    def policy(self, worker) -> Any:
        if self.rollout:
            return self.sample_action()

        if self.rendering:
            self.prev_hc = self.hc

        obs = worker.state[np.newaxis, ...]
        act_dist, v, self.hc = self.parameter.actor_critic(obs, hc=self.hc)
        act = act_dist.sample()

        return int(act.numpy()[0][0])

    def on_step(self, worker):
        if not self.training and not self.rendering:
            return

        # clip {-1,0,1} and categorical
        if worker.reward < 0:
            reward = [1, 0, 0]
        elif worker.reward > 0:
            reward = [0, 0, 1]
        else:
            reward = [0, 1, 0]

        # done categorical
        teminated = [0, 1] if worker.terminated else [1, 0]

        worker.add_tracking(
            {
                "state": worker.next_state,
                "action": worker.action,
                "reward": reward,
                "terminated": teminated,
            }
        )

        if worker.done:
            for _ in range(self.config.horizon):
                worker.add_tracking(
                    {
                        "state": worker.next_state,
                        "action": random.randint(0, self.config.action_space.n - 1),
                        "reward": reward,
                        "terminated": [0, 1],
                    }
                )

            if self.training:
                self.memory.add(
                    worker.get_trackings(
                        [
                            "state",
                            "action",
                            "reward",
                            "terminated",
                        ]
                    )
                )

    def render_rgb_array(self, worker, **kwargs) -> Optional[np.ndarray]:
        from srl.utils.pygame_wrapper import PygameScreen

        # --- draw
        IMG_W = 128
        IMG_H = 128
        STR_H = 20
        PADDING = 4
        ACT_NUM = 5
        DIFF_NUM = 5
        WIDTH = (IMG_W + PADDING) * ACT_NUM + 5
        HEIGHT = (IMG_H + PADDING) * DIFF_NUM + STR_H * 7 + 5

        if self.screen is None:
            self.screen = PygameScreen(WIDTH, HEIGHT)
        self.screen.draw_fill((0, 0, 0))

        # --- act, v
        obs = worker.state[np.newaxis, ...]
        act_dist, v, _ = self.parameter.actor_critic(obs, hc=self.prev_hc)
        act_probs = act_dist.probs()[0].numpy()
        v = v[0][0]

        self.screen.draw_text(0, 0, f"v={v:.4f}", color=(255, 255, 255))

        # 横にアクション後の結果を表示
        hc_r = self.hc_rewend
        for a in range(min(ACT_NUM, self.config.action_space.n)):
            x = (IMG_W + PADDING) * a
            y = STR_H + PADDING

            self.screen.draw_text(x, y, f"{worker.env.action_to_str(a)}({act_probs[a] * 100:.2f}%) ", color=(255, 255, 255))
            y += STR_H + PADDING

            # --- next step
            size = self.config.denoiser_cfg.num_steps_conditioning
            recent_obs = np.asarray(worker.get_tracking("state", size), np.float32)[np.newaxis, ...]
            recent_act = worker.get_tracking("action", size - 1) + [a]
            recent_act = np.asarray(recent_act, np.float32)[np.newaxis, ...]
            next_obs, obs_history = self.parameter.sampler.sample(recent_obs, recent_act)
            next_img = self.config.decode_img(next_obs[0])
            diff_history_imgs = [self.config.decode_img(o[0]) for o in obs_history]

            # --- reward, done
            obs = worker.state[np.newaxis, np.newaxis, ...]
            act = np.array(a)[np.newaxis, np.newaxis, ...]
            r, d, hc_r2 = self.parameter.reward_end_model([obs, act, next_obs], hc_r)
            r = tf.nn.softmax(r).numpy()[0][0]
            reward = r[0] * -1 + r[2] * 1
            d = tf.nn.softmax(d).numpy()[0][0][1]
            done = d

            if worker.action == a:
                self.hc_rewend = hc_r2

            self.screen.draw_text(x, y, f"r={reward:.5f}", color=(255, 255, 255))
            y += STR_H + PADDING
            self.screen.draw_text(x, y, f"d={done * 100:.2f}%", color=(255, 255, 255))
            y += STR_H + PADDING

            # --- diffision draw
            self.screen.draw_image_rgb_array(x, y, next_img, (IMG_W, IMG_W), not self.config.img_color)
            y += IMG_H + PADDING
            diff_history_imgs = list(reversed(diff_history_imgs))
            for i in range(min(len(diff_history_imgs), DIFF_NUM)):
                img = diff_history_imgs[i]
                self.screen.draw_image_rgb_array(x, y, img, (IMG_W, IMG_W), not self.config.img_color)
                y += IMG_H + PADDING

        return self.screen.get_rgb_array()
