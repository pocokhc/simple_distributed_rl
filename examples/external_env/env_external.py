"""
外部で動作する環境を想定
"""

import random
from typing import Callable


class ExternalEnv:
    def __init__(self):
        self.pos = 0
        self.reward = 0
        self.done = False

    def step(self, action):
        # posが5以下なら1で終了
        # posが-5以下なら-1で終了
        if action == 0:
            self.pos -= 1
        else:
            self.pos += 1

        if self.pos >= 5:
            self.reward = 1
            self.done = True
        if self.pos <= -5:
            self.reward = -1
            self.done = True


def run_external_env(agent: Callable[[int, int], int]):
    """
    ユーザが定義したagent関数を元に実際にシミュレーションする
    agent関数の引数は[step, state]で戻り値はactionを想定
    """

    for episode in range(5):
        env = ExternalEnv()
        act_history = []
        pos_history = [env.pos]
        for step in range(30):
            action = agent(step, env.pos)
            act_history.append(action)
            env.step(action)
            pos_history.append(env.pos)
            if env.done:
                break
        print(f"--- {episode} ---")
        print(f"reward: {env.reward}")
        print(f"action: {act_history}")
        print(f"state : {pos_history}")


if __name__ == "__main__":

    def sample_agent(step: int, state: int) -> int:
        return random.randint(0, 1)

    run_external_env(sample_agent)
