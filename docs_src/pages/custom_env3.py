import numpy as np

import srl
from srl import runner
from srl.algorithms import ql
from srl.envs import sample_env  # noqa F401

# Q学習
config = runner.Config(srl.EnvConfig("SampleEnv"), rl_config=ql.Config())

# 学習
parameter, memory, history = runner.train(config, timeout=10)

# 評価
rewards = runner.evaluate(config, parameter, max_episodes=100)
print("100エピソードの平均結果", np.mean(rewards))

# 可視化
reward = runner.render_terminal(config, parameter)
print(reward)

# animation
render = runner.animation(config, parameter)
render.create_anime(scale=3).save("_SampleEnv.gif")
# render.display(scale=3)  # for notebook
