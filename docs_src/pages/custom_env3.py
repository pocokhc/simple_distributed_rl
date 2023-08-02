import numpy as np

import srl
from srl.algorithms import ql

# 実装したEnvファイルをimportし、registerに登録
from srl.envs import sample_env  # noqa F401

# Q学習
runner = srl.Runner(srl.EnvConfig("SampleEnv"), rl_config=ql.Config())

# 学習
runner.train(timeout=10)

# 評価
rewards = runner.evaluate(max_episodes=100)
print("100エピソードの平均結果", np.mean(rewards))

# 可視化
runner.render_terminal()

# animation
render = runner.animation_save_gif("_SampleEnv.gif", render_scale=3)
