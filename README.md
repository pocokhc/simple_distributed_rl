(まだ作成中です)(WIP)

# Simple Reinforcement Learning (シンプルな強化学習)

シンプルな強化学習フレームワークを目指して作成しました。
以下の特徴があります。

+ 分散強化学習の標準サポート
+ 経験メモリーの標準サポート
+ カスタマイズ可能な環境
+ カスタマイズ可能な強化学習アルゴリズム


# Install

今のところは github からの pip install を想定しています。

``` bash
pip install git+https://github.com/pocokhc/simple_rl
```


# Usage
## Sequential Training

``` python
from srl import rl
from srl.runner import sequence
from srl.runner.callbacks import PrintProgress, RenderingEpisode

# config
config = sequence.Config(
    env_name="FrozenLake-v1",  # select env
    rl_config=rl.ql.Config(),  # select rl & rl config
    memory_config=rl.memory.replay_memory.Config(),  # select memory & memory config
)

# train
config.set_play_config(timeout=60, training=True, callbacks=[PrintProgress()])
episode_rewards, parameter, memory = sequence.play(config)

# test
config.set_play_config(max_episodes=10, callbacks=[PrintProgress()])
sequence.play(config, parameter)

# rendering
config.set_play_config(max_episodes=1, callbacks=[RenderingEpisode()])
sequence.play(config, parameter)
```

## Distribute Training

``` python
from srl import rl
from srl.runner import sequence, mp
from srl.runner.callbacks import PrintProgress, RenderingEpisode

# config
config = sequence.Config(
    env_name="FrozenLake-v1",  # select env
    rl_config=rl.ql.Config(),  # select rl & rl config
    memory_config=rl.memory.replay_memory.Config(),  # select memory & memory config
)

# train
mp_config = mp.Config(worker_num=2)  # select distribute config
mp_config.set_train_config(timeout=60, callbacks=[TrainFileLogger(enable_log=False, enable_checkpoint=False)])
parameter = mp.train(config, mp_config)

# test
config.set_play_config(max_episodes=10, callbacks=[PrintProgress()])
sequence.play(config, parameter)

# rendering
config.set_play_config(max_episodes=1, callbacks=[RenderingEpisode()])
sequence.play(config, parameter)
```

# Custome

実装例は以下のファイルを参照してください。

|path                      |   |
|--------------------------|---|
|examples/env/my_env_gym.py|GymEnvの実装例|
|examples/env/my_env.py    |本ライブラリ用のEnvの実装例|
|examples/rl/my_rl.py      |アルゴリズムの実装例|
|examples/sample_custom.py |自作環境を使った実行例|




# 実装アルゴリズム

TODO

|||
|---|---|
|a|a|





# Other Info


* Sequence flow

![sequence diagram](diagrams/sync_flow.png)


* Distribute flow

![sequence diagram](diagrams/runner_mp_flow.png)
![sequence diagram](diagrams/runner_mp_flow_trainer.png)
![sequence diagram](diagrams/runner_mp_flow_worker.png)

* Class diagram

![sequence diagram](diagrams/class.png)



