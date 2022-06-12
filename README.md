
# Simple Distributed Reinforcement Learning (シンプルな分散強化学習)

シンプルな分散強化学習フレームワークを目指して作成しました。  
どちらかというと学習用フレームワークに近いかもしれません。  
以下の特徴があります。  

+ カスタマイズ可能な環境クラスの提供
+ カスタマイズ可能な強化学習アルゴリズムクラスの提供
+ 環境とアルゴリズム間のインタフェースの自動調整
+ 分散強化学習のサポート

また本フレームワークの解説は[Qiita記事](https://qiita.com/pocokhc/items/a2f1ba993c79fdbd4b4d)にて増やしていく予定なのでそちらも参照してください。


# Install

github からの pip install を想定しています。

``` bash
pip install git+https://github.com/pocokhc/simple_distributed_rl
```

or

``` bash
git clone https://github.com/pocokhc/simple_distributed_rl.git
cd simple_distributed_rl
pip install .

# run sample
python examples/minimum_runner.py
```

## Using library

+ numpy
+ tensorflow
+ tensorflow-addons
+ matplotlib
+ pillow
+ opencv-python
+ pandas


### Option library

+ gym の環境を使う場合に必要
  + gym
  + pygame
+ cpu/gpu 情報を記録したい場合に必要
  + psutil
  + pynvml

# Usage

``` python
import numpy as np

import srl
from srl.runner import mp, sequence

# ---------------------
# Configのパラメータは、引数補完または元コードを参照してください。
# For the parameters of Config, refer to the argument completion or the original code.
#
# srl.envs.Config   : Env Config
# srl.rl.xxx.Config : Algorithm hyperparameter
# sequence.Config   : Basic Running Config
# mp.Config         : Distributed training Config
# ---------------------


def main():
    # env config
    # (Run "pip install gym pygame" to use the gym environment)
    env_config = srl.envs.Config("FrozenLake-v1")

    # rl algorithm config
    rl_config = srl.rl.ql.Config()

    # running config
    config = sequence.Config(env_config, rl_config)

    # (option) load parameter
    # config.set_parameter_path(parameter_path="params.dat")

    # --- train
    if True:
        # sequence training
        parameter, remote_memory, history = sequence.train(config, timeout=60)
    else:
        # distributed training
        mp_config = mp.Config(actor_num=2)  # distributed config
        parameter, remote_memory, history = mp.train(config, mp_config, timeout=60)

    # (option) save parameter
    # parameter.save("params.dat")

    # --- evaluate
    rewards = sequence.evaluate(config, parameter, max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards)}")

    # --- rendering
    sequence.render(config, parameter)

    # --- animation
    reward, render = sequence.render(config, parameter, mode="", enable_animation=True)
    render.create_anime(fps=3).save("FrozenLake.gif")


if __name__ == "__main__":
    main()

```

![](FrozenLake.gif)

# Customize

オリジナル環境とアルゴリズムの作成例は以下のファイルを参考にしてください。

examples/custom_env.ipynb  
examples/custom_rl.ipynb  


# Algorithms

## ModelFree
### ValueBase

|Algorithm |Observation|Action|ProgressRate||
|----------|-----------|----------|------------|---|
|QL        |Discrete   |Discrete  |100%|Basic Q Learning|
|QL_agent57|Discrete   |Discrete  | 99%|QL + Agent57|
|DQN       |Continuous |Discrete  |100%||
|C51       |Continuous |Discrete  | 99%|Categorical DQN|[Paper](https://arxiv.org/abs/1707.06887)|
|Rainbow   |Continuous |Discrete  |100%||
|R2D2      |Continuous |Discrete  | 99%||[Paper](https://openreview.net/forum?id=r1lyTjAqYX)|
|Agent57   |Continuous |Discrete  | 70%||[Paper](https://arxiv.org/abs/2003.13350)|
|Agent57_light|Continuous|Discrete  | 70%|Agent57 - (LSTM,MultiStep)|


### PolicyBase/ActorCritic

|Algorithm              |Observation|Action    |ProgressRate||
|-----------------------|-----------|----------|------------|---|
|VanillaPolicyDiscrete  |Discrete   |Discrete  |100%||
|VanillaPolicyContinuous|Discrete   |Continuous|100%||
|REINFORCE              |           |          | 0%||
|A3C/A2C                |           |          | 0%||
|TRPO                   |Continuous |Continuous| 0%||
|PPO                    |Continuous |Continuous| 0%||
|DDPG/TD3               |Continuous |Continuous| 0%||
|SAC                    |Continuous |Continuous| 50%||[Paper](https://arxiv.org/abs/1812.05905)|


## Model Base

|Algorithm|Observation|Action     |ProgressRate||
|---------|-----------|-----------|------------|---|
|MCTS     |Discrete   |Discrete   | 0%||
|AlphaZero|Continuous |Discrete   | 0%||[Paper](https://arxiv.org/abs/1712.01815)|
|MuZero   |Continuous |Discrete   | 0%||[Paper](https://www.nature.com/articles/s41586-020-03051-4)|
|DynaQ    |Discrete   |Discrete   | 10%||
|WorldModels|         |           | 0%||
|DreamerV2  |         |           | 0%||




# Diagrams

## Overview

* sequence flow

![](diagrams/overview-sequence.drawio.png)

* distributed flow

![](diagrams/overview-distributed.drawio.png)

* multiplay flow

![](diagrams/overview-multiplay.drawio.png)




## PlayFlow

![](diagrams/playflow.png)

## Distribute flow

* main

![](diagrams/runner_mp_flow.png)

* Trainer

![](diagrams/runner_mp_flow_trainer.png)

* Workers

![](diagrams/runner_mp_flow_worker.png)


## Class diagram

![](diagrams/class_rl.png)

![](diagrams/class_env.png)



# Interface

|   |           |          |Type|
|---|-----------|----------|------|
|env|action     |          |Space|
|env|observation|          |Space|
|rl |action     |Discrete  |int|
|rl |action     |Continuous|list[float]|
|rl |observation|Discrete  |np.ndarray(dtype=int)|
|rl |observation|Continuous|np.ndarray(dtype=float)|

* Space(srl.base.env.spaces)

|class               |Type       |
|--------------------|-----------|
|DiscreteSpace       |int        |
|ArrayDiscreteSpace  |list[int]  |
|ContinuousSpace     |float      |
|ArrayContinuousSpace|list[float]|
|BoxSpace            |np.ndarray |


# 開発環境/動作確認環境

+ windows10
+ Python(3.9.13, 3.7.9)
+ CPUx1: Core i7-8700 3.2GHz
+ GPUx1: NVIDIA GeForce GTX 1060 3GB
+ memory 48GB

