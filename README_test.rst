
.. |(latest) release| image:: https://img.shields.io/github/release/pocokhc/simple_distributed_rl.svg?logo=github&style=popout
  :target: https://github.com/pocokhc/simple_distributed_rl/releases/latest

============================================================================
Simple Distributed Reinforcement Learning (シンプルな分散強化学習)
============================================================================

シンプルな分散強化学習フレームワークを目指して作成しました。  
どちらかというと学習用フレームワークに近いかもしれません。  
以下の特徴があります。  

+ 標準で分散強化学習のサポート
+ 環境とアルゴリズム間のインタフェースの自動調整
+ 上記を満たすカスタマイズ可能な環境クラスの提供
+ 上記を満たすカスタマイズ可能な強化学習アルゴリズムクラスの提供
+ 有名な強化学習アルゴリズムの提供
+ （新しいアルゴリズムへの対応）

また本フレームワークの解説は[Qiita記事](https://qiita.com/pocokhc/items/a2f1ba993c79fdbd4b4d)にして記載しています。
アルゴリズムの解説等をしているのでハイパーパラメータの設定等に迷ったりしたら見てみてください。

Document: <https://pocokhc.github.io/simple_distributed_rl/>

1.Install/Download
====================

numpyライブラリだけ必須になるので入れてください。

.. code-block:: console

    $ pip install numpy

SRLはGitHubからインストールまたはダウンロードをして使うことができます。

Install
-------

インストールするコマンド例は以下です。

.. code-block:: console

    $ pip install git+https://github.com/pocokhc/simple_distributed_rl


or

.. code-block:: console

    $ git clone https://github.com/pocokhc/simple_distributed_rl.git
    $ cd simple_distributed_rl
    $ pip install .


Download(No install)
---------------------

srlディレクトリに実行パスが通っていればダウンロードだけでも使えます。

.. code-block:: console

    # Download srl files
    $ git clone https://github.com/pocokhc/simple_distributed_rl.git

.. code-block:: python

    import os
    import sys

    assert os.path.isdir("./simple_distributed_rl/srl/")  # srlがここにある想定です
    sys.path.insert(0, "./simple_distributed_rl/")

    import srl
    print(srl.__version__)


Using library
---------------------

その他、機能によっては以下ライブラリが必要になります。

+ Tensorflow が必要なアルゴリズムを使う場合に必要
   + tensorflow
   + tensorflow-addons
   + tensorflow-probability
+ Torch が必要なアルゴリズムを使う場合に必要
   + <https://pytorch.org/get-started/locally/>
+ RGBの描画関係を使用する場合に必要
   + matplotlib
   + pillow
   + opencv-python
   + pygame
+ 統計情報を扱う場合に必要
   + pandas
+ OpenAI Gym の環境を使う場合に必要
   + gym
   + pygame
+ Profile情報を表示する場合に必要
   + psutil
   + pynvml

Tensorflow,Torchを除いたライブラリを一括でインストールするコマンドは以下です。

.. code-block:: console

    $ pip install matplotlib pillow opencv-python pygame pandas gym psutil pynvml
    

2.Usage
====================

Basic run of study
---------------------

.. literalinclude:: docs_src/pages/quickstart1.py


Commonly run Example
------------------------

.. literalinclude:: docs_src/pages/quickstart2.py

.. image:: FrozenLake.gif


3.Framework Overview
========================

Sequence flow
---------------

.. image:: diagrams/overview-sequence.drawio.png

Distributed flow
------------------

.. image:: diagrams/overview-distributed.drawio.png

Runner
---------

.. image:: diagrams/overview-runner.drawio.png

Simplified pseudo code
--------------------------


.. code-block:: python

  # Initializing phase
  env.reset()
  worker.on_reset(env)
  env.render()

  # 1 episode loop
  while not env.done:
      action = worker.policy(env)
      worker.render(env)
      env.step(action)
      worker.on_step(env)
      env.render()


4.Algorithms
=================

ModelFree
-------------

ValueBase
^^^^^^^^^^^

.. list-table::
    :widths: 5 20 40
    :header-rows: 1

    * - Algorithm
      - Observation
      - Action
      - Framework
      - ProgressRate
      - 
    * - QL
      - Discrete
      - Discrete
      - 
      - 100%
      - Basic Q Learning
    * - DQN
      - Continuous
      - Discrete
      - Tensorflow/Torch
      - 100%
      - 
    * - C51
      - Continuous
      - Discrete
      - Tensorflow
      - 99%
      - CategoricalDQN
    * - Rainbow
      - Continuous
      - Discrete
      - Tensorflow/Torch
      - 100%
      - 
    * - R2D2
      - Continuous
      - Discrete
      - Tensorflow
      - 100%
      - 
    * - Agent57
      - Continuous
      - Discrete
      - Tensorflow
      - 100%
      - 


PolicyBase/ActorCritic
^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
    :widths: 5 20 40
    :header-rows: 1

    * - Algorithm
      - Observation
      - Action
      - Framework
      - ProgressRate
      - 
    * - VanillaPolicy
      - Discrete
      - Both
      - 
      - 100%
      - 
    * - A3C/A2C
      - 
      - 
      - 
      - 0%
      - 
    * - TRPO
      - Continuous
      - 
      - 
      - -
      - 
    * - PPO
      - Continuous
      - 
      - 
      - 0%
      - 
    * - DDPG/TD3
      - Continuous
      - Continuous
      - Tensorflow
      - 100%
      - 
    * - SAC
      - Continuous
      - Continuous
      - Tensorflow
      - 100%
      - 


## AlphaSeries

|Algorithm  |Observation|Action  |Frameworks|ProgressRate||
|-----------|-----------|--------|----------|----|---|
|MCTS       |Discrete   |Discrete|          |100%|MDP base|
|AlphaZero  |Image      |Discrete|Tensorflow|100%|MDP base|
|MuZero     |Image      |Discrete|Tensorflow|100%|MDP base|
|StochasticMuZero|Image |Discrete|Tensorflow|100%|MDP base|

## ModelBase

|Algorithm  |Observation|Action     |Frameworks|ProgressRate|
|-----------|-----------|-----------|----------|----|
|DynaQ      |Discrete   |Discrete   | 10%|

### WorldModels

|Algorithm  |Observation|Action     |Frameworks|ProgressRate|
|-----------|-----------|-----------|----------|----|
|WorldModels|Continuous |Discrete   |Tensorflow|100%|
|PlaNet     |Continuous |Discrete   |Tensorflow,tensorflow-probability|100%|
|Dreamer    |           |           |Tensorflow,tensorflow-probability|  0%|
|DreamerV2  |           |           ||  0%|

## Offline

|Algorithm  |Observation|Action     |Frameworks|ProgressRate|
|-----------|-----------|-----------|----------|----|
|CQL        |Discrete   |Discrete   ||  0%|

## その他(Original)

|Algorithm    |Observation|Action  |Type     |Frameworks|ProgressRate||
|-------------|-----------|--------|---------|----------|----|---|
|QL_agent57   |Discrete   |Discrete|ValueBase|          | 80%|QL + Agent57|
|Agent57_light|Continuous |Discrete|ValueBase|Tensorflow|100%|Agent57 - (LSTM,MultiStep)|
|SearchDynaQ  |Discrete   |Discrete|ModelBase/ValueBase|| 80%|original|

# 5. Customize

オリジナル環境とアルゴリズムの作成例は以下のファイルを参考にしてください。

examples/custom_env.ipynb  
examples/custom_rl.ipynb  

# 6. Detailed framework information

## Multiplay flow

![overview-multiplay.drawio.png](diagrams/overview-multiplay.drawio.png)

## Play flow

![playflow.png](diagrams/playflow.png)

## Distribute flow

+ **main**

![runner_mp_flow.png](diagrams/runner_mp_flow.png)

+ **Trainer**

![runner_mp_flow_trainer.png](diagrams/runner_mp_flow_trainer.png)

+ **Workers**

![runner_mp_flow_worker.png](diagrams/runner_mp_flow_worker.png)

## Class diagram

![class_rl.png](diagrams/class_rl.png)

![class_env.png](diagrams/class_env.png)

## Interface

+ Env input/output type

|   |           |Type|
|---|-----------|------|
|Env|action     |Space|
|Env|observation|Space|

+ RL input/output type

|   |          |           |Type|
|---|----------|-----------|------|
|RL |Discrete  |action     |int|
|RL |Discrete  |observation|np.ndarray(dtype=int)|
|RL |Continuous|action     |list[float]|
|RL |Continuous|observation|np.ndarray(dtype=np.float32)|

+ Space(srl.base.env.spaces)

|Class               |Type       |
|--------------------|-----------|
|DiscreteSpace       |int        |
|ArrayDiscreteSpace  |list[int]  |
|ContinuousSpace     |float      |
|ArrayContinuousSpace|list[float]|
|BoxSpace            |np.ndarray |

# 7. Development environment

Look "./dockers/"

+ PC1
  + windows11
  + CPUx1: Core i7-8700 3.2GHz
  + GPUx1: NVIDIA GeForce GTX 1060 3GB
  + memory 48GB
+ PC2
  + windows11
  + CPUx1: Core i9-12900 2.4GHz
  + GPUx1: NVIDIA GeForce RTX 3060 12GB
  + memory 32GB
