[README in English](README-en.md)

[![(latest) release | GitHub](https://img.shields.io/github/release/pocokhc/simple_distributed_rl.svg?logo=github&style=popout)](https://github.com/pocokhc/simple_distributed_rl/releases/latest)

# シンプルな分散強化学習フレームワーク（SRL ; Simple distributed Reinforcement Learning framework）

シンプルな分散強化学習フレームワークを目指して作成しています。  
以下の特徴があります。  

+ 分散強化学習のサポート
+ 環境とアルゴリズム間のインタフェースの自動調整
+ Gym/Gymnasiumの環境に対応
+ カスタマイズ可能な環境クラスの提供
+ カスタマイズ可能な強化学習アルゴリズムクラスの提供
+ 有名な強化学習アルゴリズムの提供
+ （新しいアルゴリズムへの対応）

**ドキュメント**

<https://pocokhc.github.io/simple_distributed_rl/>


**アルゴリズムの解説記事（Qiita）**

<https://qiita.com/pocokhc/items/a2f1ba993c79fdbd4b4d>


# 1. インストール

- Pypiからインストールできます。

``` bash
# 基本的な機能のみをインストール
pip install srl-frametowrk

# 主要な拡張機能や補助的なライブラリも含めてインストール（※TensorFlowやPyTorchは含まれません）
pip install srl-frametowrk[full]
```

Tensorflow/PyTorchは含まれないので別途インストールしてください。

+ Tensorflow
  + <https://www.tensorflow.org/install?hl=ja>
  + tensorflow-probability[tf]
+ PyTorch
  + <https://pytorch.org/get-started/locally/>


インストールされるライブラリの詳細や、ダウンロードのみでの使い方は以下のドキュメントを参照してください。  
<https://pocokhc.github.io/simple_distributed_rl/pages/install.html>


# 2. 使い方

簡単な使い方は以下です。

``` python
import srl
from srl.algorithms import ql  # qlアルゴリズムのimport


def main():
    # runnerの作成
    runner = srl.Runner("Grid", ql.Config())

    # 学習
    runner.train(timeout=10)

    # 学習結果の評価
    rewards = runner.evaluate()
    print(f"evaluate episodes: {rewards}")

    # 可視化例
    runner.animation_save_gif("Grid.gif")


if __name__ == "__main__":
    main()

```

animation_save_gifで生成された画像は以下です。

![Grid.gif](Grid.gif)

使い方の詳細は以下のドキュメントを参照してください。
<https://pocokhc.github.io/simple_distributed_rl/pages/howtouse.html>


# 3. 自作の環境とアルゴリズム

自作の環境とアルゴリズムの作成に関しては以下のドキュメントを参照してください。

+ [環境の作成方法](https://pocokhc.github.io/simple_distributed_rl/pages/custom_env.html)
+ [外部環境との連携方法](./examples/external_env/)
+ [アルゴリズムの作成方法](https://pocokhc.github.io/simple_distributed_rl/pages/custom_algorithm.html)


# 4. アルゴリズム
## モデルフリー
### 価値ベース

|Algorithm |Observation|Action  |Tensorflow|Torch|ProgressRate||
|----------|-----------|--------|----------|-----|------------|---|
|QL        |Discrete   |Discrete|-         |- |100%|Basic Q Learning|
|DQN       |Continuous |Discrete|✔        |✔|100%||
|C51       |Continuous |Discrete|✔        |- |99%|CategoricalDQN|
|Rainbow   |Continuous |Discrete|✔        |✔|100%||
|R2D2      |Continuous |Discrete|✔        |- |100%||
|Agent57   |Continuous |Discrete|✔        |✔|100%||
|SND       |Continuous |Discrete|✔        |- |100%||
|Go-Explore|Continuous |Discrete|✔        |- |100%|DQN base, R2D3 memory base|

### 方策ベース/ActorCritic

|Algorithm     |Observation|Action    |Tensorflow|Torch|ProgressRate|
|--------------|-----------|----------|----------|-----|---|
|VanillaPolicy |Discrete   |Both      |-         |-    |100%|
|A3C/A2C       |-          |-         |-         |-    |-   |
|TRPO          |-          |-         |-         |-    |-   |
|PPO           |Continuous |Both      |✔        |-    |100%|
|DDPG/TD3      |Continuous |Continuous|✔        |-    |100%|
|SAC           |Continuous |Both      |✔        |-    |100%|

## Alphaシリーズ

|Algorithm  |Observation|Action  |Tensorflow|Torch|ProgressRate||
|-----------|-----------|--------|----------|-----|---|---|
|MCTS       |Discrete   |Discrete|-         |-    |100%|MDP base|
|AlphaZero  |Image      |Discrete|✔        |-    |100%|MDP base|
|MuZero     |Image      |Discrete|✔        |-    |100%||
|StochasticMuZero|Image |Discrete|✔        |-    |100%||

## モデルベース

|Algorithm  |Observation|Action     |Framework|ProgressRate|
|-----------|-----------|-----------|---------|----|
|DynaQ      |Discrete   |Discrete   |-        |100%|

### WorldModelシリーズ

|Algorithm  |Observation|Action     |Tensorflow|Torch|ProgressRate||
|-----------|-----------|-----------|----------|-----|---|---|
|WorldModels|Continuous |Discrete   |✔        |-    |100%||
|PlaNet     |Continuous |Discrete   |✔(+tensorflow-probability)|-|100%||
|Dreamer    |Continuous |Both       |-|-|merge DreamerV3|
|DreamerV2  |Continuous |Both       |-|-|merge DreamerV3|
|DreamerV3  |Continuous |Both       |✔(+tensorflow-probability)|-|100%||

## オフライン

|Algorithm  |Observation|Action     |Framework|ProgressRate|
|-----------|-----------|-----------|----------|----|
|CQL        |Discrete   |Discrete   |          |  0%|

## オリジナル

|Algorithm     |Observation|Action  |Type     |Tensorflow|Torch|ProgressRate||
|--------------|-----------|--------|---------|----------|-----|---|---|
|QL_agent57    |Discrete   |Discrete|ValueBase|-         |-    |80%|QL + Agent57|
|Agent57_light |Continuous |Discrete|ValueBase|✔        |✔   |100%|Agent57 - (LSTM,MultiStep)|
|SearchDynaQ   |Discrete   |Discrete|ModelBase|-         |-    |100%|original|
|GoDynaQ       |Discrete   |Discrete|ModelBase|-         |-    |99%|original|
|GoDQN         |Continuous |Discrete|ValueBase|✔        |-    |90%|original|


# 5. オンライン分散学習

ネットワーク経由での分散学習は以下のドキュメントを参照してください。

+ [Distributed Learning (Online)](https://pocokhc.github.io/simple_distributed_rl/pages/distributed.html)

またクラウドサービスとの連携はQiita記事を参照

+ [クラウドサービスを利用した分散強化学習（無料編）](https://qiita.com/pocokhc/items/f7a32ee6c62cba54d6ab)
+ [クラウドサービスを利用した分散強化学習（kubernetes編）](https://qiita.com/pocokhc/items/56c930e1e401ce156141)
+ [クラウドサービスを利用した分散強化学習（GKE/有料編）](https://qiita.com/pocokhc/items/e08aab0fe56566ab9407)


# 6. フレームワークの概要

![overview-sequence.drawio.png](diagrams/overview-sequence.drawio.png)

+ 疑似コード

※実装側の動作に関してはアルゴリズムの作成方法を参照

``` python
# 学習単位の初期化
env.setup()
worker.setup()
trainer.setup()

for episode in range(N)
  # 1エピソードの初期化
  env.reset()
  worker.reset()

  # 1エピソードのループ
  while not env.done:
      # アクションを取得
      action = worker.policy()

      # 描画
      env.render()
      worker.render()

      # 環境の1stepを実行
      env.step(action)
      worker.on_step()

      # 学習
      trainer.train()

  # 終了後の描画
  env.render()

# 学習単位の終了
env.teardown()
worker.teardown()
trainer.teardown()
```


# 7. 開発環境

Look [dockers folder](./dockers/)

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
