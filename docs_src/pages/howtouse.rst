.. _howtouse:

===============
How To Use
===============

基本は以下の手順です。

1. EnvConfig で環境を設定する。
2. RLConfig でRLアルゴリズムを設定する。
3. Runner で動かす。


EnvConfig
=====================

実行する環境を設定します。
基本は登録されているIDを指定するだけです。

.. code-block:: python

    import srl
    env_config = srl.EnvConfig("Grid")

Gym/Gymnasium がインストールされていればそれらのIDも指定できます。

.. code-block:: python

    import srl
    env_config = srl.EnvConfig("FrozenLake-v1")

自作の環境を用意したい場合は :ref:`custom_env` を見てください。
また、その他環境に対して設定できる項目は :ref:`env_config` を見てください。


RLConfig
=====================

実行するアルゴリズムを設定します。
各アルゴリズムには必ずConfigが定義されているのでそれを呼び出します。

.. code-block:: python

    # 使うアルゴリズムを読み込み
    from srl.algorithms import ql
    rl_config = ql.Config()

各アルゴリズムにはそれぞれハイパーパラメータがあるので、基本はコンストラクタまたは変数で値を指定します。

.. code-block:: python

    # 割引率を変更する例
    rl_config = ql.Config(discount=0.5)

    # インスタンス後に書き換えてもいい
    rl_config.discount = 0.3

各アルゴリズムのパラメータは srl.algorithms 配下のそれぞれのコードを見てください。
また、一部のハイパーパラメータはコンストラクタから指定できません。
一部のハイパーパラメータと共通のパラメータに関しては :ref:`rl_config` を見てください。


Runner
=====================

EnvConfigとRLConfigを元に実際に実行するRunnerを作成します。

.. code-block:: python

    import srl

    runner = srl.Runner(env_config, rl_config)

    # envはIDのみでも可能
    runner = srl.Runner("Grid", rl_config)

    # envのみの指定も可能(ただしアルゴリズムを使うものは利用できない)
    runner = srl.Runner("Grid")

Runnerを作成したら、基本的な流れは学習と評価です。

最もシンプルな学習は以下です。

.. literalinclude:: ../../examples/sample_basic.py

学習と評価を別々で実行できる形式です。

.. literalinclude:: ../../examples/sample_commonly.py

.. image:: ../../Grid.gif

その他、Runnerで用意されているものは以下です。
引数や他のRunnerの機能に関しては :ref:`runner` を見てください。


Evaluate
---------------------

学習せずにシミュレーションし、報酬を返します。

.. literalinclude:: howtouse_eval.py


Replay Window
---------------------

シミュレーションして、その結果を見返す機能です。
1step毎の様子を見ることができます。

.. literalinclude:: howtouse_replay_window.py


Render Terminal
---------------------

print出力の形式で1エピソードシミュレーションします。

.. literalinclude:: howtouse_render_terminal.py


Render Window
---------------------

pygameにて1エピソードを描画します。
pygameのwindowが表示できる環境である必要があります。

.. literalinclude:: howtouse_render_window.py


Animation
---------------------

映像として残せるようにRGBデータを保存しながらシミュレーションします。
アニメーションは、'matplotlib.animation.ArtistAnimation' で作成されます。

.. literalinclude:: howtouse_animation.py


Manual play Terminal
---------------------

Terminal上で手動プレイします。
環境によっては動作しない場合があります。

.. literalinclude:: ../../examples/sample_play_grid.py


Manual play Window
---------------------

pygame上で手動プレイします。
環境によっては動作しない場合があります。

'key_bind' は設定しなくても遊べますが、設定するとより環境にあった入力方法でプレイすることができます。

.. literalinclude:: ../../examples/sample_play_atari.py

