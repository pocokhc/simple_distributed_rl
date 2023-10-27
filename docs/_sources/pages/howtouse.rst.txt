.. _howtouse:

===============
How To Use
===============

基本は以下の手順を想定しています。

1. EnvConfigで環境を設定する。
2. RLConfigでアルゴリズムを設定する。
3. Runnerで動かす。


EnvConfig
=====================

実行する環境を指定します。

.. code-block:: python

    import srl
    env_config = srl.EnvConfig("Grid")

Gym/Gymnasium がインストールされていればそれらのIDも指定できます。

.. code-block:: python

    import srl
    env_config = srl.EnvConfig("FrozenLake-v1")

| 自作の環境を用意したい場合は :ref:`custom_env` を見てください。
| また、ID以外に環境に対して設定できる項目は :ref:`env_config` を見てください。

RLConfig
=====================

| 実行するアルゴリズムを指定します。
| 各アルゴリズムには必ずConfigが定義されているのでそれを呼び出します。

.. code-block:: python

    # 使うアルゴリズムを読み込み
    from srl.algorithms import ql
    rl_config = ql.Config()

各アルゴリズムのハイパーパラメータはConfigの変数で値を指定できます。

.. code-block:: python

    # 割引率を変更する例
    rl_config = ql.Config(discount=0.5)

    # インスタンス後に書き換えてもいい
    rl_config.discount = 0.3

| 各アルゴリズムのハイパーパラメータについては srl.algorithms 配下のそれぞれのコードを見てください。
| また、共通パラメータに関しては :ref:`rl_config` を参照してください。


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

以下は学習と評価を分けて実行する例です。

.. literalinclude:: ../../examples/sample_commonly.py

.. image:: ../../Grid.gif

| Runnerで用意している実行形式は以下です。
| 引数や他のRunnerの機能に関しては :ref:`runner` を見てください。


Evaluate
---------------------

学習せずにシミュレーションし、報酬を返します。

.. literalinclude:: howtouse_eval.py


Replay Window
---------------------

| シミュレーションして、その結果を見返す機能です。
| 1step毎の様子を見ることができます。(GUIで表示されます)
| pygameのwindowが表示できる環境である必要があります。

.. literalinclude:: howtouse_replay_window.py


Render Terminal
---------------------

print出力の形式で1エピソードシミュレーションします。

.. literalinclude:: howtouse_render_terminal.py


Render Window
---------------------

1エピソードを描画します。
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

