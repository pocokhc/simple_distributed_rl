.. _howtouse:

===============
How To Use
===============

大きく以下となります。

1. 環境を設定（EnvConfig）
2. アルゴリズムを設定（RLConfig）
3. Runnerで動かす


1. EnvConfig
=====================

実行する環境を指定します。

.. code-block:: python

    import srl
    env_config = srl.EnvConfig("Grid")

Gym/Gymnasium の環境とも互換があり指定できます。

.. code-block:: python

    import srl
    env_config = srl.EnvConfig("FrozenLake-v1")

| 自作の環境を用意したい場合は :ref:`custom_env` を見てください。
| また、ID以外にEnvConfigに設定できる項目は :ref:`env_config` を見てください。


Gym/Gymnasiumに対応していない環境の読み込み
------------------------------------------------

| 'gym_make_func' 'gymnasium_make_func' に読み込む関数を指定することができます。
| 例は 'gym-retro' を読み込む例です。

.. literalinclude:: ../../examples/sample_gym_retro.py



2. RLConfig
=====================

| 実行するアルゴリズムを指定します。
| 各アルゴリズムにはConfigがあるのでそれを呼び出します。

.. code-block:: python

    # 使うアルゴリズムを読み込み
    from srl.algorithms import ql
    rl_config = ql.Config()

各アルゴリズムのハイパーパラメータはConfigの変数で値を指定できます。

.. code-block:: python

    # 割引率を変更する例
    rl_config = ql.Config(discount=0.5)

    # インスタンス後に書き換えも可能
    rl_config.discount = 0.3

| 各アルゴリズムのハイパーパラメータについては srl.algorithms 配下のそれぞれのコードを見てください。
| また、共通パラメータに関しては :ref:`rl_config` を参照してください。


3. Runner
=====================

EnvConfigとRLConfigを元に実際に実行するRunnerを作成します。

.. code-block:: python

    import srl

    # Runnerの引数にEnvConfigとRLConfigを指定
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    runner = srl.Runner(env_config, rl_config)

    # envはIDのみでも指定可能
    runner = srl.Runner("Grid", rl_config)

    # envのみの指定も可能(ただしアルゴリズムを使うものは利用できない)
    runner = srl.Runner("Grid")

Runnerを作成したら後は任意の関数を実行して学習します。


Basic run of study
--------------------------

.. literalinclude:: ../../examples/sample_basic.py


Commonly run Example
--------------------------

.. literalinclude:: ../../examples/sample_commonly.py

.. image:: ../../Grid.gif

引数や他のRunnerの機能に関しては :ref:`runner` を見てください。


4. Runner functions
==========================

Runnerで実行できる各関数に関してです。

Train
---------------------

| 学習をします。
| 学習後のParameterとMemoryがRunner内に保存されます。

.. code-block:: python

    runner.train(max_episode=10)


Rollout
---------------------

| 経験を集める時に使います。
| 実際に学習環境でエピソードを実行しますが、学習はしません。
| 実行後はMemoryがRunner内に保存されます。

.. code-block:: python

    runner.rollout(max_episode=10)


Train Only
---------------------

| エピソードは実行せず、Trainerの学習のみを実施します。
| Memoryにbatchがない状態など、学習出来ない場合で実行すると無限ループになるので注意してください。

.. code-block:: python

    runner.train_only(max_train_count=10)


Train Multiprocessing
---------------------

| multiprocessing による分散学習を実施します。

.. code-block:: python

    runner.train_mp(max_train_count=10)


Evaluate
---------------------

学習せずにシミュレーションし、報酬を返します。

.. literalinclude:: howtouse_eval.py


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

.. literalinclude:: howtouse_animation.py


Replay Window
---------------------

| シミュレーションして、その結果を見返す機能です。
| 1step毎の様子を見ることができます。(GUIで表示されます)
| pygameのwindowが表示できる環境である必要があります。

.. literalinclude:: howtouse_replay_window.py


Manual play Terminal
---------------------

Terminal上で手動プレイします。
環境によっては動作しない場合があります。

.. literalinclude:: ../../examples/sample_play_grid.py


Manual play Window
---------------------

| pygame上で手動プレイします。
| 'key_bind' は設定しなくても遊べますが、設定するとより環境にあった入力方法でプレイすることができます。

.. literalinclude:: ../../examples/sample_play_atari.py

