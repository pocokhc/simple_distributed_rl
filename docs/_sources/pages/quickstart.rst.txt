.. _quickstart:

===============
Getting Started
===============


Basic run of study
==================

最もシンプルな学習は以下です。

.. literalinclude:: quickstart1.py


Commonly run Example
=====================

学習と評価を別々で実行できる形式です。

.. literalinclude:: ../../examples/sample_commonly.py

.. image:: ../../Grid.gif


How to load external environment
======================================

本フレームワークは Gym/Gymnasium の環境に対応しています。

Gym/Gymnasium に登録されている環境ならそのまま読み込むことが出来ます。
（フレームワーク内では 'gym.make' または 'gymnasium.make' で読み込んでいます）

.. literalinclude:: quickstart_use_env1.py


'gymnasium.make' 以外の関数で環境を読み込む方法
------------------------------------------------

'gymnasium.make' 以外で読み込む場合は、引数にその関数を指定してください。
例は 'gym-retro' を読み込む例です。

.. literalinclude:: quickstart_use_env2.py


Other Run Methods
=====================

Evaluate
---------------------

学習せずにシミュレーションし、報酬を返します。

.. literalinclude:: quickstart_eval.py


Replay Window
---------------------

シミュレーションした結果を後から見返す機能です。
1step毎の様子を見返すことができます。

.. literalinclude:: quickstart_replay_window.py


Render Terminal
---------------------

print出力の形式で1エピソードシミュレーションします。

.. literalinclude:: quickstart_render_terminal.py


Render Window
---------------------

pygameにて1エピソードを描画します。
pygameのwindowが表示できる環境である必要があります。

.. literalinclude:: quickstart_render_window.py


Animation
---------------------

映像として残せるようにRGBデータを保存しながらシミュレーションします。
アニメーションは、'matplotlib.animation.ArtistAnimation' で作成されます。

.. literalinclude:: quickstart_animation.py


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

