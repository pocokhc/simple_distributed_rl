.. _quickstart:

===============
Getting Started
===============


Basic run of study
==================

最もシンプルな学習は以下です。

.. literalinclude:: ../../examples/sample_basic.py


How to load external environment
======================================

本フレームワークは Gym/Gymnasium の環境に対応しています。

Gym/Gymnasium に登録されている環境ならそのまま読み込むことが出来ます。
（Gym/Gymnasiumをインストールしている必要があります）
（フレームワーク内では 'gym.make' または 'gymnasium.make' で読み込んでいます）

.. literalinclude:: quickstart_use_env1.py


'gymnasium.make' 以外の関数で環境を読み込む方法
------------------------------------------------

'gymnasium.make' 以外で読み込む場合は、引数にその関数を指定してください。
例は 'gym-retro' を読み込む例です。

.. literalinclude:: ../../examples/sample_gym_retro.py

