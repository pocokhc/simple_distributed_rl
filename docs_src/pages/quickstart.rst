.. _quickstart:

===============
Getting Started
===============


Basic run of study
==================

最もシンプルな学習は以下です。

.. literalinclude:: ../../examples/sample_basic.py


How to load Gym/Gymnasium
------------------------------------------------

本フレームワークは Gym/Gymnasium の環境に対応しています。

| Gym/Gymnasium に登録されている環境はそのまま読み込むことが出来ます。  
| （Gym/Gymnasiumをインストールしている必要があります）  
| （フレームワーク内では 'gym.make' または 'gymnasium.make' で読み込んでいます）  

.. literalinclude:: quickstart_use_env1.py


自作環境を読み込む方法
------------------------------------------------

:ref:`custom_env` のページを参照ください。


Gym/Gymnasiumに対応していない環境の読み込み
------------------------------------------------

| 'gym_make_func' に読み込む関数を指定することができます。
| 例は 'gym-retro' を読み込む例です。

.. literalinclude:: ../../examples/sample_gym_retro.py

