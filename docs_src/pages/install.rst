.. _install:

============
Installation
============

.. currentmodule:: srl

必須ライブラリはnumpyだけです。  
ただ使う機能によっては他のライブラリをインストール必要があります。  

.. code-block:: console

    $ pip install numpy


Option Libraries
====================

使う機能によって以下ライブラリが必要になります。

+ Tensorflow が必要なアルゴリズムを使用する場合
   + tensorflow
   + tensorflow-probability
+ Torch が必要なアルゴリズムを使用する場合
   + https://pytorch.org/get-started/locally/
+ 主に画像関係の機能を使用する場合
   + pillow
   + opencv-python
   + pygame
+ 主にhistoryによる統計情報を扱う場合
   + pandas
   + matplotlib
+ OpenAI Gym の環境を使用する場合
   + gym or gymnasium
   + pygame
+ ハードウェアの統計情報を表示する場合
   + psutil
   + pynvml
+ クラウド/ネットワークによる分散学習を使用する場合
   + redis
   + pika
   + paho-mqtt
+ 学習を管理する場合
  + mlflow

一括インストールコマンドは以下です。
（Tensorflow、Torch、クラウド分散学習用ライブラリ、mlflowを除く）

.. code-block:: console

    $ pip install matplotlib pillow opencv-python pygame pandas gymnasium psutil pynvml



Installation
============

本フレームワークはGitHubからインストールまたはダウンロードをして使う事を想定しています。  
GitHubから直接インストールするコマンドは以下です。  

.. code-block:: console

    $ pip install git+https://github.com/pocokhc/simple_distributed_rl

ダウンロードしてインストールする方法は以下です。

.. code-block:: console

    $ git clone https://github.com/pocokhc/simple_distributed_rl.git
    $ cd simple_distributed_rl
    $ pip install .


Download(No install)
====================

srlディレクトリに実行パスが通っていればダウンロードだけでも使えます。

.. code-block:: console

    # Download srl files
    $ git clone https://github.com/pocokhc/simple_distributed_rl.git

.. code-block:: python

    import os
    import sys

    assert os.path.isdir("./simple_distributed_rl/srl/")  # Location of srl
    sys.path.insert(0, "./simple_distributed_rl/")

    import srl
    print(srl.__version__)


Sample code
====================

動作確認用のサンプルコードは以下です。

.. literalinclude:: ../../examples/sample_basic.py

