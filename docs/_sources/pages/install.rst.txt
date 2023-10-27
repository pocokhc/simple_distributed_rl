.. _install:

============
Installation
============

.. currentmodule:: srl

必須ライブラリはnumpyだけです。
ただ使う機能によっては他のライブラリをインストール必要があります。（後述）

.. code-block:: console

    $ pip install numpy

SRLはGitHubからインストールまたはダウンロードをして使うことができます。


Installation
============

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



Option library
====================

使う機能によって以下ライブラリが必要になります。

+ Tensorflow が必要なアルゴリズムを使う場合
   + tensorflow
   + tensorflow-probability
+ Torch が必要なアルゴリズムを使う場合
   + https://pytorch.org/get-started/locally/
+ 主にRGBの描画関係を使用する場合
   + pillow
   + opencv-python
   + pygame
+ 主にhistoryによる統計情報を扱う場合
   + pandas
   + matplotlib
+ OpenAI Gym の環境を使う場合
   + gym or gymnasium
   + pygame
+ ハードウェアの統計情報を表示する場合
   + psutil
   + pynvml
+ Redisによる分散学習を使う場合
   + redis

Tensorflow,Torchを除いたライブラリを一括でインストールするコマンドは以下です。

.. code-block:: console

    $ pip install matplotlib pillow opencv-python pygame pandas gymnasium psutil pynvml redis
