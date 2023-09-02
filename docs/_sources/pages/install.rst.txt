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

    assert os.path.isdir("./simple_distributed_rl/srl/")  # srlがここにある想定です
    sys.path.insert(0, "./simple_distributed_rl/")

    import srl
    print(srl.__version__)


Sample code
====================

動作確認用のサンプルコードは以下です。

.. literalinclude:: ../../examples/sample_basic.py



Using library
====================

その他、機能によっては以下ライブラリが必要になります。

+ Tensorflow が必要なアルゴリズムを使う場合に必要
   + tensorflow
   + tensorflow-probability
+ Torch が必要なアルゴリズムを使う場合に必要
   + <https://pytorch.org/get-started/locally/>
+ RGBの描画関係を使用する場合に必要
   + pillow
   + opencv-python
   + pygame
+ 統計情報を扱う場合に必要
   + pandas
   + matplotlib
+ OpenAI Gym の環境を使う場合に必要
   + gym or gymnasium
   + pygame
+ Profile情報を表示する場合に必要
   + psutil
   + pynvml

Tensorflow,Torchを除いたライブラリを一括でインストールするコマンドは以下です。

.. code-block:: console

    $ pip install matplotlib pillow opencv-python pygame pandas gymnasium psutil pynvml
