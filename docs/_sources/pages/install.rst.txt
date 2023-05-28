.. _install:

============
Installation
============

.. currentmodule:: srl


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

動作確認用のサンプルコードは以下です。

.. literalinclude:: ../../examples/sample_basic.py


Download(No install)
====================

srlディレクトリに実行パスが通っていればダウンロードだけでも使えます。

.. code-block:: python

    import os
    import sys

    assert os.path.isdir("./simple_distributed_rl/srl/")  # srlがここにある想定です
    sys.path.insert(0, "./simple_distributed_rl/")

    import srl
    print(srl.__version__)

