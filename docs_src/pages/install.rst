.. _install:

============
Installation
============

.. currentmodule:: srl


- Pypiからインストールできます。

.. code-block:: console

    # 基本的な機能のみをインストール
    $ pip install srl-framework
    
    # 主要な拡張機能や補助的なライブラリも含めてインストール（※TensorFlowやPyTorchは含まれません）
    $ pip install srl-framework[full]


Tensorflow/PyTorchは含まれないので別途インストールしてください。

+ Tensorflow
   + <https://www.tensorflow.org/install?hl=ja>
   + tensorflow-probability[tf]
+ PyTorch
   + <https://pytorch.org/get-started/locally/>



Install options
^^^^^^^^^^^^^^^^^^^^^^^^^

| SRLには追加機能を有効にするための **extras オプション** が用意されています。
| これにより、標準機能に加えて、特定のユースケースに必要な外部ライブラリも一括でインストールすることが可能です。
| 
| 利用可能な extras オプションと、それぞれに含まれるライブラリは以下です。

+ なし
   + numpy
   + opencv-python
   + Pillow
   + pygame
+ full
   + 「なし」に以下が追加されます
   + gymnasium
   + matplotlib
   + pandas
   + psutil
   + pynvml
   + redis
   + mlflow
   + pybind11
   + setuptools
+ dev
   + 「full」に以下が追加されます
   + pytest
   + pytest-mock
   + pytest-timeout


Download(No install)
============================

| 本フレームワークはインストールしないでも使う事ができます。
| 本フレームワークをダウンロード後、srlディレクトリに実行パスが通っていれば使うことができます。

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



各ライブラリについて
====================

必須ライブラリ
^^^^^^^^^^^^^^^

+ numpy


その他のライブラリ
^^^^^^^^^^^^^^^^^^

+ Tensorflow が必要なアルゴリズムを使用する場合に必要
   + tensorflow
   + tensorflow-probability
+ Torch が必要なアルゴリズムを使用する場合に必要
   + torch
+ 画像関係の機能を使用する場合に必要
   + pillow
   + opencv-python
   + pygame
+ OpenAI Gym の環境を使用する場合に必要
   + gym or gymnasium
   + pygame
+ historyによる統計情報を扱う場合に必要
   + pandas
   + matplotlib
+ ハードウェアの統計情報を表示する場合に必要
   + psutil
   + pynvml
+ クラウド/ネットワークによる分散学習を使用する場合に必要
   + redis
+ 学習を管理する場合に必要
   + mlflow
+ C++の拡張を使用する場合に必要
   + pybind11
   + setuptools


Sample code
====================

動作確認用のサンプルコードは以下です。

.. literalinclude:: ../../examples/sample_basic.py

