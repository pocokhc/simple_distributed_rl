
==============================================
Distributed Learning (Multiple PCs)
==============================================

ここではネットワーク経由で学習する方法を説明します。概要は以下です。

.. image:: ../../diagrams/overview-mp.drawio.png

フローイメージは以下です。

.. image:: ../../diagrams/runner_distributed_flow.png


各サーバとのやりとりですが、Redisを採用しています。Redis経由でのやりとりイメージは以下です。

.. image:: ../../diagrams/overview-distribution.drawio.png

学習を実行するまでのステップは以下となります。

0. redis(pip)のインストール(初回のみ)
1. Redisの起動
2. TrainerServer/ActorServerの起動
3. 学習の実施

----------------------------------
0. redis(pip)のインストール
----------------------------------

Redisへのアクセスとして、redis を使うので以下でインストールします。

.. code-block:: console

    $ pip install redis


------------------------
1. Redisサーバの起動
------------------------

| 任意のRedisサーバを用意します。
| フレームワーク上はサンプルとしてdocker-composeファイルを用意していますのでそちらを起動してください。

.. code-block:: console

    $ docker-compose -f examples/distribution/docker-compose.yml up -d


--------------------------------------------
2. TrainerServer/ActorServerの起動
--------------------------------------------

| TrainerServerとActorServerを任意のPCで起動します。
| 基本はTrainerServerは1個、ActorServerは1個以上(actor_num数)の起動を想定しています。
| ※各TrainerServer/ActorServerでも、EnvとAlgorithmが使用できる必要があります

TrainerServerの起動例です。(examples/distribution/server_trainer.py)

.. literalinclude:: ../../examples/distribution/server_trainer.py

ActorServerの起動例です。(examples/distribution/server_actor.py)

.. literalinclude:: ../../examples/distribution/server_actor.py

引数は以下です。

.. list-table::
   :widths: 3 2 20
   :header-rows: 0

   * - host
     - str
     - Redisのホスト名またはIPアドレスを指定します。
   * - port
     - int
     - Redisのポートを指定します。省略時は6379を使います。

--------------------------------------------
3. 学習の実施
--------------------------------------------

| 学習のサンプルコードは以下です。Runnerから train_distribution を呼び出すと学習します。
| 学習後はrunner内のparameterに学習済みデータが入っています。

.. literalinclude:: ../../examples/distribution/main.py

