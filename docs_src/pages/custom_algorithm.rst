.. _custom_algorithm:

=========================
Create Original Algorithm
=========================

ここでは本フレームワークでの自作アルゴリズムを作成する方法を説明します。  
構成としては大きく以下です。  

#. 概要  
#. 実装するクラスの説明  
    #. Config
    #. RemoteMemory
    #. Parameter
    #. Trainer
    #. Worker
#. 自作アルゴリズムの登録
#. Q学習実装例


概要
==========

自作アルゴリズムでは5つクラスを定義する必要があり、以下のように連携して動作します。

.. image:: ../../diagrams/overview-sequence.drawio.png

図では現れていませんが、5つ目としてハイパーパーパラメータを管理するConfigクラスがいます。
役割は以下です。

.. list-table::
   :widths: 10 30
   :header-rows: 0

   * - Config
     - + ハイパーパラメータなどのパラメータを管理するクラス
   * - Worker
     - + Environmentと連携しサンプルを収集
       + 収集したサンプルをRemoteMemoryに送信
       + 行動決定に必要な情報をParameterから読む（read only）
   * - Trainer
     - + RemoteMemoryからサンプルを取得し学習する
       + 学習後、Parameterを更新する
   * - RemoteMemory
     - + サンプルを管理
   * - Parameter
     - + 学習パラメータを保持


分散学習は以下となり各クラスが非同期で動作します。

.. image:: ../../diagrams/overview-distributed.drawio.png

アルゴリズムを作成する視点だと大きな違いはありませんが、以下の点が異なります。  

+ WorkerがRemoteMemoryにサンプルを送るタイミングとTrainerが取り出すタイミングが異なる
+ Parameter が Worker と Trainer と同期されない
  
各クラスの実装の仕方を見ていきます。


実装する各クラスの説明
=======================


Config
--------------------------------------------

強化学習アルゴリズムの種類やハイパーパラメータ等を管理するクラスです。  
基底クラスは `srl.base.rl.base.RLConfig` でこれを継承して作成します。  
  
ただ、アルゴリズムに特化したインタフェースも用意しており、  
当てはまるアルゴリズムを作成する場合はそちらを継承したほうが作成が楽になります。  

現状あるクラスは以下です。  

.. list-table::
   :widths: 15 30 10
   :header-rows: 1

   * - クラス名
     - 説明
     - 
   * - DiscreteActionConfig
     - モデルフリーでアクションが離散値のアルゴリズム
     - Q学習,DQN等
   * - ContinuousActionConfig
     - モデルフリーでアクションが連続値のアルゴリズム
     - DDPG,SAC等

RLConfig で実装が必要な関数・プロパティは以下です。

.. code-block:: python

   from dataclasses import dataclass
   from srl.base.rl.base import RLConfig
   from srl.base.define import RLTypes
   from srl.base.rl.processor import Processor

   # 必ず dataclass で書いてください
   @dataclass
   class MyConfig(RLConfig):
      
      def getName(self) -> str:
         """ ユニークな名前を返す """
         raise NotImplementedError()

      @property
      def action_type(self) -> RLTypes:
         """
         アルゴリズムが想定するアクションのタイプを返してください。
         DISCRETE  : 離散値
         CONTINUOUS: 連続値
         ANY       : どちらでも
         """
         raise NotImplementedError()

      @property
      def observation_type(self) -> RLTypes:
         """
         アルゴリズムが想定する環境から受け取る状態のタイプを返してください。
         DISCRETE  : 離散値
         CONTINUOUS: 連続値
         ANY       : どちらでも
         """
         raise NotImplementedError()

      # ------------------------------------
      # 以下は option です。（必須ではない）
      # ------------------------------------
      def assert_params(self) -> None:
         """ パラメータのassertを記載 """
         super().assert_params()  # 親クラスも呼び出してください

      def set_config_by_actor(self, actor_num: int, actor_id: int) -> None:
         """ 分散学習でactorが指定されたときに呼び出されます。
         Actor関係の初期化がある場合は記載してください。 """
         pass

      def set_processor(self, actor_num: int, actor_id: int) -> list[Processor]:
         """ 前処理を追加したい場合設定してください """
         return []


DiscreteActionConfig では action_type の実装が不要になります。(DISCRETE固定)
ContinuousActionConfig でも action_type の実装が不要になります。(CONTINUOUS固定)



RemoteMemory
--------------------------------------------

Workerが取得したサンプル(batch)をTrainerに渡す役割を持っているクラスです。  
分散学習では multiprocessing のサーバプロセス(Manager)になります。  
ですので、変数へのアクセスができなくなる点だけ制約があります。  
（全て関数経由でやり取りする必要があります）  

またよくあるクラスは `srl.base.rl.remote_memory` に別途定義しており、そちらを使うこともできます。

.. list-table::
   :widths: 15 30
   :header-rows: 0

   * - SequenceRemoteMemory
     - 来たサンプルを順序通りに取り出します。(Queueみたいな動作です)
   * - ExperienceReplayBuffer
     - サンプルをランダムに取り出します。
   * - PriorityExperienceReplay
     - サンプルを優先順位に従い取り出します。

各クラスの定義方法は後述します。
まずは基底クラスである `RLRemoteMemory` で定義する関数は以下です。

.. code-block:: python

   from srl.base.rl.base import RLRemoteMemory
   from typing import cast

   class MyRemoteMemory(RLRemoteMemory):
      def __init__(self, *args):
         """ コントラクタの引数は親に渡してください """
         super().__init__(*args)

         # self.config に上で定義した MyConfig が入ります
         self.config = cast(MyConfig, self.config)

      def length(self) -> int:
         """ メモリに保存されている数を返します """
         raise NotImplementedError()

      # call_restore/call_backupで復元できるように作成します
      def call_restore(self, data, **kwargs) -> None:
         self.buffer = data
      def call_backup(self, **kwargs):
         return self.buffer

      # -------
      # その他、好きな関数を定義可能です。
      # -------


SequenceRemoteMemory
^^^^^^^^^^^^^^^^^^^^^^^^^^

順番通りにサンプルを取り出します RemoteMemory です。  
サンプルは取り出すとなくなります。  

ハイパーパラメータは特にないので継承するだけで実装できます。
継承すると add 関数と sample 関数が追加されるのでそれを使います。

.. literalinclude:: custom_algorithm1.py


ExperienceReplayBuffer
^^^^^^^^^^^^^^^^^^^^^^^^^^

ランダムにサンプルを取り出すRemoteMemoryです。
継承すると add 関数と sample 関数が追加されるのでそれを使います。

ハイパーパラメータがあるのでRLConfigでハイパーパラメータの追加が必要です。

.. literalinclude:: custom_algorithm2.py


PriorityExperienceReplay
^^^^^^^^^^^^^^^^^^^^^^^^^^

優先順位に従ってサンプルを取り出すRemoteMemoryです。  
継承すると add,sample,update,on_step関数が追加されます。
（on_stepはworkerで使用）

ハイパーパラメータもあり、少し複雑なのでコードで説明します。

.. literalinclude:: custom_algorithm3.py


IPriorityMemoryConfig
^^^^^^^^^^^^^^^^^^^^^^^^^^

PriorityExperienceReplayはアルゴリズムがいくつかあるので、IPriorityMemoryConfig で切り替えれるようにしています。
具体的なアルゴリズムは以下です。

.. list-table::
   :widths: 15 50
   :header-rows: 1

   * - クラス名
     - 説明
   * - ReplayMemoryConfig
     - ExperienceReplayBufferと同じで、ランダムに取得します。（優先順位はありません）
   * - ProportionalMemoryConfig
     - サンプルの重要度によって確率が変わります。重要度が高いサンプルほど選ばれる確率が上がります。
   * - RankBaseMemoryConfig
     - サンプルの重要度のランキングによって確率が変わります。重要度が高いサンプルほど選ばれる確率が上がるのはProportionalと同じです。


Parameter
--------------------------------------------

パラメータを管理するクラスです。
深層学習の場合はここにニューラルネットワークを定義することを想定しています。

実装が必要な関数は以下です。


.. code-block:: python

   from srl.base.rl.base import RLParameter

   import numpy as np

   class MyParameter(RLParameter):
      def __init__(self, *args):
         """ コントラクタの引数は親に渡してください """
         super().__init__(*args)

         # self.config に上で定義した MyConfig が入ります
         self.config = cast(MyConfig, self.config)

      # call_restore/call_backupでパラメータが復元できるように作成
      def call_restore(self, data, **kwargs) -> None:
         raise NotImplementedError()
      def call_backup(self, **kwargs):
         raise NotImplementedError()

      # その他任意の関数を作成できます
      # （パラメータに関するTrainer/Workerで共通する処理など）



Trainer
--------------------------

学習を定義する部分です。  
RemoteMemory から経験を受け取ってParameterを更新します。  

実装が必要な関数は以下です。

.. code-block:: python

   from srl.base.rl.base import RLTrainer

   class MyTrainer(RLTrainer):
      def __init__(self, *args):
         """ コントラクタの引数は親に渡してください """
         super().__init__(*args)

         # config,parameter,memory がそれぞれ入ります。
         self.config = cast(MyConfig, self.config)
         self.parameter = cast(MyParameter, self.parameter)
         self.remote_memory = cast(MyRemoteMemory, self.remote_memory)

      def get_train_count(self) -> int:
         """ 学習回数を返します """
         raise NotImplementedError()

      def train(self) -> dict:
         """
         RemoteMemory からbatchを受け取ってParameterを更新する形で学習を定義します。
         戻り値は任意で、学習情報を辞書形式で返します。
         """
         raise NotImplementedError()



Worker
--------------------------------------------

実際に環境と連携して経験を収集するクラスです。  
役割は、Parameterを参照してアクションを決める事と、サンプルをRemoteMemoryに送信する事です。  
  
RLWorkerとRLTrainerのフローをすごく簡単に書くと以下です。  

.. code-block:: python

   env.reset()
   worker.on_reset()
   while:
      action = worker.policy()
      env.step(action)
      worker.on_step()
      trainer.train()

RLWorkerもアルゴリズムに合わせたインタフェースのクラスを用意しています。  
基本はそちらを使用してください。
現状あるクラスは以下です。

.. list-table::
   :widths: 15 30 10
   :header-rows: 1

   * - クラス名
     - 説明
     - 
   * - DiscreteActionWorker
     - モデルフリーでアクションが離散値のアルゴリズム
     - Q学習,DQN等
   * - ContinuousActionWorker
     - モデルフリーでアクションが連続値のアルゴリズム
     - DDPG,SAC等
   * - ModelBaseWorker
     - 上記以外のアルゴリズム
     - 


DiscreteActionWorker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DiscreteActionWorkerで実装が必要な関数は以下です。

.. code-block:: python

   from srl.base.rl.algorithms.discrete_action import DiscreteActionWorker

   class MyWorker(DiscreteActionWorker):
      def __init__(self, *args):
         """ コントラクタの引数は親に渡してください """
         super().__init__(*args)

         # config,parameter,memory がそれぞれ入ります。
         self.config = cast(MyConfig, self.config)
         self.parameter = cast(MyParameter, self.parameter)
         self.remote_memory = cast(MyRemoteMemory, self.remote_memory)

      def call_on_reset(self, state: np.ndarray, invalid_actions: list[int]) -> dict:
         """ エピソードの最初に呼ばれる関数

         Args:
               state (np.ndarray): 環境の初期状態
               invalid_actions (List[int]): 初期状態での有効でないアクションのリスト
         
         Returns:
               Info: 任意の情報
         """
         raise NotImplementedError()

      def call_policy(self, state: np.ndarray, invalid_actions: list[int]) -> tuple[int, dict]:
         """ このターンで実行するアクションを返す関数

         Args:
               state (np.ndarray): 現在の状態
               invalid_actions (List[int]): 現在の有効でないアクションのリスト

         Returns: (
               int : 実行するアクション
               Info: 任意の情報
         )
         """
         raise NotImplementedError()

      def call_on_step(
         self,
         next_state: np.ndarray,
         reward: float,
         done: bool,
         next_invalid_actions: list[int],
      ) -> dict:
         """ Envが1step実行した後に呼ばれる関数

         Args:
               next_state (np.ndarray): アクション実行後の状態
               reward (float): アクション実行後の報酬
               done (bool): アクション実行後の終了状態
               next_invalid_actions (List[int]): アクション実行後の有効でないアクションのリスト

         Returns:
               dict: 情報(任意)
         """
         raise NotImplementedError()


ContinuousActionWorker
^^^^^^^^^^^^^^^^^^^^^^^^^^^

ContinuousActionWorkerで実装が必要な関数は以下です。

.. code-block:: python

   from srl.base.rl.algorithms.continuous_action import ContinuousActionWorker

   class MyWorker(ContinuousActionWorker):
      def __init__(self, *args):
         """ コントラクタの引数は親に渡してください """
         super().__init__(*args)

         # config,parameter,memory がそれぞれ入ります。
         self.config = cast(MyConfig, self.config)
         self.parameter = cast(MyParameter, self.parameter)
         self.remote_memory = cast(MyRemoteMemory, self.remote_memory)

      def call_on_reset(self, state: np.ndarray) -> Info:
         """ エピソードの最初に呼ばれる関数

         Args:
               state (np.ndarray): 環境の初期状態
         
         Returns:
               Info: 任意の情報
         """
         raise NotImplementedError()

      def call_policy(self, state: np.ndarray) -> Tuple[List[float], Info]:
         """ このターンで実行するアクションを返す関数

         Args:
               state (np.ndarray): 現在の状態

         Returns: (
               List[float]: 実行するアクション(小数の配列)
               Info       : 任意の情報
         )
         """
         raise NotImplementedError()

      def call_on_step(self, next_state: np.ndarray, reward: float, done: bool) -> dict:
         """ Envが1step実行した後に呼ばれる関数

         Args:
               next_state (np.ndarray): アクション実行後の状態
               reward (float): アクション実行後の報酬
               done (bool): アクション実行後の終了状態

         Returns:
               dict: 情報(任意)
         """
         raise NotImplementedError()


ModelBaseWorker
^^^^^^^^^^^^^^^^^^^^^^^^^^

ModelBaseWorker は実行時のクラスである EnvRun、WorkerRun を直接操作して実装するクラスです。  
（直接環境を操作できます）  
出来ることが多いのと、仕様が変わる可能性大きいので詳細は一旦保留します。(TODO)

実装が必要な関数は以下です。

.. code-block:: python

   from srl.base.rl.algorithms.modelbase import ModelBaseWorker

   from srl.base.env.base import EnvRun
   from srl.base.rl.worker import RLWorker, WorkerRun

   class MyWorker(ModelBaseWorker):
      def __init__(self, *args):
         """ コントラクタの引数は親に渡してください """
         super().__init__(*args)

         # config,parameter,memory がそれぞれ入ります。
         self.config = cast(MyConfig, self.config)
         self.parameter = cast(MyParameter, self.parameter)
         self.remote_memory = cast(MyRemoteMemory, self.remote_memory)

      def call_on_reset(self, state: np.ndarray, env: EnvRun, worker: WorkerRun) -> dict:
         """ エピソードの最初に呼ばれる関数

         Args:
               state (np.ndarray): 環境の初期状態
               env: EnvRun
               worker: WorkerRun
            
         Returns:
               Info: 任意の情報
         """
         raise NotImplementedError()

      def call_policy(self, state: np.ndarray, env: EnvRun, worker: WorkerRun) -> tuple[RLActionType, dict]:
         """ このターンで実行するアクションを返す関数

         Args:
               state (np.ndarray): 現在の状態
               env: EnvRun
               worker: WorkerRun

         Returns: (
               RLAction: 実行するアクション
               Info    : 任意の情報
         )
         """
         raise NotImplementedError()

      def call_on_step(
         self,
         next_state: np.ndarray,
         reward: float,
         done: bool,
         env: EnvRun,
         worker: WorkerRun,
      ) -> dict:
         """ Envが1step実行した後に呼ばれる関数

         Args:
               next_state (np.ndarray): アクション実行後の状態
               reward (float): アクション実行後の報酬
               done (bool): アクション実行後の終了状態
               env: EnvRun
               worker: WorkerRun

         Returns:
               dict: 情報(任意)
         """
         raise NotImplementedError()


Worker共通のプロパティ・関数
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Workerで共通して持っているプロパティ・関数は以下となります。

.. code-block:: python

   from typing import Optional
   from srl.base.define import RLAction

   class RLWorker:
      @property
      def training(self) -> bool:
         """ training かどうかを返します """
         return self._training

      @property
      def distributed(self) -> bool:
         """ 分散実行中かどうかを返します """
         return self._distributed

      def render_terminal(self, env, worker, **kwargs) -> None:
         """ 
         描画用の関数です。
         実装するとrenderによる描画が可能になります。
         """
         pass

      def render_rgb_array(self, env, worker, **kwargs) -> Optional[np.ndarray]:
         """ 
         描画用の関数です。
         実装するとrenderによる描画が可能になります。
         """
         return None

      @property
      def player_index(self) -> int:
         """ 複数人実行する環境にて、自分のプレイヤー番号を返します """
         return self._player_index

      def get_invalid_actions(self, env=None) -> List[RLAction]:
         """ 有効でないアクションを返します(離散限定) """
         return invalid_actions
      
      def get_valid_actions(self, env=None) -> List[RLAction]:
         """ 有効なアクションを返します(離散限定) """
         return valid_actions
      
      def sample_action(self, env=None) -> RLAction:
         """ ランダムなアクションを返します """
         return action


自作アルゴリズムの登録
=========================

以下で登録します。  
第2引数以降の entry_point は、`モジュールパス + ":" + クラス名`で、  
モジュールパスは `importlib.import_module` で呼び出せる形式である必要があります。

.. code-block:: python

   from srl.base.rl.registration import register
   register(
      MyConfig(),
      __name__ + ":MyRemoteMemory",
      __name__ + ":MyParameter",
      __name__ + ":MyTrainer",
      __name__ + ":MyWorker",
   )


実装例(Q学習)
=================

.. literalinclude:: custom_algorithm4.py

renderの表示例

.. code-block:: text

   ### 0, action None, rewards[0.000] (0.0s)
   env   {}
   work0 None
   ......
   .   G.
   . . X.
   .P   .
   ......

   ←: 0.17756
   ↓: 0.16355
   →: 0.11174
   *↑: 0.37473
   ### 1, action 3(↑), rewards[-0.040] (0.0s)
   env   {}
   work0 {}
   ......
   .   G.
   .P. X.
   .    .
   ......

   ←: 0.27779
   ↓: 0.20577
   →: 0.27886
   *↑: 0.49146
   ### 2, action 3(↑), rewards[-0.040] (0.0s)
   env   {}
   work0 {}
   ......
   .P  G.
   . . X.
   .    .
   ......

   ←: 0.34255
   ↓: 0.29609
   *→: 0.61361
   ↑: 0.34684
   ### 3, action 2(→), rewards[-0.040] (0.0s)
   env   {}
   work0 {}
   ......
   .   G.
   .P. X.
   .    .
   ......

   ←: 0.27779
   ↓: 0.20577
   →: 0.27886
   *↑: 0.49146
   ### 4, action 3(↑), rewards[-0.040] (0.0s)
   env   {}
   work0 {}
   ......
   .P  G.
   . . X.
   .    .
   ......

   ←: 0.34255
   ↓: 0.29609
   *→: 0.61361
   ↑: 0.34684
   ### 5, action 2(→), rewards[-0.040] (0.0s)
   env   {}
   work0 {}
   ......
   . P G.
   . . X.
   .    .
   ......

   ←: 0.37910
   ↓: 0.44334
   *→: 0.76733
   ↑: 0.46368
   ### 6, action 2(→), rewards[-0.040] (0.0s)
   env   {}
   work0 {}
   ......
   .  PG.
   . . X.
   .    .
   ......

   ←: 0.47941
   ↓: 0.39324
   *→: 0.92425
   ↑: 0.59087
   ### 7, action 2(→), rewards[1.000], done(env) (0.0s)
   env   {}
   work0 {}
   ......
   .   P.
   . . X.
   .    .
   ......

   [0.760000005364418]

.. image:: custom_algorithm4.gif
