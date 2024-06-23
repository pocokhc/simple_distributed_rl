.. _custom_algorithm:

=========================
Make Original Algorithm
=========================

ここでは本フレームワークでの自作アルゴリズムを作成する方法を説明します。

+ 1.概要  
+ 2.実装するクラスの説明  
   + 2-1.Config
   + 2-2.Memory
   + 2-3.Parameter
   + 2-4.Trainer
   + 2-5.Worker
+ 3.自作アルゴリズムの登録
+ 4.型アノテーション
+ 5.Q学習実装例


1. 概要
==========

| 自作アルゴリズムは5つクラスが以下のように連携して動作します。
| ※図にはないですが、他にハイパーパラメータを管理するConfigクラスがあります
| ※WorkerRunとEnvRunはフレームワーク内の内容になるので意識する必要はありません

.. image:: ../../diagrams/overview-sequence.drawio.png

| それぞれの役割は以下です。

.. list-table::
   :widths: 8 30
   :header-rows: 0

   * - Config
     - + ハイパーパラメータを管理
       + 実行時のSpace情報を管理
   * - Memory
     - + Workerが収集したサンプルを管理
   * - Parameter
     - + 学習パラメータを保持
   * - Trainer
     - + Memoryからサンプルを取得し学習する
       + 学習後、Parameterを更新する
   * - Worker
     - + Environmentと連携しサンプルを収集
       + 収集したサンプルをMemoryに送信
       + 行動決定に必要な情報をParameterから読む


分散学習は以下となり各クラスが非同期で動作します。

.. image:: ../../diagrams/overview-mp.drawio.png

同期的な学習と以下の点が異なります。  

+ WorkerがMemoryにサンプルを送るタイミングとTrainerが取り出すタイミングが異なる
+ ParameterがWorkerとTrainerで同期されない

各クラスの実装の仕方を見ていきます。


2. 実装する各クラスの説明
================================

2-1. Config
--------------------------------------------

強化学習アルゴリズムの種類やハイパーパラメータを管理するクラスです。  
基底クラスは `srl.base.rl.base.RLConfig` でこれを継承して作成します。  

RLConfig で実装が必要な関数・プロパティは以下です。  

.. code-block:: python

   from dataclasses import dataclass
   from srl.base.rl.config import RLConfig
   from srl.base.define import RLBaseTypes
   from srl.base.rl.processor import Processor

   # 必ず dataclass で書いてください
   @dataclass
   class MyConfig(RLConfig):

      # 任意のハイパーパラメータを定義
      hoo_param: float = 0
      
      def __post_init__(self):
         super().__post_init__()  # 親のコンストラクタも呼んでください

      def get_name(self) -> str:
         """ ユニークな名前を返す """
         raise NotImplementedError()

      def get_base_action_type(self) -> RLBaseTypes:
         """
         アルゴリズムが想定するアクションのタイプ(srl.base.define.RLBaseTypes)を返してください。
         """
         raise NotImplementedError()

      def get_base_observation_type(self) -> RLBaseTypes:
         """
         アルゴリズムが想定する環境から受け取る状態のタイプ(srl.base.define.RLBaseTypes)を返してください。
         """
         raise NotImplementedError()

      def get_framework(self) -> str:
         """
         使うフレームワークを指定してください。
         return ""           : なし
         return "tensorflow" : Tensorflow
         return "torch"      : Torch
         """
         raise NotImplementedError()

      # ------------------------------------
      # 以下は option です。（なくても問題ありません）
      # ------------------------------------
      def assert_params(self) -> None:
         """ パラメータのassertを記載 """
         super().assert_params()  # 親クラスも呼び出してください

      def setup_from_env(self, env: EnvRun) -> None:
         """ env初期化後に呼び出されます。env関係の初期化がある場合は記載してください。 """
         pass
         
      def setup_from_actor(self, actor_num: int, actor_id: int) -> None:
         """ 分散学習でactorが指定されたときに呼び出されます。Actor関係の初期化がある場合は記載してください。 """
         pass

      def get_processors(self) -> List[Optional[Processor]]:
         """ 前処理を追加したい場合設定してください """
         return []

      def get_used_backup_restore(self) -> bool:
         """ MCTSなど、envのbackup/restoreを使う場合はTrueを返してください"""
         return False



2-2. Memory
--------------------------------------------

| Workerが取得したサンプル(batch)をTrainerに渡す役割を持っているクラスです。
| 以下の3種類から継承します。
| （RLMemoryを直接継承することでオリジナルのMemoryを作成することも可能です）
| （オリジナルのMemoryの作成例は`srl.algorithms.world_models`の実装を参考にしてください）

.. list-table::
   :widths: 15 30
   :header-rows: 0

   * - SequenceMemory
     - 来たサンプルを順序通りに取り出します。(Queueみたいな動作です)
   * - ExperienceReplayBuffer
     - サンプルをランダムに取り出します。
   * - PriorityExperienceReplay
     - サンプルを優先順位に従い取り出します。


SequenceMemory
^^^^^^^^^^^^^^^^^^^^^^^^^^

順番通りにサンプルを取り出しますMemoryです。サンプルは取り出すとなくなります。

.. literalinclude:: custom_algorithm1.py


ExperienceReplayBuffer
^^^^^^^^^^^^^^^^^^^^^^^^^^

| ランダムにサンプルを取り出すMemoryです。
| これを使う場合は、Configに `RLConfigComponentExperienceReplayBuffer` を継承する必要があります。

.. literalinclude:: custom_algorithm2.py


PriorityExperienceReplay
^^^^^^^^^^^^^^^^^^^^^^^^^^

| 優先順位に従ってサンプルを取り出すMemoryです。
| これを使う場合は、Configにも `RLConfigComponentPriorityExperienceReplay` を継承する必要があります。

このアルゴリズムはConfigにより切り替えることができます。

.. list-table::
   :widths: 15 50
   :header-rows: 1

   * - クラス名
     - 説明
   * - ReplayMemory
     - ExperienceReplayBufferと同じで、ランダムに取得します。（優先順位はありません）
   * - ProportionalMemory
     - サンプルの重要度によって確率が変わります。重要度が高いサンプルほど選ばれる確率が上がります。
   * - RankBaseMemory
     - サンプルの重要度のランキングによって確率が変わります。重要度が高いサンプルほど選ばれる確率が上がるのはProportionalと同じです。

.. literalinclude:: custom_algorithm3.py


2-3. Parameter
--------------------------------------------

| パラメータを管理するクラスです。
| 深層学習の場合はここにニューラルネットワークを定義することを想定しています。

実装が必要な関数は以下です。


.. code-block:: python

   from srl.base.rl.parameter import RLParameter

   import numpy as np

   class MyParameter(RLParameter):
      def __init__(self, *args):
         """ コントラクタの引数は親に渡してください """
         super().__init__(*args)

         # self.config に上で定義した MyConfig が入ります
         self.config: MyConfig

      # call_restore/call_backupでパラメータが復元できるように作成
      def call_restore(self, data, **kwargs) -> None:
         raise NotImplementedError()
      def call_backup(self, **kwargs):
         raise NotImplementedError()

      # その他任意の関数を作成できます。
      # 分散学習ではTrainer/Worker間で値を保持できない点に注意（backup/restoreした値のみ共有されます）


2-4. Trainer
--------------------------

| 学習を定義する部分です。
| Memoryから経験を受け取ってParameterを更新します。  

実装が必要な関数は以下です。

.. code-block:: python

   from srl.base.rl.trainer import RLTrainer

   class MyTrainer(RLTrainer):
      def __init__(self, *args):
         """ コントラクタの引数は親に渡してください """
         super().__init__(*args)

         # 以下の変数を持ちます。
         self.config: MyConfig
         self.parameter: MyParameter
         self.memory: IRLMemoryTrainer

      def train(self) -> None:
         """
         self.memory から batch を受け取り学習を定義します。
         self.memory は以下の関数が定義されています。

         self.memory.is_warmup_needed() : warmup中かどうかを返します
         self.memory.sample()           : batchを返します
         self.memory.update()           : ProportionalMemory の場合 update で使います

         ・学習したら回数を数えてください
         self.train_count += 1

         ・(option)必要に応じてinfoを設定します
         self.info = {"loss": 0.0}
         """
         raise NotImplementedError()
      

   # --- 実装時に関数内で使う事を想定しているプロパティ・関数となります
   trainer = MyTrainer()
   trainer.distributed  # property, bool : 分散実行中かどうかを返します
   trainer.train_only   # property, bool : 学習のみかどうかを返します


Worker
--------------------------------------------

| 実際に環境と連携して経験を収集するクラスです。
| 役割は、Parameterを参照してアクションを決める事と、サンプルをMemoryに送信する事です。

フローをすごく簡単に書くと以下です。

.. code-block:: python

   env.reset()
   worker.on_reset()
   while:
      action = worker.policy()
      env.step(action)
      worker.on_step()
      trainer.train()

※v0.15.0からRLWorkerを直接継承する方法に変更しました
※v0.16.0からInfoが戻り値ではなく、内部変数になりました。

.. code-block:: python

   from srl.base.rl.worker import RLWorker
   from srl.base.rl.worker_run import WorkerRun

   class MyWorker(RLWorker):
      def __init__(self, *args):
         """ コントラクタの引数は親に渡してください """
         super().__init__(*args)

         # 以下の変数が設定されます
         self.config: MyConfig 
         self.parameter: MyParameter
         self.memory: IRLMemoryWorker

      def on_reset(self, worker: WorkerRun):
         """ エピソードの最初に呼ばれる関数 """
         raise NotImplementedError()

      def policy(self, worker: WorkerRun) -> RLActionType:
         """ このターンで実行するアクションを返す関数、この関数のみ実装が必須になります

         Returns:
               RLActionType : 実行するアクション
         """
         raise NotImplementedError()

      def on_step(self, worker: WorkerRun):
         """ Envが1step実行した後に呼ばれる関数 """
         raise NotImplementedError()

      def render_terminal(self, worker, **kwargs) -> None:
         """ 
         描画用の関数です。
         実装するとrenderによる描画が可能になります。
         """
         pass

      def render_rgb_array(self, worker, **kwargs) -> Optional[np.ndarray]:
         """ 
         描画用の関数です。
         実装するとrenderによる描画が可能になります。
         """
         return None

   # --- 実装時に関数内で使う事を想定しているプロパティ・関数となります
   worker = MyWorker()
   worker.training     # property, bool : training かどうかを返します
   worker.distributed  # property, bool : 分散実行中かどうかを返します
   worker.rendering    # property, bool : renderがあるエピソードかどうかを返します
   worker.observation_space  # property , SpaceBase : RLWorkerが受け取るobservation_spaceを返します
   worker.action_space       # property , SpaceBase : RLWorkerが受け取るaction_spaceを返します
   worker.get_invalid_actions() # function , List[RLAction] : 有効でないアクションを返します(離散限定)
   worker.sample_action()       # function , RLAction : ランダムなアクションを返します

また、情報は WorkerRun から基本取り出して使います。
情報の例は以下です。

.. code-block:: python

   class MyWorker(RLWorker):
      def on_reset(self, worker):
         worker.state           # 初期状態
         worker.player_index    # 初期プレイヤーのindex
         worker.invalid_action  # 初期有効ではないアクションリスト

      def policy(self, worker) :
         worker.state           # 状態
         worker.player_index    # プレイヤーのindex
         worker.invalid_action  # 有効ではないアクションリスト

      def on_step(self, worker: "WorkerRun") -> dict:
         worker.prev_state      # step前の状態(policyのworker.stateと等価)
         worker.state           # step後の状態
         worker.prev_action     # step前の前のアクション
         worker.action          # step前のアクション(policyで返したアクションと等価)
         worker.reward          # step後の即時報酬
         worker.done            # step後に終了フラグが立ったか
         worker.terminated      # step後にenvが終了フラグを立てたか
         worker.player_index    # 次のプレイヤーのindex
         worker.prev_invalid_action  # step前の有効ではないアクションリスト
         worker.invalid_action       # step後の有効ではないアクションリスト


3. 自作アルゴリズムの登録
=========================

以下で登録します。  
第2引数以降の entry_point は、`モジュールパス + ":" + クラス名`で、  
モジュールパスは `importlib.import_module` で呼び出せる形式である必要があります。

.. code-block:: python

   from srl.base.rl.registration import register
   register(
      MyConfig(),
      __name__ + ":MyMemory",
      __name__ + ":MyParameter",
      __name__ + ":MyTrainer",
      __name__ + ":MyWorker",
   )


4. 型アノテーション
=========================

動作に影響はないですが、ジェネリック型を追加し実装を簡単にしています。
適用方法は以下です。

.. code-block:: python

   @dataclass
   class Config(RLConfig):
      pass

   # RLParameter[TConfig]
   #   TConfig : RLConfig型
   class Parameter(RLParameter[Config]):
      pass

   # RLTrainer[TConfig, _TParameter]
   #   TConfig    : RLConfig型
   #   TParameter : RLParameter型
   class Trainer(RLTrainer[Config, Parameter]):
      pass

   # RLWorker[TConfig, _TParameter]
   #   TConfig    : RLConfig型
   #   TParameter : RLParameter型
   class Worker(RLWorker[Config, Parameter]):
      pass



5. 実装例(Q学習)
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
