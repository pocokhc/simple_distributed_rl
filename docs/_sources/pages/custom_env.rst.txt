.. _custom_env:

===========================
Create Original Environment
===========================

ここでは本フレームワークでの自作環境を作成する方法を説明します。  
構成としては大きく以下です。  

+ 1.実装するクラスの説明
   + 1-1.OpenAI gymクラス
   + 1-2.Baseクラス
   + 1-3.一人プレイ（singleplay）
   + 1-4.ターン制二人プレイ(turnbase2player)
+ 2.Spaceクラスの説明
+ 3.登録
+ 4.実装例(singleplay)
+ 5.テスト
+ 6.Q学習による学習



実装クラス
==========

自作環境を作成する方法は現在以下の二つがあります。

+ (OpenAI gym)[https://github.com/openai/gym]（以下gym）の環境を用意する
+ 本フレームワークで実装されている 'srl.base.env.EnvBase' を継承する

それぞれについて説明します。


gym の環境を利用
--------------------------------------------

gym の環境を利用を利用する場合は別途 `pip install gym` が必要です。  
また、一部本フレームワークに特化した内容にはならない点があります。  
1-2で説明があるBaseクラスを見比べてもらえれば分かりますが、アルゴリズム向けに特化した情報が増えています。  

※gym側の更新等により不正確な情報になる可能性があります。より正確な情報は公式(<https://github.com/openai/gym>)を見てください。

具体的なコードは以下です。
gym.Envを継承して作成します。

.. code-block:: python

   import gym
   from gym import spaces
   import numpy as np

   class MyGymEnv(gym.Env):
      # 利用できるrender_modesを指定するようです
      metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 4}
      
      def __init__(self, render_mode: str | None = None):
         self.render_mode = render_mode
         """
         initで以下2つの変数を定義する必要あり
         spaces.Space型については省略します。
         
         self.action_space      : アクションが取りうる範囲を指定
         self.observation_space : 状態が取りうる範囲を指定
         """
         
         self.action_space: spaces.Space = spaces.Discrete(2)
         self.observation_space: spaces.Space = spaces.Box(-1, 1, shape=(1,))

      def reset(self, *, seed=None, options=None)-> tuple[np.ndarray, dict]:
         super().reset(seed=seed)
         """ 1エピソードの最初に実行。（初期化処理を実装）
         return 初期状態, 情報
         """
         return np.array([0], dtype=np.float32), {}
      
      def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
         """ actionを元に1step進める処理を実装
         return (
               1step後の状態,
               即時報酬,
               予定通り終了したらTrue(terminated),
               予想外で終了したらTrue(truncated),
               情報(任意),
         )
         """
         return np.array([0], dtype=np.float32), 0.0, True, False, {}

      def render(self):
         """
         描画処理を書きます。
         """
         pass

環境クラスを作成したら登録します。
登録時のid名は "名前-vX" という形式である必要があります。

.. code-block:: python

   import gym.envs.registration

   gym.envs.registration.register(
      id="MyGymEnv-v0",
      entry_point=__name__ + ":MyGymEnv",
      max_episode_steps=10,
   )

以下のように呼び出せます。

.. code-block:: python

   import gym
   env = gym.make("MyGymEnv-v0")

   observation = env.reset()

   for _ in range(10):
      observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
      env.render()

      if terminated or truncated:
         observation = env.reset()

   env.close()


EnvBaseクラスを利用する方法
--------------------------------------------


本フレームワーク共通で使用する基底クラス(`srl.base.env.EnvBase`)の説明です。  
本フレームワークで環境を使う場合は、このクラスを継承している必要があります。

ただ、EnvBaseは複数人のプレイ環境も実装できるようにいくつか余分な情報が入っています。
これは作成する環境がどういったものかである程度決定できる情報があり、それに特化したクラスも用意しています。
環境を作成する場合は、以下のクラスから選んで継承し作成してください。

+ EnvBase         : 0から環境を作る場合に継承するクラス
+ SinglePlayEnv   : 一人用の環境を作る場合に継承するクラス
+ TurnBase2Player : 二人用ターン制の環境を作る場合に継承するクラス

EnvBase を継承した後に実装が必要な関数・プロパティは以下です。

.. code-block:: python

   from typing import Any
   from srl.base.env import EnvBase
   from srl.base.spaces.space import SpaceBase
   from srl.base.define import EnvActionType, EnvObservationType, EnvObservationTypes, InfoType


   class MyEnvBase(EnvBase):

      @property
      def action_space(self) -> SpaceBase:
         """ アクションの取りうる範囲を返します(SpaceBaseは後述) """
         raise NotImplementedError()

      @property
      def observation_space(self) -> SpaceBase:
         """ 状態の取りうる範囲を返します(SpaceBaseは後述) """
         raise NotImplementedError()

      @property
      def observation_type(self) -> EnvObservationTypes:
         """ 状態の種類を返します。
         EnvObservationType は列挙型で以下です。
         DISCRETE  : 離散
         CONTINUOUS: 連続
         GRAY_2ch  : グレー画像(2ch)
         GRAY_3ch  : グレー画像(3ch)
         COLOR     : カラー画像
         SHAPE2    : 2次元空間
         SHAPE3    : 3次元空間
         """
         raise NotImplementedError()

      @property
      def max_episode_steps(self) -> int:
         """ 1エピソードの最大ステップ数 """
         raise NotImplementedError()

      @property
      def player_num(self) -> int:
         """ プレイヤー人数 """
         raise NotImplementedError()

      def reset(self) -> tuple[EnvObservationType, InfoType]:
         """ 1エピソードの最初に実行。（初期化処理を実装）
         Returns: (
               init_state : 初期状態
               info       : 任意の情報
         )
         """
         raise NotImplementedError()

      def step(self, action: EnvActionType) -> tuple[EnvObservationType, list[float], bool, InfoType]:
         """ actionとplayerを元に1step進める処理を実装

         Args:
               action (EnvAction): player_index action

         Returns: (
               next_state : 1step後の状態
               [
                  player1 reward,  : player1の報酬
                  player2 reward,  : player2の報酬
                  ...
               ],
               done,  : episodeが終了したか
               info,  : 任意の情報
         )
         """
         raise NotImplementedError()

      @property
      def next_player_index(self) -> int:
         """ 次のプレイヤーindex """
         raise NotImplementedError()

      # backup/restore で現環境を復元できるように実装
      def backup(self) -> Any:
         raise NotImplementedError()
      def restore(self, data: Any) -> None:
         raise NotImplementedError()


一人プレイ用のクラス
--------------------------

一人プレイ用に特化した基底クラスの説明です。
（`srl.base.env.genre.singleplay.SinglePlayEnv`）

.. code-block:: python

   from typing import Any

   from srl.base.env.genre import SinglePlayEnv
   from srl.base.spaces.space import SpaceBase
   from srl.base.define import EnvObservationTypes, InfoType


   class MySinglePlayEnv(SinglePlayEnv):

      @property
      def action_space(self) -> SpaceBase:
         """ アクションの取りうる範囲を返します(SpaceBaseは後述) """
         raise NotImplementedError()

      @property
      def observation_space(self) -> SpaceBase:
         """ 状態の取りうる範囲を返します(SpaceBaseは後述) """
         raise NotImplementedError()

      @property
      def observation_type(self) -> EnvObservationTypes:
         """ 状態の種類を返します。
         EnvObservationType は列挙型で以下です。
         DISCRETE  : 離散
         CONTINUOUS: 連続
         GRAY_2ch  : グレー画像(2ch)
         GRAY_3ch  : グレー画像(3ch)
         COLOR     : カラー画像
         SHAPE2    : 2次元空間
         SHAPE3    : 3次元空間
         """
         raise NotImplementedError()

      @property
      def max_episode_steps(self) -> int:
         """ 1エピソードの最大ステップ数 """
         raise NotImplementedError()

      def call_reset(self) -> tuple[EnvObservationType, InfoType]:
         """ 1エピソードの最初に実行。（初期化処理を実装）
         Returns: (
               EnvObservation (observation_spaceの型): 初期状態
               Info                                 : 任意の情報
         )
         """
         raise NotImplementedError()

      def call_step(self, action) -> tuple[EnvObservationType, float, bool, InfoType]:
         """ actionを元に1step進める処理を実装
         Returns: (
               next_state : 1step後の状態
               reward     : 即時報酬
               done       : 終了したかどうか
               info       : 任意の情報
         )
         """
         raise NotImplementedError()

      # backup/restore で現環境を復元できるように実装
      def backup(self) -> Any:
         raise NotImplementedError()
      def restore(self, data: Any) -> None:
         raise NotImplementedError()


ターン制二人プレイ用のクラス
--------------------------------------------

ターン制二人プレイに特化した基底クラスの説明です。  
(`srl.base.env.genre.turnbase2player.TurnBase2Player`)  

.. code-block:: python

   from typing import Any

   from srl.base.env.genre import TurnBase2Player
   from srl.base.spaces.space import SpaceBase
   from srl.base.define import EnvObservationTypes, InfoType


   class MyTurnBase2Player(TurnBase2Player):

      @property
      def action_space(self) -> SpaceBase:
         """ アクションの取りうる範囲を返します(SpaceBaseは後述) """
         raise NotImplementedError()

      @property
      def observation_space(self) -> SpaceBase:
         """ 状態の取りうる範囲を返します(SpaceBaseは後述) """
         raise NotImplementedError()

      @property
      def observation_type(self) -> EnvObservationTypes:
         """ 状態の種類を返します。
         EnvObservationType は列挙型で以下です。
         DISCRETE  : 離散
         CONTINUOUS: 連続
         GRAY_2ch  : グレー画像(2ch)
         GRAY_3ch  : グレー画像(3ch)
         COLOR     : カラー画像
         SHAPE2    : 2次元空間
         SHAPE3    : 3次元空間
         """
         raise NotImplementedError()

      @property
      def max_episode_steps(self) -> int:
         """ 1エピソードの最大ステップ数 """
         raise NotImplementedError()

      @property
      def next_player_index(self) -> int:
         """ 次ターンのプレイヤー番号を返す """
         raise NotImplementedError()

      def call_reset(self) -> tuple[EnvObservationType, InfoType]:
         """ 1エピソードの最初に実行。（初期化処理を実装）
         Returns: (
               EnvObservation (observation_spaceの型): 初期状態
               Info                                 : 任意の情報
         )
         """
         raise NotImplementedError()


      def call_step(self, action) -> tuple[EnvObservationType, float, float, bool, InfoType]:
         """ action を元に1step進める処理を実装
         Returns: (
               next_state     : 1step後の状態
               player1 reward : player1の即時報酬
               player2 reward : player2の即時報酬
               done       : 終了したかどうか
               info       : 情報(任意)
         )
         """
         raise NotImplementedError()

      # backup/restore で現環境を復元できるように実装
      def backup(self) -> Any:
         raise NotImplementedError()
      def restore(self, data: Any) -> None:
         raise NotImplementedError()


その他オプション
--------------------------------------------

その他必須ではないですが、設定できる関数・プロパティとなります。

.. code-block:: python

   # --- 追加情報
   @property
   def reward_info(self) -> dict:
      """ 報酬に関する情報を返す """
      return {
         "min": None,
         "max": None,
         "baseline": None,
      }

   # --- 実行に関する関数
   def close(self) -> None:
      """ 終了処理を実装 """
      pass
   
   def get_invalid_actions(self, player_index: int) -> list:
      """ 無効なアクションがある場合は配列で返す
      （SinglePlayEnv では call_get_invalid_actions で設定）
      """
      return []

   def set_seed(self, seed: Optional[int] = None) -> None:
      """ 乱数のseedを設定 """
      pass

   # --- AI
   def make_worker(self, name: str) -> Optional["srl.base.rl.base.WorkerBase"]:
      """ 環境に特化したAIを返す """
      return None

   # --- 描画に関する関数
   def render_terminal(self, **kwargs) -> None:
      """ 現在の状況をprintで表示する用に実装 """
      pass

   def render_rgb_array(self, **kwargs) -> np.ndarray | None:
      """ 現在の状況を RGB の画像配列で返す """
      return None

   def action_to_str(self, action: Union[str, EnvActionType]) -> str:
      """ アクションを文字列に変換する """
      return str(action)

   @property
   def render_interval(self) -> float:
      """ 描画速度を返す """
      return 1000 / 60

   # --- プレイ時に関する関数
   def get_key_bind(self) -> KeyBindType:
      """ キー配置とアクションを紐づける """
      return None


Spaceクラスについて
====================

Spaceクラスは、アクション・状態の取りうる範囲を決めるクラスとなります。
現状5種類あり以下となります。

.. list-table::
   :widths: 5 20 40
   :header-rows: 1

   * - SpaceClass
     - 型
     - 概要
   * - DiscreteSpace
     - int
     - 1つの整数を表します。
       例えば DiscreteSpace(5) とした場合、0～4 の値を取ります。
   * -  ArrayDiscreteSpace
     - list[int]
     - 整数の配列を取ります。
       例えば ArrayDiscreteSpace(3, low=-1, high=2) とした場合、[-1, 1, 0] や [2, 1, -1] 等の値を取ります。
   * - ContinuousSpace
     - float
     - 1つの小数を表します。
       例えば ContinuousSpace(low=-1, high=1) とした場合、-1～1 の値を取ります。
   * - ArrayContinuousSpace
     - list[float]
     - 小数の配列を取ります。
       例えば ArrayContinuousSpace(3, low=-1, high=1) とした場合、[0.1, -0.5, 0.9] 等の値を取ります。
   * - BoxSpace
     - NDArray[np.float32]
     - numpy配列を指定の範囲内で取り扱います。
       例えば BoxSpace(shape=(1, 2), low=-1, high=1) とした場合、[[-0.1, 1.0]] や [[0.1, 0.2] 等の値を取ります。

詳細はコード(`srl.base.env.spaces`)を見てください。


自作環境の登録
====================

作成した環境は以下で登録して使います。
引数は以下です。

.. list-table::
    :widths: 5 20 40
    :header-rows: 1

    * - 引数
      - 説明
      - 備考
    * - id
      - ユニークな名前
      - 被らなければ特に制限はありません
    * - entry_point
      - `モジュールパス + ":" + クラス名`
      - モジュールパスは `importlib.import_module` で呼び出せる形式である必要があります
    * - kwargs
      - クラス生成時の引数
      - 


.. code-block:: python

   from srl.base.env import registration

   registration.register(
      id="SampleEnv",
      entry_point=__name__ + ":SampleEnv",
      kwargs={},
   )


実装例
====================

左右に動け、左が穴で右がゴールなシンプルな環境を実装します。

.. literalinclude:: ../../srl/envs/sample_env.py

実装した環境をとりあえず動かすコードは以下です。

.. literalinclude:: custom_env1.py


テスト
====================

最低限ですが、ちゃんと動くか以下でテストできます。

.. literalinclude:: custom_env2.py


Q学習による学習
====================

以下のように学習できます。

.. literalinclude:: custom_env3.py

.. image:: custom_env3.gif
