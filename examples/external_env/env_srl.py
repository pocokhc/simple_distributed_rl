"""
外部環境と同じ動作をするSRL用の環境
"""

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from env_external import ExternalEnv

from srl.base.env import registration
from srl.base.env.base import EnvBase
from srl.base.spaces.discrete import DiscreteSpace

registration.register(id="ExternalEnv", entry_point=__name__ + ":SrlExternalEnv")


@dataclass
class SrlExternalEnv(EnvBase[DiscreteSpace, int, DiscreteSpace, int]):
    @property
    def action_space(self):
        return DiscreteSpace(2)

    @property
    def observation_space(self):
        return DiscreteSpace(11, start=-5)

    @property
    def player_num(self) -> int:
        return 1

    @property
    def max_episode_steps(self) -> int:
        return 20

    # -----------------------------------------------------------
    # ・学習に使用
    # 何かしらの方法で外部環境を1step進めた状態を作り、結果を返す
    # -----------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, **kwargs) -> Any:
        self.env = ExternalEnv()
        return self.env.pos

    def step(self, action) -> Tuple[int, float, bool, bool]:
        self.env.step(action)
        return self.env.pos, self.env.reward, self.env.done, False

    # -----------------------------------------------------------
    # ・実行に使用
    # 1step進めた後の外部環境から来る状態(external_state)をSRLが認識できる形に変換
    # direct_step, decode_action の2つの関数を実装する必要あり
    # -----------------------------------------------------------
    def direct_step(self, step: int, external_state: int) -> Tuple[bool, int, bool]:
        """外部環境からくる情報を元にSRL用の情報を作成しreturnする"""
        is_start_episode = step == 0  # エピソードの開始かどうか
        is_end_episode = False  # エピソードの終了かどうか（分からなくても動作します）

        # 状態をSRL側が分かる形(observation_space型)に変換
        # ここでは外部環境の状態とobservation_spaceが両方intなので変換はなし
        srl_env_state = external_state

        # 複数プレイヤーがいる場合は現在のプレイヤーindexを指定
        self.next_player = 0

        return is_start_episode, srl_env_state, is_end_episode

    def decode_action(self, srl_env_action: int) -> int:
        """アクションを外部環境が分かる形に変換"""
        # SRL側のアクションを外部環境が分かる形に変換
        # ここではSRL側のアクション(action_space)と外部環境のアクションが両方intなので変換はなし
        external_action = srl_env_action
        return external_action

    @property
    def can_simulate_from_direct_step(self) -> bool:
        """
        外部環境側の中身を、ここから復元できない場合はFalse、今回は出来ない場合を想定
        Trueにし、backup/restoreを実装すると、MCTS等のbackup/restoreを利用するアルゴリズムが利用できます
        """
        return False
