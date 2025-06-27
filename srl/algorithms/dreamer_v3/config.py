from dataclasses import dataclass, field
from typing import Any, List, Tuple

from srl.base.define import SpaceTypes
from srl.base.rl.algorithms.base_ppo import RLConfig
from srl.base.rl.processor import RLProcessor
from srl.base.spaces.space import SpaceBase
from srl.rl.memories.replay_buffer import ReplayBufferConfig
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.lr_scheduler import LRSchedulerConfig

"""
paper: https://browse.arxiv.org/abs/2301.04104v1
ref: https://github.com/danijar/dreamerv3
"""


@dataclass
class Config(RLConfig):
    #: Batch size
    batch_size: int = 32
    #: <:ref:`ReplayBufferConfig`>
    memory: ReplayBufferConfig = field(default_factory=lambda: ReplayBufferConfig())

    # --- RSSM
    #: 決定的な遷移のユニット数、内部的にはGRUのユニット数
    rssm_deter_size: int = 4096
    #: 確率的な遷移のユニット数
    rssm_stoch_size: int = 32
    #: 確率的な遷移のクラス数（rssm_use_categorical_distribution=Trueの場合有効）
    rssm_classes: int = 32
    #: 隠れ状態のユニット数
    rssm_hidden_units: int = 1024
    #: Trueの場合、LayerNormalization層が追加されます
    rssm_use_norm_layer: bool = True
    #: Falseの場合、確率的な遷移をガウス分布、Trueの場合カテゴリカル分布で表現します
    rssm_use_categorical_distribution: bool = True
    #: RSSM Activation
    rssm_activation: Any = "silu"
    #: カテゴリカル分布で保証する最低限の確率（rssm_use_categorical_distribution=Trueの場合有効）
    rssm_unimix: float = 0.01

    # --- other model layers
    #: 学習する報酬の分布のタイプ
    #:
    #: Parameters:
    #:   "linear": MSEで学習（use_symlogの影響を受けます）
    #:   "normal": ガウス分布による学習（use_symlogの影響はうけません）
    #:   "normal_fixed_scale": ガウス分布による学習ですが、分散は1で固定（use_symlogの影響はうけません）
    #:   "twohot": TwoHotエンコーディングによる学習（use_symlogの影響を受けます）
    reward_type: str = "twohot"
    #: reward_typeが"twohot"の時のみ有効、bins
    reward_twohot_bins: int = 255
    #: reward_typeが"twohot"の時のみ有効、low
    reward_twohot_low: int = -20
    #: reward_typeが"twohot"の時のみ有効、high
    reward_twohot_high: int = 20
    #: reward modelの隠れ層
    reward_layer_sizes: Tuple[int, ...] = (1024, 1024, 1024, 1024)
    #: continue modelの隠れ層
    cont_layer_sizes: Tuple[int, ...] = (1024, 1024, 1024, 1024)
    #: critic modelの隠れ層
    critic_layer_sizes: Tuple[int, ...] = (1024, 1024, 1024, 1024)
    #: actor modelの隠れ層
    actor_layer_sizes: Tuple[int, ...] = (1024, 1024, 1024, 1024)
    #: 各層のactivation
    dense_act: Any = "silu"
    #: symlogを使用するか
    use_symlog: bool = True

    # --- encoder/decoder
    #: 入力がIMAGE以外の場合の隠れ層
    encoder_decoder_mlp: Tuple[int, ...] = (1024, 1024, 1024, 1024)
    #: decoder出力層の分布
    #:
    #: Parameters:
    #:   "linear": mse
    #:   "normal": 正規分布
    encoder_decoder_dist: str = "linear"
    #: [入力がIMAGEの場合] Conv2Dのユニット数
    cnn_depth: int = 96
    #: [入力がIMAGEの場合] ResBlockの数
    cnn_blocks: int = 0
    #: [入力がIMAGEの場合] activation
    cnn_activation: Any = "silu"
    #: [入力がIMAGEの場合] 正規化層を追加するか
    #:
    #: Parameters:
    #:  "none": 何もしません
    #:  "layer": LayerNormalization層が追加されます
    cnn_normalization_type: str = "layer"
    #: [入力がIMAGEの場合] 画像を縮小する際のアルゴリズム
    #:
    #: Parameters:
    #:   "stride": Conv2Dのスライドで縮小します
    #:   "stride3": Conv2Dの3スライドで縮小します
    cnn_resize_type: str = "stride"
    #: [入力がIMAGEの場合] 画像縮小後のサイズ
    cnn_resized_image_size: int = 4
    #: [入力がIMAGEの場合] Trueの場合、画像の出力層をsigmoidにします。Falseの場合はLinearです。
    cnn_use_sigmoid: bool = False

    # --- loss params
    #: free bit
    free_nats: float = 1.0  # 1nat ~ 1.44bit
    #: reconstruction loss rate
    loss_scale_pred: float = 1.0
    #: dynamics kl loss rate
    loss_scale_kl_dyn: float = 0.5
    #: rep kl loss rate
    loss_scale_kl_rep: float = 0.1
    #: 序盤はworld modelのみ学習します
    warmup_world_model: int = 0

    # --- actor/critic
    #: critic target update interval
    critic_target_update_interval: int = 0  # 0 is disable target
    #: critic target soft update tau
    critic_target_soft_update: float = 0.02
    #: critic model type
    #:
    #: Parameters:
    #:   "linear" : MSEで学習（use_symlogの影響を受けます）
    #:   "normal" : 正規分布（use_symlogの影響は受けません）
    #:   "normal_fixed_scale": 分散1固定の正規分布（use_symlogの影響は受けません）
    #:   "twohot" : TwoHotカテゴリカル分布（use_symlogの影響を受けます）
    critic_type: str = "twohot"
    #: critic_typeが"dreamer_v3"の時のみ有効、bins
    critic_twohot_bins: int = 255
    #: critic_typeが"dreamer_v3"の時のみ有効、low
    critic_twohot_low: int = -20
    #: critic_typeが"dreamer_v3"の時のみ有効、high
    critic_twohot_high: int = 20
    #: actor model type
    #:
    #: Parameters:
    #:   "categorical"        : カテゴリカル分布
    #:   "gumbel_categorical" : Gumbelカテゴリ分布
    actor_discrete_type: str = "categorical"
    #: カテゴリカル分布で保証する最低限の確率（actionタイプがDISCRETEの時のみ有効）
    actor_discrete_unimix: float = 0.01
    #: actionが連続値の時、正規分布をtanhで-1～1に丸めるか
    actor_continuous_enable_normal_squashed: bool = True

    # --- Behavior
    #: horizonのstep数
    horizon: int = 15
    #: "actor" or "random", random is debug.
    horizon_policy: str = "actor"
    #: horizon時の価値の計算方法
    #:
    #: Parameters:
    #:  "simple"   : 単純な総和
    #:  "discount" : 割引報酬
    #:  "ewa"      : EWA
    #:  "h-return" : λ-return
    critic_estimation_method: str = "h-return"
    #: EWAの係数、小さいほど最近の値を反映（"ewa"の時のみ有効）
    horizon_ewa_disclam: float = 0.1
    #: λ-returnの係数（"h-return"の時のみ有効）
    horizon_h_return: float = 0.95
    #: 割引率
    discount: float = 0.997

    # ---Training
    #: dynamics model training flag
    enable_train_model: bool = True
    #: critic model training flag
    enable_train_critic: bool = True
    #: actor model training flag
    enable_train_actor: bool = True
    #: batch length
    batch_length: int = 64
    #: dynamics model learning rate
    lr_model: float = 1e-4
    #: <:ref:`LRSchedulerConfig`>
    lr_model_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: critic model learning rate
    lr_critic: float = 3e-5
    #: <:ref:`LRSchedulerConfig`>
    lr_critic_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: actor model learning rate
    lr_actor: float = 3e-5
    #: <:ref:`LRSchedulerConfig`>
    lr_actor_scheduler: LRSchedulerConfig = field(default_factory=lambda: LRSchedulerConfig())
    #: loss計算の方法
    #:
    #: Parameters:
    #:   "dreamer_v1" : Vの最大化
    #:   "dreamer_v2" : Vとエントロピーの最大化
    #:   "dreamer_v3" : V2 + パーセンタイルによる正規化
    actor_loss_type: str = "dreamer_v3"
    #: actionがCONTINUOUSの場合のReinforceとDynamics backpropの比率
    actor_reinforce_rate: float = 0.0
    #: entropy rate
    entropy_rate: float = 0.0003
    #: baseline
    #:
    #: Parameters:
    #:   "v"  : -v
    #:   other: none
    reinforce_baseline: str = "v"

    # --- other
    #: action ε-greedy(for debug)
    epsilon: float = 0
    #: 報酬の前処理
    #:
    #: Parameters:
    #:   "none": なし
    #:   "tanh": tanh
    clip_rewards: str = "none"

    def get_changeable_parameters(self) -> List[str]:
        return [
            "free_nats",
            "loss_scale_pred",
            "loss_scale_kl_dyn",
            "loss_scale_kl_rep",
            "critic_target_update_interval",
            "critic_target_soft_update",
            "horizon",
            "horizon_policy",
            "critic_estimation_method",
            "horizon_ewa_disclam",
            "horizon_h_return",
            "discount",
            "enable_train_model",
            "enable_train_actor",
            "enable_train_critic",
            "batch_length",
            "batch_size",
            "lr_model",
            "lr_critic",
            "lr_actor",
            "actor_loss_type",
            "entropy_rate",
            "reinforce_baseline",
            "epsilon",
            "clip_rewards",
        ]

    def set_dreamer_v1(self):
        # --- RSSM
        self.rssm_deter_size = 200
        self.rssm_stoch_size = 30
        self.rssm_hidden_units = 400
        self.rssm_use_norm_layer = False
        self.rssm_use_categorical_distribution = False
        self.rssm_activation = "elu"
        self.rssm_unimix = 0
        # --- other model layers
        self.reward_type = "normal_fixed_scale"
        self.reward_layer_sizes = (400, 400)
        self.cont_layer_sizes = (400, 400)
        self.critic_layer_sizes = (400, 400, 400)
        self.actor_layer_sizes = (400, 400, 400, 400)
        self.dense_act = "elu"
        self.use_symlog = False
        # --- encoder/decoder
        self.encoder_decoder_mlp = (400, 400, 400, 400)
        self.encoder_decoder_dist = "normal"
        self.cnn_depth = 32
        self.cnn_blocks = 0
        self.cnn_resized_image_size = 1
        self.cnn_activation = "relu"
        self.cnn_normalization_type = "none"
        self.cnn_resize_type = "stride"
        self.cnn_resized_image_size = 4
        self.cnn_use_sigmoid = False
        # --- loss params
        self.free_nats = 3.0
        self.loss_scale_pred = 1.0
        self.loss_scale_kl_dyn = 0.5
        self.loss_scale_kl_rep = 0.5
        # --- actor/critic
        self.critic_target_update_interval = 0
        self.critic_type = "normal_fixed_scale"
        self.actor_discrete_unimix = 0
        # Behavior
        self.horizon = 15
        self.critic_estimation_method: str = "ewa"
        self.horizon_ewa_disclam = 0.1
        self.discount: float = 0.99
        # Training
        self.batch_size = 50
        self.batch_length = 50
        self.lr_model = 6e-4
        self.lr_critic = 8e-5
        self.lr_actor = 8e-5
        self.actor_loss_type = "dreamer_v1"

    def set_dreamer_v2(self):
        # --- RSSM
        self.rssm_deter_size = 1024
        self.rssm_stoch_size = 32
        self.rssm_classes = 32
        self.rssm_hidden_units = 1024
        self.rssm_use_norm_layer = False
        self.rssm_use_categorical_distribution = True
        self.rssm_activation = "elu"
        self.rssm_unimix = 0
        # --- other model layers
        self.reward_type = "linear"
        self.reward_layer_sizes = (400, 400, 400, 400)
        self.cont_layer_sizes = (400, 400, 400, 400)
        self.critic_layer_sizes = (400, 400, 400, 400)
        self.actor_layer_sizes = (400, 400, 400, 400)
        self.dense_act = "elu"
        self.use_symlog = False
        # --- encoder/decoder
        self.encoder_decoder_mlp = (400, 400, 400, 400)
        self.encoder_decoder_dist = "normal"
        self.cnn_depth = 48
        self.cnn_blocks = 0
        self.cnn_resized_image_size = 1
        self.cnn_activation = "relu"
        self.cnn_normalization_type = "none"
        self.cnn_resize_type = "stride"
        self.cnn_resized_image_size = 4
        self.cnn_use_sigmoid = False
        # --- loss params
        self.free_nats = 0.0
        self.loss_scale_pred = 1.0
        self.loss_scale_kl_dyn = 0.8
        self.loss_scale_kl_rep = 0.2
        # --- actor/critic
        self.critic_target_update_interval = 100
        self.critic_target_soft_update = 1
        self.critic_type = "normal_fixed_scale"
        self.actor_discrete_unimix = 0
        # Behavior
        self.horizon = 15
        self.critic_estimation_method: str = "h-return"
        self.horizon_h_return = 0.95
        self.discount: float = 0.99
        # Training
        self.batch_size = 16
        self.batch_length = 50
        self.lr_model = 1e-4
        self.lr_critic = 2e-4
        self.lr_actor = 8e-5
        self.actor_loss_type = "dreamer_v2"
        self.entropy_rate: float = 2e-3
        self.reinforce_baseline: str = "v"

    def set_dreamer_v3(self):
        # --- RSSM
        self.rssm_deter_size = 4096
        self.rssm_stoch_size = 32
        self.rssm_classes = 32
        self.rssm_hidden_units = 1024
        self.rssm_use_norm_layer = True
        self.rssm_use_categorical_distribution = True
        self.rssm_activation = "silu"
        self.rssm_unimix = 0.01
        # --- other model layers
        self.reward_type = "twohot"
        self.reward_twohot_bins = 255
        self.reward_twohot_low = -20
        self.reward_twohot_high = 20
        self.reward_layer_sizes = (1024, 1024, 1024, 1024)
        self.cont_layer_sizes = (1024, 1024, 1024, 1024)
        self.critic_layer_sizes = (1024, 1024, 1024, 1024)
        self.actor_layer_sizes = (1024, 1024, 1024, 1024)
        self.dense_act = "silu"
        self.use_symlog = True
        # --- encoder/decoder
        self.encoder_decoder_mlp = (1024, 1024, 1024, 1024, 1024)
        self.encoder_decoder_dist = "linear"
        self.cnn_depth = 96
        self.cnn_blocks = 0
        self.cnn_resized_image_size = 4
        self.cnn_activation = "silu"
        self.cnn_normalization_type = "layer"
        self.cnn_resize_type = "stride"
        self.cnn_resized_image_size = 4
        self.cnn_use_sigmoid = False
        # --- loss params
        self.free_nats = 1.0
        self.loss_scale_pred = 1.0
        self.loss_scale_kl_dyn = 0.5
        self.loss_scale_kl_rep = 0.1
        # --- actor/critic
        self.critic_target_update_interval = 0
        self.critic_type = "twohot"
        self.critic_twohot_bins = 255
        self.critic_twohot_low = -20
        self.critic_twohot_high = 20
        self.actor_discrete_unimix = 0.01
        # Behavior
        self.horizon = 333
        self.critic_estimation_method: str = "h-return"
        self.horizon_h_return = 0.95
        self.discount: float = 0.997
        # Training
        self.batch_size = 16
        self.batch_length = 64
        self.lr_model = 1e-4
        self.lr_critic = 3e-5
        self.lr_actor = 3e-5
        self.actor_loss_type = "dreamer_v3"
        self.entropy_rate: float = 3e-4
        self.reinforce_baseline: str = "v"

    def get_name(self) -> str:
        return "DreamerV3"

    def get_processors(self, prev_observation_space: SpaceBase) -> List[RLProcessor]:
        if prev_observation_space.is_image():
            if self.cnn_resize_type == "stride3":
                return [
                    ImageProcessor(
                        image_type=SpaceTypes.COLOR,
                        resize=(96, 96),
                        normalize_type="0to1",
                    )
                ]
            else:
                return [
                    ImageProcessor(
                        image_type=SpaceTypes.COLOR,
                        resize=(64, 64),
                        normalize_type="0to1",
                    )
                ]
        return []

    def get_framework(self) -> str:
        return "tensorflow"

    def validate_params(self) -> None:
        super().validate_params()
        if not (self.horizon >= 0):
            raise ValueError(f"assert {self.horizon} >= 0")
