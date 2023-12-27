import logging
import random
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

from srl.base.define import DoneTypes, EnvObservationTypes, RLBaseTypes, RLTypes
from srl.base.exception import UndefinedError
from srl.base.rl.base import RLParameter, RLTrainer, RLWorker
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.rl.registration import register
from srl.base.rl.worker_run import WorkerRun
from srl.rl.functions import common
from srl.rl.functions.common_tf import symexp, symlog
from srl.rl.memories.experience_replay_buffer import ExperienceReplayBuffer, ExperienceReplayBufferConfig
from srl.rl.models.tf.distributions.bernoulli_dist_block import BernoulliDistBlock
from srl.rl.models.tf.distributions.categorical_dist_block import CategoricalDistBlock, CategoricalGradDist
from srl.rl.models.tf.distributions.categorical_gumbel_dist_block import CategoricalGumbelDistBlock
from srl.rl.models.tf.distributions.linear_block import LinearBlock
from srl.rl.models.tf.distributions.normal_dist_block import NormalDistBlock
from srl.rl.models.tf.distributions.twohot_dist_block import TwoHotDistBlock
from srl.rl.processors.image_processor import ImageProcessor
from srl.rl.schedulers.scheduler import SchedulerConfig
from srl.utils.common import compare_less_version

kl = keras.layers
tfd = tfp.distributions

logger = logging.getLogger(__name__)

"""
paper: https://browse.arxiv.org/abs/2301.04104v1
ref: https://github.com/danijar/dreamerv3
"""


# ------------------------------------------------------
# config
# ------------------------------------------------------
@dataclass
class Config(RLConfig, ExperienceReplayBufferConfig):
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
    #:   "linear": MSEで学習（use_symlogの影響を受けます）
    #:   "twohot": TwoHotエンコーディングによる学習（use_symlogの影響を受けます）
    #:   "normal": ガウス分布による学習（use_symlogの影響はうけません）
    #:   "normal_fixed_stddev": ガウス分布による学習ですが、分散は1で固定（use_symlogの影響はうけません）
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
    #: [入力がIMAGEの場合] Conv2Dのユニット数
    cnn_depth: int = 96
    #: [入力がIMAGEの場合] ResBlockの数
    cnn_blocks: int = 0
    #: [入力がIMAGEの場合] activation
    cnn_activation: Any = "silu"
    #: [入力がIMAGEの場合] 正規化層を追加するか
    #:  "none": 何もしません
    #:  "layer": LayerNormalization層が追加されます
    cnn_normalization_type: str = "layer"
    #: [入力がIMAGEの場合] 画像を縮小する際のアルゴリズム
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

    # --- actor/critic
    #: critic target update interval
    critic_target_update_interval: int = 0  # 0 is disable target
    #: critic target soft update tau
    critic_target_soft_update: float = 0.02
    #: critic model type
    #:   "linear"    : MSEで学習（use_symlogの影響を受けます）
    #:   "normal"    : 正規分布（use_symlogの影響は受けません）
    #:   "dreamer_v1": 分散1固定の正規分布（use_symlogの影響は受けません）
    #:   "dreamer_v2": 分散1固定の正規分布（use_symlogの影響は受けません）
    #:   "dreamer_v3": TwoHotカテゴリカル分布（use_symlogの影響を受けます）
    critic_type: str = "dreamer_v3"
    #: critic_typeが"dreamer_v3"の時のみ有効、bins
    critic_twohot_bins: int = 255
    #: critic_typeが"dreamer_v3"の時のみ有効、low
    critic_twohot_low: int = -20
    #: critic_typeが"dreamer_v3"の時のみ有効、high
    critic_twohot_high: int = 20
    #: actor model type
    #:   "categorical"        : カテゴリカル分布
    #:   "gumbel_categorical" : Gumbelカテゴリ分布
    actor_discrete_type: str = "categorical"
    #: カテゴリカル分布で保証する最低限の確率（actionタイプがDISCRETEの時のみ有効）
    actor_discrete_unimix: float = 0.01

    # --- Behavior
    #: horizonのstep数
    horizon: int = 15
    #: "actor" or "random", random is debug.
    horizon_policy: str = "actor"
    #: horizon時の価値の計算方法
    #:  "simple"    : 単純な総和
    #:  "discount"  : 割引報酬
    #:  "dreamer_v1": EWA
    #:  "dreamer_v2": λ-return
    #:  "dreamer_v3": λ-return
    critic_estimation_method: str = "dreamer_v3"
    #: EWAの係数、小さいほど最近の値を反映（critic_estimation_method=="dreamer_v1"の時のみ有効）
    horizon_ewa_disclam: float = 0.1
    #: λ-returnの係数（critic_estimation_method=="dreamer_v2,v3"の時のみ有効）
    horizon_return_lambda: float = 0.95
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
    lr_model: float = 1e-4  # type: ignore , type OK
    #: critic model learning rate
    lr_critic: float = 3e-5  # type: ignore , type OK
    #: actor model learning rate
    lr_actor: float = 3e-5  # type: ignore , type OK
    #: loss計算の方法
    #:   "dreamer_v1" : Vの最大化
    #:   "dreamer_v2" : Vとエントロピーの最大化
    #:   "dreamer_v3" : V2 + パーセンタイルによる正規化
    actor_loss_type: str = "dreamer_v3"
    #: entropy rate
    entropy_rate: float = 0.0003
    #: baseline
    #:   "v"  : -v
    #:   other: none
    reinforce_baseline: str = "v"

    # --- other
    #: action ε-greedy(for debug)
    epsilon: float = 0
    #: 報酬の前処理
    #:   "none": なし
    #:   "tanh": tanh
    clip_rewards: str = "none"

    def __post_init__(self):
        super().__post_init__()

        self.lr_model: SchedulerConfig = SchedulerConfig(cast(float, self.lr_model))
        self.lr_critic: SchedulerConfig = SchedulerConfig(cast(float, self.lr_critic))
        self.lr_actor: SchedulerConfig = SchedulerConfig(cast(float, self.lr_actor))

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
            "horizon_return_lambda",
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
        self.reward_type = "normal_fixed_stddev"
        self.reward_layer_sizes = (400, 400)
        self.cont_layer_sizes = (400, 400)
        self.critic_layer_sizes = (400, 400, 400)
        self.actor_layer_sizes = (400, 400, 400, 400)
        self.dense_act = "elu"
        self.use_symlog = False
        # --- encoder/decoder
        self.encoder_decoder_mlp = (400, 400, 400, 400)
        self.cnn_depth = 32
        self.cnn_blocks = 0
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
        self.critic_type = "dreamer_v1"
        self.actor_discrete_unimix = 0
        # Behavior
        self.horizon = 15
        self.critic_estimation_method: str = "dreamer_v1"
        self.horizon_ewa_disclam = 0.1
        self.discount: float = 0.99
        # Training
        self.batch_size = 50
        self.batch_length = 50
        self.lr_model.set_constant(6e-4)
        self.lr_critic.set_constant(8e-5)
        self.lr_actor.set_constant(8e-5)
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
        self.cnn_depth = 48
        self.cnn_blocks = 0
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
        self.critic_type = "dreamer_v2"
        self.actor_discrete_unimix = 0
        # Behavior
        self.horizon = 15
        self.critic_estimation_method: str = "dreamer_v2"
        self.horizon_return_lambda = 0.95
        self.discount: float = 0.99
        # Training
        self.batch_size = 16
        self.batch_length = 50
        self.lr_model.set_constant(1e-4)
        self.lr_critic.set_constant(2e-4)
        self.lr_actor.set_constant(8e-5)
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
        self.cnn_depth = 96
        self.cnn_blocks = 0
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
        self.critic_type = "dreamer_v3"
        self.critic_twohot_bins = 255
        self.critic_twohot_low = -20
        self.critic_twohot_high = 20
        self.actor_discrete_unimix = 0.01
        # Behavior
        self.horizon = 15
        self.critic_estimation_method: str = "dreamer_v3"
        self.horizon_return_lambda = 0.95
        self.discount: float = 0.997
        # Training
        self.batch_size = 16
        self.batch_length = 64
        self.lr_model.set_constant(1e-4)
        self.lr_critic.set_constant(3e-5)
        self.lr_actor.set_constant(3e-5)
        self.actor_loss_type = "dreamer_v3"
        self.entropy_rate: float = 3e-4
        self.reinforce_baseline: str = "v"

    def set_processor(self) -> List[Processor]:
        return [
            ImageProcessor(
                image_type=EnvObservationTypes.COLOR,
                resize=(64, 64),
                enable_norm=True,
            )
        ]

    @property
    def base_action_type(self) -> RLBaseTypes:
        return RLBaseTypes.ANY

    @property
    def base_observation_type(self) -> RLBaseTypes:
        return RLBaseTypes.ANY

    def get_use_framework(self) -> str:
        return "tensorflow"

    def getName(self) -> str:
        return "DreamerV3"

    def assert_params(self) -> None:
        super().assert_params()
        self.assert_params_memory()


register(
    Config(),
    __name__ + ":Memory",
    __name__ + ":Parameter",
    __name__ + ":Trainer",
    __name__ + ":Worker",
)


# ------------------------------------------------------
# Memory
# ------------------------------------------------------
class Memory(ExperienceReplayBuffer):
    pass


# ------------------------------------------------------
# network
# ------------------------------------------------------
class _RSSM(keras.Model):
    def __init__(
        self,
        deter: int,
        stoch: int,
        classes: int,
        hidden_units: int,
        unimix: float,
        activation: Any,
        use_norm_layer: bool,
        use_categorical_distribution: bool,
    ):
        super().__init__()
        self.use_categorical_distribution = use_categorical_distribution
        self.stoch_size = stoch
        self.classes = classes
        self.unimix = unimix

        # --- img step
        self.img_in_layers = [kl.Dense(hidden_units)]
        if use_norm_layer:
            self.img_in_layers.append(kl.LayerNormalization())
        self.img_in_layers.append(kl.Activation(activation))
        self.rnn_cell = kl.GRUCell(deter)
        self.img_out_layers = [kl.Dense(hidden_units)]
        if use_norm_layer:
            self.img_out_layers.append(kl.LayerNormalization())
        self.img_out_layers.append(kl.Activation(activation))

        # --- obs step
        self.obs_layers = [kl.Dense(hidden_units)]
        if use_norm_layer:
            self.obs_layers.append(kl.LayerNormalization())
        self.obs_layers.append(kl.Activation(activation))

        # --- dist
        if self.use_categorical_distribution:
            self.img_cat_dist_layers = [
                kl.Dense(stoch * classes, kernel_initializer="zeros"),
                kl.Reshape((stoch, classes)),
            ]
            self.obs_cat_dist_layers = [
                kl.Dense(stoch * classes, kernel_initializer="zeros"),
                kl.Reshape((stoch, classes)),
            ]
        else:
            self.img_norm_dist_block = NormalDistBlock(stoch * classes, (), (), ())
            self.obs_norm_dist_block = NormalDistBlock(stoch * classes, (), (), ())

    def img_step(self, prev_stoch, prev_deter, prev_onehot_action, training: bool = False):
        # --- NN
        x = tf.concat([prev_stoch, prev_onehot_action], -1)
        for layer in self.img_in_layers:
            x = layer(x, training=training)
        x, deter = cast(Any, self.rnn_cell(x, [prev_deter], training=training))
        deter = deter[0]
        for layer in self.img_out_layers:
            x = layer(x, training=training)

        # --- dist
        if self.use_categorical_distribution:
            for h in self.img_cat_dist_layers:
                x = h(x)
            # (batch, stoch, classes) -> (batch * stoch, classes)
            batch = x.shape[0]
            x = tf.reshape(x, (batch * self.stoch_size, self.classes))
            dist = CategoricalGradDist(x, self.unimix)
            # (batch * stoch, classes) -> (batch, stoch, classes) -> (batch, stoch * classes)
            stoch = tf.cast(
                tf.reshape(dist.sample(), (batch, self.stoch_size, self.classes)),
                tf.float32,
            )
            stoch = tf.reshape(stoch, (batch, self.stoch_size * self.classes))
            # (batch * stoch, classes)
            probs = dist.probs()
            prior = {"stoch": stoch, "probs": probs}
        else:
            dist = self.img_norm_dist_block.call_grad_dist(x)
            prior = {
                "stoch": dist.sample(),
                "mean": dist.mean(),
                "stddev": dist.stddev(),
            }

        return deter, prior

    def obs_step(self, deter, embed, training=False):
        # --- NN
        x = tf.concat([deter, embed], -1)
        for layer in self.obs_layers:
            x = layer(x, training=training)

        # --- dist
        if self.use_categorical_distribution:
            for h in self.obs_cat_dist_layers:
                x = h(x)
            # (batch, stoch, classes) -> (batch * stoch, classes)
            batch = x.shape[0]
            x = tf.reshape(x, (batch * self.stoch_size, self.classes))
            dist = CategoricalGradDist(x, self.unimix)
            # (batch * stoch, classes) -> (batch, stoch, classes) -> (batch, stoch * classes)
            stoch = tf.cast(
                tf.reshape(dist.sample(), (batch, self.stoch_size, self.classes)),
                tf.float32,
            )
            stoch = tf.reshape(stoch, (batch, self.stoch_size * self.classes))
            # (batch * stoch, classes)
            probs = dist.probs()
            post = {"stoch": stoch, "probs": probs}
        else:
            dist = self.obs_norm_dist_block.call_grad_dist(x)
            post = {
                "stoch": dist.sample(),
                "mean": dist.mean(),
                "stddev": dist.stddev(),
            }

        return post

    def get_initial_state(self, batch_size: int = 1):
        stoch = tf.zeros((batch_size, self.stoch_size * self.classes), dtype=tf.float32)
        deter = self.rnn_cell.get_initial_state(None, batch_size, dtype=tf.float32)
        return stoch, deter

    @tf.function
    def compute_train_loss(self, embed, actions, stoch, deter, undone, batch_size, batch_length, free_nats):
        # (seq*batch, shape) -> (seq, batch, shape)
        embed = tf.reshape(embed, (batch_length, batch_size) + embed.shape[1:])
        undone = tf.reshape(undone, (batch_length, batch_size) + undone.shape[1:])

        # --- batch seq step
        stochs = []
        deters = []
        if self.use_categorical_distribution:
            post_probs = []
            prior_probs = []
            for i in range(batch_length):
                deter, prior = self.img_step(stoch, deter, actions[i], training=True)
                post = self.obs_step(deter, embed[i], training=True)
                stoch = post["stoch"]
                stochs.append(stoch)
                deters.append(deter)
                post_probs.append(post["probs"])
                prior_probs.append(prior["probs"])
                # 終了時は初期化
                stoch = stoch * undone[i]
                deter = deter * undone[i]
            post_probs = tf.stack(post_probs, axis=0)
            prior_probs = tf.stack(prior_probs, axis=0)

            post_dist = tfd.OneHotCategorical(probs=post_probs)
            prior_dist = tfd.OneHotCategorical(probs=prior_probs)

        else:
            post_mean = []
            post_std = []
            prior_mean = []
            prior_std = []
            for i in range(batch_length):
                deter, prior = self.img_step(stoch, deter, actions[i], training=True)
                post = self.obs_step(deter, embed[i], training=True)
                stoch = post["stoch"]
                stochs.append(stoch)
                deters.append(deter)
                post_mean.append(post["mean"])
                post_std.append(post["stddev"])
                prior_mean.append(prior["mean"])
                prior_std.append(prior["stddev"])
                # 終了時は初期化
                stoch = stoch * undone[i]
                deter = deter * undone[i]

            post_mean = tf.stack(post_mean, axis=0)
            post_std = tf.stack(post_std, axis=0)
            prior_mean = tf.stack(prior_mean, axis=0)
            prior_std = tf.stack(prior_std, axis=0)

            post_dist = tfd.Normal(post_mean, post_std)
            prior_dist = tfd.Normal(prior_mean, prior_std)

        stochs = tf.stack(stochs, axis=0)
        deters = tf.stack(deters, axis=0)

        # (seq, batch, shape) -> (seq*batch, shape)
        stochs = tf.reshape(stochs, (batch_length * batch_size,) + stochs.shape[2:])
        deters = tf.reshape(deters, (batch_length * batch_size,) + deters.shape[2:])
        feats = tf.concat([stochs, deters], -1)

        # --- KL loss
        kl_loss_dyn = tfd.kl_divergence(tf.stop_gradient(post_dist), prior_dist)
        kl_loss_rep = tfd.kl_divergence(post_dist, tf.stop_gradient(prior_dist))
        kl_loss_dyn = tf.reduce_mean(tf.maximum(kl_loss_dyn, free_nats))
        kl_loss_rep = tf.reduce_mean(tf.maximum(kl_loss_rep, free_nats))

        return stochs, deters, feats, kl_loss_dyn, kl_loss_rep, stoch, deter

    def build(self, config: Config, embed_size: int):
        self._embed_size = embed_size
        in_stoch, in_deter = self.get_initial_state()
        in_onehot_action = np.zeros((1, config.action_num), dtype=np.float32)
        in_embed = np.zeros((1, embed_size), dtype=np.float32)
        deter, prior = self.img_step(in_stoch, in_deter, in_onehot_action)
        post = self.obs_step(deter, in_embed)
        self.built = True
        return tf.concat([post["stoch"], deter], axis=1)

    def summary(self, config: Config, **kwargs):
        _stoch, _deter = self.get_initial_state()
        in_deter = kl.Input((_deter.shape[1],), batch_size=1)
        in_stoch = kl.Input((_stoch.shape[1],), batch_size=1)
        in_onehot_action = kl.Input((config.action_num,), batch_size=1)
        in_embed = kl.Input((self._embed_size,), batch_size=1)

        deter, prior = self.img_step(in_stoch, in_deter, in_onehot_action)
        post = self.obs_step(deter, in_embed)
        model = keras.Model(
            inputs=[in_stoch, in_deter, in_onehot_action, in_embed],
            outputs=post,
            name="RSSM",
        )
        return model.summary(**kwargs)


class _ImageEncoder(keras.Model):
    def __init__(
        self,
        img_shape: tuple,
        depth: int,
        res_blocks: int,
        activation,
        normalization_type: str,
        resize_type: str,
        resized_image_size: int,
    ):
        super().__init__()
        assert normalization_type in ["none", "layer"]
        self._in_shape = img_shape
        self.img_shape = img_shape

        _size = int(np.log2(min(img_shape[-3], img_shape[-2])))
        _resize = int(np.log2(resized_image_size))
        assert _size > _resize
        self.stages = _size - _resize

        if resize_type == "stride":
            assert img_shape[-2] % (2**self.stages) == 0
            assert img_shape[-3] % (2**self.stages) == 0
        elif resize_type == "stride3":
            assert (img_shape[-2] % ((2 ** (self.stages - 1)) * 3)) == 0
            assert (img_shape[-3] % ((2 ** (self.stages - 1)) * 3)) == 0
        elif resize_type == "max":
            assert img_shape[-2] % (2**self.stages) == 0
            assert img_shape[-3] % (2**self.stages) == 0
        else:
            raise NotImplementedError(resize_type)

        _conv_kw: dict = dict(
            padding="same",
            kernel_initializer=tf.initializers.TruncatedNormal(),
            bias_initializer="zero",
        )

        self.blocks = []
        for i in range(self.stages):
            # --- cnn
            use_bias = normalization_type == "none"
            if resize_type == "stride":
                cnn_layers = [kl.Conv2D(depth, 4, 2, use_bias=use_bias, **_conv_kw)]
            elif resize_type == "stride3":
                s = 2 if i else 3
                k = 5 if i else 4
                cnn_layers = [kl.Conv2D(depth, k, s, use_bias=use_bias, **_conv_kw)]
            elif resize_type == "mean":
                cnn_layers = [
                    kl.Conv2D(depth, 3, 1, use_bias=use_bias, **_conv_kw),
                    kl.AveragePooling2D((3, 3), (2, 2), padding="same"),
                ]
            elif resize_type == "max":
                cnn_layers = [
                    kl.Conv2D(depth, 3, 1, use_bias=use_bias, **_conv_kw),
                    kl.MaxPooling2D((3, 3), (2, 2), padding="same"),
                ]
            else:
                raise NotImplementedError(resize_type)
            if normalization_type == "layer":
                cnn_layers.append(kl.LayerNormalization())
            cnn_layers.append(kl.Activation(activation))

            # --- res
            res_blocks_layers = []
            for _ in range(res_blocks):
                res_layers = []
                if normalization_type == "layer":
                    res_layers.append(kl.LayerNormalization())
                res_layers.append(kl.Activation(activation))
                res_layers.append(kl.Conv2D(depth, 3, 1, use_bias=True, **_conv_kw))
                if normalization_type == "layer":
                    res_layers.append(kl.LayerNormalization())
                res_layers.append(kl.Activation(activation))
                res_layers.append(kl.Conv2D(depth, 3, 1, use_bias=True, **_conv_kw))
                res_blocks_layers.append(res_layers)

            self.blocks.append([cnn_layers, res_blocks_layers])
            depth *= 2

        self.out_layers = []
        if res_blocks > 0:
            self.out_layers.append(kl.Activation(activation))
        self.out_layers.append(kl.Flatten())

        dummy, img_shape = self._call(np.zeros((1,) + img_shape), return_size=True)
        self.resized_img_shape = img_shape[1:]
        self.out_size = dummy.shape[1]

    @tf.function
    def call(self, x, training=False):
        return self._call(x, training=training)

    def _call(self, x, training=False, return_size=False):
        x = x - 0.5
        for block in self.blocks:
            # --- cnn
            for h in block[0]:
                x = h(x, training=training)
            # --- res
            for res_blocks in block[1]:
                skip = x
                for h in res_blocks:
                    x = h(x, training=training)
                x += skip

        x_out = x
        for h in self.out_layers:
            x_out = h(x_out, training=training)

        if return_size:
            return x_out, x.shape
        else:
            return x_out

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self._in_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self._call(x), name=name)
        return model.summary(**kwargs)


class _ImageDecoder(keras.Model):
    def __init__(
        self,
        encoder: _ImageEncoder,
        use_sigmoid: bool,
        depth: int,
        res_blocks: int,
        activation,
        normalization_type: str,
        resize_type: str,
    ):
        super().__init__()
        self.use_sigmoid = use_sigmoid

        stages = encoder.stages
        depth = depth * 2 ** (encoder.stages - 1)
        img_shape = encoder.img_shape
        resized_img_shape = encoder.resized_img_shape

        _conv_kw: dict = dict(
            kernel_initializer=tf.initializers.TruncatedNormal(),
            bias_initializer="zero",
        )
        self.in_layer = kl.Dense(resized_img_shape[0] * resized_img_shape[1] * resized_img_shape[2])
        self.reshape_layer = kl.Reshape([resized_img_shape[0], resized_img_shape[1], resized_img_shape[2]])
        self.blocks = []
        for i in range(encoder.stages):
            # --- res
            res_blocks_layers = []
            for _ in range(res_blocks):
                res_layers = []
                if normalization_type == "layer":
                    res_layers.append(kl.LayerNormalization())
                res_layers.append(kl.Activation(activation))
                res_layers.append(kl.Conv2D(depth, 3, 1, padding="same", **_conv_kw))
                if normalization_type == "layer":
                    res_layers.append(kl.LayerNormalization())
                res_layers.append(kl.Activation(activation))
                res_layers.append(kl.Conv2D(depth, 3, 1, padding="same", **_conv_kw))
                res_blocks_layers.append(res_layers)

            if i == stages - 1:
                depth = img_shape[-1]
            else:
                depth //= 2

            # --- cnn
            use_bias = normalization_type == "none"
            if resize_type == "stride":
                cnn_layers = [kl.Conv2DTranspose(depth, 4, 2, use_bias=use_bias, padding="same", **_conv_kw)]
            elif resize_type == "stride3":
                s = 3 if i == stages - 1 else 2
                k = 5 if i == stages - 1 else 4
                cnn_layers = [kl.Conv2DTranspose(depth, k, s, use_bias=use_bias, padding="same", **_conv_kw)]
            elif resize_type == "max":
                cnn_layers = [
                    kl.UpSampling2D((2, 2)),
                    kl.Conv2D(depth, 3, 1, use_bias=use_bias, padding="same", **_conv_kw),
                ]
            else:
                raise NotImplementedError(resize_type)
            if normalization_type == "layer":
                cnn_layers.append(kl.LayerNormalization())
            cnn_layers.append(kl.Activation(activation))

            self.blocks.append([res_blocks_layers, cnn_layers])

    def call(self, x):
        x = self.in_layer(x)
        x = self.reshape_layer(x)

        for block in self.blocks:
            # --- res
            for res_blocks in block[0]:
                skip = x
                for h in res_blocks:
                    x = h(x)
                x += cast(Any, skip)
            # --- cnn
            for h in block[1]:
                x = h(x)

        if self.use_sigmoid:
            x = tf.nn.sigmoid(x)
        else:
            x = cast(Any, x) + 0.5

        return x

    @tf.function
    def compute_train_loss(self, feat, states):
        y_pred = self(feat)
        # MSE
        return tf.reduce_mean(tf.square(states - y_pred))

    def sample(self, x) -> Any:
        return self(x)

    def build(self, input_shape):
        self.__input_shape = input_shape
        super().build(self.__input_shape)

    def summary(self, name="", **kwargs):
        x = kl.Input(shape=self.__input_shape)
        name = self.__class__.__name__ if name == "" else name
        model = keras.Model(inputs=x, outputs=self.call(x), name=name)
        return model.summary(**kwargs)


class _LinearEncoder(keras.Model):
    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...],
        activation: str,
    ):
        super().__init__()

        self.hidden_layers = []
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))

        self.out_size: int = hidden_layer_sizes[-1]

    @tf.function
    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x


class _LinearDecoder(keras.Model):
    def __init__(
        self,
        out_size: int,
        hidden_layer_sizes: Tuple[int, ...],
        activation: str,
        use_symlog: bool,
    ):
        super().__init__()
        self.use_symlog = use_symlog

        self.hidden_layers = []
        for i in reversed(range(len(hidden_layer_sizes))):
            self.hidden_layers.append(kl.Dense(hidden_layer_sizes[i], activation=activation))
        self.out_layer = kl.Dense(out_size)

    def call(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        x = self.out_layer(x)
        return x

    @tf.function
    def compute_train_loss(self, feat, y):
        y_pred = self(feat)
        if self.use_symlog:
            y = symlog(y)
        # MSE
        return tf.reduce_mean(tf.square(y - y_pred))

    def sample(self, x) -> Any:
        y = self(x)
        if self.use_symlog:
            y = symexp(y)
        return y


# ------------------------------------------------------
# Parameter
# ------------------------------------------------------
class Parameter(RLParameter):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config

        # --- encode/decode
        if self.config.observation_type == RLTypes.IMAGE:
            self.encode = _ImageEncoder(
                self.config.observation_shape,
                self.config.cnn_depth,
                self.config.cnn_blocks,
                self.config.cnn_activation,
                self.config.cnn_normalization_type,
                self.config.cnn_resize_type,
                self.config.cnn_resized_image_size,
            )
            self.decode = _ImageDecoder(
                self.encode,
                self.config.cnn_use_sigmoid,
                self.config.cnn_depth,
                self.config.cnn_blocks,
                self.config.cnn_activation,
                self.config.cnn_normalization_type,
                self.config.cnn_resize_type,
            )
            logger.info("Encoder/Decoder: Image")
        else:
            self.encode = _LinearEncoder(
                self.config.encoder_decoder_mlp,
                self.config.dense_act,
            )
            self.decode = _LinearDecoder(
                self.config.observation_shape[-1],
                self.config.encoder_decoder_mlp,
                self.config.dense_act,
                use_symlog=self.config.use_symlog,
            )
            logger.info("Encoder/Decoder: Linear" + ("(symlog)" if self.config.use_symlog else ""))

        # --- dynamics
        self.dynamics = _RSSM(
            self.config.rssm_deter_size,
            self.config.rssm_stoch_size,
            self.config.rssm_classes,
            self.config.rssm_hidden_units,
            self.config.rssm_unimix,
            self.config.rssm_activation,
            self.config.rssm_use_norm_layer,
            self.config.rssm_use_categorical_distribution,
        )

        # --- reward
        if self.config.reward_type == "linear":
            self.reward = LinearBlock(
                1,
                self.config.reward_layer_sizes,
                self.config.dense_act,
                use_symlog=self.config.use_symlog,
            )
            logger.info("Reward  : Linear" + ("(symlog)" if self.config.use_symlog else ""))
        elif self.config.reward_type == "twohot":
            self.reward = TwoHotDistBlock(
                self.config.reward_twohot_bins,
                self.config.reward_twohot_low,
                self.config.reward_twohot_high,
                self.config.reward_layer_sizes,
                self.config.dense_act,
                use_symlog=self.config.use_symlog,
            )
            logger.info("Reward  : TwoHot" + ("(symlog)" if self.config.use_symlog else ""))
        elif self.config.reward_type == "normal":
            self.reward = NormalDistBlock(1, self.config.reward_layer_sizes, (), (), self.config.dense_act)
            logger.info("Reward  : Normal")
        elif self.config.reward_type == "normal_fixed_stddev":
            self.reward = NormalDistBlock(1, self.config.reward_layer_sizes, (), (), self.config.dense_act, 1)
            logger.info("Reward  : Normal fixed stddev")
        else:
            raise UndefinedError(self.config.reward_type)

        # --- continue
        self.cont = BernoulliDistBlock(1, self.config.cont_layer_sizes, self.config.dense_act)
        logger.info("Continue: Bernoulli")

        # --- critic
        if self.config.critic_type == "linear":
            self.critic = LinearBlock(
                1,
                self.config.critic_layer_sizes,
                self.config.dense_act,
                self.config.use_symlog,
            )
            self.critic_target = LinearBlock(
                1,
                self.config.critic_layer_sizes,
                self.config.dense_act,
                self.config.use_symlog,
            )
            logger.info("Critic  : Linear" + ("(symlog)" if self.config.use_symlog else ""))
        elif self.config.critic_type == "normal":
            self.critic = NormalDistBlock(1, self.config.critic_layer_sizes, (), (), self.config.dense_act)
            self.critic_target = NormalDistBlock(1, self.config.critic_layer_sizes, (), (), self.config.dense_act)
            logger.info("Critic  : Normal")
        elif self.config.critic_type in ["dreamer_v1", "dreamer_v2"]:
            self.critic = NormalDistBlock(
                1,
                self.config.critic_layer_sizes,
                (),
                (),
                self.config.dense_act,
                fixed_stddev=1,
            )
            self.critic_target = NormalDistBlock(
                1,
                self.config.critic_layer_sizes,
                (),
                (),
                self.config.dense_act,
                fixed_stddev=1,
            )
            logger.info("Critic  : Normal(stddev=1)")
        elif self.config.critic_type == "dreamer_v3":
            self.critic = TwoHotDistBlock(
                self.config.critic_twohot_bins,
                self.config.critic_twohot_low,
                self.config.critic_twohot_high,
                self.config.critic_layer_sizes,
                self.config.dense_act,
                use_symlog=self.config.use_symlog,
            )
            self.critic_target = TwoHotDistBlock(
                self.config.critic_twohot_bins,
                self.config.critic_twohot_low,
                self.config.critic_twohot_high,
                self.config.critic_layer_sizes,
                self.config.dense_act,
                use_symlog=self.config.use_symlog,
            )
            logger.info("Critic  : TwoHot" + ("(symlog)" if self.config.use_symlog else ""))
        else:
            raise UndefinedError(self.config.critic_type)

        # --- actor
        if self.config.action_type == RLTypes.DISCRETE:
            if self.config.actor_discrete_type == "categorical":
                self.actor = CategoricalDistBlock(
                    self.config.action_num,
                    self.config.actor_layer_sizes,
                    unimix=self.config.actor_discrete_unimix,
                )
                logger.info(f"Actor   : Categorical(unimix={self.config.actor_discrete_unimix})")
            elif self.config.actor_discrete_type == "gumbel_categorical":
                self.actor = CategoricalGumbelDistBlock(
                    self.config.action_num,
                    self.config.actor_layer_sizes,
                )
                logger.info("Actor   : GumbelCategorical")
            else:
                raise UndefinedError(self.config.actor_discrete_type)
        elif self.config.action_type == RLTypes.CONTINUOUS:
            self.actor = NormalDistBlock(self.config.action_num, self.config.actor_layer_sizes, (), ())
            logger.info("Actor   : Normal")
        else:
            raise ValueError(self.config.action_type)

        # --- build
        self.encode.build((None,) + self.config.observation_shape)
        embed = self.dynamics.build(self.config, self.encode.out_size)
        self.decode.build((None, embed.shape[1]))
        self.reward.build((None, embed.shape[1]))
        self.cont.build((None, embed.shape[1]))
        self.critic.build((None, embed.shape[1]))
        self.critic_target.build((None, embed.shape[1]))
        self.actor.build((None, embed.shape[1]))

        # --- sync target
        self.critic_target.set_weights(self.critic.get_weights())

    def call_restore(self, data: Any, **kwargs) -> None:
        self.encode.set_weights(data[0])
        self.dynamics.set_weights(data[1])
        self.decode.set_weights(data[2])
        self.reward.set_weights(data[3])
        self.cont.set_weights(data[4])
        self.critic.set_weights(data[5])
        self.critic_target.set_weights(data[5])
        self.actor.set_weights(data[6])

    def call_backup(self, **kwargs) -> Any:
        return [
            self.encode.get_weights(),
            self.dynamics.get_weights(),
            self.decode.get_weights(),
            self.reward.get_weights(),
            self.cont.get_weights(),
            self.critic.get_weights(),
            self.actor.get_weights(),
        ]

    def summary(self, **kwargs):
        self.encode.summary(**kwargs)
        self.dynamics.summary(self.config, **kwargs)
        self.decode.summary(**kwargs)
        self.reward.summary(**kwargs)
        self.cont.summary(**kwargs)
        self.critic.summary(**kwargs)
        self.actor.summary(**kwargs)


# ------------------------------------------------------
# Trainer
# ------------------------------------------------------
class Trainer(RLTrainer):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.lr_sch_model = self.config.lr_model.create_schedulers()
        self.lr_sch_critic = self.config.lr_critic.create_schedulers()
        self.lr_sch_actor = self.config.lr_actor.create_schedulers()

        if compare_less_version(tf.__version__, "2.11.0"):
            self._model_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_model.get_rate())
            self._critic_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_critic.get_rate())
            self._actor_opt = keras.optimizers.Adam(learning_rate=self.lr_sch_actor.get_rate())
        else:
            self._model_opt = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_model.get_rate())
            self._critic_opt = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_critic.get_rate())
            self._actor_opt = keras.optimizers.legacy.Adam(learning_rate=self.lr_sch_actor.get_rate())

        self.seq_action = [[] for _ in range(self.config.batch_size)]
        self.seq_next_state = [[] for _ in range(self.config.batch_size)]
        self.seq_reward = [[] for _ in range(self.config.batch_size)]
        self.seq_undone = [[] for _ in range(self.config.batch_size)]
        self.seq_unterminated = [[] for _ in range(self.config.batch_size)]
        self.stoch, self.deter = self.parameter.dynamics.get_initial_state(self.batch_size)

    def train(self) -> None:
        if self.memory.is_warmup_needed():
            return
        self.train_info = {}

        # --- create sequence batch
        # 各batchにbatch_seq溜まるまでエピソードを追加する
        actions = []
        next_states = []
        rewards = []
        undone = []
        unterminated = []

        def _f(arr1, arr2, i):
            arr1.append(arr2[i][: self.config.batch_length])
            arr2[i] = arr2[i][self.config.batch_length :]

        for i in range(self.config.batch_size):
            while len(self.seq_action[i]) < self.config.batch_length:
                batch = self.memory.sample(1, self.train_count)[0]
                episode_len = len(batch["actions"])
                self.seq_action[i].extend(batch["actions"])
                self.seq_next_state[i].extend(batch["next_states"])
                self.seq_reward[i].extend(batch["rewards"])
                self.seq_undone[i].extend([1 for _ in range(episode_len - 1)] + [0])
                self.seq_unterminated[i].extend(
                    [1 for _ in range(episode_len - 1)] + [0 if batch["terminated"] else 1]
                )
            _f(actions, self.seq_action, i)
            _f(next_states, self.seq_next_state, i)
            _f(rewards, self.seq_reward, i)
            _f(undone, self.seq_undone, i)
            _f(unterminated, self.seq_unterminated, i)
        actions = np.asarray(actions, dtype=np.float32)
        next_states = np.asarray(next_states, dtype=np.float32)
        rewards = np.asarray(rewards, dtype=np.float32)[..., np.newaxis]
        undone = np.asarray(undone, dtype=np.float32)[..., np.newaxis]
        unterminated = np.asarray(unterminated, dtype=np.float32)[..., np.newaxis]

        # (batch, seq, shape) -> (seq, batch, shape)
        actions = tf.transpose(actions, [1, 0, 2])
        # (batch, seq, shape) -> (seq, batch, shape) -> (seq*batch, shape)
        _t = list(range(len(next_states.shape)))
        _t[0], _t[1] = _t[1], _t[0]
        next_states = tf.reshape(
            tf.transpose(next_states, _t),
            (self.config.batch_size * self.config.batch_length,) + next_states.shape[2:],
        )
        rewards = tf.reshape(
            tf.transpose(rewards, [1, 0, 2]),
            (self.config.batch_size * self.config.batch_length,) + rewards.shape[2:],
        )
        undone = tf.reshape(
            tf.transpose(undone, [1, 0, 2]),
            (self.config.batch_size * self.config.batch_length,) + undone.shape[2:],
        )
        unterminated = tf.reshape(
            tf.transpose(unterminated, [1, 0, 2]),
            (self.config.batch_size * self.config.batch_length,) + unterminated.shape[2:],
        )

        # ------------------------
        # RSSM
        # ------------------------
        self.parameter.encode.trainable = True
        self.parameter.decode.trainable = True
        self.parameter.dynamics.trainable = True
        self.parameter.reward.trainable = True
        self.parameter.actor.trainable = False
        self.parameter.critic.trainable = False
        with tf.GradientTape() as tape:
            embed = self.parameter.encode(next_states, training=True)
            (
                stochs,
                deters,
                feats,
                kl_loss_dyn,
                kl_loss_rep,
                self.stoch,
                self.deter,
            ) = self.parameter.dynamics.compute_train_loss(
                embed,
                actions,
                self.stoch,
                self.deter,
                undone,
                self.config.batch_size,
                self.config.batch_length,
                self.config.free_nats,
            )

            # --- embed loss
            decode_loss = self.parameter.decode.compute_train_loss(feats, next_states)
            reward_loss = self.parameter.reward.compute_train_loss(feats, rewards)
            cont_loss = self.parameter.cont.compute_train_loss(feats, unterminated)

            loss = (
                self.config.loss_scale_pred * (decode_loss + reward_loss + cont_loss)
                + self.config.loss_scale_kl_dyn * kl_loss_dyn
                + self.config.loss_scale_kl_rep * kl_loss_rep
            )

        if self.config.enable_train_model:
            variables = [
                self.parameter.encode.trainable_variables,
                self.parameter.dynamics.trainable_variables,
                self.parameter.decode.trainable_variables,
                self.parameter.reward.trainable_variables,
            ]
            grads = tape.gradient(loss, variables)
            for i in range(len(variables)):
                self._model_opt.apply_gradients(zip(grads[i], variables[i]))

            if self.lr_sch_model.update(self.train_count):
                self._model_opt.learning_rate = self.lr_sch_model.get_rate()

            self.train_info["decode_loss"] = np.mean(decode_loss.numpy())
            self.train_info["reward_loss"] = np.mean(reward_loss.numpy())
            self.train_info["cont_loss"] = np.mean(cont_loss.numpy())
            self.train_info["kl_loss"] = np.mean(kl_loss_dyn.numpy())

        if (not self.config.enable_train_actor) and (not self.config.enable_train_critic):
            # WorldModelsのみ学習
            self.train_count += 1
            return

        self.parameter.encode.trainable = False
        self.parameter.decode.trainable = False
        self.parameter.dynamics.trainable = False
        self.parameter.reward.trainable = False

        # ------------------------
        # Actor
        # ------------------------
        V = None
        if self.config.enable_train_actor:
            self.parameter.actor.trainable = True
            self.parameter.critic.trainable = False
            with tf.GradientTape() as tape:
                actor_loss, act_v_loss, entropy_loss, V = self._compute_horizon_step(stochs, deters, feats)
            grads = tape.gradient(actor_loss, self.parameter.actor.trainable_variables)
            self._actor_opt.apply_gradients(zip(grads, self.parameter.actor.trainable_variables))
            if act_v_loss is not None:
                self.train_info["act_v_loss"] = -np.mean(act_v_loss)
            if entropy_loss is not None:
                self.train_info["entropy_loss"] = -np.mean(entropy_loss)
            self.train_info["actor_loss"] = actor_loss.numpy() / self.config.horizon

            if self.lr_sch_actor.update(self.train_count):
                self._actor_opt.learning_rate = self.lr_sch_actor.get_rate()

        # ------------------------
        # critic
        # ------------------------
        if self.config.enable_train_critic:
            if V is None:
                actor_loss, act_v_loss, entropy_loss, V = self._compute_horizon_step(stochs, deters, feats)

            self.parameter.actor.trainable = False
            self.parameter.critic.trainable = True
            with tf.GradientTape() as tape:
                critic_loss = self.parameter.critic.compute_train_loss(feats, tf.stop_gradient(V))
            grads = tape.gradient(critic_loss, self.parameter.critic.trainable_variables)
            self._critic_opt.apply_gradients(zip(grads, self.parameter.critic.trainable_variables))
            self.train_info["critic_loss"] = critic_loss.numpy()

            if self.lr_sch_critic.update(self.train_count):
                self._critic_opt.learning_rate = self.lr_sch_critic.get_rate()

            # --- target update
            if self.config.critic_target_update_interval > 0:
                if self.train_count % self.config.critic_target_update_interval == 0:
                    self.parameter.critic_target.set_weights(self.parameter.critic.get_weights())
                else:
                    self.parameter.critic_target.set_weights(
                        (1 - self.config.critic_target_soft_update)
                        * np.array(self.parameter.critic.get_weights(), dtype=object)
                        + (self.config.critic_target_soft_update)
                        * np.array(self.parameter.critic.get_weights(), dtype=object)
                    )

        self.train_count += 1
        return

    @tf.function
    def _compute_horizon_step(self, stoch, deter, feat):
        horizon_feat = [feat]
        log_pi = None
        entropy = None
        for t in range(self.config.horizon):
            # --- calc action
            if self.config.action_type == RLTypes.DISCRETE:
                dist = self.parameter.actor.call_grad_dist(feat)
                action = dist.sample()
                if self.config.horizon_policy == "random":
                    action = tf.one_hot(
                        np.random.randint(0, self.config.action_num - 1, size=stoch.shape[0]),
                        self.config.action_num,
                    )
                # use entropy
                if t == 0:
                    log_probs = dist.log_probs()
                    entropy = -tf.reduce_sum(tf.exp(log_probs) * log_probs, axis=-1)
            elif self.config.action_type == RLTypes.CONTINUOUS:
                dist = self.parameter.actor.call_grad_dist(feat)
                action = dist.sample()
                if self.config.horizon_policy == "random":
                    action = tf.random.normal(action.shape)
                # use H and logpi
                if t == 0:
                    log_pi = dist.log_prob(action)
                    entropy = -log_pi
            else:
                raise UndefinedError(self.config.action_type)

            # --- rssm step
            deter, prior = self.parameter.dynamics.img_step(stoch, deter, action)
            stoch = prior["stoch"]
            feat = tf.concat([stoch, deter], -1)
            horizon_feat.append(feat)
        horizon_feat = tf.stack(horizon_feat)

        horizon_reward = self.parameter.reward.call_grad_dist(horizon_feat).mode()
        if self.config.critic_target_update_interval > 0:
            horizon_v = self.parameter.critic_target.call_grad_dist(horizon_feat).mode()
        else:
            horizon_v = self.parameter.critic.call_grad_dist(horizon_feat).mode()
        horizon_cont = tf.cast(self.parameter.cont.call_grad_dist(horizon_feat).mode(), dtype=tf.float32)

        # --- compute V
        # (horizon, batch_size*batch_length, shape) -> (batch_size*batch_length, shape)
        V = _compute_V(
            self.config.critic_estimation_method,
            horizon_reward,
            horizon_v,
            horizon_cont,
            self.config.discount,
            self.config.horizon_ewa_disclam,
            self.config.horizon_return_lambda,
        )

        # --- compute actor
        act_v_loss = None
        entropy_loss = None
        if self.config.actor_loss_type == "dreamer_v1":
            # Vの最大化
            actor_loss = -tf.reduce_mean(V / self.config.horizon)
        elif self.config.actor_loss_type in ["dreamer_v2", "dreamer_v3"]:
            if self.config.actor_loss_type == "dreamer_v3":
                # パーセンタイルの計算
                d5 = tfp.stats.percentile(V, 5)
                d95 = tfp.stats.percentile(V, 95)
                adv = V / tf.maximum(1.0, d95 - d5)
            else:
                adv = V
            adv /= self.config.horizon

            if self.config.action_type == RLTypes.DISCRETE:
                # dynamics backprop
                act_v_loss = tf.reduce_mean(adv)
            elif self.config.action_type == RLTypes.CONTINUOUS:
                # reinforce
                if self.config.reinforce_baseline == "v":
                    adv = adv - horizon_v
                act_v_loss = tf.reduce_mean(log_pi * tf.stop_gradient(adv))
            else:
                raise UndefinedError(self.config.action_type)

            # entropy
            entropy_loss = tf.reduce_mean(entropy)

            # Vの最大化 + entropyの最大化
            actor_loss = -(act_v_loss + self.config.entropy_rate * entropy_loss)
        else:
            raise UndefinedError(self.config.actor_loss_type)

        return actor_loss, act_v_loss, entropy_loss, V


def _compute_V(
    critic_estimation_method: str,
    horizon_reward,
    horizon_v,
    horizon_cont,
    discount: float,
    horizon_ewa_disclam: float,
    horizon_return_lambda: float,
) -> Any:
    horizon = horizon_reward.shape[0]
    batch = horizon_reward.shape[1]

    cont1 = []
    cont2 = []
    done = []
    cont3 = []
    rewards = []
    _is_cont = tf.ones((batch, 1), dtype=tf.float32)
    for t in range(horizon):
        rewards.append(horizon_reward[t] * _is_cont)
        cont1.append(_is_cont)
        done.append((1 - horizon_cont[t]) * _is_cont)
        if t == horizon - 1:
            cont3.append(_is_cont)
        else:
            cont3.append(tf.zeros((batch, 1), dtype=tf.float32))

        # --- 一度終了したらその後はなし
        _is_cont = _is_cont * horizon_cont[t]
        cont2.append(_is_cont)

    if critic_estimation_method == "simple":
        V = tf.reduce_sum(rewards, axis=0)
    elif critic_estimation_method == "discount":
        disc = tf.constant([discount**t for t in range(horizon)], dtype=tf.float32)
        disc = tf.expand_dims(tf.tile(tf.expand_dims(disc, axis=1), [1, batch]), axis=-1)
        rewards = tf.stack(rewards, axis=0)
        V = tf.reduce_sum(rewards * disc, axis=0)
    elif critic_estimation_method in ["dreamer_v1", "ewa"]:
        VN = []
        v = tf.zeros((batch, 1), dtype=tf.float32)
        for t in reversed(range(horizon)):
            v = (rewards[t] * (1.0 - cont3[t]) + horizon_v[t] * cont3[t]) + cont2[t] * discount * v
            VN.insert(0, v)

        # EWA
        V = VN[0]
        for t in range(1, horizon):
            V = (1 - horizon_ewa_disclam) * V + horizon_ewa_disclam * VN[t]

    elif critic_estimation_method in ["dreamer_v2", "dreamer_v3", "h-return"]:
        V = tf.zeros((batch, 1), dtype=tf.float32)
        for t in reversed(range(horizon)):
            a = (1 - horizon_return_lambda) * horizon_v[t] + horizon_return_lambda * V
            b = horizon_v[t]
            V = rewards[t] + cont2[t] * discount * (a * (1.0 - cont3[t]) + b * cont3[t])
    else:
        raise UndefinedError(critic_estimation_method)

    return V


# ------------------------------------------------------
# Worker
# ------------------------------------------------------
class Worker(RLWorker):
    def __init__(self, *args):
        super().__init__(*args)
        self.config: Config = self.config
        self.parameter: Parameter = self.parameter

        self.screen = None

    def on_reset(self, worker: WorkerRun) -> dict:
        self.stoch, self.deter = self.parameter.dynamics.get_initial_state()
        if self.config.action_type == RLTypes.DISCRETE:
            self.action = tf.one_hot([self.sample_action()], self.config.action_num, dtype=tf.float32)
        elif self.config.action_type == RLTypes.CONTINUOUS:
            self.action = tf.constant([self.sample_action() for _ in range(self.config.action_num)], dtype=tf.float32)
        else:
            raise UndefinedError(self.config.action_type)

        # 初期状態へのstep
        state = worker.state.astype(np.float32)
        self._rssm_step(state, self.action)
        self._recent_actions = [self.action.numpy()[0]]
        self._recent_next_states = [state]
        self._recent_rewards = [0]

        return {}

    def policy(self, worker: WorkerRun) -> Tuple[Any, dict]:
        self.feat = None

        self._rssm_step(worker.state.astype(np.float32), self.action)

        # debug
        if random.random() < self.config.epsilon:
            env_action = self.sample_action()
            if self.config.action_type == RLTypes.DISCRETE:
                self.action = tf.one_hot([env_action], self.config.action_num, dtype=tf.float32)
            elif self.config.action_type == RLTypes.CONTINUOUS:
                self.action = env_action
            else:
                raise UndefinedError(self.config.action_type)
            return env_action, {}

        dist = self.parameter.actor.call_dist(self.feat)
        if self.config.action_type == RLTypes.DISCRETE:  # int
            self.action = dist.sample(onehot=True)
            env_action = int(np.argmax(self.action[0]))
        elif self.config.action_type == RLTypes.CONTINUOUS:  # float,list[float]
            self.action = dist.sample()
            env_action = self.action[0].numpy()
            env_action = env_action * (self.config.action_high - self.config.action_low) + self.config.action_low
            env_action = np.clip(env_action, self.config.action_low, self.config.action_high).tolist()
        else:
            raise UndefinedError(self.config.action_type)

        return env_action, {}

    def _rssm_step(self, state, action):
        embed = self.parameter.encode(state[np.newaxis, ...])
        deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
        post = self.parameter.dynamics.obs_step(deter, embed)
        self.feat = tf.concat([post["stoch"], deter], axis=1)
        self.deter = deter
        self.stoch = post["stoch"]

    def on_step(self, worker: WorkerRun) -> dict:
        if not self.training:
            return {}

        clip_rewards_fn = dict(none=lambda x: x, tanh=tf.tanh)[self.config.clip_rewards]
        reward = clip_rewards_fn(worker.reward)

        # 1episodeのbatch
        self._recent_actions.append(self.action.numpy()[0])
        self._recent_next_states.append(worker.state.astype(np.float32))
        self._recent_rewards.append(reward)

        if worker.done:
            self.memory.add(
                {
                    "actions": self._recent_actions,
                    "next_states": self._recent_next_states,
                    "rewards": self._recent_rewards,
                    "terminated": worker.done_reason == DoneTypes.TERMINATED,
                }
            )

        return {}

    def render_terminal(self, worker, **kwargs) -> None:
        assert self.feat is not None
        if worker.done:
            self.policy(worker)

        # --- decode
        pred_state = self.parameter.decode.sample(self.feat)[0].numpy()
        pred_reward = self.parameter.reward.call_dist(self.feat).mode()[0][0].numpy()
        pred_cont = self.parameter.cont.call_dist(self.feat).prob()[0][0].numpy()
        value = self.parameter.critic.call_dist(self.feat).mode()[0][0].numpy()

        print(pred_state)
        print(f"reward: {pred_reward:.5f}, done: {(1-pred_cont)*100:4.1f}%, v: {value:.5f}")

        if self.config.action_type == RLTypes.DISCRETE:
            act_dist = self.parameter.actor.call_dist(self.feat)
            act_probs = act_dist.probs().numpy()[0]
            maxa = np.argmax(act_probs)

            def _render_sub(a: int) -> str:
                # rssm step
                action = tf.one_hot([a], self.config.action_num, axis=1)
                deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
                feat = tf.concat([prior["stoch"], deter], axis=1)

                # サンプルを表示
                next_state = self.parameter.decode.sample(feat)
                reward = self.parameter.reward.call_dist(feat).mode()
                cont = self.parameter.cont.call_dist(feat).prob()
                value = self.parameter.critic.call_dist(feat).mode()
                s = f"{act_probs[a]*100:4.1f}%"
                s += f", {next_state[0].numpy()}"
                s += f", reward {reward.numpy()[0][0]:.5f}"
                s += f", done {(1-cont.numpy()[0][0])*100:4.1f}%"
                s += f", value {value.numpy()[0][0]:.5f}"
                return s

            common.render_discrete_action(maxa, worker.env, self.config, _render_sub)
        elif self.config.action_type == RLTypes.CONTINUOUS:
            action = self.parameter.actor.call_dist(self.feat).mode()
            deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
            feat = tf.concat([prior["stoch"], deter], axis=1)

            next_state = self.parameter.decode.sample(feat)
            reward = self.parameter.reward.call_dist(feat).mode()
            cont = self.parameter.cont.call_dist(feat).prob()
            value = self.parameter.critic.call_dist(feat).mode()

            print(f"next  : {next_state[0].numpy()}")
            print(f"reward: {reward[0][0].numpy()}")
            print(f"done  : {(1-cont[0][0].numpy())*100:4.1f}%")
            print(f"value : {value[0].numpy()}")

        else:
            raise ValueError(self.config.action_type)

    def render_rgb_array(self, worker, **kwargs) -> Optional[np.ndarray]:
        if self.config.observation_type != RLTypes.IMAGE:
            return None

        assert self.feat is not None
        if worker.done:
            self.policy(worker)

        state = worker.state

        from srl.utils import pygame_wrapper as pw

        _view_action = 4
        _view_sample = 3
        IMG_W = 64
        IMG_H = 64
        STR_H = 15
        PADDING = 4
        WIDTH = (IMG_W + PADDING) * _view_action + 5
        HEIGHT = (IMG_H + PADDING + STR_H * 3) * (_view_sample + 1) + 5

        if self.screen is None:
            self.screen = pw.create_surface(WIDTH, HEIGHT)
        pw.draw_fill(self.screen, color=(0, 0, 0))

        # --- decode
        pred_state = self.parameter.decode.sample(self.feat)[0].numpy()
        pred_reward = self.parameter.reward.call_dist(self.feat).mode()[0][0].numpy()
        pred_cont = self.parameter.cont.call_dist(self.feat).prob()[0][0].numpy()
        value = self.parameter.critic.call_dist(self.feat).mode()[0][0].numpy()
        rmse = np.sqrt(np.mean((state - pred_state) ** 2))

        img1 = np.clip(state * 255, 0, 255).astype(np.int64)
        img2 = np.clip(pred_state * 255, 0, 255).astype(np.int64)

        pw.draw_text(self.screen, 0, 0, "original", color=(255, 255, 255))
        pw.draw_image_rgb_array(self.screen, 0, STR_H, img1)
        pw.draw_text(
            self.screen,
            IMG_W + PADDING,
            0,
            f"decode(RMSE: {rmse:.4f})",
            color=(255, 255, 255),
        )
        pw.draw_image_rgb_array(self.screen, IMG_W + PADDING, STR_H, img2)

        pw.draw_text(
            self.screen,
            IMG_W * 2 + PADDING + 10,
            10,
            f"reward: {pred_reward:.4f}",
            color=(255, 255, 255),
        )
        pw.draw_text(
            self.screen,
            IMG_W * 2 + PADDING + 10,
            20,
            f"V     : {value:.4f}",
            color=(255, 255, 255),
        )
        pw.draw_text(
            self.screen,
            IMG_W * 2 + PADDING + 10,
            30,
            f"done  : {1-pred_cont:.4f}",
            color=(255, 255, 255),
        )

        if self.config.action_type == RLTypes.DISCRETE:
            act_dist = cast(CategoricalDistBlock, self.parameter.actor).call_dist(self.feat)
            act_probs = act_dist.probs().numpy()[0]

            # 横にアクション後の結果を表示
            for a in range(self.config.action_num):
                if a in self.get_invalid_actions():
                    continue
                if a > _view_action:
                    break

                pw.draw_text(
                    self.screen,
                    (IMG_W + PADDING) * a,
                    20 + IMG_H,
                    f"{worker.env.action_to_str(a)}({act_probs[a]*100:.1f}%)",
                    color=(255, 255, 255),
                )

                action = tf.one_hot([a], self.config.action_num, axis=1)
                deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
                feat = tf.concat([prior["stoch"], deter], axis=1)

                next_state = self.parameter.decode.sample(feat)
                reward = self.parameter.reward.call_dist(feat).mode()
                cont = self.parameter.cont.call_dist(feat).mode()
                value = self.parameter.critic.call_dist(feat).mode()

                n_img = next_state[0].numpy() * 255
                reward = reward.numpy()[0][0]
                cont = cont.numpy()[0][0]
                value = value.numpy()[0][0]

                x = (IMG_W + PADDING) * a
                y = 20 + IMG_H + STR_H
                pw.draw_text(
                    self.screen,
                    x,
                    y + STR_H * 0,
                    f"r={reward:.3f}",
                    color=(255, 255, 255),
                )
                pw.draw_text(
                    self.screen,
                    x,
                    y + STR_H * 1,
                    f"d={1-cont:.3f}",
                    color=(255, 255, 255),
                )
                pw.draw_text(
                    self.screen,
                    x,
                    y + STR_H * 2,
                    f"V={value:.3f}",
                    color=(255, 255, 255),
                )
                pw.draw_image_rgb_array(self.screen, x, y + STR_H * 3, n_img)

        elif self.config.action_type == RLTypes.CONTINUOUS:
            action = cast(NormalDistBlock, self.parameter.actor).call_dist(self.feat).mode()
            deter, prior = self.parameter.dynamics.img_step(self.stoch, self.deter, action)
            feat = tf.concat([prior["stoch"], deter], axis=1)

            next_state = self.parameter.decode.sample(feat)
            reward = self.parameter.reward.call_dist(feat).mode()
            cont = self.parameter.cont.call_dist(feat).prob()
            value = self.parameter.critic.call_dist(feat).mode()

            n_img = next_state[0].numpy() * 255
            s = f"act {action.numpy()[0][0]:.5f}"
            s += f", reward {reward.numpy()[0][0]:.5f}"
            s += f", done {(1-cont.numpy()[0][0])*100:4.1f}%"
            s += f", value {value.numpy()[0][0]:.5f}"

            x = IMG_W + PADDING
            y = 20 + IMG_H + STR_H + (IMG_H + PADDING + STR_H * 3)
            pw.draw_text(self.screen, x, y + STR_H * 1, s, color=(255, 255, 255))
            pw.draw_image_rgb_array(self.screen, x, y + STR_H * 3, n_img)

        else:
            raise ValueError(self.config.action_type)

        return pw.get_rgb_array(self.screen)
