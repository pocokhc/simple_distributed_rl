import pytest

pytest.importorskip("tensorflow")
import tensorflow as tf

from srl.rl.tf.distributions.categorical_gumbel_dist_block import CategoricalGumbelDist


@pytest.fixture
def logits() -> tf.Tensor:
    return tf.Variable([[2.0, 1.0, 0.1], [0.5, 1.5, 1.0]], dtype=tf.float32)


@pytest.fixture
def dist(logits) -> CategoricalGumbelDist:
    return CategoricalGumbelDist(logits)


def test_mode(dist: CategoricalGumbelDist):
    # What: 最も高いlogitのインデックスが返ることを確認
    result = dist.mode()
    expected = tf.cast(tf.Variable([0, 1]), tf.dtypes.int64)
    tf.debugging.assert_equal(result, expected)


def test_probs_sum_to_one(dist: CategoricalGumbelDist):
    # What: softmaxの出力が確率分布（合計が1）になっていること
    probs = dist.probs()
    tf.debugging.assert_near(tf.reduce_sum(probs, axis=-1), tf.ones([2]), atol=1e-5)


def test_sample_shape(dist: CategoricalGumbelDist):
    # What: sample()が(batch, 1)のshapeを返す
    sample = dist.sample()
    assert sample.shape == (2, 1)


def test_sample_topk_shape(dist: CategoricalGumbelDist):
    # What: sample_topk()が(batch, k)のshapeを返す
    k = 5
    topk = dist.sample_topk(k)
    assert topk.shape == (2, k)


def test_sample_topk_onehot_shape(dist: CategoricalGumbelDist):
    # What: sample_topk(onehot=True)が(batch, k, num_classes)のshapeを返す
    k = 3
    onehot_topk = dist.sample_topk(k, onehot=True)
    assert onehot_topk.shape == (2, k, 3)


def test_rsample_shape(dist: CategoricalGumbelDist):
    # What: rsample()が(batch, num_classes)のsoftmaxされた確率を返す
    sample = dist.rsample()
    assert sample.shape == (2, 3)
    tf.debugging.assert_near(tf.reduce_sum(sample, axis=-1), tf.ones([2]), atol=1e-5)


def test_log_probs_shape(dist: CategoricalGumbelDist):
    # What: log_probs()が(batch, num_classes)の形で返る
    log_probs = dist.log_probs()
    assert log_probs.shape == (2, 3)


def test_log_prob_shape(dist: CategoricalGumbelDist):
    # What: log_prob()が(batch, 1)の形で返る
    actions = tf.Variable([[0], [1]], dtype=tf.float32)
    log_prob = dist.log_prob(actions)
    assert log_prob.shape == (2, 1)


def test_entropy_shape(dist: CategoricalGumbelDist):
    # What: entropy()が(batch,)の形で返る
    entropy = dist.entropy()
    assert entropy.shape == (2,)


def test_compute_train_loss(dist: CategoricalGumbelDist):
    # What: compute_train_loss()が(batch,)の形で返る
    y = tf.one_hot([0, 1], depth=3)
    loss = dist.compute_train_loss(y)
    assert loss.shape == (2,)


@pytest.mark.parametrize("method_name", ["probs", "entropy", "rsample"])
def test_gradient_through_methods(method_name: str):
    # What: logitsに対して勾配が計算されるか（probs, entropy, rsample）

    logits = tf.Variable([[1.0, 2.0, 3.0], [0.5, 0.1, -0.3]], dtype=tf.float32)
    dist = CategoricalGumbelDist(logits)

    with tf.GradientTape() as tape:
        output = getattr(dist, method_name)()
        output = tf.reduce_sum(output)  # スカラー化して勾配を計算可能にする

    grads = tape.gradient(output, logits)

    assert grads is not None, f"{method_name}: Gradient should not be None"
    if method_name != "rsample":
        assert tf.reduce_any(tf.not_equal(grads, 0.0)), f"{method_name}: Gradient should not be all zero"


def test_gradient_through_log_prob():
    # What: log_probの出力からlogitsへの勾配が流れるか

    logits = tf.Variable([[1.0, 2.0, 3.0], [0.5, 0.1, -0.3]], dtype=tf.float32)
    dist = CategoricalGumbelDist(logits)

    # テスト用のアクション（argmaxに対応するone-hot）
    actions = tf.Variable([[2], [0]], dtype=tf.float32)

    with tf.GradientTape() as tape:
        log_prob = dist.log_prob(actions)
        log_prob = tf.reduce_sum(log_prob)

    grads = tape.gradient(log_prob, logits)

    assert grads is not None, "log_prob: Gradient should not be None"
    assert tf.reduce_any(tf.not_equal(grads, 0.0)), "log_prob: Gradient should not be all zero"


def test_backpropagation_from_compute_train_loss():
    # What: compute_train_loss()のlossに対してlogitsへの勾配が正しく計算されるか確認

    # テスト用のlogitsはVariableとして定義（勾配計算の対象）
    logits = tf.Variable([[2.0, 1.0, 0.1], [0.5, 1.5, 1.0]], dtype=tf.float32)
    dist = CategoricalGumbelDist(logits)
    y = tf.one_hot([0, 1], depth=3)

    with tf.GradientTape() as tape:
        loss = dist.compute_train_loss(y)
        loss = tf.reduce_mean(loss)  # スカラーにすることで勾配計算が可能

    grads = tape.gradient(loss, logits)

    # 勾配がNoneでなく、少なくとも1つが非ゼロであることを確認
    assert grads is not None, "Gradient should not be None"
    assert tf.reduce_any(tf.not_equal(grads, 0.0)), "Gradient should not be all zero"
