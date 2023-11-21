import srl
from srl.algorithms import ql
from srl.runner.distribution import RedisParameters

runner = srl.Runner("Grid", ql.Config())
runner.train_distribution(
    RedisParameters(host="redis-internal-service"),
    actor_num=1,
    max_train_count=20_000,
)
print(runner.evaluate())
