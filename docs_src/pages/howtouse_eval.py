import srl
from srl.algorithms import ql

runner = srl.Runner("Grid", ql.Config())
runner.train(timeout=1)

rewards = runner.evaluate(max_episodes=5)
print(f"evaluate episodes: {rewards}")
"""
evaluate episodes: [0.76, 0.8, 0.72, 0.76, 0.84]
"""
