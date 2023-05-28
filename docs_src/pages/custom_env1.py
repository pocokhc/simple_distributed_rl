import srl
from srl.envs import sample_env  # noqa F401  # 実装したEnvファイルをimport

env = srl.make_env("SampleEnv")

state = env.reset(render_mode="terminal")
total_reward = 0
env.render()

while not env.done:
    action = env.sample()
    env.step(action)
    total_reward += env.reward
    print(f"step {env.step_num}, action {action}, reward {env.reward}, done {env.done}")
    env.render()
print(total_reward)
