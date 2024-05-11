import srl

# 実装したEnvファイルをimportし、registerに登録
from srl.envs import sample_env  # noqa F401

env = srl.make_env("SampleEnv")
env.setup(render_mode="terminal")

state = env.reset()
total_reward = 0
env.render()

while not env.done:
    action = env.sample_action()
    env.step(action)
    total_reward += env.reward
    print(f"step {env.step_num}, action {action}, reward {env.reward}, done {env.done}")
    env.render()
print(total_reward)
