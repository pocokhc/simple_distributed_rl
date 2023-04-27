import os

from model import create_config

# --- ディレクトリがあるかないかで kaggle 環境かローカルかを調べる
KAGGLE_PATH = "/kaggle_simulations/agent/"
if os.path.isdir(KAGGLE_PATH):
    is_local = False
    path = os.path.join(KAGGLE_PATH, "parameter.dat")
else:
    is_local = True
    path = os.path.join(os.path.dirname(__file__), "parameter.dat")

# --- make
config = create_config()
env = config.make_env()
parameter = config.make_parameter()
parameter.load(path)
worker = config.make_worker(parameter)


# --- agent
def my_agent(observation, configuration):
    env.direct_step(observation, configuration)

    if env.is_start_episode:
        worker.on_reset(env, env.next_player_index)
    action = worker.policy(env)

    return env.decode_action(action)


# -------------------------------------
# ローカル確認用
# 関数の下でも if で分けるなら大丈夫な模様
if is_local:
    import time

    import kaggle_environments
    import numpy as np

    config.model_summary()

    # kaggleのライブラリで動作するか検証
    kaggle_env = kaggle_environments.make("connectx", debug=True)
    for players in [
        [my_agent, "random"],
        ["random", my_agent],
        [my_agent, "negamax"],
        ["negamax", my_agent],
    ]:
        # 10episode実行
        rewards = []
        t0 = time.time()
        for _ in range(10):
            steps = kaggle_env.run(players)
            rewards.append([steps[-1][0]["reward"], steps[-1][1]["reward"]])

        # 結果
        rewards = np.mean(rewards, axis=0)
        print(f"rewards {rewards}, {time.time() - t0:.3f}s, {players}")
