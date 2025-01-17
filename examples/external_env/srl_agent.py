import os

from srl_model import create_runner

# --- load model
runner = create_runner()
param_path = os.path.join(os.path.dirname(__file__), "_parameter.dat")
runner.load_parameter(param_path)

# --- setup env, worker
env = runner.make_env()
worker = runner.make_worker()
worker.on_start()


# --- 外部環境で指定されたフォーマットでagentを定義
# ※envとworkerのインスタンスを何かしらの形で保持する必要あり
# ※1～4をagent内で実行する必要あり
def my_agent(step: int, state: int) -> int:
    # 1. direct_step で外部から状態を元にstepを進める
    is_start_episode, is_end_episode = env.direct_step(step, state)

    # 2. もしエピソードの最初ならworkerのon_resetを実施
    if is_start_episode:
        worker.on_reset(env.next_player)

    # 3. workerからactionを取得
    srl_env_action = worker.policy()

    # 4. srlのアクションから外部環境のactionに変換
    external_action = env.decode_action(srl_env_action)
    return external_action
