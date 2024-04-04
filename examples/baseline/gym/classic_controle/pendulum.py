import os

import numpy as np

import srl
from srl.algorithms import agent57, agent57_light, ddpg, dqn, dreamer_v3, ppo, r2d2, rainbow, sac, search_dqn
from srl.utils import common

base_dir = os.path.dirname(__file__)
ENV_NAME = "Pendulum-v1"

BASE_TRAIN = 200 * 200
BASE_LR = 0.001
BASE_BLOCK = (64, 64)


def _run(name, rl_config, is_mp):
    common.logger_print()

    rl_config.memory_compress = False
    runner = srl.Runner(ENV_NAME, rl_config)
    runner.set_history_on_file(
        os.path.join(base_dir, f"_{ENV_NAME}_{name}" + ("_mp" if is_mp else "")),
        enable_eval=True,
        eval_episode=10,
    )
    runner.model_summary()
    if is_mp:
        runner.train_mp(max_train_count=BASE_TRAIN)
    else:
        runner.train(max_train_count=BASE_TRAIN)
    rewards = runner.evaluate()
    print(f"[{name}] evaluate episodes: {np.mean(rewards)}")


def main_dqn(is_mp):
    rl_config = dqn.Config(
        lr=BASE_LR,
        target_model_update_interval=1000,
    )
    rl_config.hidden_block.set(BASE_BLOCK)
    rl_config.memory.warmup_size = 1000
    rl_config.memory.capacity = 10_000
    _run("DQN", rl_config, is_mp)


def main_rainbow(is_mp):
    rl_config = rainbow.Config(
        lr=BASE_LR,
        target_model_update_interval=1000,
    )
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    rl_config.memory.warmup_size = 1000
    rl_config.memory.capacity = 10_000
    _run("Rainbow", rl_config, is_mp)


def main_agent57_light(is_mp):
    rl_config = agent57_light.Config(
        lr_ext=BASE_LR,
        lr_int=BASE_LR,
        target_model_update_interval=1000,
    )
    rl_config.enable_intrinsic_reward = False
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    rl_config.memory.warmup_size = 1000
    rl_config.memory.capacity = 10_000
    _run("Agent57_light", rl_config, is_mp)


def main_r2d2(is_mp):
    rl_config = r2d2.Config(
        lr=BASE_LR,
        target_model_update_interval=1000,
        burnin=5,
        sequence_length=2,
    )
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 1000
    _run("R2D2", rl_config, is_mp)


def main_agent57(is_mp):
    rl_config = agent57.Config(
        lr_ext=BASE_LR,
        lr_int=BASE_LR,
        target_model_update_interval=1000,
        burnin=5,
        sequence_length=2,
    )
    rl_config.enable_intrinsic_reward = False
    rl_config.hidden_block.set_dueling_network(BASE_BLOCK)
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 1000
    _run("Agent57", rl_config, is_mp)


def main_ppo(is_mp):
    rl_config = ppo.Config(lr=BASE_LR)
    rl_config.hidden_block.set(BASE_BLOCK)
    rl_config.policy_block.set(BASE_BLOCK)
    rl_config.value_block.set(BASE_BLOCK)
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 1000
    _run("PPO", rl_config, is_mp)


def main_ddpg(is_mp):
    rl_config = ddpg.Config(lr=BASE_LR)
    rl_config.policy_block.set(BASE_BLOCK)
    rl_config.q_block.set(BASE_BLOCK)
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 1000
    _run("DDPG", rl_config, is_mp)


def main_sac(is_mp):
    rl_config = sac.Config(lr_policy=BASE_LR, lr_q=BASE_LR)
    rl_config.policy_hidden_block.set(BASE_BLOCK)
    rl_config.q_hidden_block.set(BASE_BLOCK)
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 1000
    _run("SAC", rl_config, is_mp)


def main_dreamer_v3(is_mp):
    rl_config = dreamer_v3.Config(lr_model=BASE_LR, lr_critic=BASE_LR, lr_actor=BASE_LR)
    rl_config.set_dreamer_v3()
    rl_config.rssm_deter_size = 64
    rl_config.rssm_stoch_size = 4
    rl_config.rssm_classes = 4
    rl_config.rssm_hidden_units = 256
    rl_config.reward_layer_sizes = (256,)
    rl_config.cont_layer_sizes = (256,)
    rl_config.encoder_decoder_mlp = BASE_BLOCK
    rl_config.critic_layer_sizes = BASE_BLOCK
    rl_config.actor_layer_sizes = BASE_BLOCK
    rl_config.batch_size = 32
    rl_config.batch_length = 15
    rl_config.horizon = 5
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 50
    rl_config.free_nats = 0.1
    rl_config.warmup_world_model = 1_000
    _run("DreamerV3", rl_config, is_mp)


def main_search_dqn(is_mp=False):
    rl_config = search_dqn.Config(lr=BASE_LR)
    rl_config.set_layer_sizes(BASE_BLOCK[0], BASE_BLOCK)
    rl_config.memory.capacity = 10_000
    rl_config.memory.warmup_size = 100
    _run("SearchDQN", rl_config, is_mp)


def compare1():
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_old/_{ENV_NAME}_DQN"),
            os.path.join(base_dir, f"_{ENV_NAME}_SearchDQN"),
            # os.path.join(base_dir, f"_{ENV_NAME}_Rainbow"),
            # os.path.join(base_dir, f"_{ENV_NAME}_Agent57_light"),
            # os.path.join(base_dir, f"_{ENV_NAME}_R2D2"),
            # os.path.join(base_dir, f"_{ENV_NAME}_Agent57"),
            # os.path.join(base_dir, f"_{ENV_NAME}_PPO"),
            # os.path.join(base_dir, f"_{ENV_NAME}_DDPG"),
            # os.path.join(base_dir, f"_{ENV_NAME}_SAC"),
            # os.path.join(base_dir, f"_{ENV_NAME}_DreamerV3"),
        ]
    )
    histories.plot(
        "train",
        "eval_reward0",
        title=f"Train:{BASE_TRAIN}, lr={BASE_LR}, block={BASE_BLOCK}",
    )
    histories.plot(
        "time",
        "eval_reward0",
        title=f"Train:{BASE_TRAIN}, lr={BASE_LR}, block={BASE_BLOCK}",
    )


def compare2():
    histories = srl.Runner.load_histories(
        [
            os.path.join(base_dir, f"_{ENV_NAME}_DQN_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_Rainbow_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_Agent57_light_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_R2D2_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_Agent57_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_PPO_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_DDPG_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_SAC_mp"),
            os.path.join(base_dir, f"_{ENV_NAME}_DreamerV3_mp"),
        ]
    )
    histories.plot(
        "train",
        "eval_reward0",
        title=f"Train:{BASE_TRAIN}, lr={BASE_LR}, block={BASE_BLOCK}",
    )
    histories.plot(
        "time",
        "eval_reward0",
        title=f"Train:{BASE_TRAIN}, lr={BASE_LR}, block={BASE_BLOCK}",
    )


if __name__ == "__main__":
    # main_dqn(is_mp=False)
    # main_dqn(is_mp=True)
    # main_rainbow(is_mp=False)
    # main_rainbow(is_mp=True)
    # main_agent57_light(is_mp=False)
    # main_agent57_light(is_mp=True)
    # main_r2d2(is_mp=False)
    # main_r2d2(is_mp=True)
    # main_agent57(is_mp=False)
    # main_agent57(is_mp=True)
    # main_ppo(is_mp=False)
    # main_ppo(is_mp=True)
    # main_ddpg(is_mp=False)
    # main_ddpg(is_mp=True)
    # main_sac(is_mp=False)
    # main_sac(is_mp=True)
    # main_dreamer_v3(is_mp=False)
    # main_dreamer_v3(is_mp=True)
    main_search_dqn(is_mp=False)
    # compare1()
    # compare2()

    if False:
        # import cProfile

        # cProfile.run("main()", filename="main.prof")
        # import pstats

        # pip install snakeviz
        # snakeviz main.prof
        # 1行目：呼び出した関数の数と実行時間
        # Ordered by: 出力のソート方法
        # --
        # ncalls: 呼び出し回数
        # ☆tottime: subfunctionの実行時間を除いた時間(別の関数呼び出しにかかった時間を除く)
        # ☆percall: tottimeをncallsで割った値
        # cumtime: この関数とそのsubfuntionに消費された累積時間
        # percall: cumtimeを呼び出し回数で割った値
        # sts = pstats.Stats("main.prof")
        # sts.strip_dirs().sort_stats(-1).print_stats()

        from line_profiler import LineProfiler  # pip install line_profiler

        prf = LineProfiler()
        # prf.add_module(WorkerRun)
        prf.add_module(search_dqn.search_dqn.Trainer)
        # prf.add_module(env_run.EnvRun)
        # prf.add_function(RLTrainer.train)
        # prf.add_function(core._play_trainer_only)
        # prf.add_function(ddqn.CommonInterfaceParameter.calc_target_q)
        # prf.add_function(core_play._play)
        # prf.add_function(core._play_train_only)
        prf.runcall(main_search_dqn)
        prf.print_stats()
        """
        Line: 全体の中の行数
        Hits: 呼び出し回数
        Time: かかった時間
        Per Hit: １呼び出しあたりにかかった時間
        % Time: その関数内でかかった時間の割合
        Line Contents: その行の内容
        """


"""
07:07:03  7.47s( 12.9m left),    51st/s,   1046st,  1045mem,   5ep,    47tr, 200st,-1243.9 -1169.0 -1094.0 re|epsilon 0.100|sync  1|lr 0.001|loss 2.448
07:07:13 17.48s( 11.0m left),    59st/s,   1643st,  1642mem,   8ep,   644tr, 200st,-1559.8 -1478.9 -1388.0 re|epsilon 0.100|sync  1|lr 0.001|loss 0.319
07:07:33 37.75s( 15.0m left),    42st/s,   2509st,  2508mem,  12ep,  1510tr, 200st,-1557.2 -1361.1 -1168.3 re|epsilon 0.100|sync  2|lr 0.001|loss 0.033
07:08:13   1.3m( 12.3m left),    49st/s,   4492st,  4491mem,  22ep,  3493tr, 200st,-1675.3 -1510.7 -1370.4 re|epsilon 0.100|sync  4|lr 0.001|loss 0.030
07:09:36   2.7m( 11.7m left),    46st/s,   8342st,  8341mem,  41ep,  7343tr, 200st,-1531.6 -1264.9 -1048.6 re|epsilon 0.100|sync  8|lr 0.001|loss 0.057
07:12:18   5.4m(  9.1m left),    46st/s,  15837st, 10000mem,  79ep, 14838tr, 200st,-1024.6 -480.5 -2.176 re|epsilon 0.100|sync 15|lr 0.001|loss 0.284
07:17:40  10.7m(  3.6m left),    46st/s,  30887st, 10000mem, 154ep, 29888tr, 200st,-1010.2 -234.0 -1.156 re|epsilon 0.100|sync 30|lr 0.001|loss 0.286
2024-03-30 07:21:31,895 srl.base.run.core_play _play 243 [INFO] [RunNameTypes.main] loop end(max_train_count over.)
07:21:31  14.6m( 0.00s left),    43st/s,  40999st, 10000mem, 204ep, 40000tr, 200st,-525.8 -186.6 -2.351 re|epsilon 0.100|sync 40|lr 0.001|loss 0.195
2024-03-30 07:21:31,923 srl.runner.runner_facade evaluate 677 [INFO] add callback PrintProgress
2024-03-30 07:21:31,923 srl.runner.runner _base_run_play_before 995 [INFO] HistoryOnFile is disabled.
### env: Pendulum-v1, rl: DQN:tensorflow, max episodes: 10
2024-03-30 07:21:31,924 srl.base.run.core_play play 87 [INFO] tf.device(/GPU)
2024-03-30 07:21:31,924 srl.base.run.core_play _play 138 [INFO] [RunNameTypes.main] loop start
07:21:32  1.00s( 2.34s left),   650st/s,    651st, 10000mem,   3ep, 200st,-432.0 -227.8 -123.1 re|epsilon  0
2024-03-30 07:21:34,878 srl.base.run.core_play _play 243 [INFO] [RunNameTypes.main] loop end(episode_count over.)
07:21:34  2.95s( 0.00s left),   690st/s,   2000st, 10000mem,  10ep, 200st,-348.1 -209.8 -123.6 re
"""
