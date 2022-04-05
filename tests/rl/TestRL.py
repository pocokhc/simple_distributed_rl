import numpy as np
import srl.envs.grid  # noqa F401
import srl.envs.igrid  # noqa F401
import srl.envs.oneroad  # noqa F401
from srl.runner import mp, sequence
from srl.runner.callbacks import PrintProgress


class TestRL:
    def __init__(self):
        self.parameter = None
        self.config = None

    def play_test(self, tester, rl_config):

        env_list = [
            "FrozenLake-v1",
            "OneRoad-v0",
        ]

        for env_name in env_list:
            config = sequence.Config(
                env_name=env_name,
                rl_config=rl_config,
            )
            with tester.subTest(("sequence", env_name, rl_config.getName())):
                self._sequence(config)

            with tester.subTest(("mp", env_name, rl_config.getName())):
                self._mp(config)

    def _sequence(self, config):
        # --- train
        config.set_play_config(max_train_count=10, training=True)
        episode_rewards, parameter, memory = sequence.play(config)

        # --- test
        config.set_play_config(max_episodes=1)
        episode_rewards, _, _ = sequence.play(config, parameter)

    def _mp(self, config):
        # --- train
        mp_config = mp.Config(worker_num=2)
        mp_config.set_train_config(max_train_count=10)
        parameter = mp.train(config, mp_config)

        # --- test
        config.set_play_config(max_episodes=1)
        episode_rewards, _, _ = sequence.play(config, parameter)

    def play_verify(
        self,
        tester,
        env_name,
        rl_config,
        train_count,
        test_episodes,
    ):
        base_score = {
            "Grid-v0": 0.7,  # 0.7318 ぐらい
            "Pendulum-v1": -500,  # -179.51776165585284ぐらい
            "IGrid-v0": 1,  # 乱数要素なし
        }
        assert env_name in base_score

        config = sequence.Config(
            env_name=env_name,
            rl_config=rl_config,
        )

        config.set_play_config(max_steps=1, training=True, callbacks=[PrintProgress(max_progress_time=5)])
        config.set_play_config(max_steps=train_count, training=True, callbacks=[PrintProgress(max_progress_time=5)])
        episode_rewards, parameter, memory = sequence.play(config)

        config.set_play_config(max_episodes=test_episodes)
        episode_rewards, _, _ = sequence.play(config, parameter)
        print(f"{np.mean(episode_rewards)} >= {base_score[env_name]}")
        tester.assertTrue(np.mean(episode_rewards) >= base_score[env_name])

        self.parameter = parameter
        self.config = config

    def check_policy(self, tester):
        assert self.config.env_name == "Grid-v0"

        env = self.config.create_env()
        V, _Q = env.env.calc_action_values()
        Q = {}
        for k, v in _Q.items():
            new_k = env.observation_encode(k)
            new_k = str(new_k.tolist())
            Q[new_k] = v

        worker = self.config.create_worker(self.parameter)
        worker.set_training(False)

        # 数ステップ回してactionを確認
        done = True
        for step in range(100):
            if done:
                state = env.reset()
                done = False
                total_reward = 0
                valid_actions = env.fetch_valid_actions()
                worker.on_reset(state, valid_actions, env)

            # action
            env_action, worker_action = worker.policy(state, valid_actions, env)
            if valid_actions is not None:
                assert env_action in valid_actions

            # -----------
            # policyのアクションと最適アクションが等しいか確認
            key = str(state.tolist())
            true_a = np.argmax(list(Q[key].values()))
            pred_a = env.action_decode(env_action)
            print(f"{state}: {true_a} == {pred_a}")
            tester.assertTrue(true_a == pred_a)
            # -----------

            # env step
            next_state, reward, done, env_info = env.step(env_action)
            step += 1
            total_reward += reward
            next_valid_actions = env.fetch_valid_actions()

            # rl step
            work_info = worker.on_step(
                state, worker_action, next_state, reward, done, valid_actions, next_valid_actions, env
            )
            state = next_state

    def check_action_values(self, tester):
        assert self.config.env_name == "Grid-v0"

        env = self.config.create_env()
        V, Q = env.env.calc_action_values()
        for s, q in Q.items():
            q = list(q.values())
            rl_s = env.observation_encode(s)
            true_a = np.argmax(q)

            rl_q = self.parameter.get_action_values(rl_s, env.env.actions)
            rl_a = np.argmax(rl_q)

            diff = abs(q[true_a] - rl_q[true_a])
            print(s, true_a, rl_a, diff)

            tester.assertTrue(true_a == rl_a)
            tester.assertTrue(diff < 0.2)
