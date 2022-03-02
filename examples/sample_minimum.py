import srl.rl.memory.registory
from srl import rl
from srl.base.rl.env_for_rl import create_env_for_rl
from srl.runner import mp, sequence
from srl.runner.callbacks import PrintProgress, RenderingEpisode
from srl.runner.callbacks_mp import TrainFileLogger


def main_raw():

    # parameter
    env_name = "FrozenLake-v1"
    rl_config = rl.ql.Config()
    memory_config = rl.memory.replay_memory.Config()
    training = True

    # init
    rl_config.assert_params()
    env = create_env_for_rl(env_name, rl_config)
    memory = srl.rl.memory.registory.make(memory_config)
    rl_module = rl.make(rl_config.getName())
    parameter = rl_module.Parameter(rl_config)
    trainer = rl_module.Trainer(rl_config, parameter)
    worker = rl_module.Worker(rl_config, parameter, 0)
    worker.set_training(training)

    # episode loop
    for episode in range(1):

        state = env.reset()
        done = False
        step = 0
        total_reward = 0
        valid_actions = env.fetch_valid_actions()
        worker.on_reset(state, valid_actions)

        for step in range(env.max_episode_steps):

            # action
            env_action, worker_action = worker.policy(state, valid_actions)
            if valid_actions is not None:
                assert env_action in valid_actions

            # env step
            next_state, reward, done, env_info = env.step(env_action)
            step += 1
            total_reward += reward
            next_valid_actions = env.fetch_valid_actions()

            # rl step
            work_return = worker.on_step(
                state, worker_action, next_state, reward, done, valid_actions, next_valid_actions
            )
            if training:
                batch, priority, work_info = work_return
                memory.add(batch, priority)
                train_info = trainer.train(memory)
            else:
                work_info = work_return
                train_info = None

            # render
            env.render()
            worker.render(state, valid_actions)
            print(
                "{} action {}, reward: {}, done: {}, info: {} {} {}".format(
                    step, env_action, reward, done, env_info, work_info, train_info
                )
            )

            # step after
            if done:
                break
            state = next_state
            valid_actions = next_valid_actions

        # episode result
        print(f"{episode} step: {step}, reward: {total_reward}")


def main_use_runner(is_mp):
    config = sequence.Config(
        env_name="FrozenLake-v1",
        rl_config=rl.ql.Config(),
        memory_config=rl.memory.replay_memory.Config(),
    )

    # --- train
    if not is_mp:
        # sequence training
        config.set_play_config(timeout=60, training=True, callbacks=[PrintProgress()])
        episode_rewards, parameter, memory = sequence.play(config)
    else:
        # distibute training
        mp_config = mp.Config(worker_num=2)
        mp_config.set_train_config(timeout=60, callbacks=[TrainFileLogger(enable_log=False, enable_checkpoint=False)])
        parameter = mp.train(config, mp_config)

    # --- test
    config.set_play_config(max_episodes=10, callbacks=[PrintProgress()])
    sequence.play(config, parameter)

    # --- rendering
    config.set_play_config(max_episodes=1, callbacks=[RenderingEpisode(step_stop=True)])
    sequence.play(config, parameter)


if __name__ == "__main__":

    # main_raw()
    # main_use_runner(is_mp=False)
    main_use_runner(is_mp=True)
