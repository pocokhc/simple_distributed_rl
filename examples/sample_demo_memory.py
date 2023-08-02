import srl
from srl.algorithms import ql_agent57  # algorithm load
from srl.utils import common

common.logger_print()


def main():
    rl_config = ql_agent57.Config(memory_warmup_size=1, batch_size=1)

    runner = srl.Runner("Grid", rl_config)

    # demo play
    rl_config.memory.set_demo_memory(playing=True)
    runner.play_window(enable_remote_memory=True)
    rl_config.memory.set_demo_memory(playing=False)

    print(runner.remote_memory.length())

    # demo memory train
    runner.train_only(timeout=30, enable_eval=True)

    rewards = runner.evaluate()
    print(f"evaluate episodes: {rewards}")

    runner.render_terminal()


if __name__ == "__main__":
    main()
