import srl
from srl.algorithms import ql_agent57
from srl.utils import common


def _create_runner():
    rl_config = ql_agent57.Config(multisteps=1)
    rl_config.memory.set_proportional()

    # use demo_memory
    rl_config.memory.enable_demo_memory = True

    return srl.Runner("EasyGrid", rl_config)


def play():
    runner = _create_runner()

    # 手動で経験の収集
    runner.rl_config.memory.select_memory = "demo"

    runner.play_terminal(enable_memory=True)
    runner.save_memory("_sample_demo_memory.dat")


def main():
    runner = _create_runner()
    runner.load_memory("_sample_demo_memory.dat")
    runner.rl_config.memory.select_memory = "main"
    runner.train(max_train_count=100_000)

    rewards = runner.evaluate()
    print(f"evaluate episodes: {rewards}")

    runner.render_terminal(step_stop=True)


if __name__ == "__main__":
    common.logger_print()
    play()
    main()
