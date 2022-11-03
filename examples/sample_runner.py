import numpy as np

import srl
from srl import runner

# --- env & algorithm load
from srl.envs import ox  # isort: skip # noqa F401
from srl.algorithms import ql  # isort: skip

# --- save parameter path
_parameter_path = "_parameter_ox_QL.dat"


def _create_config():
    env_config = srl.EnvConfig("OX")
    rl_config = ql.Config()
    config = runner.Config(env_config, rl_config)
    return config


def train():
    config = _create_config()

    if True:
        # sequence training
        parameter, remote_memory, history = runner.train(config, timeout=10)
    else:
        # distributed training
        parameter, remote_memory, history = runner.mp_train(config, timeout=10)

    # save parameter
    parameter.save(_parameter_path)


def evaluate():
    config = _create_config()
    config.rl_config.parameter_path = _parameter_path
    rewards = runner.evaluate(config, max_episodes=100)
    print(f"Average reward for 100 episodes: {np.mean(rewards, axis=0)}")


def render():
    config = _create_config()
    config.rl_config.parameter_path = _parameter_path
    runner.render(config)


def animation():
    config = _create_config()
    config.rl_config.parameter_path = _parameter_path
    render = runner.animation(config)
    render.create_anime().save("_OX.gif")


def test_play():
    config = _create_config()
    config.rl_config.parameter_path = _parameter_path
    history = runner.test_play(config)
    history.replay()


def training_history():
    #
    # 'runner.train' を実行すると tmp 配下に log ディレクトリが生成されます。
    # その log ディレクトリをロードすると学習過程を読み込むことができます。
    # (不要な log ディレクトリは削除して問題ありません)
    #
    # --- English
    # Running 'runner.train' will create a log directory under tmp.
    # You can load the log directory to read the training process.
    # (you can delete the log directory if you don't need it)
    #
    log_dir = "tmp/YYYYMMDD_HHMMSS_XXX_XXX"
    history = runner.load_history(log_dir)

    # get raw data
    logs = history.get_logs()

    # get pandas DataFrame
    # (Run "pip install pandas" to use the history.get_df())
    df_logs = history.get_df()
    print(df_logs)

    # plot
    # (Run "pip install matplotlib pandas" to use the history.plot())
    history.plot()


if __name__ == "__main__":
    train()
    evaluate()
    render()
    animation()
    test_play()
    # training_history()
