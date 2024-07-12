import mlflow
import srl
from srl.algorithms import ql, vanilla_policy
from srl.utils import common

common.logger_print()

# > mlflow ui --backend-store-uri mlruns
mlflow.set_tracking_uri("mlruns")


def create_ql_runner():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    return srl.Runner(env_config, rl_config)


def train_ql():
    runner = create_ql_runner()
    runner.set_mlflow()
    runner.train(timeout=30)


def load_ql_parameter():
    runner = create_ql_runner()
    runner.load_parameter_from_mlflow()
    rewards = runner.evaluate()
    print(rewards)


def train_vanilla_policy():
    env_config = srl.EnvConfig("Grid")
    rl_config = vanilla_policy.Config()
    runner = srl.Runner(env_config, rl_config)

    runner.set_mlflow()
    mlflow.set_experiment("MyExperimentName")
    with mlflow.start_run(run_name="MyRunName"):
        runner.train(timeout=30)


if __name__ == "__main__":
    train_ql()
    load_ql_parameter()
    train_vanilla_policy()
