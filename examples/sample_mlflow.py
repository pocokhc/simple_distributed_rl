import mlflow

import srl
from srl.algorithms import ql, vanilla_policy
from srl.utils import common

#
# MLFlow docs: https://mlflow.org/docs/latest/ml/
#
# Launch the MLflow Web UI (see the "mlruns" directory)
# > mlflow ui --backend-store-uri mlruns
#
# Save the data in the "mlruns" directory
mlflow.set_tracking_uri("mlruns")


def create_ql_runner():
    env_config = srl.EnvConfig("Grid")
    rl_config = ql.Config()
    return srl.Runner(env_config, rl_config)


def train_ql(timeout=30):
    runner = create_ql_runner()

    # Configuring Data Collection for MLFlow
    runner.set_mlflow(checkpoint_interval=10)

    runner.train(timeout=timeout)

    # Create an HTML video for all the parameters stored in MLFlow.
    runner.make_html_all_parameters_in_mlflow()


def load_ql_parameter():
    runner = create_ql_runner()

    # Load parameters saved in MLFlow
    runner.load_parameter_from_mlflow()

    rewards = runner.evaluate()
    print(rewards)


def train_vanilla_policy(timeout=30):
    env_config = srl.EnvConfig("Grid")
    rl_config = vanilla_policy.Config()
    runner = srl.Runner(env_config, rl_config)

    # Run directly using MLFlow
    runner.set_mlflow()
    mlflow.set_experiment("MyExperimentName")
    with mlflow.start_run(run_name="MyRunName"):
        runner.train(timeout=timeout)


if __name__ == "__main__":
    common.logger_print()

    train_ql()
    load_ql_parameter()
    train_vanilla_policy()
