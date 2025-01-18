# 実装したEnvファイルをimportし、registerに登録
from srl.envs import sample_env  # noqa F401
from srl.test.env import env_test

env_test("SampleEnv")
