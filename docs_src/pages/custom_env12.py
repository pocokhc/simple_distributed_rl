import srl

# 実装したEnvファイルをimportし、registerに登録
from srl.envs import sample_env  # noqa F401

runner = srl.Runner("SampleEnv")
runner.render_terminal()
