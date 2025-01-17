"""
学習したモデルを元に、外部環境を経由でエージェントを動作させる
"""

import srl_agent
from env_external import run_external_env

run_external_env(srl_agent.my_agent)
