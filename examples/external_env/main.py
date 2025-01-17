"""
学習したモデルを元に、外部の環境を利用してエージェントを動作させる
"""

import srl_agent
from env_external import run_external_env

run_external_env(srl_agent.my_agent)
