import os
import sys

env_dir = os.path.join(os.path.dirname(__file__), "..", "..", "envs")
sys.path.insert(0, env_dir)

algorithm_dir = os.path.join(os.path.dirname(__file__), "..", "..", "algorithms")
sys.path.insert(0, algorithm_dir)

from envs import load_all  # noqa E402

load_all()
