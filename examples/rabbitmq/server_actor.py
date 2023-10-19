from srl.runner.rabbitmq import actor_run_forever
from srl.utils import common

if __name__ == "__main__":
    common.logger_print()
    actor_run_forever("127.0.0.1")
