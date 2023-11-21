from srl.runner.distribution import RedisParameters, trainer_run_forever
from srl.utils import common


def main():
    common.logger_print()
    trainer_run_forever(RedisParameters(host="redis-internal-service"), None)


if __name__ == "__main__":
    main()
