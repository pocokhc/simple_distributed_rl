from srl.runner import distribution
from srl.utils import common


def main():
    common.logger_print()

    memory_params = None
    # memory_params = distribution.RabbitMQParameters(host="localhost", ssl=False)
    # memory_params = distribution.MQTTParameters(host="localhost")
    # memory_params = distribution.GCPParameters(project_id="YOUR_PROJECT_ID")

    distribution.actor_run_forever(distribution.RedisParameters(host="localhost"), memory_params)


if __name__ == "__main__":
    main()
