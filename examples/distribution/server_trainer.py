from srl.runner.distribution import RabbitMQParameters, RedisParameters, trainer_run_forever


def main():
    trainer_run_forever(
        RedisParameters(host="localhost"),
        RabbitMQParameters(host="localhost", ssl=False),
    )


if __name__ == "__main__":
    main()
