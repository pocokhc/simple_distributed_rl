from srl.runner.distribution import ServerParameters, trainer_run_forever


def main():
    params = ServerParameters(redis_host="localhost", rabbitmq_host="localhost")
    trainer_run_forever(params)


if __name__ == "__main__":
    main()
