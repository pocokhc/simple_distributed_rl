import pytest
import pytest_mock
import pytest_timeout  # noqa F401

import srl
from srl.algorithms import ql_agent57
from srl.base.exception import DistributionError
from tests.distribution.runner.distribution_mock.server_mock import (
    create_gcp_mock,
    create_paho_mock,
    create_pika_mock,
    create_redis_mock,
)

try:
    from srl.runner.distribution import actor_run_forever, trainer_run_forever
    from srl.runner.distribution.callback import ActorServerCallback, TrainerServerCallback
    from srl.runner.distribution.connectors.parameters import (
        GCPParameters,
        MQTTParameters,
        RabbitMQParameters,
        RedisParameters,
    )
    from srl.runner.distribution.task_manager import TaskConfig, TaskManager
except ModuleNotFoundError:
    pass


@pytest.mark.parametrize("server", ["", "redis", "pika", "paho", "gcp"])
@pytest.mark.parametrize("enable_actor_thread", [False, True])
@pytest.mark.timeout(60)  # pip install pytest_timeout
def test_server_actor(mocker: pytest_mock.MockerFixture, server, enable_actor_thread):
    pytest.importorskip("redis")
    create_redis_mock(mocker)
    if server == "":
        memory_params = None
    elif server == "redis":
        memory_params = RedisParameters(host="test")
    elif server == "pika":
        pytest.importorskip("pika")
        create_pika_mock(mocker)
        memory_params = RabbitMQParameters(host="test")
    elif server == "paho":
        pytest.skip("mockがうまくいかないのでskip")
        create_paho_mock(mocker)
        memory_params = MQTTParameters(host="test")
    elif server == "gcp":
        pytest.skip("GCPは要検討")
        create_gcp_mock(mocker)
        memory_params = GCPParameters(project_id="test")
    else:
        raise

    redis_params = RedisParameters(host="test")

    # --- create task
    context = srl.RunContext(srl.EnvConfig("Grid"), ql_agent57.Config())
    context.rl_config.setup(context.env_config.make())
    context.timeout = 10
    context.max_memory = 100
    context.training = True
    task_manager = TaskManager(redis_params, "client")
    task_manager.create_task(
        TaskConfig(
            context,
            callbacks=[],
            enable_actor_thread=enable_actor_thread,
            actor_parameter_sync_interval=0,
        ),
        context.rl_config.make_parameter(),
    )
    if memory_params is None:
        task_manager.setup_memory(redis_params.create_memory_receiver(), is_purge=True)
    else:
        task_manager.setup_memory(memory_params.create_memory_receiver(), is_purge=True)

    # --- run
    c2 = mocker.Mock(spec=ActorServerCallback)
    try:
        actor_run_forever(
            redis_params,
            memory_params,
            keepalive_interval=0,
            callbacks=[c2],
            framework="",
            run_once=True,
        )
    except DistributionError:
        pass

    assert c2.on_polling.call_count > 0

    if memory_params is None:
        memory_params = redis_params
    qsize = memory_params.create_memory_sender().memory_size()
    assert qsize == -1 or qsize > 0
    memory_receiver = memory_params.create_memory_receiver()
    assert memory_receiver.memory_recv() is not None

    batch = memory_receiver.memory_recv()
    print(batch)


@pytest.mark.parametrize("server", ["", "redis", "pika", "paho", "gcp"])
@pytest.mark.parametrize("dist_option", [False, True])
@pytest.mark.timeout(60)  # pip install pytest_timeout
def test_server_trainer(mocker: pytest_mock.MockerFixture, server, dist_option):
    pytest.importorskip("redis")
    create_redis_mock(mocker)
    if server == "":
        memory_params = None
    elif server == "redis":
        memory_params = RedisParameters(host="test")
    elif server == "pika":
        pytest.importorskip("pika")
        create_pika_mock(mocker)
        memory_params = RabbitMQParameters(host="test")
    elif server == "paho":
        pytest.skip("mockがうまくいかないのでskip")
        create_paho_mock(mocker)
        memory_params = MQTTParameters(host="test")
    elif server == "gcp":
        pytest.skip("GCPは要検討")
        create_gcp_mock(mocker)
        memory_params = GCPParameters(project_id="test")
    else:
        raise

    redis_params = RedisParameters(host="test")

    # --- create task
    rl_config = ql_agent57.Config()
    rl_config.memory.warmup_size = 10
    rl_config.batch_size = 1
    context = srl.RunContext(srl.EnvConfig("Grid"), rl_config)
    context.rl_config.setup(context.env_config.make())
    context.max_train_count = 5
    context.training = True
    task_manager = TaskManager(redis_params, "client")
    task_manager.create_task(
        TaskConfig(
            context,
            callbacks=[],
            enable_trainer_thread=dist_option,
            trainer_parameter_send_interval=0,
        ),
        context.rl_config.make_parameter(),
    )
    if memory_params is None:
        memory_params2 = redis_params
    else:
        memory_params2 = memory_params
    task_manager.setup_memory(redis_params.create_memory_receiver(), is_purge=True)

    # add batch
    memory_sender = memory_params2.create_memory_sender()
    base_memory = rl_config.make_memory()
    serialize_func = base_memory.get_worker_funcs()["add"][1]
    for _ in range(100):
        batch = {
            "states": ["1,3", "1,3"],
            "actions": [0],
            "probs": [0.25],
            "ext_rewards": [-0.03999999910593033],
            "int_rewards": [5.0],
            "invalid_actions": [[], []],
            "done": False,
            "discount": 0.9999,
        }
        memory_sender.memory_add(("add", serialize_func(batch, 0)))

    # --- run
    assert task_manager.get_trainer("train") == ""
    c2 = mocker.Mock(spec=TrainerServerCallback)
    trainer_run_forever(
        RedisParameters(host="test"),
        memory_params,
        keepalive_interval=0,
        callbacks=[c2],
        framework="",
        run_once=True,
        is_remote_memory_purge=False,
    )
    assert c2.on_polling.call_count > 0
    assert int(task_manager.get_trainer("train")) > 0
