import pytest

from srl.base.env.registration import register as register_env
from srl.base.rl.registration import register as register_rl
from srl.utils.common import is_package_installed
from tests.quick.base.rl import worker_run_stub


@pytest.fixture(scope="function", autouse=True)
def scope_function():
    register_env(id="Stub", entry_point=worker_run_stub.__name__ + ":WorkerRunStubEnv", check_duplicate=False)
    register_rl(worker_run_stub.WorkerRunStubRLConfig(), "", "", "", worker_run_stub.__name__ + ":WorkerRunStubRLWorker", check_duplicate=False)
    yield


def test_env_play():
    from srl.test.env import env_test

    env_test("Stub", test_render_window=is_package_installed("pygame"))
