import unittest

from srl.test import TestRL
from srl.utils.common import is_package_installed

try:
    from srl.algorithms import muzero
    from srl.envs import grid
    from srl.rl.models.alphazero_image_block import AlphaZeroImageBlock
except ModuleNotFoundError:
    pass


@unittest.skipUnless(is_package_installed("tensorflow"), "no module")
class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_simple_check(self):
        self.tester.simple_check(muzero.Config(), enable_image=True)

    def test_simple_check_mp(self):
        self.tester.simple_check_mp(muzero.Config(), enable_image=True)

    def test_verify_grid(self):
        rl_config = muzero.Config(
            num_simulations=20,
            discount=0.9,
            batch_size=16,
            memory_warmup_size=200,
            memory_name="ReplayMemory",
            lr_init=0.002,
            lr_decay_steps=10_000,
            v_min=-2,
            v_max=2,
            unroll_steps=1,
            input_image_block=AlphaZeroImageBlock,
            input_image_block_kwargs={"n_blocks": 1, "filters": 16},
            dynamics_blocks=1,
            enable_rescale=False,
            weight_decay=0,
        )
        rl_config.processors = [grid.LayerProcessor()]
        self.tester.verify_singleplay(
            "EasyGrid",
            rl_config,
            1500,
            test_num=10,
            # is_valid=True,
        )

    def test_verify_grid_PER(self):
        rl_config = muzero.Config(
            num_simulations=20,
            discount=0.9,
            batch_size=16,
            memory_warmup_size=200,
            lr_init=0.002,
            lr_decay_steps=10_000,
            v_min=-2,
            v_max=2,
            unroll_steps=1,
            input_image_block=AlphaZeroImageBlock,
            input_image_block_kwargs={"n_blocks": 1, "filters": 16},
            dynamics_blocks=1,
            enable_rescale=False,
            weight_decay=0,
        )
        rl_config.processors = [grid.LayerProcessor()]
        self.tester.verify_singleplay(
            "EasyGrid",
            rl_config,
            3000,
            test_num=10,
        )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_sequence", verbosity=2)
