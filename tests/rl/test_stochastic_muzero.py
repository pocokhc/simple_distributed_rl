import unittest

import srl
from srl.envs import grid
from srl.rl.models.alphazero_image_block import AlphaZeroImageBlock
from srl.test import TestRL


class Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tester = TestRL()

    def test_sequence(self):
        self.tester.play_sequence(srl.rl.stochastic_muzero.Config(), enable_image=True)

    def test_mp(self):
        self.tester.play_mp(srl.rl.stochastic_muzero.Config(), enable_image=True)

    def test_verify_grid(self):
        rl_config = srl.rl.stochastic_muzero.Config(
            num_simulations=20,
            discount=0.9,
            batch_size=16,
            memory_warmup_size=200,
            memory_name="ReplayMemory",
            lr_init=0.005,
            lr_decay_steps=10_000,
            v_min=-2,
            v_max=2,
            unroll_steps=1,
            input_image_block=AlphaZeroImageBlock,
            input_image_block_kwargs={"n_blocks": 1, "filters": 16},
            dynamics_blocks=1,
            enable_rescale=False,
            weight_decay=0,
            codebook_size=4,
        )
        rl_config.processors = [grid.LayerProcessor()]
        self.tester.play_verify_singleplay(
            "EasyGrid",
            rl_config,
            5000,
            test_num=10,
            is_valid=True,
        )


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_verify_grid", verbosity=2)
