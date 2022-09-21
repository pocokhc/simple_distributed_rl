import unittest

from srl.base.rl.remote_memory.experience_replay_buffer import \
    ExperienceReplayBuffer


class Test(unittest.TestCase):
    def test_memory(self):
        capacity = 10

        memory = ExperienceReplayBuffer(None)
        memory.init(capacity)

        self.assertTrue(memory.length() == 0)

        # add
        for i in range(100):
            memory.add((i, i, i, i))
        self.assertTrue(memory.length() == capacity)

        # sample
        batchs = memory.sample(5)
        self.assertTrue(len(batchs) == 5)

        # sample over
        batchs = memory.sample(20)
        self.assertTrue(len(batchs) == 10)

        # restore/backup
        dat = memory.backup(compress=True)
        memory2 = ExperienceReplayBuffer(None)
        memory2.init(capacity)
        memory2.restore(dat)
        self.assertTrue(memory2.length() == 10)

        # restore over
        memory2 = ExperienceReplayBuffer(None)
        memory2.init(20)
        memory2.restore(dat)
        self.assertTrue(memory2.length() == 10)
        memory2.add((11, 11, 11, 11))
        self.assertTrue(memory2.length() == 11)
        self.assertTrue(memory2.memory[10] == (11, 11, 11, 11))

        # restore min
        memory2 = ExperienceReplayBuffer(None)
        memory2.init(5)
        memory2.restore(dat)
        self.assertTrue(memory2.length() == 5)
        self.assertTrue(memory2.memory[4] == (99, 99, 99, 99))
        memory2.add((11, 11, 11, 11))
        self.assertTrue(memory2.length() == 5)
        self.assertTrue(memory2.memory[0] == (11, 11, 11, 11))


if __name__ == "__main__":
    unittest.main(module=__name__, defaultTest="Test.test_memory", verbosity=2)
