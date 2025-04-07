from abc import ABC, abstractmethod


class AInputBlock(ABC):
    @abstractmethod
    def create_dummy_data(self, np_dtype, batch_size: int = 1, timesteps: int = 1):
        raise NotImplementedError()

    @abstractmethod
    def to_tf_one_batch(self, data, tf_dtype, add_expand_dim: bool = True):
        raise NotImplementedError()

    @abstractmethod
    def to_tf_batches(self, data, tf_dtype):
        raise NotImplementedError()
