from abc import ABC

import numpy as np
import torch


class AInputBlock(ABC):
    def to_torch_one_batch(self, data, device, torch_dtype, add_expand_dim: bool = True) -> torch.Tensor:
        tensor = torch.tensor(np.asarray(data), dtype=torch_dtype, device=device)
        return tensor.unsqueeze(0) if add_expand_dim else tensor

    def to_torch_batches(self, data, device, torch_dtype) -> torch.Tensor:
        return torch.tensor(np.asarray(data), dtype=torch_dtype, device=device)
