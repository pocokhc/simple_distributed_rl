import gym
import gym.spaces
import numpy as np


def tuple_to_box(space: gym.spaces.Tuple):
    low = []
    high = []
    shape_num = 0
    for s in space.spaces:
        if isinstance(s, gym.spaces.Discrete):
            low.append(0)
            high.append(s.n - 1)
            shape_num += 1
            continue

        # Boxはとりあえずshapeが1のみ対応
        if isinstance(s, gym.spaces.Box) and len(s.shape) == 1:
            for _ in range(s.shape[0]):
                low.append(s.low)
                high.append(s.high)
                shape_num += 1
            continue

        raise ValueError(f"Unimplemented: {s.__class__.__name__}")

    new_space = gym.spaces.Box(low=np.array(low), high=np.array(high), shape=(len(space.spaces),))
    return new_space
