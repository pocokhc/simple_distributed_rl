import cProfile
import random

from tqdm import tqdm

from srl.rl.memories.priority_memories.proportional_memory import ProportionalMemory


def main():
    capacity = 100_000_000
    memory = ProportionalMemory(capacity, 0.8, 0.4, 1000)
    # memory = RankBaseMemory(capacity, 0.8, 0.4, 1000)

    _run(memory)


def _run(memory):
    warmup_size = 10_000
    batch_size = 64
    epochs = 10_000

    # warmup
    step = 0
    for _ in range(warmup_size + batch_size):
        r = random.random()
        memory.add((step, step, step, step), r)
        step += 1

    for _ in tqdm(range(epochs)):
        # add
        r = random.random()
        memory.add((step, step, step, step), r)
        step += 1

        # sample
        batches, weights, update_args = memory.sample(batch_size, step)
        assert len(batches) == batch_size
        assert len(weights) == batch_size

        # update priority
        priorities = [random.random() for _ in range(batch_size)]
        memory.update(update_args, priorities)


if __name__ == "__main__":
    cProfile.run("main()", filename="main.prof")

    import pstats

    sts = pstats.Stats("main.prof")
    sts.strip_dirs().sort_stats(-1).print_stats()

    # --- gui
    # pip install snakeviz
    # snakeviz main.prof

    # --- 見方
    # 呼び出した関数数 function calls in 掛かった時間 seconds
    # Ordered by: 出力のソート方法
    #
    # ncalls: 呼び出し回数
    # ☆tottime: subfunctionの実行時間を除いた時間(別の関数呼び出しにかかった時間を除く)
    # ☆percall: tottimeをncallsで割った値
    # cumtime: この関数とそのsubfuntionに消費された累積時間
    # percall: cumtimeを呼び出し回数で割った値
    #
