from srl.rl.schedulers.scheduler import SchedulerConfig


def main():
    sch = SchedulerConfig()
    sch.add_linear(0.5, 1.0, 100)
    sch.add_linear(1.0, 0.1, 200)
    sch.add_cosine(0.7, 0.2, 200)
    sch.add_cosine_with_hard_restarts(0.7, 0.2, 500, 5)
    sch.add_polynomial(1.5, 0.3, 200)
    sch.add(0.2)
    sch.plot()


def main2():
    sch = SchedulerConfig()
    sch.add_linear(-1, 2, 1000)
    # sch.add_cosine(-1, 2, 1000)
    # sch.add_polynomial(0.5, -0.5, 10000, 2)
    sch.add(0.2)
    sch.plot()


if __name__ == "__main__":
    # main()
    main2()
