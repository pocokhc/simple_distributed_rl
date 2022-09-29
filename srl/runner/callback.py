from abc import ABC


class Callback(ABC):
    def on_episodes_begin(self, info) -> None:
        pass  # do nothing

    def on_episodes_end(self, info) -> None:
        pass  # do nothing

    def on_episode_begin(self, info) -> None:
        pass  # do nothing

    def on_episode_end(self, info) -> None:
        pass  # do nothing

    def on_step_action_before(self, info) -> None:
        pass  # do nothing

    def on_step_begin(self, info) -> None:
        pass  # do nothing

    def on_step_end(self, info) -> None:
        pass  # do nothing

    def on_skip_step(self, info) -> None:
        pass  # do nothing

    # 外部から途中停止用
    def intermediate_stop(self, info) -> bool:
        return False

    # -------------------------
    # TrainerCallback
    # -------------------------
    def on_trainer_start(self, info) -> None:
        pass  # do nothing

    def on_trainer_train(self, info) -> None:
        pass  # do nothing

    def on_trainer_end(self, info) -> None:
        pass  # do nothing


class MPCallback(ABC):
    # all
    def on_init(self, info) -> None:
        pass  # do nothing

    # main
    def on_start(self, info) -> None:
        pass  # do nothing

    def on_polling(self, info) -> None:
        pass  # do nothing

    def on_end(self, info) -> None:
        pass  # do nothing
