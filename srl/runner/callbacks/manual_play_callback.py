from srl.base.context import RunContext
from srl.base.env.env_run import EnvRun
from srl.base.run.callback import RunCallback
from srl.base.run.core_play import RunStateActor


class ManualPlayCallback(RunCallback):
    def __init__(self, env: EnvRun, action_division_num: int):
        env.action_space.create_division_tbl(action_division_num)
        self.action_num = env.action_space.create_encode_space_DiscreteSpace().n

    def on_step_action_after(self, context: RunContext, state: RunStateActor, **kwargs) -> None:
        state.env.render()
        invalid_actions = state.env.invalid_actions

        print("- select action -")
        arr = []
        for action in range(self.action_num):
            if action in invalid_actions:
                continue
            a1 = str(action)
            a2 = state.env.action_to_str(action)
            if a1 == a2:
                arr.append(f"{a1}")
            else:
                arr.append(f"{a1}({a2})")
        print(" ".join(arr))
        for i in range(10):
            try:
                action = int(input("> "))
                if (action not in invalid_actions) and (0 <= action < self.action_num):
                    break
            except Exception:
                pass
            print(f"invalid action({10 - i} times left)")
        else:
            raise ValueError()

        # アクションでpolicyの結果を置き換える
        manual_action = state.env.action_space.decode_from_space_DiscreteSpace(action)
        state.action = manual_action
        state.workers[state.worker_idx].override_action(manual_action)
