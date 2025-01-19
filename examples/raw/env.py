import srl

# --- load env
from srl.envs import grid, ox  # isort: skip # noqa F401


def main(env_name: str, render_interval: float = 1000 / 5):
    env = srl.make_env(env_name)
    env.set_render_options(interval=render_interval)

    print(f"action_space     : {env.action_space}")
    print(f"observation_space: {env.observation_space}")
    print(f"player_num       : {env.player_num}")

    env.setup(render_mode="window")  # "terminal" or "rgb_array" or "window"
    env.reset()
    env.render()

    while not env.done:
        action = env.sample_action()
        env.step(action)

        print(f"action          : {action}")
        print(f"step            : {env.step_num}")
        print(f"next_player     : {env.next_player}")
        print(f"state           : {env.state}")
        print(f"invalid_actions : {env.get_invalid_actions()}")
        print(f"reward [rewards]: {env.reward} {env.rewards}")
        print(f"info            : {env.info}")
        print(f"done            : {env.done} ({env.done_type})")
        print(f"done_reason     : {env.done_reason}")

        env.render()

    env.close()


if __name__ == "__main__":
    main("Grid")
    print("=" * 40)
    main("OX")
