import srl

# --- load env
from srl.envs import grid, ox  # isort: skip # noqa F401


def main(env_name):
    env = srl.make_env(env_name)

    print(f"action_space     : {env.action_space}")
    print(f"observation_space: {env.observation_space}")
    print(f"player_num       : {env.player_num}")

    env.set_render_options(interval=1000 / 10)
    env.reset(render_mode="window")  # "terminal" or "rgb_array" or "window"
    env.render()

    while not env.done:
        action = env.sample_action()
        env.step(action)

        print(f"action           : {action}")
        print(f"step             : {env.step_num}")
        print(f"next_player_index: {env.next_player_index}")
        print(f"state            : {env.state}")
        print(f"invalid_actions  : {env.get_invalid_actions()}")
        print(f"reward [rewards] : {env.reward} {env.step_rewards}")
        print(f"info             : {env.info}")
        print(f"done             : {env.done}")
        print(f"done_reason      : {env.done_reason}")

        env.render()

    env.close()


if __name__ == "__main__":
    main("Grid")
    print("=" * 40)
    main("OX")
