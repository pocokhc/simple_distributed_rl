import srl

# --- load env
from srl.envs import grid  # isort: skip # noqa F401


def main():
    env = srl.make_env("Grid")

    # env information
    print(f"action_space     : {env.action_space}")
    print(f"observation_type : {env.observation_type}")
    print(f"observation_space: {env.observation_space}")
    print(f"player_num       : {env.player_num}")

    env.reset(render_mode="terminal")
    env.render()

    while not env.done:
        action = env.sample()
        env.step(action)

        # env status
        print(f"step             : {env.step_num}")
        print(f"next_player_index: {env.next_player_index}")
        print(f"state            : {env.state}")
        print(f"invalid_actions  : {env.get_invalid_actions()}")
        print(f"rewards          : {env.step_rewards}")
        print(f"info             : {env.info}")
        print(f"done             : {env.done}")
        print(f"done_reason      : {env.done_reason}")

        env.render()

    env.close()


if __name__ == "__main__":
    main()
