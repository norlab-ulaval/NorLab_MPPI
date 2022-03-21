#!/usr/bin/env python3
import numpy as np

def verify_gym_classic_control_install() -> None:
    """
    Minimal OpenAi gym classic_control install verification.
    :return: None
    """

    print("\nDS | Start Gym classic control install check")

    try:
        import gym

        print(f'DS | Gym version:    {gym.__version__}')
        # env: gym.wrappers.time_limit.TimeLimit = gym.make('Pendulum-v1')
        env: gym.wrappers.time_limit.TimeLimit = gym.make('CartPole-v1')
        env.reset()
        output = env.render("rgb_array")
        action = env.action_space.sample()
        next_obs = env.step(action=action)
        assert isinstance(next_obs[0], np.ndarray)
        env.close()
        print("\nDS | Gym classic control install is good to go!\n")
    except Exception as e:
        # Note: The exception scope is large on purpose

        # (PRIORITY) ToDo:validate >> next bloc ↓↓
        raise Exception(
            f"DS | (!) Something is wrong with Gym classic control. It's probably the Dockerized-SNOW python interpreter\n"
            f"{e}"
            ) from e

    return None

def verify_gym_box2d_install() -> None:
    """
    Minimal OpenAi gym classic_control install verification.
    :return: None
    """

    print("\nDS | Start Gym Box2D install check")

    try:
        import gym

        print(f'DS | Gym version:    {gym.__version__}')
        env: gym.wrappers.time_limit.TimeLimit = gym.make('CarRacing-v0')
        env.reset()
        output = env.render("rgb_array")
        action = env.action_space.sample()
        next_obs = env.step(action=action)
        assert isinstance(next_obs[0], np.ndarray)
        env.close()
        print("\nDS | Gym Box2D install is good to go!\n")
    except Exception as e:
        # Note: The exception scope is large on purpose

        # (Priority) ToDo:validate >> next bloc ↓↓
        raise Exception(
            f"DS | (!) Something is wrong with Gym Box2D. It's probably the Dockerized-SNOW python interpreter\n"
            f"{e}"
            ) from e

    return None
