# To run model inference with trained models
from warnings import filterwarnings

import numpy as np

filterwarnings(action="ignore", category=UserWarning)


def rl_model(env, model, iterations, **kwargs):
    max_steps = kwargs.get("max_steps", 100)
    masking = kwargs.get("masking", True)
    LOUD = kwargs.get("LOUD", True)

    def _single_iteration():

        state, _ = env.reset()
        initial_state = state.copy()

        states = []
        rewards = []
        actions = []

        step = 0
        while True:
            step += 1

            if masking:
                action, _ = model.predict(state, action_masks=env.action_masks())
            else:
                action, _ = model.predict(state)

            state, reward, terminated, truncated, _ = env.step(action)

            states.append(state)
            rewards.append(reward)
            actions.append(int(action))

            done = terminated or truncated or step >= max_steps
            if done:
                break

        states = np.array(states)
        rewards = np.array(rewards)
        actions = np.array(actions)
        return (
            initial_state,
            states,
            rewards,
            actions,
        )

    best_initial_state = []
    best_states = []
    best_actions = []

    best_rewards = np.array([-10])

    for _ in range(iterations):
        initial_state, states, rewards, actions = _single_iteration()

        if np.max(rewards) > np.max(best_rewards):
            best_initial_state = initial_state.copy()
            best_states = states.copy()
            best_rewards = rewards.copy()
            best_actions = actions.copy()

    if LOUD:
        rewards = best_rewards.copy()
        print(
            f"Final Number of Groups: {np.sum(best_states[-1]) - 1}\n"
            f"Max Reward: {np.max(rewards):2.5f}, "
            f"Groups: {np.sum(best_states[np.argmax(rewards)]) - 1}"
        )

    return best_initial_state, best_states, best_rewards, best_actions


def marl_model(env, model, iterations, **kwargs):
    max_steps = kwargs.get("max_steps", 100)
    seed = kwargs.get("seed", None)

    def _single_iteration():
        env.reset()
        env.action_space(env.possible_agents[0]).seed(seed)
        obs, _, _, _, _ = env.last()
        state, action_mask = obs.values()
        initial_state = state

        states = []
        rewards = []
        actions = {agent: [] for agent in env.possible_agents}

        step = 0
        for agent in env.agent_iter():
            action = int(model.predict(state, action_masks=action_mask)[0])
            env.step(action)

            obs, reward, terminated, truncated, _ = env.last()
            state, action_mask = obs.values()

            actions[agent].append(action)

            if agent == env.agents[-1]:
                step += 1
                states.append(state)
                rewards.append(reward)

            if terminated or truncated or step >= max_steps:
                break

        states = np.array(states)
        return (
            initial_state,
            states,
            rewards,
            actions,
        )

    best_initial_state = []
    best_states = []
    best_actions = []
    best_rewards = [-10]

    for _ in range(iterations):
        initial_state, states, rewards, actions = _single_iteration()

        if np.max(rewards) > np.max(best_rewards):
            best_initial_state = initial_state.copy()
            best_states = states.copy()
            best_rewards = rewards.copy()
            best_actions = actions.copy()

    return best_initial_state, best_states, best_rewards, best_actions
