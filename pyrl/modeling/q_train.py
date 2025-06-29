import gymnasium as gym
import numpy as np


class QTableBaseClass:
    def __init__(self, env):
        self.env = env
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.q_table = np.zeros((self.num_states, self.num_actions))

    def _generate_episode(self, action_fn, update_q=True, gamma=0.99, alpha=0.1):
        state, _ = self.env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        while not done:
            # Select the best action based on learned Q-table
            action = action_fn(state)

            # Collect next state, reward
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated

            # Update Q-table
            if update_q:
                self._update_q_table(state, action, reward, next_state, gamma, alpha)

            episode_reward += reward
            episode_length += 1
            state = next_state

        return episode_reward, episode_length

    def _update_q_table(self, state, action, reward, next_state, gamma, alpha):
        old_value = self.q_table[state, action]
        expected_q = np.mean(self.q_table[next_state])
        self.q_table[state, action] = (1 - alpha) * old_value + alpha * (
            reward + gamma * expected_q
        )

    def _get_policy(self, q_table=None):
        if q_table is None:
            self.policy = {
                state: np.argmax(self.q_table[state])
                for state in range(self.num_states)
            }
        else:
            self.policy = {
                state: np.argmax(q_table[state]) for state in range(self.num_states)
            }
        # print(self.num_states)
        # self.policy = {
        #     state: np.argmax(self.q_table[state]) for state in range(self.num_states)
        # }

    def _use_policy(self, state):
        return self.policy[state]

    def _sample_actions(self, state):
        return self.env.action_space.sample()

    def train(self, episodes, gamma, alpha):
        episode_rewards = []
        episode_lengths = []

        for episode in range(episodes):

            reward, length = self._generate_episode(
                self._sample_actions, update_q=True, gamma=gamma, alpha=alpha
            )

            episode_rewards.append(reward)
            episode_lengths.append(length)

        return episode_rewards, episode_lengths

    def test(self, num_episodes, q_table=None):
        self._get_policy(q_table)
        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            reward, length = self._generate_episode(self._use_policy, False)

            episode_rewards.append(reward)
            episode_lengths.append(length)

        return episode_rewards, episode_lengths

    def get_q_table(self):
        return self.q_table


class QLearning(QTableBaseClass):
    def __init__(self, env):
        super().__init__(env)

    def _update_q_table(self, state, action, reward, next_state, gamma, alpha):
        old_value = self.q_table[state, action]
        expected_q = np.max(self.q_table[next_state])
        self.q_table[state, action] = (1 - alpha) * old_value + alpha * (
            reward + gamma * expected_q
        )


class SARSA(QTableBaseClass):
    def __init__(self, env):
        super().__init__(env)

    def _update_q_table(self, state, action, reward, next_state, gamma, alpha):
        old_value = self.q_table[state, action]
        expected_q = np.mean(self.q_table[next_state])
        self.q_table[state, action] = (1 - alpha) * old_value + alpha * (
            reward + gamma * expected_q
        )


class DoubleQLearning(QTableBaseClass):
    def __init__(self, env):
        super().__init__(env)
        self.q_table = np.zeros((2, self.num_states, self.num_actions))

    def _update_q_table(self, state, action, reward, next_state, gamma, alpha):
        table = np.random.randint(2)

        current_table = self.q_table[table, state, action]

        best_next_action = np.argmax(self.q_table[table, next_state])
        other_table = self.q_table[1 - table, next_state, best_next_action]

        self.q_table[table, state, action] = (1 - alpha) * current_table + alpha * (
            reward + gamma * other_table
        )

    def get_q_table(self):
        return 0.5 * (self.q_table[0] + self.q_table[1])
