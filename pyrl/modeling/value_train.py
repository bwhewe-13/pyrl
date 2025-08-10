import numpy as np


class ValueBaseClass:
    """Base Class for Value Based RL Algorithms. This allows for easy training
    and testing of Value Based RL algorithms like Q-Learning, Double Q-learning,
    SARSA, and Expected SARSA.
    """

    def __init__(self, env, **kwargs):
        """Initialization of the Value Based RL Algorithms Base Class.

        Args:
            env: Gymnasium environment.

        Kwargs:
            gamma (float, default = 0.99): Discount factor for future rewards.
                Value between 0 and 1 where 0 favors immediate rewards and 1
                favors future rewards.
            alpha (float, default = 0.1): Learning rate or the value that controls
                how much the Q Table will be upated. Value between 0 and 1 where
                values close to 1 value recent experiences.
            epsilon (float, default = 0.9): Exploration vs exploitation rate.
                Value between 0 and 1 where values close to 1 favor exploration
                and values close to 0 favor exploitation.
            epsilon_decay (float, default = 0.999): Rate at which the epsilon
                value changes through training. To keep epsilon from changing,
                set epsilon_decay = 1.
            epsilon_min (float, default = 0.01): If using epsilon decay, the
                minimum value in which epsilon can go.
        """
        self.env = env
        self.num_states = self.env.observation_space.n
        self.num_actions = self.env.action_space.n
        self.q_table = np.zeros((self.num_states, self.num_actions))
        # Kwargs
        self.gamma = kwargs.get("gamma", 0.99)
        self.alpha = kwargs.get("alpha", 0.1)
        self.epsilon = kwargs.get("epsilon", 0.9)
        self.epsilon_decay = kwargs.get("epsilon_decay", 0.999)
        self.epsilon_min = kwargs.get("epsilon_min", 0.01)

    def epsilon_greedy(self, state):
        """Function for either exploring or exploiting the current policy. If
        a random number is less than epsilon, a random action is taken (action
        space is being explored), else the best policy is exploited.

        Args:
            state:

        Returns:
            action, whether random or current policy.
        """
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state, :])

    def _update_q_table(self, state, action, reward, next_state, next_action):
        """Placeholder to be overwritten by child classes."""
        pass

    def run_episode(self, action_fn, update_q=True):
        """Function to run an episode, either for training or testing.

        Args:
            action_fn (function): The action argument takes the state as the only
                argument. This allows for using the same process for training
                and testing the models. Either epsilon_greedy(state) for training
                or use_policy(state) for testing.
            update_q (bool, default = True): Whether to update the Q-Table.
                update_q = False when testing.

        Returns:
            episode_reward (float): Total reward for the current episode
            episode_steps (int): Number of steps taken for the current epsiode

        """
        # Initialize parameters for the episode
        done = False
        episode_reward = 0
        episode_steps = 0

        # Reset the state and select an action
        state, _ = self.env.reset()
        action = action_fn(state)

        while not done:
            # Collect next state, reward
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            next_action = action_fn(next_state)

            # Update Q-table
            if update_q:
                self._update_q_table(state, action, reward, next_state, next_action)

            # Record rewards and steps
            episode_reward += reward
            episode_steps += 1

            # Update state and action
            state = next_state
            action = next_action

        return episode_reward, episode_steps

    def get_policy(self, q_table=None):
        """Retrieves the optimal policy to use with testing a model.

        Args:
            q_table (np.array, default=None): Current Q Table to use. If None,
                the currently trained q_table is used for selecting the optimal
                actions. If not None, a pretrained Q Table can be used with the
                testing loop.

        Returns:
            None

        """
        if q_table is None:
            self.policy = {
                state: np.argmax(self.q_table[state])
                for state in range(self.num_states)
            }
        else:
            self.policy = {
                state: np.argmax(q_table[state]) for state in range(self.num_states)
            }

    def get_q_table(self):
        """Retrieves the current Q Table. Used with Double Q-Learning for
        averaging out the two tables.

        Returns:
            q_table (2D np.array): Numpy array for the state-action pairs
        """
        return self.q_table

    def use_policy(self, state):
        """Retrieves the current policy given a specific state.

        Args:
            state:

        Returns:
            Current policy for the given state.

        """
        return self.policy[state]

    def train(self, num_episodes):
        """Allows for training the model over a given number of episodes.

        Args:
            num_episodes (int): Total number of episodes to train on.

        Returns:
            total_rewards (list): Rewards for each episode.
                len(total_rewards) = num_episodes.
            total_steps (list): Steps taken for each episode.
                len(total_rewards) = num_episodes.
        """

        # Initialize the rewards and steps histories
        total_rewards = []
        total_steps = []

        # Iterate over the number of episodes
        for _ in range(num_episodes):

            # Run a single episode
            reward, steps = self.run_episode(self.epsilon_greedy, update_q=True)

            # Update the reward and step histories
            total_rewards.append(reward)
            total_steps.append(steps)

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return total_rewards, total_steps

    def test(self, num_episodes=1, q_table=None):
        """Testing a RL model. The Q Table can be trained or loaded.

        Args:
            num_episodes (int, default = 1): Total number of episodes to test on.
            q_table (2D np.array, default = None): Numpy array for the
                state-action pairs. If None, the trained Q Table is used.

        Returns:
            total_rewards (list): Rewards for each episode.
                len(total_rewards) = num_episodes.
            total_steps (list): Steps taken for each episode.
                len(total_rewards) = num_episodes.
        """
        # Get the policy for the given Q Table
        self.get_policy(q_table)

        # Initialize the rewards and steps histories
        total_rewards = []
        total_steps = []

        # Iterate over the number of episodes
        for _ in range(num_episodes):

            # Run a single episode
            reward, steps = self.run_episode(self.use_policy, update_q=False)

            # Update the reward and step histories
            total_rewards.append(reward)
            total_steps.append(steps)

        return total_rewards, total_steps


class QLearning(ValueBaseClass):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def _update_q_table(self, state, action, reward, next_state, next_action):
        old_value = self.q_table[state, action]
        expected_q = np.max(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * expected_q
        )


class DoubleQLearning(ValueBaseClass):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)
        self.q_table = np.zeros((2, self.num_states, self.num_actions))

    def _update_q_table(self, state, action, reward, next_state, next_action):
        table = np.random.randint(2)

        current_table = self.q_table[table, state, action]

        best_next_action = np.argmax(self.q_table[table, next_state])
        other_table = self.q_table[1 - table, next_state, best_next_action]

        self.q_table[table, state, action] = (
            1 - self.alpha
        ) * current_table + self.alpha * (reward + self.gamma * other_table)

    def get_q_table(self):
        return 0.5 * (self.q_table[0] + self.q_table[1])


class SARSA(ValueBaseClass):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def _update_q_table(self, state, action, reward, next_state, next_action):
        old_value = self.q_table[state, action]
        next_value = self.q_table[next_state, next_action]
        self.q_table[state, action] = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * next_value
        )


class ExpectedSARSA(ValueBaseClass):
    def __init__(self, env, **kwargs):
        super().__init__(env, **kwargs)

    def _update_q_table(self, state, action, reward, next_state, next_action):
        old_value = self.q_table[state, action]
        expected_q = np.mean(self.q_table[next_state])
        self.q_table[state, action] = (1 - self.alpha) * old_value + self.alpha * (
            reward + self.gamma * expected_q
        )
