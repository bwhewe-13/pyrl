import gymnasium as gym
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector


class PrisonGuard01(AECEnv):

    metadata = {
        "render_modes": [],
        "name": "PrisonGuard01_v0",
    }

    def __init__(self, board_size=7, max_steps=100):
        super().__init__()
        self.board_size = board_size
        self.max_steps = max_steps
        self.agents = ["prisoner", "guard"]
        self.possible_agents = self.agents[:]

        self.observation_spaces = {
            a: gym.spaces.Dict(
                {
                    "observation": gym.spaces.MultiDiscrete(
                        [self.board_size * self.board_size] * 3
                    ),
                    "action_mask": gym.spaces.MultiBinary(4),
                }
            )
            for a in self.agents
        }
        # Actions = [Right, Left, Up, Down]
        self.action_spaces = {a: gym.spaces.Discrete(4) for a in self.agents}

    def _flatten_loc(self, loc_x, loc_y):
        return loc_x + self.board_size * loc_y

    def _mask_actions(self, loc_x, loc_y, agent):
        # 0 - invalid action, 1 - valid action
        action_mask = [
            loc_x != 0,
            loc_x != self.board_size - 1,
            loc_y != 0,
            loc_y != self.board_size - 1,
        ]
        if agent == "prisoner":
            return np.array(action_mask, dtype=np.int8)

        guard_action_mask = [
            loc_x - 1 != self.escape_x,
            loc_x + 1 != self.escape_x,
            loc_y - 1 != self.escape_y,
            loc_y + 1 != self.escape_y,
        ]
        # Combine edge action masking with escape action masking
        action_mask = np.array(
            [min(ii, jj) for ii, jj in zip(action_mask, guard_action_mask)],
            dtype=np.int8,
        )
        return action_mask

    def reset(self, seed=None, options=None):
        self.step_count = 0

        self.agents = self.possible_agents[:]
        self.rewards = {a: 0 for a in self.agents}
        self._cumulative_rewards = {a: 0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations = {a: False for a in self.agents}
        self.infos = {a: {} for a in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.prisoner_x = 0
        self.prisoner_y = 0

        self.guard_x = self.board_size - 1
        self.guard_y = self.board_size - 1

        self.escape_x = np.random.randint(2, self.board_size - 2)
        self.escape_y = np.random.randint(2, self.board_size - 2)

        self.observations = (
            self._flatten_loc(self.prisoner_x, self.prisoner_y),
            self._flatten_loc(self.guard_x, self.guard_y),
            self._flatten_loc(self.escape_x, self.escape_y),
        )

    def observe(self, agent):
        self.observations = (
            self._flatten_loc(self.prisoner_x, self.prisoner_y),
            self._flatten_loc(self.guard_x, self.guard_y),
            self._flatten_loc(self.escape_x, self.escape_y),
        )

        if agent == "prisoner":
            action_mask = self._mask_actions(self.prisoner_x, self.prisoner_y, agent)
        elif agent == "guard":
            action_mask = self._mask_actions(self.guard_x, self.guard_y, agent)

        return {
            "observation": self.observations,
            "action_mask": action_mask.copy(),
        }

    def _move_agent(self, action):
        agent = self.agent_selection
        if agent == "prisoner":
            if action == 0 and self.prisoner_x > 0:
                self.prisoner_x -= 1
            elif action == 1 and self.prisoner_x < self.board_size - 1:
                self.prisoner_x += 1
            elif action == 2 and self.prisoner_y > 0:
                self.prisoner_y -= 1
            elif action == 3 and self.prisoner_y < self.board_size - 1:
                self.prisoner_y += 1
        elif agent == "guard":
            if action == 0 and self.guard_x > 0:
                self.guard_x -= 1
            elif action == 1 and self.guard_x < self.board_size - 1:
                self.guard_x += 1
            elif action == 2 and self.guard_y > 0:
                self.guard_y -= 1
            elif action == 3 and self.guard_y < self.board_size - 1:
                self.guard_y += 1

    def step(self, action):
        # Update agent location
        self._move_agent(action)

        # Prisoner caught
        if (self.prisoner_x, self.prisoner_y) == (self.guard_x, self.guard_y):
            self.rewards = {"prisoner": -1, "guard": 1}
            self.terminations = {a: True for a in self.agents}
            self.infos = {
                "prisoner": {"is_success": True},
                "guard": {"is_success": False},
            }
        # Prisoner got away
        elif (self.prisoner_x, self.prisoner_y) == (self.escape_x, self.escape_y):
            self.rewards = {"prisoner": 1, "guard": -1}
            self.terminations = {a: True for a in self.agents}
            self.infos = {
                "prisoner": {"is_success": False},
                "guard": {"is_success": True},
            }

        if self.step_count > self.max_steps:
            self.rewards = {"prisoner": 0, "guard": 0}
            self.truncations = {"prisoner": True, "guard": True}

        # Move to next agent
        next_agent = self._agent_selector.next()

        if self.agent_selection == self.agents[-1]:
            self.step_count += 1

        self.agent_selection = next_agent
        self._accumulate_rewards()

    def render(self):
        grid = np.full((self.board_size, self.board_size), " ")
        grid[self.prisoner_y, self.prisoner_x] = "P"
        grid[self.guard_y, self.guard_x] = "G"
        grid[self.escape_y, self.escape_x] = "E"
        print(f"{grid} \n")

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]


if __name__ == "__main__":
    from pyrl.modeling import ppo_train

    env = PrisonGuard01()

    # env = ppo_train.marl_action_masker(env)
    env.reset()
    env.action_space(env.possible_agents[0])
    obs, _, _, _, _ = env.last()
    state, action_mask = obs.values()
    initial_state = state
    print(initial_state)
    max_steps = 10

    states = []
    rewards = []
    actions = {agent: [] for agent in env.possible_agents}

    step = 0
    for agent in env.agent_iter():
        action = int(env.action_spaces[agent].sample())
        env.step(action)

        obs, reward, terminated, truncated, _ = env.last()
        state, action_mask = obs.values()
        print(agent, action, state)

        actions[agent].append(action)

        if agent == env.agents[-1]:
            step += 1
            states.append(state)
            rewards.append(reward)

        if terminated or truncated or step >= max_steps:
            break

    states = np.array(states)
