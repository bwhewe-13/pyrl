import gymnasium as gym
import numpy as np
import pettingzoo
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam


class TensorboardCallback(BaseCallback):
    # https://stable-baselines3.readthedocs.io/en/master/common/logger.html

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []

    def _on_training_start(self):
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning_rate": self.model.learning_rate,
            "n_steps": self.model.n_steps,
            "batch_size": self.model.batch_size,
            "n_epochs": self.model.n_epochs,
            "gamma": self.model.gamma,
            "gae_lambda": self.model.gae_lambda,
            "clip_range": self.model.clip_range(0),
            # "clip_range_vf": self.model.clip_range_vf
            "ent_coef": self.model.ent_coef,
            "vf_coef": self.model.vf_coef,
            "max_grad_norm": self.model.max_grad_norm,
            "target_kl": self.model.target_kl,
            "seed": self.model.seed,
        }

        metric_dict = {
            "rollout/ep_len_mean": 0.0,
            "rollout/ep_rew_mean": 0.0,
            "train/value_loss": 0.0,
            "train/loss": 0.0,
            "train/approx_kl": 0.0,
            "train/entropy_loss": 0.0,
            "train/explained_variance": 0.0,
        }

        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self):
        for info in self.locals["infos"]:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
        return True

    def _on_rollout_end(self):
        if self.episode_rewards:
            self.logger.record("max_ep_rew", np.max(self.episode_rewards))
            self.logger.record("mean_ep_rew", np.mean(self.episode_rewards))
            self.logger.record("std_ep_rew", np.std(self.episode_rewards))
            self.episode_rewards = []


def rl_action_masker(env):
    env.reset()

    def mask_fn(env):
        return env.get_wrapper_attr("get_action_masks")()

    return ActionMasker(env, mask_fn)


def marl_action_masker(env):
    env = MARLActionMaskWrapper(env)
    env.reset()

    def mask_fn(env):
        return env.action_mask()

    return ActionMasker(env, mask_fn)


class MARLActionMaskWrapper(pettingzoo.utils.BaseWrapper, gym.Env):

    def reset(self, seed=None, options=None):
        super().reset(seed, options)

        self.observation_space = super().observation_space(self.possible_agents[0])[
            "observation"
        ]
        self.action_space = super().action_space(self.possible_agents[0])

        return self.observe(self.agent_selection), {}

    def step(self, action):
        current_agent = self.agent_selection

        super().step(action)

        next_agent = self.agent_selection
        return (
            self.observe(next_agent),
            self._cumulative_rewards[current_agent],
            self.terminations[current_agent],
            self.truncations[current_agent],
            self.infos[current_agent],
        )

    def observe(self, agent):
        return super().observe(agent)["observation"]

    def action_mask(self):
        return super().observe(self.agent_selection)["action_mask"]
