import numpy as np
import optuna
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


class TrialEvalCallBack(EvalCallback):

    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=5000, verbose=0):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune if necessary
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class MaskableTrialEvalCallBack(MaskableEvalCallback):

    def __init__(self, eval_env, trial, n_eval_episodes=250, eval_freq=5000, verbose=0):
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self):
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            super()._on_step()
            self.eval_idx += 1
            self.trial.report(self.last_mean_reward, self.eval_idx)
            self.trial.report(np.mean(self._is_success_buffer), self.eval_idx)
            # Prune if necessary
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


class ProximalPolicyOptimization:

    def __init__(self, env, metric, **kwargs):
        self.env = env
        self.metric = metric
        self.marl = kwargs.get("marl", False)
        self._create_search_parameters(**kwargs)

    def _create_search_parameters(self, **kwargs):
        self.seed = kwargs.get("seed", None)
        self.tensorboard_log = kwargs.get("tensorboard_log", None)

        self.hpo_params = {
            "total_timesteps": kwargs.get("total_timesteps", [500_000]),
            "learning_rate": kwargs.get("learning_rate", [1e-6, 0.1]),
            "n_steps": kwargs.get("n_steps", [2048]),
            "batch_size": kwargs.get("batch_size", [64]),
            "n_epochs": kwargs.get("n_epochs", [5, 20]),
            "gamma": kwargs.get("gamma", [0.9, 0.99]),
            "gae_lambda": kwargs.get("gae_lambda", [0.8, 0.9999]),
            "clip_range": kwargs.get("clip_range", [0.0, 0.5]),
            "clip_range_vf": kwargs.get("clip_range_vf", [1e-8, 0.5]),
            "normalize_advantage": kwargs.get("normalize_advantage", [True]),
            "ent_coef": kwargs.get("ent_coef", [0.0, 0.5]),
            "vf_coef": kwargs.get("vf_coef", [0.0, 1.0]),
            "max_grad_norm": kwargs.get("max_grad_norm", [0.0, 1.0]),
            "target_kl": kwargs.get("target_kl", [None]),
        }

    def _sample_hyperparameters(self, trial):
        self.total_timesteps = trial.suggest_categorical(
            "total_timesteps", self.hpo_params["total_timesteps"]
        )

        # Create kwargs dict
        self.ppo_args = {
            "verbose": 0,
            "seed": self.seed,
            "tensorboard_log": self.tensorboard_log,
        }
        self.ppo_args["learning_rate"] = trial.suggest_float(
            "learning_rate",
            self.hpo_params["learning_rate"][0],
            self.hpo_params["learning_rate"][1],
            log=True,
        )
        self.ppo_args["n_steps"] = trial.suggest_categorical(
            "n_steps", self.hpo_params["n_steps"]
        )
        self.ppo_args["batch_size"] = trial.suggest_categorical(
            "batch_size", self.hpo_params["batch_size"]
        )
        self.ppo_args["n_epochs"] = trial.suggest_int(
            "n_epochs", self.hpo_params["n_epochs"][0], self.hpo_params["n_epochs"][1]
        )
        self.ppo_args["gamma"] = trial.suggest_float(
            "gamma",
            self.hpo_params["gamma"][0],
            self.hpo_params["gamma"][1],
        )
        self.ppo_args["gae_lambda"] = trial.suggest_float(
            "gae_lambda",
            self.hpo_params["gae_lambda"][0],
            self.hpo_params["gae_lambda"][1],
        )
        self.ppo_args["clip_range"] = trial.suggest_float(
            "clip_range",
            self.hpo_params["clip_range"][0],
            self.hpo_params["clip_range"][1],
        )
        self.ppo_args["clip_range_vf"] = trial.suggest_float(
            "clip_range_vf",
            self.hpo_params["clip_range_vf"][0],
            self.hpo_params["clip_range_vf"][1],
        )
        self.ppo_args["normalize_advantage"] = trial.suggest_categorical(
            "normalize_advantage", self.hpo_params["normalize_advantage"]
        )
        self.ppo_args["ent_coef"] = trial.suggest_float(
            "ent_coef",
            self.hpo_params["ent_coef"][0],
            self.hpo_params["ent_coef"][1],
        )
        self.ppo_args["vf_coef"] = trial.suggest_float(
            "vf_coef",
            self.hpo_params["vf_coef"][0],
            self.hpo_params["vf_coef"][1],
        )
        self.ppo_args["max_grad_norm"] = trial.suggest_float(
            "max_grad_norm",
            self.hpo_params["max_grad_norm"][0],
            self.hpo_params["max_grad_norm"][1],
        )
        self.ppo_args["target_kl"] = trial.suggest_categorical(
            "target_kl", self.hpo_params["target_kl"]
        )[0]

    def objective(self, trial):
        self._sample_hyperparameters(trial)

        model = MaskablePPO("MlpPolicy", self.env, **self.ppo_args)

        # Create evaluation environment
        eval_env = Monitor(self.env, info_keywords=("is_success",))
        eval_callback = MaskableTrialEvalCallBack(eval_env, trial)

        model.learn(
            total_timesteps=self.total_timesteps,
            callback=eval_callback,
            progress_bar=True,
        )

        metrics = {
            "reward": eval_callback.last_mean_reward,
            "explained_variance": explained_variance(model),
            "success_rate": np.mean(eval_callback._is_success_buffer),
        }

        if eval_callback.is_pruned:
            raise optuna.exceptions.TrialPruned()

        return metrics[self.metric]

    def run(self, **kwargs):
        n_trials = kwargs.get("n_trials", 100)
        study_name = kwargs.get("study_name", None)
        storage = kwargs.get("storage", None)
        output = kwargs.get("output", "hyperparameters_search")

        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage,
            load_if_exists=True,
        )
        study.optimize(self.objective, n_trials=n_trials)
        study.trials_dataframe().to_csv(f"{output}.csv")
        print(f"Study Saved to: {output}.csv")
        return study.best_params


# Based on function from Stable Baselines3
def explained_variance(model):
    y_pred = model.rolllout_buffer.values.flatten()
    y_true = model.rolllout_buffer.returns.flatten()
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
