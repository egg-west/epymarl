from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
import numpy as np
from copy import deepcopy


class MTRLRunner:
    """Training in a set of environments"""
    def __init__(self, args, logger, wandb_logger=None, env_id=None):
        self.args = args
        self.logger = logger
        self.wandb_logger = wandb_logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env_id = env_id
        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.interpolate_test_envs = None
        self.extrapolate_test_envs = None
        self.episode_limit = self.env.episode_limit
        self.n_agents = self.get_env_info()["n_agents"]
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, task_embedding=None, agent_embedding=None):
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)

        while not terminated:

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()],
                "task_indices": [[self.env_id for _ in range(self.n_agents)]],
                "task_indices_global": [[self.env_id]],
            }
            #print(f'{len(pre_transition_data["obs"][0])=}')# 5
            #print(f'{pre_transition_data["obs"][0][0].shape=}')# (80,)

            self.batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            if self.args.independent_task_encoder:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, task_embedding=task_embedding, agent_embedding=agent_embedding)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            if test_mode and self.args.render:
                self.env.render()
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)

            self.t += 1

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()],
            "task_indices": [[self.env_id for _ in range(self.n_agents)]],
            "task_indices_global": [[self.env_id]],
        }
        if test_mode and self.args.render:
            print(f"Episode return: {episode_return}")
        self.batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        if self.args.independent_task_encoder:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode, task_embedding=task_embedding, agent_embedding=agent_embedding)         
        else:
            actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)

        timely_test_stats = deepcopy(env_info)
        timely_test_stats["returns"] = episode_return
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        cur_stats.update({k: cur_stats.get(k, 0) + env_info.get(k, 0) for k in set(cur_stats) | set(env_info)})
        cur_stats["n_episodes"] = 1 + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = self.t + cur_stats.get("ep_length", 0)

        if not test_mode:
            self.t_env += self.t

        cur_returns.append(episode_return)
        cur_stats["epsilon"] = self.mac.action_selector.epsilon
        cur_stats["return_mean"] = np.mean(cur_returns)
        cur_stats["return_std"] = np.std(cur_returns)
        #stats_to_return = deepcopy(cur_stats)
        #stats_to_return["returns"] = cur_returns

        # if test_mode and (len(self.test_returns) == self.args.test_nepisode):
        #     self._log(cur_returns, cur_stats, log_prefix)
        # elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
        #     self._log(cur_returns, cur_stats, log_prefix)
        #     if hasattr(self.mac.action_selector, "epsilon"):
        #         self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
        #         self.wandb_logger.log({"train/epsilon":self.mac.action_selector.epsilon}, self.t_env)
        #     self.log_train_stats_t = self.t_env

        #return self.batch, stats_to_return
        return self.batch, timely_test_stats

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        self.wandb_logger.log({"eval/" + prefix + "return_mean":np.mean(returns)}, self.t_env)
        self.wandb_logger.log({"eval/" + prefix + "return_std":np.std(returns)}, self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat("eval/" + prefix + k + "_mean", v/stats["n_episodes"], self.t_env)
                self.wandb_logger.log({"eval/" + prefix + k + "_mean":v/stats["n_episodes"]}, self.t_env)
        stats.clear()
