from distrib_rl.experience import Timestep, Trajectory
from distrib_rl.policies import policy_factory
from distrib_rl.marl import OpponentSelector
import numpy as np
import torch
import time


class MARLAgent(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.leftover_obs = None
        self.opponent_policy = None
        self.opponent_num = -1
        self.opponent_selector = OpponentSelector(cfg)
        self.save_both_teams = True
        self.ep_rewards = []
        self.current_ep_rew = 0
        self.policies = None

    @torch.no_grad()
    def gather_timesteps(
        self,
        policy,
        policy_epoch,
        env,
        num_timesteps=None,
        num_seconds=None,
        num_eps=None,
    ):
        if self.opponent_policy is None:
            self.init_opponent_policy(env)

        obs = self.leftover_obs
        if obs is None:
            obs = env.reset()

        n_agents = np.shape(obs)[0]
        agents_to_save = n_agents if self.save_both_teams else n_agents // 2

        self.policies = [policy for _ in range(n_agents // 2)] + [
            self.opponent_policy for _ in range(n_agents // 2)
        ]
        policies = self.policies
        experience_trajectories = 0

        trajectories = [Trajectory(policy_epoch) for _ in range(n_agents)]

        cumulative_timesteps = 0
        start_time = time.time()
        while True:
            actions = []
            ts = [Timestep() for _ in range(n_agents)]
            for i in range(n_agents):
                action, log_prob = policies[i].get_action(obs[i], deterministic=False)
                ts[i].action = action
                ts[i].log_prob = log_prob
                actions.append(action)

            next_obs, rews, terminated, truncated, _ = env.step(np.asarray(actions))

            done = terminated or truncated

            for i in range(n_agents):
                if self.save_both_teams:
                    self.current_ep_rew += rews[i]
                elif i < n_agents // 2:
                    self.current_ep_rew += rews[i]

                ts[i].reward = rews[i]
                ts[i].obs = obs[i].copy()
                ts[i].done = 1 if done else 0
                trajectories[i].register_timestep(ts[i])

            cumulative_timesteps += 1
            if done:
                self.ep_rewards.append(self.current_ep_rew / agents_to_save)
                self.current_ep_rew = 0

                for i in range(agents_to_save):
                    trajectories[i].final_obs = next_obs[i]
                    experience_trajectories += 1
                    yield trajectories[i]

                # todo: Implement a proper opponent evaluation & selection scheme and delete this.
                result = sum(trajectories[0].rewards) > sum(trajectories[-1].rewards)
                self.opponent_selector.submit_result(self.opponent_num, result)
                self.get_next_opponent(policy)

                next_obs = env.reset()

                n_agents = np.shape(next_obs)[0]
                agents_to_save = n_agents if self.save_both_teams else n_agents // 2

                self.policies = [policy for _ in range(n_agents // 2)] + [
                    self.opponent_policy for _ in range(n_agents // 2)
                ]
                policies = self.policies

                trajectories = [Trajectory(policy_epoch) for _ in range(n_agents)]

            obs = next_obs
            if (
                (num_timesteps is not None and cumulative_timesteps >= num_timesteps)
                or (num_seconds is not None and time.time() - start_time >= num_seconds)
                or (num_eps is not None and experience_trajectories >= num_eps)
            ):
                break

        self.leftover_obs = next_obs.copy()

        for i in range(agents_to_save):
            trajectories[i].final_obs = next_obs[i]
            experience_trajectories += 1
            yield trajectories[i]

    def get_next_opponent(self, policy):
        opponent_weights, opponent_num = self.opponent_selector.get_opponent()
        if type(opponent_weights) not in (tuple, list, np.ndarray, np.array):
            self.opponent_policy.set_trainable_flat(policy.get_trainable_flat().copy())
        else:
            self.opponent_policy.set_trainable_flat(opponent_weights.copy())
        self.opponent_num = opponent_num
        self.save_both_teams = opponent_num == -1

    def init_opponent_policy(self, env):
        models = policy_factory.get_from_cfg(self.cfg, env)
        self.opponent_policy = models["policy"]
        self.opponent_policy.to(self.cfg["device"])
        models.clear()

    def _get_policy_action(self, policy, obs, timestep, evaluate=False):
        raise NotImplementedError
