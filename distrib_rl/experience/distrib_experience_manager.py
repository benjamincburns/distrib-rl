from distrib_rl.experience import ExperienceReplay
import numpy as np
import time

class DistribExperienceManager(object):
    def __init__(self, cfg, client=None, server=None):
        self.cfg = cfg
        self.client = client
        self.server = server
        self.experience = ExperienceReplay(cfg)

    def get_timesteps_as_batches(self, num_timesteps):
        times = [("start", time.perf_counter())]
        if self.server is None:
            return None

        exp = self.experience

        n_collected = 0
        acts = []
        probs = []
        rews = []
        obses = []
        done = []
        frews = []
        vals = []
        advs = []
        rets = []
        noise_indices = []
        ep_rews = []

        trajectories = self.server.get_n_timesteps(num_timesteps)
        times.append(("retrieve timesteps from redis", time.perf_counter()))

        for trajectory in trajectories:
            (
                actions,
                log_probs,
                rewards,
                obs,
                dones,
                future_rewards,
                values,
                advantages,
                pred_rets,
                ep_rew,
                noise_idx,
            ) = trajectory
            acts += actions
            probs += log_probs
            rews += rewards
            obses += obs
            done += dones
            frews += future_rewards
            vals += values
            advs += advantages
            rets += pred_rets
            noise_indices.append(noise_idx)
            ep_rews.append(ep_rew)
            n_collected += len(actions)

        times.append(("concat components", time.perf_counter()))

        if len(acts) > 0:
            exp.register_trajectory(
                (
                    acts,
                    probs,
                    rews,
                    np.asarray(obses),
                    done,
                    frews,
                    vals,
                    advs,
                    rets,
                    noise_indices,
                    ep_rews,
                ),
                serialized=True,
            )
        else:
            return None

        discarded_timesteps = self.server.discarded_timesteps
        self.server.discarded_timesteps = 0
        times.append(("register trajectory", time.perf_counter()))

        prev_time = times[0][1]
        for i, t in enumerate(times):
            if i == 0:
                continue
            print(f"DistribExperienceManager.get_timesteps_as_batches, {t[0]}: {t[1]-prev_time}")
            prev_time = t[1]

        return n_collected, self.server.steps_per_second, discarded_timesteps

    def cleanup(self):
        self.experience.clear()
