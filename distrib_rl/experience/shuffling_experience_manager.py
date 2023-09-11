import time
from typing import Any, Dict
from distrib_rl.distrib.redis_server import RedisServer
from distrib_rl.experience.distrib_experience_manager import DistribExperienceManager

import numpy as np


class ShufflingExperienceManager(object):
    def __init__(self, server: RedisServer, cfg: Dict[str, Any]):
        self.cfg = cfg
        if "rng" not in self.cfg:
            self.cfg["rng"] = np.random.RandomState(self.cfg["seed"])

        self.server = server
        self.rew_mean = 0
        self.rew_std = 1
        self.ts_collected = 0
        self.ts_discarded = 0
        self.steps_per_second = 0

        if "updates_per_timestep" in self.cfg["policy_optimizer"]:
            self.ts_per_update = (
                1 / self.cfg["policy_optimizer"]["updates_per_timestep"]
            )

            if self.ts_per_update > self.cfg["policy_optimizer"]["batch_size"]:
                raise Exception(
                    "1 / updates_per_timestep must be smaller than batch_size"
                )

        elif "timesteps_per_update" in self.cfg["policy_optimizer"]:
            self.ts_per_update = self.cfg["policy_optimizer"]["timesteps_per_update"]

            if self.ts_per_update > self.cfg["policy_optimizer"]["batch_size"]:
                raise Exception("timesteps_per_update must be smaller than batch_size")

        else:
            self.ts_per_update = int(
                round(
                    self.cfg["policy_optimizer"]["new_returns_proportion"]
                    * self.cfg["experience_replay"]["max_buffer_size"]
                )
            )

        self.batch_size = self.cfg["policy_optimizer"]["batch_size"]

        self.exp_manager = DistribExperienceManager(self.cfg, server=self.server)

    def get_all_batches_shuffled(self):
        times = [("start", time.perf_counter())]
        self.ts_collected = 0
        self.ts_discarded = 0

        returns = None

        polling_times = []
        step_availability = []
        while returns is None:
            before = time.perf_counter()
            available_steps = self.exp_manager.server.available_timesteps
            if (
                len(step_availability) == 0
                or available_steps > step_availability[-1][1]
            ):
                step_availability.append((before - times[0][1], available_steps))

            returns = self.exp_manager.get_timesteps_as_batches(self.ts_per_update)
            after = time.perf_counter()
            if returns is None:
                polling_times.append(after - before)

        times.append(("wasted polls", before))
        times.append(("returns retrieved", time.perf_counter()))

        ts_collected, fps, discarded_timesteps = returns

        batches = self.exp_manager.experience.get_all_batches_shuffled(self.batch_size)
        times.append(("shuffled batches", time.perf_counter()))

        if batches:
            self.rew_mean = self.exp_manager.experience.reward_stats.mean[0]
            self.rew_std = self.exp_manager.experience.reward_stats.std[0]
            self.ts_collected = ts_collected
            self.ts_discarded = discarded_timesteps

        prev_time = times[0][1]
        for i, t in enumerate(times):
            if i == 0:
                continue
            print(
                f"ShufflingExperienceManager.get_all_batches_shuffled, {t[0]}: {t[1]-prev_time}"
            )
            prev_time = t[1]

        polling_times = np.array(polling_times)
        polling_time_mean = np.mean(polling_times) if len(polling_times) else 0
        polling_time_std = np.std(polling_times) if len(polling_times) else 0
        polling_time_max = np.max(polling_times) if len(polling_times) else 0
        polling_time_min = np.min(polling_times) if len(polling_times) else 0

        print(
            f"ShufflingExperienceManager.get_all_batches_shuffled, polling time mean: {polling_time_mean}"
        )
        print(
            f"ShufflingExperienceManager.get_all_batches_shuffled, polling time std: {polling_time_std}"
        )
        print(
            f"ShufflingExperienceManager.get_all_batches_shuffled, polling time max: {polling_time_max}"
        )
        print(
            f"ShufflingExperienceManager.get_all_batches_shuffled, polling time min: {polling_time_min}"
        )
        print(
            f"ShufflingExperienceManager.get_all_batches_shuffled, poll count: {len(polling_times)}"
        )
        print()
        # print(f"ShufflingExperienceManager.get_all_batches_shuffled, step availability: {', '.join([str(a) for a in step_availability])}")

        return batches

    def cleanup(self):
        if self.exp_manager is not None:
            self.exp_manager.cleanup()

        if self.cfg is not None:
            self.cfg.clear()
