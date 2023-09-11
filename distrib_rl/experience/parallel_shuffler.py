from distrib_rl.mpframework import Process
import cProfile as profile
import os
import logging
from datetime import datetime


class ParallelShuffler(Process):
    def __init__(self, name):  # , loop_wait_time=0.001):
        super().__init__(name)  # , loop_wait_time)
        self.cfg = None
        self.exp_manager = None
        self.server = None
        self.total_ts = 0
        self.ts_per_update = 0
        self.batch_size = 0

    def init(self):
        import numpy
        from distrib_rl.experience import DistribExperienceManager
        from distrib_rl.distrib import RedisServer

        self.task_checker.wait_for_initialization(header="initialization_data")
        self.cfg = self.task_checker.latest_data.copy()
        self.cfg["rng"] = numpy.random.RandomState(self.cfg["seed"])
        self.server = RedisServer(self.cfg["experience_replay"]["max_buffer_size"])
        self.server.connect(new_server_instance=False)
        self.exp_manager = DistribExperienceManager(self.cfg, server=self.server)

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

    def update(self, header, data):
        pass

    def run(self):
        logging.basicConfig(
            format="%(asctime)s %(levelname)s - %(processName)s:%(process)d - %(message)s",
            level=logging.INFO,
        )
        if os.environ.get("DISTRIB_RL_PROFILING") is not None:
            profile.runctx(
                statement="_run()",
                locals={"_run": super().run},
                globals={},
                filename=f"profile-{datetime.now().isoformat()}-{os.getpid()}.prof",
            )
        else:
            super().run()

    def step(self):
        pass

    def publish(self):
        if not self.results_publisher.is_empty():
            return

        publisher = self.results_publisher

        returns = self.exp_manager.get_timesteps_as_batches(self.ts_per_update)

        if returns is not None:
            ts_collected, fps, discarded_timesteps = returns

            batches = self.exp_manager.experience.get_all_batches_shuffled(
                self.batch_size
            )

            if batches:
                self._logger.debug("publishing {} batches".format(len(batches)))

                for batch in batches:
                    publisher.publish(header="experience_batch", data=batch)

                rew_mean = self.exp_manager.experience.reward_stats.mean[0]
                rew_std = self.exp_manager.experience.reward_stats.std[0]

                publisher.publish(
                    header="misc_data",
                    data=(rew_mean, rew_std, ts_collected, fps, discarded_timesteps),
                )

    def cleanup(self):
        print("SHUTTING DOWN SHUFFLING PROCESS")
        if self.server is not None:
            self.server.disconnect()

        if self.exp_manager is not None:
            self.exp_manager.cleanup()

        if self.cfg is not None:
            self.cfg.clear()

        print("SHUFFLING PROCESS SHUTDOWN COMPLETE")
