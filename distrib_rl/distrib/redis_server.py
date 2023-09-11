from redis import Redis
from distrib_rl.distrib import redis_keys
from distrib_rl.distrib.message_serialization import MessageSerializer
import time
import pyjson5 as json
import os
import logging


class RedisServer(object):
    INITIALIZING_STATUS = "REDIS_SERVER_INITIALIZING_STATUS"
    RUNNING_STATUS = "REDIS_SERVER_RUNNING_STATUS"
    STOPPING_STATUS = "REDIS_SERVER_STOPPING_STATUS"
    RESET_STATUS = "REDIS_SERVER_RESET_STATUS"
    RECONFIGURE_STATUS = "REDIS_SERVER_RECONFIGURE_STATUS"
    AWAITING_ENV_SPACES_STATUS = "REDIS_SERVER_AWAITING_ENV_SPACES_STATUS"

    def __init__(self, max_queue_size):
        self.redis = None
        self._logger = logging.getLogger("RedisServer")
        self.max_queue_size = max_queue_size
        self.internal_buffer = []

        self.last_sps_measure = time.time()
        self.accumulated_sps = 0
        self.available_timesteps = 0
        self.discarded_timesteps = 0
        self.steps_per_second = 0
        self._message_serializer = MessageSerializer()
        self.current_epoch = 0

        # TODO: hardcoded for now, but should be made configurable in the future
        self.max_policy_age = float("inf")

    def connect(self, clear_existing=False, new_server_instance=True):
        ip = os.environ.get("REDIS_HOST", default="localhost")
        port = os.environ.get("REDIS_PORT", default=6379)
        password = os.environ.get("REDIS_PASSWORD", default=None)
        self.redis = Redis(host=ip, port=port, password=password)
        if clear_existing:
            self.redis.flushall()

        if new_server_instance:
            self.redis.set(
                redis_keys.SERVER_CURRENT_STATUS_KEY, RedisServer.INITIALIZING_STATUS
            )
            self.redis.set(redis_keys.NEW_DATA_AMOUNT_KEY, 0)

    def get_n_timesteps(self, n):
        self._update_buffer()
        n_collected = 0
        returns = []

        if self.available_timesteps < n:
            return returns

        while n_collected < n:
            if len(self.internal_buffer) == 0 and (n_collected < n):
                self._logger.debug(
                    "Buffer empty, but still need to collect {} more timesteps.".format(
                        n - n_collected
                    )
                )

            ret = self.internal_buffer.pop(-1)
            trajectory, num_timesteps, policy_epoch = ret
            self.available_timesteps -= num_timesteps

            returns.append(trajectory)
            n_collected += num_timesteps

        return returns

    def get_up_to_n_timesteps(self, n):
        self._update_buffer()
        if len(self.internal_buffer) == 0:
            return []

        n_collected = 0
        buffer = self.internal_buffer
        returns = []

        while len(buffer) > 0 and n_collected < n:
            ret = buffer.pop(-1)
            trajectory, num_timesteps, _ = ret
            self.available_timesteps -= num_timesteps

            returns.append(trajectory)
            n_collected += num_timesteps

        return returns

    def get_policy_rewards(self):
        # rewards are pushed as packed/compressed lists of scalar values
        # atomic_pop_all returns all entries for a given key as a list, giving
        # us a list of lists of reward scalars. We need to flatten it before we
        # return.
        reward_lists = self._atomic_pop(
            redis_keys.CLIENT_POLICY_REWARD_KEY, self.max_queue_size
        )

        # Actual flattening happens here. Not very readable, but it's supposedly
        # the fastest way to flatten a list[list[Any]], per
        # https://stackoverflow.com/a/952952
        return [
            reward
            for reward_list in reward_lists
            for reward in self._message_serializer.unpack(reward_list)
        ]

    def push_update(
        self,
        policy_params,
        val_params,
        strategy_frames,
        strategy_history,
        current_epoch,
    ):
        red = self.redis

        self.current_epoch = current_epoch

        packed_policy = self._message_serializer.pack(policy_params)
        packed_val = self._message_serializer.pack(val_params)
        packed_frames = self._message_serializer.pack(strategy_frames)
        packed_history = self._message_serializer.pack(strategy_history)

        pipe = red.pipeline()
        pipe.set(redis_keys.SERVER_POLICY_PARAMS_KEY, packed_policy)
        pipe.set(redis_keys.SERVER_VAL_PARAMS_KEY, packed_val)
        pipe.set(redis_keys.SERVER_STRATEGY_FRAMES_KEY, packed_frames)
        pipe.set(redis_keys.SERVER_STRATEGY_HISTORY_KEY, packed_history)
        pipe.set(redis_keys.SERVER_CURRENT_UPDATE_KEY, current_epoch)
        pipe.execute()

    def push_cfg(self, cfg):
        self._configure_serialization(cfg)

        dev = cfg["device"]
        rng = cfg["rng"]

        del cfg["rng"]
        cfg["device"] = "cpu"
        self.redis.set(redis_keys.SERVER_CONFIG_KEY, json.dumps(cfg))

        cfg["rng"] = rng
        cfg["device"] = dev

    def _configure_serialization(self, cfg):
        networking_cfg = cfg.get("networking", {})
        compression_type = networking_cfg.get("compression", None)
        if compression_type:
            self._message_serializer = MessageSerializer(
                compression_type=compression_type
            )
        else:
            self._message_serializer = MessageSerializer()

    def signal_ready(self):
        self.redis.set(redis_keys.SERVER_CURRENT_STATUS_KEY, RedisServer.RUNNING_STATUS)

        pipe = self.redis.pipeline()
        pipe.delete(redis_keys.CLIENT_EXPERIENCE_KEY)
        pipe.delete(redis_keys.CLIENT_POLICY_REWARD_KEY)
        pipe.set(redis_keys.NEW_DATA_AMOUNT_KEY, 0)

        # Short sleep to let any pre-connected clients update their policies before we erase the existing data.
        time.sleep(1)
        pipe.execute()

    def get_env_spaces(self):
        self.redis.set(
            redis_keys.SERVER_CURRENT_STATUS_KEY, RedisServer.AWAITING_ENV_SPACES_STATUS
        )
        in_space, out_space = None, None

        while in_space is None or out_space is None:
            data = self.redis.get(redis_keys.ENV_SPACES_KEY)
            if data is None:
                time.sleep(0.1)
                continue

            in_space, out_space = self._message_serializer.unpack(data)
        return in_space, out_space

    def _update_buffer(self):
        returns = self._atomic_pop(redis_keys.CLIENT_EXPERIENCE_KEY)
        collected_timesteps = 0
        discarded_timesteps = self.discarded_timesteps
        for packed_trajectories in returns:
            trajectories = self._message_serializer.unpack(packed_trajectories)
            for serialized_trajectory, policy_epoch in trajectories:
                if (
                    abs(self.current_epoch - policy_epoch) <= self.max_policy_age
                    and self.available_timesteps < self.max_queue_size
                ):
                    n_timesteps = len(serialized_trajectory[0])
                    self.available_timesteps += n_timesteps
                    collected_timesteps += n_timesteps
                    self.internal_buffer.append(
                        (serialized_trajectory, n_timesteps, policy_epoch)
                    )
                elif self.available_timesteps >= self.max_queue_size:
                    break
                else:
                    self.discarded_timesteps += len(serialized_trajectory[0])
            if self.available_timesteps >= self.max_queue_size:
                break

        if self.discarded_timesteps - discarded_timesteps > 0:
            self._logger.debug(
                f"Discarded {self.discarded_timesteps - discarded_timesteps} old or excess timesteps"
            )

        self._update_sps(collected_timesteps)
        self._trim_buffer()

    def _trim_buffer(self):
        discarded = self.discarded_timesteps
        # get rid of old trajectories
        for item in self.internal_buffer:
            _, num_timesteps, policy_epoch = item
            if self.current_epoch - policy_epoch > self.max_policy_age:
                self.discarded_timesteps += num_timesteps
                self.available_timesteps -= num_timesteps
                self.internal_buffer.remove(item)

        if self.discarded_timesteps - discarded > 0:
            self._logger.debug(
                f"Discarded {self.discarded_timesteps - discarded} old timesteps"
            )

        discarded = self.discarded_timesteps

        # drop excess trajectories
        while self.available_timesteps > self.max_queue_size:
            ret = self.internal_buffer.pop(0)
            _, num_timesteps, _ = ret
            self.discarded_timesteps += num_timesteps
            self.available_timesteps -= num_timesteps
            del ret

        if self.discarded_timesteps - discarded > 0:
            self._logger.debug(
                f"Discarded {self.discarded_timesteps - discarded} extra timesteps"
            )

    def _update_sps(self, collected_timesteps):
        self.accumulated_sps += collected_timesteps
        elapsed = time.time() - self.last_sps_measure

        if elapsed >= 1:
            self.steps_per_second = self.accumulated_sps / elapsed
            self.accumulated_sps = 0
            self.last_sps_measure = time.time()

    def _atomic_pop(self, key, count=-1):
        if count == -1:
            count = self.max_queue_size

        pipe = self.redis.pipeline()
        # pipe.command("LPOP", key, count)
        pipe.lpop(key, count=count)
        # pipe.lrange(key, 0, count)
        # pipe.delete(key)
        packed_results = pipe.execute()[0]
        if packed_results is None:
            return []
        return packed_results

    def disconnect(self):
        if self.redis is not None:
            self.redis.flushall()
            print("\nATTEMPTING TO SET REDIS TO STOPPING STATUS")
            self.redis.set(
                redis_keys.SERVER_CURRENT_STATUS_KEY, RedisServer.STOPPING_STATUS
            )
            self.redis.close()

        del self.internal_buffer
        self.internal_buffer = []
