from typing import Literal
from pydantic import constr
from distrib_rl.experiments.config_adjusters.model import BaseAdjusterConfig


class NullAdjuster(object):
    def __init__(self, adjustment_config: "NullAdjusterConfig", cfg):
        self.name = adjustment_config.name

    def step(self):
        return False

    def adjust_config(self, cfg):
        pass

    def get_name(self):
        return self.name

    def reset_config(self, cfg):
        pass

    def reset(self):
        pass

    def is_done(self):
        return False

    def reset_per_increment(self):
        return False


class NullAdjusterConfig(BaseAdjusterConfig[NullAdjuster]):
    _adjustor_type = NullAdjuster
    type: Literal["null"]
    name: constr(min_length=1)
