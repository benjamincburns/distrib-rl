from functools import reduce
import operator
from typing import Literal
from distrib_rl.experiments.config_adjusters import Adjuster

from pydantic import BaseModel
from distrib_rl.experiments.config_adjusters.model import BaseAdjusterConfig


class BasicAdjuster(Adjuster):
    def __init__(self, adjustment_config: "BasicAdjusterConfig", cfg):
        super().__init__()
        self.keys = adjustment_config.key_set
        self.begin = adjustment_config.range.begin
        self.end = adjustment_config.range.end
        self.increment = adjustment_config.range.increment
        self.reset_per_increment = adjustment_config.full_reset_per_increment

        cfg_entry = reduce(operator.getitem, self.keys[:-1], cfg)
        self.original_cfg_value = cfg_entry[self.keys[-1]]
        self.current_adjusted_value = self.begin

    def reset_per_increment(self):
        return self.reset_per_increment

    def step(self):
        if self.is_done():
            return True

        self.current_adjusted_value += self.increment
        return False

    def adjust_config(self, cfg):
        cfg_entry = reduce(operator.getitem, self.keys[:-1], cfg)

        adjusted_value = self.current_adjusted_value

        # round off floating math errors
        adjusted_value *= 1e5
        adjusted_value = round(adjusted_value)
        adjusted_value /= 1e5

        # cast adjusted value back to initial type
        adjusted_value = type(self.original_cfg_value)(adjusted_value)

        cfg_entry[self.keys[-1]] = adjusted_value
        self.current_adjusted_value = adjusted_value

    def get_name(self):
        name = ""
        for key in self.keys:
            name = "{}_{}".format(name, key)
        if name[0] == "_":
            name = name[1:]

        adjusted_value = self.current_adjusted_value

        # round off floating math errors
        adjusted_value *= 1e5
        adjusted_value = round(adjusted_value)
        adjusted_value /= 1e5

        name = "{}_{}".format(name, adjusted_value)
        return name

    def reset_config(self, cfg):
        cfg_entry = reduce(operator.getitem, self.keys[:-1], cfg)
        cfg_entry[self.keys[-1]] = self.original_cfg_value

        self.current_adjusted_value = self.begin

    def reset(self):
        self.current_adjusted_value = self.begin

    def is_done(self):
        return self.current_adjusted_value >= self.end


class BasicAdjusterRange(BaseModel):
    # would be really nice if I could tell pydantic to infer the type of these
    # fields from the type of the field being mutated
    begin: float
    end: float
    increment: float


class BasicAdjusterConfig(BaseAdjusterConfig[BasicAdjuster]):
    _adjustor_type = BasicAdjuster
    type: Literal["basic"]
    key_set: list[str]
    range: BasicAdjusterRange
    full_reset_per_increment: bool = False
