from distrib_rl.experiments.config_adjusters import (
    Adjuster,
    ListAdjusterConfig,
    BasicAdjusterConfig,
)
from distrib_rl.experiments.config_adjusters.model import BaseAdjusterConfig

from typing_extensions import Annotated
from typing import Union, Literal
from pydantic import Field


class ParallelAdjuster(Adjuster):
    def __init__(self, adjustment_config: "ParallelAdjusterConfig", cfg):
        super().__init__()
        for adjuster_config in adjuster_config.adjusters:
            adjustment = adjuster_config.to_adjuster(cfg)
            self.adjustments.append(adjustment)

    def step(self):
        done = True
        for adjustment in self.adjustments:
            if not adjustment.step():
                done = False
        return done

    def adjust_config(self, cfg):
        for adjustment in self.adjustments:
            adjustment.adjust_config(cfg)

    def reset_per_increment(self):
        reset_this_increment = False
        for adjustment in self.adjustments:
            if adjustment.reset_per_increment:
                reset_this_increment = True

        return reset_this_increment


ParallelAdjustable = Annotated[
    Union[ListAdjusterConfig, BasicAdjusterConfig],
    Field(discriminator="type"),
]


class ParallelAdjusterConfig(BaseAdjusterConfig[ParallelAdjuster]):
    _adjustor_type = ParallelAdjuster
    type: Literal["parallel"]
    adjusters: list[ParallelAdjustable]
