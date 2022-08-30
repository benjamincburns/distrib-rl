from distrib_rl.experiments.config_adjusters import Adjuster
from distrib_rl.experiments.config_adjusters.model import BaseAdjusterConfig

from . import ListAdjusterConfig
from . import BasicAdjusterConfig

from pydantic import Field
from typing import Literal, Union
from typing_extensions import Annotated


class GridAdjuster(Adjuster):
    def __init__(self, adjuster_config: "GridAdjusterConfig", cfg):
        super().__init__()
        self.current_adjustment_target = 0
        self.reset_this_increment = False

        for adjuster_config in adjuster_config.adjusters:
            adjustment = adjuster_config.to_adjuster(cfg)
            self.adjustments.append(adjustment)

    def step(self):
        self.reset_this_increment = False
        self.grid_step()

    def adjust_config(self, cfg):
        for adjustment in self.adjustments:
            adjustment.adjust_config(cfg)

    def reset_per_increment(self):
        return self.reset_this_increment

    def grid_step(self):
        idx = self.current_adjustment_target
        adj = self.adjustments

        while adj[idx].is_done():
            adj[idx].reset()
            idx += 1

            if idx >= len(self.adjustments):
                idx = 0
                break

        if adj[idx].reset_per_increment:
            self.reset_this_increment = True
        adj[idx].step()
        self.current_adjustment_target = 0


GridAdjustable = Annotated[
    Union[ListAdjusterConfig, BasicAdjusterConfig],
    Field(discriminator="type"),
]


class GridAdjusterConfig(BaseAdjusterConfig[GridAdjuster]):
    _adjustor_type = GridAdjuster
    type: Literal["grid"]
    adjusters: list[GridAdjustable]
