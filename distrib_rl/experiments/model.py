from typing import Optional, Union
from typing_extensions import Annotated
from pydantic import BaseModel, root_validator

from distrib_rl.experiments.config_adjusters import (
    BasicAdjusterConfig,
    ListAdjusterConfig,
    NullAdjusterConfig,
    GridAdjusterConfig,
    ParallelAdjusterConfig,
)

from pydantic import Field

AdjustmentConfig = Annotated[
    Union[
        BasicAdjusterConfig,
        ListAdjusterConfig,
        NullAdjusterConfig,
        GridAdjusterConfig,
        ParallelAdjusterConfig,
    ],
    Field(discriminator="type"),
]


class TerminalConditionsConfig(BaseModel):
    policy_reward: Optional[float]
    max_timesteps: Optional[int]
    max_epoch: Optional[int]

    @root_validator
    def has_at_least_one_terminal_condition(cls, values):
        for key, item in values.items():
            if (
                key in ["policy_reward", "max_timesteps", "max_epoch"]
                and item is not None
            ):
                return values
        raise ValueError("At least one terminal condition must be specified")


class ExperimentConfig(BaseModel):
    experiment_name: str
    config_file: str
    num_trials_per_adjustment: int
    steps_per_save: int
    config_adjustments: list[AdjustmentConfig]
    terminal_conditions: TerminalConditionsConfig
