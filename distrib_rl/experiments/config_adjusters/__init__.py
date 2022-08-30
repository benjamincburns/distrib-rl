from .adjuster import Adjuster
from .model import BaseAdjusterConfig
from .null_adjuster import NullAdjuster, NullAdjusterConfig
from .basic_adjuster import BasicAdjuster, BasicAdjusterConfig, BasicAdjusterRange
from .list_adjuster import ListAdjuster, ListAdjusterConfig
from .grid_adjuster import GridAdjuster, GridAdjusterConfig, GridAdjustable
from .parallel_adjuster import (
    ParallelAdjuster,
    ParallelAdjusterConfig,
    ParallelAdjustable,
)

from typing_extensions import Annotated
from typing import Union
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
