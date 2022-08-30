from typing_extensions import Annotated
from typing import Any, Literal, Union
from pydantic import BaseModel, Field, constr


class BasicAdjustmentRange(BaseModel):
    # would be really nice if I could tell pydantic to infer the type of these
    # fields from the type of the field being mutated
    begin: float
    end: float
    increment: float


class BasicAdjusterConfig(BaseModel):
    type: Literal["basic"]
    key_set: list[str]
    range: BasicAdjustmentRange
    full_reset_per_increment: bool = False


class ListAdjusterConfig(BaseModel):
    type: Literal["list"]
    key_set: list[str]

    # would be really nice if I could tell pydantic to infer the type of this
    # list from the type of the field being mutated
    values: list[Any]

    full_reset_per_increment: bool = False


class NullAdjusterConfig(BaseModel):
    type: Literal["null"]
    name: constr(min_length=1)


GridAdjustable = Annotated[
    Union[ListAdjusterConfig, BasicAdjusterConfig],
    Field(discriminator="type"),
]


class GridAdjusterConfig(BaseModel):
    type: Literal["grid"]
    adjusters: dict[str, GridAdjustable]


ParallelAdjustable = Annotated[
    Union[ListAdjusterConfig, BasicAdjusterConfig],
    Field(discriminator="type"),
]


class ParallelAdjusterConfig(BaseModel):
    type: Literal["parallel"]
    adjusters: dict[str, ParallelAdjustable]


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
