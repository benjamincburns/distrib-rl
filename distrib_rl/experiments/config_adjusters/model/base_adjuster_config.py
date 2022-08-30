from typing import Generic, TypeVar
from pydantic import BaseModel

from distrib_rl.experiments.config_adjusters import Adjuster


AdjusterT = TypeVar("AdjusterT")


class BaseAdjusterConfig(Generic[AdjusterT], BaseModel):
    _adjustor_type: AdjusterT = Adjuster

    def to_adjuster(self, cfg) -> AdjusterT:
        return self._adjustor_type(self, cfg)
