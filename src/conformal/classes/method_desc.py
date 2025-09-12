from typing import Type, Any
from dataclasses import dataclass, field

from conformal.classes.conformalizers import BaseRegionPredictor


@dataclass
class ConformalMethodDescription:
    name: str
    name_mathtext: str
    base_model_name: str
    score_name: str
    class_name: str
    cls: Type[BaseRegionPredictor]
    kwargs: dict[str, Any] = field(default_factory=dict)
    instance: BaseRegionPredictor = field(init=False, repr=False)
