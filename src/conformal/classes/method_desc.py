from dataclasses import dataclass

from conformal.classes.conformalizers import BaseRegionPredictor


@dataclass
class ConformalMethodDescription:
    name: str
    name_mathtext: str
    base_model_name: str
    score_name: str
    class_name: str
    instance: BaseRegionPredictor
