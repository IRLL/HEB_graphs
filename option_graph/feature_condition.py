# OptionGraph for explainable hierarchical reinforcement learning
# Copyright (C) 2021 Mathïs FEDERICO <https://www.gnu.org/licenses/>

import numpy as np

class FeatureCondition():

    def __init__(self, feature_id:str, image=None) -> None:
        self.feature_id = feature_id
        self.image = image

    def __call__(self, observations:np.ndarray) -> np.ndarray:
        raise NotImplementedError
