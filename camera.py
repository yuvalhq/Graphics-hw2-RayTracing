from typing import List

import numpy as np


class Camera:
    def __init__(
        self,
        position: List[float],
        look_at: List[float],
        up_vector: List[float],
        screen_distance: float,
        screen_width: float,
    ):
        self.position = np.array(position)
        self.look_at = np.array(look_at)
        self.up_vector = np.array(up_vector)
        self.screen_distance = screen_distance
        self.screen_width = screen_width
