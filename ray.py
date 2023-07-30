import numpy as np


class Ray:
    def __init__(self, source: np.ndarray, direction: np.ndarray):
        self.source = source
        self.direction = direction

    def at(self, t: float) -> np.ndarray:
        return self.source + (t * self.direction)

    @classmethod
    def ray_between_points(cls, source: np.ndarray, dest: np.ndarray) -> "Ray":
        direction = dest - source
        direction /= np.linalg.norm(direction)
        return cls(source, direction)

    def __neg__(self) -> "Ray":
        return Ray(self.source, -self.direction)
