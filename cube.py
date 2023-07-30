from typing import List, Optional

import numpy as np

from base_surface import Material, Surface
from ray import Ray

NORMALS = np.array(
    [
        np.array([1, 0, 0], dtype=np.float64),
        np.array([-1, 0, 0], dtype=np.float64),
        np.array([0, 1, 0], dtype=np.float64),
        np.array([0, -1, 0], dtype=np.float64),
        np.array([0, 0, 1], dtype=np.float64),
        np.array([0, 0, -1], dtype=np.float64),
    ]
)


class Cube(Surface):
    def __init__(self, position: List[float], scale: float, material: Material):
        super().__init__(material)
        self.position = np.array(position)
        self.scale = scale
        self.half_scale = scale / 2.0

    def intersect(self, ray: Ray) -> Optional[np.ndarray]:
        """
        Return the intersection point using the slab method.
        """
        t_near = float("-inf")
        t_far = float("inf")

        for i in range(3):
            if ray.direction[i] == 0:
                if (
                    ray.source[i] < self.position[i] - self.half_scale
                    or ray.source[i] > self.position[i] + self.half_scale
                ):
                    return None
            else:
                t1 = (
                    self.position[i] - self.half_scale - ray.source[i]
                ) / ray.direction[i]
                t2 = (
                    self.position[i] + self.half_scale - ray.source[i]
                ) / ray.direction[i]
                if t1 > t2:
                    t1, t2 = t2, t1

                t_near = max(t_near, t1)
                t_far = min(t_far, t2)

                if t_near > t_far or t_far < 0:
                    return None

        t = t_near if t_near >= 0 else t_far
        return ray.at(t) if t >= 0 else None

    def normal_at_point(self, point: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        """
        Calculate the normal at the given point, assuming it's on the cub's surface.
        This calculation uses the fact that the face center closest to a point on the surface
        of a cube is the center of the face the point is on.
        """
        point_repeated = np.repeat(point[np.newaxis, :], len(NORMALS), axis=0)
        centers = self.position + NORMALS * self.half_scale
        point_to_centers_vectors = np.linalg.norm(point_repeated - centers, axis=1)
        closest = np.argmin(point_to_centers_vectors)
        return NORMALS[closest]
