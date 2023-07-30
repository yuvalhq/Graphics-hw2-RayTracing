from typing import List, Optional

import numpy as np

from base_surface import Material, Surface
from ray import Ray


class Sphere(Surface):
    def __init__(self, position: List[int], radius: float, material: Material):
        super().__init__(material)
        self.position = np.array(position)
        self.radius = radius

    def intersect(self, ray: Ray) -> Optional[np.ndarray]:
        """
        Return the intersection point and distance from the source to the intersection point
        using the algebraic method shown in class.
        """
        oc = ray.source - self.position
        a = ray.direction @ ray.direction
        b = 2.0 * oc @ ray.direction
        c = oc @ oc - self.radius**2

        discriminant = b**2 - 4 * a * c
        if discriminant < 0:
            return None

        t1 = (-b - np.sqrt(discriminant)) / (2 * a)
        t2 = (-b + np.sqrt(discriminant)) / (2 * a)
        if t1 >= 0 and t2 >= 0:
            return ray.at(min(t1, t2))

        t = max(t1, t2)
        return ray.at(t) if t >= 0 else None

    def normal_at_point(self, point: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        normal = point - self.position
        return normal / np.linalg.norm(normal)
