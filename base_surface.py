from typing import List, Optional, Tuple

import numpy as np

from material import Material
from ray import Ray


class Surface:
    def __init__(self, material: Material):
        self.material = material

    def intersect(self, ray: Ray) -> Optional[np.ndarray]:
        """
        Return the intersection point between the ray and surface, or None if they don't intersect.
        """
        raise NotImplementedError()

    def reflection_ray(self, ray: Ray, intersection: np.ndarray) -> Ray:
        """
        Receive a ray and intersection point on the surface, and return the reflected ray.
        """
        normal = self.normal_at_point(intersection, ray.direction)
        reflection_vec = ray.direction - 2 * (ray.direction @ normal) * normal
        reflection_vec /= np.linalg.norm(reflection_vec)
        return Ray(intersection, reflection_vec)

    def normal_at_point(self, point: np.ndarray, ray_vec: np.ndarray) -> np.ndarray:
        """
        Return the normal vector of the surface at the given point.
        """
        raise NotImplementedError()


def get_closest_surface(
    ray: Ray,
    surfaces: List[Surface],
    source_surface: Surface = None,
) -> Tuple[Surface, np.ndarray]:
    """
    Receive a ray and the list of surfaces, and return a pair of the closest surface and its
    intersection point with the ray.
    :param ray: The shot ray.
    :param surfaces: A list of the surfaces in the scene.
    :param source_surface: An optional parameter indicating the surface, the ray is shot from.
    This surface will be ignored when searching for the closest surface.
    :returns: A pair of the closest surface and the intersection point.
    """
    closest_surface = None
    closest_intersection = None
    min_dist = float("inf")

    for surface in surfaces:
        if surface == source_surface:
            continue

        intersection = surface.intersect(ray)
        if intersection is None:
            continue

        dist = np.linalg.norm(intersection - ray.source)
        if dist < min_dist:
            closest_surface = surface
            closest_intersection = intersection
            min_dist = dist

    return closest_surface, closest_intersection
