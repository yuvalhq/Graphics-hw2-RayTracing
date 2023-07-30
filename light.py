from typing import List

import numpy as np

import vector
from base_surface import Surface, get_closest_surface
from consts import EPSILON
from ray import Ray


class Light:
    def __init__(
        self,
        position: List[float],
        color: List[float],
        specular_intensity: float,
        shadow_intensity: float,
        radius: float,
    ):
        self.position = np.array(position)
        self.color = np.array(color)
        self.specular_intensity = specular_intensity
        self.shadow_intensity = shadow_intensity
        self.radius = radius

    @staticmethod
    def is_path_clear(
        source: np.ndarray,
        dest: np.ndarray,
        surfaces: List[Surface],
    ) -> bool:
        """
        Returns true iff the light source a surface at the given dest point without hitting
        any other surface on the way.
        This method expects the dest point to be on a surface and the source to be a light source.
        """
        light_ray = Ray.ray_between_points(source, dest)
        _, light_intersection = get_closest_surface(light_ray, surfaces)
        return light_intersection is None or np.allclose(
            dest, light_intersection, atol=EPSILON
        )

    def calculate_intensity(
        self,
        surfaces: List[Surface],
        root_number_shadow_rays: int,
        point: np.ndarray,
    ) -> float:
        """
        Receive the list of surfaces, the number of shadow rays to case and a point, and
        return the light intensity value on the point measured as coefficient in the range [0, 1].
        The calculation is done according to the "soft shadows" algorithm.
        If the number of shadow rays is 1, we calculate "hard shadows" instead.
        """
        if root_number_shadow_rays == 1:
            return 1.0 - self.shadow_intensity * (
                not Light.is_path_clear(self.position, point, surfaces)
            )

        normal = Ray.ray_between_points(self.position, point)
        vec1, vec2 = vector.orthonormal_vector_pair(normal.direction)

        row_indices = np.repeat(
            np.arange(root_number_shadow_rays),
            root_number_shadow_rays,
        )[:, np.newaxis]
        col_indices = np.tile(
            np.arange(root_number_shadow_rays),
            root_number_shadow_rays,
        )[:, np.newaxis]

        number_shadow_rays = root_number_shadow_rays * root_number_shadow_rays
        vec1_repeated = np.repeat(vec1[np.newaxis, :], number_shadow_rays, axis=0)
        vec2_repeated = np.repeat(vec2[np.newaxis, :], number_shadow_rays, axis=0)

        half_r = self.radius / 2.0
        top_left = self.position - (half_r * vec1) - (half_r * vec2)
        grid_square_length = self.radius / root_number_shadow_rays
        corners1 = (
            top_left
            + row_indices * grid_square_length * vec1_repeated
            + col_indices * grid_square_length * vec2_repeated
        )
        corners2 = (
            corners1
            + grid_square_length * vec1_repeated
            + grid_square_length * vec2_repeated
        )

        min_coords = np.minimum(corners1, corners2)
        max_coords = np.maximum(corners1, corners2)
        light_sources = np.random.uniform(min_coords, max_coords)
        light_hit_cnt = sum(
            Light.is_path_clear(light_source, point, surfaces)
            for light_source in light_sources
        )
        return (1 - self.shadow_intensity) + self.shadow_intensity * (
            light_hit_cnt / (root_number_shadow_rays**2)
        )

    def calculate_phong_specularity(
        self,
        point: np.ndarray,
        v: np.ndarray,
        surface: Surface,
        surfaces: List[Surface],
        root_number_shadow_rays: int,
    ) -> np.ndarray:
        """
        Calculate the phong specularity color at the given point, assuming that it's on a surface.

        :param point: The point to calculate the phong specularity color of.
        :param v: The ray to calculate the specularity for.
        At the first call, this ray will be from shot the viewer.
        :param surface: The surface the point is on.
        :param surfaces: The list of all surfaces in the scene.
        :param root_number_shadow_rays: The root of the number of shadow rays to cast.
        """
        l = Ray.ray_between_points(point, self.position)
        normal = surface.normal_at_point(point, -l.direction)
        reflected_ray = surface.reflection_ray(-l, point)

        light_intensity = self.calculate_intensity(
            surfaces, root_number_shadow_rays, point
        )
        diffuse = surface.material.diffuse_color * (normal @ l.direction)
        specular = (
            surface.material.specular_color
            * self.specular_intensity
            * (v.direction @ reflected_ray.direction) ** surface.material.shininess
        )
        return (diffuse + specular) * self.color * light_intensity
