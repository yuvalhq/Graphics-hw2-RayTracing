from typing import List, Optional

import numpy as np

from base_surface import Surface, get_closest_surface
from light import Light
from ray import Ray
from scene import SceneSettings


def calculate_color(
    ray: Ray,
    surfaces: List[Surface],
    lights: List[Light],
    scene_settings: SceneSettings,
    source_surface: Optional[Surface] = None,
    iteration: int = 0,
) -> np.ndarray:
    if iteration == scene_settings.max_recursions:
        return scene_settings.background_color

    surface, intersection = get_closest_surface(
        ray, surfaces, source_surface=source_surface
    )
    if not surface:
        return scene_settings.background_color

    color = calculate_phong_specularity(
        intersection, ray, surface, surfaces, lights, scene_settings
    ) * (1 - surface.material.transparency)
    if surface.material.transparency > 0:
        color += (
            calculate_color(
                Ray(intersection, ray.direction),
                surfaces,
                lights,
                scene_settings,
                surface,
                iteration + 1,
            )
            * surface.material.transparency
        )
    if surface.material.is_reflective():
        color += (
            calculate_color(
                surface.reflection_ray(ray, intersection),
                surfaces,
                lights,
                scene_settings,
                surface,
                iteration + 1,
            )
            * surface.material.reflection_color
        )

    return color


def calculate_phong_specularity(
    point: np.ndarray,
    ray: Ray,
    surface: Surface,
    surfaces: List[Surface],
    lights: List[Light],
    scene_settings: SceneSettings,
) -> np.ndarray:
    return sum(
        light.calculate_phong_specularity(
            point,
            -ray,
            surface,
            surfaces,
            scene_settings.root_number_shadow_rays,
        )
        for light in lights
    )
