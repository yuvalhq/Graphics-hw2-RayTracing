from typing import List

import numpy as np

from camera import Camera
from cube import Cube
from infinite_plane import InfinitePlane
from light import Light
from material import Material
from sphere import Sphere


class SceneSettings:
    def __init__(
        self,
        background_color: List[float],
        root_number_shadow_rays: float,
        max_recursions: float,
    ):
        self.background_color = np.array(background_color)
        self.root_number_shadow_rays = int(root_number_shadow_rays)
        self.max_recursions = max_recursions


def parse_scene_file(file_path):
    surfaces, materials, lights = [], [], []
    camera = None
    scene_settings = None
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            if obj_type == "cam":
                camera = Camera(
                    params[:3], params[3:6], params[6:9], params[9], params[10]
                )
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], params[3], params[4])
            elif obj_type == "mtl":
                material = Material(
                    params[:3], params[3:6], params[6:9], params[9], params[10]
                )
                materials.append(material)
            elif obj_type == "sph":
                material = materials[int(params[4]) - 1]
                sphere = Sphere(params[:3], params[3], material)
                surfaces.append(sphere)
            elif obj_type == "pln":
                material = materials[int(params[4]) - 1]
                plane = InfinitePlane(params[:3], params[3], material)
                surfaces.append(plane)
            elif obj_type == "box":
                material = materials[int(params[4]) - 1]
                cube = Cube(params[:3], params[3], material)
                surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                lights.append(light)
            else:
                raise ValueError("Unknown object type: {}".format(obj_type))
    return camera, scene_settings, surfaces, lights
