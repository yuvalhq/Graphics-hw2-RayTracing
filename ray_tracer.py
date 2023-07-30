import argparse
import itertools
from typing import List

import numpy as np
from PIL import Image

from base_surface import Surface
from camera import Camera
from colors import calculate_color
from consts import COLOR_CHANNELS, COLOR_SCALE
from light import Light
from progressbar import progressbar
from ray import Ray
from scene import SceneSettings, parse_scene_file


class RayTracer:
    def __init__(
        self,
        camera: Camera,
        scene_settings: SceneSettings,
        surfaces: List[Surface],
        lights: List[Light],
    ):
        self.camera = camera
        self.scene_settings = scene_settings
        self.surfaces = surfaces
        self.lights = lights

        self.v = Ray.ray_between_points(self.camera.position, self.camera.look_at)
        self.p_c = self.v.at(self.camera.screen_distance)
        self.v_right = np.cross(self.v.direction, self.camera.up_vector)
        self.v_right /= np.linalg.norm(self.v_right)
        self.v_up = np.cross(self.v_right, self.v.direction)
        self.v_up /= np.linalg.norm(self.v_up)

    def construct_ray_through_pixel(
        self, height: int, width: int, i: int, j: int
    ) -> Ray:
        ratio = self.camera.screen_width / width
        p = (
            self.p_c
            + ((j - width // 2) * ratio * self.v_right)
            - ((i - height // 2) * ratio * self.v_up)
        )
        return Ray.ray_between_points(self.camera.position, p)

    def ray_trace(self, img_mat: np.ndarray) -> None:
        height, width, _ = img_mat.shape

        for i, j in progressbar(
            itertools.product(range(height), range(width)),
            count=height * width,
            prefix="Computing: ",
        ):
            ray = self.construct_ray_through_pixel(height, width, i, j)

            color = calculate_color(
                ray,
                self.surfaces,
                self.lights,
                self.scene_settings,
            )
            img_mat[i][j] = np.clip(color, 0, 1) * COLOR_SCALE


def save_image(image_array: np.ndarray, save_path: str) -> None:
    image = Image.fromarray(np.uint8(image_array))
    image.save(save_path)


def main():
    parser = argparse.ArgumentParser(description="Python Ray Tracer")
    parser.add_argument("scene_file", type=str, help="Path to the scene file")
    parser.add_argument("output_image", type=str, help="Name of the output image file")
    parser.add_argument("--width", type=int, default=500, help="Image width")
    parser.add_argument("--height", type=int, default=500, help="Image height")
    args = parser.parse_args()

    camera, scene_settings, surfaces, lights = parse_scene_file(args.scene_file)
    ray_tracer = RayTracer(camera, scene_settings, surfaces, lights)
    img_mat = np.zeros((args.height, args.width, COLOR_CHANNELS))
    ray_tracer.ray_trace(img_mat)
    save_image(img_mat, args.output_image)


if __name__ == "__main__":
    main()
