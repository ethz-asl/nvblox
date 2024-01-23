#!/usr/bin/python3

import argparse
import open3d as o3d
import json
import numpy as np
from typing import Optional
from moviepy.editor import ImageSequenceClip

# Methods to generate a spherical camera trajectory.


def _trans_t(t):
    r"""Taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py.
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rot_phi(phi):
    r"""Taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py.
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _rot_theta(th):
    r"""Taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py.
    """
    return np.array(
        [
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1],
        ],
        dtype=np.float32,
    )


def _pose_spherical(theta: float,
                    phi: float,
                    radius: float,
                    offset: Optional[np.ndarray] = None,
                    vec_up: Optional[np.ndarray] = None):
    """
    Taken from https://github.com/sxyu/svox2/blob/master/opt/util/util.py.
    Generate spherical rendering poses, from NeRF. Forgive the code horror
    :return: r (3,), t (3,)
    """
    c2w = _trans_t(radius)
    c2w = _rot_phi(phi / 180.0 * np.pi) @ c2w
    c2w = _rot_theta(theta / 180.0 * np.pi) @ c2w
    c2w = (np.array(
        [[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
        dtype=np.float32,
    ) @ c2w)
    if vec_up is not None:
        vec_up = vec_up / np.linalg.norm(vec_up)
        vec_1 = np.array([vec_up[0], -vec_up[2], vec_up[1]])
        vec_2 = np.cross(vec_up, vec_1)
        trans = np.eye(4, 4, dtype=np.float32)
        trans[:3, 0] = vec_1
        trans[:3, 1] = vec_2
        trans[:3, 2] = vec_up
        c2w = trans @ c2w
    c2w = c2w @ np.diag(np.array([-1, 1, -1, 1], dtype=np.float32))
    if offset is not None:
        c2w[:3, 3] += offset
    return c2w


def pose_spiral(num_views=150,
                elevation=-45.0,
                elevation2=-12.0,
                radius=0.85,
                offset=np.array([0.0, 0.0, 0.0]),
                vec_up_nerf_format=np.array([0., 0., -1.])):
    r"""Adapted from https://github.com/sxyu/svox2/blob/master/opt/
    render_imgs_circle.py"""
    assert (num_views % 2 == 0)
    num_views = num_views // 2
    angles = np.linspace(-270, 90, num_views + 1)[:-1]
    elevations = np.linspace(elevation, elevation2, num_views)
    c2ws = [
        _pose_spherical(
            angle,
            ele,
            radius,
            offset,
            vec_up=vec_up_nerf_format,
        ) for ele, angle in zip(elevations, angles)
    ]
    c2ws += [
        _pose_spherical(
            angle,
            ele,
            radius,
            offset,
            vec_up=vec_up_nerf_format,
        ) for ele, angle in zip(reversed(elevations), angles)
    ]
    c2ws = np.stack(c2ws, axis=0)
    return c2ws


class MeshSpinner:

    # Set the animation callback.
    def spin(self, vis):
        ctr = vis.get_view_control()
        if self.index < len(self.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                self.trajectory.parameters[self.index], allow_arbitrary=True)
            if self.output and self.index > 0:
                image_float = np.asarray(vis.capture_screen_float_buffer(True))
                image_uint8 = (image_float * 255).astype(np.uint8)
                self.images.append(image_uint8)
        self.index = self.index + 1
        return False

    def set_view_control_from_json(self, file_path):
        # Load the file.
        with open(file_path) as file:
            params = json.load(file)
            print(params)
            # Set the controls.
            ctr = self.vis.get_view_control()
            ctr.change_field_of_view(params['trajectory'][0]['field_of_view'])
            ctr.set_front(params['trajectory'][0]['front'])
            ctr.set_lookat(params['trajectory'][0]['lookat'])
            ctr.set_up(params['trajectory'][0]['up'])
            ctr.set_zoom(params['trajectory'][0]['zoom'])

    def generate_trajectory(self):
        self.trajectory = o3d.camera.PinholeCameraTrajectory()
        traj = pose_spiral(num_views=180, radius=10.,
                           elevation=45, elevation2=25,
                           offset=np.array([3.0, 1.3, 0]),
                           vec_up_nerf_format=np.array([0., 0., -1.]))
        traj = np.concatenate([traj[-1][None], traj], axis=0)
        all_params = []
        for curr_extr in traj:
            camera_params = o3d.camera.PinholeCameraParameters()
            camera_params_current = self.vis.get_view_control(
            ).convert_to_pinhole_camera_parameters()
            camera_params.intrinsic = camera_params_current.intrinsic

            # camera_params.extrinsic = curr_extr.T
            camera_params.extrinsic = np.linalg.inv(curr_extr)
            all_params.append(camera_params)
        self.trajectory.parameters = all_params
        self.index = 0

    def visualize_ply(
            self, ply_path: str, do_normal_coloring: bool, view: str,
            output_path: str):
        self.output = False
        # Load the mesh.
        mesh = o3d.io.read_triangle_mesh(ply_path)
        print(mesh)
        mesh.compute_vertex_normals()

        # Create a window.
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(mesh)
        # Color the mesh by normals.
        self.vis.get_render_option().light_on = False
        if do_normal_coloring:
            self.vis.get_render_option().mesh_color_option = \
                o3d.visualization.MeshColorOption.Normal

        # Set the view if necessary.
        if view:
            self.set_view_control_from_json(view)

        # Generate a trajectory
        self.generate_trajectory()

        # Spin once before we start
        self.spin(self.vis)

        # Set up the output if we need to:
        if output_path:
            self.index = 0
            self.output = True
            self.images = []

        self.spin(self.vis)
        self.index = 0

        self.vis.register_animation_callback(self.spin)

        self.vis.run()

        pinhole = self.vis.get_view_control().convert_to_pinhole_camera_parameters()
        print("Focal length: ")
        print(pinhole.intrinsic.get_focal_length())
        print("PP:", pinhole.intrinsic.get_principal_point(),
              " Width: ", pinhole.intrinsic.width,
              " Height: ", pinhole.intrinsic.height)
        print(pinhole.extrinsic)

        self.vis.destroy_window()

        # If we're outputting the video, then... do it.
        if self.output:
            for image in self.images:
                print(image.shape)
            clip = ImageSequenceClip(self.images, fps=10)
            clip.write_videofile(str(output_path), fps=10, bitrate="8M", codec='mpeg4')


parser = argparse.ArgumentParser(description="Visualize a PLY mesh.")
parser.add_argument("ply_path", type=str,
                    help="Path to the ply file to visualize.")

parser.add_argument("--view", type=str,
                    help="Path to the view json file to use.")

parser.add_argument(
    "--normal_coloring", dest="do_normal_coloring", action='store_const',
    const=True, default=False,
    help="Flag indicating if we should just color the mesh by normals.")

parser.add_argument("--output_path", type=str,
                    help="Path to output the mp4 file.")

args = parser.parse_args()
mesh_spinner = MeshSpinner()
mesh_spinner.visualize_ply(
    args.ply_path, args.do_normal_coloring, args.view, args.output_path)
