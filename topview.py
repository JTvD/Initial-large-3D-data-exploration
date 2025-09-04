from pathlib import Path
import open3d as o3d
import numpy as np
import pandas as pd
import argparse
import os

import pointcloud_utils as ply_utils


class TopView():
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.width = 1920
        self.height = 1080
        self.filter_value = 0.995

    def render(self):
        ''' Function to create a top view image (PNG) from a mesh (PLY) '''
        if not os.path.isdir(self.output_folder):
            os.mkdir(self.output_folder)

        for ply_file in Path(self.input_folder).glob('*.ply.gz'):
            # create paths to save the png images
            base_filename = ply_file.name.split('.')[0]
            unzipped_ply = ply_utils.unzip_ply(ply_file)

            # Replace either '_full_' or '_mg_' with the appropriate suffix
            if '_full_' in base_filename:
                rgb_im_name = base_filename.replace('_full_', '_rgb_', 1) + '.png'
                ndvi_im_name = base_filename.replace('_full_', '_ndvi_', 1) + '.png'
            else:
                rgb_im_name = base_filename + '_rgb.png'
                ndvi_im_name = base_filename + '_ndvi.png'
            rgb_im_path = os.path.join(self.output_folder, rgb_im_name)
            ndvi_im_path = os.path.join(self.output_folder, ndvi_im_name)

            # Load pointcloud
            df, pcd = ply_utils.load_df_pointcloud(unzipped_ply)

            # Remove outliers
            pcd = self.remove_outliers(pcd)

            # Render the RGB image
            self.render_image(pcd, rgb_im_path)

            # Render the NDVI image
            ndvi_pcd = self.ndvi_as_color(pcd, df)
            self.render_image(ndvi_pcd, ndvi_im_path)

    def remove_outliers(self, pcd: o3d.geometry.PointCloud | o3d.geometry.TriangleMesh) \
            -> o3d.geometry.PointCloud | o3d.geometry.TriangleMesh:
        """ Filter out RGB color outliers using a quantile threshold and normalize color values.
            Same options is available in Cloudcompare: edit -> colors -> levels)
            Args:
                pcd (o3d.geometry.PointCloud or Mesh): The input point cloud.
            Returns:
                np.ndarray: The filtered and normalized color array.
        """
        # Find the max value based on the quantile
        if pcd.get_geometry_type() == o3d.geometry.Geometry.Type.PointCloud:
            color_arr = np.asarray(pcd.colors)
        elif pcd.get_geometry_type() == o3d.geometry.Geometry.Type.TriangleMesh:
            color_arr = np.asarray(pcd.vertex_colors)
        color_tr = np.transpose(color_arr)
        r_max = np.quantile(color_tr[0], self.filter_value)
        g_max = np.quantile(color_tr[1], self.filter_value)
        b_max = np.quantile(color_tr[2], self.filter_value)

        # Set outlier to threshold value
        color_tr[0] = np.where((color_tr[0] > r_max), r_max, color_tr[0])
        color_tr[1] = np.where((color_tr[1] > g_max), g_max, color_tr[1])
        color_tr[2] = np.where((color_tr[2] > b_max), b_max, color_tr[2])

        # Normalize to prevent values above 1.0
        filtered_color_arr = np.transpose(color_tr / np.max([r_max, g_max, b_max]))
        if pcd.get_geometry_type() == o3d.geometry.Geometry.Type.PointCloud:
            pcd.colors = o3d.utility.Vector3dVector(filtered_color_arr)
        elif pcd.get_geometry_type() == o3d.geometry.Geometry.Type.TriangleMesh:
            pcd.vertex_colors = o3d.utility.Vector3dVector(filtered_color_arr)
        return pcd

    def render_image(self, pcd: o3d.geometry.PointCloud, file_path: str):
        """ Create open3D vizualization of the mesh.
            Args:
                pcd (o3d.geometry.PointCloud): The input point cloud.
                file_path (str): The path to the output image file.
            Returns:
                None
        """
        # Create the open3d Visualizer class
        vis = o3d.visualization.Visualizer()
        # Create the window to take the image
        vis.create_window(width=self.width, height=self.height, visible=False)
        # add the mr and sl meshes to the window and update
        vis.add_geometry(pcd)
        vis.update_geometry(pcd)
        # Set render parameters from json
        vis.get_render_option().load_from_json('./render_parameters.json')
        vis.update_renderer()
        # Render the view
        vis.capture_screen_image(file_path, do_render=True)
        vis.destroy_window()

    def jet_colormap(self, values: np.ndarray) -> np.ndarray:
        """ Map values in [0, 1] to RGB using a jet colormap.
            Args:
                values (np.ndarray): Input array of values in [0, 1].
            Returns:
                np.ndarray: Output array of shape (N, 3) with RGB colors.
        """
        values = np.clip(values, 0, 1)
        r = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
        g = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
        b = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
        return np.stack([r, g, b], axis=-1)

    def ndvi_as_color(self, pcd: o3d.geometry.PointCloud | o3d.geometry.TriangleMesh,
                      df: pd.DataFrame) \
            -> o3d.geometry.PointCloud | o3d.geometry.TriangleMesh:
        """ Map NDVI values to colors for visualization.
            Args:
                pcd (o3d.geometry.PointCloud | o3d.geometry.TriangleMesh):
                    The input point cloud or mesh.
                df (pd.DataFrame): The dataframe containing NDVI values.
            Returns:
                None
        """
        # Get the NDVI values from the dataframe
        ndvi_values = df['ndvi'].values / 255
        # Map NDVI values to colors (e.g., using a colormap)
        colors = self.jet_colormap(ndvi_values)
        if pcd.get_geometry_type() == o3d.geometry.Geometry.Type.PointCloud:
            pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        elif pcd.get_geometry_type() == o3d.geometry.Geometry.Type.TriangleMesh:
            pcd.vertex_colors = o3d.utility.Vector3dVector(colors[:, :3])
        return pcd


if __name__ == '__main__':
    ''' Retrieve the arguments specified by the user '''
    parser = argparse.ArgumentParser(description='Render top-view images from meshes')
    parser.add_argument('--files_path',
                        default='./example_data', type=str,
                        help='The path to the folder with the ply-meshes (default = ./ )')
    parser.add_argument('--output_path',
                        default='./topviews/', type=str,
                        help='The path to the folder where the rendered images will be saved.')
    args = parser.parse_args()

    topview_renderer = TopView(args.files_path, args.output_path)
    topview_renderer.render()
