import pointcloud_utils as ply_utils
import pandas as pd
import numpy as np
import open3d as o3d
from pathlib import Path
from skimage.color import rgb2hsv


class pointcloud_analysis():
    def __init__(self, input_folder):
        self.input_folder = input_folder

    def analyze(self):
        ''' Function to analyze pointcloud files (PLY) '''
        for ply_file in Path(self.input_folder).glob('*.ply.gz'):
            # create paths to save the png images
            base_filename = ply_file.name.split('.')[0]
            unzipped_ply = ply_utils.unzip_ply(ply_file)

            # Load pointcloud
            df, pcd = ply_utils.load_df_pointcloud(unzipped_ply)
            # Convert wvl1-wvl4 columns to Int64
            for col in ['wvl1', 'wvl2', 'wvl3', 'wvl4']:
                if col in df.columns:
                    df[col] = df[col].astype('Int64')
            print(f'Loaded {base_filename} with {len(df)} points.')

            # Example analysis: Print basic statistics of scalar fields
            # print(df.describe())
            if pcd.get_geometry_type() == o3d.geometry.Geometry.Type.PointCloud:
                print("It's a PointCloud!")
            elif pcd.get_geometry_type() == o3d.geometry.Geometry.Type.TriangleMesh:
                print("It's a TriangleMesh!")
                (triangle_area, avg_face_rgb, avg_face_nir) = self.area(pcd, df)
                # Compute the overall statistics
                print(f"Total triangle Area - Mean: {round(triangle_area.sum(), 2)}")
                # Print mean and std for each channel (R, G, B) separately
                print(f"Average Face R - Mean: {round(avg_face_rgb[:, 0].mean(), 2)}, Std: {round(avg_face_rgb[:, 0].std(), 2)}")
                print(f"Average Face G - Mean: {round(avg_face_rgb[:, 1].mean(), 2)}, Std: {round(avg_face_rgb[:, 1].std(), 2)}")
                print(f"Average Face B - Mean: {round(avg_face_rgb[:, 2].mean(), 2)}, Std: {round(avg_face_rgb[:, 2].std(), 2)}")
                print(f"Average Face NIR - Mean: {round(avg_face_nir.mean(), 2)}, Std: {round(avg_face_nir.std(), 2)}")
            # Color analysis
            print(f"Greenness Index - Mean: {round(self.greenness_index(df).mean(), 2)}, Std: {round(self.greenness_index(df).std(), 2)}")
            print(f"Hue - Mean: {round(self.hue(df).mean(), 2)}, Std: {round(self.hue(df).std(), 2)}")
            print(f"NDVI - Mean: {round(self.ndvi(df).mean(), 2)}, Std: {round(self.ndvi(df).std(), 2)}")
            print(f"PSRI - Mean: {round(self.psri(df).mean(), 2)}, Std: {round(self.psri(df).std(), 2)}")
            print(f"NPCI - Mean: {round(self.npci(df).mean(), 2)}, Std: {round(self.npci(df).std(), 2)}")
            print("\n")

    def area(self, mesh: o3d.geometry.TriangleMesh, df: pd.DataFrame = None):
        """ Compute the area of the triangles in the point cloud
        Args:
            mesh (o3d.geometry.TriangleMesh): The input triangle mesh
            df (pd.DataFrame, optional): DataFrame containing point cloud data
        Returns:
            tuple: A tuple containing the triangle area list, average face RGB colors, and average face NIR values
        """
        # The color value is the average color of the 3 points of the triangle
        triangle_area_list = []
        avg_face_rgb = []
        avg_face_nir = []

        points_np = np.asarray(mesh.vertices)
        triangles_np = np.asarray(mesh.triangles)
        # analyse the 16 bit rgb & nir values
        rgb_16b = df[["wvl1", "wvl2", "wvl3"]].to_numpy()
        nir_16b = df["wvl4"].to_numpy()
        for i, triangle in enumerate(triangles_np):
            # Compute triangle area using Heron's formula
            # https://www.cuemath.com/measurement/semi-perimeter-of-triangle/
            AB = np.sqrt(np.sum((points_np[triangle[1]] - points_np[triangle[0]]) **2))
            AC = np.sqrt(np.sum((points_np[triangle[2]] - points_np[triangle[0]]) **2))
            BC = np.sqrt(np.sum((points_np[triangle[2]] - points_np[triangle[1]]) **2))
            semi_perimeter = (AB + AC + BC) / 2
            triangle_area = np.sqrt(semi_perimeter * (semi_perimeter-AB) * (semi_perimeter - AC) * (semi_perimeter * BC))

            triangle_area_list.append(triangle_area)
            avg_face_rgb.append((rgb_16b[triangle[0]] + rgb_16b[triangle[1]] + rgb_16b[triangle[2]]) / 3)
            avg_face_nir.append((nir_16b[triangle[0]] + nir_16b[triangle[1]] + nir_16b[triangle[2]]) / 3)

        triangle_area_list = np.array(triangle_area_list)
        avg_face_rgb = np.array(avg_face_rgb)
        avg_face_nir = np.array(avg_face_nir)
        return triangle_area_list, avg_face_rgb, avg_face_nir

    def greenness_index(self, df: pd.DataFrame) -> pd.Series:
        """ Compute the greenness index (Normalized Difference Vegetation Index) for each point in the point cloud
            Greenness: (2*G-R-B)/(R+G+B)
        Args:
            df (pd.DataFrame): DataFrame containing the point cloud data with RGB and NIR channels
        Returns:
            pd.Series: Series containing the greenness index for each point
        """
        red = df["wvl1"]
        green = df["wvl2"]
        blue = df["wvl3"]
        a = (2 * green - red - blue)
        b = (red + green + blue)
        greenness = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        return greenness

    def ndvi(self, df: pd.DataFrame) -> pd.Series:
        """ Compute the Normalized Difference Vegetation Index (NDVI) for each point in the point cloud
            NDVI: (NIR - R) / (NIR + R)
        Args:
            df (pd.DataFrame): DataFrame containing the point cloud data with RGB and NIR channels
        Returns:
            pd.Series: Series containing the NDVI for each point
        """
        red = df["wvl1"]
        nir = df["wvl4"]
        a = (nir - red)
        b = (nir + red)
        ndvi = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
        return ndvi

    def psri(self, df: pd.DataFrame) -> pd.Series:
        """ Compute the Plant Senescence Reflectance Index (PSRI) for each point in the point cloud
            PSRI: (R - G) / NIR
        Args:
            df (pd.DataFrame): DataFrame containing the point cloud data with RGB channels
        Returns:
            pd.Series: Series containing the PSRI for each point
        """
        red = df["wvl1"]
        green = df["wvl2"]
        nir = df["wvl4"]
        a = (red - green)
        psri = np.divide(a, nir, out=np.zeros_like(a), where=nir != 0)
        return psri

    def npci(self, df: pd.DataFrame) -> pd.Series:
        """ Compute the Normalized Pigment Chlorophyll Index (NPCI) for each point in the point cloud
            NPCI: (R - G) / NIR
        Args:
            df (pd.DataFrame): DataFrame containing the point cloud data with RGB channels
        Returns:
            pd.Series: Series containing the NPCI for each point
        """
        red = df["wvl1"]
        green = df["wvl2"]
        nir = df["wvl4"]
        a = (red - green)
        npci = np.divide(a, nir, out=np.zeros_like(a), where=nir != 0)
        return npci

    def hue(self, df: pd.DataFrame) -> pd.Series:
        """ Compute the hue for each point in the point cloud
            Hue: angle in degrees [0, 360]
            https://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2hsv
        Args:
            df (pd.DataFrame): DataFrame containing the point cloud data with RGB channels
        Returns:
            pd.Series: Series containing the hue for each point
        """
        # 16 bit values to 0-1 values.
        rgb_16b = df[["wvl1", "wvl2", "wvl3"]].values.astype(np.float32) / 65535.0
        hsv_16 = rgb2hsv(rgb_16b)
        hue = np.mean(hsv_16[:, 0]) * 360
        return hue


if __name__ == '__main__':

    ## Loading the DATA
    input_folder = './example_data'
    # for ply_file in Path(input_folder).glob('*.ply.gz'):
    #         # create paths to save the png images
    #         base_filename = ply_file.name.split('.')[0]
    #         unzipped_ply = ply_utils.unzip_ply(ply_file)
    analyzer = pointcloud_analysis(input_folder)
    analyzer.analyze()
