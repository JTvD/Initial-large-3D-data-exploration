from pathlib import Path
from shutil import copyfileobj
from plyfile import PlyData, PlyElement
import numpy as np
import pandas as pd
import open3d as o3d
import gzip


def unzip_ply(filename: Path) -> Path:
    """ Unzip ply file and save a copy of the unzipped file.
        Args:
            filename (Path): The path to the gzipped ply file.
        Returns:
            Path: The path to the unzipped ply file.
    """
    with gzip.open(filename, 'rb') as f_in:
        out_path = filename.with_suffix('')
        with open(out_path, 'wb') as f_out:
            copyfileobj(f_in, f_out)
    return out_path


def zip_ply(filename: Path) -> Path:
    """ Zip a ply file to a gzipped ply file.
        Args:
            filename (Path): The path to the uncompressed ply file.
        Returns:
            Path: The path to the gzipped ply file.
    """
    out_path = filename.with_suffix(filename.suffix + '.gz')
    with open(filename, 'rb') as f_in:
        with gzip.open(out_path, 'wb') as f_out:
            copyfileobj(f_in, f_out)
    return out_path


def datetype_mapper(df: pd.DataFrame) -> dict:
    """Map DataFrame dtypes to PLY file dtypes.
        Args:
            df (pd.DataFrame): The input DataFrame.
        Returns:
            dict: A dictionary mapping column names to PLY dtypes.
    """
    dtype_dict = {"x": "f4", "y": "f4", "z": "f4", "red": "u1", "green": "u1", "blue": "u1"}
    dtype_map = {
        "float32": "f4",
        "float64": "f8",
        "int32": "i4",
        "int64": "i4",
        "uint8": "u1",
        "uint16": "u2",
        "int16": "i2",
        "bool": "u1"
    }
    for colname, dtype in df.dtypes.items():
        if colname not in dtype_dict:
            dtype_str = str(dtype)
            if dtype_str in dtype_map:
                dtype_dict[colname] = dtype_map[dtype_str]
            else:
                raise ValueError(f"Unsupported dtype: {dtype_str} for column {colname}")
    return dtype_dict


def save_df_pointcloud(filename: Path, df: pd.DataFrame,
                       pcd_object: o3d.geometry.PointCloud = None):
    """ Open3D cannot save/load pointclouds with 3D scalars fields, therefore we use plyfile.
        Args:
            filename (str): The path to save the ply file.
            df (pd.DataFrame): The dataframe containing scalarfield values for each point.
            pcd_object (o3d.geometry.PointCloud): The Open3D point cloud object to save.
    """
    # Point cloud
    if hasattr(pcd_object, 'colors'):
        df[['x', 'y', 'z']] = pd.DataFrame(np.asarray(pcd_object.points)).astype('float32')
        df[['red', 'green', 'blue']] = pd.DataFrame(np.asarray(pcd_object.colors)*255).astype('uint8')
    # Mesh
    else:
        df[['x', 'y', 'z']] = pd.DataFrame(np.asarray(pcd_object.vertices)).astype('float32')
        df[['red', 'green', 'blue']] = pd.DataFrame(np.asarray(pcd_object.vertex_colors)*255).astype('uint8')

    vertices = df.to_numpy()
    dtype_dict = datetype_mapper(df)
    vertices = df.to_records(index=False, column_dtypes=dtype_dict)

    # Mesh
    if hasattr(pcd_object, 'triangles') and len(pcd_object.triangles) > 0:
        faces = np.asarray(pcd_object.triangles)
        face_dtype = [("vertex_index", "i4", (3,))]
        faces_np = np.array([(tuple(f),) for f in faces], dtype=face_dtype)
        ply = PlyData([PlyElement.describe(vertices, "vertex"),
                       PlyElement.describe(faces_np, 'face')], text=False)
    else:
        ply = PlyData([PlyElement.describe(vertices, "vertex")], text=False)

    ply.write(filename)


def load_df_pointcloud(filename: Path, return_pointcloud: bool = True,
                       df_drop: bool = True) -> tuple:
    """ Load pointcloud and scalarfields.
        Args:
            filename (str): The path to the ply file.
            return_pointcloud (bool): Whether to return the pointcloud object.
            df_drop (bool): Whether to drop the pointcloud columns from the dataframe.
        Returns:
            tuple: A tuple containing the dataframe and the pointcloud object (if requested).
    """

    plydata = PlyData.read(filename)
    df = pd.DataFrame(np.array(plydata["vertex"].data))

    # Check if 'face' element exists and is not empty, then add as new column
    if 'face' in plydata and len(plydata['face'].data) > 0:
        triangles = np.array(plydata['face'].data)
        tri_arr = np.stack(triangles['vertex_index'])
        df[['triangle1', 'triangle2', 'triangle3']] = pd.DataFrame(tri_arr, index=df.index[:len(tri_arr)]).astype('Int64')
    if return_pointcloud is True:
        if all(col in df.columns for col in ["triangle1", "triangle2", "triangle3", "red", "green", "blue"]):
            pcd = df_to_mesh(df)
            if df_drop:
                df.drop(["triangle1", "triangle2", "triangle3", "red", "green", "blue"], axis=1, inplace=True)
        elif all(col in df.columns for col in ["x", "y", "z", "red", "green", "blue"]):
            pcd = df_to_pointcloud(df)
            if df_drop:
                df.drop(["x", "y", "z", "red", "green", "blue"], axis=1, inplace=True)
        else:
            pcd = None
    else:
        pcd = None
    return (df, pcd)


def df_to_mesh(df: pd.DataFrame) -> o3d.geometry.TriangleMesh:
    """ Converts a dataframe to a triangle mesh.
        Requires columns: x, y, z, red, green, blue, triangle1, triangle2, triangle3
        Args:
            df (pd.DataFrame): The input dataframe.
        Returns:
            o3d.geometry.TriangleMesh: The resulting triangle mesh.
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.array(df[["x", "y", "z"]].values.tolist()))
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.array(df[["red", "green", "blue"]].values.tolist()) / 255)
    # Triangles: expects columns named triangle1, triangle2, triangle3 (indices)
    if all(col in df.columns for col in ["triangle1", "triangle2", "triangle3"]):
        sub_df = df.dropna(subset=['triangle1', 'triangle2', 'triangle3'])
        triangles = np.array(sub_df[["triangle1", "triangle2", "triangle3"]].values.tolist(), dtype=np.int32)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
    return mesh


def df_to_pointcloud(df: pd.DataFrame) -> o3d.geometry.PointCloud:
    """ Converts a dataframe to a pointcloud
        Requires columns: x, y, z, red, green, blue
        Args:
            df (pd.DataFrame): The input dataframe.
        Returns:
            o3d.geometry.PointCloud: The resulting point cloud.
       """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(df[["x", "y", "z"]].values.tolist()))
    pcd.colors = o3d.utility.Vector3dVector(np.array(df[["red", "green", "blue"]].values.tolist()) / 255)
    return pcd


if __name__ == "__main__":
    for ply_file in Path("example_data").glob('*.ply.gz'):

        # Load pointcloud
        unzipped_ply = unzip_ply(ply_file)
        df, pcd = load_df_pointcloud(unzipped_ply)

        # Store pointcloud
        output_file = Path('test_output.ply')
        save_df_pointcloud(output_file, df, pcd)
        zipped_path = zip_ply(output_file)
