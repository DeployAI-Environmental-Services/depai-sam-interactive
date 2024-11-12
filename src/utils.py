import io
import os
from io import BytesIO
from typing import Tuple
import uuid
import tifffile as tiff
from owslib.wms import WebMapService
from pyproj import Transformer, CRS
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling,
    transform_bounds,
)
import numpy as np
from PIL import Image


def save_numpy_as_geotiff(array, template_file, output_file, count=1):
    # Read the template file
    with rasterio.open(template_file) as src:
        template_profile = src.profile
        # template_transform = src.transform
        # template_crs = src.crs

    # Update the template profile with the array data
    template_profile.update(dtype=np.float32, count=count)

    # Create the output GeoTIFF file
    with rasterio.open(output_file, "w", **template_profile) as dst:
        dst.write(array, 1)


def format_float(f):
    return "%.6f" % (float(f),)


def shape_to_table_row(row):
    bounds = row.bounds.values[0]
    if row.geom_type.values[0] == "Point":
        return {
            "x_min": format_float(bounds[0]),
            "y_min": format_float(bounds[1]),
            "id": str(row["_leaflet_id"].values[0]),
        }
    return {
        "x_min": format_float(bounds[0]),
        "y_min": format_float(bounds[1]),
        "x_max": format_float(bounds[2]),
        "y_max": format_float(bounds[3]),
        "id": str(row["_leaflet_id"].values[0]),
    }


def bounds_to_table_row(polygon):
    bounds = polygon.bounds
    return {
        "x_min": format_float(bounds[0]),
        "y_min": format_float(bounds[1]),
        "x_max": format_float(bounds[2]),
        "y_max": format_float(bounds[3]),
        "id": "bbox",
    }


def id_generator():
    return str(uuid.uuid4())


def download_from_wms(
    wms_url: str,
    bbox: Tuple,
    layer: str,
    image_format: str,
    output_dir: str,
    resolution: int,
):
    wms = WebMapService(wms_url)
    # Specify the desired bounding box
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    xmin, ymin, xmax, ymax = bbox[0], bbox[1], bbox[2], bbox[3]
    xmin_t, ymin_t = transformer.transform(xmin, ymin)  # pylint: disable=E0633
    xmax_t, ymax_t = transformer.transform(xmax, ymax)  # pylint: disable=E0633
    width = int((xmax_t - xmin_t) / resolution)
    height = int((ymax_t - ymin_t) / resolution)

    # Request the image from the WMS
    image = wms.getmap(
        layers=[layer],
        srs="EPSG:4326",
        bbox=bbox,
        size=(width, height),
        format=image_format,
    )
    output_filename = id_generator()
    output_filepath = os.path.join(output_dir, output_filename + ".tif")
    img = np.array(Image.open(BytesIO(image.read())))
    pixel_size_x = (xmax - xmin) / width
    pixel_size_y = (ymax - ymin) / height
    transform = rasterio.transform.from_origin(xmin, ymax, pixel_size_x, pixel_size_y)  # type: ignore
    crs_to = CRS.from_epsg(4326)
    with rasterio.open(
        output_filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=img.dtype,
        crs=crs_to,
        transform=transform,
    ) as dst:
        dst.write(np.moveaxis(img, 2, 0))  # Assuming img is a 3-band RGB image

    return output_filepath


def geographic_to_pixel_bbox(
    bbox_geo: np.array,
    image_width: int,
    image_height: int,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
) -> np.array:
    # Calculate the conversion factors
    lat_range = max_latitude - min_latitude
    lon_range = max_longitude - min_longitude
    # lat_factor = image_height / lat_range
    # lon_factor = image_width / lon_range

    # Convert the bounding box coordinates to pixel coordinates
    x_min = ((bbox_geo[:, 0] - min_longitude) / lon_range * image_width).astype(int)
    y_min = ((max_latitude - bbox_geo[:, 3]) / lat_range * image_height).astype(int)
    x_max = ((bbox_geo[:, 2] - min_longitude) / lon_range * image_width).astype(int)
    y_max = ((max_latitude - bbox_geo[:, 1]) / lat_range * image_height).astype(int)

    # Create the pixel bounding box array
    pixel_bbox = np.column_stack((x_min, y_min, x_max, y_max))

    return pixel_bbox


def geographic_to_pixel_point(
    point_geo: np.array,
    image_width: int,
    image_height: int,
    min_latitude: float,
    max_latitude: float,
    min_longitude: float,
    max_longitude: float,
) -> np.array:
    # Calculate the conversion factors
    lat_range = max_latitude - min_latitude
    lon_range = max_longitude - min_longitude
    # lat_factor = image_height / lat_range
    # lon_factor = image_width / lon_range

    # Convert the bounding box coordinates to pixel coordinates
    x_pixel = ((point_geo[:, 0] - min_longitude) / lon_range * image_width).astype(int)
    y_pixel = ((max_latitude - point_geo[:, 1]) / lat_range * image_height).astype(int)

    # Create the pixel bounding box array
    pixel_coord = np.column_stack((x_pixel, y_pixel))

    return pixel_coord


def reproject_to_epsg4326(input_path, output_path):
    with rasterio.open(input_path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        dst_meta = src.meta.copy()
        dst_meta.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )
        reprojected_bounds = transform_bounds(src.crs, dst_crs, *src.bounds)
        with rasterio.open(output_path, "w", **dst_meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest,
                )

        with rasterio.open(output_path) as reprojected_tiff:
            # Read the image data (for simplicity, take the first 3 bands if RGB)
            bands = [
                reprojected_tiff.read(i)
                for i in range(1, min(4, reprojected_tiff.count + 1))
            ]
            img_array = np.stack(bands, axis=-1).astype(np.uint8)

            # Normalize and scale to 8-bit if the image is not already in 8-bit range
            if img_array.dtype != np.uint8:
                img_array = (
                    255
                    * (img_array - img_array.min())
                    / (img_array.max() - img_array.min())
                ).astype(np.uint8)

            # Convert to a PIL image (handling RGB or single-band grayscale)
            if img_array.shape[2] == 3:
                png_image = Image.fromarray(img_array, mode="RGB")
            else:
                raise ValueError("Unsupported number of bands for PNG export")

            # Save the PNG to bytes
            img_byte_arr = io.BytesIO()
            png_image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
            png_image.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)

            # Save PNG to disk
            png_file_path = output_path.rsplit(".", 1)[0] + ".png"
            with open(png_file_path, "wb") as f:
                f.write(img_byte_arr.getbuffer())

    return reprojected_bounds, img_byte_arr.getvalue(), png_file_path


# if __name__ == "__main__":
#     from config import *

#     download_from_wms(
#         WMS_URL,
#         (1.25, 43.5, 1.5, 43.75),
#         "s2cloudless-2020",
#         "image/jpeg",
#         "/Users/syam/Documents/code/dino-sam/src/",
#         10,
#     )
