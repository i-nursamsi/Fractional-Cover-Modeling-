import numpy as np
import tensorflow as tf
from osgeo import gdal, osr
import os
import shutil
import datetime
from tqdm import tqdm
import scipy.ndimage
import sys
import re
import concurrent.futures

# --- Import the unmixcover module ---
UNMIXCOVER_DIR = r"/sandisk1/u8022291/Automating_Workflow"
if UNMIXCOVER_DIR not in sys.path:
    sys.path.append(UNMIXCOVER_DIR)
import unmixcover

# --- Global Variables and Configuration ---
# 🔑 INPUT TILES ARE IN THE SENTINEL2MONTHLY FOLDER NOW
GEE_TILES_DIR = r"/sandisk1/u8022291/Automating_Workflow/Sentinel2Monthly"
# 🔑 FC OUTPUT TILES WILL BE TEMPORARILY SAVED IN THE DATA FOLDER
FC_TILES_OUTPUT_DIR = r"/sandisk1/u8022291/Automating_Workflow/Data"
os.makedirs(FC_TILES_OUTPUT_DIR, exist_ok=True)

TFLITE_MODEL_PATH = r"/sandisk1/u8022291/Automating_Workflow/pkgdata/fcModel_256x128x256.tflite"

INPUT_REFLECTANCE_SCALE_FACTOR = 10000.0
INPUT_NODATA_VALUE = 0
OUTPUT_NULL_VALUE = 255
TILE_SIZE = 512
MODEL_INPUT_BANDS_ORDER = ['B2', 'B3', 'B4', 'B8', 'B11', 'B12']
TARGET_CRS_EPSG = "EPSG:3577"
SMOOTHING_RADIUS_PIXELS = 1.0

# --- Helper Functions ---
def read_raster_window_as_numpy(dataset, band_index, x_offset, y_offset, window_xsize, window_ysize):
    try:
        band = dataset.GetRasterBand(band_index)
        data = band.ReadAsArray(x_offset, y_offset, window_xsize, window_ysize)
        return data
    except Exception as e:
        print(f"Error reading window from band {band_index} at ({x_offset},{y_offset}) with size ({window_xsize},{window_ysize}): {e}")
        return None

def write_numpy_array_to_geotiff_window(output_dataset, band_index, data_array, x_offset, y_offset):
    try:
        out_band = output_dataset.GetRasterBand(band_index)
        out_band.WriteArray(data_array, x_offset, y_offset)
    except Exception as e:
        print(f"Error writing window to band {band_index} at ({x_offset},{y_offset}): {e}")

def load_tflite_model(model_path):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print(f"Successfully loaded TFLite model from: {model_path}")
        return interpreter
    except Exception as e:
        print(f"Error loading TFLite model: {e}")
        return None

def get_model_input_details(interpreter):
    input_details = interpreter.get_input_details()
    if not input_details:
        raise ValueError("No input tensors found in the TFLite model.")
    return input_details[0]['index'], input_details[0]['shape'], input_details[0]['dtype']

def get_model_output_details(interpreter):
    output_details = interpreter.get_output_details()
    if not output_details:
        raise ValueError("No output tensors found in the TFLite model.")
    return output_details[0]['index'], output_details[0]['shape'], output_details[0]['dtype']

def fudge_sentinel2_to_landsat_etm(s2_bands_stacked, in_null_val):
    fudged_bands = np.zeros_like(s2_bands_stacked)
    null_mask = np.any(s2_bands_stacked == in_null_val, axis=0)
    c0 = np.array([-0.0022, 0.0031, 0.0064, 0.012, 0.0079, -0.0042])
    c1 = np.array([0.9551, 1.0582, 0.9871, 1.0187, 0.9528, 0.9688])
    fudged_bands[0, :, :] = (s2_bands_stacked[0, :, :] * c1[0]) + c0[0]
    fudged_bands[1, :, :] = (s2_bands_stacked[1, :, :] * c1[1]) + c0[1]
    fudged_bands[2, :, :] = (s2_bands_stacked[2, :, :] * c1[2]) + c0[2]
    fudged_bands[3, :, :] = (s2_bands_stacked[3, :, :] * c1[3]) + c0[3]
    fudged_bands[4, :, :] = (s2_bands_stacked[4, :, :] * c1[4]) + c0[4]
    fudged_bands[5, :, :] = (s2_bands_stacked[5, :, :] * c1[5]) + c0[5]
    for i in range(fudged_bands.shape[0]):
        fudged_bands[i][null_mask] = in_null_val
    return fudged_bands

def apply_spatial_smoothing(fc_chunk_uint8, smoothing_radius, nodata_value):
    smoothed_chunk = np.copy(fc_chunk_uint8)
    if smoothing_radius > 0:
        for i in range(smoothed_chunk.shape[0]):
            smoothed_band = scipy.ndimage.gaussian_filter(
                input=fc_chunk_uint8[i],
                sigma=smoothing_radius,
                mode='nearest',
                truncate=3.0
            )
            smoothed_chunk[i] = smoothed_band.astype(np.uint8)
    return smoothed_chunk

def run_fractional_cover_prediction_chunked(input_file_path, output_dir, output_filename_prefix):
    interpreter = load_tflite_model(TFLITE_MODEL_PATH)
    if interpreter is None:
        return None
    input_index, model_input_shape, model_input_dtype = get_model_input_details(interpreter)
    output_index, model_output_shape, model_output_dtype = get_model_output_details(interpreter)
    print(f"\nOpening Sentinel-2 input composite: {os.path.basename(input_file_path)}...")
    input_ds = gdal.Open(input_file_path, gdal.GA_ReadOnly)
    if input_ds is None:
        print(f"Failed to open input composite: {input_file_path}. Skipping.")
        return None
    xsize = input_ds.RasterXSize
    ysize = input_ds.RasterYSize
    input_geotransform = input_ds.GetGeoTransform()
    input_projection_wkt = input_ds.GetProjection()
    num_input_bands_in_file = input_ds.RasterCount
    if len(MODEL_INPUT_BANDS_ORDER) != num_input_bands_in_file:
        print(f"Warning: Expected {len(MODEL_INPUT_BANDS_ORDER)} bands but found {num_input_bands_in_file}. Skipping {os.path.basename(input_file_path)}.")
        del input_ds
        return None
    intermediate_output_filename = f"{output_filename_prefix}_Interm.tif"
    intermediate_output_file_path = os.path.join(output_dir, intermediate_output_filename)
    output_bands = 3
    driver = gdal.GetDriverByName("GTiff")
    options = ['COMPRESS=DEFLATE', 'PREDICTOR=2', 'BIGTIFF=IF_NEEDED']
    output_ds = driver.Create(intermediate_output_file_path, xsize, ysize, output_bands, gdal.GDT_Byte, options)
    if output_ds is None:
        print(f"Failed to create intermediate output GeoTIFF: {intermediate_output_file_path}. Skipping.")
        del input_ds
        return None
    output_ds.SetGeoTransform(input_geotransform)
    output_ds.SetProjection(input_projection_wkt)
    for i in range(output_bands):
        output_ds.GetRasterBand(i + 1).SetNoDataValue(OUTPUT_NULL_VALUE)
    print(f"Starting chunked prediction with spatial smoothing for {os.path.basename(input_file_path)}...")
    total_chunks = ((ysize + TILE_SIZE - 1) // TILE_SIZE) * ((xsize + TILE_SIZE - 1) // TILE_SIZE)
    input_band_indices = [MODEL_INPUT_BANDS_ORDER.index(band_name) + 1 for band_name in MODEL_INPUT_BANDS_ORDER]
    with tqdm(total=total_chunks, desc=f"Processing {os.path.basename(input_file_path)}") as pbar:
        for y_offset in range(0, ysize, TILE_SIZE):
            for x_offset in range(0, xsize, TILE_SIZE):
                window_xsize = min(TILE_SIZE, xsize - x_offset)
                window_ysize = min(TILE_SIZE, ysize - y_offset)
                current_chunk_bands_raw = [read_raster_window_as_numpy(input_ds, idx, x_offset, y_offset, window_xsize, window_ysize) for idx in input_band_indices]
                if any(chunk is None for chunk in current_chunk_bands_raw):
                    pbar.update(1)
                    continue
                surface_reflectance_chunk_scaled = np.stack(current_chunk_bands_raw, axis=0).astype(np.float32) / INPUT_REFLECTANCE_SCALE_FACTOR
                fudged_surface_reflectance_chunk = fudge_sentinel2_to_landsat_etm(surface_reflectance_chunk_scaled, INPUT_NODATA_VALUE / INPUT_REFLECTANCE_SCALE_FACTOR)
                try:
                    output_fc_array_raw = unmixcover.unmix_fractional_cover(fudged_surface_reflectance_chunk, interpreter, inNull=INPUT_NODATA_VALUE/INPUT_REFLECTANCE_SCALE_FACTOR)
                except unmixcover.UnmixError:
                    pbar.update(1)
                    continue
                sum_of_components = np.sum(output_fc_array_raw, axis=0)
                sum_of_components[sum_of_components == 0] = 1e-6
                normalized_fc_array = output_fc_array_raw / sum_of_components[np.newaxis, :, :]
                output_fc_array_scaled_chunk = np.round(normalized_fc_array * 100).astype(np.uint8)
                output_fc_array_scaled_chunk = np.clip(output_fc_array_scaled_chunk, 0, 100)
                if SMOOTHING_RADIUS_PIXELS > 0:
                    smoothed_chunk = apply_spatial_smoothing(fc_chunk_uint8=output_fc_array_scaled_chunk, smoothing_radius=SMOOTHING_RADIUS_PIXELS, nodata_value=OUTPUT_NULL_VALUE)
                    current_sum = np.sum(smoothed_chunk, axis=0)
                    deviation = 100 - current_sum
                    smoothed_chunk[0] = np.clip(smoothed_chunk[0] + deviation, 0, 100)
                else:
                    smoothed_chunk = output_fc_array_scaled_chunk
                for i in range(output_bands):
                    write_numpy_array_to_geotiff_window(output_ds, i + 1, smoothed_chunk[i], x_offset, y_offset)
                pbar.update(1)
    output_ds.FlushCache()
    del output_ds
    del input_ds
    final_output_filename = f"{output_filename_prefix}_Final.tif"
    final_output_file_path = os.path.join(output_dir, final_output_filename)
    gdal.Warp(
        final_output_file_path,
        intermediate_output_file_path,
        dstSRS=TARGET_CRS_EPSG,
        resampleAlg=gdal.GRA_NearestNeighbour,
        srcNodata=OUTPUT_NULL_VALUE,
        dstNodata=OUTPUT_NULL_VALUE,
    )
    try:
        os.remove(intermediate_output_file_path)
    except Exception as e:
        print(f"Warning: Could not remove intermediate file {intermediate_output_file_path}: {e}")
    return final_output_file_path

# --- Parallel Image Processing Function ---
def process_image(file_name):
    # 🔑 Input files are now from the GEE_TILES_DIR
    file_path = os.path.join(GEE_TILES_DIR, file_name)
    
    # 🔑 The output filename will be based on the input filename
    output_filename_prefix = os.path.splitext(file_name)[0]
    print(f"\n--- Starting processing for {file_name} ---")
    
    # 🔑 The output is saved to the FC_TILES_OUTPUT_DIR
    final_output_path = run_fractional_cover_prediction_chunked(file_path, FC_TILES_OUTPUT_DIR, output_filename_prefix)
    
    if final_output_path:
        print(f"--- Successfully saved output to: {final_output_path} ---")
    else:
        print(f"--- Fractional cover prediction failed for {file_name} ---")

# --- Main execution block ---
if __name__ == '__main__':
    gdal.AllRegister()
    print(f"Processing all TIF files in: {GEE_TILES_DIR}")
    # Retrieve all files ending with .tif
    input_files = [f for f in os.listdir(GEE_TILES_DIR) if f.endswith('.tif')]
    if not input_files:
        print("No TIF files found in the specified directory. Exiting.")
    else:
        # Use as many workers as CPUs available
        with concurrent.futures.ProcessPoolExecutor() as executor:
            executor.map(process_image, input_files)