import os
import glob
import rasterio
from rasterio.merge import merge
import numpy as np
import re
from tqdm import tqdm
from collections import defaultdict

##DONT FORGET TO CHANGE THE NAME OF THE FILE OUPUT OF WHICH NAMED AFTER THE ROI

# --- Configuration ---
# Set this to the directory where your FractionalCover_HPC_Paralell.py script saves its output.
FC_TILES_DIR = r"/sandisk1/u8022291/Automating_Workflow/Data"
COMBINED_OUTPUT_DIR = r"/sandisk1/u8022291/Automating_Workflow/FractionalCover_Results"
os.makedirs(COMBINED_OUTPUT_DIR, exist_ok=True)

# --- Function to combine tiles for a single week into a single TIFF ---
def combine_weekly_tiles_to_weekly_tiff(tile_files, output_filepath):
    """
    Combines a list of GeoTIFF tiles for a single weekly composite into a single GeoTIFF file.
    """
    if not tile_files:
        print(f"  No tiles found to combine for {os.path.basename(output_filepath)}. Skipping.")
        return None

    src_files_to_mosaic = []
    for f in tile_files:
        try:
            src = rasterio.open(f)
            src_files_to_mosaic.append(src)
        except rasterio.errors.RasterioIOError as e:
            print(f"  Warning: Could not open {os.path.basename(f)} for mosaicking. Skipping this file. Error: {e}")
            continue

    if not src_files_to_mosaic:
        print(f"  No valid source files opened for mosaicking for {os.path.basename(output_filepath)}. Skipping.")
        return None

    try:
        mosaic, out_transform = merge(src_files_to_mosaic)
    except MemoryError as e:
        print(f"  MemoryError merging tiles for {os.path.basename(output_filepath)}: {e}. "
              "Consider running on a machine with more RAM. Skipping.")
        for src in src_files_to_mosaic:
            src.close()
        return None
    except Exception as e:
        print(f"  Error merging tiles for {os.path.basename(output_filepath)}: {e}. Skipping.")
        for src in src_files_to_mosaic:
            src.close()
        return None

    # Update metadata for the output TIFF
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "count": mosaic.shape[0],
        "dtype": 'uint8',
        "compress": "deflate",
        "predictor": 2
    })

    # Write the mosaicked image to file
    try:
        with rasterio.open(output_filepath, "w", **out_meta) as dest:
            dest.write(mosaic)
        print(f"  Successfully combined {len(src_files_to_mosaic)} tiles into {os.path.basename(output_filepath)}")
        return output_filepath
    except Exception as e:
        print(f"  Error writing combined image to {os.path.basename(output_filepath)}: {e}. Skipping.")
        return None
    finally:
        for src in src_files_to_mosaic:
            src.close()

# --- Main script execution ---
if __name__ == "__main__":
    print("Starting FC weekly tile combination process...")

    # Use a specific glob pattern to find the final output files from the previous step.
    # The output from the previous script is expected to be named `S2_Weekly_*_Final.tif`.
    all_fc_files = glob.glob(os.path.join(FC_TILES_DIR, "S2_Weekly_*_Final.tif"))

    if not all_fc_files:
        print(f"No weekly FC tiles found in {FC_TILES_DIR}. Exiting.")

    # Use a nested defaultdict to group files by their full date range (which represents a week)
    files_by_week = defaultdict(list)

    for filepath in all_fc_files:
        # 🔑 REVISED REGEX to capture the full weekly date range, accounting for chunk info
        # It now matches anything between the date range and '_Final.tif' non-greedily
        match = re.search(r'S2_Weekly_(\d{8}_\d{8}).*?_Final\.tif', os.path.basename(filepath))
        if match:
            # Capture the full date range, e.g., '20180401_20180407'
            date_range = match.group(1)
            files_by_week[date_range].append(filepath)
        else:
            print(f"  Warning: Skipping file with unrecognized naming convention: {os.path.basename(filepath)}")

    if not files_by_week:
        print(f"No properly named weekly FC tiles found in {FC_TILES_DIR}. Exiting.")

    sorted_weeks = sorted(files_by_week.keys())
    for date_range in tqdm(sorted_weeks, desc="Combining weekly tiles"):
        files_for_this_week = files_by_week[date_range]

        # Generate the new output filename for the final weekly composite
        # The output name is now S2_FC_{start_date}_{end_date}.tif
        output_filename = f"S2_FC_{date_range}_CHC.tif"
        output_filepath = os.path.join(COMBINED_OUTPUT_DIR, output_filename)

        created_tiff_path = combine_weekly_tiles_to_weekly_tiff(files_for_this_week, output_filepath)
        if created_tiff_path:
            pass
        else:
            print(f"  Failed to combine tiles for week {date_range}.")

    print("\nFC tile combination process completed.")
    print(f"Final combined weekly TIFF images are saved to: {COMBINED_OUTPUT_DIR}")
