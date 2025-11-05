# 🛰️ Fractional Cover (FC) Analysis Pipeline

This document outlines the **end-to-end process** for conducting **Fractional Cover (FC)** analysis on satellite imagery (**Sentinel-2** and **MODIS**) using a High-Performance Computing (**HPC**) environment. It covers data retrieval, individual tile processing, and final output combination.

---

## Pipeline Overview

This pipeline automates the process of calculating Fractional Cover (FC) – the proportion of **exposed soil**, **photosynthetic vegetation**, and **non-photosynthetic vegetation** within each pixel of satellite imagery.

It is broken down into four main stages:

1. **GEE Imagery Export (Step 0):** Processes raw Sentinel-2 imagery in Google Earth Engine and exports weekly composites to Google Drive.
2. **Data Retrieval (Step 1):** Downloads these exported GeoTIFF satellite imageries from Google Drive to the HPC.
3. **FC Prediction (Step 2):** Processes each individual image tile to calculate its fractional cover.
4. **Tile Combination (Step 3):** Merges the processed individual tiles into larger, coherent weekly or daily composite maps of the entire region of interest.

---

## Prerequisites (HPC Setup)

Before running the pipeline, ensure the following are set up on your HPC environment:

* **HPC Access:** You have an active user account on the HPC with SSH access.
* **Google Earth Engine (GEE) Setup (for Step 0):**
    * Python environment with `earthengine-api` and `geemap` installed.
    * `ee.Authenticate()` has been run and you are authenticated to your GEE account.
    * A GEE project (e.g., `ee-ilyasnursyamsi05` or your own) is active.
* **rclone Installation & Configuration:**
    * `rclone` is installed in your user directory (e.g., `~/bin`).
    * Your `PATH` environment variable is correctly set in `~/.bashrc` or `~/.bash_profile` to include `~/bin`.
    * A Google Drive remote (e.g., named **gdrive**) is configured using `rclone config`.
    * **Crucially for HPC compute nodes:** **Proxy settings** are configured in your `~/.bash_profile` **AND** explicitly in your PBS job scripts to allow outbound internet access.

    > **Example Proxy Settings:**
    >
    > ```bash
    > export ftp_proxy="http://139.86.9.82:8080/"
    > export http_proxy="http://139.86.9.82:8080/"
    > export https_proxy="http://139.86.9.82:8080/"
    > export no_proxy="localhost, 127.0.0.1"
    > export PATH ftp_proxy http_proxy https_proxy no_proxy # Ensure all are exported
    > ```

* **Miniconda Installation & Environment:**
    * Miniconda is installed in your home directory (e.g., `~/miniconda3`).
    * A Conda environment (e.g., **fc_env**) is created and contains all necessary Python libraries (e.g., `numpy`, `tensorflow`, `rasterio`, `gdal`, `tqdm`, `scipy.ndimage`, and the **unmixcover** module).
    * The `environment.yml` file defines these dependencies.
    * The `unmixcover` module itself is located within your `WORKFLOW_DIR`.

---

## Step 0: GEE Imagery Export (Weekly Sentinel-2 Composites) ☁️

This initial step is performed **outside** the HPC job, typically in a local Python environment (like a Jupyter Notebook) that has access to Google Earth Engine.

* **Purpose:** To process raw Sentinel-2 imagery, apply cloud masking, generate weekly composite GeoTIFFs, and export them directly to a specified Google Drive folder.
* **Tool Used:** Python script (e.g., `download_Sentinel2_FC_Oct_Nov(WeeklyComposite)_2015_2024_onefolder.ipynb`).
* **Input:**
    * Raw Sentinel-2 imagery from GEE collections (Level-1C or Harmonized Level-2A).
    * A defined Region of Interest (ROI).
* **Output:** Weekly Sentinel-2 GeoTIFF composites, saved to a designated folder in your Google Drive.

### Key Parameters in the GEE Export Script

| Parameter | Description | Example Value |
| :--- | :--- | :--- |
| `ROI_PATH` | Path to your Region of Interest asset in GEE. | `"projects/ee-ilyasnursyamsi05/assets/CHC"` |
| `start_year`, `end_year` | Defines the range of years for processing. | N/A |
| `COMMON_EXPORT_FOLDER` | The Google Drive folder where GeoTIFFs are saved. | `'GEE_Exports_Weekly_OctNov15_24_CHC'` |

* **Output Naming Convention (in Google Drive):** `S2_Weekly_YYYYMMDD_YYYYMMDD.tif`.

---

By following this guide, you should be able to successfully run and manage your Fractional Cover analysis pipeline on the HPC!
