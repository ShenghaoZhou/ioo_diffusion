import os
import multiprocessing
from zipfile import ZipFile
from io import BytesIO
import requests
import time
import argparse
import functools

urls = (
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip",
    "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_03_difficult/V2_03_difficult.zip",
)

def download_and_unzip(url, target_dir="."):
    """Downloads a zip file from a URL and extracts it."""
    try:
        filename = url.split("/")[-1]
        # Extract the base name without extension for the directory name
        dirname = filename.replace(".zip", "")
        extract_path = os.path.join(target_dir, dirname)

        # Check if the directory already exists (implying successful download/extraction)
        if os.path.exists(extract_path):
            print(f"Directory {extract_path} already exists. Skipping download for {filename}.")
            return

        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for bad status codes

        with ZipFile(BytesIO(response.content)) as zip_ref:
            print(f"Unzipping {filename} to {extract_path}...")
            os.makedirs(extract_path, exist_ok=True)
            # List all files in the zip
            all_files = zip_ref.namelist()
            # Exclude "__MACOSX" from the list of files
            all_files = [f for f in all_files if not f.startswith("__MACOSX")]
            # Filter the files we need
            files_to_extract = [
                f for f in all_files if 'mav0/imu0' in f or 'mav0/state_groundtruth_estimate0' in f]
            # Extract the filtered files
            for file in files_to_extract:
                zip_ref.extract(file, extract_path)
                
        print(f"Successfully downloaded and extracted {filename}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
    except ZipFile.BadZipFile:
        print(f"Error: Downloaded file from {url} is not a valid zip file or is corrupted.")
    except Exception as e:
        print(f"An unexpected error occurred for {url}: {e}")

if __name__ == "__main__":
    start_time = time.time()
    # Create a pool of worker processes
    # Adjust the number of processes based on your CPU cores and network bandwidth
    num_processes = min(len(urls), os.cpu_count() or 1)
    print(f"Using {num_processes} processes for downloading and extracting.")
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Download and extract EuRoC MAV dataset sequences.")
    parser.add_argument(
        "--target_dir",
        type=str,
        default=".",
        help="Directory where the datasets will be downloaded and extracted."
    )
    args = parser.parse_args()

    # Ensure the target directory exists
    target_directory = args.target_dir
    os.makedirs(target_directory, exist_ok=True)
    print(f"Target directory set to: {os.path.abspath(target_directory)}")

    # Create a partial function with the target directory fixed
    # This binds the target_dir argument for the pool.map call later
    download_func = functools.partial(download_and_unzip, target_dir=target_directory)

    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use pool.map to apply the function to each URL
        # We use a partial function or lambda if download_and_unzip needs more args fixed
        pool.map(download_func, urls)

    end_time = time.time()
    print(f"\nAll downloads and extractions completed in {end_time - start_time:.2f} seconds.")

