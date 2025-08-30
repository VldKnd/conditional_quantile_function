import json
from pathlib import Path
from infrastructure.training import train_from_json_file
from infrastructure.testing import test_from_json_file
import os

def process_config(file_path: str):
    """
    This is the function that gets called for each config file.
    'config_data' is the parsed JSON content from the file.
    'file_path' is the path to the config file.
    """
    print(f"--- Processing: {file_path} ---")
    file_folder_path = os.path.dirname(file_path)
    try:
        print(f"Training: {file_path}")
        train_from_json_file(file_path)
    except Exception as exception:
        with open(f"{file_folder_path}/train_error.log", "w") as error_file:
            error_file.write(str(exception))
        print(f"Error: Could not train the model for {file_path}. Error logged in {file_folder_path}/train_error.log")

    try:
        print(f"Testing: {file_path}")
        test_from_json_file(file_path, verbose=True)
    except Exception as exception:
        with open(f"{file_folder_path}/test_error.log", "w") as error_file:
            error_file.write(str(exception))
        print(f"Error: Could not test the model for {file_path}. Error logged in {file_folder_path}/test_error.log")

    print("-" * (20 + len(str(file_path))))


def find_and_process_configs(root_directory):
    """
    Scans a root directory for subfolders containing a 'config.json' file
    and calls process_config() for each one found.
    """
    root_path = Path(root_directory)
    if not root_path.is_dir():
        print(f"Error: Directory not found at '{root_directory}'")
        return

    print(f"Starting scan in '{root_path}'...")

    for item in root_path.iterdir():
        if item.is_dir():
            config_file = item / 'config.json'

            if config_file.is_file():
                try:
                    process_config(config_file)
                except json.JSONDecodeError:
                    print(f"Error: Could not parse JSON from '{config_file}'")
                except Exception as e:
                    print(f"An unexpected error occurred with {config_file}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--target_folder', type=str, default=f"{os.getcwd()}/experiments")
    args = parser.parse_args()
    TARGET_FOLDER = args.target_folder
    find_and_process_configs(TARGET_FOLDER)