from huggingface_hub import snapshot_download

# Define repository details
repo_id = "nvidia/PhysicalAI-SmartSpaces"  # Replace with your repository ID
repo_type = "dataset"  # Change to "model" or "space" if applicable

# Define the folder to download and the local directory to save files
folder_name = "MTMC_Tracking_2024"

# Download the specified folder while excluding .h5 files
snapshot_download(
    repo_id=repo_id,
    repo_type=repo_type,
    allow_patterns=f"{folder_name}/test/scene_082/*",  # Include all files in the specified folder
    # ignore_patterns="*.h5",  # Exclude all .h5 files
    local_dir_use_symlinks=False
)
