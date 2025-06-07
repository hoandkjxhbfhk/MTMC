#!/bin/bash

# Check if the F05_01 directory name is passed as an argument
if [ -z "$1" ]; then
  echo "Usage: $0 F05_01"
  exit 1
fi

# Set the base directory name (e.g., F05_01)
BASE_DIR_NAME="$1"

# Define the path to the 'data' directory containing the target directories
DATA_DIR="data"

# Process each directory in the 'data' folder that matches the base name pattern
for dir in "$DATA_DIR"/"$BASE_DIR_NAME"*; do
  # Ensure the directory is valid
  if [ -d "$dir" ]; then
    echo "Processing directory: $dir"

    # Navigate to the subdirectory
    cd "$dir" || exit

    # Iterate over each subfolder in the videos directory
    for video_dir in videos/*; do
      # Get the base name (ID number) of the folder
      id=$(basename "$video_dir")

      # Create a new folder in the root of this subdirectory named after the ID
      mkdir -p "$id"

      # Move the corresponding video file to the new folder
      if [ -f "$video_dir/$id.mp4" ]; then
        mv "$video_dir/$id.mp4" "$id/"
        echo "Moved $video_dir/$id.mp4 to $id/"
      else
        echo "Warning: Video file $video_dir/$id.mp4 not found."
      fi

      # Move the corresponding annotation file from the annotations folder
      if [ -f "annotations/$id.txt" ]; then
        mv "annotations/$id.txt" "$id/"
        echo "Moved annotations/$id.txt to $id/"
      else
        echo "Warning: Annotation file annotations/$id.txt not found."
      fi
    done

    # Only delete the directories if they've been processed successfully
    if [ -d "videos" ]; then
      rm -rf videos
      echo "Deleted videos directory in $dir."
    fi

    if [ -d "annotations" ]; then
      rm -rf annotations
      echo "Deleted annotations directory in $dir."
    fi

    # Show the structure after processing for this directory

    # Return to the base directory for the next loop iteration
    cd - > /dev/null
  else
    echo "Error: Directory $dir not found."
  fi
done

echo "All directories processed."
