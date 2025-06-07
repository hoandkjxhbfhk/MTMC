#!/bin/bash

# Check if the script received an argument (name of the data folder)
if [ -z "$1" ]; then
  echo "Usage: $0 <data_folder_name>"
  exit 1
fi

# Assign the argument to a variable
DATA_FOLDER=$1

# Navigate to the annotations directory
ANNOTATIONS_DIR="data/$DATA_FOLDER/annotations"

# Check if the annotations directory exists
if [ ! -d "$ANNOTATIONS_DIR" ]; then
  echo "Directory $ANNOTATIONS_DIR does not exist!"
  exit 1
fi

# Find the last modified .zip file in the annotations directory
LAST_MODIFIED_ZIP=$(ls -t "$ANNOTATIONS_DIR"/*.zip 2>/dev/null | head -n 1)

# Check if any zip files are found
if [ -z "$LAST_MODIFIED_ZIP" ]; then
  echo "No zip files found in $ANNOTATIONS_DIR"
  exit 1
fi

# Extract the last modified zip file, overwrite any existing files
unzip -o "$LAST_MODIFIED_ZIP" -d "$ANNOTATIONS_DIR"

# Confirmation message
echo "Extracted $LAST_MODIFIED_ZIP to $ANNOTATIONS_DIR"
