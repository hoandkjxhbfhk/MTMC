#!/bin/bash

# Check if the script received an argument
if [ -z "$1" ]; then
  echo "Usage: $0 <name>"
  exit 1
fi

# Assign the argument to a variable
NAME=$1

# Create directories inside data with the given name
mkdir -p data/"$NAME"/annotations
mkdir -p data/"$NAME"/videos

src_dir="/home/minhlv/MTMC/Annotations/MOT_anno/$NAME"
dest_dir="data/$NAME/videos"

cp -r "$src_dir"/* "$dest_dir"

echo "Copied $src_dir to $dest_dir" 

# Confirmation message
echo "Directories created: data/$NAME/annotations and data/$NAME/videos"
