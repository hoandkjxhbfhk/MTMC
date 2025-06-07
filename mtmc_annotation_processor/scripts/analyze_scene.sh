#!/bin/bash

data_dir=$1

bash scripts/extract_latest_zip.sh $data_dir

python src/check_camera_label_dist.py --scene $data_dir