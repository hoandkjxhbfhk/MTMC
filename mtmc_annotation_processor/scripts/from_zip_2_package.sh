#!/bin/bash

data_dir=$1

bash scripts/extract_latest_zip.sh $data_dir

python src/restore_annotations.py --scene $data_dir

bash scripts/mtmc_dataset_packaging.sh $data_dir

python src/merge_annotations.py --scene $data_dir