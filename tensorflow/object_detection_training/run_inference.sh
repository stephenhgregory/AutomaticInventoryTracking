#!/usr/bin/env bash
# Used to retrain inference using trained Object Detection models
# NOTE: Should be run from same directory as this file

# Do inference with SSD Mobilenet V2 FPN Lite (my_ssd_mobilenet_v2_fpnlite) and save result predictions
# (Saved result predictions for visual analysis only)
python scripts/inference/run_inference.py \
--saved_model_dir=exported-models/my_ssd_mobilenet_v2_fpnlite/saved_model \
--image_dir=images/test \
--label_map=annotations/label_map.pbtxt \
--save_images \
--save_image_dir=images/results

## Do inference with SSD Mobilenet V2 FPN Lite (my_ssd_mobilenet_v2_fpnlite)
#python scripts/inference/run_inference.py \
#--saved_model_dir=exported-models/my_ssd_mobilenet_v2_fpnlite/saved_model \
#--image_dir=images/test \
#--label_map=annotations/label_map.pbtxt
