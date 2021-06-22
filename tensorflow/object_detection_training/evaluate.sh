#!/usr/bin/env bash
# Used to retrain object detection models using TensorFlow 2 Object Detection API
# NOTE: Should be run from same directory as this file

# Evaluate SSD Mobilenet V2 FPN Lite (my_ssd_mobilenet_v2_fpnlite)
python scripts/train_eval/model_main_tf2.py --model_dir=models/my_ssd_mobilenet_v2_fpnlite --pipeline_config_path=models/my_ssd_mobilenet_v2_fpnlite/pipeline.config --checkpoint_dir=models/my_ssd_mobilenet_v2_fpnlite