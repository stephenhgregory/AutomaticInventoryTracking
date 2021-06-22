#!/usr/bin/env bash
# Used to export a model from checkpoint files that can be later used for inference
# NOTE: Should be run from same directory as this file

# Export SSD Mobilenet V2 FPN Lite (my_ssd_mobilenet_v2_fpnlite)
python scripts/utils/exporter_main_v2.py python --input_type image_tensor \
--pipeline_config_path models/my_ssd_mobilenet_v2_fpnlite/pipeline.config \
--trained_checkpoint_dir models/my_ssd_mobilenet_v2_fpnlite \
--output_directory ./exported-models/my_ssd_mobilenet_v2_fpnlite
