#!/bin/bash
set -e
############################################# INPUT #############################################

# path to scene directory downloaded; recommended to have the path inside ""
WORK_DIR="lse_raw_data/Dragon Max"

# path to output directory
OUT_DIR="formatted_data/$(basename "$WORK_DIR")"

# path to event camera intrinsics
PROPHESEE_CAM_F="lse_raw_data/ecam_intrinsics.json"

# path to relative extrinsics between the rgb and event camera
REL_CAM_F="lse_raw_data/rel_extrinsics.json"

# in microseconds; window to create the BIIs (event frames) in
DELTA_T=5000 
############################################# INPUT END ###########################################

# path to temporary output
TMP_OUT="tmp"


# Step 1: Format raw data to llff format (intermediate format)
python format_raw_step1.py --work_dir "$WORK_DIR" \
                           --targ_dir "$TMP_OUT" \
                           --delta_t "$DELTA_T" \
                           --prophesee_cam_f "$PROPHESEE_CAM_F" \
                           --rel_cam_f "$REL_CAM_F"


# Step 2: Format llff data to final format
python format_raw_step2.py --src_dir "$TMP_OUT" \
                           --targ_dir "$OUT_DIR"

# Step 3: generate corresponding ecam dataset.json
python update_dataset.py --dataset_dir "$OUT_DIR"

# step 4: clean up temporary output
rm -rf "$TMP_OUT"

echo DONE!